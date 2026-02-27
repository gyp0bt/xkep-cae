"""Physics-Informed Neural Network (PINN) training for thermal GNN.

Physics-Informed ロスの導入:
  L_total = L_data + λ * L_phys
  L_phys  = ||K @ ΔT_pred − f_shifted||² / ||f_shifted||²

ここで:
  K:         FEM全体剛性行列（伝導 + 対流）
  f_shifted: 右辺ベクトル（ΔT基準に変換済み: f − T_min · K @ 1）
  ΔT_pred:   GNN予測の温度上昇
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from xkep_cae.thermal.dataset import (
    ThermalProblemConfig,
    place_heat_sources,
    sample_to_graph_data,
)
from xkep_cae.thermal.fem import (
    assemble_thermal_system,
    make_irregular_rect_mesh,
    make_rect_mesh,
    solve_steady_thermal,
)
from xkep_cae.thermal.gnn import graph_dict_to_pyg


def generate_pinn_sample(
    nodes: np.ndarray,
    conn: np.ndarray,
    boundary_edges: dict[str, np.ndarray],
    config: ThermalProblemConfig,
    rng: np.random.Generator,
) -> dict:
    """PINN学習用サンプル生成（FEM行列付き）.

    通常のグラフデータに加え、FEM剛性行列 K と
    ΔT基準の右辺ベクトル f_shifted を格納する。

    Args:
        nodes: (N, 2) 節点座標
        conn: (Ne, 4) 接続配列
        boundary_edges: 境界辺 dict
        config: 問題設定
        rng: 乱数生成器

    Returns:
        graph dict with keys:
            "x": (N, 6) ノード特徴量
            "edge_index": (2, E) エッジインデックス
            "y": (N, 1) ターゲット ΔT
            "K_dense": (N, N) FEM全体行列（dense）
            "f_shifted": (N,) ΔT基準の右辺ベクトル
    """
    q_nodal, centers = place_heat_sources(nodes, config, rng)

    K, f = assemble_thermal_system(
        nodes,
        conn,
        boundary_edges,
        k=config.k,
        h_conv=config.h_conv,
        t=config.t,
        T_inf=config.T_inf,
        q_nodal=q_nodal,
    )
    T = solve_steady_thermal(K, f)

    sample = {
        "q_nodal": q_nodal,
        "T": T,
        "flux": None,
        "centers": centers,
        "nodes": nodes,
        "conn": conn,
    }
    graph = sample_to_graph_data(sample)

    # ΔT基準の右辺ベクトル: f_shifted = f - T_min * K @ ones
    T_min = float(T.min())
    ones_vec = np.ones(len(T))
    f_shifted = f - T_min * K.dot(ones_vec)

    graph["K_dense"] = K.toarray().astype(np.float32)
    graph["f_shifted"] = f_shifted.astype(np.float32)

    return graph


def generate_pinn_dataset(
    config: ThermalProblemConfig,
    n_samples: int,
    seed: int = 42,
) -> list[dict]:
    """PINN学習用データセット生成.

    Args:
        config: 問題設定
        n_samples: サンプル数
        seed: 乱数シード

    Returns:
        list of graph data dicts（K, f 付き）
    """
    rng = np.random.default_rng(seed)
    nodes, conn, edges = make_rect_mesh(config.Lx, config.Ly, config.nx, config.ny)

    dataset = []
    for _ in range(n_samples):
        graph = generate_pinn_sample(nodes, conn, edges, config, rng)
        dataset.append(graph)

    return dataset


def generate_pinn_dataset_irregular(
    config: ThermalProblemConfig,
    n_samples: int,
    perturbation: float = 0.3,
    seed: int = 42,
) -> list[dict]:
    """不規則メッシュ上のPINN学習用データセット生成.

    Args:
        config: 問題設定
        n_samples: サンプル数
        perturbation: ノード摂動量（要素サイズ比、0〜0.45推奨）
        seed: 乱数シード

    Returns:
        list of graph data dicts（K, f 付き）
    """
    rng = np.random.default_rng(seed)
    nodes, conn, edges = make_irregular_rect_mesh(
        config.Lx,
        config.Ly,
        config.nx,
        config.ny,
        perturbation=perturbation,
        seed=seed + 1000,
    )

    dataset = []
    for _ in range(n_samples):
        graph = generate_pinn_sample(nodes, conn, edges, config, rng)
        dataset.append(graph)

    return dataset


def graph_dict_to_pyg_pinn(graph_dict: dict, config: ThermalProblemConfig) -> Data:
    """PINN用の PyG Data 変換（K, f 付き）.

    Args:
        graph_dict: generate_pinn_sample() の出力
        config: 問題設定

    Returns:
        PyG Data with K_dense, f_shifted tensors
    """
    data = graph_dict_to_pyg(graph_dict, config)
    data.K_dense = torch.from_numpy(graph_dict["K_dense"])
    data.f_shifted = torch.from_numpy(graph_dict["f_shifted"])
    return data


def compute_physics_loss(
    pred_dt: torch.Tensor,
    K_dense: torch.Tensor,
    f_shifted: torch.Tensor,
) -> torch.Tensor:
    """Physics-informed 残差ロスを計算.

    L_phys = ||K @ ΔT_pred − f_shifted||² / ||f_shifted||²

    Args:
        pred_dt: (N,) 非正規化ΔT予測値
        K_dense: (N, N) FEM全体行列
        f_shifted: (N,) ΔT基準右辺ベクトル

    Returns:
        正規化残差 MSE（スカラー）
    """
    residual = torch.mv(K_dense, pred_dt) - f_shifted
    f_norm_sq = (f_shifted**2).mean() + 1e-8
    return (residual**2).mean() / f_norm_sq


def train_model_pinn(
    model: nn.Module,
    train_data: list[Data],
    val_data: list[Data],
    *,
    lambda_phys: float = 0.1,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    normalize_target: bool = True,
    grad_clip: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Physics-Informed 学習ループ.

    L_total = L_data + λ_phys × L_phys

    サンプル単位で反復（K行列がサンプル固有のためバッチ化不可）。

    Args:
        model: GNNモデル
        train_data: 学習データリスト（K_dense, f_shifted 付き PyG Data）
        val_data: 検証データリスト
        lambda_phys: 物理ロスの重み
        epochs: エポック数
        lr: 学習率
        weight_decay: L2正則化
        normalize_target: ターゲット標準化
        grad_clip: 勾配クリッピング閾値
        verbose: 進捗表示

    Returns:
        history: {
            "train_loss": [...],
            "val_loss": [...],
            "data_loss": [...],
            "phys_loss": [...],
            "y_mean": float,
            "y_std": float,
        }
    """
    y_mean, y_std = 0.0, 1.0
    if normalize_target:
        all_y = torch.cat([d.y for d in train_data])
        y_mean = float(all_y.mean())
        y_std = float(max(all_y.std().item(), 1e-8))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "data_loss": [],
        "phys_loss": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0

        perm = torch.randperm(len(train_data))

        for idx in perm:
            data = train_data[idx.item()]
            optimizer.zero_grad()

            pred = model(data)  # (N, 1) normalized
            target = (data.y - y_mean) / y_std

            # Data loss (MSE on normalized target)
            l_data = criterion(pred, target)

            # Physics loss (on denormalized ΔT)
            dt_pred = pred.squeeze(-1) * y_std + y_mean
            l_phys = compute_physics_loss(dt_pred, data.K_dense, data.f_shifted)

            loss = l_data + lambda_phys * l_phys
            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_data_loss += l_data.item()
            epoch_phys_loss += l_phys.item()

        n = len(train_data)
        avg_data = epoch_data_loss / n
        avg_phys = epoch_phys_loss / n
        history["data_loss"].append(avg_data)
        history["phys_loss"].append(avg_phys)
        history["train_loss"].append(avg_data + lambda_phys * avg_phys)

        # Validate (data loss only)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_data:
                pred = model(data)
                target = (data.y - y_mean) / y_std
                val_loss += criterion(pred, target).item()
        val_loss /= max(len(val_data), 1)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1:4d} | "
                f"Data: {avg_data:.6f} | "
                f"Phys: {avg_phys:.6f} | "
                f"Val: {val_loss:.6f} | "
                f"LR: {current_lr:.2e}"
            )

    history["y_mean"] = y_mean
    history["y_std"] = y_std
    return history
