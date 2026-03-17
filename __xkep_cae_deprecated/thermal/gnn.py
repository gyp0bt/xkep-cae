"""GNNサロゲートモデル — メッシュベースの温度場予測.

アーキテクチャ:
  入力: ノード特徴量 (座標 + 発熱密度) → エンコーダ MLP
  → GraphNet (message passing) × L 層
  → デコーダ MLP → 温度上昇 ΔT

Message Passing 方式:
  m_ij = MLP_msg(x_i || x_j || e_ij)
  x_i' = MLP_upd(x_i || Σ_j m_ij)

エッジ特徴量:
  e_ij = [Δx/Lx, Δy/Ly, ||Δr||/L_diag]
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


class EdgeConvLayer(MessagePassing):
    """エッジ情報付きメッセージパッシング層（dropout対応）."""

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, dropout: float = 0.0):
        super().__init__(aggr="add")
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.upd_mlp(torch.cat([x, out], dim=-1))
        return self.norm(out + x) if x.shape[-1] == out.shape[-1] else self.norm(out)

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class ThermalGNN(nn.Module):
    """メッシュベースGNNサロゲートモデル.

    Args:
        node_in_dim: ノード入力特徴量の次元 (default=3: x, y, q)
        edge_dim: エッジ特徴量の次元 (default=3: dx, dy, dist)
        hidden_dim: 隠れ層の次元
        n_layers: メッセージパッシング層数
    """

    def __init__(
        self,
        node_in_dim: int = 3,
        edge_dim: int = 3,
        hidden_dim: int = 64,
        n_layers: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                EdgeConvLayer(hidden_dim, hidden_dim, edge_dim, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.encoder(data.x)
        for layer in self.layers:
            x = layer(x, data.edge_index, data.edge_attr)
        return self.decoder(x)


def compute_edge_attr(
    nodes: np.ndarray,
    edge_index: np.ndarray,
    Lx: float,
    Ly: float,
) -> np.ndarray:
    """エッジ特徴量を計算.

    Args:
        nodes: (N, 2) 節点座標
        edge_index: (2, E) エッジインデックス
        Lx, Ly: 正規化用の長さ

    Returns:
        edge_attr: (E, 3) [dx/Lx, dy/Ly, dist/L_diag]
    """
    L_diag = np.sqrt(Lx**2 + Ly**2)
    src = edge_index[0]
    dst = edge_index[1]
    dx = nodes[dst, 0] - nodes[src, 0]
    dy = nodes[dst, 1] - nodes[src, 1]
    dist = np.sqrt(dx**2 + dy**2)
    return np.column_stack([dx / Lx, dy / Ly, dist / L_diag]).astype(np.float32)


def graph_dict_to_pyg(graph_dict: dict, config) -> Data:
    """numpy dict → PyG Data に変換.

    Args:
        graph_dict: dataset.sample_to_graph_data() の出力
        config: ThermalProblemConfig

    Returns:
        PyG Data オブジェクト
    """
    nodes_xy = np.column_stack(
        [
            graph_dict["x"][:, 0] * config.Lx,
            graph_dict["x"][:, 1] * config.Ly,
        ]
    )
    edge_attr = compute_edge_attr(nodes_xy, graph_dict["edge_index"], config.Lx, config.Ly)
    return Data(
        x=torch.from_numpy(graph_dict["x"]),
        edge_index=torch.from_numpy(graph_dict["edge_index"]),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.from_numpy(graph_dict["y"]),
    )


def _compute_target_stats(data_list: list[Data]) -> tuple[float, float]:
    """学習データのターゲット統計量を計算."""
    all_y = torch.cat([d.y for d in data_list])
    return float(all_y.mean()), float(all_y.std())


def train_model(
    model: ThermalGNN,
    train_data: list[Data],
    val_data: list[Data],
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    normalize_target: bool = True,
    grad_clip: float = 1.0,
    verbose: bool = True,
) -> dict:
    """GNNモデルの学習（DataLoader バッチ処理 + ターゲット正規化）.

    Args:
        model: ThermalGNN モデル
        train_data: 学習データリスト
        val_data: 検証データリスト
        epochs: エポック数
        lr: 学習率
        weight_decay: L2正則化
        batch_size: ミニバッチサイズ
        normalize_target: ターゲットを標準化するか
        grad_clip: 勾配クリッピングの閾値（0以下で無効）
        verbose: 進捗表示

    Returns:
        history: {"train_loss": [...], "val_loss": [...], "y_mean": float, "y_std": float}
    """
    from torch_geometric.loader import DataLoader

    # ターゲット正規化
    y_mean, y_std = 0.0, 1.0
    if normalize_target:
        y_mean, y_std = _compute_target_stats(train_data)
        y_std = max(y_std, 1e-8)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    history: dict[str, list] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch)
            target = (batch.y - y_mean) / y_std
            loss = criterion(pred, target)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / n_batches
        history["train_loss"].append(train_loss)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch)
                target = (batch.y - y_mean) / y_std
                val_loss += criterion(pred, target).item()
                n_val += 1
        val_loss /= max(n_val, 1)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1:4d} | "
                f"Train MSE: {train_loss:.6f} | "
                f"Val MSE: {val_loss:.6f} | "
                f"LR: {current_lr:.2e}"
            )

    history["y_mean"] = y_mean
    history["y_std"] = y_std
    return history


def evaluate_model(
    model: ThermalGNN,
    test_data: list[Data],
    y_mean: float = 0.0,
    y_std: float = 1.0,
) -> dict:
    """モデルの評価指標を計算.

    Args:
        model: 学習済みモデル
        test_data: テストデータリスト
        y_mean: ターゲット平均（学習時の正規化に対応）
        y_std: ターゲット標準偏差

    Returns:
        dict with keys:
            "mse": 平均二乗誤差
            "mae": 平均絶対誤差
            "r2": 決定係数
            "max_error": 最大誤差
            "relative_error_pct": 相対誤差の平均 [%]
    """
    model.eval()
    all_pred = []
    all_true = []
    with torch.no_grad():
        for data in test_data:
            pred_norm = model(data).numpy().flatten()
            pred = pred_norm * y_std + y_mean  # 逆正規化
            true = data.y.numpy().flatten()
            all_pred.append(pred)
            all_true.append(true)

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    max_err = np.max(np.abs(pred - true))

    # 相対誤差（ΔTが小さい節点を除外）
    mask = true > true.max() * 0.01
    if np.any(mask):
        rel_err = np.mean(np.abs(pred[mask] - true[mask]) / true[mask]) * 100
    else:
        rel_err = 0.0

    return {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "max_error": float(max_err),
        "relative_error_pct": float(rel_err),
    }
