"""GNNサロゲートモデル用データセット生成.

固定メッシュ上にランダム発熱体を配置 → FEMで温度分布を計算
→ グラフデータ（PyG Data形式）に変換。

問題設定:
- 長方形プレート (Lx × Ly)、固定メッシュ (nx × ny)
- 固定パラメータ: k, h_conv, t, T_inf
- 変数: 発熱体の位置（面内ランダム座標）
- 発熱体は矩形で面積が一定（w_heat × h_heat）
- 1サンプルにつき n_sources 個の発熱体を配置
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xkep_cae.thermal.fem import (
    assemble_thermal_system,
    compute_heat_flux,
    make_rect_mesh,
    solve_steady_thermal,
)


@dataclass
class ThermalProblemConfig:
    """熱伝導問題の設定."""

    Lx: float = 0.1  # [m] x方向長さ
    Ly: float = 0.1  # [m] y方向長さ
    nx: int = 20  # x方向要素数
    ny: int = 20  # y方向要素数
    k: float = 200.0  # [W/(m·K)] 熱伝導率（アルミニウム相当）
    h_conv: float = 25.0  # [W/(m²·K)] 対流熱伝達率
    t: float = 0.002  # [m] 板厚
    T_inf: float = 25.0  # [°C] 周囲温度
    q_value: float = 5.0e6  # [W/m³] 発熱密度
    w_heat: float = 0.01  # [m] 発熱体のx方向サイズ
    h_heat: float = 0.01  # [m] 発熱体のy方向サイズ
    n_sources_min: int = 1  # 最小発熱体数
    n_sources_max: int = 5  # 最大発熱体数


def place_heat_sources(
    nodes: np.ndarray,
    config: ThermalProblemConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """ランダム発熱体を配置し、節点ごとの発熱密度を返す.

    Args:
        nodes: (N, 2) 節点座標
        config: 問題設定
        rng: 乱数生成器

    Returns:
        q_nodal: (N,) 節点ごとの発熱密度 [W/m³]
        centers: 発熱体中心座標リスト [(cx, cy), ...]
    """
    N = len(nodes)
    q_nodal = np.zeros(N)
    n_sources = rng.integers(config.n_sources_min, config.n_sources_max + 1)

    # 発熱体の中心座標をランダム生成（発熱体がプレート内に収まるように）
    margin_x = config.w_heat / 2
    margin_y = config.h_heat / 2
    centers = []
    for _ in range(n_sources):
        cx = rng.uniform(margin_x, config.Lx - margin_x)
        cy = rng.uniform(margin_y, config.Ly - margin_y)
        centers.append((cx, cy))

        # 発熱体領域内の節点に発熱を割当
        in_source = (
            (nodes[:, 0] >= cx - config.w_heat / 2)
            & (nodes[:, 0] <= cx + config.w_heat / 2)
            & (nodes[:, 1] >= cy - config.h_heat / 2)
            & (nodes[:, 1] <= cy + config.h_heat / 2)
        )
        q_nodal[in_source] = config.q_value

    return q_nodal, centers


def generate_single_sample(
    nodes: np.ndarray,
    conn: np.ndarray,
    boundary_edges: dict[str, np.ndarray],
    config: ThermalProblemConfig,
    rng: np.random.Generator,
) -> dict:
    """1サンプルのFEM計算を実行.

    Returns:
        dict with keys:
            "q_nodal": (N,) 節点発熱密度
            "T": (N,) 温度分布
            "flux": (Ne, 2) 要素中心熱流束
            "centers": 発熱体中心座標リスト
            "nodes": (N, 2) 節点座標
            "conn": (Ne, 4) 接続配列
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
    flux = compute_heat_flux(nodes, conn, T, config.k, config.t)

    return {
        "q_nodal": q_nodal,
        "T": T,
        "flux": flux,
        "centers": centers,
        "nodes": nodes,
        "conn": conn,
    }


def mesh_to_edge_index(conn: np.ndarray) -> np.ndarray:
    """要素接続からグラフのエッジインデックスを生成.

    Q4要素の辺接続から無向グラフのエッジリストを構築。

    Args:
        conn: (Ne, 4) 接続配列

    Returns:
        edge_index: (2, E) エッジインデックス（双方向）
    """
    edge_set = set()
    for elem in conn:
        n = len(elem)
        for i in range(n):
            j = (i + 1) % n
            a, b = int(elem[i]), int(elem[j])
            if a > b:
                a, b = b, a
            edge_set.add((a, b))

    edges = np.array(sorted(edge_set), dtype=np.int64)
    # 双方向
    edge_index = np.concatenate([edges, edges[:, ::-1]], axis=0).T
    return edge_index


def sample_to_graph_data(sample: dict) -> dict:
    """FEMサンプルをGNN入力形式に変換.

    ノード特徴量 (6次元):
        - x座標 (正規化)
        - y座標 (正規化)
        - 発熱密度 (バイナリ: 0/1)
        - 最近接境界までの距離 (正規化)
        - 境界フラグ (0/1)
        - 発熱ポテンシャル (Σ 1/(r² + ε), 正規化)

    ターゲット:
        - 温度上昇 ΔT = T - T_inf

    Returns:
        dict with keys:
            "x": (N, 6) ノード特徴量
            "edge_index": (2, E) エッジインデックス
            "y": (N, 1) ターゲット（温度上昇）
    """
    nodes = sample["nodes"]
    q = sample["q_nodal"]
    T = sample["T"]

    Lx = nodes[:, 0].max()
    Ly = nodes[:, 1].max()
    q_max = q.max() if q.max() > 0 else 1.0

    # 境界距離: min(x, Lx-x, y, Ly-y)
    dist_to_boundary = np.minimum(
        np.minimum(nodes[:, 0], Lx - nodes[:, 0]),
        np.minimum(nodes[:, 1], Ly - nodes[:, 1]),
    )
    max_dist = min(Lx, Ly) / 2.0

    # 境界フラグ
    tol = 1e-10
    on_boundary = (
        (nodes[:, 0] < tol)
        | (nodes[:, 0] > Lx - tol)
        | (nodes[:, 1] < tol)
        | (nodes[:, 1] > Ly - tol)
    ).astype(float)

    # 発熱ポテンシャル: 各ノードから全発熱ノードへの 1/(r²+ε) の総和
    heat_potential = np.zeros(len(nodes))
    heat_nodes = np.where(q > 0)[0]
    if len(heat_nodes) > 0:
        eps = (min(Lx, Ly) * 0.05) ** 2
        for hi in heat_nodes:
            r2 = np.sum((nodes - nodes[hi]) ** 2, axis=1)
            heat_potential += 1.0 / (r2 + eps)
        hp_max = heat_potential.max()
        if hp_max > 0:
            heat_potential /= hp_max

    # ノード特徴量: 正規化座標 + 発熱密度 + 境界距離 + 境界フラグ + ポテンシャル
    x_feat = np.column_stack(
        [
            nodes[:, 0] / Lx,
            nodes[:, 1] / Ly,
            q / q_max,
            dist_to_boundary / max_dist,
            on_boundary,
            heat_potential,
        ]
    )

    edge_index = mesh_to_edge_index(sample["conn"])

    # ターゲット: 温度上昇 ΔT
    T_inf = T.min()  # 近似的に T_inf
    y = (T - T_inf).reshape(-1, 1)

    return {
        "x": x_feat.astype(np.float32),
        "edge_index": edge_index,
        "y": y.astype(np.float32),
    }


def generate_dataset(
    config: ThermalProblemConfig,
    n_samples: int,
    seed: int = 42,
) -> list[dict]:
    """データセット生成.

    Args:
        config: 問題設定
        n_samples: サンプル数
        seed: 乱数シード

    Returns:
        list of graph data dicts
    """
    rng = np.random.default_rng(seed)
    nodes, conn, edges = make_rect_mesh(config.Lx, config.Ly, config.nx, config.ny)

    dataset = []
    for _ in range(n_samples):
        sample = generate_single_sample(nodes, conn, edges, config, rng)
        graph = sample_to_graph_data(sample)
        dataset.append(graph)

    return dataset
