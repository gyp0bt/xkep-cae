"""簡略化全結合GNNサロゲートモデル.

メッシュのエッジ構造に依存せず、全ノード間の全結合グラフで
長距離相互作用を1ホップで伝搬する。

アーキテクチャ:
  入力: ノード特徴量 (6D) → エンコーダ MLP
  → FullyConnectedLayer × L 層（全ノード間メッセージパッシング）
  → デコーダ MLP → 温度上昇 ΔT

全結合グラフの利点:
  - メッシュの受容野制約を回避（1ホップで任意ノード間情報伝搬）
  - 発熱ポテンシャル特徴量が不要になる可能性

制約:
  - エッジ数 O(N²) → 大規模メッシュ向けではない
  - 固定メッシュ（ノード数一定）が前提
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


class FCEdgeConvLayer(MessagePassing):
    """全結合グラフ用メッセージパッシング層.

    メッシュ版 EdgeConvLayer と同一構造だが、
    全ノード間エッジで動作することを前提とする。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr="mean")  # 全結合 → mean 集約で安定化
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


class FullyConnectedThermalGNN(nn.Module):
    """全結合グラフベースの温度場サロゲートモデル.

    Args:
        node_in_dim: ノード入力特徴量の次元
        edge_dim: エッジ特徴量の次元 (dx, dy, dist)
        hidden_dim: 隠れ層の次元
        n_layers: メッセージパッシング層数
        dropout: ドロップアウト率
    """

    def __init__(
        self,
        node_in_dim: int = 6,
        edge_dim: int = 3,
        hidden_dim: int = 64,
        n_layers: int = 4,
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
                FCEdgeConvLayer(hidden_dim, hidden_dim, edge_dim, dropout=dropout)
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


# ---------------------------------------------------------------------------
# 全結合グラフ構築ユーティリティ
# ---------------------------------------------------------------------------


def make_fc_edge_index(n_nodes: int) -> np.ndarray:
    """N個のノードの全結合エッジインデックスを生成.

    自己ループは含めない。

    Args:
        n_nodes: ノード数

    Returns:
        edge_index: (2, N*(N-1)) 双方向エッジインデックス
    """
    src = []
    dst = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return np.array([src, dst], dtype=np.int64)


def compute_fc_edge_attr(
    nodes: np.ndarray,
    edge_index: np.ndarray,
    Lx: float,
    Ly: float,
) -> np.ndarray:
    """全結合エッジの特徴量を計算.

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


def graph_dict_to_pyg_fc(graph_dict: dict, config) -> Data:
    """numpy dict → 全結合グラフの PyG Data に変換.

    メッシュエッジの代わりに全ノード間の全結合エッジを使用。

    Args:
        graph_dict: dataset.sample_to_graph_data() の出力
        config: ThermalProblemConfig

    Returns:
        PyG Data オブジェクト（全結合エッジ付き）
    """
    n_nodes = graph_dict["x"].shape[0]
    nodes_xy = np.column_stack(
        [
            graph_dict["x"][:, 0] * config.Lx,
            graph_dict["x"][:, 1] * config.Ly,
        ]
    )

    # 全結合エッジ
    fc_edge_index = make_fc_edge_index(n_nodes)
    fc_edge_attr = compute_fc_edge_attr(nodes_xy, fc_edge_index, config.Lx, config.Ly)

    return Data(
        x=torch.from_numpy(graph_dict["x"]),
        edge_index=torch.from_numpy(fc_edge_index),
        edge_attr=torch.from_numpy(fc_edge_attr),
        y=torch.from_numpy(graph_dict["y"]),
    )
