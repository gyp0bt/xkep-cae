"""簡略化全結合GNNサロゲートモデル.

メッシュのエッジ構造に依存せず、全ノード間の全結合グラフで
長距離相互作用を1ホップで伝搬する。

改善版アーキテクチャ（v2）:
  入力: ノード特徴量 (6D) → エンコーダ MLP
  → FCAttentionLayer × L 層（ソフトアテンション + 距離カーネルエッジ特徴量）
  → デコーダ MLP → 温度上昇 ΔT

v1 からの改善:
  1. mean集約 → アテンション重み付きadd集約（重要なノードに集中）
  2. エッジ特徴量に距離カーネル追加（逆距離、ガウシアン2スケール）→ 6D
  3. メッセージ = attention_weight × value（GAT類似の構造）
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class FCAttentionLayer(MessagePassing):
    """アテンション付き全結合メッセージパッシング層.

    各ノードが全ノードからメッセージを受け取る際に、
    学習されたアテンション重みで重要なノードに集中する。

    attention_ij = softmax_j( MLP_att([x_i || x_j || e_ij]) )
    value_ij     = MLP_val([x_j || e_ij])
    agg_i        = Σ_j attention_ij * value_ij
    x_i'         = LayerNorm( MLP_upd([x_i || agg_i]) + x_i )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr="add")
        # アテンションスコア: 送受信ノード + エッジ情報から1スカラー
        self.att_net = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels, 1),
        )
        # バリュー: 送信ノード + エッジ情報から特徴量
        self.val_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )
        # 更新
        self.upd_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )
        self.norm = nn.LayerNorm(out_channels)
        self.att_dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.upd_mlp(torch.cat([x, out], dim=-1))
        return self.norm(out + x) if x.shape[-1] == out.shape[-1] else self.norm(out)

    def message(self, x_i, x_j, edge_attr, index):
        # アテンションスコア（受信ノード index でsoftmax）
        alpha = self.att_net(torch.cat([x_i, x_j, edge_attr], dim=-1))
        alpha = softmax(alpha, index)
        alpha = self.att_dropout(alpha)
        # バリュー（送信ノードの情報 + エッジ距離情報）
        val = self.val_mlp(torch.cat([x_j, edge_attr], dim=-1))
        return val * alpha


class FullyConnectedThermalGNN(nn.Module):
    """全結合グラフベースの温度場サロゲートモデル（アテンション版）.

    Args:
        node_in_dim: ノード入力特徴量の次元
        edge_dim: エッジ特徴量の次元 (v2: 6D = 位置差分3 + 距離カーネル3)
        hidden_dim: 隠れ層の次元
        n_layers: メッセージパッシング層数
        dropout: ドロップアウト率
    """

    def __init__(
        self,
        node_in_dim: int = 6,
        edge_dim: int = 6,
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
                FCAttentionLayer(hidden_dim, hidden_dim, edge_dim, dropout=dropout)
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
    """全結合エッジの特徴量を計算（距離カーネル付き 6D）.

    特徴量:
        0: dx/Lx          — x方向変位（正規化）
        1: dy/Ly          — y方向変位（正規化）
        2: dist/L_diag    — ユークリッド距離（正規化）
        3: 1/(dist_n+0.1) — 逆距離カーネル（Green関数的減衰）
        4: exp(-dist_n²/0.05) — 狭域ガウシアンカーネル（近距離重視）
        5: exp(-dist_n²/0.5)  — 広域ガウシアンカーネル（中距離まで）

    Args:
        nodes: (N, 2) 節点座標
        edge_index: (2, E) エッジインデックス
        Lx, Ly: 正規化用の長さ

    Returns:
        edge_attr: (E, 6)
    """
    L_diag = np.sqrt(Lx**2 + Ly**2)
    src = edge_index[0]
    dst = edge_index[1]
    dx = nodes[dst, 0] - nodes[src, 0]
    dy = nodes[dst, 1] - nodes[src, 1]
    dist = np.sqrt(dx**2 + dy**2)
    dist_n = dist / L_diag

    # 距離カーネル
    inv_dist = 1.0 / (dist_n + 0.1)
    gauss_narrow = np.exp(-(dist_n**2) / 0.05)
    gauss_wide = np.exp(-(dist_n**2) / 0.5)

    return np.column_stack([dx / Lx, dy / Ly, dist_n, inv_dist, gauss_narrow, gauss_wide]).astype(
        np.float32
    )


def graph_dict_to_pyg_fc(graph_dict: dict, config) -> Data:
    """numpy dict → 全結合グラフの PyG Data に変換.

    メッシュエッジの代わりに全ノード間の全結合エッジを使用。
    エッジ特徴量に距離カーネルを含む（6D）。

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


# ---------------------------------------------------------------------------
# ハイブリッドグラフ: メッシュエッジ + 長距離ショートカット
# ---------------------------------------------------------------------------


def graph_dict_to_pyg_hybrid(graph_dict: dict, config) -> Data:
    """numpy dict → ハイブリッドグラフの PyG Data に変換.

    メッシュのQ4エッジ（局所伝搬）に加え、発熱ノード↔全ノードの
    長距離ショートカットエッジを追加。

    エッジ構成:
        - メッシュエッジ（Q4要素辺、双方向）
        - 発熱ノード → 全ノード（双方向、メッシュエッジと重複しない分のみ）
    エッジ特徴量は全結合版と同じ 6D（距離カーネル付き）。

    Args:
        graph_dict: dataset.sample_to_graph_data() の出力
        config: ThermalProblemConfig

    Returns:
        PyG Data オブジェクト（ハイブリッドエッジ付き）
    """
    n_nodes = graph_dict["x"].shape[0]
    nodes_xy = np.column_stack(
        [
            graph_dict["x"][:, 0] * config.Lx,
            graph_dict["x"][:, 1] * config.Ly,
        ]
    )

    # 1) メッシュエッジ（既存の Q4 接続ベース）
    mesh_ei = graph_dict["edge_index"]  # (2, E_mesh)
    mesh_edge_set = set(zip(mesh_ei[0].tolist(), mesh_ei[1].tolist(), strict=True))

    # 2) 発熱ノードの特定（特徴量 index=2 = 正規化発熱密度）
    q_feat = graph_dict["x"][:, 2]
    heat_node_ids = np.where(q_feat > 0.5)[0]

    # 3) 長距離ショートカット: 発熱ノード ↔ 全ノード（メッシュ重複除外）
    lr_src, lr_dst = [], []
    for h in heat_node_ids:
        for j in range(n_nodes):
            if h != j:
                # 双方向で追加（メッシュにない分だけ）
                if (int(h), j) not in mesh_edge_set:
                    lr_src.append(int(h))
                    lr_dst.append(j)
                if (j, int(h)) not in mesh_edge_set:
                    lr_src.append(j)
                    lr_dst.append(int(h))

    # 4) エッジ統合（メッシュ + 長距離ショートカット、重複除去）
    if lr_src:
        lr_ei = np.array([lr_src, lr_dst], dtype=np.int64)
        # 長距離エッジ内の重複を除去
        lr_pairs = set()
        unique_lr_src, unique_lr_dst = [], []
        for s, d in zip(lr_ei[0], lr_ei[1], strict=True):
            if (s, d) not in lr_pairs:
                lr_pairs.add((s, d))
                unique_lr_src.append(s)
                unique_lr_dst.append(d)
        lr_ei = np.array([unique_lr_src, unique_lr_dst], dtype=np.int64)
        combined_ei = np.concatenate([mesh_ei, lr_ei], axis=1)
    else:
        combined_ei = mesh_ei

    # 5) エッジ特徴量（距離カーネル付き 6D）
    edge_attr = compute_fc_edge_attr(nodes_xy, combined_ei, config.Lx, config.Ly)

    return Data(
        x=torch.from_numpy(graph_dict["x"]),
        edge_index=torch.from_numpy(combined_ei),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.from_numpy(graph_dict["y"]),
    )
