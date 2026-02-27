"""接触プリスクリーニングGNN用データ生成パイプライン.

設計仕様: docs/contact/contact-prescreening-gnn-design.md (Step 1)

撚線テストケースからGNN学習用の接触ラベル付きグラフデータを
自動生成する。各セグメントペアに対し、実際に接触（gap < threshold）
したか否かのバイナリラベルを付与する。

入力: 撚線メッシュ + 変形後のセグメント座標
出力: {
    "node_features": (N_seg, 10),  # セグメント特徴量
    "edge_index":    (2, E),        # 候補ペア（broadphase出力）
    "edge_features": (E, 7),        # エッジ特徴量
    "labels":        (E,),          # 接触ラベル（0 or 1）
}
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.broadphase import broadphase_aabb
from xkep_cae.contact.geometry import closest_point_segments


def extract_segments(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    u: np.ndarray | None = None,
    *,
    ndof_per_node: int = 6,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """メッシュからセグメント端点を抽出する.

    Args:
        node_coords: (N_nodes, 3) 初期節点座標
        connectivity: (N_elem, 2) 接続配列（梁要素 [node_a, node_b]）
        u: (N_nodes * ndof_per_node,) 変位ベクトル（None の場合は初期形状）
        ndof_per_node: 1節点あたりの自由度数

    Returns:
        [(x0, x1), ...] 各セグメントの変形後端点座標
    """
    segments = []
    for na, nb in connectivity:
        x0 = node_coords[na].copy()
        x1 = node_coords[nb].copy()
        if u is not None:
            x0 += u[na * ndof_per_node : na * ndof_per_node + 3]
            x1 += u[nb * ndof_per_node : nb * ndof_per_node + 3]
        segments.append((x0, x1))
    return segments


def compute_segment_features(
    segments: list[tuple[np.ndarray, np.ndarray]],
    radii: np.ndarray,
    *,
    wire_ids: np.ndarray | None = None,
    layer_ids: np.ndarray | None = None,
) -> np.ndarray:
    """セグメント特徴量を計算する.

    特徴量 (10D):
        [x, y, z, dx, dy, dz, L_seg, r_contact, wire_id, layer_id]

    Args:
        segments: セグメント端点リスト
        radii: (N_seg,) 各セグメントの接触半径
        wire_ids: (N_seg,) 素線ID（None の場合は 0）
        layer_ids: (N_seg,) 層番号（None の場合は 0）

    Returns:
        (N_seg, 10) 特徴量配列
    """
    n = len(segments)
    features = np.zeros((n, 10), dtype=np.float32)

    for i, (x0, x1) in enumerate(segments):
        mid = 0.5 * (x0 + x1)
        direction = x1 - x0
        length = np.linalg.norm(direction)
        if length > 1e-30:
            direction = direction / length

        features[i, 0:3] = mid  # x, y, z
        features[i, 3:6] = direction  # dx, dy, dz
        features[i, 6] = length  # L_seg
        features[i, 7] = radii[i] if i < len(radii) else 0.0  # r_contact
        features[i, 8] = wire_ids[i] if wire_ids is not None else 0
        features[i, 9] = layer_ids[i] if layer_ids is not None else 0

    return features


def compute_edge_features(
    segments: list[tuple[np.ndarray, np.ndarray]],
    candidates: list[tuple[int, int]],
    *,
    wire_ids: np.ndarray | None = None,
    layer_ids: np.ndarray | None = None,
) -> np.ndarray:
    """エッジ特徴量を計算する.

    特徴量 (7D):
        [Δx, Δy, Δz, dist, cos_angle, same_wire, same_layer]

    Args:
        segments: セグメント端点リスト
        candidates: 候補ペアリスト [(i, j), ...]
        wire_ids: (N_seg,) 素線ID
        layer_ids: (N_seg,) 層番号

    Returns:
        (E, 7) エッジ特徴量配列
    """
    n_edges = len(candidates)
    edge_feat = np.zeros((n_edges, 7), dtype=np.float32)

    for k, (i, j) in enumerate(candidates):
        mid_i = 0.5 * (segments[i][0] + segments[i][1])
        mid_j = 0.5 * (segments[j][0] + segments[j][1])
        delta = mid_j - mid_i
        dist = np.linalg.norm(delta)

        dir_i = segments[i][1] - segments[i][0]
        li = np.linalg.norm(dir_i)
        dir_j = segments[j][1] - segments[j][0]
        lj = np.linalg.norm(dir_j)
        if li > 1e-30 and lj > 1e-30:
            cos_angle = abs(float(dir_i @ dir_j) / (li * lj))
        else:
            cos_angle = 0.0

        edge_feat[k, 0:3] = delta
        edge_feat[k, 3] = dist
        edge_feat[k, 4] = cos_angle
        edge_feat[k, 5] = 1.0 if (wire_ids is not None and wire_ids[i] == wire_ids[j]) else 0.0
        edge_feat[k, 6] = 1.0 if (layer_ids is not None and layer_ids[i] == layer_ids[j]) else 0.0

    return edge_feat


def label_contact_pairs(
    segments: list[tuple[np.ndarray, np.ndarray]],
    candidates: list[tuple[int, int]],
    radii: np.ndarray,
    *,
    gap_threshold_ratio: float = 1.0,
) -> np.ndarray:
    """候補ペアに接触ラベルを付与する.

    gap < threshold（= gap_threshold_ratio * (r_i + r_j)）なら接触(1)、
    それ以外は非接触(0)。

    Args:
        segments: セグメント端点リスト
        candidates: 候補ペアリスト
        radii: (N_seg,) 各セグメントの接触半径
        gap_threshold_ratio: ギャップ閾値の半径比（1.0 = 接触ちょうど）

    Returns:
        (E,) バイナリラベル配列
    """
    labels = np.zeros(len(candidates), dtype=np.int32)
    for k, (i, j) in enumerate(candidates):
        result = closest_point_segments(
            segments[i][0],
            segments[i][1],
            segments[j][0],
            segments[j][1],
        )
        threshold = gap_threshold_ratio * (radii[i] + radii[j])
        if result.distance < threshold:
            labels[k] = 1
    return labels


def generate_prescreening_sample(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    radii: np.ndarray,
    u: np.ndarray | None = None,
    *,
    ndof_per_node: int = 6,
    wire_ids: np.ndarray | None = None,
    layer_ids: np.ndarray | None = None,
    broadphase_margin: float = 0.01,
    gap_threshold_ratio: float = 1.0,
) -> dict:
    """1つの変形状態から接触プリスクリーニング用サンプルを生成する.

    Args:
        node_coords: (N_nodes, 3) 初期節点座標
        connectivity: (N_elem, 2) 接続配列
        radii: (N_seg,) 各セグメントの接触半径
        u: 変位ベクトル
        ndof_per_node: 1節点あたりの自由度数
        wire_ids: (N_seg,) 素線ID
        layer_ids: (N_seg,) 層番号
        broadphase_margin: broadphase 探索マージン
        gap_threshold_ratio: 接触判定の閾値比

    Returns:
        {
            "node_features": (N_seg, 10),
            "edge_index": (2, E),
            "edge_features": (E, 7),
            "labels": (E,),
            "n_contact": int,
            "n_candidates": int,
        }
    """
    segments = extract_segments(node_coords, connectivity, u, ndof_per_node=ndof_per_node)

    candidates = broadphase_aabb(segments, radii, margin=broadphase_margin)

    if not candidates:
        return {
            "node_features": compute_segment_features(
                segments, radii, wire_ids=wire_ids, layer_ids=layer_ids
            ),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_features": np.zeros((0, 7), dtype=np.float32),
            "labels": np.zeros(0, dtype=np.int32),
            "n_contact": 0,
            "n_candidates": 0,
        }

    node_feat = compute_segment_features(segments, radii, wire_ids=wire_ids, layer_ids=layer_ids)

    edge_idx = np.array(candidates, dtype=np.int64).T  # (2, E)

    edge_feat = compute_edge_features(segments, candidates, wire_ids=wire_ids, layer_ids=layer_ids)

    labels = label_contact_pairs(
        segments,
        candidates,
        radii,
        gap_threshold_ratio=gap_threshold_ratio,
    )

    return {
        "node_features": node_feat,
        "edge_index": edge_idx,
        "edge_features": edge_feat,
        "labels": labels,
        "n_contact": int(labels.sum()),
        "n_candidates": len(candidates),
    }
