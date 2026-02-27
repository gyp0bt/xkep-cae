"""k_pen推定MLモデル用特徴量抽出ユーティリティ.

設計仕様: docs/contact/kpen-estimation-ml-design.md (Step 1-2)

接触解析ケースから k_pen 推定用の特徴量ベクトル (12D) を抽出する。
グリッドサーチで最適 k_pen を同定し、学習データを生成するための
パイプラインを提供する。

特徴量 (12D):
    # 材料・断面（4D）
    log10(E), log10(I), log10(12EI/L³), r/L

    # メッシュ特性（3D）
    log10(n_segments), n_wires, lay_angle

    # 接触幾何（5D）
    log10(n_active_est+1), gap_mean/r, gap_std/r,
    frac_near_contact, cos_angle_mean
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.broadphase import broadphase_aabb
from xkep_cae.contact.geometry import closest_point_segments
from xkep_cae.contact.prescreening_data import extract_segments


def extract_kpen_features(
    *,
    E: float,
    Iy: float,
    L_elem: float,
    r_contact: float,
    n_segments_per_wire: int,
    n_wires: int,
    lay_angle: float,
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    radii: np.ndarray,
    u: np.ndarray | None = None,
    ndof_per_node: int = 6,
    broadphase_margin: float = 0.01,
) -> np.ndarray:
    """k_pen推定用の特徴量ベクトル (12D) を抽出する.

    Args:
        E: ヤング率
        Iy: 断面二次モーメント
        L_elem: 要素長さ
        r_contact: 接触半径（被膜込み）
        n_segments_per_wire: 素線あたりセグメント数
        n_wires: 素線数
        lay_angle: 撚り角 [rad]
        node_coords: (N, 3) 節点座標
        connectivity: (N_elem, 2) 接続配列
        radii: (N_seg,) 各セグメントの接触半径
        u: 変位ベクトル（None の場合は初期形状）
        ndof_per_node: 1節点あたりの自由度数
        broadphase_margin: broadphase 探索マージン

    Returns:
        (12,) 特徴量ベクトル
    """
    segments = extract_segments(node_coords, connectivity, u, ndof_per_node=ndof_per_node)

    candidates = broadphase_aabb(segments, radii, margin=broadphase_margin)

    # 接触幾何の統計量を計算
    gaps = []
    cos_angles = []
    for i_seg, j_seg in candidates:
        result = closest_point_segments(
            segments[i_seg][0],
            segments[i_seg][1],
            segments[j_seg][0],
            segments[j_seg][1],
        )
        gap = result.distance - (radii[i_seg] + radii[j_seg])
        gaps.append(gap)

        # セグメント間角度
        dir_i = segments[i_seg][1] - segments[i_seg][0]
        dir_j = segments[j_seg][1] - segments[j_seg][0]
        li = np.linalg.norm(dir_i)
        lj = np.linalg.norm(dir_j)
        if li > 1e-30 and lj > 1e-30:
            cos_angles.append(abs(float(dir_i @ dir_j) / (li * lj)))
        else:
            cos_angles.append(0.0)

    gaps_arr = np.array(gaps) if gaps else np.array([0.0])
    cos_arr = np.array(cos_angles) if cos_angles else np.array([0.0])

    # 材料・断面 (4D)
    ref_stiffness = 12.0 * E * Iy / L_elem**3
    feat_material = np.array(
        [
            np.log10(max(E, 1e-30)),
            np.log10(max(Iy, 1e-30)),
            np.log10(max(ref_stiffness, 1e-30)),
            r_contact / max(L_elem, 1e-30),
        ]
    )

    # メッシュ特性 (3D)
    feat_mesh = np.array(
        [
            np.log10(max(n_segments_per_wire, 1)),
            float(n_wires),
            lay_angle,
        ]
    )

    # 接触幾何 (5D)
    n_active_est = np.sum(gaps_arr < 0)  # 貫通しているペア数
    gap_mean = float(np.mean(gaps_arr))
    gap_std = float(np.std(gaps_arr))
    frac_near = float(np.mean(gaps_arr < 2.0 * r_contact))

    feat_contact = np.array(
        [
            np.log10(max(n_active_est + 1, 1)),
            gap_mean / max(r_contact, 1e-30),
            gap_std / max(r_contact, 1e-30),
            frac_near,
            float(np.mean(cos_arr)),
        ]
    )

    return np.concatenate([feat_material, feat_mesh, feat_contact]).astype(np.float32)


def extract_kpen_features_from_mesh(
    mesh,
    *,
    E: float,
    Iy: float,
    r_contact: float | None = None,
    u: np.ndarray | None = None,
    ndof_per_node: int = 6,
    broadphase_margin: float = 0.01,
) -> np.ndarray:
    """TwistedWireMesh から k_pen 特徴量を抽出する（高レベルAPI）.

    Args:
        mesh: TwistedWireMesh インスタンス
        E: ヤング率
        Iy: 断面二次モーメント
        r_contact: 接触半径（None の場合は mesh.radii[0] を使用）
        u: 変位ベクトル
        ndof_per_node: 1節点あたりの自由度数
        broadphase_margin: broadphase 探索マージン

    Returns:
        (12,) 特徴量ベクトル
    """
    if r_contact is None:
        r_contact = float(mesh.radii[0])

    # 要素長さの推定（最初のセグメント）
    na, nb = mesh.connectivity[0]
    L_elem = float(np.linalg.norm(mesh.node_coords[nb] - mesh.node_coords[na]))

    # 撚り角の推定
    if hasattr(mesh, "lay_angles") and mesh.lay_angles:
        lay_angle = float(mesh.lay_angles[0])
    else:
        lay_angle = 0.0

    n_segments_per_wire = len(mesh.connectivity) // max(mesh.n_strands, 1)

    return extract_kpen_features(
        E=E,
        Iy=Iy,
        L_elem=L_elem,
        r_contact=r_contact,
        n_segments_per_wire=n_segments_per_wire,
        n_wires=mesh.n_strands,
        lay_angle=lay_angle,
        node_coords=mesh.node_coords,
        connectivity=mesh.connectivity,
        radii=mesh.radii,
        u=u,
        ndof_per_node=ndof_per_node,
        broadphase_margin=broadphase_margin,
    )
