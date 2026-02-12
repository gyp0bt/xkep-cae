"""高レベルAPI: ラベルベースの平面ひずみソルバー.

Protocol ベースの要素・材料・アセンブリを使い、
ラベル指定の節点・要素・境界条件から変位を解く。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae.assembly import assemble_global_stiffness
from xkep_cae.elements.quad4 import Quad4PlaneStrain
from xkep_cae.elements.tri3 import Tri3PlaneStrain
from xkep_cae.elements.tri6 import Tri6PlaneStrain
from xkep_cae.materials.elastic import PlaneStrainElastic


def solve_plane_strain(
    node_coord_array: np.ndarray,
    node_label_df_mapping: dict[int, tuple[bool, bool]],
    node_label_load_mapping: dict[int, tuple[float, float]],
    E: float,
    nu: float,
    *,
    thickness: float = 1.0,
    elem_quads: np.ndarray | None = None,
    elem_tris: np.ndarray | None = None,
    elem_tri6: np.ndarray | None = None,
    size_threshold: int = 4000,
) -> dict[int, tuple[float, float]]:
    """ラベル指定の境界条件・荷重から、ラベル→変位のマッピングを解く高レベルAPI.

    Q4 / TRI3 / TRI6 混在の平面歪み・線形弾性を仮定する。
    Protocol ベースの要素・材料・アセンブリを内部で使用する。

    Args:
        node_coord_array: (N, 3+) 先頭列が label、続く x, y [, z]
        node_label_df_mapping: {label: (ux_free, uy_free)} Dirichlet BC
        node_label_load_mapping: {label: (Fx, Fy)} 節点荷重
        E: ヤング率
        nu: ポアソン比
        thickness: 厚み
        elem_quads: (Ne4, 4) Q4 節点ラベル配列。None 可。
        elem_tris: (Ne3, 3) TRI3 節点ラベル配列。None 可。
        elem_tri6: (Ne6, 6) TRI6 節点ラベル配列。None 可。
        size_threshold: AMG ソルバーを使う DOF 数閾値

    Returns:
        {label: (ux, uy)} ラベル→変位マッピング
    """
    has_q = elem_quads is not None and len(elem_quads) > 0
    has_t3 = elem_tris is not None and len(elem_tris) > 0
    has_t6 = elem_tri6 is not None and len(elem_tri6) > 0
    if not (has_q or has_t3 or has_t6):
        raise ValueError(
            "要素が空です。elem_quads / elem_tris / elem_tri6 のいずれかを指定してください。"
        )

    # ラベル→内部インデックスのマッピング構築
    used_label_list: list[np.ndarray] = []
    if has_q:
        used_label_list.append(np.asarray(elem_quads, dtype=int).ravel())
    if has_t3:
        used_label_list.append(np.asarray(elem_tris, dtype=int).ravel())
    if has_t6:
        used_label_list.append(np.asarray(elem_tri6, dtype=int).ravel())

    used_labels = np.unique(np.concatenate(used_label_list))
    used_labels.sort()

    labels = node_coord_array[:, 0].astype(int)
    label_to_row: dict[int, int] = {int(lab): i for i, lab in enumerate(labels)}
    label_to_new: dict[int, int] = {int(lab): i for i, lab in enumerate(used_labels)}

    # nodes_xy（内部インデックス順）
    nodes_xy = np.empty((used_labels.size, 2), dtype=float)
    for lab, idx in label_to_new.items():
        r = label_to_row[lab]
        nodes_xy[idx, 0] = float(node_coord_array[r, 1])
        nodes_xy[idx, 1] = float(node_coord_array[r, 2])

    # Protocol ベースの要素・材料
    mat = PlaneStrainElastic(E, nu)
    element_groups: list[
        tuple[Quad4PlaneStrain | Tri3PlaneStrain | Tri6PlaneStrain, np.ndarray]
    ] = []

    if has_q:
        q = np.asarray(elem_quads, int)
        conn_q = np.vectorize(label_to_new.get)(q)
        element_groups.append((Quad4PlaneStrain(), conn_q))

    if has_t3:
        t_arr = np.asarray(elem_tris, int)
        conn_t3 = np.vectorize(label_to_new.get)(t_arr)
        element_groups.append((Tri3PlaneStrain(), conn_t3))

    if has_t6:
        t6_arr = np.asarray(elem_tri6, int)
        conn_t6 = np.vectorize(label_to_new.get)(t6_arr)
        element_groups.append((Tri6PlaneStrain(), conn_t6))

    # アセンブリ
    K: sp.csr_matrix = assemble_global_stiffness(
        nodes_xy,
        element_groups,
        mat,
        thickness=thickness,
    )

    ndof = K.shape[0]
    if ndof != 2 * used_labels.size:
        raise RuntimeError("内部DOF数と使用ラベル数が整合していません。")

    # 荷重ベクトル
    f = np.zeros(ndof, dtype=float)
    for lab, (Fx, Fy) in node_label_load_mapping.items():
        lab_int = int(lab)
        if lab_int not in label_to_new:
            continue
        i = label_to_new[lab_int]
        f[2 * i] += float(Fx)
        f[2 * i + 1] += float(Fy)

    # Dirichlet 境界条件
    fixed_dofs: list[int] = []
    for lab, (ux_free, uy_free) in node_label_df_mapping.items():
        lab_int = int(lab)
        if lab_int not in label_to_new:
            continue
        i = label_to_new[lab_int]
        if not ux_free:
            fixed_dofs.append(2 * i)
        if not uy_free:
            fixed_dofs.append(2 * i + 1)

    fixed_dofs_arr = np.array(sorted(set(fixed_dofs)), dtype=int)

    # ソルブ
    from xkep_cae.bc import apply_dirichlet
    from xkep_cae.solver import solve_displacement

    Kbc, fbc = apply_dirichlet(K, f, fixed_dofs_arr, values=0.0)
    u, info = solve_displacement(Kbc, fbc, size_threshold=size_threshold, use_pyamg=True)

    # 解ベクトル → ラベル辞書へマッピング
    uN2 = u.reshape(-1, 2)
    node_label_displacement_mapping: dict[int, tuple[float, float]] = {}
    for lab, idx in label_to_new.items():
        ux, uy = uN2[idx, 0], uN2[idx, 1]
        node_label_displacement_mapping[int(lab)] = (float(ux), float(uy))

    return node_label_displacement_mapping
