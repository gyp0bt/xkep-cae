from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import scipy.sparse as sp

from typing import Dict, Tuple, Optional
import numpy as np
import scipy.sparse as sp


def assemble_K_from_arrays_mixed(
    elem_quads: np.ndarray | None,
    elem_tris: np.ndarray | None,
    node_coord_array: np.ndarray,
    E: float,
    nu: float,
    *,
    thickness: float = 1.0,
    elem_tri6: np.ndarray | None = None,  # ★追加（kw-only）
) -> sp.csr_matrix:
    """要素（Q4, TRI3, TRI6 混在）と節点配列（label,x,y[,z]）から全体剛性 K を返す。

    ラベル→内部インデックスの詰め直しを行う。登場ラベルのみ使用。

    Args:
        elem_quads: (Ne4,4) 節点ラベルの配列。None 可。
        elem_tris:  (Ne3,3) 節点ラベルの配列。None 可。
        node_coord_array: (N, 3+) 先頭列が label、続く x,y[,z]
        E: ヤング率
        nu: ポアソン比
        thickness: 厚み
        elem_tri6: (Ne6,6) TRI6 の節点ラベル配列。None 可。

    Returns:
        K: CSR剛性（内部インデックス順）
    """
    if (
        (elem_quads is None or len(elem_quads) == 0)
        and (elem_tris is None or len(elem_tris) == 0)
        and (elem_tri6 is None or len(elem_tri6) == 0)  # ★ TRI6 もチェック
    ):
        raise ValueError(
            "要素が空です。Q4 / TRI3 / TRI6 のいずれかを指定してください。"
        )

    labels = node_coord_array[:, 0].astype(int)
    if np.unique(labels).size != labels.size:
        raise ValueError("node_coord_array の label が重複しています。")

    used = []
    if elem_quads is not None and len(elem_quads) > 0:
        used.append(np.asarray(elem_quads, int).ravel())
    if elem_tris is not None and len(elem_tris) > 0:
        used.append(np.asarray(elem_tris, int).ravel())
    if elem_tri6 is not None and len(elem_tri6) > 0:  # ★追加
        used.append(np.asarray(elem_tri6, int).ravel())

    used_labels = np.unique(np.concatenate(used))
    used_labels.sort()

    label_to_row: dict[int, int] = {int(l): i for i, l in enumerate(labels)}
    label_to_new: dict[int, int] = {int(l): i for i, l in enumerate(used_labels)}

    # nodes_xy（内部インデックス順）
    nodes_xy = np.empty((used_labels.size, 2), dtype=float)
    for lab, i in label_to_new.items():
        r = label_to_row[lab]
        nodes_xy[i, 0] = float(node_coord_array[r, 1])
        nodes_xy[i, 1] = float(node_coord_array[r, 2])

    # 接続（内部インデックス化）
    conn_quads_int = None
    conn_tris_int = None
    conn_tri6_int = None  # ★追加

    if elem_quads is not None and len(elem_quads) > 0:
        q = np.asarray(elem_quads, int)
        conn_quads_int = np.vectorize(label_to_new.get)(q)

    if elem_tris is not None and len(elem_tris) > 0:
        t_arr = np.asarray(elem_tris, int)
        conn_tris_int = np.vectorize(label_to_new.get)(t_arr)

    if elem_tri6 is not None and len(elem_tri6) > 0:  # ★追加
        t6_arr = np.asarray(elem_tri6, int)
        conn_tri6_int = np.vectorize(label_to_new.get)(t6_arr)

    from .assembly import assemble_global_stiffness_mixed

    # アセンブリ（CSR）
    # K = assemble_global_stiffness_mixed(
    K = assemble_global_stiffness_mixed(
        nodes_xy,
        conn_quads_int,
        conn_tris_int,
        conn_tri6_int,  # ★追加
        E,
        nu,
        t=thickness,
    )
    return K


def solve_plane_strain_from_label_maps(
    elem_quads: Optional[np.ndarray],
    elem_tris: Optional[np.ndarray],
    node_coord_array: np.ndarray,
    node_label_df_mapping: Dict[int, Tuple[bool, bool]],
    node_label_load_mapping: Dict[int, Tuple[float, float]],
    E: float,
    nu: float,
    *,
    thickness: float = 1.0,
    size_threshold: int = 4000,
    elem_tri6: Optional[np.ndarray] = None,  # ★追加（kw-only）
) -> Dict[int, Tuple[float, float]]:
    """ラベル指定の境界条件・荷重から、ラベル→変位のマッピングを解く高レベルAPI.

    Q4 / TRI3 / TRI6 混在の平面歪み・線形弾性を仮定する。
    """
    # -----------------------------
    # 1) 使用ラベル・マッピング構築
    # -----------------------------
    if (
        (elem_quads is None or len(elem_quads) == 0)
        and (elem_tris is None or len(elem_tris) == 0)
        and (elem_tri6 is None or len(elem_tri6) == 0)  # ★追加
    ):
        raise ValueError(
            "要素が空です。elem_quads / elem_tris / elem_tri6 のいずれかを指定してください。"
        )

    # np.asarray して flatten → 使用ラベル集合
    used_label_list = []
    if elem_quads is not None and len(elem_quads) > 0:
        used_label_list.append(np.asarray(elem_quads, dtype=int).ravel())
    if elem_tris is not None and len(elem_tris) > 0:
        used_label_list.append(np.asarray(elem_tris, dtype=int).ravel())
    if elem_tri6 is not None and len(elem_tri6) > 0:  # ★追加
        used_label_list.append(np.asarray(elem_tri6, dtype=int).ravel())

    used_labels = np.unique(np.concatenate(used_label_list))
    used_labels.sort()  # 内部インデックス順を決める

    # ラベル → 内部節点インデックス
    label_to_index: Dict[int, int] = {int(lab): i for i, lab in enumerate(used_labels)}

    # -----------------------------
    # 2) 全体剛性 K を構築（既存API使用）
    # -----------------------------
    K: sp.csr_matrix = assemble_K_from_arrays_mixed(
        elem_quads=elem_quads,
        elem_tris=elem_tris,
        node_coord_array=node_coord_array,
        E=E,
        nu=nu,
        thickness=thickness,
        elem_tri6=elem_tri6,  # ★追加
    )

    ndof = K.shape[0]
    if ndof != 2 * used_labels.size:
        raise RuntimeError("内部DOF数と使用ラベル数が整合していません。")

    # -----------------------------
    # 3) 荷重ベクトル f を構築
    # -----------------------------
    f = np.zeros(ndof, dtype=float)
    for lab, (Fx, Fy) in node_label_load_mapping.items():
        lab_int = int(lab)
        if lab_int not in label_to_index:
            continue
        i = label_to_index[lab_int]
        f[2 * i] += float(Fx)
        f[2 * i + 1] += float(Fy)

    # -----------------------------
    # 4) Dirichlet 境界条件 → fixed_dofs
    # -----------------------------
    fixed_dofs: list[int] = []
    for lab, (ux_free, uy_free) in node_label_df_mapping.items():
        lab_int = int(lab)
        if lab_int not in label_to_index:
            continue
        i = label_to_index[lab_int]
        if not ux_free:
            fixed_dofs.append(2 * i)
        if not uy_free:
            fixed_dofs.append(2 * i + 1)

    fixed_dofs_arr = np.array(sorted(set(fixed_dofs)), dtype=int)

    # -----------------------------
    # 5) 拘束適用 → ソルブ
    # -----------------------------
    from .bc import apply_dirichlet  # 循環import回避のためにここで

    # from .bc import (
    # apply_dirichlet_penalty as apply_dirichlet,
    # )  # 循環import回避のためにここで
    from .solver import solve_displacement

    Kbc, fbc = apply_dirichlet(K, f, fixed_dofs_arr, values=0.0)
    u, info = solve_displacement(
        Kbc, fbc, size_threshold=size_threshold, use_pyamg=True
    )

    # -----------------------------
    # 6) 解ベクトル → ラベル辞書へマッピング
    # -----------------------------
    node_label_displacement_mapping: Dict[int, Tuple[float, float]] = {}
    uN2 = u.reshape(-1, 2)  # (Nnode_used, 2)

    for lab, idx in label_to_index.items():
        ux, uy = uN2[idx, 0], uN2[idx, 1]
        node_label_displacement_mapping[int(lab)] = (float(ux), float(uy))

    return node_label_displacement_mapping
