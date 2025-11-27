import sys, os

sys.path.append(os.getcwd())

from pycae.api import solve_plane_strain_from_label_maps
import numpy as np


def _test_square1():
    # 単位正方形Q4
    nodes = np.array(
        [
            [10, 0.0, 0.0],
            [11, 1.0, 0.0],
            [12, 1.0, 1.0],
            [13, 0.0, 1.0],
        ],
        dtype=float,
    )
    elem_quads = np.array([[10, 11, 12, 13]], dtype=int)
    elem_tris = None

    node_label_df_mapping = {
        10: (False, False),
        11: (False, False),
    }

    # 右側に水平荷重
    node_label_load_mapping = {
        12: (0.0, 1.0),
        13: (0.0, 1.0),
    }

    u_map = solve_plane_strain_from_label_maps(
        elem_quads=elem_quads,
        elem_tris=elem_tris,
        node_coord_array=nodes,
        node_label_df_mapping=node_label_df_mapping,
        node_label_load_mapping=node_label_load_mapping,
        E=200e3,
        nu=0.3,
        thickness=1.0,
    )

    print("square 1")
    print(u_map[12])  # {10: (0,0), 11: (ux,uy), 12:(ux,uy), 13:(ux,0)} のような形


def _test_tri1():

    nodes = np.array(
        [
            [10, 0.0, 0.0],
            [11, 1.0, 0.0],
            [12, 0.5, 1.0],
        ],
        dtype=float,
    )
    elem_quads = None
    elem_tris = np.array([[10, 11, 12]], dtype=int)

    # 左下(10) 全拘束, 左上(13) y拘束だけなど、好きに
    node_label_df_mapping = {
        10: (False, False),
        11: (False, False),
    }

    node_label_load_mapping = {
        12: (0.0, 1.0),
    }

    u_map = solve_plane_strain_from_label_maps(
        elem_quads=elem_quads,
        elem_tris=elem_tris,
        node_coord_array=nodes,
        node_label_df_mapping=node_label_df_mapping,
        node_label_load_mapping=node_label_load_mapping,
        E=200e3,
        nu=0.3,
        thickness=1.0,
    )

    print("triangle 1")
    print(u_map[12])  # {10: (0,0), 11: (ux,uy), 12:(ux,uy), 13:(ux,0)} のような形


def _test_square1_tri1():

    nodes = np.array(
        [
            [1, 0.000000e00, 0.000000e00, 0.000000e00],
            [2, 1.000000e00, 0.000000e00, 0.000000e00],
            [3, 0.000000e00, 1.000000e00, 0.000000e00],
            [4, 1.000000e00, 1.000000e00, 0.000000e00],
            [5, 0.500000e00, 2.000000e00, 0.000000e00],
        ],
        dtype=float,
    )
    elem_quads = np.array([[1, 2, 4, 3]], dtype=int)
    elem_tris = np.array([[3, 4, 5]], dtype=int)

    # 左下(10) 全拘束, 左上(13) y拘束だけなど、好きに
    node_label_df_mapping = {
        1: (False, False),
        2: (False, False),
    }

    node_label_load_mapping = {
        5: (0.0, 1.0),
    }

    u_map = solve_plane_strain_from_label_maps(
        elem_quads=elem_quads,
        elem_tris=elem_tris,
        node_coord_array=nodes,
        node_label_df_mapping=node_label_df_mapping,
        node_label_load_mapping=node_label_load_mapping,
        E=200e3,
        nu=0.3,
        thickness=1.0,
    )

    print("square-triangle 1")
    print(u_map[4])
    print(u_map[5])


def _test_tri6():
    import numpy as np
    from scipy.sparse import lil_matrix, csr_matrix

    # ----------------------------------------
    # TRI6 節点（label, x, y）
    # 1つの二次三角形：1-2-3, 中点4,5,6
    # 座標は単純な正三角形に近い形
    # ----------------------------------------
    nodes = np.array(
        [
            [10, 0.0, 0.0],  # 1
            [11, 1.0, 0.0],  # 2
            [12, 0.5, 1.0],  # 3
            [13, 0.5, 0.0],  # 4 (1-2 midpoint)
            [14, 0.75, 0.5],  # 5 (2-3 midpoint)
            [15, 0.25, 0.5],  # 6 (3-1 midpoint)
        ],
        dtype=float,
    )

    # TRI6 要素：label を 6個並べる
    elem_tri6 = np.array([[10, 11, 12, 13, 14, 15]], dtype=int)

    # ----------------------------------------
    # 変位拘束（左下 10 を全部拘束）
    # ----------------------------------------
    node_label_df_mapping = {
        10: (False, False),  # ux=0, uy=0
        11: (False, False),  # ux=0, uy=0
        13: (False, False),  # ux=0, uy=0
        # 他は自由
    }

    # ----------------------------------------
    # 荷重（節点12 に Fx=1.0）
    # ----------------------------------------
    node_label_load_mapping = {
        12: (0.0, 1.0),
    }

    # ----------------------------------------
    # TRI6を直接アセンブル（簡易）
    # ※ solve_plane_strain_from_label_maps は TRI3/Q4専用なので使わない
    # ----------------------------------------
    from pycae.elements.tri6 import (
        tri6_ke_plane_strain,
    )  # :contentReference[oaicite:1]{index=1}
    from pycae.materials.elastic import (
        constitutive_plane_strain,
    )  # :contentReference[oaicite:2]{index=2}
    from pycae.bc import apply_dirichlet  # :contentReference[oaicite:3]{index=3}
    from pycae.solver import solve_displacement  # :contentReference[oaicite:4]{index=4}

    D = constitutive_plane_strain(200e3, 0.3)
    t = 1.0

    # --- 内部インデックス化 ---
    labels = nodes[:, 0].astype(int)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    # (6,2) 座標抽出
    xy = nodes[:, 1:3]

    # --- 要素剛性 Ke ---
    Ke = tri6_ke_plane_strain(xy, D, t)

    # --- 全体剛性 K（12x12） ---
    ndof = 12
    K = lil_matrix((ndof, ndof), dtype=float)

    # edofs = [2*i, 2*i+1 ...]
    edofs = []
    for lab in elem_tri6[0]:
        i = label_to_idx[int(lab)]
        edofs.extend([2 * i, 2 * i + 1])
    edofs = np.array(edofs, dtype=int)

    # 1要素なので散布は単純コピー
    for ii, I in enumerate(edofs):
        K[I, edofs] += Ke[ii, :]

    K = K.tocsr()

    # --- 荷重 f ---
    f = np.zeros(ndof, dtype=float)
    for lab, (Fx, Fy) in node_label_load_mapping.items():
        i = label_to_idx[lab]
        f[2 * i] = Fx
        f[2 * i + 1] = Fy

    # --- Dirichlet ---
    fixed_dofs = []
    for lab, (ux_free, uy_free) in node_label_df_mapping.items():
        i = label_to_idx[lab]
        if not ux_free:
            fixed_dofs.append(2 * i)
        if not uy_free:
            fixed_dofs.append(2 * i + 1)
    fixed_dofs = np.array(fixed_dofs, dtype=int)

    Kbc, fbc = apply_dirichlet(K, f, fixed_dofs, values=0.0)

    u, info = solve_displacement(Kbc, fbc)

    # --- 結果マップ ---
    uN2 = u.reshape(-1, 2)
    out = {
        lab: (uN2[label_to_idx[lab], 0], uN2[label_to_idx[lab], 1]) for lab in labels
    }

    print("TRI6 displacement at node 12:", out[12])
    return out[12]


if __name__ == "__main__":
    _test_square1()
    _test_tri1()
    _test_square1_tri1()
    _test_tri6()
