from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix

from pycae.api import solve_plane_strain_from_label_maps
from pycae.elements.tri6 import tri6_ke_plane_strain
from pycae.materials.elastic import constitutive_plane_strain
from pycae.bc import apply_dirichlet
from pycae.solver import solve_displacement


def test_square1_tensile():
    """Q4単位正方形 引張試験"""
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

    node_label_df_mapping = {10: (False, False), 11: (False, False)}
    node_label_load_mapping = {12: (0.0, 1.0), 13: (0.0, 1.0)}

    u_map = solve_plane_strain_from_label_maps(
        elem_quads=elem_quads,
        elem_tris=None,
        node_coord_array=nodes,
        node_label_df_mapping=node_label_df_mapping,
        node_label_load_mapping=node_label_load_mapping,
        E=200e3, nu=0.3, thickness=1.0,
    )
    assert 12 in u_map
    ux, uy = u_map[12]
    assert isinstance(ux, float)
    assert isinstance(uy, float)


def test_tri1_tensile():
    """TRI3単体 引張試験"""
    nodes = np.array(
        [
            [10, 0.0, 0.0],
            [11, 1.0, 0.0],
            [12, 0.5, 1.0],
        ],
        dtype=float,
    )
    elem_tris = np.array([[10, 11, 12]], dtype=int)

    node_label_df_mapping = {10: (False, False), 11: (False, False)}
    node_label_load_mapping = {12: (0.0, 1.0)}

    u_map = solve_plane_strain_from_label_maps(
        elem_quads=None,
        elem_tris=elem_tris,
        node_coord_array=nodes,
        node_label_df_mapping=node_label_df_mapping,
        node_label_load_mapping=node_label_load_mapping,
        E=200e3, nu=0.3, thickness=1.0,
    )
    assert 12 in u_map


def test_square_tri_mixed_tensile():
    """Q4+TRI3混在メッシュ 引張試験"""
    nodes = np.array(
        [
            [1, 0.0, 0.0, 0.0],
            [2, 1.0, 0.0, 0.0],
            [3, 0.0, 1.0, 0.0],
            [4, 1.0, 1.0, 0.0],
            [5, 0.5, 2.0, 0.0],
        ],
        dtype=float,
    )
    elem_quads = np.array([[1, 2, 4, 3]], dtype=int)
    elem_tris = np.array([[3, 4, 5]], dtype=int)

    node_label_df_mapping = {1: (False, False), 2: (False, False)}
    node_label_load_mapping = {5: (0.0, 1.0)}

    u_map = solve_plane_strain_from_label_maps(
        elem_quads=elem_quads,
        elem_tris=elem_tris,
        node_coord_array=nodes,
        node_label_df_mapping=node_label_df_mapping,
        node_label_load_mapping=node_label_load_mapping,
        E=200e3, nu=0.3, thickness=1.0,
    )
    assert 4 in u_map
    assert 5 in u_map


def test_tri6_tensile():
    """TRI6単体 引張試験"""
    nodes = np.array(
        [
            [10, 0.0, 0.0],
            [11, 1.0, 0.0],
            [12, 0.5, 1.0],
            [13, 0.5, 0.0],
            [14, 0.75, 0.5],
            [15, 0.25, 0.5],
        ],
        dtype=float,
    )
    elem_tri6 = np.array([[10, 11, 12, 13, 14, 15]], dtype=int)

    node_label_df_mapping = {
        10: (False, False),
        11: (False, False),
        13: (False, False),
    }
    node_label_load_mapping = {12: (0.0, 1.0)}

    D = constitutive_plane_strain(200e3, 0.3)
    t = 1.0

    labels = nodes[:, 0].astype(int)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    xy = nodes[:, 1:3]

    Ke = tri6_ke_plane_strain(xy, D, t)

    ndof = 12
    K = lil_matrix((ndof, ndof), dtype=float)

    edofs = []
    for lab in elem_tri6[0]:
        i = label_to_idx[int(lab)]
        edofs.extend([2 * i, 2 * i + 1])
    edofs = np.array(edofs, dtype=int)

    for ii, I in enumerate(edofs):
        K[I, edofs] += Ke[ii, :]
    K = K.tocsr()

    f = np.zeros(ndof, dtype=float)
    for lab, (Fx, Fy) in node_label_load_mapping.items():
        i = label_to_idx[lab]
        f[2 * i] = Fx
        f[2 * i + 1] = Fy

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

    uN2 = u.reshape(-1, 2)
    out = {lab: (uN2[label_to_idx[lab], 0], uN2[label_to_idx[lab], 1]) for lab in labels}

    assert 12 in out
    _, uy = out[12]
    assert uy > 0.0, "TRI6の頂点12はy方向正の変位を持つべき"
