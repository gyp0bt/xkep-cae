from __future__ import annotations

import numpy as np
import pytest

pymesh = pytest.importorskip("pymesh", reason="pymeshが必要なテスト")


@pytest.mark.external
def test_cutter_sample1():
    """実メッシュ（pymesh依存）でのQ4/TRI3混在ソルブ"""
    from pymesh import mesher

    from xkep_cae.api import solve_plane_strain_from_label_maps

    mesh_filepath = "mesh_cutter_sample1.inp"
    mesh = mesher(mesh_filepath, verbose=False)
    elem_arr = mesh.get_element_array(allow_polymorphism=True, invalid_node=0)
    if np.isin(0, elem_arr):
        elem_quads = elem_arr[~np.isin(elem_arr[:, -1], 0)][:, 1:]
        elem_tris = elem_arr[np.isin(elem_arr[:, -1], 0)][:, 1:-1]
    elif elem_arr.shape[1] == 5:
        elem_quads = elem_arr[:, 1:]
        elem_tris = None
    else:
        elem_quads = None
        elem_tris = elem_arr[:, 1:]

    node_coord_array = mesh.get_node_coord_array()
    nodes = np.array([[i["label"], i["x"], i["y"], i["z"]] for i in node_coord_array], dtype=float)

    gfix = mesh.get_node_labels_with_nset("gfix")
    gmove = mesh.get_node_labels_with_nset("gmove")

    node_label_df_mapping = {i: (False, False) for i in gfix}
    node_label_load_mapping = {i: (1.0e-3, 0.0) for i in gmove}

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
    assert 1 in u_map
