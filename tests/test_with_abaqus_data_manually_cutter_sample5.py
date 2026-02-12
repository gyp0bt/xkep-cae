from __future__ import annotations
import pytest
import numpy as np

pymesh = pytest.importorskip("pymesh", reason="pymeshが必要なテスト")


@pytest.mark.external
def test_cutter_sample5():
    """TRI6実メッシュ（pymesh依存）ソルブ"""
    from pymesh import mesher
    from pycae.api import solve_plane_strain_from_label_maps

    mesh_filepath = "go_cutter_sample5.inp"
    mesh = mesher(mesh_filepath, verbose=False)
    elem_arr = mesh.get_element_array(allow_polymorphism=True, invalid_node=0)

    node_coord_array = mesh.get_node_coord_array()
    nodes = np.array(
        [[i["label"], i["x"], i["y"], i["z"]] for i in node_coord_array], dtype=float
    )

    gfix = mesh.get_node_labels_with_nset("gfix")
    gmove = mesh.get_node_labels_with_nset("gmove")

    node_label_df_mapping = {i: (False, False) for i in gfix}
    node_label_load_mapping = {i: (1.0e-3, 0.0) for i in gmove}

    u_map = solve_plane_strain_from_label_maps(
        elem_quads=None,
        elem_tris=None,
        elem_tri6=elem_arr,
        node_coord_array=nodes,
        node_label_df_mapping=node_label_df_mapping,
        node_label_load_mapping=node_label_load_mapping,
        E=200e3, nu=0.3, thickness=1.0,
    )
    assert 1 in u_map
