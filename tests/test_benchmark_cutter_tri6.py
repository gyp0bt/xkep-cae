"""TRI6実メッシュベンチマーク: Abaqusパーサー使用."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

MESH_FILE = Path("go_cutter_sample5.inp")


@pytest.mark.external
def test_cutter_sample5():
    """TRI6実メッシュソルブ（Abaqusパーサー使用）"""
    if not MESH_FILE.exists():
        pytest.skip(f"メッシュファイルが見つかりません: {MESH_FILE}")

    from xkep_cae.api import solve_plane_strain
    from xkep_cae.io.abaqus_inp import read_abaqus_inp

    mesh = read_abaqus_inp(MESH_FILE)

    node_coord_array = mesh.get_node_coord_array()
    nodes = np.array(
        [[n["label"], n["x"], n["y"], n["z"]] for n in node_coord_array],
        dtype=float,
    )

    elem_arr = mesh.get_element_array(allow_polymorphism=True, invalid_node=0)
    elem_arr_np = np.array(elem_arr, dtype=int)

    gfix = mesh.get_node_labels_with_nset("gfix")
    gmove = mesh.get_node_labels_with_nset("gmove")

    node_label_df_mapping = {i: (False, False) for i in gfix}
    node_label_load_mapping = {i: (1.0e-3, 0.0) for i in gmove}

    u_map = solve_plane_strain(
        node_coord_array=nodes,
        node_label_df_mapping=node_label_df_mapping,
        node_label_load_mapping=node_label_load_mapping,
        E=200e3,
        nu=0.3,
        thickness=1.0,
        elem_tri6=elem_arr_np[:, 1:],
    )
    assert 1 in u_map
