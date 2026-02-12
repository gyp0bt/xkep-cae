"""実メッシュ（Q4/TRI3混在）ベンチマーク: Abaqusパーサー使用."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

MESH_FILE = Path("mesh_cutter_sample1.inp")


@pytest.mark.external
def test_cutter_sample1():
    """実メッシュでのQ4/TRI3混在ソルブ（Abaqusパーサー使用）"""
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

    # Q4/TRI3判別（invalid_node=0でパディングされている場合）
    if np.isin(0, elem_arr_np):
        elem_quads = elem_arr_np[~np.isin(elem_arr_np[:, -1], 0)][:, 1:]
        elem_tris = elem_arr_np[np.isin(elem_arr_np[:, -1], 0)][:, 1:-1]
    elif elem_arr_np.shape[1] == 5:
        elem_quads = elem_arr_np[:, 1:]
        elem_tris = None
    else:
        elem_quads = None
        elem_tris = elem_arr_np[:, 1:]

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
        elem_quads=elem_quads,
        elem_tris=elem_tris,
    )
    assert 1 in u_map
