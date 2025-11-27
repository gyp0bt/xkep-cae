## メモ

```
# 最終的にはこの形で使用
from pycae.api import solve_plane_strain_from_label_maps
u_map = solve_plane_strain_from_label_maps(
    elem_quads=None,
    elem_tris=None,
    elem_tri6=elem_arr,
    node_coord_array=nodes,
    node_label_df_mapping=node_label_df_mapping,
    node_label_load_mapping=node_label_load_mapping,
    E=200e3,
    nu=0.3,
    thickness=1.0,
)
```

## サンプルコード

```
# 別途pymesh(pymes)をインストール必要
# http://elnhub.is.sei.co.jp:8929/analysis/osaka/cae2g/nishioka/pymes
import sys, os

sys.path.append(os.getcwd())

import pandas as pd
from pycae.api import solve_plane_strain_from_label_maps
import numpy as np
from pymesh import mesher


def _test_sample5(mesh_filepath: str):
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
        E=200e3,
        nu=0.3,
        thickness=1.0,
    )

    org_node_coord = mesh.get_node_coord()
    new_node_coord = org_node_coord.copy()
    for k, uv in u_map.items():
        new_node_coord[k][0] += uv[0] * 1000.0
        new_node_coord[k][1] += uv[1] * 1000.0
    mesh.update_node_coord(new_node_coord)
    mesh.dump(mesh_filepath[:-4] + ".result.inp", to_caeap=True)

    print("sample 1")
    print(u_map[1])
    return u_map[1]


if __name__ == "__main__":
    data = dict(u=[])
    mesh_filepath = "go_cutter_sample6.inp"
    uv = _test_sample5(mesh_filepath)
    data["u"].append(uv[0])
    data["u"].append(uv[1])
    mesh_filepath = "go_cutter_sample5.inp"
    uv = _test_sample5(mesh_filepath)
    data["u"].append(uv[0])
    data["u"].append(uv[1])
    df = pd.DataFrame(data)
    print(df.to_csv())

```