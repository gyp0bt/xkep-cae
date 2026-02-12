# pycae

ニッチドメイン問題を解くための自作有限要素ソルバー基盤。
支配方程式・構成則・要素・更新則・積分スキーマをモジュール化し、
組み合わせて問題特化ソルバーを構成する。

## 現在の状態

線形弾性・平面ひずみソルバーが実装済み。
Q4/TRI3/TRI6/Q4_BBAR要素、Abaqusベンチマーク完了。

次のマイルストーン: 空間Timoshenko梁要素の実装に向けたアーキテクチャ再構成。

## ドキュメント

- [ロードマップ](docs/roadmap.md) — 全体開発計画（Phase 1〜8）
- [実装状況](docs/status/status-001.md) — 最新のステータス

## 使用方法

```python
from pycae.api import solve_plane_strain_from_label_maps

u_map = solve_plane_strain_from_label_maps(
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

```python
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

## 依存ライブラリ

- numpy, scipy（必須）
- pyamg（大規模問題時のAMGソルバー、オプション）
- pymesh（メッシュI/O、テスト用）
- numba（TRI6高速化、オプション）

## 運用

本プロジェクトはCodexとClaude Codeの2交代制で運用。
引き継ぎ情報は `docs/status/` 配下のステータスファイルを参照のこと。
