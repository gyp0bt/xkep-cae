# pycae

ニッチドメイン問題を解くための自作有限要素ソルバー基盤。
支配方程式・構成則・要素・更新則・積分スキーマをモジュール化し、
組み合わせて問題特化ソルバーを構成する。

## 現在の状態

線形弾性・平面ひずみソルバーが実装済み。
Q4/TRI3/TRI6/Q4_BBAR要素、Abaqusベンチマーク完了。
Phase 1（アーキテクチャ再構成）完了: Protocol導入、pyproject.toml、pytest統一、CI設定。

次のマイルストーン: Phase 2 空間梁要素の実装。

## ドキュメント

- [ロードマップ](docs/roadmap.md) — 全体開発計画（Phase 1〜8）
- [実装状況](docs/status/status-002.md) — 最新のステータス
- [status-001](docs/status/status-001.md) — プロジェクト棚卸しとロードマップ策定

## インストール

```bash
pip install -e ".[dev]"
```

## テスト実行

```bash
pytest tests/ -v -m "not external"
```

## 使用方法

### レガシーAPI（関数ベース）

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

### Protocol API（クラスベース）

```python
import numpy as np
from pycae.elements.quad4 import Quad4PlaneStrain
from pycae.elements.tri3 import Tri3PlaneStrain
from pycae.materials.elastic import PlaneStrainElastic
from pycae.assembly import assemble_global_stiffness
from pycae.bc import apply_dirichlet
from pycae.solver import solve_displacement

# 材料・要素のオブジェクト生成
mat = PlaneStrainElastic(E=200e3, nu=0.3)
q4 = Quad4PlaneStrain()
t3 = Tri3PlaneStrain()

# アセンブリ
K = assemble_global_stiffness(
    nodes_xy,
    [(q4, conn_q4), (t3, conn_t3)],
    mat,
    thickness=1.0,
)

# 境界条件適用 → ソルブ
Kbc, fbc = apply_dirichlet(K, f, fixed_dofs)
u, info = solve_displacement(Kbc, fbc)
```

## 依存ライブラリ

- numpy, scipy（必須）
- pyamg（大規模問題時のAMGソルバー、オプション）
- numba（TRI6高速化、オプション）

## 運用

本プロジェクトはCodexとClaude Codeの2交代制で運用。
引き継ぎ情報は `docs/status/` 配下のステータスファイルを参照のこと。
