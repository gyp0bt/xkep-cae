# xkep-cae

ニッチドメイン問題を解くための自作有限要素ソルバー基盤。
支配方程式・構成則・要素・更新則・積分スキーマをモジュール化し、
組み合わせて問題特化ソルバーを構成する。

> **名前の由来**: kepler（物理モジュール向けCAEアプリ）の派生系 → xkep

## 現在の状態

線形弾性・平面ひずみソルバーが実装済み。
Q4/TRI3/TRI6/Q4_BBAR/Q4_EAS要素、Abaqusベンチマーク完了。
Phase 1（アーキテクチャ再構成）完了。Protocol API に一本化済み（レガシーAPI削除）。
Phase 2.1/2.2（2D梁要素）完了: Euler-Bernoulli梁、Timoshenko梁。
Abaqus .inp パーサー自前実装済み（pymesh代替）。
**Q4要素にEAS-4（Simo-Rifai）を実装し、デフォルトに設定。**
せん断ロッキングと体積ロッキングを同時に抑制。
**Cowper (1966) のν依存せん断補正係数 `kappa="cowper"` をTimoshenko梁に実装（Abaqus準拠）。**
**ロードマップに Cosserat rod（Phase 2.5）および撚線モデル（Phase 4.6: 拡張ファイバー理論）を追加。**

次のマイルストーン: Phase 2.3 Timoshenko梁（3D空間）→ Phase 2.5 Cosserat rod。

## ドキュメント

- [ロードマップ](docs/roadmap.md) — 全体開発計画（Phase 1〜8）
- [Abaqus差異](docs/abaqus-differences.md) — xkep-cae と Abaqus の既知の差異
- [実装状況](docs/status/status-008.md) — 最新のステータス
- [status-007](docs/status/status-007.md) — Cowper κ(ν)実装・Abaqus比較テスト
- [status-006](docs/status/status-006.md) — EAS-4 Q4要素・B-barバグ修正
- [status-005](docs/status/status-005.md) — レガシー削除・Q4 D行列修正
- [status-004](docs/status/status-004.md) — Phase 2.1/2.2 梁要素 & Abaqusパーサー
- [status-003](docs/status/status-003.md) — リネーム & Phase 1 完了
- [status-002](docs/status/status-002.md) — Phase 1 アーキテクチャ再構成
- [status-001](docs/status/status-001.md) — プロジェクト棚卸しとロードマップ策定

## インストール

```bash
pip install -e ".[dev]"
```

## テスト実行

```bash
pytest tests/ -v -m "not external"
```

## Lint / Format

```bash
ruff check xkep_cae/ tests/
ruff format xkep_cae/ tests/
```

## 使用方法

### 高レベルAPI（ラベルベース）

```python
from xkep_cae.api import solve_plane_strain

u_map = solve_plane_strain(
    node_coord_array=nodes,
    node_label_df_mapping={1: (False, False), 2: (False, False)},
    node_label_load_mapping={5: (1.0, 0.0)},
    E=200e3,
    nu=0.3,
    thickness=1.0,
    elem_quads=elem_q4,
    elem_tris=elem_t3,
)
```

### Protocol API（低レベル）

```python
from xkep_cae.elements.quad4_eas_bbar import Quad4EASPlaneStrain  # EAS-4 (推奨)
from xkep_cae.elements.tri3 import Tri3PlaneStrain
from xkep_cae.materials.elastic import PlaneStrainElastic
from xkep_cae.assembly import assemble_global_stiffness
from xkep_cae.bc import apply_dirichlet
from xkep_cae.solver import solve_displacement

mat = PlaneStrainElastic(E=200e3, nu=0.3)

K = assemble_global_stiffness(
    nodes_xy,
    [(Quad4EASPlaneStrain(), conn_q4), (Tri3PlaneStrain(), conn_t3)],
    mat,
    thickness=1.0,
)

Kbc, fbc = apply_dirichlet(K, f, fixed_dofs)
u, info = solve_displacement(Kbc, fbc)
```

### 梁要素

```python
from xkep_cae.elements.beam_eb2d import EulerBernoulliBeam2D
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection2D

sec = BeamSection2D.rectangle(b=10.0, h=10.0)
beam = EulerBernoulliBeam2D(section=sec)
mat = BeamElastic1D(E=200e3)

K = assemble_global_stiffness(nodes_xy, [(beam, conn)], mat)
```

## 依存ライブラリ

- numpy, scipy（必須）
- pyamg（大規模問題時のAMGソルバー、オプション）
- numba（TRI6高速化、オプション）
- ruff（開発時lint/format）

## 運用

本プロジェクトはCodexとClaude Codeの2交代制で運用。
引き継ぎ情報は `docs/status/` 配下のステータスファイルを参照のこと。
