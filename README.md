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
Phase 2.3/2.4（3D梁要素 & 断面拡張）完了: 3D Timoshenko梁（12DOF）、BeamSection（Iy/Iz/J）。
SCF（スレンダネス補償係数）を2D/3D Timoshenko梁に実装。
2D/3D断面力ポスト処理（`BeamForces2D/3D`, `beam2d/3d_section_forces()`）実装済み。
せん断応力ポスト処理（`beam2d/3d_max_shear_stress()`、ねじり+横せん断）実装済み。
**Phase 2.6（数値試験フレームワーク）完了: 3点曲げ・4点曲げ・引張・ねん回・周波数応答試験。**
**CSV出力対応、Abaqusライクテキスト入力対応。整合質量行列・Rayleigh減衰・FRF計算を実装。**
**Phase 2.5 前半完了: Cosserat rod 四元数回転実装。** 四元数演算モジュール（15関数）、
Cosserat rod 線形化要素（B行列＋1点ガウス求積）、[設計仕様書](docs/cosserat-design.md)を実装。314テストパス。
Abaqus .inp パーサー自前実装済み（pymesh代替、`*BEAM SECTION` / `*TRANSVERSE SHEAR STIFFNESS` 対応）。
Q4要素にEAS-4（Simo-Rifai）を実装し、デフォルトに設定。
Cowper (1966) のν依存せん断補正係数 `kappa="cowper"` をTimoshenko梁に実装（Abaqus準拠）。

次のマイルストーン: Phase 2.5 後半（内力ベクトル・幾何剛性） → Phase 3 幾何学的非線形。

## ドキュメント

- [ロードマップ](docs/roadmap.md) — 全体開発計画（Phase 1〜8）
- [Abaqus差異](docs/abaqus-differences.md) — xkep-cae と Abaqus の既知の差異
- [Cosserat rod 設計仕様書](docs/cosserat-design.md) — 四元数回転・Cosserat rod の設計
- [実装状況](docs/status/status-013.md) — 最新のステータス（Cosserat rod 四元数回転実装）
- [status-012](docs/status/status-012.md) — 数値試験フレームワーク（Phase 2.6）
- [status-011](docs/status/status-011.md) — 2D断面力ポスト処理 & せん断応力 & 数値試験ロードマップ
- [status-010](docs/status/status-010.md) — 3Dアセンブリテスト & 内力ポスト処理 & ワーピング検討
- [status-009](docs/status/status-009.md) — 3D Timoshenko梁 & 断面モデル拡張 & SCF
- [status-008](docs/status/status-008.md) — Cosserat rod & 撚線モデル ロードマップ拡張
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

### 3D梁要素

```python
from xkep_cae.elements.beam_timo3d import TimoshenkoBeam3D
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection

sec = BeamSection.rectangle(b=10.0, h=20.0)
beam = TimoshenkoBeam3D(section=sec, kappa_y="cowper", kappa_z="cowper")
mat = BeamElastic1D(E=200e3, nu=0.3)

K = assemble_global_stiffness(nodes_xyz, [(beam, conn)], mat)

# 解析後の断面力計算
from xkep_cae.elements.beam_timo3d import beam3d_section_forces
forces_1, forces_2 = beam.section_forces(coords_elem, u_elem, mat)
print(f"軸力: {forces_1.N:.3f}, せん断力: {forces_1.Vy:.3f}, モーメント: {forces_1.Mz:.3f}")
```

### Cosserat rod（四元数ベース幾何学的厳密梁）

```python
from xkep_cae.elements.beam_cosserat import CosseratRod
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection

sec = BeamSection.circle(d=10.0)
beam = CosseratRod(section=sec, kappa_y="cowper", kappa_z="cowper")
mat = BeamElastic1D(E=200e3, nu=0.3)

coords = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
Ke = beam.local_stiffness(coords, mat)

# 一般化歪みの計算
strains = beam.compute_strains(coords, u_elem)
print(f"軸伸び: {strains.gamma[0]:.6f}, ねじり: {strains.kappa[0]:.6f}")
```

### 数値試験フレームワーク

```python
from xkep_cae.numerical_tests import (
    NumericalTestConfig, run_test, run_all_tests,
    export_static_csv, parse_test_input,
)

# 関数引数で指定
cfg = NumericalTestConfig(
    name="bend3p", beam_type="timo2d", E=200e3, nu=0.3,
    length=100.0, n_elems=10, load_value=1000.0,
    section_shape="rectangle", section_params={"b": 10.0, "h": 20.0},
)
result = run_test(cfg)
print(f"FEM: {result.displacement_max:.6f}, 解析解: {result.displacement_analytical:.6f}")

# Abaqusライクテキスト入力
cfg = parse_test_input("""
*TEST, TYPE=BEND3P
*BEAM SECTION, SECTION=RECT
 10.0, 20.0
*ELASTIC
 200000.0, 0.3
*SPECIMEN
 100.0, 10
*LOAD
 1000.0
""", beam_type="timo2d")
result = run_test(cfg)

# CSV出力
export_static_csv(result, output_dir="./results")
```

### 周波数応答試験

```python
from xkep_cae.numerical_tests import (
    FrequencyResponseConfig, run_frequency_response,
    export_frequency_response_csv,
)

cfg = FrequencyResponseConfig(
    beam_type="timo2d", E=200e3, nu=0.3, rho=7.85e-9,
    length=100.0, n_elems=10,
    section_shape="rectangle", section_params={"b": 10.0, "h": 20.0},
    freq_min=10.0, freq_max=5000.0, n_freq=200,
    excitation_type="displacement", excitation_dof="uy",
    damping_alpha=0.0, damping_beta=1e-7,
)
result = run_frequency_response(cfg)
print(f"推定固有振動数: {result.natural_frequencies}")
export_frequency_response_csv(result, output_dir="./results")
```

## 依存ライブラリ

- numpy, scipy（必須）
- pyamg（大規模問題時のAMGソルバー、オプション）
- numba（TRI6高速化、オプション）
- ruff（開発時lint/format）

## 運用

本プロジェクトはCodexとClaude Codeの2交代制で運用。
引き継ぎ情報は `docs/status/` 配下のステータスファイルを参照のこと。
