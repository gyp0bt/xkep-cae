# status-208: BackendRegistry 完全廃止 + 被膜モデル物理検証テスト

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-18
**ブランチ**: `claude/check-status-todos-QYQ7y`
**テスト数**: 412 passed（fast テスト、81 deselected）

---

## 概要

status-207 の TODO 3件を完了。BackendRegistry パターンを完全廃止し O2 条例違反 2件を解消。
被膜モデル（Kelvin-Voigt）の物理検証テスト 18件を新規作成。coating strategy の import バグも修正。

## 1. BackendRegistry 完全廃止（O2 条例違反解消）

### 変更内容

`xkep_cae/numerical_tests/_backend.py` を全面書き換え:
- `BackendRegistry` クラス + `backend` シングルトンを**完全削除**
- 純粋関数として直接実装に置換

| 旧（BackendRegistry） | 新（直接実装） |
|---|---|
| `backend.apply_dirichlet(K, f, dofs)` | `_apply_dirichlet(K, f, dofs)` |
| `backend.solve(K, f)` | `_solve_linear(K, f)` |
| `backend.ke_func_factory(cfg, sec)` | `_ke_func_factory(cfg, sec)` |
| `backend.section_force_computer(...)` | `_section_force_computer(...)` |
| `backend.beam2d_lumped_mass_local(...)` | `_beam2d_lumped_mass_local(...)` |
| `backend.beam3d_lumped_mass_local(...)` | `beam3d_lumped_mass_local(...)` |
| `backend.beam2d_mass_global(...)` | `_beam2d_mass_global(...)` |
| `backend.beam3d_mass_global(...)` | `beam3d_mass_global(...)` |
| `backend.beam3d_length_and_direction(...)` | `beam3d_length_and_direction(...)` |
| `backend.cr_assembler_factory(...)` | `_cr_assembler_factory(...)` |
| `backend.cosserat_nl_assembler_factory(...)` | `_cosserat_nl_assembler_factory(...)` |
| `backend.transient_config_class` | `_TransientConfigInput` |
| `backend.transient_solver(...)` | `_transient_solver(...)` |

### 新規実装

- **2D 梁剛性行列**: `_eb2d_ke()` (EB 2D), `_timo2d_ke()` (Timoshenko 2D) — 教科書定式化
- **2D 質量行列**: `_beam2d_lumped_mass_local()`, `_beam2d_mass_global()` — 整合/集中
- **Newmark-β 過渡応答ソルバー**: `_transient_solver()` — HHT-α 対応、NR 反復
- **断面力計算**: `_section_force_computer()` — `_NodeForces2DOutput` / `_NodeForces3DOutput` データクラス
- **3D 関数**: `xkep_cae.elements._beam_cr` の既存関数を直接使用

### 更新ファイル

| ファイル | 変更 |
|---------|------|
| `_backend.py` | 全面書き換え（BackendRegistry → 純粋関数） |
| `runner.py` | `backend.*` → 直接関数呼び出し |
| `frequency.py` | 同上 |
| `dynamic_runner.py` | 同上 |
| `wire_bending_benchmark.py` | `backend._bending_oscillation_runner` → Process Architecture 遅延 import |

### 結果

```
O2 条例違反: 2件 → 0件
O3 条例違反: 0件 → 0件
契約違反: 0件
```

## 2. 被膜モデル物理検証テスト（18件新規）

**新規ファイル**: `xkep_cae/contact/coating/tests/test_physics.py`

### テスト一覧

| カテゴリ | テスト名 | 検証内容 |
|---------|---------|---------|
| 弾性力 | `test_no_compression_no_force` | δ=0 → f=0 |
| 弾性力 | `test_force_proportional_to_compression` | f ∝ δ（線形性） |
| 弾性力 | `test_force_direction_action_reaction` | A側+B側=0（作用反作用） |
| 弾性力 | `test_force_direction_repulsive` | 法線方向反発力 |
| 粘性 | `test_viscous_force_zero_when_no_damping` | c=0 → 粘性力ゼロ |
| 粘性 | `test_viscous_force_proportional_to_rate` | f_visc ∝ δ̇ |
| 粘性 | `test_stable_at_equilibrium` | δ=const → 弾性のみ |
| 剛性 | `test_effective_stiffness_composition` | k_eff = k + c/dt |
| 剛性 | `test_stiffness_symmetry` | K = K^T |
| 剛性 | `test_stiffness_positive_semi_definite` | 固有値 ≥ 0 |
| 剛性 | `test_no_compression_no_stiffness` | δ=0 → K=0 |
| 摩擦 | `test_friction_zero_without_compression` | δ=0 → 摩擦ゼロ |
| 摩擦 | `test_friction_zero_with_zero_mu` | μ=0 → 摩擦ゼロ |
| 摩擦 | `test_friction_uses_elastic_force_only` | p_n = k*δ（粘性項含まず） |
| 摩擦 | `test_coulomb_slip_limit` | \|q\| ≤ μ*p_n |
| 摩擦 | `test_friction_history_reset_on_separation` | 分離時リセット |
| 摩擦剛性 | `test_friction_stiffness_symmetry` | K_fric = K_fric^T |
| 摩擦剛性 | `test_friction_stiffness_zero_when_separated` | δ=0 → K_fric=0 |

## 3. coating strategy import バグ修正

`xkep_cae/contact/coating/strategy.py` で `return_mapping_core` / `tangent_2x2_core` の import が
旧名（prefix なし）を参照していた。正しい `_return_mapping_core` / `_tangent_2x2_core` に修正。

## 4. contracts C14 チェッカー動作確認

deprecated ディレクトリ不在でも `contracts/validate_process_contracts.py` は正常動作確認済み。

## テスト結果

```
412 passed, 81 deselected (slow/external), 4284 warnings
ruff check: All checks passed
ruff format: 111 files already formatted
契約違反: 0件
条例違反: 0件
```

## 次のタスク

- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動の Phase2 xfail 解消
- [ ] numerical_tests の slow テスト復旧（backendあり→直接実装で動くようになった）
