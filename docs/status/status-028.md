# status-028: Phase 3.4 TL定式化 + Phase 5.4 非線形動解析 + 動的三点曲げ

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-18

## 概要

status-027 の TODO を実行。Phase 3.4（Q4要素のTL定式化）、Phase 5.2（mass_matrix()メソッド追加）、Phase 5.4（非線形動解析ソルバー）、数値三点曲げ試験の非線形動解析対応を実装。テスト数 498 → 556（+58テスト）。Phase 4.3（von Mises 3D）は凍結。

## 実装内容

### Phase 5.2: 梁要素に mass_matrix() メソッド追加（+14テスト）

**更新ファイル**: `xkep_cae/elements/beam_eb2d.py`, `beam_timo2d.py`, `beam_timo3d.py`, `beam_cosserat.py`

- EB2D, Timo2D, Timo3D, Cosserat の各梁要素クラスに `mass_matrix()` メソッドを追加
- consistent / lumped の両タイプ対応
- 既存の `*_mass_global()` / `*_lumped_mass_local()` 関数をラップ

### Phase 3.4: Q4要素のTL定式化（+27テスト）

**新規ファイル**: `xkep_cae/elements/continuum_nl.py`, `tests/test_continuum_nl.py`

Total Lagrangian（TL）定式化による幾何学的非線形解析:

- **変形勾配** F = I + H, H = ∂u/∂X（参照配置上のガウス点で計算）
- **Green-Lagrangeひずみ** E = 0.5*(F^T F - I)
- **第二Piola-Kirchhoff応力** S = D:E（Saint-Venant Kirchhoff材料）
- **線形化B行列** B_L = B_0 + B_NL（変位依存の非線形項を含む）
- **内力ベクトル** f_int = ∫ B_L^T S dV₀
- **幾何剛性行列** K_geo = ∫ G^T Σ G dV₀（初期応力剛性）
- **材料剛性行列** K_mat = ∫ B_L^T D B_L dV₀
- **NRソルバー統合**: `make_nl_assembler_q4()`, `make_nl_assembler_q4_combined()`

テストカテゴリ:
| テストクラス | テスト数 | 検証内容 |
|-------------|---------|---------|
| TestBasicNonlinear | 5 | ゼロ変位, 線形剛性一致 |
| TestSmallDisplacementLimit | 3 | f_int ≈ K_lin @ u |
| TestTangentFiniteDifference | 6 | 有限差分による接線検証（3形状×3変位レベル） |
| TestSymmetry | 3 | 対称性 |
| TestPatchTest | 2 | 一様引張, 一様せん断のGL歪み検証 |
| TestGreenLagrangeStrain | 4 | 純粋引張, 回転不変性, 二軸, 単純せん断 |
| TestNewtonRaphsonIntegration | 3 | NR統合テスト |
| TestEnergyConsistency | 1 | f_int = dU/du（有限差分） |

### Phase 5.4: 非線形動解析ソルバー（+6テスト）

**更新ファイル**: `xkep_cae/dynamics.py`, `tests/test_dynamics.py`

- `NonlinearTransientConfig` データクラス
- `NonlinearTransientResult` データクラス
- `solve_nonlinear_transient()` 関数:
  - Newmark予測子 → NR反復で非線形平衡を解く
  - HHT-α数値減衰対応
  - 初期加速度: M·a₀ = f(0) - f_int(u₀) - C·v₀
  - 参照ノルム: max(||f_ext||, ||M·a||, ||f_int||, 1.0)

テスト:
| テスト名 | 検証内容 |
|---------|---------|
| test_linear_matches_linear_solver | 線形ばねでの線形ソルバーとの一致 |
| test_nonlinear_converges_each_step | Duffing振動子の全ステップ収束 |
| test_nonlinear_energy_conservation | 無減衰系のエネルギー保存（<1%） |
| test_nonlinear_static_convergence_with_damping | 減衰系の静的平衡収束 |
| test_fixed_dofs_respected | 固定DOFの境界条件遵守 |
| test_hardening_spring_frequency_shift | 硬化ばねの周波数シフト（FFT） |

### 数値三点曲げ試験の非線形動解析対応（+11テスト）

**新規ファイル**: `xkep_cae/numerical_tests/dynamic_runner.py`
**更新ファイル**: `xkep_cae/numerical_tests/core.py`, `xkep_cae/numerical_tests/__init__.py`, `tests/test_numerical_tests.py`

- `DynamicTestConfig` データクラス:
  - ステップ荷重 / ランプ荷重
  - Rayleigh減衰（α, β）
  - 整合 / 集中質量行列
  - 2D/3D梁タイプ対応
- `DynamicTestResult` データクラス:
  - 時刻歴（変位/速度/加速度）
  - 静的解析解との最終ステップ比較
- `run_dynamic_test()`, `run_dynamic_tests()` 公開API

テスト:
| テスト名 | 検証内容 |
|---------|---------|
| test_config_validation | コンフィグバリデーション |
| test_config_invalid_name | 無効試験名拒否 |
| test_config_invalid_rho | 負の密度拒否 |
| test_step_load_converges | ステップ荷重収束 |
| test_damped_step_converges_to_static | 高減衰で静的解に収束（<10%） |
| test_ramp_load | ランプ荷重の変位増加 |
| test_eb2d_beam_type | EB2D梁タイプ |
| test_3d_beam_type | 3D Timoshenko梁タイプ |
| test_lumped_mass | 集中質量行列 |
| test_dynamic_overshoot | 動的増幅（静的解超過） |
| test_run_via_public_api | 公開API統合 |

## テスト

**全体テスト結果**: 556 passed, 2 skipped

内訳:
- 既存テスト: 498 passed（変更なし）
- Phase 5.2 mass_matrix(): +14
- Phase 3.4 TL定式化: +27
- Phase 5.4 非線形動解析: +6
- 動的三点曲げ: +11

## コミット履歴

1. `feat: 梁要素に mass_matrix() メソッドを追加 (Phase 5.2)` — 14テスト追加
2. `feat: Q4要素の幾何学的非線形定式化 (Phase 3.4 TL/UL)` — 27テスト追加
3. `feat: 非線形動解析ソルバー (Phase 5.4 Newton-Raphson + Newmark-β)` — 6テスト追加
4. `feat: 数値三点曲げ試験の非線形動解析対応 (11テスト)` — 11テスト追加

## TODO（残タスク）

- [ ] Phase 4.3: von Mises 3D 弾塑性テスト実行（45テスト計画済み）— **凍結中**
- [ ] Phase 3.4: Updated Lagrangian（参照配置更新）の実装
- [ ] Phase C: 梁–梁接触モジュール実装
- [ ] 陽解法（Central Difference）
- [ ] モーダル減衰

## 確認事項・懸念

- Phase 3.4 は TL（Total Lagrangian）定式化のみ実装。UL（Updated Lagrangian）の参照配置更新は未実装。大変形問題での適用範囲に制限あり。
- `continuum_nl.py` は Q4 平面ひずみ要素のみ対応。TRI3/TRI6/Q4_EAS への拡張は未対応。
- 動的三点曲げの `dynamic_runner.py` は線形梁要素を使用（f_int = K·u）。Cosserat rod の非線形内力を使う場合は別途コールバック関数の実装が必要。
- 非線形動解析ソルバーの `solve_nonlinear_transient()` は全DOF密行列化するため、大規模問題（DOF > 数千）には不適。疎行列対応が将来必要。

---
