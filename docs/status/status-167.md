# status-167: AL完全削除 — プロセスアーキテクチャ完全移行 Session 1

[← README](../../README.md) | [← status-index](status-index.md) | [← status-166](status-166.md)

**日付**: 2026-03-14
**テスト数**: ~2260 + 327 process テスト（AL削除により ~215テスト減）

## 概要

AL（Augmented Lagrangian）ソルバー（solver_hooks.py, line_search.py）を完全削除。
NCP Semi-smooth Newton ソルバー一本に統一。

**背景**: ALソルバーの7本撚線安定の主張は誤りであった。不適切なテスト（まともでないテストで収束をもってOKとしていた）に基づく評価であり、信頼性がなかった。

## 削除ファイル一覧

### ソースコード（3ファイル, ~2,600行）
| ファイル | 行数 | 内容 |
|---------|------|------|
| `xkep_cae/contact/solver_hooks.py` | 2,126 | AL接触ソルバー（Outer/Inner Newton） |
| `xkep_cae/contact/line_search.py` | ~300 | merit function + backtracking（AL専用） |
| `tests/contact/test_line_search.py` | ~160 | line_search.pyのテスト |

### テストファイル（10ファイル, ~215テスト, ~8,500行）
| 削除ファイル | NCP版 |
|------------|-------|
| `tests/contact/test_solver_hooks.py` | AL固有テスト（NCP版不要） |
| `tests/contact/test_real_beam_contact.py` | `test_real_beam_contact_ncp.py` |
| `tests/contact/test_twisted_wire_contact.py` | `test_twisted_wire_contact_ncp.py` |
| `tests/contact/test_friction_validation.py` | `test_friction_validation_ncp.py` |
| `tests/contact/test_beam_contact_penetration.py` | `test_beam_contact_penetration_ncp.py` |
| `tests/contact/test_coated_wire_integration.py` | `test_coated_wire_integration_ncp.py` |
| `tests/contact/test_large_scale_contact.py` | `test_large_scale_contact_ncp.py` |
| `tests/contact/test_hysteresis.py` | `test_hysteresis_ncp.py` |
| `tests/test_s3_benchmark_timing.py` | AL比較ベンチマーク（廃止） |
| `tests/test_wire_bending_benchmark.py` | AL使用ベンチマーク（廃止） |

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/__init__.py` | solver_hooks/line_search import除去、`__all__`更新 |
| `xkep_cae/contact/pair.py` | `evaluate_contact_forces()`, `update_al_multipliers()` 除去 |
| `xkep_cae/contact/law_normal.py` | `update_al_multiplier()` 関数除去 |
| `tests/contact/test_law_normal.py` | `TestUpdateALMultiplier` クラス除去 |
| `tests/contact/test_solver_ncp.py` | `TestNCPSolverComparison` クラス除去（AL比較） |
| `xkep_cae/numerical_tests/wire_bending_benchmark.py` | AL import除去、ローカルデータ型定義、use_ncp=True デフォルト化、docstring修正 |
| `xkep_cae/tuning/executor.py` | AL依存関数をNotImplementedErrorスタブに置換 |

## 残存AL参照（低リスク）

- `tests/generate_verification_plots.py`: 3箇所の遅延import（関数内import、pytestで実行されない）
- `xkep_cae/contact/pair.py`: `ContactState.lambda_n` フィールド（NCP/smooth_penaltyでも使用、AL専用ではない）
- `docs/status/` 各種: 歴史的記録として残す

## S3復帰トスアップ

### 既存テストの精査結果

`test_ncp_bending_oscillation.py` の実態:
- **Phase 1（曲げ）**: 完全に変位制御。`prescribed_dofs` で端部回転角を処方、`f_ext = np.zeros(ndof)`。docstringに「モーメント荷重」と書かれていたが誤り → 修正済み
- **Phase 2（揺動）**: z方向サイクル変位制御。xfail（status-143: 接触活性セット変動で不収束）
- `use_updated_lagrangian=True` + `adaptive_timestepping=True` で安定

### S3復帰ロードマップ

1. **Phase 2 xfail 解消**: 接触活性セット変動の安定化（chattering_window, contact_stabilization パラメータ調整）
2. **7本撚線曲げ揺動の再ベースライン**: NCP + Process API経由での安定収束を確認
3. **19本 → 37本スケールアップ**: 段階的に規模拡大
4. `tuning/executor.py` のNCP版作成

## テスト結果

- 全テスト通過（slow/external除く）
- process テスト: 327 パス
- 契約検証: 0件違反
- lint: 0件

## TODO（次セッション）

- [ ] テスト名正規化: `test_*_ncp.py` → `test_*.py`（AL版削除済みで名前衝突なし）
- [ ] NCP直接呼出テストのProcess API移行（Session 3）
- [ ] S3: Phase 2 xfail 解消に着手
- [ ] `tuning/executor.py` のNCP版実装
- [ ] `generate_verification_plots.py` のAL遅延import除去
