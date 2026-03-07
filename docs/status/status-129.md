# status-129: status-128 TODO消化 — CR梁摩擦調査・被膜gap修正・CI修正・S3機能統合

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-07
**テスト数**: 2271（fast: 1691 + 1 xfailed / slow: 362 - 9 xfailed / deprecated: 218）

## 概要

status-128のTODO 6項目を消化。

1. **CR梁の摩擦接触不収束の原因調査** → v_refフォールバック修正 + 根本原因特定
2. **被膜テストのgapパラメータ最適化** → gap=0で剛性比較テストxfail解除
3. **CI test-fast失敗の原因特定** → pyyaml依存追加で修正
4. **7本NCP曲げ揺動の大角度収束改善** → S3機能統合、10度のみ収束（15度以上は不収束）
5. **19本NCP径方向圧縮のCI環境収束** → CI test-slowの失敗はpyyaml欠損が原因
6. **旧ソルバーコード削除検討** → xfailテスト残存のため時期尚早

## 実施内容

### 1. CR梁摩擦接触不収束の原因調査

**根本原因**:
- CR梁のCR接線剛性行列は数値微分（中心差分, ε=1e-7）で計算
- 回転DOFのSO(3)多様体上でε=1e-7は不適切（Rodrigues/四元数変換の非局所性）
- ただしε値変更（1e-5等）は既存テストに副作用あり（Timo3D摩擦テスト破壊）
- μ=0.01のみ収束、μ≥0.05で全て不収束

**実施した修正**:
- `beam_timo3d.py:_build_local_axes()`: v_refが梁軸と平行な場合のフォールバック追加
  - 大変形でv_refが梁軸と平行になるとValueError → 自動で直交軸を再選択
- `solver_ncp.py`: J_t_t正則化εを1e-4→1e-3に強化（ratio > 1のケース）
- テスト更新: `test_parallel_reference_raises` → `test_parallel_reference_fallback`
- CR摩擦テストxfail理由を更新（数値微分K_TとNCPの結合不安定）

**結論**: CR梁の摩擦接触は解析的接線剛性行列の導出が必要。数値微分では本質的に限界がある。

### 2. 被膜テストのgapパラメータ最適化

**問題**: `test_coated_vs_bare_stiffness`で被膜付き変位 > 素線変位 * 1.05

**原因**: gap = `_COATING.thickness * 4 = 0.2e-3` が配置半径を増大させ、接触配置変化が被膜剛性寄与を打ち消していた

**修正**: gap = 0 に変更（同一配置で剛性寄与のみを比較）
- 比率: 1.19 → 0.997（被膜で変位が減少 = 剛性増加を正しく検出）
- xfail解除、5テスト全PASS

### 3. CI test-fast失敗の原因特定と修正

**CIログ取得方法の発見**:
- `gh api "repos/.../actions/jobs/{id}/logs"` → Azure Blob Storage (productionresultssa*.blob.core.windows.net) にリダイレクト → Forbidden
- **回避策**: `gh api "repos/.../actions/runs/{id}/logs"` でrun全体のZIPログを直接取得可能

**失敗原因**: `test_tuning_schema.py::TestTuningYAMLAPI::test_task_yaml_roundtrip`
- PyYAMLがdev依存に含まれていなかった
- `ModuleNotFoundError: No module named 'yaml'`

**修正**: `pyproject.toml` の `[project.optional-dependencies] dev` に `pyyaml>=6.0` を追加

### 4. 7本NCP曲げ揺動の大角度収束改善

**S3機能のベンチマーク関数統合**:
- `wire_bending_benchmark.py`: `_build_contact_manager()`と`run_bending_oscillation()`にS3パラメータ追加
  - `adjust_initial_penetration`, `contact_force_ramp`, `k_pen_continuation` 等

**収束角度の系統的調査**:

| 角度 | ステップ数 | S3機能 | 結果 |
|------|-----------|--------|------|
| 10度 | 5-10 | - | ✅ 収束（21-31秒） |
| 15度 | 15 | line_search + adjust | ❌ 不収束 |
| 20度 | 20 | 同上 | ❌ 不収束 |
| 45度 | 30-60 | 全S3有効 | ❌ 不収束 |
| 45度 | 60 | tol_force=1e-3 | ❌ 不収束 |

**結論**: CR梁+NCP接触で15度以上の曲げ揺動は根本的に不収束。解析的K_T導出またはアルゴリズム改善が必要。

### 5. 19本NCP径方向圧縮のCI環境収束

- CIログ分析で test-fast のみ失敗（pyyaml問題）、test-slow はtest-fast依存で未実行が多い
- pyyaml追加でtest-fast問題は解消 → test-slowも実行されるようになるはず
- 19本テスト自体はローカルで既にPASS（status-112で確認済み）

### 6. 旧ソルバーコード削除検討

**判定**: 時期尚早
- CR梁摩擦テストがxfail（NCP版で未解決）
- 7本曲げ揺動テスト45度以上がxfail
- 全NCP版テストがPASSするまで旧コードは保持

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/elements/beam_timo3d.py` | `_build_local_axes` v_refフォールバック |
| `xkep_cae/contact/solver_ncp.py` | J_t_t正則化ε: 1e-4→1e-3 |
| `xkep_cae/numerical_tests/wire_bending_benchmark.py` | S3機能パラメータ統合 |
| `pyproject.toml` | dev依存にpyyaml>=6.0追加 |
| `tests/test_beam_timo3d.py` | `test_parallel_reference_fallback`に変更 |
| `tests/contact/test_real_beam_contact_ncp.py` | CR摩擦xfail理由更新 |
| `tests/contact/test_coated_wire_integration_ncp.py` | gap=0 + xfail解除 |

## 設計上の懸念・ユーザーへの確認事項

1. **CR梁の解析的接線剛性**: 数値微分K_Tが摩擦NCP接触で致命的な精度不足。解析的導出は複雑だが避けて通れない
2. **7本NCP曲げ揺動の壁**: 10度以上で不収束はCR梁+NCP接触の組み合わせ問題。Timo3D線形梁でのNCP曲げ揺動も検討すべきか
3. **CIログ取得**: `gh api .../runs/{id}/logs` でZIP取得可能。ただし `productionresultssa*.blob.core.windows.net` の許可も推奨

## TODO

- [ ] CR梁の解析的接線剛性行列の導出（回転パラメータ化を含む）
- [ ] Timo3D線形梁でのNCP曲げ揺動45度テスト（CR梁限界の切り分け）
- [ ] CI test-slow の安定化確認（pyyaml修正後）
- [ ] 被膜モデルでの物理的gapの取り扱い改善（現状gap=0は剛性比較専用）
- [ ] 旧ソルバーコード削除（全NCP版xfail解消後）

## 運用フィードバック

### 効果的な点
- CIログのZIP取得（`gh api .../runs/{id}/logs`）でCI失敗の原因を即座に特定
- 角度の系統的パラメータスイープで収束限界を定量化
- Agent toolによるCR梁原因分析の並列調査が効率的

### 非効果的な点
- ε値変更の副作用調査に時間がかかった（変更→テスト→revertの繰り返し）
- 7本曲げ揺動テストの実行が各2-3分で、パラメータ探索に時間がかかる

---

[← README](../../README.md)
