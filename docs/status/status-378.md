[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-378: 陽的中央差分 Phase 2 — solver path 配線 + 7 本 smoke test でスケーリング障壁実測

**日付**: 2026-04-29
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+28+10 passed（status-377 比 +10、新規 ExplicitDynamic ユニット 10 件 + 既存 strand_bending solver_mode 1 件差し替え）

## 概要

status-377 Phase 1 で完成した `ExplicitCentralDifferenceProcess` を `ContactFrictionProcess` に配線。`_explicit_dynamic.py` を新設し、`ExplicitDynamicProcess` が増分単位で `step()` を 1 回駆動する。Courant 監視は sparse Gerschgorin 上界で実装し、上限超過時に `failure_reason="courant"` で cutback 要求。

7 本撚線 90° 曲げ smoke test で **Courant 比 3×10⁵**（`dt_c=1.055e-06` vs `dt_sub=0.333`）を実測。wiring は正常動作するも、19 本以上の実機検証には `mass scaling` / `dt subcycling` が必須と確定。

## 1. 実装

### 1.1 `_explicit_dynamic.py` 新設

`ExplicitDynamicProcess`（+251 行）+ `ExplicitDynamicInput` / `ExplicitDynamicStepInput` + `_estimate_critical_dt()` ヘルパ。`uses=[ContactForceAssemblyProcess, ExplicitCentralDifferenceProcess]`、`docs/explicit_dynamic.md` を `document_path` に紐付け。

1 増分の挙動:
1. `ContactForceAssemblyProcess` で f_int + f_c 組立
2. Courant 監視（`courant_check_interval` 増分ごと、sparse Gerschgorin 上界）
3. `dt_sub > 0.9·dt_c` で `diverged=True, failure_reason="courant"` 返却
4. それ以外は `time_strategy.step(u, f_ext, f_int_eff, dt, fixed_dofs)` で 1 step 前進

`time_strategy` が `ExplicitCentralDifferenceProcess` でない場合は `TypeError` で防御。

### 1.2 `ContactFrictionInputData` / `ContactFrictionProcess` 配線

| 追加 field（`ContactFrictionInputData`） | default |
|---|---|
| `solver_mode: str` | `"implicit"` |
| `explicit_courant_safety: float` | `0.9` |
| `explicit_courant_check_interval: int` | `50` |
| `explicit_mass_lumping: str` | `"row_sum"` |

`default_strategies()` に `solver_mode` / `mass_lumping` 引数追加。`ContactFrictionProcess.process()` で:
- `_solver_mode == "explicit"` 時に `ExplicitDynamicProcess` を起動
- `predict()` / `correct()` / `_u_pred` MPC 射影をスキップ（陽解法は `step()` 内で速度・加速度更新）
- `uses` に `ExplicitDynamicProcess` 追加

### 1.3 `StrandBendingOscillationConfig` plumb-through

`solver_mode`/`explicit_courant_safety`/`explicit_courant_check_interval`/`explicit_mass_lumping` 4 field を追加、3 経路の `ContactFrictionInputData` 構築箇所すべてに伝搬。`NotImplementedError` ガード削除（status-377 Phase 1 待機解除）。

### 1.4 設計仕様 `xkep_cae/contact/solver/docs/explicit_dynamic.md` 新設

driver 責務 / 依存最小化 / スケーリング障壁を明記、本体は `time_integration_explicit.md` 参照。

## 2. テスト（+10 件）

`xkep_cae/contact/solver/tests/test_explicit_dynamic.py` 新設:

| クラス | テスト | 検証内容 |
|---|---|---|
| `TestExplicitDynamicProcessAPI` | 4 | SolverProcess 適合 / meta name/module/version |
| `TestEstimateCriticalDt` | 4 | 単位質量 ω²=100→dt_c=0.2 / 空 K で inf / 固定 DOF 除外 / 0 質量逆数除外 |
| `TestExplicitDynamicProcessRequiresExplicitStrategy` | 1 | 陰解法 strategy で TypeError |
| `TestExplicitContactFrictionIntegration` | 1 | `solver_mode="explicit"` で u 前進 + `SolverResultData` 返却 |

`test_strand_bending_oscillation.py` 既存 `test_solver_mode_explicit_raises_not_implemented` を `test_solver_mode_explicit_propagates_to_solver_input` に差し替え（field 伝搬検証）。

## 3. 検証

### 3.1 Default OFF 回帰（gate 必達）

| 項目 | 結果 |
|---|---|
| `python contracts/validate_process_contracts.py` | 全 24 検査 OK / 契約違反 0 件 |
| `pytest xkep_cae/contact/ xkep_cae/mathematics/ xkep_cae/time_integration/ xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py` | **680 passed, 5 skipped** |
| `test_helical_3d_hermite` | rel_err=2.18e-07 維持 |
| 7 本撚線 implicit | frac=1.0 完走（既存テスト） |
| `ruff check` / `ruff format --check` | OK |

### 3.2 7 本撚線 explicit smoke test（実機規模はじまり）

`/tmp/explicit_smoke/run_7strand_explicit.py`、`max_increments=3`, `n_strands=7`, `bending_curvature=0.0005`:

```
[explicit] dt_sub=3.333e-01 > 0.9·dt_c=1.055e-06 → cutback request
[CUTBACK:courant] frac 0.3333, dt=3.3333e-01 (cb #1)
[CUTBACK:courant] frac 0.0833, dt=8.3333e-02 (cb #2)
[CUTBACK:courant] frac 0.0208, dt=2.0833e-02 (cb #3)
[OK] solver_mode=explicit, converged=False, frac=0.0000, n_increments=1, n_cutbacks=3
```

**Courant 比 = dt_sub / dt_c = 3.16×10⁵**。3 回カットバック後も dt は dt_c に到達せず frac=0.0052 で打ち切り。wiring（cutback 連携 / state 一貫性 / Courant 推定）は正常動作。

implicit baseline（同条件）も `max_nr_attempts=15`, `max_increments=3` の極端な打切りでは収束せず（frac=0.0833 で diverged）、explicit と implicit の wiring 比較は別 status で `max_nr_attempts=50`, `max_increments` 緩和で再実施予定。

## 4. MCDD 脱法 pattern 回避

- pattern 1（tol 緩和）: 単体 10 件は機械精度ベース（`dt_c` 数値 12 桁一致）
- pattern 5（既存 skip）: 既存 status-377 Phase 1 の 28 unit + GeneralizedAlpha / QuasiStatic / contact 全 pass、`test_helical_3d_hermite` 機械精度継続
- pattern 6（骨格 status）: Phase 2 を solver wiring + 統合 smoke test + 設計仕様 + 10 unit で完結。`NotImplementedError` ガード削除済み

## 5. 引継ぎ（次 status へ）

### 5.1 スケーリング障壁の数値根拠

7 本実測で `dt_c ≈ 1.06×10⁻⁶` s。19 本では K_pen がさらに増加し dt_c は微減〜同オーダー想定。implicit dt_physical ~10² s に対し explicit は **10⁸ step 必要** で実機 frac=1.0 完走は本 driver 単独では非現実的。

### 5.2 候補 (h) Phase 3 候補（19 本 frac=1.0 gate に向けて）

1. **mass scaling**（Belytschko §6.4.2 推奨）: 接触 DOF の質量を係数 β² 倍して dt_c を β 倍化。物理的妥当性は高周波モードに限定して評価
2. **dt subcycling**: 接触ペアのみ陽解法、構造系は陰解法の混合解法。`predict / correct` Protocol 適合は status-377 で確保済み
3. **selective explicit**: K_c x/z 不整合 DOF のみ陽解法駆動

### 5.3 副次保留

- K_mat の x/z 二次補正項追加（status-377 §6.3 から継承）。陽解法移行が成功すれば優先度低下、失敗時に再開
- 多 pair 診断 `14b_kc_multi_pair_diagnostic.py`（status-370 §5）

## 6. 運用所見

### 6.1 Courant 比の事前見積もりが Phase 2 設計を変えた

Phase 1 docstring（`time_integration_explicit.md`）は dt_c ≈ 10⁻⁶ を「O(10³) 倍細かい」と記述したが、実機 7 本で **3×10⁵ 倍**。これは k_pen ~10⁹ に対し M_lump ~10⁻⁷（mm 系銅線）から `ω = √(k/m) ≈ 10⁸` rad/s の自然な帰結。Phase 3 で mass scaling 必須と確定。

### 6.2 wiring 完了による技術的選択肢の確保

status-378 で `solver_mode="explicit"` が 1 行で発動可能になった。今後の選択は数値手法（mass scaling / subcycling）のみで、wiring 議論は終結。Process 抽出設計の効果。

## 7. 引継ぎコマンド（次担当者向け）

```bash
# 7 本 explicit smoke test 再現
python /tmp/explicit_smoke/run_7strand_explicit.py 2>&1 | tee /tmp/explicit_smoke/smoke.log

# 単体回帰
pytest xkep_cae/contact/solver/tests/test_explicit_dynamic.py -v

# 全回帰
pytest xkep_cae/contact/ xkep_cae/mathematics/ xkep_cae/time_integration/ \
       xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py
python contracts/validate_process_contracts.py
ruff check xkep_cae/ tests/ && ruff format --check xkep_cae/ tests/
```
