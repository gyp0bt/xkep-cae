# status-362: 仮説 C 候補 (c) — ContactBacktrackingLineSearchProcess 実装

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-22
**テスト数**: 前 status +6（`TestContactBacktrackingLineSearchProcessAPI` 6 件）

## 概要

status-361 で特定された「19本撚線 Type D stall の真因は mixed (C+D) 領域
（active flip + tangent 不整合の同時発火、Type 分布 16.6% 突出）」を
直接抑制する **仮説 C 候補 (c) line search 強化** を実装。
`ContactBacktrackingLineSearchProcess` を新設し、既存 `NCPLineSearch` の
||R_u|| 全体発散判定では捉えられない接触残差 / active flip の過剰増加を
検知して α を半減する backtracking を NR 反復内に追加した。

## 1. 実装

### 1.1 新規 Process: `ContactBacktrackingLineSearchProcess`

場所: `xkep_cae/contact/solver/_newton_steps.py`（+112 行）

入力: 現在 `u`, 既存 line search 後の `du`, 事前 `n_active_pre` /
`contact_res_pre`, 及び `compute_trial: Callable[[np.ndarray],
tuple[float, int]]`（u_try を受け、接触残差ノルム / active ペア数を返す）。

出力: 採択 α, `du_accepted = α·du`, 反復数, reason タグ
（`accepted`/`active_flip`/`residual_growth`/`min_alpha`/`max_steps`/
`trial_failed`）。

判定基準:

- **active flip**: `|n_active_try − n_active_pre| ≤
  max(flip_threshold_abs, flip_ratio × n_active_pre)` でなければ reject
- **residual growth**: `contact_res_try / contact_res_pre ≤
  residual_ratio_threshold` でなければ reject
- 両方 OK で採択 / いずれか NG なら α を `alpha_decay` 倍（既定 0.5）に半減

### 1.2 `_newton_dynamic.py` への組込

既存 `LineSearchUpdateProcess` の後、`u += du` の前に backtracking フックを
挿入。トリガー条件（mixed (C+D) の狭義検知）:

1. `att >= 2`: 収束率履歴が 2 以上あること（att=0/1 除外、初期活性化回避）
2. `n_active >= 1`: 接触活性化後のみ（初期侵入 step 除外）
3. `active_set_changed`: 今回の NR snapshot で active 集合変化あり
4. `_conv_rate > rate_threshold`（既定 0.85）: Type D 気味の収束率

発動時はペア状態を `list(manager.pairs)` でスナップショット、Process 内で
`compute_trial` が `ContactForceAssemblyProcess` を複数回呼ぶ間にペア状態が
ドリフトするため、採択後に `manager.pairs[:] = snapshot` で復元。
次 NR 反復冒頭の `force_proc` 呼び出しで manager 状態は新 u で再構築される。

### 1.3 `NewtonDynamicProcess.uses` への追加

```python
uses = [
    ContactForceAssemblyProcess,
    ConvergenceCheckProcess,
    TangentAssemblyProcess,
    LinearSolveProcess,
    LineSearchUpdateProcess,
    ContactBacktrackingLineSearchProcess,  # 新規
    TangentFDDiagnosticProcess,
    ContactKcComponentFDDiagnosticProcess,
]
```

### 1.4 config の縦方向プラミング

新設 Process は default OFF。opt-in するための config 経路を整備:

| 層 | 追加 field（全 9 項） |
|----|---|
| `NewtonDynamicInput` | `contact_backtracking_enabled / _max_steps / _active_flip_threshold / _active_flip_ratio / _residual_ratio / _alpha_decay / _min_alpha / _mixed_only / _rate_threshold` |
| `ContactFrictionInputData` | 同上 9 field |
| `ContactFrictionProcess` | `getattr(input_data, ...)` で `NewtonDynamicInput` に bridge |
| `StrandBendingOscillationConfig` | 同上 9 field、3 箇所の `ContactFrictionInputData(...)` 呼び出しで plumb-through |

## 2. Process 単体テスト

`xkep_cae/contact/solver/tests/test_process.py` に
`TestContactBacktrackingLineSearchProcessAPI` を追加（6 テスト、全 pass）:

1. `test_protocol_conformance` — `SolverProcess` サブクラス確認
2. `test_accept_alpha_one_when_no_flip` — α=1.0 即採択
3. `test_backtrack_on_excessive_active_flip` — flip > threshold で α 段階半減
4. `test_backtrack_on_contact_residual_growth` — 残差比 > threshold で α 半減
5. `test_min_alpha_guard` — α < min_alpha で打切り
6. `test_trial_failure_halves_alpha` — `compute_trial` 例外時の α 半減継続

## 3. 7本撚線 90° 曲げ回帰検証（default OFF / opt-in）

**default OFF**: 既存動作に変更なし、`pytest xkep_cae/contact/solver/
xkep_cae/numerical_tests/` 163 passed 6 skipped 1 xfailed（93s）で
回帰なし確認。

**opt-in 実測**（`work/beam_hysteresis/19_hypothesis_c_backtracking_7strand.py`、
status-359 採択設定 smoothing_delta=1000 + `contact_backtracking_enabled=True`）:

| 指標 | status-359 baseline | 候補 (c) | Δ% |
|------|---|---|---|
| frac | 1.0000 | **1.0000** ✓ | 0.00% |
| n_increments | 475 | 473 | -0.4% |
| n_cutbacks | 53 | 55 | +3.8% |
| elapsed [s] | 259.92 | **285.64** | **+9.9%** |
| BT 発動数 | - | 54（全 NR 反復の ~3%） | - |

**判定: frac=1.0 完走 OK + elapsed ≤ 1.20x OK**。backtracking 有効化により
+9.9% の overhead を許容できれば 7 本撚線は完全に回帰なしで動作する。
mixed 狭義検知の trigger 条件（`att≥2 & n_active≥1 & active_set_changed &
_conv_rate>0.85`）は status-361 の Type 分布（7 本 mixed=1.2%）と整合し、
BT は散発的にしか発動しない。

## 4. 19本撚線 90° 曲げ検証（MCDD 凍結解除条件候補）

**実測**（`work/beam_hysteresis/20_hypothesis_c_backtracking_19strand.py`、
default `smoothing_delta=2000` + `contact_backtracking_enabled=True`）:

| 指標 | status-339 baseline | 候補 (c) | Δ% |
|------|---|---|---|
| frac | 0.4839 | **0.5153** | **+6.5%** |
| n_increments | 271 | 318 | +17.3% |
| n_cutbacks | 39 | 38 | -2.6% |
| elapsed [s] | 534.68 | 729.36 | +36.4% |
| 最終停滞時 NR Type 分布 | D+E:67%, E:28% | **D+E:51%, E:43%** | - |
| BT 発動数 | - | 52（全 NR 反復の ~1%） | - |

**判定: frac=1.0 完走 NG（未完走）、ただし stall 点で +6.5% 改善は達成**。

### 所見

1. **部分的前進**: backtracking 有効化で 19 本撚線の stall 点が
   `frac=0.4839 → 0.5153`（+3.14pt / +6.5%）改善。これは status-358/360 で
   却下された候補 (a)/(a') よりは明確に良い成績。
2. **frac=1.0 未達**: 最終停滞時の NR Type 分布 `D+E:51%, E:43%` は
   status-360 の `D+E:67%, E:28%` より E 単独比率が高く、mixed (D+E) の
   割合は減っている → BT が mixed 領域を一部抑制していることを示唆。
3. **trigger 条件の保守性**: 19 本撚線での BT 発動数 52 は全 NR 反復の ~1%
   のみ。mixed_only + 厳格な trigger が 19 本 stall で本当に必要な BT
   発動を取りこぼしている可能性。
4. **MCDD 凍結解除条件「frac=1.0 完走」には未達**。次セッション (status-363)
   でパラメータ感度探索（`rate_threshold=0.7` に緩和 / `active_flip_ratio=0.15`
   に厳格化 / `mixed_only=False` 全反復発動）を実施、効果不十分なら候補 (d)
   接触凍結モード適用や Phase C-3' s-tracking 経路の再検討が必要。

## 5. トリガー条件設計の設計判断

初期実装では `att >= 0, n_active >= 0` で発動していたが、7 本撚線で
初期接触活性化（att=0 で n_active 0 → 75）にも backtracking が発動し、
正常な大量活性化を誤って reject する不具合を発見。以下で修正:

- `att >= 2` を必須化（収束率履歴を担保、att=0 の conv_rate=1.0 デフォルト
  判定を回避）
- `n_active >= 1` を必須化（初期侵入 step を除外）
- `active_flip_threshold` は `max(abs, ratio × n_active_pre)` の相対判定に
  変更（デフォルト abs=3, ratio=0.3、n_active=20 なら 6 まで許容、
  n_active=100 なら 30 まで許容）

これらは「mixed (C+D) 狭義検知」という仮説 C (c) の意図に整合する。

## ファイル変更

| ファイル | 変更 |
|---------|------|
| `xkep_cae/contact/solver/_newton_steps.py` | **+112 行**: `ContactBacktrackingLineSearchProcess` + Input/Output |
| `xkep_cae/contact/solver/_newton_dynamic.py` | **~70 行**: config 9 field + `uses` 追加 + 主ループへの組込 |
| `xkep_cae/contact/solver/__init__.py` | Process export 追加 |
| `xkep_cae/core/data.py` | `ContactFrictionInputData` に 9 field 追加 |
| `xkep_cae/contact/solver/process.py` | `NewtonDynamicInput` への bridge 9 field |
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `StrandBendingOscillationConfig` 9 field + 3 箇所 plumb-through |
| `xkep_cae/contact/solver/tests/test_process.py` | `TestContactBacktrackingLineSearchProcessAPI` 6 テスト |
| `work/beam_hysteresis/19_hypothesis_c_backtracking_7strand.py` | **新規**: 7 本撚線 opt-in 回帰検証 |
| `work/beam_hysteresis/20_hypothesis_c_backtracking_19strand.py` | **新規**: 19 本撚線 MCDD 凍結解除条件検証 |
| `docs/status/status-362.md` | **新規**: 本ファイル |
| `docs/status/status-index.md` | status-362 行追加 |
| `README.md` | status-362 要約追記 |

## Gate

- ruff check xkep_cae/ tests/ ✓
- ruff format --check xkep_cae/ tests/ ✓
- `python contracts/validate_process_contracts.py` 契約違反 **0 件** / 条例違反 **0 件**
- `pytest xkep_cae/contact/solver/` 109 passed 5 skipped（+6 新規テスト）
- `pytest xkep_cae/contact/solver/ xkep_cae/numerical_tests/` 163 passed
  6 skipped 1 xfailed（default OFF 回帰確認）

## 引継ぎ（status-363 へ）

1. **候補 (c) パラメータ感度探索**（19 本 frac=1.0 完走を目指す）:
   - `contact_backtracking_rate_threshold=0.7`（default 0.85 から緩和 →
     conv_rate 0.7〜0.85 の D.slow 領域も取り込み BT 発動数を増やす）
   - `contact_backtracking_active_flip_ratio=0.15`（default 0.3 から厳格化 →
     flip 許容量を減らし mixed (C+D) をより積極的に抑制）
   - `contact_backtracking_mixed_only=False`（全反復発動、コスト約 2x だが
     19 本 mixed 16.6% でも全体の 84% は無駄な BT コスト）
   - 上記 3 パラメータを `ParameterSweepBenchmarkProcess` で掃引、
     frac=1.0 完走判定 + elapsed 比較で最適 working point 決定。
2. **候補 (d) 接触凍結モードの 19 本適用**（(c) 感度探索で効果不十分な場合）:
   status-284 の 7 本 frac 0.40→0.70 達成手法を 19 本に適用、
   `chattering_freeze_enabled=True` のパラメータチューニング。
3. **候補 (e) Phase C-3' s-tracking 経路の再検討**（最終手段）:
   status-357 で active 集合振動支配領域には波及しないと判定された
   Phase C-3' の (ii) s-tracking 経路を 19 本実機で再評価、active 集合
   振動中の FD 整合性を直接改善するアプローチ。
4. **Phase E C24 候補**: `@verified_by` VerifyProcess の `process()` 内で
   実際に FD 整合検証が呼ばれるか AST 検査（MCDD 脱法パターン 2 裏口対策）。
