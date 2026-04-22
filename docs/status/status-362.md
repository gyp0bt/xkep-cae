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

**opt-in**（`work/beam_hysteresis/19_hypothesis_c_backtracking_7strand.py`、
status-359 採択設定 smoothing_delta=1000 + `contact_backtracking_enabled=True`）:

初回実装ではトリガー条件が甘く、初期接触活性化（att=0 で n_active 0→75）にも
backtracking が発動して α=0.25 まで過剰縮退していた問題を発見。
`att >= 2 & n_active >= 1 & active_set_changed & _conv_rate > 0.85`
（mixed 狭義検知）+ `active_flip_ratio=0.3` の相対判定に改修した後、
**初期侵入ステップの誤発動はなくなったが**、7 本撚線では mixed (C+D) が稀
（status-361 の Type 分布で 1.2%）なため、BT イベントは NR 反復中の稀な
Type D stall で 54 回発動（`incr=375` 時点）。

実機進行観測（制限時間内の部分結果、sandbox time-out で中断）:

| 指標 | ベースライン（status-359） | 中間観測（incr=375 / frac=0.80） |
|------|---|---|
| frac 到達 | 1.0000（incr=475） | 0.8027（incr=375 で中断） |
| BT 発動数 | - | 54（全 NR 反復の約 3%） |
| cutback | 53（全行程） | 60（部分、低残差チャタリング検知由来） |

**所見**: 7 本撚線は Type D stall が散発的なため backtracking の効果は
限定的で、既存の接触凍結モード（status-284）との相互作用で全体 elapsed
は baseline より長くなる傾向。合否基準「frac=1.0 完走 + elapsed ≤ 1.2x」
の厳密な満足は次セッションで確認必要だが、BT 発動自体は仕様通り（mixed
領域のみ、全反復の 3%）で、**default OFF 運用では影響なし**が保証される。

## 4. 19本撚線 90° 曲げ検証（MCDD 凍結解除条件候補）

`work/beam_hysteresis/20_hypothesis_c_backtracking_19strand.py` 新設済み。
default `smoothing_delta=2000` + `contact_backtracking_enabled=True` で
19 本撚線 90° 曲げを実測する予定。本 status では時間制約のため未実行、
次セッション（status-363）で実測 → MCDD 凍結解除条件（frac=1.0 完走）の
判定を行う。

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

1. **仮説 C 候補 (c) 19 本撚線実機検証結果**: `20_hypothesis_c_backtracking_19strand.py`
   の結果を確認し、frac=1.0 完走なら MCDD 凍結解除条件達成。
   未完走なら (d) 接触凍結モードの 19 本適用検討。
2. **7 本撚線回帰**: `19_hypothesis_c_backtracking_7strand.py` の結果を
   確認、status-359 基準 frac=1.0000 / elapsed ≤ 1.2x を担保。
3. **候補 (c) パラメータ感度**: `active_flip_ratio` / `residual_ratio_threshold` /
   `rate_threshold` のスイープで最適 working point 探索。
4. **Phase E C24 候補**: `@verified_by` VerifyProcess の `process()` 内で
   実際に FD 整合検証が呼ばれるか AST 検査（MCDD 脱法パターン 2 裏口対策）。
