# status-366: 候補 (e) 接触減衰 escape hatch — Phase 2 (NR ソルバー配線 + ContactDampingEnergyMonitorProcess)

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-23
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7 passed

## 概要

status-365 Phase 1 引継ぎ 1. に対応し、候補 (e) 接触減衰 escape hatch を
NR ソルバーに配線完了。default `c_n=0` で既存動作不変。`c_n>0` で
`ContactNormalDampingProcess` が NR 反復ごとに `f_damp / K_damp` を
組み立て、`ContactDampingEnergyMonitorProcess`（新設 PostProcess）で
E_damp_cumulative / E_strain 比を監査する。

## 1. 実装

### 1.1 `ContactFrictionProcess` の Strategy 追加（§4.1）

- `damping_slot = StrategySlot(object, required=False,
  default_types=(ContactNormalDampingProcess,))` を `contact/solver/process.py`
  に追加。`uses` グラフから `ContactNormalDampingProcess` が到達可能化。
- `SolverStrategies.damping` フィールド（`object | None = None`）追加、
  `core/data.py::default_strategies()` が `ContactNormalDampingProcess()` を
  注入する。

### 1.2 NR ループへの f_damp / K_damp 加算（§4.2）

`contact/solver/_newton_dynamic.py`:

- `NewtonDynamicInput.contact_damping_coefficient: float = 0.0` 追加
- `DynamicStepOutput.damping_energy_rate: float = 0.0` 追加（収束時の
  瞬時消散率、呼び出し側で dt 乗算して累積）
- NR ループで `effective_residual` 適用後に `R_u += f_damp`、
  `effective_stiffness` 適用後に `K_T += K_damp` を加算。発動条件は
  `_damping_enabled & dt_sub > 1e-30 & manager.pairs` の AND。`c1 = γ/(β·dt)`
  は `_time_strategy.gamma / (_time_strategy.beta * dt_sub)` から算出。

### 1.3 `ContactDampingEnergyMonitorProcess` 新設（§4.3）

`contact/damping/monitor.py`（182 行、PostProcess）:

- 入力: `damping_energy_history`（tuple of `(load_frac, E_damp_cumulative)`）
  + `energy_history`（`EnergyHistory`）+ `budget_ratio`（0 で検査無効）+
  `log_every_n_steps`
- 出力: `max_ratio / final_ratio / final_damping_energy / final_strain_energy
  / n_violations / budget_violated / report`（人間向けテキスト）
- 非侵襲監査（読み取り専用）。validation スクリプトから呼ばれ budget 超過
  を検知。

### 1.4 設定 plumb-through（§4.4）

- `ContactFrictionInputData` に `contact_damping_coefficient` +
  `contact_damping_energy_budget_ratio` 追加
- `SolverResultData.damping_energy_history` 公開（tuple、default 空）
- `ContactFrictionProcess` が成功インクリメントごとに
  `damping_energy_rate * dt_sub` を累積して履歴に追加
- `StrandBendingOscillationProcess` 3 call site（MPC 曲げ / 自由端 / 揺動）
  で `cfg.contact_damping_coefficient` / `..._energy_budget_ratio` を
  solver 入力へ転送

## 2. ユニットテスト

`contact/damping/tests/test_monitor.py`（7 件、`@binds_to` 紐付け）:

| テスト | 検証 |
|---|---|
| `test_empty_damping_history` | 空入力で全ゼロ short-circuit |
| `test_within_budget_no_violation` | 比 0.01 < budget 0.05、violations=0 |
| `test_budget_violation_flags_correctly` | 2/2 件のうち 1 件超過カウント |
| `test_budget_zero_disables_check` | budget=0 で violations=0 固定 |
| `test_report_is_str` | report 文字列かつ "DampingMonitor" を含む |
| `test_strain_fallback_when_shorter` | entries が短い場合は末尾 strain で padding |
| `test_output_types` | 出力型（float / int / bool / str）厳密チェック |

## 3. Gate

- `pytest xkep_cae/contact/damping/` → **19 passed**（12 strategy + 7 monitor）
- `pytest xkep_cae/contact/` → **446 passed, 5 skipped**（+7 monitor、回帰なし）
- `pytest tests/` → **314 passed, 11 skipped**（回帰なし）
- `pytest xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py`
  → **18 passed**（3 call site 配線の smoke 回帰なし）
- `python contracts/validate_process_contracts.py` → **契約違反 0 件 /
  条例違反 0 件**（全 24 検査 OK）
- `ruff check / format --check` → **All checks passed**

## 4. 設計判断

### 4.1 damping の 2 回呼び出しは単純化と引き換えに微量の冗長計算

NR 反復内で `f_damp` 用に 1 回 + `K_damp` 用に 1 回、計 2 回
`ContactNormalDampingProcess.process()` を呼ぶ。1 回分は
`(g_shape ⊗ g_shape)` 構築が支配的で pair 数に線形。19 本撚線 Type D
stall 領域では damping コストは tangent 組み立てより 1〜2 桁低く、
実測で顕在化したら戻り値共有の refactor に切り替える。

### 4.2 Generalized-α の C 行列を書き換えない設計

`GeneralizedAlphaProcess.C` はコンストラクタ固定。接触ペアは NR 反復ごとに
active 集合が変動するため C に組み込むと時間積分モジュールが接触マネージャ
に依存する責務分離違反。本実装は「ペア依存処理層で f_damp/K_damp を
組み立て、NR 残差/接線に加算」の接触側責務に収めている。

### 4.3 `@verified_by` は付与せず

`TermExpansionContract("K_c_term_expansion")` の 5 項分解は K_c の解析的
接線拡張であり、減衰項は Generalized-α の C 行列経路の代替で別系統。
`K_damp` は rank-1 outer product で解析解が明白、status-365 の
`test_tangent_matches_fd_under_v_is_c1_u` で機械精度整合済み。

## 5. ファイル変更

| ファイル | 変更 |
|---------|------|
| `xkep_cae/contact/solver/process.py` | `damping_slot` / `ContactNormalDampingProcess` import / `contact_damping_coefficient` plumb / `damping_energy_history` 累積 + 2 return 追加 |
| `xkep_cae/contact/solver/_newton_dynamic.py` | `ContactNormalDampingInput/Process` import / `contact_damping_coefficient` field / `uses` 追加 / NR 内 R_u/K_T 加算 / `damping_energy_rate` 返却 |
| `xkep_cae/contact/damping/monitor.py` | **新規** 182 行（`ContactDampingEnergyMonitorProcess`） |
| `xkep_cae/contact/damping/__init__.py` | monitor Input/Output/Process 公開 |
| `xkep_cae/contact/damping/tests/test_monitor.py` | **新規** 7 テスト |
| `xkep_cae/contact/damping/docs/contact_damping.md` | Phase 2 完了へ更新 |
| `xkep_cae/core/data.py` | `SolverStrategies.damping` / `default_strategies` 注入 / `ContactFrictionInputData` 2 field / `SolverResultData.damping_energy_history` |
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | 3 call site で damping cfg を solver 入力へ転送 |
| `docs/status/status-366.md` | **新規** 本ファイル |
| `docs/status/status-index.md` | status-366 行追加 |
| `README.md` | 現在状況に Phase 2 追記 |
| `docs/roadmap.md` | 候補 (e) Phase 2 完了行追記 |

## 6. 引継ぎ（status-367 へ）

1. **validation (§4.4 の §4.4)**: `work/beam_hysteresis/23_contact_damping_7strand_sweep.py`
   新設で 7 本撚線 90° 曲げを c_n ∈ {0, 0.01, 0.02, 0.05, 0.10, 0.20} × k_pen·dt
   で実測、Papailiou 解析解との比較で `E_damp / W_load`・`E_damp / E_strain`
   の budget 許容線を確立。
2. **19 本撚線 Type D stall 解消検証**: `work/beam_hysteresis/24_contact_damping_19strand.py`
   新設で 7 本で特定した最小 c_n を 19 本に適用、**MCDD 凍結解除条件**
   `frac=1.0 完走 + E_damp/E_strain < budget` を判定。
3. **default 化判断**: (1)(2) の結果を踏まえ、`StrandBendingOscillationConfig`
   の default を `contact_damping_coefficient=0.0` のまま保つか小さな値に
   変更するか決定。
4. **不十分時の副次候補**: (d) 接触凍結モード 19 本適用（status-284 の 7 本
   frac 0.40→0.70 手法再評価）/ (f) Phase C-3' s-tracking の 19 本再評価。
5. **MCDD Phase E C25 候補**: `@verified_by` challenge-test fixture 義務化
   は damping validation 完了後に検討。
