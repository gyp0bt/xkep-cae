# status-368: 候補 (d) 接触凍結モード 19 本再評価（nr_max=30 で +16.6% / frac=1.0 未達で候補クローズ）

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-24
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7 passed（回帰なし）

## 概要

status-367 引継ぎ 1. に対応。status-284 で 7 本撚線にて `frac 0.40→0.70`
を達成した `chattering_freeze_*` の既定パラメータは 7 本向けチューニング。
19 本 Type D stall 本体（status-339: frac=0.4839 / status-357 以降の現行
baseline: frac=0.3739）に対し 3 パラメータ × 6 ケースの感度掃引で MCDD
凍結解除条件達成可否を実測。

**結論**:

- **Case B（`chattering_freeze_nr_max=30`, default の 2x）が最良**: frac=0.5642
  （default 0.3739 比 **+50.9%**、status-339 baseline 0.4839 比 **+16.6%**）
- **他 5 ケース全て効果軽微または悪化**（max_cycles / tol_factor の単独増加
  は全く効かず、combined は逆相関で悪化）
- **MCDD 凍結解除条件（frac=1.0 完走）未達** → **候補 (d) クローズ**
- `chattering_freeze_nr_max=15` default は変更せず（7 本系の status-284
  パラメータを維持、19 本向けは opt-in escape hatch として運用）
- 次候補は **(f) Phase C-3' s-tracking 19 本再評価**

## 1. 掃引設計

### 1.1 Infrastructure plumb-through

`StrandBendingOscillationConfig` に 4 field を公開（既定値は
status-284 と同一）:

```python
chattering_freeze_enabled: bool = True
chattering_freeze_max_cycles: int = 5
chattering_freeze_nr_max: int = 15
chattering_freeze_tol_factor: float = 10.0
```

3 箇所の `ContactFrictionInputData` 組立（MPC 経路 / `free_end_mode` 経路 /
揺動フェーズ）で plumb-through。`ContactFrictionInputData` 側は status-284
で既に同名 field を保有（defaults 一致）、既存動作不変。

### 1.2 掃引軸（6 ケース）

| case | enabled | max_cycles | nr_max | tol_factor | 意図 |
|------|:---:|:---:|:---:|:---:|---|
| default | T | 5  | 15 | 10.0  | status-284 既定（7本用、比較基準） |
| A: more_cycles | T | **10** | 15 | 10.0  | 凍結サイクル数倍増 |
| B: longer_nr   | T | 5  | **30** | 10.0  | 凍結中 NR 反復上限倍増 |
| C: loose_tol   | T | 5  | 15 | **100.0** | 凍結 tol 10x 緩和 |
| D: combined    | T | **10** | **30** | **100.0** | A+B+C 全緩和 |
| E: disabled    | **F** | — | — | — | freeze 完全停止（反証ケース） |

問題設定: 19 本撚線 / 90° 曲げ（κ=0.015）/ `smoothing_delta` 自動（2000）/
`contact_backtracking` default OFF / `max_increments=1500`。

## 2. 実測結果

### 2.1 掃引サマリ（`work/beam_hysteresis/25_freeze_param_sweep_19strand.py`）

| case | frac | incr | cb | elapsed | Δfrac/default | Δfrac/status-339 |
|---|---:|---:|---:|---:|---:|---:|
| status-339 baseline | 0.4839 | 271 | 39 | 534.68s | — | — |
| default (5/15/10) | 0.3739 | 177 | 18 | 245.50s | — | **-22.7%** |
| A: more_cycles=10 | 0.3739 | 177 | 18 | 252.22s | +0.0% | -22.7% |
| **B: nr_max=30**  | **0.5642** | **356** | **35** | **863.22s** | **+50.9%** | **+16.6%** |
| C: tol_factor=100 | 0.3739 | 177 | 18 | 248.34s | +0.0% | -22.7% |
| D: combined | 0.4830 | 268 | 29 | 587.98s | +29.2% | -0.2% |
| E: disabled | 0.4661 | 264 | 39 | 1281.48s | +24.7% | -3.7% |

注: `status-339` の baseline=0.4839 は Phase C-3'（status-356）/ C-3'
反証（status-357）導入前の値。現行 code では default=0.3739 に退化済み
（status-357 既報）で、本掃引の比較基準は「default case」。

### 2.2 最終停滞時の NR Type 分布

| case | 内訳 | frac |
|---|---|---:|
| default / A / C | `D+E:69%, E:25%` (36/50 att) | 0.3739 |
| **B: nr_max=30** | **`D+E:56%, E:40%`** (50/50 att) | 0.5642 |
| D: combined | `D+E:52%, E:44%` (50/50 att) | 0.4830 |
| E: disabled | **`D+E:98%, E:1%`** (200/50 att) | 0.4661 |

**キー所見**:

- **E: disabled の `D+E:98%`**: freeze を無効化すると D+E（tangent 不整合
  + active flip）に 200 反復ハマる。status-339 の freeze 有効運用が
  「D+E ロック回避の支柱」であることが確定。
- **B: nr_max=30 の `D+E:56%`**: 凍結中 NR 反復上限を 15→30 に拡張する
  ことで、D+E 領域での収束チャンスが拡大し mixed 比率が 13 ポイント
  低下。status-362 の backtracking line search（`D+E:51%, E:43%`）と
  同じパターンの改善。
- **A/C の no-op**: `max_cycles=10` と `tol_factor=100` の単独増加は
  default と**完全に同じ停止点**（incr/cb/elapsed もバイト一致）。理由:
  default では (i) max_cycles=5 の上限に到達する前に (ii) nr_max=15 の
  反復上限で凍結失敗となり、早期 cutback へ分岐するため、max_cycles /
  tol_factor は発動機会を得ない。
- **D: combined の逆相関**: nr_max=30 単独（B=0.5642）を max_cycles=10 +
  tol_factor=100 と組み合わせると frac=0.4830 に劣化（-14.4%）。tol_factor
  緩和は凍結 "成功" と判定されやすくなり、本来はもっと NR 反復で戻すべき
  局面を早期に抜ける副作用と解釈可能。

## 3. 判定

### 3.1 MCDD 凍結解除条件未達

Case B の frac=0.5642 は status-339 baseline 比 **+16.6%** / default 比
**+50.9%** の有意な改善だが、**frac=1.0 完走には及ばない**。したがって
**候補 (d) クローズ**。これは status-363（候補 (c) パラメータ感度掃引で
BT 既定が最良と確定）と同じパターン。

### 3.2 default 変更は実施しない

19 本 Type D stall の主因は **K_c x/z カップリング不整合**（status-344
`mat_only rel_err 44%`）であり、freeze 拡張は症状緩和（D+E 領域での
NR 反復チャンス拡大）にすぎない。以下の理由で default を変更しない:

1. `chattering_freeze_nr_max=15` は status-284 で **7 本向けに最適化**
   されたパラメータ。30 に引き上げると 7 本系で不必要な反復コストを
   支払う可能性があり、回帰リスクを負う。
2. Case B は 19 本で elapsed 245s → 863s（3.5x 増）を代償としており、
   改善比率（frac +50.9%）と elapsed コスト（+251%）は不釣り合い。
3. MCDD 凍結解除条件未達の改善を default 化するのは脱法パターン 1
   （目標緩和）に抵触するリスクがある。

代替運用: `StrandBendingOscillationConfig.chattering_freeze_nr_max=30`
を 19 本以上の大規模撚線向けの **opt-in escape hatch** として公開（本
status で plumb-through 済み）。設計仕様書の更新は次 status 以降に
統合（本 status は実験記録に集中）。

### 3.3 Case B の本質: nr_max=30 が効く理由

Case E（disabled）で D+E:98% に 200 反復ハマる事実から、**freeze mode
は D+E ロック回避に必須**と確定。default の freeze_nr_max=15 は D+E
領域で (i) 凍結する / (ii) 凍結中 NR で再収束を試みる / (iii) 失敗で
cutback、のサイクルを早期に抜けており、15→30 拡張で (ii) の成功率が
上がることで全体 frac が前進する構造と解釈できる。

ただし nr_max 無限大にしても D+E 自体の数学的根本（K_c 不整合）は
解消されないため、**nr_max>30 でさらなる改善は期待薄**（MCDD 本命は
候補 (f) Phase C-3' s-tracking 19 本再評価）。

## 4. ファイル変更

| ファイル | 変更 |
|---------|------|
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `StrandBendingOscillationConfig` に `chattering_freeze_*` 4 field 追加 + 3 経路で `ContactFrictionInputData` へ plumb-through |
| `work/beam_hysteresis/25_freeze_param_sweep_19strand.py` | **新規**（6 ケース掃引スクリプト） |
| `docs/status/status-368.md` | **新規** 本ファイル |
| `docs/status/status-index.md` | status-368 行追加 |
| `README.md` | 現在状況に status-368 追記 |
| `docs/roadmap.md` | 候補 (d) 結果行追加 |

## 5. Gate

- `ruff check` / `ruff format --check`: **OK**
- `pytest xkep_cae/contact/`: **446 passed 5 skipped**（回帰なし）
- `pytest xkep_cae/contact/solver/tests/test_process.py -k freeze`: **4 passed**
- 契約違反 **0 件**（全 24 検査 OK） / 条例違反 **0 件**

## 6. 引継ぎ（status-369 へ）

1. **最優先: 候補 (f) Phase C-3' s-tracking 19 本再評価** — status-355/356
   の `K_hermite_adj` + `K_closest/K_st` active×adj 拡張は 7 本では
   FD 機械精度達成 / 19 本では frac=0.3739 退化（status-357）で保留中。
   本質的には active 集合変動領域での K_c x/z カップリング不整合
   （status-344 mat_only rel_err 44%）への対策であり、候補 (d) / (e) /
   (c) が全て症状緩和でしか効かなかった以上、**MCDD 凍結解除条件を
   クリアするにはここに戻るしかない**。次手は active 集合変動下での
   `K_hermite_adj` フル項拡張（status-354 実験を再評価）or 新 term
   `KcActiveFlipStiffness` 追加（Term Expansion Contract 6 項化）。
2. **副次: Case B の 19 本 opt-in 推奨化** — `chattering_freeze_nr_max=30`
   を 19 本以上向けのガイドラインとして README / `docs/roadmap.md` に
   明記。ただし本 status では「実装本体無変更」を優先し、次 status で
   設計仕様統合。
3. **MCDD Phase E C25 候補** は引き続き保留（症状緩和策が出揃った
   現状では契約拡張より MCDD 本命（K_c 不整合解消）に集中）。
4. **凍結中 TODO 棚卸し**（Phase C-3 以降再開禁止 → 候補 (d) クローズで
   status-363 §TODO と同じ）。

## 7. 運用所見

- **掃引スクリプトの反復性**: `22_bt_parameter_sweep_19strand.py` / 本
  `25_freeze_param_sweep_19strand.py` は同じ 6 ケース × 1 軸掃引の型。
  共通 infra を `ParameterSweepBenchmarkProcess`（status-315）に統合
  する価値あり（ただし現状 MCDD 本命から外れるため MCDD 完了後の
  改善候補）。
- **実測 1h 超**: 19 本 90° 曲げ 6 ケース掃引は実測 **~65 分**
  （245 + 252 + 863 + 248 + 588 + 1281 ≈ 3477s）。開発ループとしては
  重く、次候補 (f) Phase C-3' s-tracking では先に 7 本で FD gate テスト
  を固めてから 19 本実測に進む開発順序が効率的。
