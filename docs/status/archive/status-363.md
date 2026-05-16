# status-363: 仮説 C 候補 (c) パラメータ感度掃引 — 4 ケース全却下、BT 既定が局所最適

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-23
**テスト数**: 前 status と同一（work script のみ追加、本体無変更）

## 概要

status-362 で `ContactBacktrackingLineSearchProcess`（以下 BT）既定設定が
7 本撚線 `frac=1.0000 完走 / +9.9% elapsed`（回帰なし）、19 本撚線
`frac=0.5153（baseline 0.4839 比 +6.5%）stall`（MCDD 凍結解除条件未達）
だったことを受け、**status-362 引継ぎ 1. の候補 (c) パラメータ感度掃引**を
実施。3 パラメータ（`rate_threshold` 緩和 / `active_flip_ratio` 厳格化 /
`mixed_only=False`）の単独・組合せを 4 ケースで実測し、**全ケースが
frac=1.0 未達**、かつ **BT 既定設定が実測最良点に最も近い局所最適**である
ことを確定した。

実装本体（`xkep_cae/`、`tests/`、`contracts/`）は**無変更**、新規 work
script（`22_bt_parameter_sweep_19strand.py`）1 本のみ追加。

## 1. 掃引軸と実装

新規スクリプト `work/beam_hysteresis/22_bt_parameter_sweep_19strand.py`
（+228 行）で以下 4 ケースを `StrandBendingOscillationProcess` 経由で
順次実行。`max_increments=1500` で runaway 抑制、`smoothing_delta` は
default（自動 2000）、`contact_backtracking_enabled=True` 固定。

| case | rate_threshold | active_flip_ratio | mixed_only | 意図 |
|------|---|---|---|---|
| A: relaxed_trigger | **0.70** | 0.30 | True  | D.slow 領域も BT 発動、発動数増加 |
| B: strict_flip     | 0.85 | **0.15** | True  | flip 許容量半減、mixed 強抑制 |
| C: always_on       | 0.85 | 0.30 | **False** | 全 NR 反復で BT チェック |
| D: combined (A+B+C)| **0.70** | **0.15** | **False** | 3 パラメータ全部 |

単一フィールド掃引しか扱えない `ParameterSweepBenchmarkProcess` では
3 フィールド同時掃引が困難なため、カスタム work script で対応。

## 2. 実測結果（19 本撚線 90° 曲げ、baseline 参照つき）

```
case       frac   incr   cb   elapsed[s]   Δfrac%base   Δfrac%bt
noBT     0.4839    271   39       534.68           --         --
BTdef    0.5153    318   38       729.36         6.5%         --
A        0.5153    318   38       744.92         6.5%      -0.0%
B        0.4701    256   28       532.20        -2.8%      -8.8%
C        0.4817    276   27       939.30        -0.5%      -6.5%
D        0.5156    313   35      1229.22         6.6%       0.1%
```

BT 発動統計（全 4 ケース合計、log grep ベース）:
`[BT:accepted]=388件` / `[BT:min_alpha]=147件` / `NR 診断出力=281件`。

### 2.1 ケース別所見

**Case A: `rate_threshold=0.70`（relaxed trigger）**
- frac=0.5153 / incr=318 / cb=38 で **BT default と完全一致**。
- 実測上 D.slow (conv_rate 0.70–0.85) 領域で追加発動するはずの BT が
  このケースでは **全く新規ヒットを生んでいない** ことを示唆。
- 理由推定: status-362 の NR Type 分布で `D+E` モードでは `_conv_rate`
  がすぐ 0.9 超まで上がる（Type D.stall は rate≈0.95–0.99）ため、
  閾値 0.70/0.85 の差は実効的に影響しない。
- elapsed +2.1% は実行時のばらつき範囲。

**Case B: `active_flip_ratio=0.15`（strict flip）**
- frac=0.4701（**BT default 比 -8.8% 悪化**）、elapsed は short。
- flip 許容量を `0.3*n_active_pre` から `0.15*n_active_pre` に半減すると、
  正当な active 増加まで rejection されて小 α に収束、物理的に意味ある
  接触活性化を抑制してしまい、むしろ **baseline(no BT) 0.4839 よりも
  悪い** 0.4701 で stall。厳格化は逆効果。

**Case C: `mixed_only=False`（always_on）**
- frac=0.4817（**BT default 比 -6.5% 悪化**）、elapsed +28.8%。
- 全 NR 反復で BT 判定すると非 mixed 領域（A/C/E 単独）でも α 半減が
  入り、収束進行を無用に減速。**BT の価値は mixed (C+D) 狭義検知に
  あることを再確認**。
- 7本撚線で mixed 1.2% だった status-361 の Type 分布所見と整合。

**Case D: 3 パラメータ全部（combined）**
- frac=0.5156（**BT default 比 +0.06%**、実測誤差範囲）、elapsed +68.5%。
- Case A 単独と誤差の範囲で一致、**Case B/C の悪化は Case D では
  Case A の保護効果で吸収**されている構造。
- 2 倍近い elapsed を払っても **frac 改善は事実上なし**。

### 2.2 最終停滞時 NR Type 分布（Case D、monitor ログより）

```
NR Type distribution (38 iterations): [D+E:26(68%), E:10(26%), A:1(3%), -:1(3%)]
Last 10 iterations: [D+E:10]
Last snapshot: R_c=2.06e-05, R_s=2.71e-06, rate=0.990, active=44, sliding=307
```

D+E（mixed）占有 68% は status-362 の BT default 最終分布
`D+E:51%, E:43%` よりむしろ **mixed 比率が高い**。Case D で BT を
過積極に firing しても stall 領域の Type 分布自体を改善できておらず、
active 集合振動が line search では根本抑制できないことを示す。

## 3. 判定と帰結

### 3.1 パラメータ感度掃引の結論

**全 4 ケースで MCDD 凍結解除条件「19 本 frac=1.0 完走」未達**。加えて:

1. **BT default 設定 (`rate_threshold=0.85 / active_flip_ratio=0.3 /
   mixed_only=True`) が実測最良点に最も近い局所最適**（Case D との
   差は +0.06% で実測誤差範囲）。
2. **緩和（A）は無変化、厳格化（B/C）は悪化**という明確なワンサイド
   感度を確認。
3. **候補 (c) line search 強化は status-362 で既に効果ほぼ全量を抽出済み**。
   パラメータチューニングではこれ以上 frac が伸びないことが確定。

### 3.2 BT default 変更の判定

`StrandBendingOscillationConfig.contact_backtracking_*` の default 値を
**変更しない**。現設定が本掃引で局所最適として確認されたため。
`22_bt_parameter_sweep_19strand.py` は**失敗実験の記録**として残置
（status-358 `15_hypothesis_c_7strand.py` / status-360
`16_hypothesis_c_aprime_19strand.py` と対称の扱い）。

### 3.3 候補 (c) クローズ宣言

status-362〜363 を通じて:

- **候補 (c) = `ContactBacktrackingLineSearchProcess` 実装 + 感度掃引**
  → **7本 frac=1.0 回帰なし達成 / 19本 frac +6.5% 改善止まり**
- **以降の候補 (c) 再試行は実施しない**（line search アプローチとして
  7/19 本の期待効果を既に抽出済み）。
- 19 本 stall の真因（mixed C+D での K_c x/z カップリング不整合 +
  active 振動同時発火）は line search では対処困難と確定。

## 4. 次セッション (status-364) 候補

status-362 で提示された (d)/(e)/(f) のうち、実装コストと期待効果
から **候補 (e) 接触減衰 escape hatch** を最有力に再設定。理由:

1. 19 本 stall の根本要因（active 振動支配）に対して BT は「静的 α
   半減」での対処だが、**減衰項 `c_n * v_n` は動的に振動そのものを
   抑える**ため本質的に強い。
2. 非物理的であるが、`E_damp / E_strain < budget_ratio` の **energy
   budget 制約** で定量監査可能（ユーザー提案、status-362 §引継ぎ 3.）。
3. 7 本撚線 / Papailiou 解析解を validation case として budget_ratio
   を empirical に決定可能（1/2/5/10/20% 掃引）。

副次候補:

- **候補 (d) 接触凍結モードの 19 本適用** — status-284 の 7 本
  frac 0.40→0.70 手法。パラメータチューニングのみで実装コスト低。
- **候補 (f) Phase C-3' s-tracking 経路の再検討** — status-357 で
  active 振動支配領域には波及しないと判定済みだが、19 本実機規模
  で再評価（最終手段）。

## 5. Phase E 進捗（本 status は維持）

status-360 の C21/C22/C23（`TermExpansionContract.term_names` 重複 +
`contracts` ClassVar 同名契約重複 + `@verified_by` 検証 Process 継承
必須）は本 status でも無変更で pass。全 23 契約検査 OK。

## ファイル変更

| ファイル | 変更 |
|---------|------|
| `work/beam_hysteresis/22_bt_parameter_sweep_19strand.py` | **新規 228 行**: 4 ケース掃引スクリプト（失敗実験記録として残置） |
| `docs/status/status-363.md` | **新規**: 本ファイル |
| `docs/status/status-index.md` | status-363 行追加 |
| `README.md` | status-363 要約追記 |
| `docs/roadmap.md` | 掃引結論追記 |

実装本体（`xkep_cae/`、`tests/`、`contracts/`）は**無変更**。

## Gate

- ruff check xkep_cae/ tests/ ✓
- ruff format --check xkep_cae/ tests/ ✓
- `python contracts/validate_process_contracts.py` 契約違反 **0 件** /
  条例違反 **0 件**（全 23 契約検査 OK）
- 実測ログ: `/tmp/bt_sweep_19strand_1776912267.log`
  （tee 保存、4 ケース計 ~55 min、47k 行）

## 引継ぎ（status-364 へ）

1. **候補 (e) 接触減衰 escape hatch 実装**（最優先）:
   - 新規 `ContactNormalDampingProcess`（仮称）を `xkep_cae/contact/solver/`
     に追加、`-c_n * v_n * n̂` の減衰力を接触ペア単位で組み上げ。
     接線剛性への寄与は `c_n / dt * I_nn`（Generalized-α の `γ/β dt`
     項に同期）。
   - `StrandBendingOscillationConfig` に `contact_damping_coefficient`
     + `contact_damping_budget_ratio` を追加、default OFF。
   - `ContactDampingEnergyMonitorProcess`（仮称）で `E_damp =
     Σ c_n v_n² dt` を積算、`E_strain` との比を 10 step 毎に出力。
   - validation: 7 本撚線 Papailiou 解析解 vs 減衰 1/2/5/10/20% で
     energy budget 許容線を確立、次に 19 本撚線で budget 内最大
     減衰を探索。
2. **候補 (d) 接触凍結モードの 19 本適用**（(e) で効果不十分時）:
   status-284 の 7 本 frac 0.40→0.70 手法を 19 本に適用、
   `chattering_freeze_enabled=True` のパラメータチューニング。
3. **候補 (f) Phase C-3' s-tracking 経路の 19 本再評価**（最終手段）:
   status-357 判定を 19 本実機規模で再検証。
4. **Phase E C24 候補**: `@verified_by` VerifyProcess の `process()`
   内で実際に FD 整合検証が呼ばれるか AST 検査（MCDD 脱法 pattern 2
   裏口対策）。
5. **`Strand3DContourProcess` の既存 BeamRender/StressContour3D との
   統合**（status-362 から継承、tube rendering / color mapping の
   共通ユーティリティ化）。
