[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-388: status-387 訂正・撤回 + 妥当性テスト透明性ルール策定（独立解析解 3 個以上同時一致を必須化）+ 単梁 explicit + UL は **L_arc 不伸長性 gate で全 n_inc で大破綻**

**日付**: 2026-05-02
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12+5+17+11+6 passed
（status-387 と同数、実装本体無変更）

## 概要

ユーザーから **STA2 厳罰** + **妥当性テストの透明性策定** + **3 個以上の解析解
同時一致** の要求を受け、status-387 を撤回し本 status-388 で訂正する。

## 1. status-387 の二重ミス

### 1.1 解析解の取り違え（90° vs 86°）

status-387 は単梁 90° 曲げの解析解 73.30mm（quarter circle u = √2 · 2L/π）を
gate 基準として使用していた。しかし実装の BC は `bending_curvature=0.015` ×
`L=100mm` = 1.5 rad ≈ **86°** で、正しい解析解は:

- `R = L/θ = 100/1.5 = 66.67 mm`
- `|u_x|_anal = |R sin θ - L| = 33.50 mm`
- `|u_z|_anal = R(1 - cos θ) = 61.96 mm`
- `|u|_anal = sqrt(u_x² + u_z²) = 70.44 mm`（3 個目の独立指標としてはカウント不可）

implicit baseline `|u|=70.45 mm` は **正しい解析解 70.44 mm にほぼ完全一致**
（err 0.01%）であり、status-387 が「err 3.90%」と報告したのは 90° 解析解との
誤った比較だった。

### 1.2 単一指標 (max\|u\|) のみで判定

`|u|=72.88 mm` の n_inc=8000 ケースが `|u|_anal_90=73.30mm` と「**err 0.58%**」で
一致したのを以て「精度 gate 達成」と判定した。これは **単一指標一致は偶然の交差を
許容する**（STA2 該当）の典型例。複数の独立解析解で交差検証していなかった。

## 2. 妥当性テスト透明性ルール（CLAUDE.md に追記）

ユーザー要求「**3 個以上の解析解同時一致**」を CLAUDE.md「STA2 防止ルール」
セクションに追記:

> **「max\|u\| 単一指標一致」は偶然の交差を許容するため STA2 該当**。物理的妥当性を
> 主張するには **独立な解析解 3 個以上の同時一致** が必須。
>
> **最低 3 指標**（互いに独立、kinematics と energetics-or-geometric の両方を含む）:
>
> 1. 位置成分 1（例: 先端 u_x）
> 2. 位置成分 2（例: 先端 u_z）
> 3. kinematics と独立な指標 — エネルギー量 (SE/W_ext/M_reaction) / 不伸長性
>    (L_arc) / 曲率分布 / 内部断面回転 のいずれか
>
> `|u|` ノルムは (1)(2) から導出されるため独立指標としてカウント不可。
> SE は実装の `0.5 u^T f_int` が MPC 拘束 DOF 消去で信頼できない場合があり、
> その場合は L_arc 等の geometric 指標で代替する。

## 3. 訂正版実機検証 — 3 指標 (|u_x|, |u_z|, L_arc) AND gate

### 3.1 解析解 3 指標（実 BC κ=0.015, L=100mm, θ=1.5 rad）

| 指標 | 値 | 物理的意味 |
|------|---:|------------|
| `\|u_x\|_anal` | 33.500 mm | 縮み方向（kinematic） |
| `\|u_z\|_anal` | 61.951 mm | 曲げ方向（kinematic） |
| `L_arc_anal` | 100.000 mm | 不伸長性（geometric、最強の独立指標） |
| SE (診断のみ) | 71.79 N·mm | gate 外（MPC 拘束消去で 0.5 u^T f_int 信頼性なし） |

### 3.2 多重集合 + 不伸長性 gate（10% AND）

実装座標系で u_x/u_z の役割が解析解と入れ替わる（処方回転向き依存）ため、
`{|u_x|, |u_z|}` を **多重集合**として sort 後 pair-wise 比較。

| label | frac | \|u_x\|/\|u_z\| 多重集合 [mm] | kin err | L_arc [mm] | L err | gate |
|-------|----:|----------------------------:|--------:|----------:|------:|:----:|
| analytical | — | 33.500, 61.951 | — | 100.000 | — | — |
| **implicit baseline (n=20)** | 1.000 | 33.482, 61.981 |  **0.1%** ✓ | **100.000** | **0.0%** ✓ | **PASS** ✅ |
| exp_n_inc=200 | 1.000 |  1.438,  6.414 | 95.7% ✗ | 105.628 |  5.6% ✓ | FAIL |
| exp_n_inc=500 | 1.000 |  4.448, 12.589 | 86.7% ✗ | 116.943 | 16.9% ✗ | FAIL |
| exp_n_inc=1000 | 1.000 |  8.580, 19.583 | 74.4% ✗ | 131.997 | 32.0% ✗ | FAIL |
| exp_n_inc=2000 | 1.000 | 14.901, 29.379 | 55.5% ✗ | 154.505 | 54.5% ✗ | FAIL |
| exp_n_inc=4000 | 1.000 | 24.218, 43.101 | 30.4% ✗ | 187.117 | 87.1% ✗ | FAIL |
| exp_n_inc=6000 | 1.000 | 31.509, 53.569 | 13.5% ✗ | 212.379 |112.4% ✗ | FAIL |
| **exp_n_inc=8000 (旧 sweet spot)** | 1.000 | 37.710, 62.366 | **12.6%** ✗ | **233.754** | **133.8%** ✗ | **FAIL** ❌ |
| exp_n_inc=10000 | 1.000 | 43.203, 70.101 | 29.0% ✗ | 252.624 |152.6% ✗ | FAIL |
| exp_n_inc=12000 | 1.000 | 48.186, 77.084 | 43.8% ✗ | 269.704 |169.7% ✗ | FAIL |
| exp_n_inc=16000 | 1.000 | 57.066, 89.465 | 70.3% ✗ | 300.065 |200.1% ✗ | FAIL |
| n_inc=8000 + α=5.0 + relax=200 | 1.000 | 12.714, 14.419 | 76.7% ✗ | 126.883 | 26.9% ✗ | FAIL |
| n_inc=12000 + α=5.0 + relax=200 | 1.000 | 14.986, 17.030 | 72.5% ✗ | 133.682 | 33.7% ✗ | FAIL |
| n_inc=16000 + α=5.0 + relax=500 | 1.000 | 16.982, 19.117 | 69.1% ✗ | 139.383 | 39.4% ✗ | FAIL |

### 3.3 結論 — explicit + UL は **全 n_inc で梁が伸びる非物理解**

- **implicit baseline は 3 指標すべて PASS**（err 0.1% / 0.0%）。これが妥当性の
  ground truth。
- **explicit + UL は 全 13 ケース FAIL**:
  - n_inc=200: L_arc は 5.6% で gate 通過するが kinematic 95.7% off で完全 FAIL
  - n_inc=8000: kinematic は 12.6% に近接するが **L_arc は 134% 過大**
    （**梁が 100mm → 233.75mm に伸びる、2.3x 非物理ストレッチ**）
  - n_inc=16000: L_arc 200% 過大（**3x スケール、300mm**）
- **damping + relax 併用** は L_arc を改善する代わり kinematic を破壊（n_inc=8000
  で L_arc 234→127、kinematic 12.6%→76.7%）— トレードオフ関係で 3 指標同時 PASS
  は不可能。
- **n_inc を増やすと L_arc は単調に過大化**（線形にほぼ比例: 100, 106, 117, 132,
  155, 187, 212, 234, 253, 270, 300）。これは UL 凍結による f_int(u_incr)≈0 で
  軸方向拘束力が消失し、BC 駆動の質量慣性で慣性的に節点が引き伸ばされる現象。

### 3.4 status-387 の「max\|u\| sweet spot」の真相

n_inc=8000 で `|u|≈73mm` が 90° 解析解 73.30 と一致したのは:

- 実梁長 100mm が 234mm に伸び（`L_arc=233.75`）、曲率が `1.5/100=0.015` から
  実効的に `1.5/233 ≈ 0.0064` に減少
- 半径 R_eff = 234/1.5 = 156mm の円弧で先端が回転
- 偶然 (u_x, u_z) が (37.7, 62.4) になり |u| ≈ 73mm 相当に一致
- **しかし u_x も u_z も実 BC 解析解 (33.5, 62.0) と片側 12.6% off**

つまり「単梁 sweet spot」は **梁が伸びる + 曲率が薄まる + 座標が偶然一致** という
完全に非物理的な座標一致で、3 指標 AND gate で確実に検出される。

## 4. MCDD 凍結解除条件

| 条件 | 状態（status-388 訂正後） |
|------|---|
| (1) Phase E 完了 | ✅ status-357 |
| (2) 19 本 frac=1.0 完走 | ❌ 未達続行 |
| (3) max\|u_trans\| < L_strand × 10 | ✅ implicit / 一部 explicit |
| (4) `KcNormalDirectionStiffness` FD rel_err < 1e-2 | ✅ status-356（2.18×10⁻⁷） |
| **(5) 解の精度 < 10%** | **❌ 未達続行（status-387 判定撤回、3 指標 AND gate で全 explicit ケース FAIL）** |
| **(5') 透明性ルール: 3 指標 AND gate (status-388)** | **❌ implicit 単梁のみ PASS** |

## 5. 実装変更まとめ

- `CLAUDE.md`:
  - 「STA2 防止ルール」セクションに「**妥当性テストの透明性ルール（status-388 追加・厳罰）**」を追加
    - 独立 3 指標必須化（kinematics 2 + energetics-or-geometric 1）
    - `|u|` ノルムは導出値で独立指標カウント不可
    - SE 信頼できない場合は L_arc 等で代替可
- `docs/status/status-387.md`: 冒頭に「**⚠️ 訂正通知（status-388 で撤回）**」追加
- `work/beam_hysteresis/40_explicit_n_inc_sweep.py`:
  - docstring を訂正（90° → 86°、3 指標 AND gate）
  - `_run_one()` で `tip_u_x` / `tip_u_z` / `L_arc` 抽出
  - `_analytical_circular_arc()` で実 BC の解析解 3 指標を計算
  - `_summarize()` で多重集合 + L_arc の AND gate 判定
- 実装本体（`xkep_cae/`、単体テスト、契約検査）は **無変更**

回帰: 全 24 契約検査 OK / contact + math + time_integration + strand_bending_osc =
**743 passed 5 skipped**（status-386/387 と同数）/ ruff check + format pass。

## 6. 引継ぎ — 次 status の候補

### 6.1 候補 (z2) Cosserat 梁プロトタイプ最優先（不変）

status-388 で確定: explicit + UL は **すべての n_inc で物理的に破綻**
（梁が 1.06x〜3.0x に伸びる）。status-387 の「sweet spot」は座標値の偶然交差で
真の達成ではない。**(z2) Cosserat 梁路線が唯一の本質解決路**。

UL を捨てた geometrically exact (Simo-Reissner) Cosserat 梁は:

- SO(3) 回転 DOF をネイティブに保持、reference 更新が不要
- 大回転 + 大変位での `f_int(u)` 評価が物理的に正しい
- 軸方向拘束（L_arc 保存）も exact に維持される
- explicit + 適切な mass scaling で波伝播・変形両方を正しく追従

実装規模 ~1000 行、Phase 設計から着手。

### 6.2 副次 — 既存 validation スクリプトの 3 指標化

status-388 で策定した透明性ルールに合わせ、既存 `30_implicit_vs_explicit_*.py` /
`32_*` 〜 `39_*` の validation スクリプトを順次 3 指標 AND gate に更新する作業を
TODO に積む。Cosserat 梁実装と並行可能。

### 6.3 副次 — 候補 (q3) implicit + AL n>2 復活

不変。中期 fallback。

### 6.4 副次 — 候補 (h5) bending 段階処方

不変。

## 7. MCDD 脱法 pattern 回避（本 status の自己点検）

- **pattern 1（tol 緩和）**: 10% gate を変更せず、explicit ケース全 FAIL を直接報告
- **pattern 5（既存テスト skip）**: 既存 743 test 全 pass、新規追加なし
- **pattern 6（骨格 status）**: status-387 の誤判定を 3 指標 AND gate で**実機 14
  ケース定量反証** + CLAUDE.md ルール追記 + status-387 訂正通知の三段で完結
- **pattern 8（根拠なき主張）**: 全 14 ケースで (|u_x|, |u_z|, L_arc) を表形式で
  公開、L_arc=234mm（梁伸び 2.3x）等の異常値を実証根拠として提示
- **pattern 10（TODO 先送り）**: status-387 撤回 + 訂正は本 status で完結

## 8. STA2 防止 — 自己点検チェックリスト

| 項目 | 該当チェック |
|------|---|
| increment の定義 | カットバック ≠ increment（実装変更なし、既存規範維持） |
| 結果の再現性 | tee ログ `/tmp/n_inc_sweep_v2_*.log` 保存、本 status §3 表に転記 |
| 数値の捏造禁止 | status-387 「err 0.58% 達成」誤判定を本 status で正式撤回 |
| **3 指標同時一致**（status-388 追加） | implicit baseline のみ PASS、全 explicit ケース FAIL を表形式報告 |

## 9. 引継ぎコマンド

```bash
# 訂正版 sweep（54s × 11 + 31s × 3 ≈ 11 分）
uv run --extra dev python work/beam_hysteresis/40_explicit_n_inc_sweep.py \
    2>&1 | tee /tmp/n_inc_sweep_v2_$(date +%s).log

# 回帰（status-387 と同じテスト数 743 passed 5 skipped）
uv run --extra dev pytest xkep_cae/contact/ xkep_cae/mathematics/ \
       xkep_cae/time_integration/ \
       xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py
uv run --extra dev python contracts/validate_process_contracts.py
uv run --extra dev ruff check xkep_cae/ tests/ && \
       uv run --extra dev ruff format --check xkep_cae/ tests/
```

## 10. 観察 — 開発運用

### 効果的だった点

- **「3 個以上の解析解同時一致」ルールの即時適用**: ユーザー指摘から 1 ステップで
  status-387 の誤判定を実機データで反証（implicit PASS / 全 explicit FAIL の対比
  が極めて明瞭）。L_arc=234mm の数値が出た瞬間に「梁が 2.3x に伸びている = 完全
  非物理」と即断できた。
- **多重集合一致による符号規約吸収**: 実装と解析解で u_x/u_z の役割が入れ替わる
  ことを許容する {|u_x|, |u_z|} の sort 比較は、実装座標系の差分を gate 判定から
  排除しつつ kinematic 一致を厳密に検証できる優れた設計。

### 学び — 単一指標一致の罠

status-387 で起きた事象は、**「特定の数値 (max\|u\|) が偶然解析解と一致する非物理
解」が存在する**ことを実証した教訓的事例:

1. 梁が 234mm に伸びる
2. 曲率が 1/3 に薄まる
3. 半径 R が 3 倍に増える
4. 先端変位 (u_x, u_z) ≈ (-38, +62) で |u| ≈ 73mm
5. 90° 解析解 73.30mm と「err 0.58% 一致」

座標系の単純な変位値だけ見ると正しく見えるが、実は梁が完全に物理規則を破って
ストレッチしている。**energetics or geometric の独立指標がなければ検出不可能**。

これは将来 19 本撚線等の複雑系で同じ罠を回避するための**普遍的教訓**:

- 必ず L_arc / SE / W_ext / κ(s) などの独立 anchor を gate に含める
- 単一指標 PASS は「PASS」と言わない、必ず multi-indicator gate
- gate 設計は実装に取り掛かる**前**に 3 指標を予定し、解析解値と gate 閾値を
  先行確定する

### 観察 — 次セッション向け

- **status-388 で MCDD ロードマップが clean になった**: explicit + UL は本質的に
  破綻、(z2) Cosserat が唯一の本質解決路と確定。Cosserat 着手の Phase 設計を
  status-389 で開始する（要素・歪み・接線・回転更新の 4 分割を見込む）。
- **既存 validation スクリプト 30〜39 番の 3 指標化**: TODO 化、Cosserat 実装と
  並行で順次更新。failed ケースは過去判定が信頼できないので再判定する。
