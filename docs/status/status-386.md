[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-386: 候補 (z1d) `t_cycle` 下限緩和実装 — z1d は方向自体が逆と実証、explicit + UL の精度 gate 未達続行

**日付**: 2026-05-01
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12+5+17+11+6 passed
（status-385 比 +6 = `TestTCycleMinSeconds`）

## 概要

status-385 §6.1 最有力候補 **(z1d) `t_cycle` 下限緩和** を実装。`StrandBendingOscillationConfig`
に `t_cycle_min_seconds: float = 1.0` field を追加し、`t_cycle = max(10·T1,
cfg.t_cycle_min_seconds)` で下限を外部制御可能化。default 1.0 で既存挙動完全保持。

**Gate (MCDD 凍結解除条件 (5))**: 90° 単梁曲げの解析解 73.30mm に対し
`|max|u_explicit| − u_anal| / u_anal < 0.10`。

**結論（候補 (z1d) 却下方向）**:

- **z1d 自体は設計通り動作**: t_cycle_min=0.0 で `t_cycle = 10·T1 = 67ms`、
  dt_sub が 1.0s/20=50ms → 67ms/20=3.35ms に **15x 縮小**、initial target β
  も 4.6e+04 → 3.1e+03 に同 15x 縮小（ログ確認）。
- **implicit 側 regression なし**: t_cycle_min=0.0 でも frac=1.0 完走、
  err 4.86%（baseline 3.90% との差は 1pt 未満で gate 内）。
- **explicit 側で逆効果**: 単梁 11 ケース掃引で全 FAIL。
  - selective + z1d (β_outside=10): 全 DIVERGED（target β=3e3 が β_stiff_max=1e3 超過）
  - non-selective uniform β² + z1d: frac=1.0 完走するも max|u|=0.77mm（**err 99%**）
  - 大 β_outside=2000 + z1d: 同様に max|u|=1.83mm（**err 97.5%**）
  - **逆方向対照 (n_inc=200, t_cycle 据え置き)**: max|u|=6.57mm（err 91.0%）
    で z1d 方向より **10x 改善** — z1d は逆効果と定量実証。

**真の物理原因**: 質量スケーリング β は弾性波速度を `c → c/β` に減速する。
波の梁長 L 横断時間は `β·L/c`。`β=3000`, L=100mm, c=3.81e6mm/s で **78ms**。
`t_cycle = 67ms`（z1d）では波が梁を **1 回も横断できず**、変形が先端まで伝播
しないまま frac=1.0 到達 → max|u| が解析解の 1% 程度に留まる。z1d は loading
rate を物理スケールへ揃える方向だが、mass scaling の波速減速と組合さって
実効的に dt_total < t_traverse となり精度が崩壊する。

`t_cycle_min_seconds` field は **default 1.0 で保持**（implicit 側 opt-in 無害）、
将来 explicit + Cosserat 梁等で「物理スケール loading rate」が必要になった場合の
opt-in API として残置。

## 1. 実装

### 1.1 `StrandBendingOscillationConfig` field 追加

`xkep_cae/numerical_tests/strand_bending_oscillation.py`:

```python
# status-386 候補 (z1d): t_cycle 下限緩和 — 物理 T1 ベース loading rate.
# 従来 `t_cycle = max(10·T1, 1.0)` の **下限 1 秒** を本 field で外部指定可能化。
# default 1.0 で既存挙動完全保持（implicit 7 本 frac=1.0 維持）。
#
# 0.0: 下限を完全に外し純粋 `t_cycle = 10·T1` を採用（最も aggressive、
#       loading rate を物理スケールで決定）。
t_cycle_min_seconds: float = 1.0
```

### 1.2 `t_cycle` 計算 2 箇所の修正

```python
# Before
t_cycle = max(10.0 * T1, 1.0)

# After
t_cycle = max(10.0 * T1, cfg.t_cycle_min_seconds)
```

修正箇所: `StrandMeshProcess.process()` 内 L≈920 / L≈1170（曲げ単独 / 統合モード）。
L=1488 の `t_osc = t_cycle * cfg.n_oscillation_cycles` は計算済み `t_cycle` を
継承するので自動反映。

### 1.3 単体テスト追加（+6、`TestTCycleMinSeconds`）

`xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py`:

- `test_default_is_one_second`: default 1.0 維持
- `test_field_is_overridable`: 0.0 を含む任意値で構築可能
- `test_field_accepts_intermediate_values`: 0.1 等の中間値も受理
- `test_t_cycle_floor_logic_default`: 7 本撚線 T1≈6.7ms 想定で 1.0s 支配
- `test_t_cycle_floor_logic_relaxed`: t_cycle_min=0.0 で 10·T1=67ms 支配
- `test_t_cycle_floor_logic_partial_relax`: t_cycle_min=0.1 で 10·T1<0.1 → 0.1 支配

## 2. 実機検証 — `39_z1d_t_cycle_validation.py`

### 2.1 単梁 90° カンチレバー曲げ（接触なし、L=100mm、E=130GPa）

解析解 max|u| = 73.30mm（quarter circle, R=2L/π）。

| # | label | frac | max\|u\| [mm] | err_anal | gate |
|---|-------|------|--------------|----------|------|
| 1 | implicit_baseline (default) | 1.000 | 70.45 | 3.90% | **PASS** |
| 2 | implicit_z1d_only (t_cycle_min=0.0) | 1.000 | 69.74 | 4.86% | **PASS** |
| 3 | exp_z1c_only (status-385 baseline) | 0.000 | DIVERGED | — | FAIL |
| 4 | exp_z1c_z1d (β_out=10) | 0.000 | DIVERGED | — | FAIL |
| 5 | exp_z1c_z1d_modest (β_out=3) | 0.000 | DIVERGED | — | FAIL |
| 6 | exp_z1c_z1d_partial (t_cycle_min=0.1) | 0.000 | DIVERGED | — | FAIL |
| 7 | exp_z1c_z1d_damp_relax (α=10, relax=500) | 0.000 | DIVERGED | — | FAIL |
| 8 | exp_z1d_uniform_beta (selective=False) | 1.000 | **0.77** | **98.95%** | FAIL |
| 9 | exp_z1d_uniform_damp_relax | 1.000 | **0.68** | **99.07%** | FAIL |
| 10 | exp_z1c_z1d_large_outside (β_out=2000) | 1.000 | **1.83** | **97.51%** | FAIL |
| 11 | **exp_more_increments (n_inc=200, t_cycle 据え置き)** | 1.000 | **6.57** | **91.03%** | FAIL |

### 2.2 ログ観察 — z1d は確かに動作している

| 設定 | initial target β | 比 |
|------|----------------:|----:|
| t_cycle_min=1.0 (status-385 7 本) | 4.7e+04 | baseline |
| t_cycle_min=0.1 (#6) | 4.6e+03 | 1/10 |
| t_cycle_min=0.0 (#4,#10) | 3.1e+03 | 1/15 |

15x 縮小は `t_cycle 1.0s → 0.067s` の比率と一致、(z1d) は数値的に設計通り作動。

### 2.3 cutback 後 target β 急騰の解析

selective=True + z1d で `[MASS_SCALE] β=1→1000 (target 3.099e+03, cap 1e3)` 直後に
cutback され、cutback 後 `target=5.659e+04` に急騰（18x 増加）。

**原因**: `_explicit_dynamic.py` の dt_c_beam スケーリングは
`mass_scaling_beta_outside` のみ（mask 設定時、L=380）。一方 dt_c_gers は
`mass_scaling_beta` で scaling。

```
dt_c_gers ∝ β = 1000           (stiff DOF 寄与、scale 完了)
dt_c_beam ∝ β_outside = 10     (beam DOF 寄与、scale 不足)
dt_c = min(...)                 → beam 側がボトルネック
target_β = β·dt_sub/(0.9·dt_c) = 1000·dt_sub / (0.9·10·1.64e-6)
        ≈ 226 · (dt_sub/dt_sub_init)
```

cutback で dt_sub→dt_sub/4 されても dt_c_beam は固定なので target_β はあまり減らず、
β を 1000 まで上げてしまった分だけ累積比 56700 程度になる。**この挙動は
selective + z1d の併用で常に発生**（API bug ではなく設計の数学構造）。

### 2.4 真の物理原因 — 弾性波伝播時間 vs t_cycle

mass scaling β は質量を β² 倍化、弾性波速度を `c → c/β` に減速する。
梁長 L=100mm を波が横断する時間:

| β | wave traverse time | t_cycle (z1d) | 横断回数 |
|--:|------------------:|--------------:|--------:|
| 1 | 26 μs | 67 ms | ~2600 回 ✓ |
| 1000 | 26 ms | 67 ms | 2.6 回 ✗ |
| 3000 | 78 ms | 67 ms | **0.86 回（1 回未満）** ✗ |

z1d で `t_cycle = 67ms` に縮小すると、`β=3000` 程度で波が梁を **1 回も
横断できない**。BC 端から印加された変位パルスが先端に到達する前に解析が
frac=1.0 で終了 → max|u| が解析解の 1% 程度しか出ない。

#### 逆方向対照実験 (#11)

`t_cycle=1.0s`（default）+ `n_increments=200`（10x）で dt_sub=5e-3s:

- target β は z1d (#8) と同程度 (~3e+03)
- t_total=1.0s で波横断 ~38 回（β=1000 想定）
- 結果: max|u|=6.57mm — z1d 方向 (#8 0.77mm) の **10x 改善**

精度 gate（10%）には依然届かないが、**「t_cycle 縮小は逆効果」が定量的に裏付け**
られた。

### 2.5 物理的妥当性 gate (status-381 §5)

z1d 適用ケース（#8〜#10）は max|u| ≪ 73.30mm（解析解）で under-prediction が
劇的、すべて (5) `精度 < 10%` 違反。frac=1.0 と max|u|<L_strand×10 は満たすが
解として成立せず、status-380/381 で追加された (3)(5) gate の重要性が再確認された。

## 3. MCDD 凍結解除条件 — 条件 (5) 未達続行

| 条件 | 状態 |
|------|------|
| (1) Phase E 完了 | ✅ status-357 |
| (2) 19 本 frac=1.0 完走 | △ explicit 系で精度不足のため事実上未達 |
| (3) max\|u_trans\| < L_strand × 10 | ✅ implicit / 一部 explicit |
| (4) `KcNormalDirectionStiffness` FD rel_err < 1e-2 | ✅ status-356（2.18×10⁻⁷） |
| **(5) 解の精度 < 10%** | **❌ z1d 方向追加適用でむしろ悪化** |

## 4. 実装変更まとめ

- `xkep_cae/numerical_tests/strand_bending_oscillation.py`:
  - `StrandBendingOscillationConfig.t_cycle_min_seconds: float = 1.0` 追加
  - `t_cycle = max(10.0 * T1, cfg.t_cycle_min_seconds)` に置換（2 箇所）
- `xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py`:
  - `TestTCycleMinSeconds` クラス + 6 テスト追加
- `work/beam_hysteresis/39_z1d_t_cycle_validation.py` 新設（+296 行、11 ケース）

回帰: 全 24 契約検査 OK / contact + math + time_integration + strand_bending_osc =
**743 passed 5 skipped**（status-385 比 +6）/ `test_helical_3d_hermite`
rel_err=2.18×10⁻⁷ 維持 / 7 本 implicit frac=1.0 / ruff check + format pass。

## 5. 引継ぎ — 次 status の候補

### 5.1 z1d 路線は終結（field は default 1.0 で保持）

- `t_cycle_min_seconds` を default 1.0 で残置（implicit 完全保持、explicit opt-in）
- 検証スクリプト `39_z1d_t_cycle_validation.py` は失敗実験記録として残置
  （status-358/360/363/375 と対称）

### 5.2 候補 (z2) Cosserat 梁プロトタイプ最優先

候補 (z1a/b/c/d) 全候補で精度 gate 未達が確定。**explicit + UL 路線は status-382/383
の根本欠陥（UL `update_reference` 凍結）が解消されない限り精度 gate 達成不可**。

UL を捨てた **geometrically exact (Simo-Reissner) Cosserat 梁** は:

- SO(3) 回転 DOF をネイティブに保持、reference 更新が不要
- 大回転 + 大変位での `f_int(u)` 評価が物理的に正しい
- explicit + 適切な mass scaling で波伝播・変形両方を正しく追従

実装規模 ~1000 行（要素・歪み・接線・回転更新）。

### 5.3 副次 — 候補 (q3) implicit + AL n>2 復活

status-376 で却下された候補 (g2) AL n>2 を Uzawa update under-relaxation で再試行。
Cosserat 路線が長期化した際の中期 fallback。

### 5.4 副次 — 「t_cycle 据え置き + n_increments 大」探索

(11) は `n_inc=200` で max|u|=6.57mm（z1d 方向の 10x 改善）。`n_inc=2000` 等
さらに増やせば波伝播時間に対し dt_sub が十分小さくなり、解析解に近づく可能性。
ただし全候補 (z1*) のなかで最良であっても精度 gate 達成は楽観できない（β
スケーリングと UL 凍結の本質欠陥は不変）。

## 6. MCDD 脱法 pattern 回避

- **pattern 1（tol 緩和）**: 精度 gate 0.10 を変更せず、未達と明記
- **pattern 5（既存テスト skip）**: 既存 737 test 全 pass、+6 追加
- **pattern 6（骨格 status）**: API 実装 + 6 単体テスト + 11 ケース実機検証 +
  ログ解析（target β 15x 縮小確認）+ 物理解析（波伝播時間 vs t_cycle 比）+
  逆方向対照実験 (#11) で完結
- **pattern 8（根拠なき主張）**: ログから initial target β 4.7e+04 → 3.1e+03
  の 15x 縮小、波伝播時間 78ms vs t_cycle 67ms の数値的不整合を実証根拠として提示
- **pattern 10（TODO 先送り）**: (z1d) は完結、(z2) Cosserat 梁は独立スコープなので
  別 status が適切

## 7. 引継ぎコマンド

```bash
# z1d 検証
uv run --extra dev python work/beam_hysteresis/39_z1d_t_cycle_validation.py \
    2>&1 | tee /tmp/z1d_$(date +%s).log

# 回帰
uv run --extra dev pytest xkep_cae/contact/ xkep_cae/mathematics/ \
       xkep_cae/time_integration/ xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py
uv run --extra dev python contracts/validate_process_contracts.py
uv run --extra dev ruff check xkep_cae/ tests/ && uv run --extra dev ruff format --check xkep_cae/ tests/
```

## 8. 観察 — 開発運用

### 効果的だった点

- **「単梁で精度 gate 確認するべき」のユーザー指摘**: 7 本撚線で云々する前に
  単梁で解析解 73.30mm と一致するか確認すべき、という方針修正で実装の本質課題
  （波伝播時間 vs t_cycle）に直接到達できた。
- **逆方向対照実験 (#11)**: z1d を否定するためには「z1d とは反対方向の操作で
  実際に改善する」ことを示すのが最も強い反証になる。z1d 方向 0.77mm vs 反対方向
  6.57mm の **10x 差** で「方向自体が逆」を定量的に確立。
- **target β 15x 縮小ログ**: API は設計通り動作している（pattern 8 回避）が
  物理結果は逆方向、という矛盾を target β の絶対値追跡で明確にした。

### 学び — 弾性波伝播時間が真の Δt 制約

Belytschko §6.4 の Courant 条件 `dt_sub ≤ 2/ω_max` は **数値安定性の必要条件**
だが、explicit + 大変位では別の制約 **「波が変形を伝える時間」** が暗黙に加わる。
mass scaling β はこの伝播時間を β 倍に伸ばすので、`t_total ≥ β·L_max/c_min` を
満たさないと最終解が定常解析解と一致しない。

status-385 §6.1 の予測「dt_sub 縮小 → target β 縮小 → gate 達成」は
**Courant 条件の必要条件しか考慮していなかった**。実際には dt_sub 縮小は
total time も縮小するため、波伝播時間との比 `(β·L/c) / t_total` が悪化する
（z1d 方向）。この混同を修正できたのが本 status の主な収穫。

### 観察 — 次セッション向け

- `solver_mode="explicit"` + UL 路線は status-382 で本質的不整合が確定、status-383
  で代替案 (q1) 失敗、status-385 で (z1c) cap 不足、本 status で (z1d) 方向逆と確定。
  4 status 連続で精度 gate 未達の探索を行ったため、**(z2) Cosserat 梁路線の
  着手を最優先にすべき**段階に到達。
- `t_cycle_min_seconds` field は将来 Cosserat 梁実装時に「真の物理スケール
  loading rate」が必要になる場面で有用な可能性があるため、default 1.0 で残置。
