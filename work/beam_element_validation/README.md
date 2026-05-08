# work/beam_element_validation — Phase α/β/γ/δ 系統的再検証

[← project README](../../README.md)

status-389 §2 計画に基づく **梁要素 1 つから系統的再検証** スクリプト群。
status-388 透明性ルール（独立解析解 3 個以上同時一致）を全ケースで適用する。

## Phase α — 1 要素静的検証（implicit static）

CR Timoshenko 3D 梁要素 1 つに 4 つの基礎荷重ケースを implicit static で適用、
3 指標 AND gate で foundation 健全性を確認する。**全 4 ケース PASS**（status-390）。

| スクリプト | ケース | gate 3 指標 | 結果 |
|---|---|---|---|
| `41_alpha1_axial_tension.py` | 純軸引張 F_x=100 N | u_x / u_z(=0) / L_arc | **PASS** (機械精度) |
| `42_alpha2_pure_bending_small.py` | 純粋曲げ small κ M_y=10 N·mm | \|u_z\| / \|θ_y\| / \|f_int\| | **PASS** (機械精度) |
| `43_alpha3_pure_bending_large.py` | 純粋曲げ large κ θ_y=0.15 rad | \|u_x\| / \|u_z\| / L_chord (Hermite 解) | **PASS** (機械精度) |
| `44_alpha4_pure_shear.py` | cantilever 横荷重 F_z=0.01 N | \|u_z\| / \|θ_y\| / \|M_base\| | **PASS** (機械精度) |

### 重要な学び（α-3 から）

1 要素 CR は **chord 長保存制約** により、純粋曲げで `α = θ_R/2` だけ chord rotation
する Hermite 解を出す。これは true circular arc（curve length 保存）とは異なり、
特に `u_x` で 25% の差が出る。これは **1 要素の本質的離散化誤差** で、Phase γ で
n_elements を増やすと circular arc に収束するはず。**CR 局所剛性 + Battini-Pacoste
接線そのものは正しく動作**しており、foundation は健全。

### 符号規約

実装の局所剛性は XZ 平面で `Ke[u_z, θ_y] = +6 EI/L²` 規約（`M_y > 0` →
chord が y 軸まわりに正回転 → tip が −z 方向に変位）を採用。一方 status-389
plan の解析式は `u_z_tip = +M·L²/(2·EI)` と書いており、両者で **符号が逆**。
status-388 透明性ルール「絶対値多重集合一致」を踏襲し、kinematic 量は
`compare_abs=True` で吸収する（gate 判定に影響なし）。

## Phase β — 1 要素動的検証（status-391 で完了、両ケース PASS）

α 完了後の explicit dynamics を 1 要素で検証。**全 2 ケース PASS**（status-391）:

| スクリプト | ケース | gate 3 指標 | 結果 |
|---|---|---|---|
| `45_beta1_free_vibration.py` | 自由振動 v_z(tip)=1 mm/s | T_FE / \|u_z_max\| / E_drift | **PASS** (T 0.06% / u 4.85% / E 0.02%) |
| `46_beta2_explicit_quasistatic.py` | prescribed θ_y=0.15 rad slow ramp | \|u_x\| / \|u_z\| / L_chord (Hermite 解) | **PASS** (機械精度 0.000%) |

### β-2 PASS の重要含意

**1 要素 explicit + slow ramp + 質量比例減衰** は α-3 implicit Hermite 解と
**機械精度（0.000%）で完全一致**。これは:

- **CR foundation は explicit dynamics 領域でも健全**
- status-381〜387 explicit + UL の精度問題は **CR 要素自体ではなく上位層**（assembler /
  UL formulation / mass scaling 戦略）に局在
- **(z2) Cosserat 路線は absolute necessity ではない** — 主目的は explicit + 大回転
  robust 化（assembler / UL 由来の問題解消）に絞れる

### 共通ヘルパ

- `_beta_common.py`: `solve_explicit_central_diff` (leap-frog Verlet) /
  `compute_natural_frequencies_fe` (K_aa φ = ω² M_aa φ) /
  `compute_strain_energy_cr` (corotational SE) / `measure_period_zero_crossings`.
- 1 要素 12 DOF を `timo_beam3d_cr_internal_force` 直接呼出で駆動、assembler 経由なし.

## Phase γ — multi-element 検証（status-392 で完了、4/5 ケース PASS + O(1/n²) 収束実証）

n_elements ∈ {1, 2, 4, 8, 16} で α-3 を再実施し circular arc への収束を確認。
**4/5 ケース PASS**（n=1 のみ FAIL は α-3 で実証済み chord 長保存制約による既知の
25% 離散化誤差で期待通り）。

| スクリプト | ケース | gate 3 指標 | 結果 |
|---|---|---|---|
| `47_gamma_multi_element_convergence.py` | n_elements ∈ {1,2,4,8,16}、θ_y=0.15 rad | \|u_x\| / \|u_z\| / L_chord (arc 解) | **4/5 PASS** |

| n_elements | err(\|u_x\|) [%] | err(\|u_z\|) [%] | err(L_chord) [%] | gate (10%) |
| ---: | ---: | ---: | ---: | :---: |
|  1 | 24.95 | 0.094 | 0.094 | **FAIL** (u_x のみ) |
|  2 |  6.23 | 0.023 | 0.023 | PASS |
|  4 |  1.56 | 0.006 | 0.006 | PASS |
|  8 |  0.39 | 0.001 | 0.001 | PASS |
| 16 |  0.10 | 0.000 | 0.000 | PASS |

- **log-log slope of err(u_x) vs n (n≥2): -2.000**（理論値 O(1/n²) と完全一致）
- **CR closed form 一致**: 全 5 ケースで \|u_x\| / \|u_z\| / L_chord すべて
  **機械精度（10⁻¹³%〜10⁻¹²%）** — 実装は CR 多要素 chord rotation 解析理論と完全整合
- **「16 要素/ピッチ厳守」規範のマージン確認**: θ=0.15 rad ≈ 8.6° 単一曲げで
  n=2 から 10% gate を通過、n=16 で 0.1% に縮小

### 共通ヘルパ

- `_gamma_common.py`: `ChainedBeamSection` (n_elements 拡張 BeamSection) /
  `assemble_internal_force` / `assemble_tangent` (要素ループ直接アセンブル) /
  `solve_static_nr_chain` (multi-element NR static、load stepping + prescribed disp 対応) /
  `compute_chord_total` / `compute_polyline_length`.
- assembler 経由を避け、`timo_beam3d_cr_*` を直接呼び出してアセンブル.

## Phase δ — 接触あり 2 本撚線（γ 完了後、次セッション副次）

最小規模の接触系（2 本撚線、平行配置、軽荷重）で 3 指標一致を確認。

## assembler / UL 1 要素再現実験（status-394 で完了、改修対象を explicit + UL のみに局在化）

status-393 §6.1 で次セッション最優先候補として明示された assembler / UL
update_reference の 1 要素規模再現実験。**4 モード中 D のみ FAIL** で改修対象を
局在化。

| スクリプト | ケース | 4 モード比較 | 結果 |
|---|---|---|---|
| `49_beta2_with_assembler_ul.py` | α-3 / β-2 と同 BC（θ_y=0.15 rad）を 4 通りの実装パスで | A: implicit+assembler+TL / B: implicit+assembler+UL / C: explicit+assembler+TL / D: explicit+assembler+UL（毎 step） | **A/B/C PASS（機械精度 0.000%）/ D FAIL（u_x 99.85% / u_z 96.14% アンダー）** |

### Mode D 失敗の物理的解釈（status-394 §3）

毎 step UL 更新 → `u_incr` がほぼゼロにリセット → `f_int(u_incr) ≈ 0` で elastic
restoring force が発達しない → reference が処方値に追従するだけで deformation が
elastic energy に変換されない。これは status-382 §3 で推定された UL update_reference
凍結の正しい診断であることを 1 要素規模で定量実証する。

### 含意

- **改修対象は explicit + UL update_reference per step の組合せのみ**に局在
- **(z2) Cosserat 路線は不要**（β-2 直接駆動 + Mode A/B/C で foundation 健全実証 +
  Mode D のみ FAIL）
- **次セッション最優先**: 候補 (z3) explicit モード TL 固定 API 化
  （`explicit_ul_update_interval=0` で update_reference を一切呼ばない解釈）+
  19 本撚線適用

## 実行方法

```bash
# 個別実行
uv run --extra dev python work/beam_element_validation/41_alpha1_axial_tension.py \
    2>&1 | tee /tmp/alpha1_$(date +%s).log

# Phase α 全 4 ケース
for i in 41 42 43 44; do
    uv run --extra dev python work/beam_element_validation/${i}_*.py 2>&1
done | tee /tmp/alpha_all_$(date +%s).log

# Phase β 全 2 ケース
for i in 45 46; do
    uv run --extra dev python work/beam_element_validation/${i}_*.py 2>&1
done | tee /tmp/beta_all_$(date +%s).log

# Phase γ multi-element
uv run --extra dev python work/beam_element_validation/47_gamma_multi_element_convergence.py \
    2>&1 | tee /tmp/gamma_$(date +%s).log
```

## 共通ヘルパ

- `_alpha_common.py`: `BeamSection` / `solve_static_nr` / `MetricRow` /
  `evaluate_three_metric_gate` / `run_case` を提供。
- 1 要素 12 DOF を `xkep_cae.elements._beam_cr.timo_beam3d_cr_internal_force` /
  `timo_beam3d_cr_tangent_analytical` で直接ドライブ。assembler 経由を避け、
  foundation を最小単位で検証する。
