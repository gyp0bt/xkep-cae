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

## Phase β — 1 要素動的検証（α 完了後）

α 完了後、explicit 動的を 1 要素で検証する予定:

| スクリプト | 内容 |
|---|---|
| `45_beta1_free_vibration.py` (TODO) | 1 要素 自由振動、SDoF Timoshenko 第 1 モード |
| `46_beta2_explicit_quasistatic.py` (TODO) | 1 要素 prescribed θ_y を explicit + slow ramp |

**Phase β-2 で 1 要素 explicit が 3 指標 FAIL → (z2) Cosserat 移行根拠 absolute 確定**、
PASS なら CR foundation 健全 + (z2) は explicit + 大回転 robust 化に絞れる。

## Phase γ — multi-element 検証（β 完了後）

n_elements ∈ {2, 4, 8, 16} で α-3 を再実施し circular arc への収束を確認。
「16 要素/ピッチ厳守」の妥当性を再確認する。

## Phase δ — 接触あり 2 本撚線（γ 完了後）

最小規模の接触系（2 本撚線、平行配置、軽荷重）で 3 指標一致を確認。

## 実行方法

```bash
# 個別実行
uv run --extra dev python work/beam_element_validation/41_alpha1_axial_tension.py \
    2>&1 | tee /tmp/alpha1_$(date +%s).log

# 全 4 ケース連続実行
for i in 41 42 43 44; do
    uv run --extra dev python work/beam_element_validation/${i}_*.py 2>&1
done | tee /tmp/alpha_all_$(date +%s).log
```

## 共通ヘルパ

- `_alpha_common.py`: `BeamSection` / `solve_static_nr` / `MetricRow` /
  `evaluate_three_metric_gate` / `run_case` を提供。
- 1 要素 12 DOF を `xkep_cae.elements._beam_cr.timo_beam3d_cr_internal_force` /
  `timo_beam3d_cr_tangent_analytical` で直接ドライブ。assembler 経由を避け、
  foundation を最小単位で検証する。
