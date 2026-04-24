# Contact Damping 設計仕様 (status-365 Phase 1)

[← README](../../../../README.md)

## 概要

接触ペア単位の法線減衰力 `f_damp = -c_n v_n n̂` を組み立てる Process。
候補 (e) 接触減衰 escape hatch（status-363 §4）の **Phase 1 インフラ**で、
本 status-365 では単体 Process + ユニットテストのみ、solver への配線は
Phase 2（status-366 予定）に分離する。

## 背景

status-363 の BT line search 4 ケース感度掃引で「BT 既定が局所最適、
パラメータチューニングで frac は伸びない」と確定し、line search では
active 集合振動を根本抑制できないと判定された。

次候補 (e) は Type D stall の震源である active×mixed 領域に対し、微小な
粘性を **escape hatch** として導入する手法。Generalized-α の C 行列を
直接書き換えず、接触ペア単位で組み立てた `f_damp + K_damp` を NR 残差
/ 接線剛性に加算することで、時間積分モジュールを無変更に保つ。

## 数理

### 法線相対速度

線形形状係数 `coeff = [(1-s), s, -(1-t), -t]`（`HuberContactForce.
_contact_shape_vector` と同符号、n̂ は A→B 外向き）を使い、

```
g_shape (12,) = [coeff_0·n̂, coeff_1·n̂, coeff_2·n̂, coeff_3·n̂]
v_local (12,) = [v(A0), v(A1), v(B0), v(B1)]   （各ノードの先頭 3 DOF）
v_n = g_shape · v_local
```

ここで v_n > 0 が「ペアが閉じる向きの速度」。

### 減衰力

```
f_damp_local = -c_n * v_n * g_shape              (12,)
```

A 側と B 側で符号が反転する構造で、`n̂ ⊗ n̂` 方向のみ成分を持つ（摩擦 K_st の
接線方向とは独立）。

### 接線剛性

Generalized-α で v = c1·(u - u_pred) + const （c1 = γ/(β·dt)）なので、

```
∂v_n/∂u = c1 * g_shape
∂f_damp_local/∂u = -c_n * c1 * g_shape ⊗ g_shape
```

減衰接線寄与（NR K に **加算** する向き、`K_eff += K_damp`）は

```
K_damp_local = c_n * c1 * (g_shape ⊗ g_shape)    (12, 12)
```

対称半正定値（rank-1 ブロック）で `c_n ≥ 0` なら常に安定化側。

### 消散エネルギー率

```
E_damp_rate = Σ_active c_n * v_n²   [エネルギー/時間]
```

常に ≥ 0（散逸性）。呼び出し側が dt を乗じて `E_damp_increment` を積算する。

## Process 一覧

| Process | 用途 |
|---------|------|
| `ContactNormalDampingProcess` | 接触ペアごとに `f_damp` / `K_damp` / `energy_rate` を組み立てる単体 Process |
| `ContactDampingEnergyMonitorProcess` | `SolverResultData.damping_energy_history` と `EnergyHistory.entries[].strain_energy` を照合し、E_damp_cumulative / E_strain 比の最大値 / 最終値 / budget 超過件数を出力する PostProcess（Phase 2 追加） |

## Phase 2（status-366 完了）の配線

1. `ContactFrictionProcess.damping_slot` = `StrategySlot(object, required=False,
   default_types=(ContactNormalDampingProcess,))` を追加（`uses` グラフ経由で
   到達可能化）、default は `c_n=0` で no-op
2. `_newton_dynamic.py` の NR 反復内で、`effective_residual`/`effective_stiffness`
   適用後に `R_u += f_damp` / `K_T += K_damp` を加算（`c_n>0` かつ
   `manager.pairs` が非空、`dt_sub>1e-30` のときのみ）
3. `ContactFrictionInputData.contact_damping_coefficient` /
   `contact_damping_energy_budget_ratio` を `StrandBendingOscillationConfig`
   → `ContactFrictionInputData` → `NewtonDynamicInput` の 3 層で plumb through
4. `DynamicStepOutput.damping_energy_rate` を追加、`ContactFrictionProcess` が
   dt を乗じて `damping_energy_history` に累積、`SolverResultData` に公開
5. `ContactDampingEnergyMonitorProcess` を新設（PostProcess）。成功後に
   E_damp / E_strain の最大値・最終値・budget 超過件数を report 出力
6. 7本撚線で 1/2/5/10/20% 減衰 budget 実測 → 19本 Type D stall 検証
   （MCDD 凍結解除条件: frac=1.0 完走 + E_damp/E_strain < budget）

## MCDD との関係

Phase 1 の `ContactNormalDampingProcess` は `TermExpansionContract("K_c_term_expansion")`
の 5 項分解（material/geo/st/closest/hermite_adj）とは独立系統。減衰項は
Generalized-α の C 行列経路の代替であり、K_c の解析的接線拡張ではない。

Phase 2 での solver 配線時に、`ContactKcComponentFDDiagnosticProcess`
（status-343）と同等の診断 Process を新設するかは要検討（`K_damp` は
rank-1 な outer product なので FD 検証の必要性は低い — unit test で機械精度
の整合性を確保済み）。
