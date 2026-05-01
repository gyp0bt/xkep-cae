[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-385: 候補 (z1c) 2 段階質量スケーリング API（β_stiff + β_outside）実装 — API 完成、validation で β_stiff cap が支配的と確認、(z1d) loading rate 縮小が必須と判明

**日付**: 2026-05-01
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12+5+17+11 passed（status-384 比 +11 = `TestTwoStageMassScaling`）

## 概要

status-384 §6.1 最有力候補 **(z1c) 2 段階質量スケーリング** を実装した。
status-384 では `selective mass scaling` で β² 倍化を stiff DOF（接触ペナルティ等）
に限定したが、残り 84% の梁 DOF が β=1 のまま dt 制限を支配し、target β=8.8×10⁶
（cap 1e3 を超過）で frac<<1.0 となる課題が残った。

**(z1c) の解**: 梁 DOF にも **modest な β_outside** を別途与える。stiff DOF は
aggressive な β² 倍化（auto-tune）、梁 DOF は β_outside² の固定 modest 倍化を
適用する 2 段階構成。`mass_scaling_beta_outside` 引数を独立追加し、
`_compute_scaled_mass()` を mask に応じて両 β を切替適用する。

**API 完成**: 11 単体テスト全 pass、KE 保存 v/a リスケールも mask に応じて
selective に適用。

**validation 結果（8 ケース）**: API は設計通り動作（log で post-cutback target β
が β_outside=10 で 8.8e6 → 8.8e5 に **10x 縮小** 確認）。しかし全ケース frac=0
で divergence、または frac=0.4 程度進んだ後 max|u|=1.6e5mm の数値発散。

**根本要因**: initial target β=4.7e4（β_stiff cap 1e3〜1e4 を超過）が支配的。
β_outside による梁 DOF dt 拡大は post-cutback で機能するが、initial cutback 前
は stiff DOF の Courant 制約が dominant で β_stiff cap に当たる。

**結論**: (z1c) infrastructure は完成。MCDD 凍結解除条件 (5) 達成には
**(z1d) `t_cycle` 下限緩和** で loading rate を物理 T1 ベースに縮小し、
target β 自体を物理時間スケールに合わせて下げる必要がある。

## 1. 実装

### 1.1 `ExplicitCentralDifferenceProcess` API 拡張

`xkep_cae/time_integration/strategy.py`:

```python
def __init__(
    self,
    mass_matrix,
    *,
    mass_scaling_beta: float = 1.0,
    mass_scaling_beta_outside: float = 1.0,  # ← 新設
    ...
) -> None:
    if mass_scaling_beta_outside < 1.0:
        raise ValueError(...)
    self.mass_scaling_beta_outside = float(mass_scaling_beta_outside)
    ...

def _compute_scaled_mass(self, beta: float) -> np.ndarray:
    if self._mass_scaling_dof_mask is None:
        return (beta * beta) * self._M_lump_raw  # 既存挙動（mask=None）
    beta_out_sq = self.mass_scaling_beta_outside * self.mass_scaling_beta_outside
    scaling = np.full_like(self._M_lump_raw, beta_out_sq)
    scaling[self._mass_scaling_dof_mask] = beta * beta
    return scaling * self._M_lump_raw

def set_mass_scaling_beta_outside(
    self, beta_outside: float, *, rescale_state: bool = True
) -> None:
    """β_outside 単調増加 + KE 保存 v/a リスケール（mask=False DOF のみ）."""
    ...
```

### 1.2 `_explicit_dynamic.py` dt_c_beam 推定の更新

```python
if np.isfinite(dt_c_beam):
    if time_strategy._mass_scaling_dof_mask is None:
        dt_c_beam *= time_strategy.mass_scaling_beta            # 既存（z1a 単独）
    else:
        dt_c_beam *= time_strategy.mass_scaling_beta_outside    # 新（z1c）
dt_c = min(dt_c_gers, dt_c_beam)
```

mask=False の梁 DOF は β_outside² で倍化されるため、element-wise 物理的下限
にも β_outside を乗じて dt_c の整合を取る。

### 1.3 plumb-through

| 層 | 追加 field |
|----|-----------|
| `ExplicitCentralDifferenceProcess.__init__` | `mass_scaling_beta_outside: float = 1.0` |
| `_create_time_integration_strategy` | 同 |
| `default_strategies()` | 同 |
| `ContactFrictionInputData` | `explicit_mass_scaling_beta_outside: float = 1.0` |
| `StrandBendingOscillationConfig` | 同 |
| `strand_bending_oscillation.py` 3 経路 | `cfg.explicit_mass_scaling_beta_outside` を伝搬 |
| `process.py` | `getattr(input_data, "explicit_mass_scaling_beta_outside", 1.0)` で factory に渡す |

### 1.4 単体テスト追加（+11、`TestTwoStageMassScaling`）

`xkep_cae/contact/solver/tests/test_explicit_dynamic.py`:

- default β_outside=1.0 で z1b 等価（mask True: β², False: 1）
- β_outside=2 + mask=[T,F,T,F]: M_lump=[100,4,100,4]
- mask=None で β_outside ignored（全 DOF が β² 一律）
- β_outside<1.0 で `ValueError`
- `set_mass_scaling_beta_outside()` 単調増加 + monotone reject
- 不正値 reject
- KE 保存 v/a rescale が mask=False DOF のみ適用
- mask=None 時は β_outside 上方更新でも v 不変（β_outside 自体無効）
- `M_lump_inv` が β_outside 上昇で縮小（dt_c 拡大）
- `_create_time_integration_strategy()` 経由で β_outside 伝搬

## 2. 実機検証 — `38_z1c_two_stage_validation.py`

### 2.1 単梁 90° 曲げ（接触なし、L=100mm）

| ケース | β_out | β_stiff_max | frac | max\|u\| [mm] | 解析解誤差 | gate |
|--------|-------|-------------|------|--------------|-----------|------|
| implicit_baseline | — | — | 1.000 | 70.45 | 3.90% | PASS |
| exp_z1b_selective_only | 1 | 1e3 | 0.000 | DIVERGED | — | FAIL |
| exp_z1c_beta_outside_10 | 10 | 1e3 | 0.000 | DIVERGED | — | FAIL |
| exp_z1c_beta_outside_100 | 100 | 1e3 | 0.000 | DIVERGED | — | FAIL |

**観察**: 単梁 K がほぼ一様 → stiff threshold で 32/102 DOF が「stiff」として
誤検出されるが、initial target β=4.6e4 が β_stiff cap=1e3 を超過し cutback。
β_outside の値に関わらず stiff DOF cap で死ぬ。

### 2.2 7 本撚線 90° 曲げ（接触あり、stiff DOF 検出される）

| ケース | β_out | β_stiff_max | α | frac | max\|u\| [mm] | gate |
|--------|-------|-------------|---|------|--------------|------|
| implicit_baseline_7s | — | — | — | 0.373 | 27.84 | FAIL（Type D stall） |
| exp_z1b_only_7s | 1 | 1e3 | 0 | 0.000 | DIVERGED | FAIL |
| exp_z1c_beta_out_10_7s | 10 | 1e3 | 0 | 0.000 | DIVERGED | FAIL |
| exp_z1c_beta_out_100_betamax_1e4_7s | 100 | 1e4 | 0 | 0.000 | DIVERGED | FAIL |
| exp_z1c_beta_out_10_betamax_1e6_7s | 10 | 1e6 | 10 | 0.425 | **1.6e5** | FAIL（数値発散） |

**重要な log 観察** — z1c は確かに動作している:

```
[z1b only post-cutback]   target β=8.819e+06 > 1.000e+03 → cutback
[z1c β_out=10 post-cutback] target β=8.819e+05 > 1.000e+03 → cutback   ← 10x 縮小
[z1c β_out=100 post-cutback] target β=8.819e+05 > 1.000e+04 → cutback  ← 同じ
```

post-cutback で target β が β_outside=10 で 8.8e6 → 8.8e5 に **10x 縮小**
（梁 DOF 制約緩和の効果）。しかし initial target β=4.737e+04（cutback 前）は
β_outside に関わらず一定で、β_stiff cap 不足で発散。

### 2.3 物理的妥当性 gate（status-381 §5）

`exp_z1c_beta_out_10_betamax_1e6_7s` は frac=0.425 まで進むが、max|u|=1.6×10⁵ mm
（解析解 73.3mm の **2200倍**）で precision gate を完全違反。aggressive
scaling は frac=1.0 では nominal 進行するが解は unphysical。

## 3. 真の根本要因 — initial Courant 制約

7 本撚線で initial target β=4.7e4 を分析:

```
ω_max² ≈ K_stiff_max / M_raw_stiff    （stiff DOF が dominant）
dt_c   = 2/ω_max
target β = dt_sub / (0.9 · dt_c) = 0.05s / (1.06 μs / 1.0) ≈ 4.7e4
```

dt_sub = 0.05s（`t_cycle = max(10·T1, 1.0) = 1.0s` / `n_increments = 20`）が
loading rate を支配。**T1 (~6.7ms) と t_cycle (1s) の比 ~150x が target β の
cubed root** で寄与し、physically unnecessary な loading time が target β を
人為的に押し上げている。

(z1d) で `t_cycle` 下限を `max(10·T1, 0.1·T1)` 程度まで緩和すれば、target β
は 150x に応じて減少し（具体的には dt_sub ∝ t_cycle なので linear）、
β_outside=10 + β_stiff_max=1e3 の組合せで gate 達成可能と推定される。

## 4. 実装変更まとめ

- `xkep_cae/time_integration/strategy.py`:
  - `ExplicitCentralDifferenceProcess.__init__` に `mass_scaling_beta_outside` 追加
  - `_compute_scaled_mass()` で mask に応じて 2 段階適用
  - `set_mass_scaling_beta_outside()` API 追加（KE 保存 outside-only rescale）
  - `_create_time_integration_strategy()` factory に plumb
- `xkep_cae/contact/solver/_explicit_dynamic.py`:
  - dt_c_beam 推定で mask 設定時は β_outside を乗じる（mask=None 時は β）
- `xkep_cae/core/data.py`:
  - `default_strategies()` factory + `ContactFrictionInputData` に field 追加
- `xkep_cae/contact/solver/process.py`:
  - `_mass_scaling_beta_outside = getattr(input_data, ...)` で factory に渡す
- `xkep_cae/numerical_tests/strand_bending_oscillation.py`:
  - `StrandBendingOscillationConfig` field 追加 + 3 経路 plumb-through
- 単体テスト +11（`TestTwoStageMassScaling`）
- 検証スクリプト `work/beam_hysteresis/38_z1c_two_stage_validation.py` 新設（+250 行）

回帰: 全 24 契約検査 OK / **737 passed 5 skipped**（status-384 比 +11）/
`test_helical_3d_hermite` rel_err=2.18×10⁻⁷ 維持 / 7 本 implicit frac=1.0 / ruff pass。

## 5. **MCDD 凍結解除条件 — 条件 (5) 未達**

| 条件 | 状態 |
|------|------|
| (1) Phase E 完了 | ✅ status-357 |
| (2) 19 本 frac=1.0 完走 | △ explicit 系で再評価対象、(z1d) 待ち |
| (3) max\|u_trans\| < L_strand × 10 | ✅ implicit / N/A explicit 発散ケース |
| (4) `KcNormalDirectionStiffness` FD rel_err < 1e-2 | ✅ status-356（2.18×10⁻⁷） |
| **(5) 解の精度 < 10%** | **❌ (z1c) 単独では達成不可、(z1d) 必須** |

## 6. 引継ぎ — 次 status の候補

### 6.1 候補 (z1d) 最優先 — `t_cycle` 下限緩和

`StrandBendingOscillationProcess.process()` 内 `t_cycle = max(10.0 * T1, 1.0)`
の **下限 1.0 秒** を削除し、`t_cycle = 10.0 * T1` または `t_cycle = max(10.0 * T1, 0.1 * T1)`
等の物理ベース下限に変更する。実装規模小（3 経路 1 行修正）、ただし以下を考慮:

- `n_increments` を維持しつつ `dt_sub` を 100x 程度縮小 → `target β` も 100x 縮小
- 結果として β_outside=10 + β_stiff_max=1e3 で gate 達成可能と予測
- ただし dt_sub 縮小は計算時間増加（n_increments 不変なら横ばい、増やせば伸びる）
- implicit 側の挙動も変わる可能性（既存 7 本 frac=1.0 を保つかは要 regression）

### 6.2 副次 — 候補 (z2) Cosserat 梁プロトタイプ

UL を捨てて explicit + 大回転を本質解決。中期的に最もクリーンだが実装コスト
中（~1000 行オーダー）。先に (z1d) で gate 達成できるか確認してから判断。

### 6.3 副次 — 候補 (q3) implicit + AL n>2 復活

19 本 implicit Type D stall の (g2) 系候補（status-376 で却下）を Uzawa update
under-relaxation で再試行。explicit 路線が (z1d) でも行き詰まったときの最終 fallback。

## 7. MCDD 脱法 pattern 回避

- **pattern 1（tol 緩和）**: 精度 gate 0.10 を変更せず、未達と明記
- **pattern 5（既存テスト skip）**: 既存 726 test 全 pass、+11 追加
- **pattern 6（骨格 status）**: API 実装 + 11 単体テスト + 8 ケース実機検証 +
  log 解析（target β 10x 縮小確認）+ 数理的観察（initial target β は
  loading rate 依存）で完結
- **pattern 8（根拠なき主張）**: 7 本実機ログ「target β=8.8e5 vs 8.8e6 で 10x 縮小」を
  実証根拠として提示
- **pattern 10（TODO 先送り）**: (z1c) infrastructure は完了、(z1d) は loading
  rate 縮小という独立スコープなので別 status が適切

## 8. 引継ぎコマンド

```bash
# z1c 検証
uv run --extra dev python work/beam_hysteresis/38_z1c_two_stage_validation.py \
    2>&1 | tee /tmp/z1c_$(date +%s).log

# 回帰
pytest xkep_cae/contact/ xkep_cae/mathematics/ xkep_cae/time_integration/ \
       xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py
python contracts/validate_process_contracts.py
ruff check xkep_cae/ tests/ && ruff format --check xkep_cae/ tests/
```

## 9. 観察 — 開発運用

### 効果的だった点

- **log 解析による設計検証**: aggregate frac/max|u| で gate FAIL でも、
  `[MASS_SCALE] target β=8.8e6 → 8.8e5` の 10x 縮小から API が設計通り
  動作していることを定量確認できた。「frac=0 だから動いていない」という
  早合点を避けられた。
- **2 段階の数学的構造分離**: `mass_scaling_beta`（auto-tune 対象）と
  `mass_scaling_beta_outside`（user-set modest）で役割を分離することで、
  既存 z1b API を破壊せずに z1c を additive に重ねられた（pattern 4 回避）。

### 学び — Courant 制約は loading rate × ω_max の積

Courant 安定条件 `dt_sub ≤ 2/ω_max` は **物理時間 dt_sub** と **数値時間スケール 1/ω_max**
の比を要求する。target β は両者の積に比例し:

- mass scaling（β² 倍化）は ω_max を縮小（数値時間スケールを引き伸ばす）
- loading rate 縮小（t_cycle ↓）は dt_sub を縮小（物理時間スケールを縮める）

(z1c) は前者（縦方向）、(z1d) は後者（横方向）。両方の組合せで初めて
target β を「mass scaling cap で吸収可能な範囲」に収められる。
status-384 の段階で「2 段階 + loading rate」が解と推定されていた理由が
本 status で **定量的に裏付け**られた（target β 4.7e4 のうち、loading rate 寄与が
~150x）。
