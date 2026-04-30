[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-379: 陽的中央差分 Phase 3 — 候補 (h1) mass scaling auto-tune で **19 本 frac=1.0 完走**

**日付**: 2026-04-30
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+28+10+11 passed（status-378 比 +11、新規 mass scaling unit 8 + auto-tune unit 3）

## 概要

status-378 Phase 2 で 7 本撚線 explicit smoke の **Courant 比 3×10⁵** が確定し、
mass scaling (Belytschko §6.4.2) または subcycling が必須要件と判明。本 status で
**候補 (h1) 集中質量スケーリング** を `ExplicitCentralDifferenceProcess` に追加し、
`ExplicitDynamicProcess` の Courant 監視に **β auto-tune** を統合。

**19 本撚線 90° 曲げ実機で**: `solver_mode="explicit"` + `mass_scaling_auto=True`
（max β=10³）で **frac=1.0000 完走**、E_kin/E_strain=**1.15%**（gate 5% 未満）、
269 incr / 31 cb / 131s。status-376 implicit + AL n=2 の 0.5746 を大幅に上回り、
**MCDD Phase E 凍結解除条件「19 本 frac=1.0 完走」を達成**。

## 1. 実装

### 1.1 `ExplicitCentralDifferenceProcess` mass scaling

`xkep_cae/time_integration/strategy.py`:

| 追加 API | 役割 |
|---------|------|
| `mass_scaling_beta: float = 1.0` (constructor) | 初期 β（>= 1.0、<1 で `ValueError`） |
| `set_mass_scaling_beta(beta)` | 上方更新（単調増加のみ、既存値以下で no-op） |

`__init__` で `_M_lump_raw = self._lump_mass(M, lumping)` を保存し、
`self.M_lump = β² · _M_lump_raw` / `M_lump_inv` を再計算。`set_mass_scaling_beta()`
は raw を係数倍するだけなので O(ndof)。

数理: $M_\mathrm{scaled} = \beta^2 \cdot M_\mathrm{raw}$ →
$\omega = \omega_\mathrm{raw}/\beta$ → $\Delta t_c = \beta \cdot \Delta t_{c,\mathrm{raw}}$。

### 1.2 `ExplicitDynamicProcess` β auto-tune

`xkep_cae/contact/solver/_explicit_dynamic.py` に `ExplicitDynamicInput` 3 field:

| field | default |
|-------|---------|
| `mass_scaling_auto: bool` | `False` |
| `mass_scaling_max_beta: float` | `100.0` |
| `kinetic_energy_budget_ratio: float` | `0.0` |

Courant 監視で違反検知時の挙動:

```python
required_extra = dt_sub / dt_safe   # > 1.0
target_beta = current_beta * required_extra
capped_beta = min(target_beta, mass_scaling_max_beta)
if capped_beta > current_beta * 1.05:   # 5% 成長閾値（数値ノイズ抑制）
    time_strategy.set_mass_scaling_beta(capped_beta)
if target_beta > mass_scaling_max_beta:
    return DynamicStepOutput(..., diverged=True, failure_reason="courant_cap")
```

`auto=False`（既定）では従来通り `failure_reason="courant"` で cutback 要求。

### 1.3 plumb-through

`ContactFrictionInputData` / `StrandBendingOscillationConfig` に 4 field 追加:
`explicit_mass_scaling_beta` / `explicit_mass_scaling_auto` /
`explicit_mass_scaling_max_beta` / `explicit_kinetic_energy_budget_ratio`。
3 経路（曲げ / 揺動 / 旧2フェーズ）で `ContactFrictionInputData` 構築箇所すべてに
伝搬。`default_strategies()` + `_create_time_integration_strategy()` に
`mass_scaling_beta` 引数追加。

### 1.4 設計仕様

- `xkep_cae/time_integration/docs/time_integration_explicit.md`
  §質量スケーリング（数理 / 準静的近似ガード / auto-tune / API、+58 行）追加
- `xkep_cae/contact/solver/docs/explicit_dynamic.md`
  §既知のスケーリング障壁 を「status-378 実測 + status-379 解決」に更新、
  §auto-tune（候補 (h1)）追加

## 2. テスト（+11 件）

### 2.1 `time_integration/tests/test_strategy.py`（+8 件）

`TestExplicitCentralDifferenceMassScaling`:

| テスト | 検証内容 |
|--------|----------|
| `test_default_beta_one_unchanged` | β=1.0 で M_lump==raw |
| `test_beta_squared_scales_lumped_mass` | β=10 → M_lump=100·raw、M_lump_inv 整合 |
| `test_beta_below_one_raises` | β<1 で ValueError |
| `test_beta_scales_critical_dt` | β=4 → dt_c=4·dt_c_raw（同 K に対し） |
| `test_set_mass_scaling_beta_increases` | 上方更新で M_lump 再計算 |
| `test_set_mass_scaling_beta_monotone` | 既存値以下では no-op |
| `test_set_mass_scaling_beta_invalid_raises` | β<1 で ValueError |
| `test_factory_passes_mass_scaling_beta` | `_create_time_integration_strategy` 伝搬 |

### 2.2 `contact/solver/tests/test_explicit_dynamic.py`（+3 件）

`TestMassScalingAutoTune`:

| テスト | 検証内容 |
|--------|----------|
| `test_auto_tune_disabled_returns_courant_failure` | `auto=False` で `failure_reason="courant"` |
| `test_auto_tune_scales_beta_within_cap` | `auto=True` で β 上方更新 + converged=True |
| `test_auto_tune_cap_reached_returns_courant_cap` | cap 到達で `failure_reason="courant_cap"` |

## 3. 検証

### 3.1 Default OFF 回帰（gate 必達）

| 項目 | 結果 |
|------|------|
| `python contracts/validate_process_contracts.py` | 全 24 検査 OK |
| `pytest xkep_cae/contact/ xkep_cae/mathematics/ xkep_cae/time_integration/ xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py` | **691 passed, 5 skipped**（status-378 比 +11） |
| `test_helical_3d_hermite` | rel_err=2.18e-07 維持 |
| 7 本撚線 implicit | frac=1.0（既存テスト） |
| `ruff check` / `ruff format --check` | OK |

### 3.2 7 本撚線 explicit + auto-tune smoke

`bending_curvature=0.0005`, `max_increments=5`, `mass_scaling_max_beta=1e8`:

```
[MASS_SCALE] Incr 1 β: 1.000e+00 → 4.737e+04 (target 4.737e+04, cap 1.000e+08)
[smoke] frac=0.2500, n_incr=5, n_cb=0, elapsed=0.27s, converged=True
最終 KE=3.76e-11, SE=3.31e-01 → ratio=1.1e-10
```

β=4.7×10⁴ に 1 増分で収束、E_kin/E_strain ≪ 1%。auto-tune の妥当性を確認。

### 3.3 19 本撚線 90° 曲げ実機（**MCDD 凍結解除 gate**）

`work/beam_hysteresis/29_mass_scaling_19strand.py auto`、`max_beta=1e3`:

```
[MASS_SCALE] Incr 10 β: 1.000e+00 → 1.000e+03 (target 4.521e+04, cap 1.000e+03)
[CUTBACK:courant_cap] frac 0.4625 ... (cb #1)
... (cap 到達後は dt 縮小カットバックで対応)
β=1.0 auto=True: frac=1.0000, incr=269, cb=31, elapsed=131.07s, converged=True
  E_kin (final) = 1.85e+10
  E_strain (final) = 1.61e+12
  E_kin / E_strain = 1.148e-02
  Gate frac=1.0:           PASS
  Gate E_kin/E_strain<5%:  PASS
  Total:                   PASS
```

**Gate 両条件 PASS**:
- frac=1.0000 完走（status-376 implicit AL n=2 の 0.5746 を **+74%** 上回る）
- E_kin/E_strain = 1.15%（5% 上限の 23%）

cap=10³ で β を制限（必要 target 4.5×10⁴ より小さい）した結果、
途中から `failure_reason="courant_cap"` で adaptive dt cutback に移行し、
mass scaling + dt 縮小の組合せで Courant 比 3×10⁵ を吸収した。

## 4. MCDD 脱法 pattern 回避

- pattern 1（tol 緩和）: 単体 11 件は機械精度ベース（β² scaling 数値完全一致 / cap 越え判定）
- pattern 5（既存 skip）: status-378 既存 680 全 pass、`test_helical_3d_hermite` 機械精度継続
- pattern 6（骨格 status）: 単体 + auto-tune 配線 + 設計仕様 + **19 本実機 frac=1.0 完走**

## 5. MCDD 凍結解除条件達成

CLAUDE.md「凍結解除条件: Phase E 完了 + 19本 frac=1.0 完走 + `KcNormalDirectionStiffness`
FD rel_err < 1e-2」のうち **19 本 frac=1.0 完走** を達成。残: Phase E（C18-C24 完了済）と
`KcNormalDirectionStiffness` FD は status-356 で rel_err=2.18e-07 機械精度継続中。

凍結中の派生 TODO（被膜圧縮モデル / リスタート方式 / ファイバー梁キャリブレーション /
7本撚線ピッチ依存性 / 空間ブロック分離）は次 status 以降で順次解凍可能。

## 6. 引継ぎ（次 status へ）

### 6.1 数値解の物理的妥当性検証（最優先、次 status）

19 本 frac=1.0 は完走したが、解の物理的妥当性（変形形状 / 接触力分布 /
ヘリカル構造保持）は陰解法解との比較で確認が必要。次 status で:

- `Strand3DContourProcess` で 19 本 explicit 解の 3D レンダリング
- 7 本撚線 implicit と explicit 結果の数値比較（変位 / 反力 / 接触力）
- 高速化: 現状 131s（19 本）の cb=31 を mass scaling cap 引き上げで削減可能

### 6.2 候補 (h2) dt subcycling / (h3) selective explicit（保留）

(h1) で gate 達成のため、subcycling / selective は当面保留。陽解法 only で
frac=1.0 を確保した上で、性能要件（1000 本撚線 6 時間以内）に対する
費用対効果を別途評価する。

### 6.3 副次 TODO（凍結解除）

- 被膜圧縮モデル整理 / 7 本撚線ピッチ依存性 / ファイバー梁キャリブレーション

## 7. 引継ぎコマンド

```bash
# 19 本 mass scaling 検証再現
python work/beam_hysteresis/29_mass_scaling_19strand.py auto 2>&1 | \
    tee /tmp/ms_19strand_auto_$(date +%s).log

# 単体回帰
pytest xkep_cae/time_integration/tests/test_strategy.py \
       xkep_cae/contact/solver/tests/test_explicit_dynamic.py -v

# 全回帰
pytest xkep_cae/contact/ xkep_cae/mathematics/ xkep_cae/time_integration/ \
       xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py
python contracts/validate_process_contracts.py
ruff check xkep_cae/ tests/ && ruff format --check xkep_cae/ tests/
```
