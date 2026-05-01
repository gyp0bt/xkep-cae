[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-382: 候補 (p3) damping + (p1) relax API 実装 — UL update_reference 凍結が真の原因と判明

**日付**: 2026-05-01
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12 passed（status-381 比 +12 = damping 5 + relax 2 + 既存 5 を合算）

## 概要

status-381 §7 の最優先 TODO「explicit 解を implicit / 解析解と一致させる」に対し、
仮説 (p1) 動的緩和 + (p3) artificial damping の 2 API を実装し、`35_explicit_accuracy_validation.py`
で効果を実測した。**結果: 両 API 実装は正しく単体テストで動作確認できたが、
単梁 90° 曲げの精度 gate（`|u_explicit − u_implicit|/u_implicit < 0.1`）は未達**。

実測で明らかになった**真の根本原因**: UL 定式化の `update_reference()` が各増分の
dynamic lag を「reference」へ凍結するため、後続の relax phase では
`f_int(u_incr) ≈ 0`（u_incr は新 reference 基準で ≒ 0）となり、構造を
quasi-static 平衡へ駆動できない。

→ **MCDD 凍結解除条件 (5)「解の精度」未達のまま**。次 status は (q1)〜(q3)
（explicit 中の UL update スキップ / 増分内 sub-cycling / implicit 系の AL n>2 復活）
を検討する必要がある。

## 1. 実装

### 1.1 候補 (p3): 質量比例 Rayleigh damping

`xkep_cae/time_integration/strategy.py` `ExplicitCentralDifferenceProcess`:

```python
def __init__(
    self, mass_matrix, *,
    mass_proportional_damping_alpha: float = 0.0,
    ...
):
    if mass_proportional_damping_alpha < 0.0:
        raise ValueError(...)
    self.mass_proportional_damping_alpha = float(mass_proportional_damping_alpha)
    ...

def step(self, u, f_ext, f_int, dt, *, fixed_dofs=None):
    ...
    a_n = self.M_lump_inv * residual
    if self.mass_proportional_damping_alpha > 0.0:
        a_n = a_n - self.mass_proportional_damping_alpha * self.vel
    ...
```

`C = α · M` を等価適用し $a_n -= α · v_{n−1/2}$。M に独立に α が damping 率として
作用するため Courant 安定性および β スケーリングと無関係に減衰を与える。

### 1.2 候補 (p1): BC 完了後の動的緩和フェーズ

`xkep_cae/contact/solver/process.py` の `ContactFrictionProcess.process()` 末尾に追加:

```python
# 主ループ完了後、frac=1.0 到達かつ explicit_relax_steps > 0 のとき
if (_solver_mode == "explicit" and _explicit_relax_steps > 0 and ...):
    _relax_cfg = ExplicitDynamicInput(
        ...,
        courant_check_interval=0,  # 負荷変化なしで K 一定、Courant 検査無効化
        mass_scaling_auto=False,   # β 上方更新も停止
    )
    for _ridx in range(1, _explicit_relax_steps + 1):
        # BC 保持 + MPC 射影 + 接触検出
        # 陽解法 1 step
        _relax_result = _explicit_proc.process(_relax_step_input)
        # 残差ノルムによる早期収束判定
        if _r_rel < _explicit_relax_tol:
            break
```

### 1.3 plumb-through（4 経路 3 field）

| 層 | field |
|----|-------|
| `ExplicitCentralDifferenceProcess.__init__` | `mass_proportional_damping_alpha` |
| `_create_time_integration_strategy` | `mass_proportional_damping_alpha` |
| `default_strategies` | `mass_proportional_damping_alpha` |
| `ContactFrictionInputData` | `explicit_mass_proportional_damping_alpha` / `explicit_relax_steps` / `explicit_relax_tol` |
| `StrandBendingOscillationConfig` | 同 3 field |
| `strand_bending_oscillation.py` 3 経路（free_end_mode / combined / 2-phase）| 同 3 field |

### 1.4 単体テスト追加（+7）

**`test_strategy.py` (+5)**:
- `test_mass_proportional_damping_default_zero`: default α=0、減衰無効
- `test_mass_proportional_damping_negative_raises`: α<0 拒否
- `test_mass_proportional_damping_decays_velocity`: SDoF で v_{n+1} = v_n·(1−α·dt)
- `test_mass_proportional_damping_independent_of_beta`: β=1 と β=10 で v 一致
- `test_factory_passes_damping_alpha`: factory plumb-through

**`test_explicit_dynamic.py` (+2)**:
- `test_explicit_relax_steps_runs`: relax phase 実行確認
- `test_explicit_relax_default_off_unchanged`: default OFF で挙動不変

## 2. 実機検証（`35_explicit_accuracy_validation.py`）

単梁 90° 曲げ（L=100mm, E=130GPa）で 6 ケース実測:

| ケース | frac | max\|u\| [mm] | 解析解誤差 | implicit 誤差 | gate |
|--------|------|--------------|-----------|--------------|------|
| implicit_baseline | 1.000 | 70.45 | 3.90% | 0.00% | **PASS** |
| exp_baseline_no_damp_no_relax | 1.000 | **35.37** | 51.74% | 49.79% | FAIL |
| exp_alpha0.5_no_relax | 1.000 | 29.54 | 59.70% | 58.07% | FAIL |
| exp_no_damp_relax500 | 1.000 | **35.41** | 51.69% | 49.73% | FAIL |
| exp_alpha0.5_relax500 | 1.000 | 29.57 | 59.67% | 58.03% | FAIL |
| exp_ninc200_no_damp | 1.000 | 31.84 | 56.57% | 54.80% | FAIL |

解析解 max\|u\| = 73.30 mm（quarter circle、R = 2L/π ≈ 63.66 mm）。

**重要観察**:
- `exp_no_damp_relax500` が **baseline と本質的に同値**（35.41 vs 35.37）
- `[RELAX] converged at step 1 (||R||/||f||=0.000e+00 < 1.0e-04)` ログで relax phase が
  即座に終了している
- damping 追加（α=0.5）はむしろ精度を悪化（35→29mm）— ローディング中に動きを抑制
- n_increments 増（20→200）も改善せず（35→31mm）— 各増分で UL update により
  lag が凍結されるため、増分数を増やしても効果なし

## 3. 真の根本原因 — UL update_reference 凍結

### 3.1 メカニズム

`xkep_cae/contact/solver/process.py` 主ループでの UL update:

```python
if _ul and hasattr(ul_assembler, "update_reference"):
    _u_incr_ul = state.u - _ul_ref_base
    ul_assembler.update_reference(_u_incr_ul)
    _ul_ref_base[:] = state.u
```

そして `_ul_internal_force_wrapper`:

```python
def _ul_internal_force_wrapper(u_total: np.ndarray) -> np.ndarray:
    u_incr = u_total - _ul_ref_base  # update 後は ≈ 0
    return input_data.callbacks.assemble_internal_force(u_incr)
```

### 3.2 explicit + UL + mass scaling での問題連鎖

1. 主ループ各増分: BC が `Δθ` 増加、explicit dynamics で構造は β² 倍化された
   inertia により応答にラグ δ
2. step 完了後 `update_reference` 呼出: 現状の状態（lag δ 込み）が新 reference に
   凍結される
3. 次増分: u_incr ≈ 0 起点で再度 BC を Δθ 進めるが、構造の **絶対** lag は累積
4. frac=1.0 到達時: 構造の絶対変位は static 解の 50% 程度しかない
5. relax phase: `f_int(u_incr) = f_int(state.u − _ul_ref_base) ≈ f_int(0) = 0`
   → 残差 0 → 構造は動かない → **relax で平衡へ駆動できない**

### 3.3 修正困難性

- **explicit 中に update_reference をスキップ**: CR 梁は大回転で UL update が必須
  （初期 reference のままでは 90° 回転での gimbal lock / 線形化破綻）。skip すると
  内力評価精度が落ち、結果が信用できなくなる。
- **TL formulation 切替**: 既存 UL アセンブラの代替実装が必要、既存テスト全
  影響、リスク大
- **増分内 sub-cycling**: 各 BC 増分で multi-step explicit を実行してから
  update_reference。実装可能だが、現状の主ループ構造大幅改修

## 4. 実装変更まとめ

- `xkep_cae/time_integration/strategy.py`:
  - `ExplicitCentralDifferenceProcess.__init__` に `mass_proportional_damping_alpha`
    引数追加（+13 行 docstring 含む）
  - `step()` で `a_n -= α · v` 適用（+5 行）
  - `_create_time_integration_strategy` plumb-through（+3 行）
- `xkep_cae/core/data.py`:
  - `default_strategies()` plumb（+2 行）
  - `ContactFrictionInputData` 3 field 追加（+13 行 docstring 含む）
- `xkep_cae/contact/solver/process.py`:
  - `_mass_proportional_damping_alpha` 取得 + plumb（+5 行）
  - 主ループ末尾に relax phase 追加（+106 行）
- `xkep_cae/numerical_tests/strand_bending_oscillation.py`:
  - `StrandBendingOscillationConfig` 3 field 追加（+13 行）
  - 3 経路 plumb-through（各 +3 行）
- 単体テスト +7（damping 5 / relax 2）
- 検証スクリプト `work/beam_hysteresis/35_explicit_accuracy_validation.py` 新設
  （+220 行）

回帰: 全 24 契約検査 OK / contact + math + time_integration + strand_bending_osc =
**704 passed 5 skipped**（status-381 比 +7、damping 5 + relax 2）/
`test_helical_3d_hermite` rel_err=2.18×10⁻⁷ 維持 / 7 本 implicit frac=1.0 / ruff pass。

## 5. **MCDD 凍結解除条件 — 条件 (5) 未達**

| 条件 | 状態 |
|------|------|
| (1) Phase E 完了 | ✅ status-357 |
| (2) 19 本 frac=1.0 完走 | ✅ status-379（implicit + explicit、ただし精度別問題） |
| (3) max\|u_trans\| < L_strand × 10 | ✅ status-381（41 mm < 1000 mm） |
| (4) `KcNormalDirectionStiffness` FD rel_err < 1e-2 | ✅ status-356（2.18×10⁻⁷） |
| **(5) 解の精度 < 10%（implicit / 解析解と一致）** | **❌ 本 status: explicit 50% アンダー** |

## 6. 引継ぎ — 次 status の候補

### 6.1 候補 (q1) 最有力 — explicit 中の UL update 周期化

main ループ各増分で update_reference を呼ぶのではなく、**N 増分ごと** に呼ぶ
（例: 10 増分単位）。これにより:
- 各 update 単位内では u_incr が累積し、f_int(u_incr) が正しく非ゼロ
- relax phase の f_int(u_incr) 評価が意味を持つ
- 大回転 update も保たれる（精度低下回避）

**実装案**: `explicit_ul_update_interval: int = 1` field 追加、default 1 で既存
挙動不変、>1 のとき N 増分ごとに update。

### 6.2 候補 (q2) — 増分内 sub-cycling

各 BC 増分で multi-step explicit を実行してから update_reference。dt_sub を
小さくし、dynamics が BC に追従するまで動かしてから reference 更新。

**実装案**: ContactFrictionProcess 主ループ内で `inner_step_count` を導入、
inner で multi-step 後 update_reference。

### 6.3 候補 (q3) — implicit + AL n>2 復活

status-376 で却下された候補 (g2) AL n>2 を、Uzawa update under-relaxation で
再試行。explicit と並行して implicit 系で 19 本 frac=1.0 を狙う。

### 6.4 凍結中 TODO（条件 (5) 達成後）

被膜圧縮モデル / リスタート方式 / ファイバー梁キャリブレーション /
7本撚線ピッチ依存性 / 空間ブロック分離 / 19本 Type D stall。

target: 1000 本撚線（10 万節点）の曲げ揺動計算 6 時間以内。

## 7. MCDD 脱法 pattern 回避

- **pattern 1（tol 緩和）**: 精度 gate 0.10 を変更せず、未達と明記
- **pattern 5（既存テスト skip）**: 既存 test 全 pass、+7 追加
- **pattern 6（骨格 status）**: API 実装 + 単体テスト + 6 ケース実機検証 +
  根本原因解析で完結
- **pattern 8（根拠なき主張）**: `35_*.py` 6 ケース max\|u\| 数値 +
  `||R||=0` ログ + UL wrapper コード参照で実証
- **pattern 10（TODO 先送り）**: API 実装は完了。次の (q1)〜(q3) は **本 status の
  範囲外** であり、UL アーキテクチャに踏み込むため別 status で扱うのが適切

## 8. 引継ぎコマンド

```bash
# 精度検証
uv run --extra dev python work/beam_hysteresis/35_explicit_accuracy_validation.py \
    2>&1 | tee /tmp/accuracy_$(date +%s).log

# 回帰
pytest xkep_cae/contact/ xkep_cae/mathematics/ xkep_cae/time_integration/ \
       xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py
python contracts/validate_process_contracts.py
ruff check xkep_cae/ tests/ && ruff format --check xkep_cae/ tests/
```

## 9. 観察 — 開発運用

### 効果的だった点

- **UL wrapper コードの直接読解**: `_ul_internal_force_wrapper` が `u_incr =
  state.u − _ul_ref_base` を渡すという 2 行を読んだことで、relax phase が
  動かない原因が即座に判明。3D 可視化や追加実機実測ではなく、ソース解析が
  決定打になった。
- **damping API と relax API の分離実装**: 単独でテスト可能なため、damping は
  正しく動作することを確認できた（独立に再利用可能）。

### 学び — UL 定式化の暗黙仮定

UL 定式化は「各増分で NR が完全収束する」前提で組まれている。各収束時点で
update_reference することで、累積回転を回避し CR 梁の線形化を維持する。
explicit dynamics は **収束保証なし** で update_reference が呼ばれるため、
UL アセンブラの暗黙仮定（u_incr が static 平衡）に違反する。

→ explicit + UL の組合せは、UL 側の前提を見直さない限り根本的に誤差が
   累積する。次 status で `explicit_ul_update_interval` 等で UL update 頻度を
   下げ、収束に近づく前に reference を凍結しない方針が有効と推定。
