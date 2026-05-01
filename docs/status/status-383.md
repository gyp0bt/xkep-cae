[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-383: 候補 (q1) explicit_ul_update_interval 実装 — 4 ケース掃引で却下、UL 凍結が真因と再確証

**日付**: 2026-05-01
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12+5 passed（status-382 比 +5 = q1 単体テスト 5）

## 概要

status-382 §6.1 最有力候補 (q1)「explicit 中の UL update_reference 周期化」を
実装。`explicit_ul_update_interval: int = 1` field を追加し、`solver_mode="explicit"`
かつ interval > 1 のとき UL `update_reference()` を **N 増分ごと** に呼び出す。
default 1 で既存挙動完全不変。

**実機検証 4 ケース掃引（単梁 90° カンチレバー曲げ、L=100mm）で全 FAIL**:

| ケース | frac | max\|u\| [mm] | 解析解誤差 | gate |
|--------|------|--------------|-----------|------|
| implicit_baseline | 1.000 | 70.45 | 3.90% | PASS |
| q1_interval1_baseline | 1.000 | 29.57 | 59.67% | FAIL |
| q1_interval5 | — | DIVERGED (NaN) | — | FAIL |
| q1_interval10 | 1.000 | 6.21×10⁶ | 8.5×10⁶% | FAIL |
| q1_interval20 | 1.000 | 5.16×10²¹ | 7.0×10²¹% | FAIL |

**結論**: 候補 (q1) **却下**。interval > 1 で UL 線形化が累積 u_incr に追従できず
explicit dynamics が爆発的発散する。status-382 §3 の「UL update_reference 凍結が
真の根本原因」の解析を実装で**再確証**する形となった。CR 梁の UL 定式化は
「各増分で u_incr が小さい」前提で組まれており、N 増分蓄積はこの前提を破壊する。

→ **MCDD 凍結解除条件 (5)「解の精度」未達のまま**。次 status は (q2) 増分内
sub-cycling、または (q3) implicit + AL n>2 復活へ移行。

## 1. 実装

### 1.1 候補 (q1): UL update_reference 周期化

`xkep_cae/contact/solver/process.py`:

```python
# status-383 候補 (q1): explicit 中の UL update_reference 周期化.
_explicit_ul_update_interval = max(
    1, int(getattr(input_data, "explicit_ul_update_interval", 1))
)

# ... 主ループ内 update_reference 呼出箇所:
_next_incr = _incr_count + 1
_do_ul_update = (
    _solver_mode != "explicit"
    or _explicit_ul_update_interval <= 1
    or (_next_incr % _explicit_ul_update_interval == 0)
)
if _ul and hasattr(ul_assembler, "update_reference") and _do_ul_update:
    _u_incr_ul = state.u - _ul_ref_base
    ul_assembler.update_reference(_u_incr_ul)
    _ul_ref_base[:] = state.u
    # MPC T 再構築（status-283）も update 時のみ実行
    ...
```

ゲート条件式（implicit には影響しない）:

| mode | interval | next_incr | do_update |
|------|----------|-----------|-----------|
| implicit | 任意 | 任意 | True（毎増分） |
| explicit | ≤ 1 | 任意 | True（既存挙動） |
| explicit | N | k | (k % N == 0) |

### 1.2 plumb-through（3 経路 1 field）

| 層 | field |
|----|-------|
| `ContactFrictionInputData` | `explicit_ul_update_interval: int = 1` |
| `StrandBendingOscillationConfig` | `explicit_ul_update_interval: int = 1` |
| `strand_bending_oscillation.py` 3 経路（free_end / combined / 2-phase）| `explicit_ul_update_interval=cfg.explicit_ul_update_interval` |

### 1.3 単体テスト追加（+5、TestExplicitULUpdateInterval クラス）

**`test_explicit_dynamic.py`**:

- `test_default_interval_one_calls_every_increment`: default `interval=1` で
  4 増分すべてに update_reference 呼出（4 回）
- `test_interval_two_calls_every_other_increment`: `interval=2` で
  2nd / 4th のみ呼出（2 回）
- `test_interval_larger_than_increments_skips_all`: `interval=100` で
  4 増分中 0 回呼出
- `test_interval_zero_treated_as_one`: `interval=0` は `max(1, 0) = 1`
  として扱う（毎増分呼出、4 回）
- `test_implicit_mode_gate_short_circuits`: gate 式の `_solver_mode != "explicit"`
  short-circuit を直接検証（implicit では interval=100 でも常に True）

`_MockULAssembler` クラスで `update_reference` 呼出回数を計測。
dt_max_fraction=1/n_target で各 step を強制分割し、確実な増分カウントを確保。

## 2. 実機検証（`36_explicit_ul_interval_validation.py`）

単梁 90° 曲げ（L=100mm, E=130GPa）で 5 ケース実測:

| ケース | interval | frac | max\|u\| [mm] | 状態 |
|--------|---------|------|--------------|------|
| implicit_baseline | — | 1.000 | 70.45 | 解析解誤差 3.90% |
| q1_interval1_baseline | 1 | 1.000 | 29.57 | status-382 と同値 |
| q1_interval5 | 5 | — | NaN | relax phase で発散 |
| q1_interval10 | 10 | 1.000 | 6.21×10⁶ | RELAX max\|v\|=10²⁰ |
| q1_interval20 | 20 | 1.000 | 5.16×10²¹ | E_kin/E_strain=10³⁸ |

**重要観察**:

- `interval=1` は status-382 結果と一致（29.57mm）— default 挙動完全保持確認
- `interval=5` で relax phase が `||R||/||f||=1.0×10²⁵`（25 step 時点）から
  `nan` へ発散、cKDTree が finite check で reject
- `interval=10`/`20` は relax phase 内で max\|v\| が 10²⁰ オーダーに発散後、
  解析自体は完走するが解は物理的妥当性を完全に失う（max\|u\| > 10⁶ mm = 1 km〜
  10²¹ mm）
- ローディング中の cutback も増加（interval=5 で 14 回）、内部力のラグ補正が
  きかず Courant 違反が頻発

## 3. (q1) 却下の数理的根拠

### 3.1 CR 梁 UL 定式化の暗黙仮定

UL 定式化の接線剛性 `K_T(u_incr)` は **u_incr が微小** という前提で第 1 次
線形化される（Crisfield §17.4）。各増分の収束時に reference を更新することで
u_incr ≪ 1 を保ち、Newton 反復の二次収束を維持する。

### 3.2 explicit + interval > 1 での問題連鎖

1. 増分 1〜N-1: 主ループで explicit step が累積 u_incr を生成（θ ≈ Δθ × N/N_total）
2. 増分 N で初めて update_reference: u_incr が大幅に蓄積した状態で参照を更新
3. しかし update 直前の **N-1 ステップ** では `K_T(u_incr_累積)` が線形化精度を
   失っているため、加速度評価が不正確
4. mass scaling auto-tune が誤った K_max を観測 → β cap 到達 → cutback 連鎖
5. relax phase 進入時には系が高エネルギー過渡応答中、damping も追従不能
6. 数値発散

### 3.3 status-382 §3 解析との整合性

status-382 は「update_reference の毎呼出が dynamic lag を凍結し f_int(u_incr)
≈ 0 になる」と解析。本 status は逆方向「update_reference を間引くと u_incr が
線形化レンジを超え K_T が誤計算される」を実測。すなわち UL 定式化は陽解法と
**根本的に非整合**であり、interval を間引いても貯めても破綻する。

→ explicit + UL 組合せの本質的限界。次候補は UL 定式化に手を入れない (q2)
sub-cycling か、explicit を諦めて implicit + AL を強化する (q3)。

## 4. 実装変更まとめ

- `xkep_cae/core/data.py`:
  - `ContactFrictionInputData.explicit_ul_update_interval: int = 1` 追加（+7 行 docstring 含む）
- `xkep_cae/numerical_tests/strand_bending_oscillation.py`:
  - `StrandBendingOscillationConfig.explicit_ul_update_interval: int = 1` 追加（+7 行）
  - 3 経路 plumb-through（各 +1 行）
- `xkep_cae/contact/solver/process.py`:
  - `_explicit_ul_update_interval` 取得 + `max(1, int(...))` clamping（+8 行 docstring 含む）
  - 主ループ内 update_reference 呼出箇所に gate 条件追加（+9 行）
- 単体テスト +5（TestExplicitULUpdateInterval クラス、`_MockULAssembler` 含む +130 行）
- 検証スクリプト `work/beam_hysteresis/36_explicit_ul_interval_validation.py` 新設（+200 行）

回帰: 全 24 契約検査 OK / contact + math + time_integration + strand_bending_osc =
**709 passed 5 skipped**（status-382 比 +5、q1 単体テスト 5）/
`test_helical_3d_hermite` rel_err=2.18×10⁻⁷ 維持 / 7 本 implicit frac=1.0 / ruff pass。

## 5. **MCDD 凍結解除条件 — 条件 (5) 未達**

| 条件 | 状態 |
|------|------|
| (1) Phase E 完了 | ✅ status-357 |
| (2) 19 本 frac=1.0 完走 | ✅ status-379（implicit + explicit） |
| (3) max\|u_trans\| < L_strand × 10 | ❌ q1 interval > 1 で発散（解析発散時はこの gate も失敗） |
| (4) `KcNormalDirectionStiffness` FD rel_err < 1e-2 | ✅ status-356（2.18×10⁻⁷） |
| **(5) 解の精度 < 10%（implicit / 解析解と一致）** | **❌ 本 status: 全 q1 ケースが FAIL** |

## 6. 引継ぎ — 次 status の候補

### 6.1 候補 (q2) 最有力 — 増分内 sub-cycling

各 BC 増分で multi-step explicit を実行してから update_reference を 1 回呼出。
dt_sub を Courant 安定性下に留め、N サブステップで dynamics を BC に追従させて
から reference を更新。これにより:

- 1 BC 増分内で u_incr が累積し、内部力が意味を持つ
- update_reference は 1 BC 増分終了時に 1 回（既存と同じ頻度）
- UL 線形化レンジは 1 BC 増分で収まるため破綻しない

**実装案**: ContactFrictionProcess 主ループ内で `inner_step_count` 導入、
explicit step を inner で multi-step 実行 → 1 inner cycle 完了で update_reference 1 回。

### 6.2 候補 (q3) — implicit + AL n>2 復活

status-376 で却下された候補 (g2) AL n>2 を、Uzawa update under-relaxation で
再試行。explicit 路線が UL 由来の本質的限界に達したため、implicit 系で
19 本 frac=1.0 を狙う。

### 6.3 候補 (h5) — bending 段階処方

19 本 implicit で `bending_curvature` を 0.005 → 0.010 → 0.015 と段階的に増加させ、
各段階で NR を完全収束させてからチェックポイントから次段階を開始する。
Newton 良条件再開で 19 本 frac=1.0 を implicit のまま達成できる可能性。

## 7. MCDD 脱法 pattern 回避

- **pattern 1（tol 緩和）**: 精度 gate 0.10 を変更せず、未達と明記
- **pattern 5（既存テスト skip）**: 既存 704 test 全 pass、+5 追加
- **pattern 6（骨格 status）**: API 実装 + 5 単体テスト + 5 ケース実機検証 +
  数理的却下根拠（CR 梁線形化レンジ）で完結
- **pattern 8（根拠なき主張）**: `36_*.py` 5 ケース max\|u\| / RELAX 発散ログ
  / cKDTree finite check 例外を実証根拠として提示
- **pattern 10（TODO 先送り）**: q1 結論「却下」を明記、次候補 q2/q3/h5 に移譲

## 8. 引継ぎコマンド

```bash
# q1 検証（5 ケース掃引）
uv run --extra dev python work/beam_hysteresis/36_explicit_ul_interval_validation.py \
    2>&1 | tee /tmp/q1_validation_$(date +%s).log

# 回帰
pytest xkep_cae/contact/ xkep_cae/mathematics/ xkep_cae/time_integration/ \
       xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py
python contracts/validate_process_contracts.py
ruff check xkep_cae/ tests/ && ruff format --check xkep_cae/ tests/
```

## 9. 観察 — 開発運用

### 効果的だった点

- **ゲート条件式の単体検証**: 完全な ContactFrictionProcess を走らせずに
  `_solver_mode != "explicit" or interval <= 1 or (next_incr % interval == 0)`
  を直接 assert することで、implicit 経路への副作用なしを契約レベルで保証
- **mock UL アセンブラ**: update_reference 呼出回数のみを計測する最小モック
  により、UL 定式化全体を再実装せずに gate 動作を検証可能
- **default-preserving design**: `interval=1` で既存挙動完全不変、回帰テスト
  709 passed が変動なし

### 学び — UL 定式化の二重制約

UL 定式化は (a) 各増分で u_incr 微小、かつ (b) 各収束時点で reference 更新の
両方を満たす必要がある。explicit dynamics は (a) を保証できないため:

- update_reference を毎呼出 → status-382: f_int(u_incr) ≈ 0 で平衡駆動不能
- update_reference を間引く → status-383: K_T(u_incr) 線形化崩壊で発散

→ UL + explicit の組合せは原理的に成立しない。explicit 路線継続には UL を
   捨てて (q2) sub-cycling で「各 BC 増分は通常 UL 動作」とするか、
   そもそも implicit に戻る (q3) しかない。
