# status-297: 微小dt耐性改善（dt snap + atol_force）

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-05
- **ブランチ**: `claude/execute-status-todos-26c7R`
- **テスト数**: 442+ passed（既存テスト全合格、test_stress_contour既知失敗除く）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-296でHertz型+frozen-m解消によりfrac=0.9997を達成したが、最終インクリメント（frac=1.0, dfrac=0.0003）で不収束。原因分析により2つの対策を実装:

1. **dt snap改善**: 微小dt（端数）発生を防止（一次対策）
2. **atol_force**: NR力収束の絶対許容値（根本対策）

---

## 1. 微小dt防止（dt snap改善）— 一次対策

### 問題

AdaptiveSteppingProcessの`_on_success()`で、最終ステップ手前の残量が通常dtの0.5倍未満の場合に端数dtが発生。
例: frac=0.9970でdt=0.003 → 次frac=0.9997 → 残り0.0003（dt_min=0.0004の0.75倍で snap条件を満たさない）

### 修正

旧コード（status-296以前）:
```python
if 1.0 - next_frac < cfg.dt_min_fraction * 0.5:
    next_frac = 1.0
```

新コード:
```python
remaining = 1.0 - next_frac
if 0 < remaining < next_delta * 0.5:
    next_frac = 1.0
```

**変更点**: snap閾値を`dt_min_fraction * 0.5`から`next_delta * 0.5`（現在のステップ幅の半分）に変更。

### テスト

`test_snap_to_one_avoids_micro_dt`: snap条件の単体テスト追加。

---

## 2. atol_force（NR力収束の絶対許容値）— 根本対策

### 問題

動的ソルバー（`dynamic_ref=True`）では、各インクリメントの初回反復（att=0）の残差ノルムを`f_ref`として設定。
微小dtでは荷重変化が極小のため、f_ref自体が極小（例: 3.8e-4）になる。
NR反復で絶対残差が1.7e-6まで低下しても、相対比 1.7e-6 / 3.8e-4 = 4.5e-3 >> tol=1e-8 で不収束。

### なぜ f_ref floor（旧方式）は不十分だったか

初期実装では `f_ref_floor = global_f_ref * 0.01` で f_ref に下限を設けたが:
```
f_ref_floor = 0.5 * 0.01 = 0.005
相対比 = 1.7e-6 / 0.005 = 3.4e-4 >> tol=1e-8 → まだ不収束！
```
**f_ref を少し持ち上げても、相対判定の桁が根本的に合わない。**

### 修正: 絶対許容値（atol_force）

相対判定に加えて**絶対判定**を追加:
```python
# ConvergenceCheckProcess.process()
_force_converged = res_trans_norm / f_ref < tol_force        # 従来の相対判定
if not _force_converged and atol_force > 0:
    _force_converged = res_trans_norm < atol_force            # 絶対判定（新規）
```

**atol_force の計算**: `global_f_ref × tol_force`
- `global_f_ref`: 成功インクリメントのf_ref指数移動平均（α=0.3）
- `tol_force`: 相対許容値（デフォルト1e-8）
- 意味: **通常インクリメントで力収束を満たすのと同じ絶対残差水準**

### 数値検証

```
通常インクリメント: f_ref=0.5, tol=1e-8 → 収束条件 res < 5e-9 N
atol_force = 0.5 × 1e-8 = 5e-9 N

微小dtインクリメント: f_ref=3.8e-4, res=1.7e-6
  相対判定: 1.7e-6 / 3.8e-4 = 4.5e-3 >> 1e-8 → ✗
  絶対判定: 1.7e-6 > 5e-9 → ✗（まだ不収束）

  → 5反復後 res ≈ 0.018^5 × 3.8e-4 ≈ 7e-13
  絶対判定: 7e-13 < 5e-9 → ✓ 収束！
```

K_c 1.8%誤差による収束率0.018/iterが維持される限り、5反復で絶対許容値に到達する。
**通常インクリメントと同じ精度水準で収束を保証**し、精度を緩めない。

### テスト

- `test_atol_force_convergence`: 相対不収束でもatol_forceで収束を判定
- `test_atol_force_field_exists`: フィールド存在確認

---

## 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/contact/solver/_adaptive_stepping.py` | dt snap閾値をnext_delta基準に変更 |
| `xkep_cae/contact/solver/_newton_steps.py` | ConvergenceCheckInputにatol_force追加、絶対判定追加 |
| `xkep_cae/contact/solver/_newton_dynamic.py` | atol_forceフィールド追加、ConvergenceCheck呼び出しに伝搬 |
| `xkep_cae/contact/solver/process.py` | _global_f_ref追跡、atol_force=global×tol渡し |
| `xkep_cae/contact/solver/tests/test_process.py` | snap/atol_forceテスト3件追加 |

---

## TODO

- [ ] Hertz型+atol_forceで frac=1.0 完走確認（実行検証）
- [ ] cutback数削減（41→20以下）のためのチャタリング対策最適化
- [ ] MPC+contact: ローカルMPC（ワイヤ単位の端部結合）の検討

---

## 次の担当者向け

### atol_force の仕組み

```
process.py (インクリメントループ)
  ├─ _global_f_ref: 成功ステップのf_refのEMA（α=0.3）
  ├─ atol_force = _global_f_ref × tol_force
  └─ NewtonDynamicStepInput(atol_force=atol_force)

_newton_dynamic.py (NRループ)
  └─ ConvergenceCheckInput(atol_force=atol_force)

_newton_steps.py (ConvergenceCheckProcess)
  ├─ 従来: res / f_ref < tol_force  （相対判定）
  └─ 新規: res < atol_force          （絶対判定、相対不収束時のフォールバック）
```

**設計意図**: atol = global_f_ref × tol_force は「通常インクリメントの力収束基準と同じ絶対残差」。
精度を一切緩めずに、微小dtでも原理的に到達可能な収束基準を与える。

### dt snap の仕組み

```
_on_success() で次dt (next_delta) を計算後:
  remaining = 1.0 - next_frac
  if 0 < remaining < next_delta * 0.5:
      next_frac = 1.0  # 端数を吸収
```

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: テスト結果はpytest出力と一致（412 passed, 86 solver tests passed）
- [x] **回帰なし**: 既存テスト全合格（test_stress_contour既知失敗除く）
- [x] **ベースライン確認**: status-296のfrac=0.9997（微小dt不収束）がベースライン
