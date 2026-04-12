# status-263: delta_hデフォルト値検討 + three_point_bend E=25回帰発見

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-29
- **ブランチ**: `claude/process-todo-items-ZJCFK`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3+9+4（変更なし）→ **合計574 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. delta_h デフォルト値の検討（status-262 TODO #1）

**結論: huber_delta_h = 0.0（現状維持）**

- 梁-梁（strand_bending）: delta_h=0.025 が最速完走。ただし 0.030 で非完走の非単調性あり
- 問題依存性が高く、グローバルデフォルト設定は時期尚早

### 2. three_point_bend_jig delta_h スイープ（status-262 TODO #2）

`contracts/bench_three_point_bend_delta_h.py` を作成。

#### STA2 教訓

最初のベンチマークは **E=25.0, n_periods=3, max_increments=200** で実行し、frac=0.42 を「壁」と誤報告した。
ユーザーから「E=25MPa で完走していた」と指摘を受け、**ベースライン未確認のまま結論を出した STA2 違反** を認識。

#### 回帰バグの発見

正しい条件（E=25, n_periods=30, max_increments=10000）でテストしたところ **frac=0.0003** で破綻。
status-234 では frac=1.0 で完走していた条件が回帰していた。

#### git bisect による原因特定

| コミット | 内容 | 結果 |
|---------|------|------|
| ef06ba0 (status-234) | SDI排除効果検証 | **GOOD** (frac=1.0 記録) |
| 049ffe9 (cc6f465直前) | Merge PR #200 | frac=0.0128 (100incr中、進行中) |
| **cc6f465** | **LM正則化の実装** | **BAD** (frac=0.0018) |
| 7500fdf (HEAD) | — | frac=0.0003 |

**原因コミット**: `cc6f465 feat: Levenberg-Marquardt正則化の実装 — K_st安全有効化基盤`

#### 回帰メカニズム（暫定分析）

cc6f465 で追加された LM 関連コードは後のコミットで削除済み。しかし以下の変更が残存:
- `consistent_st_tangent=False` が three_point_bend_jig の contact_config に明示的に渡されるようになった
- `consistent_st_tangent=True` でテストしても改善なし（frac=0.0002）

cc6f465 の直前（049ffe9）でも frac=0.0128 と status-234 の frac=1.0 からは劣化済み。
049ffe9 と status-234 の間のコミット（status-235〜238: 梁メッシュ粗化、解析的剛体表面など）が元々の劣化を引き起こし、cc6f465 がさらに悪化させた可能性がある。

**修正は次セッションに引き継ぎ。** 根本原因の特定にはさらなる調査が必要。

### 3. デフォルト設定（E=200e3）での完走確認

| 条件 | frac | incr | cutback | time |
|------|------|------|---------|------|
| E=200e3, n_periods=3（デフォルト） | **1.000** | 555 | 336 | 345s |
| E=25, n_periods=3 | 0.868 | 500 | 303 | 266s |
| E=25, n_periods=30 | **0.0003** | 7 | 10 | 9s |

---

## テスト結果

- 新規テスト: なし
- 既存テスト: 574 passed, 20 skipped, 1 xfailed（回帰なし）
- 契約違反: 0件
- lint: 全合格

---

## 再現手順

```bash
git checkout claude/process-todo-items-ZJCFK
pip install -e .
# E=25 n_periods=30 回帰確認（~9s、frac=0.0003 で即停止）
python contracts/check_rigid_surface_effect.py 2>&1 | tee /tmp/log-regression.log
# E=200e3 デフォルト完走確認（~345s）
python3 -c "
from xkep_cae.numerical_tests.three_point_bend_jig import *
cfg = DynamicThreePointBendContactJigConfig()
r = DynamicThreePointBendContactJigProcess().process(cfg)
print(f'frac={r.solver_result.load_history[-1]:.4f}')
"
# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"
# 契約検証
python contracts/validate_process_contracts.py
```

---

## 次セッションへの引き継ぎ

### 最優先: three_point_bend E=25 n_periods=30 回帰修正

1. **cc6f465 以降の変更を精査**: LMコードは削除済みだが、cc6f465 で導入された他の変更（LinearSolveProcess の `lm_lambda` フィールド追加、three_point_bend_jig の `consistent_st_tangent` パススルーなど）が残存
2. **049ffe9 時点で既に frac=0.0128**: status-234 (frac=1.0) → 049ffe9 (frac=0.0128) の劣化も別途調査が必要。status-235〜238 の梁メッシュ粗化・解析的剛体表面変更が影響している可能性
3. **bisect 再実施**: ef06ba0 → 049ffe9 の範囲でより精密な bisect を推奨

### 残課題（優先度順）

1. **three_point_bend E=25 回帰修正**（上記）
2. **Hermite 非局所 ∂g/∂u 対応**（status-262 から継続）
3. **NR 力収束改善**（status-262 から継続）

### STA2 教訓

- ベンチマーク実施前に **必ず既知の完走条件でベースライン確認** すること
- 「壁」と報告する前に、過去の完走実績と条件を照合すること
- E=25 の三点曲げは以前は完走していたので、「完走しない」は回帰バグ

---

## 懸念・設計メモ

1. **E=25 回帰の深刻度**: E=25 は低剛性テスト条件で、E=200e3（デフォルト）は正常。実用上の影響は限定的だが、以前動いていたものが壊れているのは品質問題
2. **delta_h デフォルト値**: 回帰修正後に改めて E=25 条件で delta_h スイープを実施すべき
