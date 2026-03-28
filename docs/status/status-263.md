# status-263: three_point_bend_jig delta_h スイープ検証 + デフォルト値検討

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-28
- **ブランチ**: `claude/process-todo-items-ZJCFK`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3+9+4（変更なし）→ **合計574 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. three_point_bend_jig huber_delta_h スイープ（status-262 TODO #2）

`contracts/bench_three_point_bend_delta_h.py` を作成し、剛体円柱-梁接触（DynamicThreePointBendContactJigProcess）で delta_h をスイープ。

#### スケーリング考察

| 問題 | wire_radius | auto δ | auto delta_h推定 |
|------|------------|--------|-----------------|
| strand_bending | 0.5mm | 1000/r=2000 | ≈0.015 |
| three_point_bend | 8.5mm | 5000/r≈588 | ≈0.39 |

strand_bending の最適値 delta_h/r=0.05 → three_point_bend では delta_h≈0.425 が期待値。

#### 第1ラウンド結果（delta_h=0.0, 0.005, 0.010）

小さい delta_h（0.005-0.010）は全て同一結果（frac=0.42, max_increments=500）。
→ ワイヤ径スケールに対して微小すぎて効果なし。

#### 第2ラウンド結果（広範囲スイープ, max_increments=200）

| delta_h | frac | incr | cutback | NR_avg | NR_max | time |
|---------|------|------|---------|--------|--------|------|
| 0.0 (auto δ=588) | **0.418** | 200 | 125 | 3.5 | 30 | 219s |
| 0.025 | 0.418 | 200 | 124 | 3.6 | 30 | 196s |
| 0.050 | 0.418 | 200 | 125 | 3.5 | 30 | 206s |
| 0.100 | 0.288 | 200 | 117 | 2.5 | 29 | 186s |
| 0.200 | 0.281 | 200 | 114 | 2.7 | 30 | 248s |
| 0.300 | 0.301 | 200 | 113 | 2.6 | 30 | 228s |
| 0.425 | 0.411 | 200 | 115 | 3.4 | 28 | 255s |
| 0.500 | 0.406 | 200 | 117 | 3.4 | 30 | 222s |
| 0.750 | 0.380 | 200 | 131 | 3.0 | 29 | 254s |
| 1.000 | 0.002 | 7 | 8 | 4.7 | 18 | 18s |

#### 分析

- **delta_h=0〜0.050**: ほぼ同一結果（frac=0.42）。小さい delta_h は効果なし
- **delta_h=0.1〜0.3**: 悪化（frac=0.28-0.30）。strand_bending と同じ非単調性パターン
- **delta_h=0.425**: 回復（frac=0.41）。auto delta_h≈0.39 に近い値
- **delta_h=1.0**: 完全崩壊（frac=0.002）。遷移幅が大きすぎて接触力が不正確に
- **最良値は auto (δ=5000/r)** で frac=0.42。huber_delta_h 直接指定による改善なし

### 2. delta_h デフォルト値の検討結果（status-262 TODO #1）

#### 結論: **グローバルデフォルトの設定は時期尚早**

| 項目 | 判断 | 根拠 |
|------|------|------|
| グローバルデフォルト変更 | **見送り** | 問題依存性が高い |
| 問題固有デフォルト | **見送り** | 非単調性があり安全な万能値がない |
| 現行 huber_delta_h=0.0 | **維持** | auto smoothing_delta パスが最善 |

#### 根拠

1. **問題依存性**: 梁-梁（delta_h=0.025最適）と剛体-梁（delta_h直接指定は改善なし）でスケールが完全に異なる
2. **非単調性**: 両問題で delta_h の中間値域に「谷」が存在。安全マージンが取れない
3. **auto smoothing_delta の優位性**: 剛体-梁では auto (5000/r) が最良結果。直接指定で超える値は見つからず
4. **ユーザー確認**: 7本撚線は貫入なし（status-262 で確認済み、ユーザーからも追認）

### 3. ベンチマークスクリプト

| ファイル | 内容 |
|---------|------|
| `contracts/bench_three_point_bend_delta_h.py` | 剛体円柱-梁接触 delta_h スイープ |

---

## テスト結果

- 新規テスト: なし（ベンチマーク結果の記録のみ）
- 既存テスト: 574 passed, 20 skipped, 1 xfailed（回帰なし）
- 契約違反: 0件
- 条例違反: 0件
- lint: 全合格

---

## 再現手順

```bash
git checkout claude/process-todo-items-ZJCFK
pip install -e .
# three_point_bend delta_h スイープ（~30分）
python contracts/bench_three_point_bend_delta_h.py 2>&1 | tee /tmp/log-tpb-delta-h.log
# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"
# 契約検証
python contracts/validate_process_contracts.py
# lint
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## 次セッションへの引き継ぎ

### 残課題（優先度順）

1. **auto smoothing_delta の問題固有最適化**: three_point_bend_jig の auto δ=5000/r は frac=0.42 で壁。delta_h 直接指定では改善不可。NR力収束改善（tol_force の壁）が本質的課題
2. **Hermite 非局所 ∂g/∂u 対応**: 4ノードペア外の DOF 結合（status-258 から継続）。大規模設計タスク
3. **NR力収束改善**: 中盤後〜終盤で 25 反復が力収束に不足、disp 収束で抜ける状態

### 設計メモ

- delta_h 直接指定は梁-梁（strand_bending）専用の最適化。剛体-梁では効果なし
- 剛体-梁の frac=0.42 の壁は delta_h ではなく NR 力収束自体の問題（disp 収束で抜ける）
- delta_h の非単調性は両問題で共通するが、谷の位置はスケール依存（梁-梁: 0.030, 剛体-梁: 0.1-0.3）
- `huber_delta_h` API は問題固有の手動チューニング用として有用。デフォルト値は 0.0（auto smoothing_delta 使用）が正しい設計

---

## 懸念・設計メモ

1. **delta_h 問題依存性が確認された**: 梁-梁と剛体-梁で最適レンジが完全に異なる。今後の新しい問題タイプ（例: 61本撚線）でも再スイープが必要
2. **three_point_bend_jig の frac=0.42 壁**: 200 incr/125 cutback のうち、力収束に失敗して disp 収束で抜けるインクリメントが多い。これは delta_h ではなく NR ソルバー自体の改善が必要
