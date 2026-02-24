# status-050: 実梁要素接触テスト + 長距離スライドテスト + バリデーションドキュメントTODO追加

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-24
**作業者**: Claude Code
**テスト数**: 1064（+21）

## 概要

status-049 の TODO を消化。実梁要素（Timoshenko 3D / CR梁）での接触テスト、
EI/L³ベースの自動 k_pen 推定、長距離スライド（複数セグメントを跨ぐ）テストを
新規テストファイル `tests/contact/test_real_beam_contact.py` に実装。
また、接触テストフェーズ後の優先 TODO として「バリデーション結果の図式+GIF
バリデーションドキュメント作成」を追加。

## status-049 TODO 消化状況

| TODO | 状態 | 備考 |
|------|------|------|
| 実梁要素（Timoshenko 3D / CR梁）でのマルチセグメント貫入テスト | ✅ 完了 | 15テスト（Timo3D/CR/マルチセグメント/一致性/自動k_pen/摩擦） |
| EA/L ベースの自動 k_pen 推定と適応的ペナルティの統合 | ✅ 完了 | EI/L³ベース推定に変更（曲げ支配の接触問題に適切） |
| 接触付き弧長法との統合テスト | 🔒 凍結 | 設計ドキュメント(`arc_length_contact_design.md`)で Phase 4.7 まで凍結 |
| 接触点移動の長距離スライド（複数セグメントを跨ぐ）テスト | ✅ 完了 | 6テスト（8セグメント梁スライド/境界付近/摩擦/CR梁） |

## 変更内容

### 1. 新規テストファイル: `tests/contact/test_real_beam_contact.py`

ばねモデルではなく実際の FEM 梁要素を使用した接触テスト。21テスト。

#### 梁パラメータ
- アルミニウム: E=70GPa, ν=0.33
- 円形断面 d=20mm (r=10mm)
- 梁長さ L=0.5m
- デフォルト4分割（Timo3D線形 / CR非線形）

#### テスト構成

| クラス | テスト数 | 検証内容 |
|-------|---------|---------|
| `TestTimo3DContactDetection` | 2 | 接触検出/非検出 |
| `TestTimo3DPenetrationBound` | 2 | 貫入制限、ペナルティ依存性 |
| `TestTimo3DMultiSegment` | 1 | 8分割梁での接触 |
| `TestCRBeamContactDetection` | 2 | CR梁接触検出/非検出 |
| `TestCRBeamPenetrationBound` | 1 | CR梁貫入制限 |
| `TestCRBeamMultiSegment` | 1 | CR梁8分割接触 |
| `TestTimo3DVsCRConsistency` | 1 | Timo3D vs CR小変位一致性（20%以内） |
| `TestAutoKPenEstimation` | 2 | EI/L³推定の妥当性・収束検証 |
| `TestRealBeamFrictionContact` | 3 | 摩擦収束（Timo3D/CR）・摩擦貫入非悪化 |
| `TestLongRangeSlide` | 6 | 8セグメントスライド・境界付近・摩擦・CR |

### 2. EI/L³ ベース k_pen 推定

梁の接触問題では曲げ剛性が支配的であり、軸剛性 EA/L では過大評価になる。
曲げ剛性 12EI/L³ を基準とした推定に変更:

```python
k_pen = 10 * 12EI/L³
```

EA/L ベースとの比較（d=20mm, L=125mm, E=70GPa）:
- EA/L: ~1.76×10⁸ N/m（過大）
- 12EI/L³: ~5.26×10⁴ N/m
- 推定 k_pen: ~5.26×10⁵ N/m（妥当）

### 3. 長距離スライドテスト

8セグメント梁A（各セグメント長62.5mm）の上を梁Bがスライドする。
z方向押し下げ + x方向スライド力の複合荷重。

テスト内容:
- `test_slide_contact_detected`: 接触検出
- `test_slide_penetration_bounded`: スライド中の貫入2%以下
- `test_slide_x_displacement_positive`: x方向変位が正
- `test_segment_boundary_crossing`: セグメント境界付近の接触検出
- `test_slide_with_friction`: 摩擦ありスライド収束 + 貫入制限
- `test_cr_beam_slide`: CR梁でのスライド接触

### 4. パラメータチューニングの知見

実梁接触テストの設計で得られた知見:

| 項目 | 推奨値 | 理由 |
|------|--------|------|
| 梁材料 | Al E=70GPa, d=20mm | 鋼(210GPa,80mm)は曲げに対して硬すぎる |
| 初期ギャップ | 0.5mm (search_radius比2.5%) | 過大だとNR初期に接触せず、過小だと初期から接触 |
| k_pen | 1e4〜5e5 | EI/L³ベースが妥当。EA/Lは過大 |
| 荷重 | 200〜500N | 4セグメントL=0.5m梁で〜10mm変位 |
| 荷重ステップ | 20〜30 | 接触過渡現象の安定化に必要 |
| 分割数 | 4以上 | FEM精度 + ブロードフェーズ候補確保 |

## ファイル変更

- `tests/contact/test_real_beam_contact.py` — 新規（21テスト）

## テスト結果

```
tests/contact/test_real_beam_contact.py  21 passed (227s)
全回帰テスト:                           1034 passed, 24 skipped (1042s)
合計:                                    1064テスト
```

## 確認事項

- 既存テスト1034件全パス（24 skipped は matplotlib 等の環境依存テスト）
- lint/format パス（`ruff check` + `ruff format`）
- 弧長法＋接触統合は設計ドキュメントで凍結。Phase 4.7（撚線）で座屈が必要になるまで実装しない

## TODO

### 接触テストフェーズ後の優先TODO（新規追加）

- **テストバリデーション結果の図式+GIFバリデーションドキュメント作成**
  - Phase 1〜Phase C5 の全バリデーション結果を図付きで整理
  - 接触テスト（法線力、ギャップ、Active-set遷移）のGIFアニメーション
  - `docs/verification/` に追加ドキュメント

### 残存TODO

- 弧長法＋接触統合テスト → Phase 4.7 で座屈必要時に着手
- 大規模マルチセグメント（16+セグメント）での性能評価
- 接触テスト結果の系統的ドキュメント化（テスト名・パラメータ・結果の一覧表）

---
