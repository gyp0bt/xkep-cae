# status-055: ヒステリシス可視化 + 統計ダッシュボード + 被膜モデル

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-25
**作業者**: Claude Code
**テスト数**: 1270（+36）

## 概要

status-054 の TODO を実行。3件のTODO完了 + 被膜モデル新規実装。

1. **撚線ヒステリシス可視化**: `plot_hysteresis_curve()` 荷重-変位ヒステリシス曲線描画 + `compute_hysteresis_area()` Shoelace公式によるループ面積計算（散逸エネルギー相当）。
2. **接触グラフ統計ダッシュボード**: `plot_statistics_dashboard()` 6パネル統計描画（stick/slip比率、法線反力、連結成分数、累積散逸、接触持続マップ、エッジ/ノード数）。
3. **被膜モデル（CoatingModel）**: 被膜材料特性データクラス + 環状断面特性計算 + 複合断面剛性計算 + 被膜込み接触半径。Phase 4.7 Level 1 の基盤。

## 変更内容

### 1. ヒステリシス可視化 + ループ面積計算

`xkep_cae/contact/graph.py`:

| 関数 | 引数 | 説明 |
|------|------|------|
| `plot_hysteresis_curve()` | load_factors, displacements, dof_index=None | 荷重-変位ヒステリシス曲線描画 |
| `compute_hysteresis_area()` | load_factors, displacements, dof_index=None | Shoelaceによるループ面積（非負値） |

**設計**: `CyclicContactResult` の `load_factors` / `displacements` を直接受け取る。循環import回避のため生データ入力。

### 2. 接触グラフ統計ダッシュボード

`xkep_cae/contact/graph.py`:

| 関数 | 説明 |
|------|------|
| `plot_statistics_dashboard()` | 6パネル統計ダッシュボード |

**パネル構成**:
1. stick/slip 比率の推移
2. 平均・最大法線反力の推移
3. 連結成分数の推移
4. 累積散逸エネルギーの推移
5. 接触持続マップ（横棒グラフ）
6. エッジ数・ノード数の推移

### 3. 被膜モデル（CoatingModel）

`xkep_cae/mesh/twisted_wire.py`:

| 要素 | 説明 |
|------|------|
| `CoatingModel` | 被膜材料特性（thickness, E, nu, mu） + バリデーション |
| `coating_section_properties()` | 環状断面の A/Iy/Iz/J 計算 |
| `coated_beam_section()` | 素線+被膜の複合断面剛性（EA/EI/GJ）計算 |
| `coated_contact_radius()` | 被膜込み接触半径（wire_radius + thickness） |
| `coated_radii()` | メッシュ全要素の被膜込み半径ベクトル |

**設計**:
- 被膜は理想化弾性体（温度・損傷はスコープ外）
- 環状断面として断面特性を厳密計算
- 複合断面剛性ではヤング率比 n = E_coat/E_wire で換算
- ねじり剛性はせん断弾性係数比 G_coat/G_wire で換算
- 同一材料の場合は中実円断面と厳密一致（テスト検証済み）

## ファイル変更

### 変更
- `xkep_cae/contact/graph.py` — plot_hysteresis_curve, compute_hysteresis_area, plot_statistics_dashboard 追加
- `xkep_cae/contact/__init__.py` — 新関数3件のエクスポート追加
- `xkep_cae/mesh/twisted_wire.py` — CoatingModel, coating_section_properties, coated_beam_section, coated_contact_radius, coated_radii 追加
- `xkep_cae/mesh/__init__.py` — 全メッシュモジュールのエクスポート追加

### 新規作成
- `tests/contact/test_hysteresis_viz.py` — ヒステリシス可視化テスト（11テスト: 面積計算7 + 描画4）
- `tests/contact/test_statistics_dashboard.py` — 統計ダッシュボードテスト（6テスト）
- `tests/mesh/test_coating_model.py` — 被膜モデルテスト（19テスト: CoatingModel 7 + 断面特性 5 + 等価剛性 4 + 接触半径 3）

## テスト結果

```
tests/contact/test_hysteresis_viz.py         7 passed, 4 skipped  (新規)
tests/contact/test_statistics_dashboard.py   0 passed, 6 skipped  (新規, matplotlib依存)
tests/mesh/test_coating_model.py            19 passed             (新規)
tests/mesh/test_twisted_wire.py             32 passed             (既存, 影響なし)
全テスト:                                   1270 collected
lint/format:                                ruff check + ruff format パス
```

## 確認事項

- 既存テスト影響なし
- matplotlib描画テストは matplotlib がない環境ではスキップ（既存方針通り）
- `compute_hysteresis_area()` は円の面積（Shoelace）で理論値と 1% 以内の一致を確認
- `coated_beam_section()` で同一材料 → 中実円断面と一致（相対誤差 < 1e-12）
- `mesh/__init__.py` にモジュールエクスポートを追加（既存コードは twisted_wire を直接 import していたため影響なし）

## TODO

### 次ステップ

- [ ] 撚線線（7本撚線＋被膜の7撚線）: 被膜付き7本撚線の接触解析テスト。被膜込み半径・摩擦係数・断面剛性を統合した統合テスト
- [ ] 弧長法＋接触統合テスト → Phase 4.7 で座屈必要時に着手
- [ ] 7本撚りブロック分解ソルバー（Schur補完法 or Uzawa法）
- [ ] 接触グラフ統計の可視化検証（実データでのダッシュボード描画確認）

### 設計懸念

- 被膜モデルの `coated_beam_section()` は異種材料複合断面の等価剛性を計算するが、BeamSection クラスへの直接統合（コンストラクタレベル）はまだ行っていない。現時点では手動で EA/EI/GJ を取得し、梁アセンブリに渡す使い方を想定。
- 被膜の「周方向せん断ばね＋圧縮ばね」モデル化（roadmap Level 1）は別途設計が必要。現在の実装は剛性寄与の基盤のみ。

---
