# status-056: シース（外被）モデル

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-25
**作業者**: Claude Code
**テスト数**: 1311（+41）

## 概要

status-055 の TODO「シース被膜対応」を実装。撚線/撚撚線全体を覆う円筒シース（外被）の幾何モデル。

1. **SheathModel**: シース材料特性データクラス（thickness, E, nu, mu, clearance）+ バリデーション
2. **エンベロープ半径計算**: 最外層素線配置から撚線束の外接円半径を計算
3. **シース断面特性**: 円筒管断面の A/Iy/Iz/J、等価梁剛性 EA/EI/GJ
4. **最外層素線特定**: 最外層番号・素線ID・節点インデックスの取得
5. **径方向ギャップ計算**: 最外層素線節点とシース内面の距離（接触ペア生成の前段）

## 変更内容

### SheathModel データクラス

`xkep_cae/mesh/twisted_wire.py`:

| 要素 | 説明 |
|------|------|
| `SheathModel` | シース材料特性（thickness, E, nu, mu, clearance）+ バリデーション |
| `compute_envelope_radius()` | 撚線束外接円半径（最外層配置半径 + 素線有効半径） |
| `sheath_inner_radius()` | シース内径（エンベロープ + クリアランス） |
| `sheath_section_properties()` | 円筒管断面 A/Iy/Iz/J + 内外径 |
| `sheath_equivalent_stiffness()` | 等価梁剛性 EA/EIy/EIz/GJ |
| `outermost_layer()` | 最外層の層番号 |
| `outermost_strand_ids()` | 最外層素線IDリスト |
| `outermost_strand_node_indices()` | 最外層素線の全節点インデックス |
| `sheath_radial_gap()` | 最外層節点とシース内面の径方向ギャップ |

### 設計

- **CoatingModel との違い**: CoatingModel は素線個別被膜（環状層として梁要素に帰着）。SheathModel は束全体への拘束（外圧）として作用し、個別素線の梁要素に帰着できない。
- **エンベロープ半径**: `max(lay_radius) + wire_radius + coating.thickness` で決定。3本/7本/19本撚りに対応。
- **クリアランス**: シース内面と最外層素線外表面の初期隙間。0=密着、正値=遊び。
- **径方向ギャップ**: ヘリカル配置の各節点の径方向位置 `√(x²+y²)` を使い、`r_inner - (r_nodal + r_effective)` で計算。正=非接触、負=貫入。

## ファイル変更

### 変更
- `xkep_cae/mesh/twisted_wire.py` — SheathModel, compute_envelope_radius, sheath_inner_radius, sheath_section_properties, sheath_equivalent_stiffness, outermost_layer, outermost_strand_ids, outermost_strand_node_indices, sheath_radial_gap 追加
- `xkep_cae/mesh/__init__.py` — 新要素9件のエクスポート追加

### 新規作成
- `tests/mesh/test_sheath_model.py` — シースモデルテスト（41テスト）

## テスト結果

```
tests/mesh/test_sheath_model.py           41 passed  (新規)
tests/mesh/test_coating_model.py          19 passed  (既存, 影響なし)
tests/mesh/test_twisted_wire.py           32 passed  (既存, 影響なし)
全テスト:                                 1311 collected
lint/format:                              ruff check + ruff format パス
```

### テスト内訳（41テスト）

| テストクラス | テスト数 | 内容 |
|------------|---------|------|
| TestSheathModel | 9 | 基本生成, デフォルト値, クリアランス, せん断弾性率, バリデーション5件 |
| TestComputeEnvelopeRadius | 4 | 7本/3本/19本撚り, 被膜付き |
| TestSheathInnerRadius | 3 | クリアランスなし/あり, 被膜込み |
| TestSheathSectionProperties | 6 | 断面積, 慣性モーメント, ねじり定数, Iy=Iz, 内外径, 薄肉近似 |
| TestSheathEquivalentStiffness | 5 | 正値, EA=E*A, GJ=G*J, EIy=EIz, 厚肉ほど高剛性 |
| TestOutermostLayer | 6 | 最外層番号(7/3/19), 最外層ID(7/3/19) |
| TestOutermostStrandNodes | 3 | 節点数, 全外層, 有効インデックス |
| TestSheathRadialGap | 5 | クリアランス0, 正クリアランス, 形状, 被膜込み, 3本撚り |

## 確認事項

- 既存テスト影響なし（twisted_wire 32テスト + coating 19テスト 全パス）
- エンベロープ半径は 3本/7本/19本撚りで理論値と一致
- 断面特性は円筒管の厳密解と一致（相対誤差 < 1e-12）
- 薄肉近似（2πrt, πr³t）で 1% 以内の一致
- 径方向ギャップは clearance=0 で ≈ 0（ヘリカル形状の端部変動のみ）

## TODO

### 次ステップ

- [ ] 撚線線（7本撚線＋被膜の7撚線）: 被膜付き7本撚線の接触解析テスト。被膜込み半径・摩擦係数・断面剛性を統合した統合テスト
- [ ] シース-素線接触ペア生成: 最外層素線とシース内面の接触ペアを ContactManager に統合
- [ ] シース面圧拘束: 外圧として最外層素線に作用するペナルティベース径方向拘束
- [ ] 弧長法＋接触統合テスト → Phase 4.7 で座屈必要時に着手
- [ ] 7本撚りブロック分解ソルバー（Schur補完法 or Uzawa法）
- [ ] 接触グラフ統計の可視化検証（実データでのダッシュボード描画確認）

### 設計懸念

- 現在のシースモデルは幾何・断面特性の計算基盤のみ。シースを接触解析に組み込むには、シース内面を「仮想梁」または「剛体面」として ContactManager に接触ペアを追加する必要がある。これは SheathModel 自体の設計ではなく、接触フレームワーク側の拡張が必要。
- `sheath_radial_gap()` は初期配置の径方向ギャップのみ計算。変形後のギャップ更新は接触ソルバーの幾何更新ステップで行う。
- 被膜モデルの `coated_beam_section()` は BeamSection クラスへの直接統合はまだ行っていない（手動で EA/EI/GJ を取得する運用）。

---
