# ContactGeometry Strategy

[← README](../../../../README.md)

## 概要

接触幾何の検出・ギャップ計算・制約ヤコビアン構築を Strategy として実装する。
`ContactGeometryStrategy` Protocol に従い、3種の接触検出方式を統一的に扱う。

## 具象 Process

| クラス | 概要 |
|--------|------|
| `PointToPointProcess` | 最近接点ペア（PtP）— 小規模問題向け |
| `LineToLineGaussProcess` | Line-to-Line Gauss 積分 — 大規模問題向け |
| `MortarSegmentProcess` | Mortar 法セグメント — Phase 5 で完成予定 |

## Protocol メソッド

- `detect(node_coords, connectivity, radii)`: 接触候補ペアの検出
- `compute_gap(pair, node_coords)`: ギャップ計算
- `update_geometry(pairs, node_coords, *, config)`: 全ペアの幾何情報更新
- `build_constraint_jacobian(pairs, ndof_total, ndof_per_node)`: 制約ヤコビアン G 構築

## ファクトリ

```python
create_contact_geometry_strategy(
    mode="point_to_point",  # "point_to_point" | "line_to_line" | "mortar"
    exclude_same_layer=True,
    n_gauss=2,
    auto_gauss=False,
    line_contact=False,     # True → LineToLineGauss
    use_mortar=False,       # True → MortarSegment
)
```

## 移行元

`xkep_cae_deprecated/process/strategies/contact_geometry.py` → status-179
