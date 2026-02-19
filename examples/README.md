# xkep-cae サンプル入力ファイル

[← README](../README.md)

xkep-cae の `.inp` パーサーで読み込み可能なサンプルファイル集。
Abaqus 互換の入力形式に xkep-cae 独自拡張（`*OUTPUT, FIELD ANIMATION`）を含む。

## ファイル一覧

| ファイル | 説明 | 要素タイプ | 断面 |
|---------|------|-----------|------|
| `cantilever_beam_3d.inp` | 3D片持ち梁（集中荷重） | B31 | 円形 |
| `three_point_bending.inp` | 3点曲げ試験（2D梁） | B21 | 矩形 |
| `portal_frame.inp` | 門型フレーム（柱+梁、3部材） | B31 | 矩形 |
| `l_frame_3d.inp` | L型フレーム（垂直+水平） | B31 | パイプ |
| `elastoplastic_bar.inp` | 弾塑性棒（*PLASTIC テーブル硬化） | B31 | 円形 |

## 使い方

```python
from xkep_cae.io import read_abaqus_inp
from xkep_cae.output import export_field_animation, export_field_animation_gif

# .inp ファイルの読み込み
mesh = read_abaqus_inp("examples/cantilever_beam_3d.inp")

# 初期配置の PNG 出力
export_field_animation(mesh, output_dir="output/png")

# GIF アニメーション出力（変形フレーム付き）
import numpy as np
frames = [...]  # 変形後の節点座標リスト
export_field_animation_gif(
    mesh,
    output_dir="output/gif",
    node_coords_frames=frames,
    frame_labels=["t=0.0", "t=0.5", "t=1.0"],
)
```

## 対応キーワード

| キーワード | 説明 |
|-----------|------|
| `*NODE` | 節点座標 |
| `*ELEMENT` | 要素接続配列（TYPE=, ELSET= オプション） |
| `*NSET` | ノードセット（GENERATE対応） |
| `*ELSET` | 要素セット（GENERATE対応） |
| `*MATERIAL` | 材料定義ブロック開始 |
| `*ELASTIC` | 弾性定数（E, nu） |
| `*DENSITY` | 密度 |
| `*PLASTIC` | 塑性テーブル（降伏応力-塑性ひずみ） |
| `*BEAM SECTION` | 梁断面定義（SECTION=, ELSET=, MATERIAL=） |
| `*TRANSVERSE SHEAR STIFFNESS` | 横せん断剛性 |
| `*BOUNDARY` | 境界条件 |
| `*OUTPUT, FIELD ANIMATION` | アニメーション出力設定（xkep-cae独自） |

---
