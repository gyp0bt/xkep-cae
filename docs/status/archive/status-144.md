# status-144: 3Dチューブレンダリング統一 — 2D投影線図からの完全移行

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-08
**テスト数**: 2271（変化なし、可視化のみの変更）

---

## 概要

梁要素の可視化を従来の2D投影線図（中心線のみ）から、**3Dチューブサーフェスレンダリング**（円形断面付き）に完全移行。matplotlib mplot3dの`plot_surface`で各梁要素を円筒チューブとして描画し、撚線構造の物理的妥当性が目視で判別できるようにした。

---

## 1. 新規モジュール: `render_beam_3d.py`

### コア機能

| 関数 | 用途 |
|------|------|
| `render_beam_3d()` | 低レベル: node_coords + connectivity + wire_radius から3Dチューブ描画 |
| `render_twisted_wire_3d()` | TwistedWireMesh用ラッパー（推奨API） |
| `render_multiview_3d()` | 複数視角から一括レンダリング |

### 内部実装

- `_make_tube_mesh()`: 2節点間の円筒メッシュ生成（n_circ分割）
- `_make_cap_mesh()`: 端面キャップ（円盤）メッシュ生成
- `_set_equal_aspect_3d()`: 3D軸等アスペクト比設定

### 視角プリセット (VIEW_PRESETS)

| 名前 | elev | azim | 用途 |
|------|------|------|------|
| isometric | 25° | -60° | 標準斜め視角 |
| front_xy | 0° | -90° | 正面（XY平面） |
| side_xz | 0° | 0° | 側面（XZ平面） |
| end_yz | 0° | -180° | 端面（断面が見える） |
| top_down | 90° | -90° | 上面 |
| oblique_30/60 | 20° | -30°/-60° | 斜め |
| bird_eye | 45° | -45° | 鳥瞰 |

---

## 2. ギャラリースクリプト置換

| 旧 | 新 |
|----|-----|
| `scripts/generate_multiview_gallery.py`（削除） | `scripts/generate_3d_beam_gallery.py` |
| 2D投影線図（中心線のみ） | 3Dチューブ表面レンダリング（断面表示） |

出力: 7/19/37本 × 8視角 + マルチビューパネル(2×4)

---

## 3. アニメーション出力の3D対応

`export_animation.py`に新関数追加:

| 関数 | 用途 |
|------|------|
| `export_3d_animation()` | TwistedWireMeshの3DフレームPNG出力 |
| `export_3d_animation_gif()` | 3DチューブGIFアニメーション出力 |

`wire_bending_benchmark.py`の`export_bending_oscillation_gif()`も3D版に更新。
旧2Dビュー名("xy","xz","yz")は自動的に3Dプリセットに変換（後方互換性）。

---

## 4. 検証スクリプト更新

| スクリプト | 変更 |
|-----------|------|
| `verify_7strand_bending_oscillation.py` | `plot_2d_projection()` → `plot_3d_snapshots()` |
| `verify_19strand_bending_oscillation.py` | `plot_side_view()`/`plot_cross_section()` → `_render_3d_on_ax()` |
| `run_bending_oscillation.py` | デフォルトGIF視角を `["isometric", "end_yz"]` に変更 |

---

## 5. `__init__.py`エクスポート追加

```python
from xkep_cae.output import (
    render_beam_3d, render_twisted_wire_3d, render_multiview_3d,
    VIEW_PRESETS, export_3d_animation, export_3d_animation_gif,
)
```

---

## 動作確認

- 7本撚線3Dレンダリング: OK（isometric / end_yz / bird_eye）
- マルチビューパネル(2×4): OK（8視角全て正常描画）
- 既存テスト(`test_export_animation.py`): 28 passed
- ruff check/format: All passed

---

## TODO

- [ ] Mortar接触チャタリング問題の解消（status-143より継続）
- [ ] Phase2揺動の接触活性セット変動対策（status-143より継続）
- [ ] 被膜摩擦μ=0.25の収束達成（status-143より継続）
- [ ] 19本→37本のスケールアップ（status-143より継続）
- [ ] 3Dレンダリングの高速化（37本以上で描画が重い場合はLOD対応検討）
- [ ] PyVistaへの移行検討（matplotlib mplot3dの制約：z-order問題、パフォーマンス）

---

## 設計メモ（次の担当者向け）

### なぜmatplotlib mplot3dを採用したか

- PyVistaやVTKはpyproject.tomlの依存に含まれていない
- matplotlib[plot]は既存の依存関係で利用可能
- mplot3dのplot_surfaceは円筒チューブの描画に十分
- ヘッドレス環境(Agg backend)で問題なく動作

### mplot3dの既知の制約

- **z-order問題**: 重なりの前後関係が不正確になることがある（matplotlib固有）
- **パフォーマンス**: 37本以上・n_circ=16で描画が遅くなる
- **回避策**: n_circ=8-10に削減、antialiased=False
