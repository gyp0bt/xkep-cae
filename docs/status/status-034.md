# status-034: FIELD ANIMATION出力 + .inpパーサー拡張（*ELSET / *BOUNDARY / *OUTPUT, FIELD ANIMATION）

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-18

## 概要

Abaqus .inp パーサーに `*ELSET`, `*BOUNDARY`, `*OUTPUT, FIELD ANIMATION` キーワードを追加。
梁要素のx/y/z軸方向からの2Dプロット描画機能（FIELD ANIMATION出力）を新規実装。
テスト数 701 → 741（+40テスト）。

## 実施内容

### 1. .inp パーサー拡張 (`xkep_cae/io/abaqus_inp.py`)

#### `*ELSET` キーワード追加
- 通常形式: 要素ラベルのカンマ区切りリスト
- `GENERATE` 形式: `start, end [, step]`
- 複数行対応
- `AbaqusMesh.elsets` フィールド追加（`dict[str, list[int]]`）
- `get_element_labels_with_elset()` メソッド追加
  - 明示的 `*ELSET` と `*ELEMENT` の `ELSET=` 暗黙定義の両方を検索
  - 大文字小文字区別なし

#### `*BOUNDARY` キーワード追加
- Abaqus形式準拠:
  - `node_label, dof` （単一DOF拘束）
  - `node_label, first_dof, last_dof` （DOF範囲拘束）
  - `node_label, first_dof, last_dof, value` （規定変位）
- `AbaqusBoundary` データクラス新設
- `AbaqusMesh.boundaries` フィールド追加

#### `*OUTPUT, FIELD ANIMATION` キーワード追加（独自拡張）
- Abaqusにはないxkep-cae独自キーワード
- オプション: `DIR=<出力ディレクトリ>` （デフォルト "animation"）
- データ行: ビュー方向のカンマ区切り（例: `xy, xz, yz`）
- `AbaqusFieldAnimation` データクラス新設
- `AbaqusMesh.field_animation` フィールド追加

#### `io/__init__.py` 更新
- 全データクラスを公開APIとしてエクスポート

### 2. FIELD ANIMATION出力 (`xkep_cae/output/export_animation.py`)

新規モジュール。梁要素のアニメーション画像を生成する。

#### `_collect_beam_segments(mesh, node_coords)`
- 梁要素（B21/B22/B31/B32）のセグメントを要素セット別に収集
- 変形後座標対応（2D/3D）
- 非梁要素は無視

#### `render_beam_animation_frame(mesh, view, ...)`
- xy/xz/yz の3ビュー方向に対応
- 要素セットごとに色分け描画（10色パレット）
- 凡例表示
- 全要素が画面に収まるようにマージン付きビュー自動設定
- 等アスペクト比
- matplotlib Figure/Axes を返す

#### `export_field_animation(mesh, output_dir, views, ...)`
- 複数フレーム × 複数ビュー方向の PNG 画像を出力
- 出力ディレクトリの自動作成
- フレームラベル指定可能
- 出力ファイルパスのリストを返す

#### `output/__init__.py` 更新
- `export_field_animation`, `render_beam_animation_frame` を公開APIに追加

### 3. テスト

#### パーサーテスト (`tests/test_abaqus_inp.py`) — +20テスト
- `TestElsetParsing` (8件): 基本/GENERATE/複数行/明示的取得/暗黙的取得/大文字小文字/エラー/複数セット
- `TestBoundaryParsing` (5件): 単一DOF/範囲/規定変位/複数/統合
- `TestFieldAnimationParsing` (6件): デフォルト/カスタムDir/カスタムViews/単一ビュー/未指定/完全モデル
- 既存26テストとの統合テスト1件

#### アニメーション出力テスト (`tests/test_export_animation.py`) — 21テスト（新規ファイル）
- `TestCollectBeamSegments` (6件): 基本/座標/複数セット/変形後/非梁除外/2D座標
- `TestRenderBeamAnimationFrame` (8件): xy/xz/yz/無効ビュー/凡例/タイトル/アスペクト/変形後
- `TestExportFieldAnimation` (5件): デフォルト/単一ビュー/複数フレーム/ラベル/ディレクトリ作成
- `TestIntegrationWithInpParser` (2件): フルパイプライン/変形アニメーション

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/io/abaqus_inp.py` | `*ELSET`, `*BOUNDARY`, `*OUTPUT, FIELD ANIMATION` パーサー追加。`AbaqusBoundary`, `AbaqusFieldAnimation` データクラス新設。`AbaqusMesh` に `elsets`, `boundaries`, `field_animation` フィールド追加 |
| `xkep_cae/io/__init__.py` | 全データクラスのエクスポート追加 |
| `xkep_cae/output/export_animation.py` | **新規**: FIELD ANIMATION出力モジュール |
| `xkep_cae/output/__init__.py` | `export_field_animation`, `render_beam_animation_frame` エクスポート追加 |
| `tests/test_abaqus_inp.py` | +20テスト（ELSET/BOUNDARY/FIELD ANIMATION パーサー） |
| `tests/test_export_animation.py` | **新規**: 21テスト（アニメーション出力） |

## テスト数

701 → 741（+40テスト）

## 対応キーワード一覧（.inp パーサー）

| キーワード | 状態 | 備考 |
|-----------|------|------|
| `*NODE` | 既存 | 2D/3D座標 |
| `*ELEMENT` | 既存 | ELSET=対応、継続行対応 |
| `*NSET` | 既存 | GENERATE対応 |
| `*ELSET` | **新規** | GENERATE対応 |
| `*BEAM SECTION` | 既存 | SECTION/ELSET/MATERIAL |
| `*TRANSVERSE SHEAR STIFFNESS` | 既存 | K11/K22/K12 |
| `*BOUNDARY` | **新規** | 単一DOF/範囲/規定変位 |
| `*OUTPUT, FIELD ANIMATION` | **新規** | xkep-cae独自拡張 |

## 確認事項・懸念

1. **matplotlib依存**: FIELD ANIMATION出力はmatplotlibに依存する。現在はオプショナル依存（importエラーは呼び出し時に発生）。`pyproject.toml` のオプショナル依存に追加を検討
2. **梁要素のみ対応**: 現時点では梁の線図描画と要素セット凡例のみ。連続体要素のメッシュプロットは将来拡張
3. **アニメーション形式**: 現在はPNGフレーム出力。GIF/MP4 への結合は外部ツール（ffmpeg/imagemagick）が必要

## TODO

- [ ] `pyproject.toml` に matplotlib をオプショナル依存として追加
- [ ] GIF/MP4 出力のサポート（ffmpeg連携）
- [ ] 連続体要素のメッシュプロット対応（将来）
- [ ] *MATERIAL, *ELASTIC, *DENSITY キーワードのパーサー追加

---
