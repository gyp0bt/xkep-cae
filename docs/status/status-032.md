# status-032: 過渡応答出力 TODO 消化（run_transient_steps / 非線形RF / VTKバイナリ / 要素データ / .inp統合）

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-18

## 概要

status-031 の TODO 5項目を全て実装。
テスト数 653 → 670（+17テスト）。

## 実施内容

### 1. `run_transient_steps()` 関数（ステップ列自動実行・状態引き継ぎ）

- `xkep_cae/output/database.py` に `run_transient_steps()` を追加
- ステップ列を順次実行し、前ステップの終状態（u, v）を次ステップの初期条件に自動引き継ぎ
- Newmark-β / Central Difference ソルバーに対応
- HHT-α パラメータの指定が可能
- 結果は `OutputDatabase` として一括返却
- **6テスト追加**: 1ステップ実行、2ステップ状態引き継ぎ、引数不一致エラー、陽解法、未対応ソルバーエラー、ヒストリ出力付き

### 2. 非線形反力計算（`assemble_internal_force` 対応）

- `build_output_database()` と `_extract_history_nodal()` に `assemble_internal_force` コールバックを追加
- 指定時: RF = f_int(u) + M·a（非線形内力ベース、正確）
- 未指定時: RF = K·u + M·a（従来の線形近似、後方互換）
- **2テスト追加**: 線形 vs 非線形 RF の一致確認、K なしでの非線形 RF 計算

### 3. VTK バイナリ出力モード

- `export_vtk()` に `binary=True` パラメータを追加
- Base64 エンコード形式（VTK XML binary format）で出力
- UInt32 ヘッダ + raw data を base64 エンコード
- `_add_data_array_binary()` 関数と `_VTK_DTYPE_MAP` を追加
- **2テスト追加**: binary 出力の XML 構造検証、base64 デコード可能性検証

### 4. 要素データ出力（応力・歪み等 CellData/PointData）

- `Frame` に `element_data: dict[str, np.ndarray]` フィールドを追加
- `build_output_database()` に `element_data_func` コールバックを追加
  - 変位ベクトル → 要素データ辞書（例: `{"stress_xx": ndarray}`）
  - フレーム生成時に各フレームで呼び出し
- VTK 出力時に CellData セクションとして書き出し
  - スカラー (n_elements,) / ベクトル (n_elements, n_components) 対応
- **3テスト追加**: element_data 設定、VTK CellData 出力検証、多成分要素データ

### 5. Abaqus .inp パーサーとの統合

- `mesh_from_abaqus_inp()` ブリッジ関数を追加
- `read_abaqus_inp()` のパース結果を OutputDatabase 用形式に変換:
  - `node_coords`: (n_nodes, ndim) 節点座標（2D/3D自動判定）
  - `connectivity`: [(vtk_cell_type, node_index_array), ...] 0-based
  - `node_sets`: {name: ndarray of 0-based indices}
- VTK セルタイプマッピング: B21/B31→VTK_LINE, CPS4R→VTK_QUAD, CPE3→VTK_TRIANGLE 等
- **4テスト追加**: 梁メッシュ、四角形メッシュ、3D メッシュ、OutputDatabase→VTK 統合

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/output/__init__.py` | `run_transient_steps`, `mesh_from_abaqus_inp` をエクスポートに追加 |
| `xkep_cae/output/database.py` | `run_transient_steps()`, `mesh_from_abaqus_inp()` 追加、`build_output_database()` に `assemble_internal_force`/`element_data_func` パラメータ追加 |
| `xkep_cae/output/step.py` | `Frame` に `element_data` フィールド追加 |
| `xkep_cae/output/export_vtk.py` | `binary` パラメータ追加、`_add_data_array_binary()` 追加、CellData 出力対応 |
| `tests/test_output.py` | 17テスト追加（55テスト → 計55テスト） |

## テスト数

653 → 670（+17テスト）

## 確認事項・懸念

1. **VTK binary format**: VTK XML の binary 形式は UInt32 ヘッダ + raw data の base64 エンコード。ParaView で正常に読み込めることを確認推奨
2. **element_data_func の呼び出し頻度**: フレームごとに呼ばれるため、計算コストが高い場合はフレーム数を減らすか、キャッシュ機構の追加を検討
3. **mesh_from_abaqus_inp のラベル→インデックス変換**: 1-based ラベルを 0-based インデックスに変換。不連続ラベル番号にも対応

## TODO

（status-031 の TODO を全て消化。新規 TODO なし）

---
