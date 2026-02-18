# status-031: 過渡応答出力インターフェース（Step/Frame/Increment + CSV/JSON/VTK）

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-18

## 概要

動解析の計算結果を体系的に出力するインターフェースを新規実装。
Abaqus に準じた Step / Increment / Frame の階層構造を導入し、
ヒストリ出力（時系列プロファイル）・フィールド出力（空間分布スナップショット）の
2種類の出力要求に対応。CSV・JSON・VTK/VTU（ParaView対応）の3形式でエクスポート可能。
テスト数 615 → 653（+38テスト）。

## 実施内容

### 1. xkep_cae/output/ パッケージ新規作成

```
xkep_cae/output/
├── __init__.py              # 公開API（全エクスポート定義）
├── step.py                  # Step, IncrementResult, Frame, StepResult
├── request.py               # HistoryOutputRequest, FieldOutputRequest
├── initial_conditions.py    # InitialConditions（初期変位・初期速度構築）
├── database.py              # OutputDatabase, build_output_database()
├── export_csv.py            # export_history_csv(), export_frames_csv()
├── export_json.py           # export_json()
└── export_vtk.py            # export_vtk()（.vtu + .pvd）
```

### 2. データモデル（step.py, request.py）

- **Step**: 解析ステップ（name, total_time, dt, history_output, field_output）
  - ステップのシリアル連結対応（前ステップ終状態→次ステップ初期状態）
- **IncrementResult**: NR 1収束分の結果（変位・速度・加速度・収束情報）
- **Frame**: フィールド出力スナップショット（frame_index, time, 変位・速度・加速度）
- **StepResult**: ステップ全体の結果（インクリメント列 + フレーム列 + ヒストリデータ）
- **HistoryOutputRequest**: `output,history,dt=0.01` に相当
  - 対応変数: U, V, A, RF, CF, ALLIE, ALLKE
  - node_sets 指定（nset=refmove 等）
- **FieldOutputRequest**: `output,field,num=15` に相当
  - ステップ内 N 等分のフレーム生成

### 3. 初期条件（initial_conditions.py）

- `InitialConditions` クラス（Abaqus `*INITIAL CONDITIONS` 相当）
  - `add(type="velocity", node_indices=[...], dof=0, value=10.0)`
  - `build_initial_vectors(ndof_total, ndof_per_node) → (u0, v0)`
  - TYPE=VELOCITY, TYPE=DISPLACEMENT 対応

### 4. OutputDatabase（database.py）

- `build_output_database()`: 既存ソルバー結果 → Step/Increment/Frame 階層に変換
  - TransientResult, NonlinearTransientResult, CentralDifferenceResult に対応
  - ヒストリ出力: 指定時間間隔で線形内挿し節点変数・エネルギー変数を記録
  - フィールド出力: 指定フレーム数で等分割スナップショットを記録
  - エネルギー計算: ALLKE = 0.5*v^T*M*v, ALLIE = 0.5*u^T*K*u
  - 反力計算: RF = K*u + M*a（拘束DOFのみ）

### 5. CSV エクスポート（export_csv.py）

- `export_history_csv()`: 各ステップ・各節点集合ごとに CSV ファイル
  - 列: time, U_{node}_{dof}, RF_{node}_{dof}, ALLKE, ALLIE 等
- `export_frames_csv()`: フレーム一覧 CSV + 各フレームの節点データ CSV
  - 座標 + 変位 + 速度 + 加速度

### 6. JSON エクスポート（export_json.py）

- `export_json()`: メタデータ + ステップ結果 + フレーム + ヒストリを構造化 JSON
- NumPy 配列の自動シリアライズ対応

### 7. VTK/VTU エクスポート（export_vtk.py）

- `export_vtk()`: ParaView で開ける形式
  - `.vtu` (VTK XML Unstructured Grid): 各フレームに1ファイル
    - 節点座標 + PointData（U, V, A, U_magnitude）
    - 要素接続（VTK_LINE=梁, VTK_QUAD=Q4, VTK_TRIANGLE=TRI3 等）
  - `.pvd` (ParaView Data Collection): タイムステップインデックス
- 外部 VTK ライブラリ不要（XML 直接生成）

### 8. テスト（38テスト）

`tests/test_output.py`:
- TestStep: 6テスト（基本生成、バリデーション、出力要求付き）
- TestIncrementResult: 1テスト
- TestFrame: 2テスト
- TestHistoryOutputRequest: 3テスト
- TestFieldOutputRequest: 2テスト
- TestInitialConditions: 4テスト（velocity/displacement/multiple/out_of_range）
- TestBuildOutputDatabase: 5テスト（history/field/energy保存/multi-step/不一致エラー）
- TestExportCSV: 2テスト（history CSV, frames CSV）
- TestExportJSON: 1テスト
- TestExportVTK: 3テスト（beam VTK, coords必須, timestep正確性）
- TestStepResult: 1テスト
- TestOutputDatabase: 3テスト（empty/all_frames/properties）
- TestDynamicsIntegration: 2テスト（solve_transient/central_difference との統合）
- TestWorkflow: 3テスト（完全ワークフロー/初期条件/マルチステップ）

### 9. 設計仕様書

`docs/transient-output-design.md` — 設計仕様の詳細

## テスト数

615 → 653（+38テスト）

## 確認事項・懸念

1. **非線形動解析との統合**: `solve_nonlinear_transient()` の結果も同じインターフェースで利用可能（`iterations_per_step` 属性が存在すれば自動的にインクリメント情報に反映）
2. **反力計算の簡易性**: RF の計算は `K*u + M*a` で近似しており、非線形解析では `f_int(u) + M*a` とすべき。将来的に `assemble_internal_force` コールバックを渡せる拡張を検討
3. **VTK 出力形式**: ASCII フォーマットで出力（可読性優先）。大規模モデルではバイナリ（base64エンコード）への切り替えが必要になる可能性あり
4. **ステップ連結**: 現在は `build_output_database()` に全ステップのソルバー結果を渡す方式。将来的に `run_transient_steps()` でステップ列を順次実行し自動的に前ステップの終状態を引き継ぐ関数の追加を検討

## TODO

- [ ] `run_transient_steps()` 関数の追加（ステップ列の自動実行・状態引き継ぎ）
- [ ] 非線形解析での正確な反力計算（`assemble_internal_force` 対応）
- [ ] VTK バイナリ出力モード
- [ ] 要素データ出力（応力・歪み等の CellData/PointData）
- [ ] Abaqus .inp スタイルのテキスト入力パーサーとの統合

---
