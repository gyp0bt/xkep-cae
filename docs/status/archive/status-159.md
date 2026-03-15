# status-159: Phase 5 完了 — ContactForce/Geometry Strategy 注入 + Phase 6 concrete/ 具象プロセス

[← README](../../README.md) | [← status-index](status-index.md) | [← status-158](status-158.md)

**日付**: 2026-03-11
**テスト数**: 2477（変更なし — 回帰テスト全通過）

## 概要

Process Architecture Phase 5 完了: ContactForceStrategy と ContactGeometryStrategy を solver_ncp.py に注入し、5軸 Strategy 合成を達成。
Phase 6 として concrete/ に PreProcess（StrandMesh, ContactSetup）、PostProcess（Export, BeamRender）を追加。

## 実施内容

### 1. Phase 5: ContactForceStrategy の solver_ncp.py 注入（status-158 TODO #1）

- `strategies` から `contact_force` / `contact_geometry` を取得するコードを追加
- NCP Newton ループ内の PtP 接触力計算を `_contact_force_strategy.evaluate()` に委譲
  - Mortar / Line-to-line パスは固有データ構造のため直接呼び出しを維持
  - PtP パスが Strategy 経由に統一
- `create_contact_force_strategy()` ファクトリで自動構築（strategies=None 時の後方互換）

### 2. Phase 5: ContactGeometryStrategy の solver_ncp.py 注入（status-158 TODO #2）

- NCP Newton ループ内の `_build_constraint_jacobian()` を `_contact_geometry_strategy.build_constraint_jacobian()` に委譲
- `manager.update_geometry()` は ContactManager 内部の状態管理（freeze_active_set 等）と密結合のため、現行維持
- `create_contact_geometry_strategy()` ファクトリで自動構築

### 3. default_strategies() の 5軸 Strategy 完全生成

- `contact_force` と `contact_geometry` フィールドを追加
- `default_strategies()` で全5軸 Strategy を生成するように更新
- 新パラメータ: `line_contact`, `use_mortar`, `n_gauss`, `contact_compliance`, `smoothing_delta`

### 4. Phase 6: concrete/ 具象プロセス実装（status-158 TODO #3）

- `pre_mesh.py`: StrandMeshProcess — TwistedWireMesh の PreProcess ラッパー
- `pre_contact.py`: ContactSetupProcess — ContactManager 初期化 + broadphase
- `post_export.py`: ExportProcess — CSV/JSON エクスポート
- `post_render.py`: BeamRenderProcess — 3D チューブレンダリング
- `concrete/__init__.py`: 全プロセスを公開

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver_ncp.py` | ContactForce/Geometry Strategy 注入、3分岐の PtP ケース委譲 |
| `xkep_cae/process/data.py` | default_strategies() に contact_force/contact_geometry 追加 |
| `xkep_cae/process/concrete/__init__.py` | 新プロセス4件の公開 |
| `xkep_cae/process/concrete/pre_mesh.py` | StrandMeshProcess 新規 |
| `xkep_cae/process/concrete/pre_contact.py` | ContactSetupProcess 新規 |
| `xkep_cae/process/concrete/post_export.py` | ExportProcess 新規 |
| `xkep_cae/process/concrete/post_render.py` | BeamRenderProcess 新規 |

## 検証

- `pytest xkep_cae/process/` — 202テスト全通過
- `pytest tests/contact/test_solver_ncp_s3.py` — 31テスト全通過（NCP 回帰なし）
- `pytest tests/contact/` — 568テスト全通過（26スキップ、5 xfail）
- `ruff check && ruff format --check` — 全通過

## 設計上の判断

### manager.update_geometry() の Strategy 委譲について

- ContactGeometryStrategy の `update_geometry()` は pair.state を直接操作するロジックを完全移植済み
- しかし solver_ncp.py は `manager.update_geometry(coords_def, freeze_active_set=True)` を使用
- `freeze_active_set` は Manager 内部の active set 管理と密結合
- **判断**: build_constraint_jacobian のみ Strategy 化し、update_geometry は Manager に残す
- 完全な委譲は Manager のリファクタリング（active set 管理の分離）が前提

### Mortar / Line-to-line の ContactForceStrategy 統合

- Mortar パスは固有データ構造（mortar_nodes, lam_mortar, G_mortar）を使用
- Line-to-line パスは assembly の compute_contact_force() を使用
- 現時点では PtP ケースのみ Strategy 化。Mortar/L2L は今後の拡張

## TODO（次セッション）

- [ ] 37本 NCP 収束テストの高速化（CI タイムアウト >600s 問題）
- [ ] Mortar / Line-to-line の ContactForceStrategy 完全統合
- [ ] manager.update_geometry() の Strategy 完全委譲（active set 分離が前提）
- [ ] Phase 5 設計仕様: バッチプロセス StrandBendingBatchProcess の実装
- [ ] concrete/ の 1:1 テスト追加（test_pre_mesh.py, test_pre_contact.py, test_post_export.py, test_post_render.py）

## 懸念事項・運用メモ

- CI 環境で `gh` CLI が使用不可（ローカルプロキシ経由のためホスト認識されず）。CI ログの取得は別手段が必要。
