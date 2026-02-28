# status-087: Phase S2 CPU並列化基盤 + GMRES自動有効化 + Mortar適応ペナルティ

[← README](../../README.md) | [← status-index](status-index.md)

- **日付**: 2026-02-28
- **テスト数**: 1822（fast 1525 / slow 297）（+13: S2並列化テスト13）
- **ブランチ**: claude/execute-status-todos-m3TOm

## 概要

status-086 のTODO3件を消化。Phase S2（CPU並列化）の基盤実装として、GMRES自動有効化、要素行列並列化、Broadphaseベクトル化、ブロック前処理バッチ化を実装。Mortar貫入率改善のための適応ペナルティ増大機構も追加。

## 実施内容

### 1. GMRES 自動有効化（DOF閾値ベース切替）

- `ContactConfig.linear_solver` のデフォルトを `"direct"` → `"auto"` に変更
- `ContactConfig.gmres_dof_threshold: int = 2000` を新規追加
- `_solve_linear_system()`: mode="auto" で DOF >= 閾値のとき自動的に GMRES+ILU を使用
- `_solve_saddle_point_contact()`: DOF >= 閾値のとき自動的にブロック前処理 GMRES を選択
- 全呼び出し箇所に `gmres_dof_threshold` を伝播

**変更ファイル**:
- `xkep_cae/contact/pair.py`: ContactConfig に `gmres_dof_threshold` 追加、`linear_solver` デフォルト変更
- `xkep_cae/contact/solver_ncp.py`: `_solve_linear_system`, `_solve_saddle_point_contact`, メインループ

### 2. 要素行列計算の並列化（ThreadPoolExecutor）

- `assemble_global_stiffness()` に `n_jobs` パラメータ追加（デフォルト: 1=逐次）
- `n_jobs >= 2` かつ要素数 >= 64 のとき `ThreadPoolExecutor` で並列計算
- 要素バッチを n_jobs 個に分割し、各バッチで要素剛性行列を並列計算
- COO への書き込みは逐次（メモリ安全）
- `_assemble_sequential()` / `_assemble_parallel()` に分離

**変更ファイル**: `xkep_cae/assembly.py`

### 3. Broadphase AABB ベクトル化

- AABB 計算をPythonループ → numpy一括処理に変更
- `np.minimum(x0_arr, x1_arr) - expand` で全セグメントを一度に計算
- 既存テスト全通過

**変更ファイル**: `xkep_cae/contact/broadphase.py`

### 4. ブロック前処理 Schur 対角近似バッチ化

- `_solve_saddle_point_gmres()`: n_active 回の ILU ソルブをバッチ行列ソルブに置換
  - `ilu.solve(G_A_bc_dense.T)` で全行を一括ソルブ
  - `np.einsum("ij,ji->i", ...)` で対角要素を一括計算
- `_solve_saddle_point_direct()`: n_active > 4 のとき V 計算を ThreadPoolExecutor で並列化
- フォールバック: バッチソルブが失敗した場合は従来の行ごとソルブ

**変更ファイル**: `xkep_cae/contact/solver_ncp.py`

### 5. Mortar 適応ペナルティ増大

- NCP ソルバーのステップ完了時に Mortar 重み付きギャップをチェック
- 貫入率が `tol_penetration_ratio` を超えた場合、`k_pen *= penalty_growth_factor`（上限: `k_pen_max`）
- 代表半径はアクティブペアの平均 `search_radius` で正規化
- 既存の PtP 適応ペナルティ（solver_hooks.py）と同じパラメータを再利用

**変更ファイル**: `xkep_cae/contact/solver_ncp.py`

### 6. テスト追加

**ファイル**: `tests/test_s2_parallel.py`（13テスト）

| テストクラス | テスト | 検証内容 |
|-------------|-------|---------|
| TestParallelAssembly | test_sequential_matches_parallel_small | 逐次と並列の結果一致 |
| TestParallelAssembly | test_n_jobs_minus_one | n_jobs=-1（全コア）の動作 |
| TestParallelAssembly | test_n_jobs_default_sequential | デフォルト逐次動作 |
| TestParallelAssembly | test_parallel_threshold | 閾値未満は並列化しない |
| TestGMRESAutoEnable | test_default_linear_solver_is_auto | デフォルト設定 |
| TestGMRESAutoEnable | test_gmres_dof_threshold_default | 閾値デフォルト値 |
| TestGMRESAutoEnable | test_solve_linear_system_auto_small | 小規模で直接法 |
| TestGMRESAutoEnable | test_solve_linear_system_auto_large | 大規模で反復法 |
| TestGMRESAutoEnable | test_saddle_point_auto_block_preconditioner | 鞍点系自動ブロック前処理 |
| TestBroadphaseVectorized | test_aabb_vectorized_matches_scalar | AABB一致検証 |
| TestBroadphaseVectorized | test_broadphase_result_unchanged | 候補ペア不変検証 |
| TestMortarAdaptivePenalty | test_mortar_p_n_computation | Mortar法線力計算 |
| TestMortarAdaptivePenalty | test_config_has_adaptive_mortar_params | 設定パラメータ存在 |

## 確認事項・今後の課題

- [ ] 7本撚り曲げ + Mortar の収束改善: 適応ペナルティの効果を再評価（S3以降で実施）
- [ ] 要素並列化の実際のスピードアップ測定（大規模問題: 91本撚り等）
- [ ] Broadphase グリッドビニングの並列化（現状は逐次、大規模問題でのボトルネック）
- [ ] Phase S3: 19本 → 37本 → 61本 → 91本 段階的ベンチマーク

## 開発運用メモ

- **効果的**: statusのTODOベースのタスク管理。各TODOが明確な実装ターゲット。
- **効果的**: ThreadPoolExecutorによる並列化はjoblib依存なしでstdlibのみで実現。CI環境との互換性が高い。
- **懸念**: ThreadPoolExecutor はGIL制約があるため、C拡張（numpy/scipy）を多用する計算でのみ効果的。純Python計算部分は ProcessPoolExecutor への切替検討が必要。
- **懸念**: `linear_solver` デフォルトを "auto" に変更したため、既存ユーザーが "direct" を明示指定していない場合に挙動が変わる可能性。ただし auto は小規模で direct フォールバックするため、実質影響は限定的。

---
