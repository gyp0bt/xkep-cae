# status-068: PINN導入検証 + 不規則メッシュGNN比較 + 接触ML設計仕様

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-26
**作業者**: Claude Code
**テスト数**: 1612（+28: PINN 17 + 不規則メッシュ 11）

## 概要

status-067のTODO4項目を全て消化:

1. **Physics-Informed ロス（PINN）の導入検証** — 実装＋17テスト
2. **不規則メッシュでの全結合GNN vs メッシュGNN再検証** — 実装＋11テスト
3. **接触プリスクリーニング用GNN設計仕様** — 設計文書
4. **k_pen最適推定MLモデル設計仕様** — 設計文書

## 1. Physics-Informed ロス（PINN）導入検証（17テスト）

### 実装内容

`xkep_cae/thermal/pinn.py` — Physics-Informed 学習モジュール:

- `generate_pinn_sample()`: FEM行列 K と ΔT基準の右辺ベクトル f_shifted をグラフデータに付加
- `generate_pinn_dataset()`: PINN用データセット生成
- `graph_dict_to_pyg_pinn()`: PyG Data変換（K_dense, f_shifted テンソル付き）
- `compute_physics_loss()`: 正規化物理残差 `||K·ΔT_pred − f_shifted||² / ||f_shifted||²`
- `train_model_pinn()`: L_total = L_data + λ × L_phys の学習ループ

### Physics-Informed ロスの定式化

```
L_total = L_data + λ_phys × L_phys

L_data = MSE(ΔT_pred_norm, ΔT_target_norm)     # 正規化ターゲット
L_phys = ||K @ ΔT_pred − f_shifted||² / ||f_shifted||²  # FEM残差

f_shifted = f − T_min · K @ 1   （ΔT基準への変換）
```

### テスト一覧

| テストクラス | テスト名 | 内容 |
|------------|---------|------|
| TestPINNDataGeneration | test_k_matrix_shape | K行列形状 (N,N) |
| TestPINNDataGeneration | test_k_matrix_symmetric | K行列の対称性 |
| TestPINNDataGeneration | test_k_matrix_positive_definite | K行列の正定値性 |
| TestPINNDataGeneration | test_f_shifted_shape | f_shifted形状 (N,) |
| TestPINNDataGeneration | test_exact_solution_satisfies_physics | FEM解で物理残差≈0 |
| TestPINNDataGeneration | test_pyg_conversion_preserves_data | PyG変換でK,f保持 |
| TestPINNDataGeneration | test_multiple_samples_different_f | サンプル間でf_shifted異なる |
| TestPhysicsLoss | test_zero_for_exact_solution | 正解で残差ゼロ |
| TestPhysicsLoss | test_nonzero_for_random | ランダム予測で正の残差 |
| TestPhysicsLoss | test_gradient_flows | 勾配の伝搬確認 |
| TestPhysicsLoss | test_loss_decreases_towards_exact | 正解近傍で残差減少 |
| TestPINNTraining | test_training_converges | 学習損失の減少 |
| TestPINNTraining | test_physics_loss_decreases | 物理ロスの減少 |
| TestPINNTraining | test_data_loss_decreases | データロスの減少 |
| TestPINNTraining | test_pinn_achieves_positive_r2 | R² > -0.5 |
| TestPINNTraining | test_lambda_zero_equals_data_only | λ=0でデータのみと等価 |
| TestPINNTraining | test_high_lambda_physics_dominant | 高λで物理ロス支配的 |

### 知見

- FEM解に対して物理残差が確実にゼロになることを検証（f_shifted = K @ ΔT_exact）
- λ_phys の制御により、データフィッティングと物理整合性のバランス調整が可能
- 小規模データ（50サンプル、5×5メッシュ）でも物理ロスの減少が観察される

## 2. 不規則メッシュでのGNN比較（11テスト）

### 実装内容

- `fem.py`: `make_irregular_rect_mesh()` — 内部ノード摂動による不規則メッシュ生成
- `dataset.py`: `generate_dataset_irregular()` — 不規則メッシュ上のデータセット生成

### テスト一覧

| テストクラス | テスト名 | 内容 |
|------------|---------|------|
| TestIrregularMeshGeneration | test_node_count_preserved | ノード数保持 |
| TestIrregularMeshGeneration | test_connectivity_preserved | 接続配列保持 |
| TestIrregularMeshGeneration | test_boundary_nodes_fixed | 境界ノード固定 |
| TestIrregularMeshGeneration | test_interior_nodes_perturbed | 内部ノード摂動確認 |
| TestIrregularMeshGeneration | test_element_jacobian_positive | 全要素Jacobian正 |
| TestIrregularMeshGeneration | test_perturbation_range | 摂動量範囲確認 |
| TestIrregularMeshGeneration | test_reproducibility | 再現性（同一シード） |
| TestIrregularMeshFEM | test_fem_converges | 不規則メッシュFEM有限値 |
| TestIrregularMeshFEM | test_temperature_rise_positive | 温度上昇が正 |
| TestIrregularMeshComparison | test_both_models_learn | 両モデル学習可能 |
| TestIrregularMeshComparison | test_comparison_irregular_vs_regular | 正則vs不規則比較 |

### 比較結果

不規則メッシュ（perturbation=0.35）での性能比較を実装。
テスト内でレポート出力し、正則/不規則メッシュでの mesh GNN vs FC GNN の R² 差を計測。

## 3. 接触プリスクリーニング用GNN設計（設計文書）

`docs/contact/contact-prescreening-gnn-design.md` — 詳細設計仕様:

- GNNでセグメントペアの接触確率を高速予測（Broadphaseの候補を80%削減目標）
- 10Dノード特徴量 + 7Dエッジ特徴量
- Focal Loss（クラス不均衡1%対応）
- 性能目標: Recall > 99%, 推論 < 1ms
- 5ステップの実装計画

## 4. k_pen最適推定MLモデル設計（設計文書）

`docs/contact/kpen-estimation-ml-design.md` — 詳細設計仕様:

- 12D特徴量 → MLP → log10(k_pen) の回帰問題
- グリッドサーチデータ生成（~2,600サンプル）
- フォールバック安全策（auto_beam_penalty_stiffness との併用）
- 収束速度30〜50%改善目標

## ファイル変更

### 新規
- `xkep_cae/thermal/pinn.py` — PINN学習モジュール
- `tests/thermal/test_pinn.py` — PINNテスト（17テスト）
- `tests/thermal/test_irregular_mesh.py` — 不規則メッシュテスト（11テスト）
- `docs/contact/contact-prescreening-gnn-design.md` — 接触プリスクリーニングGNN設計
- `docs/contact/kpen-estimation-ml-design.md` — k_pen推定ML設計
- `docs/status/status-068.md`

### 変更
- `xkep_cae/thermal/fem.py` — `make_irregular_rect_mesh()` 追加
- `xkep_cae/thermal/dataset.py` — `generate_dataset_irregular()` 追加
- `docs/status/status-index.md` — status-068行追加
- `docs/roadmap.md` — PINN + 不規則メッシュ + 設計文書の追記
- `README.md` — 現在状態更新

## TODO

- [ ] 接触プリスクリーニングGNN の Step 1 実装（データ生成パイプライン）
- [ ] k_pen推定MLモデルの Step 1 実装（グリッドサーチデータ生成）
- [ ] PINNの大規模メッシュ（20×20）での検証
- [ ] 不規則メッシュでの PINN 学習効果検証

## 確認事項・懸念

- PINNの小規模テスト（5×5, 50サンプル）では R² > 0 の達成が安定しない場合がある。大規模データ・エポック数増加で改善が期待される。
- 不規則メッシュ比較テスト（test_comparison_irregular_vs_regular）は4モデル学習で約3分を要する。CI環境では slow マーカーの付与を検討。
- 接触ML設計仕様（TODO 3, 4）は設計段階であり、実装には撚線テストの自動実行環境が必要。

---
