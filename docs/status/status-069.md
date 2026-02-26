# status-069: PINN大規模メッシュ検証 + 不規則メッシュPINN効果検証 + ハイブリッドGNN統一

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-26
**作業者**: Claude Code
**テスト数**: 1629（+17: PINN拡張検証 17）

## 概要

status-068のTODO 3, 4を消化。TODO 1, 2（接触ML）は撚線フェーズ次までペンディング。
不規則メッシュ比較テストのGNNアーキテクチャをFC GNN → ハイブリッドGNNに統一（計算量削減）。

1. **PINNの大規模メッシュ（20×20）検証** — 6テスト
2. **不規則メッシュでのPINN学習効果検証** — 7テスト
3. **PINN vs data-only比較** — 4テスト
4. **不規則メッシュ比較: FC GNN → ハイブリッドGNNに変更** — 計算量削減

## 1. PINNの大規模メッシュ（20×20, 441ノード）検証（6テスト）

### 実装内容

`tests/thermal/test_pinn_extended.py` — TestPINNLargeMesh:

- 20×20メッシュ（441ノード）でのK行列生成・形状・対称性・正定値性
- FEM解の物理方程式整合性（K@ΔT = f_shifted）
- PINN学習の収束・物理ロス減少・R²>0達成

### テスト結果

| テスト | 結果 |
|--------|------|
| K行列形状 (441, 441) | PASS |
| K行列対称性・正定値性 | PASS |
| FEM解物理整合性 (残差<1e-3) | PASS |
| PINN学習損失減少 | PASS |
| 物理ロス減少 | PASS |
| R² > 0 達成 | PASS |

### 知見

- 20×20メッシュ（441ノード）でもPINN学習は安定して収束
- FEM行列が441×441のdense行列となるため、PINN学習ループはCPUで約13分（80サンプル×80エポック）
- 大規模メッシュではK行列のサイズ増加がボトルネック。将来的にスパース行列演算への移行が有効

## 2. 不規則メッシュでのPINN検証（7テスト）

### 実装内容

`xkep_cae/thermal/pinn.py` — `generate_pinn_dataset_irregular()` 追加:
- 不規則メッシュ（内部ノード摂動）上でのPINNデータセット生成
- FEM行列K + f_shifted付きグラフデータの不規則メッシュ版

`tests/thermal/test_pinn_extended.py` — TestPINNIrregularMesh:

- 不規則メッシュでのK行列形状・対称性・正定値性
- 不規則メッシュFEM解の物理整合性
- 不規則メッシュ上のPINN学習収束・物理ロス減少

### テスト結果

全7テストPASS。不規則メッシュでもPINNの物理整合性が保たれる。

## 3. PINN vs data-only比較（4テスト）

### 実装内容

`tests/thermal/test_pinn_extended.py` — TestPINNvsDataOnly:

- 正則/不規則メッシュそれぞれでPINNとdata-only学習を比較
- 同一シード・同一モデル構造（ThermalGNN）での公平比較

### 比較結果

| 条件 | PINN R² | data-only R² | PINN効果(ΔR²) |
|------|---------|-------------|---------------|
| 正則メッシュ (5×5) | 0.987 | 0.981 | +0.006 |
| 不規則メッシュ (5×5, perturbation=0.35) | 0.973 | 0.952 | +0.021 |

### 知見

- **不規則メッシュでPINNの効果がより顕著**（ΔR² +0.021 vs +0.006）
- 不規則メッシュではメッシュ構造に依存しない物理ロスがdata augmentation的な効果を発揮
- 正則メッシュではdata-onlyでも十分な精度（R²=0.981）が出るため、PINNの限界効用は小さい
- 小規模データ（100サンプル、5×5メッシュ）でも安定したPINN効果を観測

## 4. 不規則メッシュ比較: FC GNN → ハイブリッドGNNに変更

### 変更内容

`tests/thermal/test_irregular_mesh.py`:
- FC GNN（全結合グラフ: O(N²) エッジ）を **ハイブリッドGNN**（メッシュエッジ + 発熱ノードショートカット）に置換
- 計算量: N*(N-1) エッジ → mesh_edges + heat_node_shortcuts（メッシュの1.5倍程度）

### ハイブリッドGNN比較結果

| 条件 | mesh GNN R² | hybrid GNN R² | hybrid-mesh差 |
|------|------------|--------------|---------------|
| 正則メッシュ | 0.935 | 0.982 | +0.047 |
| 不規則メッシュ | 0.952 | 0.956 | +0.004 |

### 知見

- ハイブリッドGNNは正則メッシュで明確な優位性（+0.047）
- 不規則メッシュでは差が縮小（+0.004）：メッシュGNNの帰納バイアス低下
- FC GNNと同等の精度をO(N²)→O(N)の計算量で達成

## ファイル変更

### 新規
- `tests/thermal/test_pinn_extended.py` — PINN拡張テスト（17テスト）
- `docs/status/status-069.md`

### 変更
- `xkep_cae/thermal/pinn.py` — `generate_pinn_dataset_irregular()` 追加
- `tests/thermal/test_irregular_mesh.py` — FC GNN → ハイブリッドGNNに変更
- `docs/status/status-index.md` — status-069行追加
- `docs/roadmap.md` — PINN大規模検証 + 不規則PINN + ハイブリッドGNN統一の追記
- `README.md` — 現在状態更新

## TODO

- [ ] 接触プリスクリーニングGNN の Step 1 実装 — 撚線フェーズ次までペンディング
- [ ] k_pen推定MLモデルの Step 1 実装 — 撚線フェーズ次までペンディング
- [ ] PINN学習のスパース行列対応（大規模メッシュ高速化）
- [ ] ハイブリッドGNN + PINN の組み合わせ検証

## 確認事項・懸念

- 20×20メッシュのPINN学習はCPU上で約13分かかる。CI環境では `slow` マーカーの付与を検討。
- 不規則メッシュでのPINN効果（ΔR²=+0.021）は有意だが、サンプル数100の小規模実験。大規模での再現性確認が望ましい。
- 接触ML設計仕様（接触プリスクリーニングGNN、k_pen推定ML）は撚線フェーズの次フェーズまでペンディング。
- 2交代制運用の観点: 全結合GNNの比較テストをハイブリッドGNNに置き換えた。FC GNNの実装コードは`gnn_fc.py`に残存するが、テストではハイブリッドグラフのみを使用する方針に変更。

---
