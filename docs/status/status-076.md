# status-076: TODO消化 — PINNスパース対応 + adaptive omega定量評価 + ML基盤

[← README](../../README.md) | [← status-index](status-index.md)

- **日付**: 2026-02-27
- **テスト数**: 1732（fast: +30、slow: +7）
- **ブランチ**: claude/execute-status-todos-O2GA1

## 概要

status-075 の TODO を消化。PINN学習のスパース行列対応、adaptive omega 効果定量評価テスト、ヒステリシスループ面積計測テスト、接触プリスクリーニングGNNデータ生成パイプライン、k_pen推定ML用特徴量抽出ユーティリティを実装。

## 実施内容

### 1. PINN学習スパース行列対応（+8テスト）

**問題**: `generate_pinn_sample()` で K 行列を `K.toarray()` で密行列化。大規模メッシュ（20×20 = 441ノード）でメモリ効率が悪い。

**解決策**:
- `generate_pinn_sample()`: K の COO スパース表現（indices, values, size）も格納
- `graph_dict_to_pyg_pinn()`: `use_sparse=True` オプション追加。PyTorch スパース COO テンソルを生成
- `compute_physics_loss()`: K が sparse か dense かを自動判定し、対応する行列ベクトル積を使用
- `train_model_pinn()`: `K_sparse` が存在する場合は自動的にスパース演算を使用

**テスト（+8テスト: fast 7 + slow 1）**:
- スパースデータ生成、dense/sparse一致検証、PyG変換、物理ロス一致、勾配伝搬、学習収束

**変更ファイル**:
- `xkep_cae/thermal/pinn.py` — スパース対応（COO格納、sparse判定、sparse.mm使用）
- `tests/thermal/test_pinn.py` — TestSparsePINN クラス追加

### 2. adaptive omega 効果定量評価テスト（+4テスト）

n_outer_max=3 と n_outer_max=5 での収束性比較、成長率バリエーション検証。

| テスト | 内容 |
|--------|------|
| 3本撚り outer3 vs outer5 | 両方収束 + 最終変位差 < 20% |
| 7本撚り outer3 | adaptive omega + n_outer_max=3 収束確認 |
| 7本撚り outer5 | adaptive omega + n_outer_max=5 収束確認 |
| growth rate 1.5/2.0/3.0 | 3パターン全て収束確認 |

**変更ファイル**:
- `tests/contact/test_twisted_wire_contact.py` — TestAdaptiveOmegaQuantitative クラス追加

### 3. 7本撚りサイクリック荷重ヒステリシスループ面積計測（+1テスト）

既存の `compute_hysteresis_area()` を利用して、loading/unloading の力-変位履歴からループ面積を自動計測するテストを追加。

**変更ファイル**:
- `tests/contact/test_twisted_wire_contact.py` — TestHysteresisLoopArea クラス追加

### 4. 接触プリスクリーニング GNN Step 1 — データ生成パイプライン（+17テスト）

設計仕様 `docs/contact/contact-prescreening-gnn-design.md` の Step 1 を実装。

**新規モジュール**: `xkep_cae/contact/prescreening_data.py`

| 関数 | 説明 |
|------|------|
| `extract_segments()` | メッシュからセグメント端点を抽出（変位対応） |
| `compute_segment_features()` | ノード特徴量 (10D) 計算 |
| `compute_edge_features()` | エッジ特徴量 (7D) 計算 |
| `label_contact_pairs()` | closest_point_segments で接触ラベル付与 |
| `generate_prescreening_sample()` | 統合サンプル生成（broadphase → ラベル） |

**テスト（+17テスト: 全 fast）**:
- セグメント抽出（3テスト）、セグメント特徴量（4テスト）、エッジ特徴量（4テスト）、ラベル付与（2テスト）、撚線メッシュ統合（4テスト）

### 5. k_pen推定MLモデル Step 1 — 特徴量抽出ユーティリティ（+7テスト）

設計仕様 `docs/contact/kpen-estimation-ml-design.md` の Step 1-2 を実装。

**新規モジュール**: `xkep_cae/contact/kpen_features.py`

| 関数 | 説明 |
|------|------|
| `extract_kpen_features()` | 12D 特徴量ベクトル抽出（材料4D + メッシュ3D + 接触幾何5D） |
| `extract_kpen_features_from_mesh()` | TwistedWireMesh からの高レベル API |

**テスト（+7テスト: 全 fast）**:
- 特徴量形状（12D）、材料特徴量の妥当性、メッシュ特徴量、3本/7本撚りメッシュ、変位付き

## ファイル変更

### 新規
- `xkep_cae/contact/prescreening_data.py` — 接触プリスクリーニングGNNデータ生成
- `xkep_cae/contact/kpen_features.py` — k_pen推定ML用特徴量抽出
- `tests/contact/test_prescreening_data.py` — プリスクリーニングデータ生成テスト（17テスト）
- `tests/contact/test_kpen_features.py` — k_pen特徴量テスト（7テスト）
- `docs/status/status-076.md` — 本ステータス

### 変更
- `xkep_cae/thermal/pinn.py` — スパース行列対応（COO格納 + sparse判定 + sparse.mm）
- `tests/thermal/test_pinn.py` — TestSparsePINN クラス追加（8テスト）
- `tests/contact/test_twisted_wire_contact.py` — adaptive omega定量評価 + ヒステリシスループ面積計測（5テスト）
- `README.md` — テスト数更新
- `docs/roadmap.md` — TODO チェックボックス更新
- `docs/status/status-index.md` — status-076 追加
- `CLAUDE.md` — 現在の状態更新

## 設計上の懸念・TODO

### 消化済み（status-075 → 076）
- [x] PINN学習スパース行列対応
- [x] adaptive omega の効果定量評価
- [x] 7本撚りサイクリック荷重でのヒステリシスループ面積計測
- [x] 接触プリスクリーニング GNN Step 1（データ生成パイプライン）
- [x] k_pen推定MLモデル Step 1（特徴量抽出ユーティリティ）

### 未解決（引き継ぎ）
- [ ] **Phase C6: 接触アルゴリズム根本整理**（[設計仕様](../contact/contact-algorithm-overhaul-c6.md)、ML に先立つ最優先）
  - [ ] C6-L1: Segment-to-segment Gauss 積分
  - [ ] C6-L2: ∂s/∂u Jacobian + 完全一貫接線
  - [ ] C6-L3: Semi-smooth Newton + NCP（Outer loop 廃止）
  - [ ] C6-L4: 接触 Schur 前処理
  - [ ] C6-L5: Mortar 離散化
- [ ] 接触プリスクリーニング GNN Step 2-5（C6 後に実施）
- [ ] k_pen推定ML v2 Step 2-7（C6 後に実施）
- [ ] PINN + ハイブリッドGNN 組み合わせ検証
- [ ] 19本撚り/37本撚りでの大規模接触テスト

### 運用上の気付き
- PINNのスパース対応は既存APIとの後方互換性を保持（`use_sparse=False`がデフォルト）
- 接触プリスクリーニングのデータ生成は、NR解法のステップごとの変位履歴が必要。現在の `newton_raphson_block_contact` は `u_history` を返さないことがある点に注意。
- k_pen 特徴量の `Iy` パラメータ名は ruff E741 対応（`I` は曖昧な変数名）

---
