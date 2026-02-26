# status-066: 2D定常熱伝導FEM + GNNサロゲートモデル — R²=0.973達成

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-26
**作業者**: Claude Code
**テスト数**: 1554（+29: thermal FEM 12 + dataset 10 + GNN 7）

## 概要

梁プロジェクトを一時停止し、二次元定常熱伝導要素の開発およびGNNサロゲートモデルの原理検証を実施。
**R² = 0.973, MAE = 0.13°C** を達成し、メッシュベースGNNで温度場予測が可能なことを実証。

## 問題設定

- 100mm × 100mm アルミニウム薄板（t=2mm）
- 10×10 Q4要素メッシュ（121ノード、440エッジ）
- k=200 W/(m·K), h=25 W/(m²·K), T∞=25°C
- 15mm×15mm の矩形発熱体（q=5×10⁶ W/m³）を1〜5個ランダム配置
- 表裏面対流 + 側面対流（フィンモデル）
- 変数: 発熱体の座標のみ（材料・メッシュ・BC固定）

## 実装内容

### 1. 熱伝導FEM (`xkep_cae/thermal/fem.py`)

- **Q4双線形要素**: 2×2 Gauss積分
  - `quad4_conductivity()`: K_e = ∫ ∇Nᵀ k t ∇N dA
  - `quad4_convection_surface()`: H_surf = ∫ Nᵀ 2h N dA
  - `quad4_heat_load()`: f_q = ∫ Nᵀ q t dA
  - `quad4_convection_load_surface()`: f_conv = ∫ Nᵀ 2h T∞ dA
- **辺対流**: `_edge_convection()` — 2節点辺の h·t 対流
- **メッシュ生成**: `make_rect_mesh()` — 均一矩形メッシュ + 境界辺辞書
- **アセンブリ**: `assemble_thermal_system()` — COO→CSR形式
- **ソルバー**: `solve_steady_thermal()` — scipy spsolve
- **熱流束**: `compute_heat_flux()` — 要素中心で q = -k∇T

### 2. データセット生成 (`xkep_cae/thermal/dataset.py`)

- `ThermalProblemConfig`: 全パラメータをdataclassで管理
- `place_heat_sources()`: ランダム矩形発熱体配置
- `generate_single_sample()`: FEM計算の1サンプル実行
- `mesh_to_edge_index()`: Q4メッシュ → 双方向エッジリスト
- `sample_to_graph_data()`: FEM結果 → ノード特徴量6次元 + ΔTターゲット
- `generate_dataset()`: 任意サンプル数のデータセット生成

**ノード特徴量（6次元）**:
1. x座標（正規化）
2. y座標（正規化）
3. 発熱密度（バイナリ 0/1）
4. 最近接境界距離（正規化）
5. 境界フラグ（0/1）
6. **発熱ポテンシャル** Σ 1/(r² + ε) — 精度に決定的

### 3. GNNサロゲートモデル (`xkep_cae/thermal/gnn.py`)

**アーキテクチャ**: EdgeConvLayer × 10層

```
ノード特徴量 (6) → Encoder MLP (6→64→64)
→ EdgeConvLayer × 10 (hidden=64, edge_dim=3, dropout=0.1)
→ Decoder MLP (64→64→1) → ΔT
```

**EdgeConvLayer**:
```
m_ij = MLP_msg(x_i || x_j || e_ij)  # メッセージ
x_i' = MLP_upd(x_i || Σ_j m_ij)     # 更新
→ LayerNorm + 残差接続
```

**エッジ特徴量（3次元）**: [Δx/Lx, Δy/Ly, ||Δr||/L_diag]

**学習設定**:
- データ: 1000サンプル (train 800 / val 100 / test 100)
- ターゲット標準化（mean/std正規化）
- Adam + ReduceLROnPlateau (patience=20, factor=0.5)
- Gradient clipping (max_norm=1.0)
- DataLoader バッチ処理 (batch_size=32)
- 300 epochs

### 4. 学習結果

| 指標 | 値 |
|------|-----|
| R² | **0.973** |
| MSE | 0.045 |
| MAE | 0.127 °C |
| 最大誤差 | 1.84 °C |
| 相対誤差（平均） | 14.1 % |
| パラメータ数 | 260,353 |
| 学習時間 | ~33分（CPU） |

### 精度改善の推移

| バージョン | 特徴量 | 層数 | データ | R² | MAE |
|-----------|--------|------|--------|-----|------|
| v1 | 3D (座標+q) | 6 | 300 | 0.894 | 0.37°C |
| v2 | 5D (+境界) | 10 | 500 | 0.919 | 0.28°C |
| v3 | 5D + dropout | 10 | 1000 | 0.915 | 0.28°C |
| **v4** | **6D (+ポテンシャル)** | **10** | **1000** | **0.973** | **0.13°C** |

**決定的な改善要因**: 発熱ポテンシャル特徴量 Σ 1/(r²+ε) の追加。これにより各ノードが全発熱体の空間分布を「見る」ことが可能になり、メッセージパッシングの受容野制約を実質的に回避。

## テスト追加（+29テスト）

| ファイル | テスト数 | 内容 |
|---------|---------|------|
| test_thermal_fem.py | 12 | メッシュ生成、要素行列、1D伝導、フィンモデル、熱流束 |
| test_dataset.py | 10 | 発熱体配置、グラフ変換、データセット生成 |
| test_gnn.py | 7 | モデル構造、エッジ特徴量、統合学習テスト |

## ファイル変更

### 新規
- `xkep_cae/thermal/__init__.py`
- `xkep_cae/thermal/fem.py` — Q4熱伝導FEM
- `xkep_cae/thermal/dataset.py` — データセット生成
- `xkep_cae/thermal/gnn.py` — GNNサロゲートモデル
- `xkep_cae/thermal/train_surrogate.py` — 学習・評価スクリプト
- `tests/thermal/__init__.py`
- `tests/thermal/test_thermal_fem.py`
- `tests/thermal/test_dataset.py`
- `tests/thermal/test_gnn.py`
- `docs/status/status-066.md`

### 変更
- `docs/status/status-index.md` — status-066追加
- `docs/roadmap.md` — GNNサロゲート項目追加
- `README.md` — 現在状態更新

## 設計上の洞察と改善方向

### GNNがこの問題で苦戦する理由

1. **受容野制約**: 10×10グリッドの対角は~14ホップ。10層MPでも全域カバー困難
2. **正則グリッド**: CNN が最適な構造。GNN は不規則メッシュ向け
3. **線形問題**: T = K⁻¹·q の Green 関数は CNN の畳み込みで直接学習可能

### 発熱ポテンシャル特徴量の成功

Σ 1/(r²+ε) は各ノードに「全発熱体の空間分布」の情報を1ホップで提供。
これはGNNの受容野制約を特徴量エンジニアリングで回避する手法。

### 次のステップ候補

- [ ] **全結合簡略化グラフ**: 発熱体ノード + メッシュ代表点の全結合グラフ → 少数ノードで長距離相互作用を1ホップで表現
- [ ] **DeepSets / Set Transformer**: 発熱体座標のセットを処理し、固定メッシュ上の温度場にデコード
- [ ] **Physics-Informed ロス**: L_phys = ||K·T_pred - Q||² を追加（収束加速）
- [ ] **MeshGraphNet / GNO**: 不規則メッシュ対応、メッシュ解像度非依存の汎化
- [ ] **FNO (Fourier Neural Operator)**: 正則グリッドでのGreen関数の直接学習

### pyproject.toml の依存関係

GNN機能を使うには以下のインストールが必要:
```bash
pip install torch torch-geometric
```

---
