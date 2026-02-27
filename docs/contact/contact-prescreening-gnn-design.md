# 接触プリスクリーニング用GNN設計仕様

[← README](../../README.md) | [← roadmap](../roadmap.md) | [← status-067](../status/status-067.md)

**日付**: 2026-02-26
**作成者**: Claude Code

---

## 1. 目的

梁–梁接触検出において、Broadphase（AABB格子）+ Narrowphase（最近点計算）の
組合せ的探索コストを削減するため、GNNベースのプリスクリーニングモデルを設計する。

**入力**: 梁セグメント座標（現在の変形形状）
**出力**: セグメントペアごとの接触確率（二値分類）

## 2. 現状の接触検出パイプライン

```
[梁セグメント座標] → Broadphase(AABB格子) → 候補ペア → Narrowphase(最近点計算) → 接触ペア
                     O(N + P_cand)            P_cand個      O(P_cand)                  P_active個
```

- `broadphase_aabb()`: 各セグメントのAABBを計算 → 空間ハッシュグリッドに配置 → 同一セル内のAABB重複判定
- セルサイズ: 平均AABB幅の1.5倍（自動推定）
- `closest_point_segments()`: 2線分間の最近接点をニュートン法で求解
- 7本撚り（126セグメント）で ~8000候補ペア → ~100接触ペア（歩留まり ~1%）

### ボトルネック

1. **Broadphaseの過剰候補**: AABB重複は接触の必要条件だが十分条件ではない → 候補の99%が実際には接触しない
2. **Narrowphaseのコスト**: `closest_point_segments` は1ペアあたり ~5 Newton反復 → 8000ペア × 5反復 × 各ステップ
3. **Active Set変動**: 大変形で毎NR反復ごとに再検出 → 検出コストが支配的

## 3. GNNプリスクリーニングの提案

### 3.1 概要

GNNでセグメント間の接触確率を高速に予測し、Narrowphaseの候補を大幅に絞る。

```
[梁セグメント座標] → GNN予測（接触確率マップ） → 高確率ペアのみ → Narrowphase
                     O(N²) or O(N·k)              P_screen個       O(P_screen)
```

目標: P_screen / P_cand < 0.2（候補を80%削減）かつ、true positive > 99%（実際の接触を見逃さない）

### 3.2 グラフ構造

```
ノード: 各梁セグメント（セグメント中点座標 + 方向ベクトル）
エッジ: 空間的近傍（k-NNまたは固定半径）+ 同一梁の隣接
ノード特徴量: [x, y, z, dx, dy, dz, L_seg, r_contact, wire_id, layer_id]  (10D)
エッジ特徴量: [Δx, Δy, Δz, dist, cos_angle, same_wire, same_layer]       (7D)
```

| 特徴量 | 説明 |
|--------|------|
| x, y, z | セグメント中点座標（正規化） |
| dx, dy, dz | セグメント方向ベクトル（単位） |
| L_seg | セグメント長（正規化） |
| r_contact | 接触半径（被膜込み） |
| wire_id | 素線ID（one-hot or 整数） |
| layer_id | 層番号（0: 中心, 1: 第1層, ...） |
| dist | セグメント間距離（正規化） |
| cos_angle | セグメント間の角度余弦 |
| same_wire | 同一素線フラグ（自己接触除外用） |
| same_layer | 同一層フラグ |

### 3.3 出力タスク

**ノードペア分類**: 各エッジ (i, j) に対して接触確率 p_ij ∈ [0, 1] を予測

```python
class ContactPrescreeningGNN(nn.Module):
    def __init__(self, node_dim=10, edge_dim=7, hidden=64, n_layers=4):
        self.encoder = MLP(node_dim, hidden)
        self.layers = [EdgeConvLayer(hidden, hidden, edge_dim) for _ in range(n_layers)]
        self.edge_classifier = MLP(2 * hidden + edge_dim, 1)  # → sigmoid → p_ij

    def forward(self, data):
        x = self.encoder(data.x)
        for layer in self.layers:
            x = layer(x, data.edge_index, data.edge_attr)
        # エッジ分類
        x_i = x[data.edge_index[0]]
        x_j = x[data.edge_index[1]]
        logits = self.edge_classifier(cat([x_i, x_j, data.edge_attr], dim=-1))
        return torch.sigmoid(logits)
```

### 3.4 損失関数

**Focal Loss**（クラス不均衡対応: 接触ペアは全体の ~1%）

```
L = -α (1 - p_t)^γ log(p_t)
  α = [0.05, 0.95]  (非接触, 接触)
  γ = 2.0
```

**追加ペナルティ**: false negative（接触見逃し）を重く罰する

```
L_total = Focal + β * FN_penalty
FN_penalty = -Σ_{contact} log(p_ij + ε)
β = 5.0
```

### 3.5 学習データ生成

既存のテストインフラから自動生成可能:

```python
def generate_prescreening_dataset(n_wires, n_load_steps, seeds):
    """撚線テストケースから接触データを生成."""
    for seed in seeds:
        mesh = make_twisted_wire_mesh(n_wires=n_wires, ...)
        result = newton_raphson_with_contact(model, loads, ...)

        for step in result.steps:
            segments = extract_segments(model, step.u)
            labels = {(i,j): 1 if gap < threshold else 0
                      for (i,j) in all_candidate_pairs}
            yield GraphData(segments, labels)
```

**データ量見積り**:
- 3本撚り × 5荷重 × 10ステップ × 10シード = 500サンプル
- 7本撚り × 3荷重 × 10ステップ × 5シード = 150サンプル
- 計 ~650サンプル（各サンプル ~8000候補ペア）

### 3.6 推論時の統合

```python
def broadphase_with_prescreening(segments, model, threshold=0.3):
    """GNNプリスクリーニング付きbroadphase."""
    # 1. AABB broadphase（高速フィルタ）
    candidates = broadphase_aabb(segments)

    # 2. GNN予測
    graph = segments_to_graph(segments, candidates)
    probs = model(graph)

    # 3. 高確率ペアのみ通過
    screened = [(i, j) for (i, j), p in zip(candidates, probs) if p > threshold]

    return screened
```

### 3.7 性能目標

| 指標 | 目標値 | 備考 |
|------|--------|------|
| Recall（接触検出率） | > 99% | 接触見逃しは力学的に危険 |
| Precision | > 10% | 現状の1%を大幅改善 |
| 候補削減率 | > 80% | Narrowphaseコスト削減 |
| 推論時間 | < 1ms | 1 NR反復あたりの追加コスト |

### 3.8 実装ステップ

1. **Step 1**: データ生成パイプライン（既存テストからの自動収集）
2. **Step 2**: グラフ構築ユーティリティ（セグメント→グラフ変換）
3. **Step 3**: GNNモデル実装 + 学習スクリプト
4. **Step 4**: 推論統合（broadphaseとの連携）
5. **Step 5**: 性能評価（精度 + 速度）

## 4. k_pen推定モデルとのエンコーダ共有

プリスクリーニング GNN と k_pen 推定 ML は同一の接触候補グラフを入力とするため、
GNN エンコーダを共有して Dual Head アーキテクチャとする。
詳細は [kpen-estimation-ml-design.md](kpen-estimation-ml-design.md)（v2）を参照。

### 4.1 共有構造

```
接触候補グラフ
     ↓
  共有 GNN エンコーダ (2-3層 EdgeConv)
     ↓
  ┌──────────────────┬──────────────────────┐
  │  Edge Head        │  Global Head          │
  │  → 接触確率 p_ij   │  → log10(k_pen)       │
  │  (本設計書の対象)    │  (k_pen推定設計書の対象) │
  └──────────────────┴──────────────────────┘
```

### 4.2 ノード/エッジ特徴量の拡張

k_pen 推定のためにノード特徴量を 10D → 12D に拡張（材料特性 E, Iy を追加）、
エッジ特徴量を 7D → 9D に拡張（接触タイプフラグを追加）。
プリスクリーニングタスクでは追加次元を無視（ゼロパディング）しても性能に影響しない。

### 4.3 学習戦略

1. **Phase A**: プリスクリーニングデータ（大量 ~5.2M エッジラベル）でエンコーダ + Edge Head を事前学習
2. **Phase B**: k_pen データ（少量 ~3,150 サンプル）で Global Head をファインチューニング
3. **Phase C**: 両タスクでマルチタスク学習

### 4.4 メリット

- プリスクリーニングの大量データでエンコーダの表現力を確保
- k_pen 推定の少量データ問題を転移学習で緩和
- 推論時にエンコーダ計算は1回で両タスクを処理（追加コスト最小）

## 5. リスクと緩和策

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| False Negative（接触見逃し） | 力学的不整合 | 閾値を低めに設定（0.1〜0.3）、AABBとの併用 |
| 推論コスト > Narrowphaseコスト | 逆効果 | 小規模問題では使わない（N > 100セグメントで有効化） |
| 学習データの偏り | 汎化失敗 | 多様な荷重・撚り角・素線数でデータ生成 |
| 大変形での精度低下 | 見逃し増加 | 変形量に応じた閾値動的調整 |

## 6. 既存コードとの関連

- `xkep_cae/contact/broadphase.py`: `broadphase_aabb()` — 統合先
- `xkep_cae/contact/pair.py`: `ContactManager.detect_candidates()` — 候補生成の呼び出し元
- `xkep_cae/contact/solver_hooks.py`: `newton_raphson_with_contact()` — 各NR反復で呼び出し
- `xkep_cae/mesh/twisted_wire.py`: `make_twisted_wire_mesh()` — テストデータ生成源

---
