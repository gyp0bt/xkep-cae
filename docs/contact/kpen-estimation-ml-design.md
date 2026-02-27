# k_pen最適推定MLモデル設計仕様（v2 — GNNベース）

[← README](../../README.md) | [← roadmap](../roadmap.md) | [← status-067](../status/status-067.md)

**日付**: 2026-02-27（v2 改訂）
**作成者**: Claude Code
**v1 → v2 変更点**: MLP→GNNベース、目的関数を反復回数→残差ベースに変更、プリスクリーニングGNNとのエンコーダ共有

---

## 1. 目的

梁–梁接触および梁–シース接触のペナルティ剛性 `k_pen` の最適値を機械学習で推定し、
Newton-Raphson の収束速度を改善する。

現在の `auto_beam_penalty_stiffness()` は経験式 `scale × 12EI/L³ / n_pairs` で
推定するが、`scale` と `n_contact_pairs` の適切な値が問題依存であり、
過小推定は追加 Outer ループ（適応的増大）、過大推定は条件数悪化を招く。

## 2. 現状の問題点

### 2.1 auto_beam_penalty_stiffness の限界

```python
def auto_beam_penalty_stiffness(E, I, L_elem, *, n_contact_pairs=1, scale=0.1, scaling="linear"):
    if scaling == "sqrt":
        return scale * 12 * E * I / L_elem**3 / max(1, np.sqrt(n_contact_pairs))
    return scale * 12 * E * I / L_elem**3 / max(1, n_contact_pairs)
```

| 問題 | 詳細 |
|------|------|
| `n_contact_pairs` の事前把握困難 | 初期化時に最終的なアクティブペア数は不明 |
| `scale` が問題依存 | 0.01〜1.0 の範囲で最適値が変動 |
| 幾何的要因の欠落 | 接触角度、接触幅、曲率の影響を無視 |
| 動的変化に追従不可 | 荷重ステップ進行に伴う最適値の変化に対応しない |

### 2.2 k_pen が収束に与える影響

```
k_pen 小 → 貫入超過 → penalty_growth_factor(=2.0)で倍増
         → 追加 Outer ループ → 総反復数増加

k_pen 大 → K_total = K_T + k_pen·g·gᵀ の条件数悪化
         → Inner NR 非収束 → GMRES フォールバック → 速度低下
         → 最悪: 全体非収束
```

### 2.3 v1 設計（MLP）の限界

v1 では12D固定次元の集計統計量を入力とするMLPを採用していたが、以下の問題が判明:

| 問題 | 詳細 |
|------|------|
| メッシュ規模外挿不可 | 3本撚り20seg/本で学習 → 7本撚り50seg/本に適用困難 |
| 接触トポロジ欠落 | gap_mean/std だけでは局所密集 vs 均一分散を区別不能 |
| 接触タイプ混在 | 梁-梁と梁-シースが同一モデル内で混在する場合に集計統計量では不足 |
| 目的関数（反復回数）の離散性 | k_pen を微小変化させても反復回数が同じ → 最適値の分解能が粗い |

## 3. v2 設計: GNNベース + エンコーダ共有

### 3.1 アーキテクチャ概要

接触候補グラフ上の GNN で局所接触パターンを学習し、Global Readout で問題全体のスカラー値（k_pen）を推定する。プリスクリーニング GNN とエンコーダを共有することで、データ効率と推論効率を両立する。

```
接触候補グラフ (Broadphase 出力)
     ↓
  共有 GNN エンコーダ (2-3層 EdgeConv)     ← プリスクリーニングと共有
     ↓
  ┌──────────────────┬──────────────────────┐
  │  Edge Head        │  Global Head          │
  │  (MLP 2層)        │  (Readout + MLP 2層)  │
  │  → 接触確率 p_ij   │  → log10(k_pen)       │
  │  (プリスクリーニング)  │  (k_pen 推定)         │
  └──────────────────┴──────────────────────┘
```

### 3.2 グラフ構造（プリスクリーニングと共通）

```
ノード: 各梁セグメント
  特徴量 (12D):
    [x, y, z,                     # 中点座標（正規化）
     dx, dy, dz,                  # 方向ベクトル（単位）
     L_seg,                       # セグメント長（正規化）
     r_contact,                   # 接触半径（被膜込み）
     wire_id,                     # 素線ID
     layer_id,                    # 層番号（0: 中心, 1: 第1層, ...）
     log10(E), log10(Iy)]         # 材料特性 ← v2 で追加

エッジ: Broadphase 候補 + 同一梁隣接
  特徴量 (9D):
    [Δx, Δy, Δz,                 # 相対位置ベクトル
     dist,                        # セグメント間距離
     cos_angle,                   # 角度余弦
     same_wire,                   # 同一素線フラグ
     same_layer,                  # 同一層フラグ
     contact_type_beam_beam,      # 梁-梁フラグ ← v2 で追加
     contact_type_beam_sheath]    # 梁-シースフラグ ← v2 で追加
```

**v1 との違い**: ノードに材料特性を追加（E, Iy）、エッジに接触タイプフラグを追加。これにより梁-梁と梁-シースを統一モデルで扱える。

### 3.3 モデルアーキテクチャ

```python
class ContactDualTaskGNN(nn.Module):
    """プリスクリーニング + k_pen推定の共有エンコーダGNN."""

    def __init__(self, node_dim=12, edge_dim=9, hidden=64, n_layers=3):
        super().__init__()
        # === 共有エンコーダ ===
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden)
        )
        self.gnn_layers = nn.ModuleList([
            EdgeConvLayer(hidden, hidden) for _ in range(n_layers)
        ])

        # === タスクヘッド: プリスクリーニング（エッジ分類）===
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden + hidden, hidden),  # [h_i || h_j || e_ij]
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # === タスクヘッド: k_pen推定（グローバル回帰）===
        self.kpen_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),  # [mean_pool || max_pool]
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, data):
        """共有エンコーダ: ノード埋め込みを計算."""
        x = self.node_encoder(data.x)                  # (N, hidden)
        e = self.edge_encoder(data.edge_attr)           # (E, hidden)
        for layer in self.gnn_layers:
            x = layer(x, data.edge_index, e)            # message passing
        return x, e

    def forward_prescreening(self, data):
        """エッジごとの接触確率を予測."""
        x, e = self.encode(data)
        x_i = x[data.edge_index[0]]
        x_j = x[data.edge_index[1]]
        logits = self.edge_classifier(torch.cat([x_i, x_j, e], dim=-1))
        return torch.sigmoid(logits.squeeze(-1))

    def forward_kpen(self, data):
        """問題全体の log10(k_pen) を予測."""
        x, _e = self.encode(data)
        # Global readout: mean + max pooling
        if hasattr(data, "batch") and data.batch is not None:
            from torch_geometric.nn import global_mean_pool, global_max_pool
            g_mean = global_mean_pool(x, data.batch)    # (B, hidden)
            g_max = global_max_pool(x, data.batch)      # (B, hidden)
        else:
            g_mean = x.mean(dim=0, keepdim=True)        # (1, hidden)
            g_max = x.max(dim=0, keepdim=True).values    # (1, hidden)
        g = torch.cat([g_mean, g_max], dim=-1)           # (B, 2*hidden)
        return self.kpen_head(g).squeeze(-1)             # (B,)
```

### 3.4 GNN の利点（v1 MLP との比較）

| 項目 | v1 MLP (12D → 1) | v2 GNN + Global Readout |
|------|-------------------|------------------------|
| **メッシュ規模** | 固定次元 → 外挿不可 | Message passing + pooling で任意 N に対応 |
| **接触パターン** | 集計統計量のみ（mean/std） | 局所構造を message passing で学習 |
| **接触タイプ** | 暗黙的統一（区別不能） | ノード/エッジ特徴で明示的に区別 |
| **データ効率** | 独立学習（2,610サンプル） | エンコーダをプリスクリーニングタスクで事前学習可 |
| **推論コスト** | <0.1ms | ~1-5ms（k_pen推定は初期化時1回のみなので許容） |

### 3.5 目的関数: 固定イテレーション残差ベース

v1 の「総反復回数最小化」から変更。

#### v1 の問題点

1. **離散値** — k_pen を微小変化させても反復回数は同じ（例: k_pen=1e5→23回、1.5e5→23回、3e5→22回）
2. **非収束の扱い** — 非収束=∞ とするとラベル欠損、閾値を設けるとバイアス混入
3. **収束判定基準依存** — NR の `tol` を変えるとラベルが変化

#### v2: 残差和スコア

```python
def compute_kpen_score(k_pen, *, n_fixed_outer=3, n_fixed_inner=10):
    """固定イテレーション数での残差スコアを計算.

    Args:
        k_pen: 評価するペナルティ剛性
        n_fixed_outer: 固定 Outer ループ回数
        n_fixed_inner: 固定 Inner NR 反復回数

    Returns:
        score: 残差スコア（小さいほど良い）
    """
    result = newton_raphson_with_contact(
        ...,
        k_pen=k_pen,
        max_outer=n_fixed_outer,
        max_iter=n_fixed_inner,
        return_residual_history=True,
    )
    # 各イテレーションの残差ノルムの対数和
    residuals = result.residual_history  # [(outer, inner, ||R||), ...]
    return sum(math.log10(r + 1e-30) for _, _, r in residuals)
```

#### v2 代替案: 収束率（対数残差の線形回帰勾配）

```python
def compute_convergence_rate(k_pen, *, n_fixed_outer=3, n_fixed_inner=10):
    """収束率を計算（対数残差の線形回帰勾配）."""
    result = newton_raphson_with_contact(
        ...,
        k_pen=k_pen,
        max_outer=n_fixed_outer,
        max_iter=n_fixed_inner,
        return_residual_history=True,
    )
    log_residuals = [math.log10(r) for _, _, r in result.residual_history if r > 0]
    if len(log_residuals) < 3:
        return 0.0  # 残差履歴が短すぎる場合
    slope, _ = np.polyfit(range(len(log_residuals)), log_residuals, 1)
    return -slope  # 正の値で大きいほど良い
```

#### 最終目的変数の定式

```python
# 推奨: 残差和スコア（連続、非収束でも計算可能）
target_log_kpen = argmin_{log10(k_pen)} score(k_pen)

# グリッドサーチ: 対数等間隔10点
k_pen_candidates = np.logspace(4, 10, 10)  # [1e4, ..., 1e10]
scores = [compute_kpen_score(k) for k in k_pen_candidates]
best_idx = np.argmin(scores)

# さらに: 最小点付近で黄金分割探索で精密化（オプション）
best_log_kpen = golden_section_search(
    lambda lk: compute_kpen_score(10**lk),
    a=np.log10(k_pen_candidates[max(0, best_idx-1)]),
    b=np.log10(k_pen_candidates[min(len(scores)-1, best_idx+1)]),
)
```

| 項目 | v1 反復回数 | v2 残差和 | v2 収束率 |
|------|-----------|---------|---------|
| **値域** | 離散（整数） | 連続（float） | 連続（float） |
| **分解能** | 粗い（1回刻み） | 高い | 高い |
| **非収束対応** | ラベル欠損 | 大きな残差値として処理 | slope ≈ 0 として処理 |
| **k_pen感度** | 広いプラトー | 鋭い最小値 | 鋭い最大値 |
| **計算コスト** | 収束まで実行 | 固定回数で打ち切り（高速） | 同左 |
| **推奨度** | — | **推奨（メイン）** | 補助指標として併用可 |

### 3.6 損失関数

#### k_pen 推定ヘッド

```python
# メイン: 回帰損失（log scale）
L_kpen = MSE(pred_log_kpen, target_log_kpen)
```

#### プリスクリーニングヘッド

```python
# Focal Loss（クラス不均衡対応）
L_prescreening = FocalLoss(pred_prob, label, alpha=[0.05, 0.95], gamma=2.0)
            + β * FN_penalty
```

#### マルチタスク学習

```python
L_total = L_prescreening + λ * L_kpen
# λ: タスク重み（Uncertainty Weighting で自動調整 or λ=0.1 固定）
```

学習戦略:
1. **Phase A**: プリスクリーニングデータ（大量）で共有エンコーダ + Edge Head を事前学習
2. **Phase B**: k_pen データ（少量）で Global Head をファインチューニング（エンコーダは LR を 0.1x に減衰）
3. **Phase C**: 両タスクでマルチタスク学習（最終調整）

### 3.7 学習データ生成

#### プリスクリーニング用（大量・ラベル安価）

```python
# 既存パイプライン prescreening_data.py を使用
# 650サンプル × ~8000エッジ/サンプル = ~5.2M エッジラベル
```

#### k_pen推定用（少量・ラベル高コスト）

```python
def generate_kpen_training_data(cases, k_pen_candidates, n_fixed_outer=3, n_fixed_inner=10):
    """残差ベースのグリッドサーチで最適 k_pen を求める."""
    for case in cases:
        mesh = create_mesh(case)

        scores = []
        for k_pen in k_pen_candidates:  # [1e4, 3e4, 1e5, ..., 1e10]
            score = compute_kpen_score(
                k_pen, n_fixed_outer=n_fixed_outer, n_fixed_inner=n_fixed_inner,
            )
            scores.append(score)

        best_idx = np.argmin(scores)
        best_log_kpen = np.log10(k_pen_candidates[best_idx])

        # グラフデータ生成（prescreening と同一フォーマット）
        graph = generate_prescreening_sample(mesh, ...)
        graph["target_log_kpen"] = best_log_kpen
        graph["residual_scores"] = scores  # 全候補のスコア（デバッグ用）
        yield graph
```

**データ量見積り**:

| ケース | パラメータ変動 | サンプル数 |
|--------|-------------|-----------|
| 3本撚り（梁-梁） | E×3, L×3, 角度×3, 荷重×5 | 135 |
| 7本撚り（梁-梁） | E×3, L×3, 角度×3, 荷重×3 | 81 |
| 交差梁（梁-梁） | E×3, L×3, 角度×5 | 45 |
| 3本撚り+シース（梁-シース） | E×3, L×3, 荷重×3 | 27 |
| 7本撚り+シース（梁-シース） | E×3, L×3, 荷重×3 | 27 |
| k_pen候補 | 10点（1e4〜1e10、対数等間隔） | ×10 |
| **計** | | **~3,150** |

**実行時間見積り**（固定iter打ち切りにより v1 比で高速化）:
- 固定 outer=3, inner=10: 1ケース ≈ 30-50ms（収束を待たない）
- 3,150ケース × 50ms = **約2.6分**（シングルスレッド）

### 3.8 推論時の統合

```python
# ContactConfig に追加
k_pen_mode: str = "manual"  # "manual" | "beam_ei" | "ml"
kpen_model_path: str | None = None  # 学習済みモデルパス

def initialize_penalty_ml(manager, model, mesh, beam_params):
    """GNNモデルで k_pen を初期化."""
    # 1. 接触候補グラフ構築（Broadphase 出力を流用）
    graph = build_contact_graph(manager, mesh, beam_params)

    # 2. k_pen 推定
    log_kpen = model.forward_kpen(graph)
    k_pen_est = 10 ** float(log_kpen)

    # 3. 妥当性チェック + フォールバック
    if not (1e2 <= k_pen_est <= 1e14):
        k_pen_est = auto_beam_penalty_stiffness(
            beam_params.E, beam_params.I, beam_params.L_elem,
            n_contact_pairs=max(1, manager.n_active),
        )

    # 4. 初期化
    manager.initialize_penalty(k_pen_est, k_t_ratio=0.5)
```

### 3.9 安全策

1. **フォールバック**: ML推定値が妥当範囲外（< 1e2 or > 1e14）なら `auto_beam_penalty_stiffness` を使用
2. **適応的増大は維持**: ML推定を初期値とし、貫入超過時は従来通り `penalty_growth_factor` で増大
3. **上限キャップ**: `k_pen_max` は ContactConfig で制御
4. **モデル未ロード時**: `k_pen_mode="ml"` でモデルがない場合は `beam_ei` にフォールバック

## 4. 実装ステップ

| Step | 内容 | 依存 | ステータス |
|------|------|------|----------|
| **1** | 特徴量抽出ユーティリティ（12D集計、ノード/エッジ特徴量） | — | ✅ 完了（status-076） |
| **2** | グラフ構築ユーティリティ（共有フォーマット、ノード12D + エッジ9D） | Step 1 | ⏳ |
| **3** | 残差ベーススコア計算 + グリッドサーチデータ生成スクリプト | Step 2 | ⏳ |
| **4** | 共有 GNN エンコーダ + Dual Head 実装 | Step 2 | ⏳ |
| **5** | 学習パイプライン（Phase A→B→C） | Step 3, 4 | ⏳ |
| **6** | `ContactConfig` に `k_pen_mode="ml"` 統合 | Step 5 | ⏳ |
| **7** | 収束速度ベンチマーク（ML推定 vs 経験式 vs 最適値） | Step 6 | ⏳ |

**Step 1 の資産活用**: `kpen_features.py` の `extract_kpen_features()` は GNN 版でもノード特徴量の材料パラメータ（E, Iy）算出に利用。`prescreening_data.py` の `extract_segments()`, `compute_segment_features()`, `compute_edge_features()` はグラフ構築の基盤。

## 5. 期待効果

| 指標 | 現状 | 目標 |
|------|------|------|
| Outer ループ回数 | 2〜5回（適応的増大） | 1〜2回 |
| Inner NR 反復 | 10〜30回 | 10〜20回 |
| 総反復数削減 | — | 30〜50% |
| 推論コスト | — | < 5ms（初期化時1回） |
| メッシュ規模汎化 | — | 3本撚りで学習 → 7/19本で適用可 |
| 接触タイプ汎化 | — | 梁-梁 + 梁-シース統一モデル |

## 6. リスクと緩和策

| リスク | 緩和策 |
|--------|--------|
| 学習データの偏り（特定の E, I 範囲） | 物理的無次元量ベースの特徴量 + ノード正規化 |
| 大規模問題への外挿 | GNN の pooling によりメッシュ規模不変 |
| 推定失敗時の非収束 | auto_beam_penalty_stiffness フォールバック |
| 動的 k_pen 変化 | 初期推定のみML、適応的増大は従来手法を維持 |
| GNN学習データ不足（3,150サンプル） | プリスクリーニングとのエンコーダ共有で事前学習 |
| 梁-シース接触のデータ偏り | シース専用ケースを追加（27+27サンプル） |
| 残差スコアの数値安定性 | log10(r + ε) で下限クリップ、外れ値除去 |

## 7. 既存コードとの関連

- `xkep_cae/contact/kpen_features.py`: Step 1 実装済み特徴量抽出（GNNノード特徴量の素材として活用）
- `xkep_cae/contact/prescreening_data.py`: グラフ構築基盤（`extract_segments`, `compute_*_features`）
- `xkep_cae/contact/law_normal.py`: `auto_beam_penalty_stiffness()` — フォールバック先
- `xkep_cae/contact/pair.py`: `ContactConfig.k_pen_mode` — `"ml"` オプション追加先
- `xkep_cae/contact/solver_hooks.py`: `newton_raphson_with_contact()` — k_pen初期化の呼び出し元
- `xkep_cae/mesh/twisted_wire.py`: テストケース生成源
- `docs/contact/contact-prescreening-gnn-design.md`: エンコーダ共有先の設計仕様

## 8. 変更履歴

| 日付 | 版 | 変更内容 |
|------|---|---------|
| 2026-02-26 | v1 | 初版（MLP 12D→1、反復回数ベース目的関数） |
| 2026-02-27 | v2 | GNNベース + エンコーダ共有に変更、目的関数を残差和ベースに変更、梁-シースの明示的対応を追加 |

---
