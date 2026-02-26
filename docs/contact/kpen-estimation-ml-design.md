# k_pen最適推定MLモデル設計仕様

[← README](../../README.md) | [← roadmap](../roadmap.md) | [← status-067](../status/status-067.md)

**日付**: 2026-02-26
**作成者**: Claude Code

---

## 1. 目的

梁–梁接触のペナルティ剛性 `k_pen` の最適値を機械学習で推定し、
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

### 2.3 観測データ

実際のテスト結果から得られるデータ:

| データ | 取得元 | 形状 |
|--------|--------|------|
| E, I | 材料・断面定義 | スカラー |
| L_elem | メッシュ | スカラー |
| r_contact | 被膜込み接触半径 | スカラー |
| n_segments_per_wire | メッシュ | int |
| n_wires | 撚線構成 | int |
| lay_angle | ヘリックス角 | float |
| n_active_pairs | Outer ループ各ステップ | int |
| gap_distribution | 各ステップ | (n_pairs,) |
| k_pen_final | 収束時の値 | float |
| total_newton_iter | 結果 | int |
| total_outer_iter | 結果 | int |

## 3. MLモデルの設計

### 3.1 問題定式化

**入力**: 断面パラメータ + メッシュ特性 + 接触幾何
**出力**: 最適 k_pen（対数スケール）

「最適」の定義: 総反復数（Inner + Outer）を最小化する k_pen

### 3.2 特徴量設計

```python
features = [
    # 材料・断面（4D）
    log10(E),                    # ヤング率
    log10(I),                    # 断面二次モーメント
    log10(12*E*I/L**3),          # 基準曲げ剛性（無次元化ベース）
    r / L,                       # 接触半径/要素長さ比

    # メッシュ特性（3D）
    log10(n_segments),           # 素線あたりセグメント数
    n_wires,                     # 素線数
    lay_angle,                   # 撚り角 [rad]

    # 接触幾何（5D）
    log10(n_active_est + 1),     # 推定アクティブペア数
    gap_mean / r,                # 平均ギャップ/半径比
    gap_std / r,                 # ギャップ分散
    frac_near_contact,           # gap < 2r のペア割合
    cos_angle_mean,              # 平均セグメント間角度
]
# 計 12D
```

### 3.3 モデルアーキテクチャ

シンプルな MLP（回帰問題）:

```python
class KPenEstimator(nn.Module):
    def __init__(self, in_dim=12, hidden=64, n_layers=3):
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # log10(k_pen) を出力
```

**出力**: `log10(k_pen)` — 対数スケール（k_pen の範囲が 1e4〜1e12 と広いため）

### 3.4 損失関数

```python
# MSE on log-scale
L_regression = MSE(pred_log_kpen, target_log_kpen)

# 収束ペナルティ（収束しなかった場合を考慮）
L_convergence = max(0, total_iter - target_iter) ** 2 / target_iter ** 2

L_total = L_regression + α * L_convergence
```

### 3.5 学習データ生成

**グリッドサーチ方式**: 各ケースで複数の k_pen を試行し、最小反復数の k_pen を正解ラベルとする

```python
def generate_kpen_training_data(cases, k_pen_candidates):
    """グリッドサーチで最適 k_pen を求める."""
    for case in cases:
        mesh = create_mesh(case)
        best_kpen = None
        best_iters = float('inf')

        for k_pen in k_pen_candidates:  # [1e4, 3e4, 1e5, ..., 1e10]
            try:
                result = newton_raphson_with_contact(
                    ..., k_pen=k_pen, max_outer=5, max_iter=30
                )
                total = result.total_outer_iterations + result.total_newton_iterations
                if total < best_iters:
                    best_iters = total
                    best_kpen = k_pen
            except ConvergenceError:
                pass

        if best_kpen is not None:
            features = extract_features(case, mesh)
            yield (features, log10(best_kpen))
```

**データ量見積り**:

| ケース | パラメータ変動 | サンプル数 |
|--------|-------------|-----------|
| 3本撚り | E×3, L×3, 角度×3, 荷重×5 | 135 |
| 7本撚り | E×3, L×3, 角度×3, 荷重×3 | 81 |
| 交差梁 | E×3, L×3, 角度×5 | 45 |
| k_pen候補 | 10点（1e4〜1e10、対数等間隔） | ×10 |
| **計** | | **~2,610** |

### 3.6 推論時の統合

```python
def estimate_kpen(model, E, I, L_elem, mesh_info, contact_info):
    """MLモデルによる k_pen 推定."""
    features = extract_features_from_params(E, I, L_elem, mesh_info, contact_info)
    log_kpen = model(features)
    return 10 ** log_kpen.item()

# newton_raphson_with_contact 内での使用
def initialize_penalty_ml(manager, model, beam_params, mesh_info):
    """MLモデルで k_pen を初期化."""
    k_pen_est = estimate_kpen(model, ...)
    manager.initialize_penalty(k_pen_est, k_t_ratio=0.5)
```

### 3.7 安全策

1. **フォールバック**: ML推定値が妥当範囲外（< 1e2 or > 1e14）なら `auto_beam_penalty_stiffness` を使用
2. **適応的増大は維持**: ML推定を初期値とし、貫入超過時は従来通り `penalty_growth_factor` で増大
3. **上限キャップ**: `k_pen_max` は ContactConfig で制御

## 4. 実装ステップ

1. **Step 1**: グリッドサーチデータ生成スクリプト（既存テストケースの自動実行）
2. **Step 2**: 特徴量抽出ユーティリティ
3. **Step 3**: MLP モデル実装 + 学習スクリプト
4. **Step 4**: `ContactConfig` に `k_pen_mode="ml"` オプション追加
5. **Step 5**: 収束速度ベンチマーク（ML推定 vs 経験式 vs 最適値）

## 5. 期待効果

| 指標 | 現状 | 目標 |
|------|------|------|
| Outer ループ回数 | 2〜5回（適応的増大） | 1〜2回 |
| Inner NR 反復 | 10〜30回 | 10〜20回 |
| 総反復数削減 | — | 30〜50% |
| 推論コスト | — | < 0.1ms |

## 6. リスクと緩和策

| リスク | 緩和策 |
|--------|--------|
| 学習データの偏り（特定の E, I 範囲） | 物理的無次元量ベースの特徴量で汎化性確保 |
| 大規模問題への外挿 | n_pairs, n_segments を対数特徴量として含める |
| 推定失敗時の非収束 | auto_beam_penalty_stiffness フォールバック |
| 動的 k_pen 変化 | 初期推定のみML、適応的増大は従来手法を維持 |

## 7. 既存コードとの関連

- `xkep_cae/contact/law_normal.py`: `auto_beam_penalty_stiffness()` — 置換先
- `xkep_cae/contact/pair.py`: `ContactConfig.k_pen_mode` — `"ml"` オプション追加先
- `xkep_cae/contact/solver_hooks.py`: `newton_raphson_with_contact()` — k_pen初期化の呼び出し元
- `xkep_cae/mesh/twisted_wire.py`: テストケース生成源

---
