"""GNNサロゲートモデルの学習・評価スクリプト.

使い方:
    python -m xkep_cae.thermal.train_surrogate

データセット: 300サンプル (train 240 / val 30 / test 30)
メッシュ: 10×10 Q4要素 (121ノード) — CPU高速化のため
GNN: hidden_dim=64, n_layers=6, edge特徴量付き, DataLoaderバッチ処理
"""

from __future__ import annotations

import time

import numpy as np

from xkep_cae.thermal.dataset import ThermalProblemConfig, generate_dataset
from xkep_cae.thermal.gnn import (
    ThermalGNN,
    evaluate_model,
    graph_dict_to_pyg,
    train_model,
)


def main():
    print("=" * 60)
    print("GNN サロゲートモデル — 2D 定常熱伝導")
    print("=" * 60)

    # --- 問題設定 ---
    config = ThermalProblemConfig(
        Lx=0.1,
        Ly=0.1,
        nx=10,
        ny=10,
        k=200.0,
        h_conv=25.0,
        t=0.002,
        T_inf=25.0,
        q_value=5.0e6,
        w_heat=0.015,
        h_heat=0.015,
        n_sources_min=1,
        n_sources_max=5,
    )

    print(f"\nプレート: {config.Lx * 1000:.0f}mm x {config.Ly * 1000:.0f}mm")
    print(
        f"メッシュ: {config.nx} x {config.ny} Q4要素 ({(config.nx + 1) * (config.ny + 1)} ノード)"
    )
    print(f"材料: k={config.k} W/(m K), h={config.h_conv} W/(m2 K)")
    print(
        f"発熱体: {config.w_heat * 1000:.0f}mm x {config.h_heat * 1000:.0f}mm, "
        f"q={config.q_value:.0e} W/m3, {config.n_sources_min}-{config.n_sources_max}個"
    )

    # --- データセット生成 ---
    n_total = 1000
    n_train = 800
    n_val = 100

    print(f"\nデータセット生成中 ({n_total} サンプル)...")
    t0 = time.time()
    raw_data = generate_dataset(config, n_samples=n_total, seed=42)
    dt = time.time() - t0
    print(f"  完了: {dt:.1f} 秒")

    # 統計
    dT_all = np.concatenate([g["y"].flatten() for g in raw_data])
    print(f"  dT 範囲: {dT_all.min():.3f} - {dT_all.max():.3f} C")
    print(f"  dT 平均: {dT_all.mean():.3f} +/- {dT_all.std():.3f} C")

    # --- PyG変換 ---
    print("\nグラフデータへ変換中...")
    pyg_data = [graph_dict_to_pyg(g, config) for g in raw_data]

    train_data = pyg_data[:n_train]
    val_data = pyg_data[n_train : n_train + n_val]
    test_data = pyg_data[n_train + n_val :]

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"  ノード/グラフ: {pyg_data[0].x.shape[0]}")
    print(f"  エッジ/グラフ: {pyg_data[0].edge_index.shape[1]}")

    # --- モデル ---
    hidden_dim = 64
    n_layers = 10
    model = ThermalGNN(
        node_in_dim=6,
        edge_dim=3,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nモデル: ThermalGNN (hidden={hidden_dim}, layers={n_layers})")
    print(f"  パラメータ数: {n_params:,}")

    # --- 学習 ---
    n_epochs = 300
    print(f"\n学習開始 (epochs={n_epochs}, batch_size=32)...")
    t0 = time.time()
    history = train_model(
        model,
        train_data,
        val_data,
        epochs=n_epochs,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=32,
        normalize_target=True,
        grad_clip=1.0,
        verbose=True,
    )
    dt = time.time() - t0
    print(f"学習完了: {dt:.1f} 秒")

    # --- 評価 ---
    print("\n" + "=" * 60)
    print("テストセット評価")
    print("=" * 60)
    y_mean = history.get("y_mean", 0.0)
    y_std = history.get("y_std", 1.0)
    metrics = evaluate_model(model, test_data, y_mean=y_mean, y_std=y_std)
    print(f"  MSE:          {metrics['mse']:.6f}")
    print(f"  MAE:          {metrics['mae']:.6f} C")
    print(f"  R2:           {metrics['r2']:.6f}")
    print(f"  最大誤差:      {metrics['max_error']:.6f} C")
    print(f"  相対誤差 (平均): {metrics['relative_error_pct']:.2f} %")

    # --- 精度基準 ---
    print("\n" + "-" * 60)
    if metrics["r2"] > 0.95:
        print("[PASS] R2 > 0.95: サロゲート精度達成")
    else:
        print(f"[NEED WORK] R2 = {metrics['r2']:.4f}: 改善の余地あり")

    if metrics["relative_error_pct"] < 5.0:
        print("[PASS] 相対誤差 < 5%: 十分な精度")
    else:
        print(f"[NEED WORK] 相対誤差 = {metrics['relative_error_pct']:.2f}%: 改善の余地あり")

    return metrics, history


if __name__ == "__main__":
    main()
