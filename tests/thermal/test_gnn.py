"""GNNサロゲートモデルのテスト.

単体テスト（モデル構造・forward pass）+ 統合テスト（小規模学習）.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from xkep_cae.thermal.dataset import (
    ThermalProblemConfig,
    generate_dataset,
)
from xkep_cae.thermal.gnn import (
    ThermalGNN,
    compute_edge_attr,
    evaluate_model,
    graph_dict_to_pyg,
    train_model,
)


class TestModelStructure:
    """モデル構造の検証."""

    def test_forward_pass_shape(self):
        model = ThermalGNN(node_in_dim=6, edge_dim=3, hidden_dim=16, n_layers=2)
        x = torch.randn(10, 6)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_attr = torch.randn(4, 3)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        out = model(data)
        assert out.shape == (10, 1)

    def test_gradients_flow(self):
        model = ThermalGNN(node_in_dim=6, edge_dim=3, hidden_dim=16, n_layers=2)
        x = torch.randn(10, 6)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_attr = torch.randn(4, 3)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        out = model(data)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_different_layer_counts(self):
        for n_layers in [1, 3, 6]:
            model = ThermalGNN(hidden_dim=16, n_layers=n_layers)
            assert len(model.layers) == n_layers


class TestEdgeFeatures:
    """エッジ特徴量計算の検証."""

    def test_edge_attr_shape(self):
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64)
        ea = compute_edge_attr(nodes, edge_index, 1.0, 1.0)
        assert ea.shape == (4, 3)

    def test_edge_attr_values(self):
        nodes = np.array([[0, 0], [1, 0]], dtype=float)
        edge_index = np.array([[0], [1]], dtype=np.int64)
        ea = compute_edge_attr(nodes, edge_index, 1.0, 1.0)
        np.testing.assert_allclose(ea[0, 0], 1.0)  # dx/Lx
        np.testing.assert_allclose(ea[0, 1], 0.0)  # dy/Ly


class TestIntegration:
    """統合テスト: 小規模データ + 短時間学習."""

    def test_small_training_converges(self):
        """小規模学習で損失が減少することを確認."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw_data = generate_dataset(config, n_samples=20, seed=42)
        pyg_data = [graph_dict_to_pyg(g, config) for g in raw_data]

        train_data = pyg_data[:15]
        val_data = pyg_data[15:]

        model = ThermalGNN(node_in_dim=6, hidden_dim=32, n_layers=3)
        history = train_model(
            model,
            train_data,
            val_data,
            epochs=50,
            lr=1e-3,
            verbose=False,
        )

        # 損失が減少していること
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_evaluate_model(self):
        """評価関数が妥当な値を返すことを確認."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw_data = generate_dataset(config, n_samples=10, seed=0)
        pyg_data = [graph_dict_to_pyg(g, config) for g in raw_data]

        model = ThermalGNN(node_in_dim=6, hidden_dim=16, n_layers=2)
        metrics = evaluate_model(model, pyg_data)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "max_error" in metrics
        assert metrics["mse"] >= 0
        assert metrics["mae"] >= 0
