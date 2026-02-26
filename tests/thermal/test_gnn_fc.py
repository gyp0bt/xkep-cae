"""簡略化全結合GNNの検証テスト.

構造テスト + 統合テスト + メッシュGNNとの性能比較。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from xkep_cae.thermal.dataset import (
    ThermalProblemConfig,
    generate_dataset,
)
from xkep_cae.thermal.gnn import (
    ThermalGNN,
    evaluate_model,
    graph_dict_to_pyg,
    train_model,
)
from xkep_cae.thermal.gnn_fc import (
    FCEdgeConvLayer,
    FullyConnectedThermalGNN,
    compute_fc_edge_attr,
    graph_dict_to_pyg_fc,
    make_fc_edge_index,
)


class TestFCEdgeIndex:
    """全結合エッジインデックスの検証."""

    def test_edge_count(self):
        """N ノードで N*(N-1) エッジ（双方向）."""
        ei = make_fc_edge_index(5)
        assert ei.shape == (2, 20)  # 5 * 4 = 20

    def test_no_self_loops(self):
        """自己ループがないこと."""
        ei = make_fc_edge_index(10)
        assert not np.any(ei[0] == ei[1])

    def test_bidirectional(self):
        """双方向: (i,j) があれば (j,i) もある."""
        ei = make_fc_edge_index(6)
        edge_set = set(zip(ei[0].tolist(), ei[1].tolist(), strict=True))
        for i in range(6):
            for j in range(6):
                if i != j:
                    assert (i, j) in edge_set


class TestFCEdgeAttr:
    """全結合エッジ特徴量の検証."""

    def test_shape(self):
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        ei = make_fc_edge_index(3)
        ea = compute_fc_edge_attr(nodes, ei, 1.0, 1.0)
        assert ea.shape == (6, 3)  # 3*2 = 6 edges

    def test_distance_normalization(self):
        """対角 (0,0)→(1,1) の距離が 1.0 (L_diag = √2 で正規化)."""
        nodes = np.array([[0, 0], [1, 1]], dtype=float)
        ei = np.array([[0], [1]], dtype=np.int64)
        ea = compute_fc_edge_attr(nodes, ei, 1.0, 1.0)
        np.testing.assert_allclose(ea[0, 2], 1.0, atol=1e-6)


class TestFCModelStructure:
    """全結合GNNモデル構造の検証."""

    def test_forward_pass_shape(self):
        model = FullyConnectedThermalGNN(node_in_dim=6, edge_dim=3, hidden_dim=16, n_layers=2)
        n = 8
        x = torch.randn(n, 6)
        # 全結合エッジ
        ei_np = make_fc_edge_index(n)
        ei = torch.from_numpy(ei_np)
        ea = torch.randn(ei.shape[1], 3)
        data = Data(x=x, edge_index=ei, edge_attr=ea)
        out = model(data)
        assert out.shape == (n, 1)

    def test_gradients_flow(self):
        model = FullyConnectedThermalGNN(node_in_dim=6, hidden_dim=16, n_layers=2)
        n = 6
        x = torch.randn(n, 6)
        ei_np = make_fc_edge_index(n)
        ei = torch.from_numpy(ei_np)
        ea = torch.randn(ei.shape[1], 3)
        data = Data(x=x, edge_index=ei, edge_attr=ea)
        out = model(data)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_mean_aggregation(self):
        """FCEdgeConvLayer が mean 集約を使用."""
        layer = FCEdgeConvLayer(16, 16, 3)
        assert layer.aggr == "mean"


class TestFCPyGConversion:
    """graph_dict → 全結合 PyG Data 変換の検証."""

    def test_fc_conversion_edge_count(self):
        config = ThermalProblemConfig(nx=3, ny=3)
        raw = generate_dataset(config, n_samples=1, seed=42)
        data = graph_dict_to_pyg_fc(raw[0], config)

        n_nodes = data.x.shape[0]  # (3+1)*(3+1) = 16
        expected_edges = n_nodes * (n_nodes - 1)
        assert data.edge_index.shape[1] == expected_edges

    def test_fc_vs_mesh_node_count(self):
        """ノード数とターゲットはメッシュ版と同じ."""
        config = ThermalProblemConfig(nx=3, ny=3)
        raw = generate_dataset(config, n_samples=1, seed=42)
        data_mesh = graph_dict_to_pyg(raw[0], config)
        data_fc = graph_dict_to_pyg_fc(raw[0], config)

        assert data_fc.x.shape == data_mesh.x.shape
        assert torch.allclose(data_fc.y, data_mesh.y)


class TestFCTraining:
    """全結合GNNの学習テスト."""

    def test_small_training_converges(self):
        """小規模学習で損失が減少する."""
        config = ThermalProblemConfig(nx=3, ny=3)
        raw = generate_dataset(config, n_samples=20, seed=42)
        pyg_data = [graph_dict_to_pyg_fc(g, config) for g in raw]

        train_data = pyg_data[:15]
        val_data = pyg_data[15:]

        model = FullyConnectedThermalGNN(node_in_dim=6, hidden_dim=32, n_layers=3)
        history = train_model(
            model,
            train_data,
            val_data,
            epochs=50,
            lr=1e-3,
            verbose=False,
        )
        assert history["train_loss"][-1] < history["train_loss"][0]


class TestFCVsMeshComparison:
    """全結合GNN vs メッシュGNNの性能比較."""

    @pytest.fixture
    def comparison_data(self):
        """比較用データセット（5×5メッシュ、70サンプル）."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_dataset(config, n_samples=70, seed=42)

        mesh_pyg = [graph_dict_to_pyg(g, config) for g in raw]
        fc_pyg = [graph_dict_to_pyg_fc(g, config) for g in raw]

        return {
            "config": config,
            "mesh_train": mesh_pyg[:50],
            "mesh_val": mesh_pyg[50:60],
            "mesh_test": mesh_pyg[60:],
            "fc_train": fc_pyg[:50],
            "fc_val": fc_pyg[50:60],
            "fc_test": fc_pyg[60:],
        }

    def test_fc_achieves_positive_r2(self, comparison_data):
        """全結合GNNが R²>0 を達成できることを確認."""
        d = comparison_data
        model_fc = FullyConnectedThermalGNN(
            node_in_dim=6,
            hidden_dim=32,
            n_layers=4,
            dropout=0.05,
        )
        history = train_model(
            model_fc,
            d["fc_train"],
            d["fc_val"],
            epochs=100,
            lr=1e-3,
            verbose=False,
        )
        metrics = evaluate_model(model_fc, d["fc_test"], history["y_mean"], history["y_std"])
        assert metrics["r2"] > 0.0, f"FC-GNN R²={metrics['r2']:.3f}"

    def test_comparison_both_converge(self, comparison_data):
        """メッシュGNNと全結合GNNの両方が学習収束すること."""
        d = comparison_data

        # メッシュGNN
        model_mesh = ThermalGNN(node_in_dim=6, hidden_dim=32, n_layers=4, dropout=0.05)
        h_mesh = train_model(
            model_mesh,
            d["mesh_train"],
            d["mesh_val"],
            epochs=80,
            lr=1e-3,
            verbose=False,
        )
        m_mesh = evaluate_model(model_mesh, d["mesh_test"], h_mesh["y_mean"], h_mesh["y_std"])

        # 全結合GNN
        model_fc = FullyConnectedThermalGNN(node_in_dim=6, hidden_dim=32, n_layers=4, dropout=0.05)
        h_fc = train_model(
            model_fc,
            d["fc_train"],
            d["fc_val"],
            epochs=80,
            lr=1e-3,
            verbose=False,
        )
        m_fc = evaluate_model(model_fc, d["fc_test"], h_fc["y_mean"], h_fc["y_std"])

        # 結果レポート
        print("\n=== メッシュGNN vs 全結合GNN ===")
        print(f"メッシュGNN: R²={m_mesh['r2']:.3f}, MAE={m_mesh['mae']:.3f}°C")
        print(f"全結合GNN:  R²={m_fc['r2']:.3f}, MAE={m_fc['mae']:.3f}°C")
        print(f"メッシュGNN パラメータ: {sum(p.numel() for p in model_mesh.parameters()):,}")
        print(f"全結合GNN  パラメータ: {sum(p.numel() for p in model_fc.parameters()):,}")

        # 両方が収束していること（R² > -0.5）
        assert m_mesh["r2"] > -0.5, f"メッシュGNN 収束失敗: R²={m_mesh['r2']:.3f}"
        assert m_fc["r2"] > -0.5, f"全結合GNN 収束失敗: R²={m_fc['r2']:.3f}"

    def test_fc_fewer_layers_sufficient(self, comparison_data):
        """全結合GNNは少ない層数で情報伝搬できることを確認.

        受容野制約がないため、2層でもメッシュ6層並みの性能が出る可能性。
        """
        d = comparison_data

        model_fc_shallow = FullyConnectedThermalGNN(
            node_in_dim=6,
            hidden_dim=32,
            n_layers=2,
            dropout=0.05,
        )
        h = train_model(
            model_fc_shallow,
            d["fc_train"],
            d["fc_val"],
            epochs=80,
            lr=1e-3,
            verbose=False,
        )
        m = evaluate_model(model_fc_shallow, d["fc_test"], h["y_mean"], h["y_std"])
        # 2層でも学習が進むこと
        assert h["train_loss"][-1] < h["train_loss"][0]
        print(f"\n全結合GNN (2層): R²={m['r2']:.3f}, MAE={m['mae']:.3f}°C")
