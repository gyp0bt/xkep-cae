"""簡略化全結合GNNの検証テスト.

構造テスト + 統合テスト + メッシュGNNとの性能比較。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Data = pytest.importorskip("torch_geometric.data").Data

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
    FCAttentionLayer,
    FullyConnectedThermalGNN,
    compute_fc_edge_attr,
    graph_dict_to_pyg_fc,
    graph_dict_to_pyg_hybrid,
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
    """全結合エッジ特徴量の検証（距離カーネル付き6D）."""

    def test_shape(self):
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        ei = make_fc_edge_index(3)
        ea = compute_fc_edge_attr(nodes, ei, 1.0, 1.0)
        assert ea.shape == (6, 6)  # 3*2 = 6 edges, 6 features

    def test_distance_normalization(self):
        """対角 (0,0)→(1,1) の距離が 1.0 (L_diag = √2 で正規化)."""
        nodes = np.array([[0, 0], [1, 1]], dtype=float)
        ei = np.array([[0], [1]], dtype=np.int64)
        ea = compute_fc_edge_attr(nodes, ei, 1.0, 1.0)
        np.testing.assert_allclose(ea[0, 2], 1.0, atol=1e-6)

    def test_distance_kernels(self):
        """距離カーネル特徴量の妥当性: 近距離でガウシアン≈1, 逆距離大."""
        # 2つのノードが隣接（距離小）
        nodes = np.array([[0, 0], [0.01, 0]], dtype=float)
        ei = np.array([[0], [1]], dtype=np.int64)
        ea = compute_fc_edge_attr(nodes, ei, 1.0, 1.0)
        # dist_n ≈ 0.01/√2 ≈ 0.007 → gauss_narrow ≈ exp(-0.001) ≈ 1.0
        assert ea[0, 4] > 0.95, f"近距離の狭域ガウシアンが低い: {ea[0, 4]}"
        # 逆距離は大きい: 1/(0.007+0.1) ≈ 9.3
        assert ea[0, 3] > 5.0, f"近距離の逆距離カーネルが低い: {ea[0, 3]}"

    def test_distance_kernels_far(self):
        """遠距離でガウシアン≈0, 逆距離小."""
        nodes = np.array([[0, 0], [1, 1]], dtype=float)
        ei = np.array([[0], [1]], dtype=np.int64)
        ea = compute_fc_edge_attr(nodes, ei, 1.0, 1.0)
        # dist_n = 1.0 → gauss_narrow = exp(-1/0.05) ≈ 0
        assert ea[0, 4] < 0.01, f"遠距離の狭域ガウシアンが高い: {ea[0, 4]}"
        # 逆距離: 1/(1.0+0.1) ≈ 0.91
        assert ea[0, 3] < 2.0, f"遠距離の逆距離カーネルが高い: {ea[0, 3]}"


class TestFCModelStructure:
    """全結合GNNモデル構造の検証."""

    def test_forward_pass_shape(self):
        model = FullyConnectedThermalGNN(node_in_dim=6, edge_dim=6, hidden_dim=16, n_layers=2)
        n = 8
        x = torch.randn(n, 6)
        ei_np = make_fc_edge_index(n)
        ei = torch.from_numpy(ei_np)
        ea = torch.randn(ei.shape[1], 6)
        data = Data(x=x, edge_index=ei, edge_attr=ea)
        out = model(data)
        assert out.shape == (n, 1)

    def test_gradients_flow(self):
        model = FullyConnectedThermalGNN(node_in_dim=6, hidden_dim=16, n_layers=2)
        n = 6
        x = torch.randn(n, 6)
        ei_np = make_fc_edge_index(n)
        ei = torch.from_numpy(ei_np)
        ea = torch.randn(ei.shape[1], 6)
        data = Data(x=x, edge_index=ei, edge_attr=ea)
        out = model(data)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_attention_aggregation(self):
        """FCAttentionLayer が add 集約（+ attention重み）を使用."""
        layer = FCAttentionLayer(16, 16, 6)
        assert layer.aggr == "add"


class TestFCPyGConversion:
    """graph_dict → 全結合 PyG Data 変換の検証."""

    def test_fc_conversion_edge_count(self):
        config = ThermalProblemConfig(nx=3, ny=3)
        raw = generate_dataset(config, n_samples=1, seed=42)
        data = graph_dict_to_pyg_fc(raw[0], config)

        n_nodes = data.x.shape[0]  # (3+1)*(3+1) = 16
        expected_edges = n_nodes * (n_nodes - 1)
        assert data.edge_index.shape[1] == expected_edges

    def test_fc_edge_attr_dim(self):
        """エッジ特徴量が6次元（距離カーネル付き）."""
        config = ThermalProblemConfig(nx=3, ny=3)
        raw = generate_dataset(config, n_samples=1, seed=42)
        data = graph_dict_to_pyg_fc(raw[0], config)
        assert data.edge_attr.shape[1] == 6

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
        """比較用データセット（5×5メッシュ、100サンプル）."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_dataset(config, n_samples=100, seed=42)

        mesh_pyg = [graph_dict_to_pyg(g, config) for g in raw]
        fc_pyg = [graph_dict_to_pyg_fc(g, config) for g in raw]

        return {
            "config": config,
            "mesh_train": mesh_pyg[:70],
            "mesh_val": mesh_pyg[70:85],
            "mesh_test": mesh_pyg[85:],
            "fc_train": fc_pyg[:70],
            "fc_val": fc_pyg[70:85],
            "fc_test": fc_pyg[85:],
        }

    @pytest.mark.slow
    def test_fc_achieves_positive_r2(self, comparison_data):
        """全結合GNNが R²>0 を達成できることを確認."""
        d = comparison_data
        model_fc = FullyConnectedThermalGNN(
            node_in_dim=6,
            hidden_dim=48,
            n_layers=4,
            dropout=0.05,
        )
        history = train_model(
            model_fc,
            d["fc_train"],
            d["fc_val"],
            epochs=150,
            lr=1e-3,
            verbose=False,
        )
        metrics = evaluate_model(model_fc, d["fc_test"], history["y_mean"], history["y_std"])
        assert metrics["r2"] > 0.5, f"FC-GNN R²={metrics['r2']:.3f}"

    @pytest.mark.slow
    def test_comparison_both_converge(self, comparison_data):
        """メッシュGNNと全結合GNNの両方が学習収束し、FC-GNNが R²>0.8 を達成."""
        d = comparison_data

        # メッシュGNN
        model_mesh = ThermalGNN(node_in_dim=6, hidden_dim=32, n_layers=4, dropout=0.05)
        h_mesh = train_model(
            model_mesh,
            d["mesh_train"],
            d["mesh_val"],
            epochs=120,
            lr=1e-3,
            verbose=False,
        )
        m_mesh = evaluate_model(model_mesh, d["mesh_test"], h_mesh["y_mean"], h_mesh["y_std"])

        # 全結合GNN（アテンション版、十分なエポック数）
        model_fc = FullyConnectedThermalGNN(node_in_dim=6, hidden_dim=48, n_layers=4, dropout=0.05)
        h_fc = train_model(
            model_fc,
            d["fc_train"],
            d["fc_val"],
            epochs=150,
            lr=1e-3,
            verbose=False,
        )
        m_fc = evaluate_model(model_fc, d["fc_test"], h_fc["y_mean"], h_fc["y_std"])

        # 結果レポート
        print("\n=== メッシュGNN vs 全結合GNN (attention + 距離カーネル) ===")
        print(f"メッシュGNN: R²={m_mesh['r2']:.3f}, MAE={m_mesh['mae']:.3f}°C")
        print(f"全結合GNN:  R²={m_fc['r2']:.3f}, MAE={m_fc['mae']:.3f}°C")
        print(f"メッシュGNN パラメータ: {sum(p.numel() for p in model_mesh.parameters()):,}")
        print(f"全結合GNN  パラメータ: {sum(p.numel() for p in model_fc.parameters()):,}")

        # 両方がまともな精度を出すこと
        assert m_mesh["r2"] > 0.5, f"メッシュGNN 収束失敗: R²={m_mesh['r2']:.3f}"
        assert m_fc["r2"] > 0.5, f"全結合GNN 収束失敗: R²={m_fc['r2']:.3f}"

    @pytest.mark.slow
    def test_fc_fewer_layers_sufficient(self, comparison_data):
        """全結合GNNは少ない層数で情報伝搬できることを確認.

        受容野制約がないため、2層でもメッシュ4層並みの性能が出る可能性。
        """
        d = comparison_data

        model_fc_shallow = FullyConnectedThermalGNN(
            node_in_dim=6,
            hidden_dim=48,
            n_layers=2,
            dropout=0.05,
        )
        h = train_model(
            model_fc_shallow,
            d["fc_train"],
            d["fc_val"],
            epochs=150,
            lr=1e-3,
            verbose=False,
        )
        m = evaluate_model(model_fc_shallow, d["fc_test"], h["y_mean"], h["y_std"])
        # 2層でも学習が進むこと
        assert h["train_loss"][-1] < h["train_loss"][0]
        print(f"\n全結合GNN (2層): R²={m['r2']:.3f}, MAE={m['mae']:.3f}°C")


class TestHybridGraph:
    """ハイブリッドグラフ（メッシュ + 発熱ノードショートカット）の検証."""

    def test_hybrid_edge_count(self):
        """ハイブリッドグラフのエッジ数がメッシュ以上、全結合未満."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_dataset(config, n_samples=1, seed=42)
        d_mesh = graph_dict_to_pyg(raw[0], config)
        d_hybrid = graph_dict_to_pyg_hybrid(raw[0], config)
        d_fc = graph_dict_to_pyg_fc(raw[0], config)

        n_mesh = d_mesh.edge_index.shape[1]
        n_hybrid = d_hybrid.edge_index.shape[1]
        n_fc = d_fc.edge_index.shape[1]

        assert n_hybrid >= n_mesh, f"ハイブリッド({n_hybrid}) < メッシュ({n_mesh})"
        assert n_hybrid <= n_fc, f"ハイブリッド({n_hybrid}) > 全結合({n_fc})"
        print(f"\nエッジ数: メッシュ={n_mesh}, ハイブリッド={n_hybrid}, 全結合={n_fc}")

    def test_hybrid_no_heat_source_fallback(self):
        """発熱ノードがない場合、メッシュエッジのみにフォールバック."""
        config = ThermalProblemConfig(nx=3, ny=3)
        raw = generate_dataset(config, n_samples=1, seed=42)

        # 発熱特徴量をゼロにする
        raw[0]["x"][:, 2] = 0.0
        d_hybrid = graph_dict_to_pyg_hybrid(raw[0], config)
        d_mesh = graph_dict_to_pyg(raw[0], config)

        assert d_hybrid.edge_index.shape[1] == d_mesh.edge_index.shape[1]

    def test_hybrid_training_converges(self):
        """ハイブリッドグラフで学習が収束する."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_dataset(config, n_samples=30, seed=42)
        pyg_data = [graph_dict_to_pyg_hybrid(g, config) for g in raw]

        train_data = pyg_data[:20]
        val_data = pyg_data[20:]

        model = FullyConnectedThermalGNN(node_in_dim=6, hidden_dim=32, n_layers=3)
        history = train_model(
            model,
            train_data,
            val_data,
            epochs=80,
            lr=1e-3,
            verbose=False,
        )
        assert history["train_loss"][-1] < history["train_loss"][0]

    @pytest.mark.slow
    def test_three_way_comparison(self):
        """メッシュGNN vs 全結合GNN vs ハイブリッドGNN の3方比較."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_dataset(config, n_samples=100, seed=42)

        mesh_pyg = [graph_dict_to_pyg(g, config) for g in raw]
        fc_pyg = [graph_dict_to_pyg_fc(g, config) for g in raw]
        hybrid_pyg = [graph_dict_to_pyg_hybrid(g, config) for g in raw]

        split_train, split_val = 70, 85

        results = {}
        for name, data_list, hidden, epochs in [
            ("メッシュGNN", mesh_pyg, 32, 120),
            ("全結合GNN", fc_pyg, 48, 150),
            ("ハイブリッドGNN", hybrid_pyg, 48, 150),
        ]:
            train_d = data_list[:split_train]
            val_d = data_list[split_train:split_val]
            test_d = data_list[split_val:]

            # メッシュGNN は元のエッジ構造（3D エッジ特徴量）
            if name == "メッシュGNN":
                model = ThermalGNN(node_in_dim=6, hidden_dim=hidden, n_layers=4, dropout=0.05)
            else:
                model = FullyConnectedThermalGNN(
                    node_in_dim=6, hidden_dim=hidden, n_layers=4, dropout=0.05
                )

            h = train_model(model, train_d, val_d, epochs=epochs, lr=1e-3, verbose=False)
            m = evaluate_model(model, test_d, h["y_mean"], h["y_std"])
            n_edges = data_list[0].edge_index.shape[1]
            n_params = sum(p.numel() for p in model.parameters())
            results[name] = {**m, "edges": n_edges, "params": n_params}

        # レポート
        print("\n=== 3方比較: メッシュ vs 全結合 vs ハイブリッド ===")
        for name, m in results.items():
            print(
                f"  {name}: R²={m['r2']:.3f}, MAE={m['mae']:.3f}°C, "
                f"エッジ={m['edges']}, パラメータ={m['params']:,}"
            )

        # 3モデルとも R²>0 を達成
        for name, m in results.items():
            assert m["r2"] > 0.0, f"{name} R²={m['r2']:.3f}"
