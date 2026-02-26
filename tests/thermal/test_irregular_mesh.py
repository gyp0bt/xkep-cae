"""不規則メッシュでの全結合GNN vs メッシュGNN比較テスト.

正則メッシュでは帰納バイアスにより mesh GNN が FC GNN を上回るが、
不規則メッシュでは FC GNN の優位性が出るか検証する。

テスト項目:
  1. 不規則メッシュ生成の品質（ノード摂動、要素品質）
  2. 不規則メッシュ上のFEM計算の妥当性
  3. FC GNN vs mesh GNN の性能比較（不規則メッシュ）
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.thermal.dataset import (
    ThermalProblemConfig,
    generate_dataset,
    generate_dataset_irregular,
)
from xkep_cae.thermal.fem import make_irregular_rect_mesh, make_rect_mesh
from xkep_cae.thermal.gnn import (
    ThermalGNN,
    evaluate_model,
    graph_dict_to_pyg,
    train_model,
)
from xkep_cae.thermal.gnn_fc import (
    FullyConnectedThermalGNN,
    graph_dict_to_pyg_fc,
)


class TestIrregularMeshGeneration:
    """不規則メッシュ生成の検証."""

    def test_node_count_preserved(self):
        """ノード数が均一メッシュと同一."""
        n_reg, _, _ = make_rect_mesh(0.1, 0.1, 5, 5)
        n_irr, _, _ = make_irregular_rect_mesh(0.1, 0.1, 5, 5)
        assert len(n_reg) == len(n_irr)

    def test_connectivity_preserved(self):
        """接続配列が均一メッシュと同一."""
        _, c_reg, _ = make_rect_mesh(0.1, 0.1, 5, 5)
        _, c_irr, _ = make_irregular_rect_mesh(0.1, 0.1, 5, 5)
        np.testing.assert_array_equal(c_reg, c_irr)

    def test_boundary_nodes_fixed(self):
        """境界ノードが摂動されないこと."""
        nodes_reg, _, _ = make_rect_mesh(0.1, 0.1, 5, 5)
        nodes_irr, _, _ = make_irregular_rect_mesh(0.1, 0.1, 5, 5, perturbation=0.3)

        Lx, Ly = 0.1, 0.1
        tol = 1e-10
        for i in range(len(nodes_reg)):
            x, y = nodes_reg[i]
            if x < tol or x > Lx - tol or y < tol or y > Ly - tol:
                np.testing.assert_allclose(nodes_irr[i], nodes_reg[i])

    def test_interior_nodes_perturbed(self):
        """内部ノードが摂動されること."""
        nodes_reg, _, _ = make_rect_mesh(0.1, 0.1, 5, 5)
        nodes_irr, _, _ = make_irregular_rect_mesh(0.1, 0.1, 5, 5, perturbation=0.3)

        Lx, Ly = 0.1, 0.1
        tol = 1e-10
        n_perturbed = 0
        for i in range(len(nodes_reg)):
            x, y = nodes_reg[i]
            if x > tol and x < Lx - tol and y > tol and y < Ly - tol:
                if not np.allclose(nodes_irr[i], nodes_reg[i]):
                    n_perturbed += 1
        assert n_perturbed > 0, "内部ノードが摂動されていない"

    def test_element_jacobian_positive(self):
        """全要素の Jacobian が正（要素が反転していない）."""
        nodes, conn, _ = make_irregular_rect_mesh(0.1, 0.1, 5, 5, perturbation=0.4)
        for elem in conn:
            xy = nodes[elem]
            # Q4要素中心 (ξ=0, η=0) のJacobian
            dN_dxi = 0.25 * np.array([-(1), (1), (1), -(1)])
            dN_deta = 0.25 * np.array([-(1), -(1), (1), (1)])
            J = np.array(
                [
                    [dN_dxi @ xy[:, 0], dN_deta @ xy[:, 0]],
                    [dN_dxi @ xy[:, 1], dN_deta @ xy[:, 1]],
                ]
            )
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            assert detJ > 0, f"要素反転: detJ={detJ}"

    def test_perturbation_range(self):
        """摂動量が指定範囲内であること."""
        nodes_reg, _, _ = make_rect_mesh(0.1, 0.1, 5, 5)
        nodes_irr, _, _ = make_irregular_rect_mesh(0.1, 0.1, 5, 5, perturbation=0.3)

        dx = 0.1 / 5
        dy = 0.1 / 5
        tol = 1e-10
        Lx, Ly = 0.1, 0.1

        for i in range(len(nodes_reg)):
            x, y = nodes_reg[i]
            if x > tol and x < Lx - tol and y > tol and y < Ly - tol:
                assert abs(nodes_irr[i, 0] - nodes_reg[i, 0]) <= 0.3 * dx + 1e-10
                assert abs(nodes_irr[i, 1] - nodes_reg[i, 1]) <= 0.3 * dy + 1e-10

    def test_reproducibility(self):
        """同一シードで同一メッシュ生成."""
        n1, _, _ = make_irregular_rect_mesh(0.1, 0.1, 5, 5, seed=99)
        n2, _, _ = make_irregular_rect_mesh(0.1, 0.1, 5, 5, seed=99)
        np.testing.assert_array_equal(n1, n2)


class TestIrregularMeshFEM:
    """不規則メッシュ上のFEM計算の妥当性."""

    def test_fem_converges(self):
        """不規則メッシュでFEM解が有限値."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_dataset_irregular(config, n_samples=3, seed=42)
        for sample in raw:
            y = sample["y"]
            assert np.all(np.isfinite(y)), "FEM解に NaN/Inf"

    def test_temperature_rise_positive(self):
        """発熱体がある場合に温度上昇が正."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_dataset_irregular(config, n_samples=5, seed=42)
        for sample in raw:
            dt_max = sample["y"].max()
            assert dt_max > 0, "最大ΔT が 0"


class TestIrregularMeshComparison:
    """不規則メッシュでの全結合GNN vs メッシュGNN比較."""

    @pytest.fixture
    def irregular_data(self):
        """不規則メッシュ上の比較用データセット（5×5メッシュ、100サンプル）."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_dataset_irregular(
            config,
            n_samples=100,
            perturbation=0.35,
            seed=42,
        )

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

    def test_both_models_learn(self, irregular_data):
        """両モデルが不規則メッシュ上で学習できること."""
        d = irregular_data

        # メッシュGNN
        model_mesh = ThermalGNN(node_in_dim=6, hidden_dim=32, n_layers=4, dropout=0.05)
        h_mesh = train_model(
            model_mesh,
            d["mesh_train"],
            d["mesh_val"],
            epochs=100,
            lr=1e-3,
            verbose=False,
        )
        assert h_mesh["train_loss"][-1] < h_mesh["train_loss"][0]

        # FC GNN
        model_fc = FullyConnectedThermalGNN(node_in_dim=6, hidden_dim=48, n_layers=4, dropout=0.05)
        h_fc = train_model(
            model_fc,
            d["fc_train"],
            d["fc_val"],
            epochs=120,
            lr=1e-3,
            verbose=False,
        )
        assert h_fc["train_loss"][-1] < h_fc["train_loss"][0]

    def test_comparison_irregular_vs_regular(self):
        """正則 vs 不規則メッシュでの性能差をレポート.

        正則メッシュでは mesh GNN の帰納バイアスが有利だが、
        不規則メッシュではその優位性が減少し、FC GNN が相対的に改善する可能性を検証。
        """
        config = ThermalProblemConfig(nx=5, ny=5)
        n_samples = 100
        split_train, split_val = 70, 85

        results = {}
        for mesh_type in ["regular", "irregular"]:
            if mesh_type == "regular":
                raw = generate_dataset(config, n_samples=n_samples, seed=42)
            else:
                raw = generate_dataset_irregular(
                    config,
                    n_samples=n_samples,
                    perturbation=0.35,
                    seed=42,
                )

            mesh_pyg = [graph_dict_to_pyg(g, config) for g in raw]
            fc_pyg = [graph_dict_to_pyg_fc(g, config) for g in raw]

            for model_type, data_list, hidden, epochs in [
                ("mesh", mesh_pyg, 32, 120),
                ("fc", fc_pyg, 48, 150),
            ]:
                train_d = data_list[:split_train]
                val_d = data_list[split_train:split_val]
                test_d = data_list[split_val:]

                if model_type == "mesh":
                    model = ThermalGNN(node_in_dim=6, hidden_dim=hidden, n_layers=4, dropout=0.05)
                else:
                    model = FullyConnectedThermalGNN(
                        node_in_dim=6, hidden_dim=hidden, n_layers=4, dropout=0.05
                    )

                h = train_model(model, train_d, val_d, epochs=epochs, lr=1e-3, verbose=False)
                m = evaluate_model(model, test_d, h["y_mean"], h["y_std"])
                key = f"{mesh_type}_{model_type}"
                results[key] = m

        # レポート
        print("\n=== 正則 vs 不規則メッシュ: mesh GNN vs FC GNN ===")
        for key in ["regular_mesh", "regular_fc", "irregular_mesh", "irregular_fc"]:
            r = results[key]
            print(f"  {key:20s}: R²={r['r2']:.3f}, MAE={r['mae']:.3f}°C")

        # FC GNN の不規則メッシュでの相対改善を確認
        # 正則メッシュでの差（mesh - fc）
        regular_gap = results["regular_mesh"]["r2"] - results["regular_fc"]["r2"]
        # 不規則メッシュでの差（mesh - fc）
        irregular_gap = results["irregular_mesh"]["r2"] - results["irregular_fc"]["r2"]

        print(f"\n  正則メッシュ mesh-fc R²差: {regular_gap:+.3f}")
        print(f"  不規則メッシュ mesh-fc R²差: {irregular_gap:+.3f}")

        # 両方のモデルが学習できていること（R² > -1）
        for key, m in results.items():
            assert m["r2"] > -1.0, f"{key}: R²={m['r2']:.3f}"
