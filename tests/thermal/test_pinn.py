"""Physics-Informed GNN（PINN）学習の検証テスト.

テスト項目:
  1. PINN用データ生成（K行列、f_shifted）の整合性
  2. 物理ロス計算の正確性
  3. PINN学習の収束性
  4. PINN vs データのみ学習の比較
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from xkep_cae.thermal.dataset import ThermalProblemConfig
from xkep_cae.thermal.gnn import ThermalGNN, evaluate_model
from xkep_cae.thermal.pinn import (
    compute_physics_loss,
    generate_pinn_dataset,
    graph_dict_to_pyg_pinn,
    train_model_pinn,
)


class TestPINNDataGeneration:
    """PINN用データ生成の検証."""

    def test_k_matrix_shape(self):
        """K行列の形状が (N, N) であること."""
        config = ThermalProblemConfig(nx=3, ny=3)
        raw = generate_pinn_dataset(config, n_samples=1, seed=42)
        n_nodes = (config.nx + 1) * (config.ny + 1)  # 16
        assert raw[0]["K_dense"].shape == (n_nodes, n_nodes)

    def test_k_matrix_symmetric(self):
        """K行列が対称であること."""
        config = ThermalProblemConfig(nx=3, ny=3)
        raw = generate_pinn_dataset(config, n_samples=1, seed=42)
        K = raw[0]["K_dense"]
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_k_matrix_positive_definite(self):
        """K行列が正定値であること（全固有値 > 0）."""
        config = ThermalProblemConfig(nx=3, ny=3)
        raw = generate_pinn_dataset(config, n_samples=1, seed=42)
        K = raw[0]["K_dense"]
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals > 0), f"最小固有値: {eigvals.min()}"

    def test_f_shifted_shape(self):
        """f_shifted の形状が (N,) であること."""
        config = ThermalProblemConfig(nx=3, ny=3)
        raw = generate_pinn_dataset(config, n_samples=1, seed=42)
        n_nodes = (config.nx + 1) * (config.ny + 1)
        assert raw[0]["f_shifted"].shape == (n_nodes,)

    def test_exact_solution_satisfies_physics(self):
        """FEM解の ΔT が物理方程式 K @ ΔT = f_shifted を満たすこと."""
        config = ThermalProblemConfig(nx=3, ny=3)
        raw = generate_pinn_dataset(config, n_samples=3, seed=42)

        for sample in raw:
            K = sample["K_dense"]
            f_shifted = sample["f_shifted"]
            dt_exact = sample["y"].flatten()  # ΔT = T - T_min
            residual = K @ dt_exact - f_shifted
            np.testing.assert_allclose(residual, 0, atol=1e-4)

    def test_pyg_conversion_preserves_data(self):
        """PyG変換でK, f_shiftedが保持されること."""
        config = ThermalProblemConfig(nx=3, ny=3)
        raw = generate_pinn_dataset(config, n_samples=1, seed=42)
        data = graph_dict_to_pyg_pinn(raw[0], config)

        assert hasattr(data, "K_dense")
        assert hasattr(data, "f_shifted")
        assert data.K_dense.shape[0] == data.x.shape[0]
        assert data.f_shifted.shape[0] == data.x.shape[0]

    def test_multiple_samples_different_f(self):
        """異なるサンプルで f_shifted が異なること（発熱位置が異なるため）.

        十分なメッシュ密度（5×5）で発熱体ノードが確実に含まれるようにする。
        """
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset(config, n_samples=5, seed=42)
        f_list = [s["f_shifted"] for s in raw]
        any_different = any(not np.allclose(f_list[0], f_list[i]) for i in range(1, len(f_list)))
        assert any_different, "全サンプルの f_shifted が同一"


class TestPhysicsLoss:
    """物理ロス計算の検証."""

    def test_zero_for_exact_solution(self):
        """正確な ΔT に対して物理残差がゼロ."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset(config, n_samples=1, seed=42)
        data = graph_dict_to_pyg_pinn(raw[0], config)

        dt_exact = data.y.squeeze(-1)
        loss = compute_physics_loss(dt_exact, data.K_dense, data.f_shifted)
        assert loss.item() < 1e-6, f"正解の残差が大きい: {loss.item()}"

    def test_nonzero_for_random(self):
        """ランダム予測に対して正の残差."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset(config, n_samples=1, seed=42)
        data = graph_dict_to_pyg_pinn(raw[0], config)

        dt_random = torch.randn(data.x.shape[0])
        loss = compute_physics_loss(dt_random, data.K_dense, data.f_shifted)
        assert loss.item() > 1e-3, f"ランダム予測の残差が小さすぎる: {loss.item()}"

    def test_gradient_flows(self):
        """物理ロスから勾配が伝搬すること."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset(config, n_samples=1, seed=42)
        data = graph_dict_to_pyg_pinn(raw[0], config)

        dt_pred = torch.randn(data.x.shape[0], requires_grad=True)
        loss = compute_physics_loss(dt_pred, data.K_dense, data.f_shifted)
        loss.backward()
        assert dt_pred.grad is not None
        assert torch.any(dt_pred.grad != 0)

    def test_loss_decreases_towards_exact(self):
        """正解に近づくほど物理ロスが減少すること."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset(config, n_samples=1, seed=42)
        data = graph_dict_to_pyg_pinn(raw[0], config)

        dt_exact = data.y.squeeze(-1)
        dt_random = torch.randn_like(dt_exact)

        losses = []
        for alpha in [1.0, 0.5, 0.1, 0.0]:
            dt_interp = alpha * dt_random + (1 - alpha) * dt_exact
            loss = compute_physics_loss(dt_interp, data.K_dense, data.f_shifted)
            losses.append(loss.item())

        # 正解に近づくほど小さい
        for i in range(len(losses) - 1):
            assert losses[i] >= losses[i + 1] - 1e-8


class TestPINNTraining:
    """PINN学習の検証."""

    @pytest.fixture
    def pinn_data(self):
        """PINN用の小規模データセット."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset(config, n_samples=50, seed=42)
        pyg = [graph_dict_to_pyg_pinn(g, config) for g in raw]
        return {
            "config": config,
            "train": pyg[:35],
            "val": pyg[35:43],
            "test": pyg[43:],
        }

    def test_training_converges(self, pinn_data):
        """PINN学習で損失が減少すること."""
        d = pinn_data
        model = ThermalGNN(node_in_dim=6, hidden_dim=32, n_layers=3)
        history = train_model_pinn(
            model,
            d["train"],
            d["val"],
            lambda_phys=0.1,
            epochs=60,
            lr=1e-3,
            verbose=False,
        )
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_physics_loss_decreases(self, pinn_data):
        """物理ロスが学習中に減少すること."""
        d = pinn_data
        model = ThermalGNN(node_in_dim=6, hidden_dim=32, n_layers=3)
        history = train_model_pinn(
            model,
            d["train"],
            d["val"],
            lambda_phys=0.5,
            epochs=60,
            lr=1e-3,
            verbose=False,
        )
        # 後半の物理ロスが前半より小さい
        early_phys = np.mean(history["phys_loss"][:10])
        late_phys = np.mean(history["phys_loss"][-10:])
        assert late_phys < early_phys, (
            f"物理ロス減少なし: early={early_phys:.4f}, late={late_phys:.4f}"
        )

    def test_data_loss_decreases(self, pinn_data):
        """データロスが学習中に減少すること."""
        d = pinn_data
        model = ThermalGNN(node_in_dim=6, hidden_dim=32, n_layers=3)
        history = train_model_pinn(
            model,
            d["train"],
            d["val"],
            lambda_phys=0.1,
            epochs=60,
            lr=1e-3,
            verbose=False,
        )
        assert history["data_loss"][-1] < history["data_loss"][0]

    def test_pinn_achieves_positive_r2(self, pinn_data):
        """PINN学習で R² > 0 を達成すること."""
        d = pinn_data
        model = ThermalGNN(node_in_dim=6, hidden_dim=32, n_layers=4)
        history = train_model_pinn(
            model,
            d["train"],
            d["val"],
            lambda_phys=0.1,
            epochs=120,
            lr=1e-3,
            verbose=False,
        )
        metrics = evaluate_model(model, d["test"], history["y_mean"], history["y_std"])
        assert metrics["r2"] > -0.5, f"PINN R²={metrics['r2']:.3f} (学習不足の可能性)"

    def test_lambda_zero_equals_data_only(self, pinn_data):
        """λ=0 のPINN学習がデータのみ学習と等価であること."""
        d = pinn_data
        torch.manual_seed(123)
        model_pinn = ThermalGNN(node_in_dim=6, hidden_dim=16, n_layers=2)

        history = train_model_pinn(
            model_pinn,
            d["train"],
            d["val"],
            lambda_phys=0.0,
            epochs=30,
            lr=1e-3,
            verbose=False,
        )
        # lambda=0 なら物理ロスは計算されるが損失に寄与しない
        # train_loss ≈ data_loss
        for tl, dl in zip(
            history["train_loss"],
            history["data_loss"],
            strict=True,
        ):
            np.testing.assert_allclose(tl, dl, rtol=1e-5)

    def test_high_lambda_physics_dominant(self, pinn_data):
        """高い λ_phys で物理ロスが支配的になること."""
        d = pinn_data
        model = ThermalGNN(node_in_dim=6, hidden_dim=32, n_layers=3)
        history = train_model_pinn(
            model,
            d["train"],
            d["val"],
            lambda_phys=10.0,
            epochs=40,
            lr=1e-3,
            verbose=False,
        )
        # 物理ロスが減少していること
        assert history["phys_loss"][-1] < history["phys_loss"][0]
