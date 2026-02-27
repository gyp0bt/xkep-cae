"""PINN拡張検証テスト — 大規模メッシュ + 不規則メッシュ + PINN効果比較.

テスト項目:
  1. 大規模メッシュ（20×20）でのPINNデータ生成・学習・精度検証
  2. 不規則メッシュでのPINNデータ生成・物理整合性・学習検証
  3. PINN vs data-only学習の比較（正則/不規則メッシュ）
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xkep_cae.thermal.dataset import ThermalProblemConfig
from xkep_cae.thermal.gnn import ThermalGNN, evaluate_model, graph_dict_to_pyg, train_model
from xkep_cae.thermal.pinn import (
    compute_physics_loss,
    generate_pinn_dataset,
    generate_pinn_dataset_irregular,
    graph_dict_to_pyg_pinn,
    train_model_pinn,
)

# =========================================================================
# 1. 大規模メッシュ（20×20）でのPINN検証
# =========================================================================


class TestPINNLargeMesh:
    """20×20メッシュ（441ノード）でのPINN検証."""

    @pytest.fixture
    def large_config(self):
        """20×20メッシュの問題設定."""
        return ThermalProblemConfig(nx=20, ny=20)

    def test_large_mesh_k_matrix_shape(self, large_config):
        """20×20メッシュのK行列形状が (441, 441)."""
        raw = generate_pinn_dataset(large_config, n_samples=1, seed=42)
        n_nodes = (large_config.nx + 1) * (large_config.ny + 1)  # 441
        assert n_nodes == 441
        assert raw[0]["K_dense"].shape == (441, 441)

    def test_large_mesh_k_matrix_properties(self, large_config):
        """大規模メッシュのK行列の対称性と正定値性."""
        raw = generate_pinn_dataset(large_config, n_samples=1, seed=42)
        K = raw[0]["K_dense"]
        # 対称性
        np.testing.assert_allclose(K, K.T, atol=1e-10)
        # 正定値性
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals > 0), f"最小固有値: {eigvals.min()}"

    def test_large_mesh_physics_exact(self, large_config):
        """大規模メッシュでFEM解が物理方程式を満たすこと."""
        raw = generate_pinn_dataset(large_config, n_samples=3, seed=42)
        for sample in raw:
            K = sample["K_dense"]
            f_shifted = sample["f_shifted"]
            dt_exact = sample["y"].flatten()
            residual = K @ dt_exact - f_shifted
            np.testing.assert_allclose(residual, 0, atol=1e-3)

    @pytest.mark.slow
    def test_large_mesh_pinn_training_converges(self, large_config):
        """20×20メッシュでPINN学習の損失が減少すること."""
        raw = generate_pinn_dataset(large_config, n_samples=80, seed=42)
        pyg = [graph_dict_to_pyg_pinn(g, large_config) for g in raw]
        train_d, val_d = pyg[:60], pyg[60:]

        model = ThermalGNN(node_in_dim=6, hidden_dim=48, n_layers=6)
        history = train_model_pinn(
            model,
            train_d,
            val_d,
            lambda_phys=0.1,
            epochs=80,
            lr=1e-3,
            verbose=False,
        )
        assert history["train_loss"][-1] < history["train_loss"][0]

    @pytest.mark.slow
    def test_large_mesh_physics_loss_decreases(self, large_config):
        """20×20メッシュで物理ロスが学習中に減少すること."""
        raw = generate_pinn_dataset(large_config, n_samples=80, seed=42)
        pyg = [graph_dict_to_pyg_pinn(g, large_config) for g in raw]
        train_d, val_d = pyg[:60], pyg[60:]

        model = ThermalGNN(node_in_dim=6, hidden_dim=48, n_layers=6)
        history = train_model_pinn(
            model,
            train_d,
            val_d,
            lambda_phys=0.5,
            epochs=80,
            lr=1e-3,
            verbose=False,
        )
        early_phys = np.mean(history["phys_loss"][:15])
        late_phys = np.mean(history["phys_loss"][-15:])
        assert late_phys < early_phys, (
            f"物理ロス減少なし: early={early_phys:.4f}, late={late_phys:.4f}"
        )

    @pytest.mark.slow
    def test_large_mesh_pinn_r2_positive(self, large_config):
        """20×20メッシュでPINN学習後にR²>0を達成."""
        raw = generate_pinn_dataset(large_config, n_samples=100, seed=42)
        pyg = [graph_dict_to_pyg_pinn(g, large_config) for g in raw]
        train_d, val_d, test_d = pyg[:70], pyg[70:85], pyg[85:]

        model = ThermalGNN(node_in_dim=6, hidden_dim=48, n_layers=6)
        history = train_model_pinn(
            model,
            train_d,
            val_d,
            lambda_phys=0.1,
            epochs=150,
            lr=1e-3,
            verbose=False,
        )
        metrics = evaluate_model(model, test_d, history["y_mean"], history["y_std"])
        assert metrics["r2"] > 0, f"PINN R²={metrics['r2']:.3f} (20×20 mesh)"


# =========================================================================
# 2. 不規則メッシュでのPINN検証
# =========================================================================


class TestPINNIrregularMesh:
    """不規則メッシュでのPINNデータ生成と学習の検証."""

    def test_irregular_k_matrix_shape(self):
        """不規則メッシュのK行列形状."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset_irregular(config, n_samples=1, perturbation=0.3, seed=42)
        n_nodes = (config.nx + 1) * (config.ny + 1)  # 36
        assert raw[0]["K_dense"].shape == (n_nodes, n_nodes)

    def test_irregular_k_matrix_symmetric(self):
        """不規則メッシュのK行列が対称."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset_irregular(config, n_samples=1, perturbation=0.3, seed=42)
        K = raw[0]["K_dense"]
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_irregular_k_matrix_positive_definite(self):
        """不規則メッシュのK行列が正定値."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset_irregular(config, n_samples=1, perturbation=0.3, seed=42)
        K = raw[0]["K_dense"]
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals > 0), f"最小固有値: {eigvals.min()}"

    def test_irregular_physics_exact(self):
        """不規則メッシュでFEM解が K@ΔT=f_shifted を満たすこと."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset_irregular(config, n_samples=3, perturbation=0.3, seed=42)
        for sample in raw:
            K = sample["K_dense"]
            f_shifted = sample["f_shifted"]
            dt_exact = sample["y"].flatten()
            residual = K @ dt_exact - f_shifted
            np.testing.assert_allclose(residual, 0, atol=1e-4)

    def test_irregular_physics_loss_zero_for_exact(self):
        """不規則メッシュの正解ΔTに対して物理ロスがゼロ."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset_irregular(config, n_samples=1, perturbation=0.3, seed=42)
        data = graph_dict_to_pyg_pinn(raw[0], config)
        dt_exact = data.y.squeeze(-1)
        loss = compute_physics_loss(dt_exact, data.K_dense, data.f_shifted)
        assert loss.item() < 1e-6, f"正解の残差: {loss.item()}"

    @pytest.mark.slow
    def test_irregular_pinn_training_converges(self):
        """不規則メッシュ上でPINN学習の損失が減少すること."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset_irregular(config, n_samples=50, perturbation=0.35, seed=42)
        pyg = [graph_dict_to_pyg_pinn(g, config) for g in raw]
        train_d, val_d = pyg[:35], pyg[35:]

        model = ThermalGNN(node_in_dim=6, hidden_dim=32, n_layers=4)
        history = train_model_pinn(
            model,
            train_d,
            val_d,
            lambda_phys=0.1,
            epochs=60,
            lr=1e-3,
            verbose=False,
        )
        assert history["train_loss"][-1] < history["train_loss"][0]

    @pytest.mark.slow
    def test_irregular_physics_loss_decreases(self):
        """不規則メッシュでPINN物理ロスが減少すること."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset_irregular(config, n_samples=50, perturbation=0.35, seed=42)
        pyg = [graph_dict_to_pyg_pinn(g, config) for g in raw]
        train_d, val_d = pyg[:35], pyg[35:]

        model = ThermalGNN(node_in_dim=6, hidden_dim=32, n_layers=4)
        history = train_model_pinn(
            model,
            train_d,
            val_d,
            lambda_phys=0.5,
            epochs=60,
            lr=1e-3,
            verbose=False,
        )
        early_phys = np.mean(history["phys_loss"][:10])
        late_phys = np.mean(history["phys_loss"][-10:])
        assert late_phys < early_phys


# =========================================================================
# 3. PINN vs data-only 比較（正則/不規則メッシュ）
# =========================================================================


class TestPINNvsDataOnly:
    """PINN学習とデータのみ学習の比較.

    正則メッシュ/不規則メッシュの両方でPINNの効果を検証する。
    """

    def _run_comparison(
        self,
        config: ThermalProblemConfig,
        raw_data: list[dict],
        n_train: int = 60,
        n_val: int = 15,
        hidden_dim: int = 32,
        n_layers: int = 4,
        epochs: int = 100,
        lambda_phys: float = 0.1,
    ) -> dict:
        """PINN vs data-only の比較を実行.

        Returns:
            {"pinn_r2": float, "data_only_r2": float,
             "pinn_history": dict, "data_only_history": dict}
        """
        pyg_pinn = [graph_dict_to_pyg_pinn(g, config) for g in raw_data]
        pyg_data = [graph_dict_to_pyg(g, config) for g in raw_data]

        # PINN学習
        torch.manual_seed(42)
        model_pinn = ThermalGNN(node_in_dim=6, hidden_dim=hidden_dim, n_layers=n_layers)
        h_pinn = train_model_pinn(
            model_pinn,
            pyg_pinn[:n_train],
            pyg_pinn[n_train : n_train + n_val],
            lambda_phys=lambda_phys,
            epochs=epochs,
            lr=1e-3,
            verbose=False,
        )
        m_pinn = evaluate_model(
            model_pinn,
            pyg_pinn[n_train + n_val :],
            h_pinn["y_mean"],
            h_pinn["y_std"],
        )

        # data-only学習
        torch.manual_seed(42)
        model_data = ThermalGNN(node_in_dim=6, hidden_dim=hidden_dim, n_layers=n_layers)
        h_data = train_model(
            model_data,
            pyg_data[:n_train],
            pyg_data[n_train : n_train + n_val],
            epochs=epochs,
            lr=1e-3,
            verbose=False,
        )
        m_data = evaluate_model(
            model_data,
            pyg_data[n_train + n_val :],
            h_data["y_mean"],
            h_data["y_std"],
        )

        return {
            "pinn_r2": m_pinn["r2"],
            "pinn_mae": m_pinn["mae"],
            "data_only_r2": m_data["r2"],
            "data_only_mae": m_data["mae"],
            "pinn_history": h_pinn,
            "data_only_history": h_data,
        }

    @pytest.mark.slow
    def test_pinn_learns_on_regular_mesh(self):
        """正則メッシュでPINNが学習できること（R² > -1）."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset(config, n_samples=80, seed=42)
        result = self._run_comparison(config, raw, n_train=55, n_val=10, epochs=80)
        assert result["pinn_r2"] > -1.0, f"PINN R²={result['pinn_r2']:.3f}"

    @pytest.mark.slow
    def test_pinn_learns_on_irregular_mesh(self):
        """不規則メッシュでPINNが学習できること（R² > -1）."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset_irregular(config, n_samples=80, perturbation=0.35, seed=42)
        result = self._run_comparison(config, raw, n_train=55, n_val=10, epochs=80)
        assert result["pinn_r2"] > -1.0, f"PINN R²={result['pinn_r2']:.3f}"

    @pytest.mark.slow
    def test_comparison_regular_mesh_report(self):
        """正則メッシュでのPINN vs data-only比較レポート."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset(config, n_samples=100, seed=42)
        result = self._run_comparison(
            config, raw, n_train=70, n_val=15, epochs=120, lambda_phys=0.1
        )

        print("\n=== 正則メッシュ: PINN vs data-only ===")
        print(f"  PINN:      R²={result['pinn_r2']:.3f}, MAE={result['pinn_mae']:.4f}")
        print(f"  Data-only: R²={result['data_only_r2']:.3f}, MAE={result['data_only_mae']:.4f}")
        diff = result["pinn_r2"] - result["data_only_r2"]
        print(f"  R²差（PINN - data-only）: {diff:+.3f}")

        # 両方が学習できていること
        assert result["pinn_r2"] > -1.0
        assert result["data_only_r2"] > -1.0

    @pytest.mark.slow
    def test_comparison_irregular_mesh_report(self):
        """不規則メッシュでのPINN vs data-only比較レポート."""
        config = ThermalProblemConfig(nx=5, ny=5)
        raw = generate_pinn_dataset_irregular(config, n_samples=100, perturbation=0.35, seed=42)
        result = self._run_comparison(
            config, raw, n_train=70, n_val=15, epochs=120, lambda_phys=0.1
        )

        print("\n=== 不規則メッシュ: PINN vs data-only ===")
        print(f"  PINN:      R²={result['pinn_r2']:.3f}, MAE={result['pinn_mae']:.4f}")
        print(f"  Data-only: R²={result['data_only_r2']:.3f}, MAE={result['data_only_mae']:.4f}")
        diff = result["pinn_r2"] - result["data_only_r2"]
        print(f"  R²差（PINN - data-only）: {diff:+.3f}")

        # 両方が学習できていること
        assert result["pinn_r2"] > -1.0
        assert result["data_only_r2"] > -1.0
