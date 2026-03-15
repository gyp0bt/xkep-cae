"""TimeIntegration Strategy 具象実装の1:1テスト."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae_deprecated.process.strategies.protocols import TimeIntegrationStrategy
from xkep_cae_deprecated.process.strategies.time_integration import (
    GeneralizedAlphaProcess,
    QuasiStaticProcess,
    TimeIntegrationInput,
    create_time_integration_strategy,
)
from xkep_cae_deprecated.process.testing import binds_to

# --- Protocol 準拠チェック ---


class TestTimeIntegrationProtocolConformance:
    """全 TimeIntegration 具象が Protocol を満たすことを検証."""

    def test_quasi_static_conformance(self):
        instance = QuasiStaticProcess()
        assert isinstance(instance, TimeIntegrationStrategy)

    def test_generalized_alpha_conformance(self):
        M = sp.eye(6, format="csr")
        instance = GeneralizedAlphaProcess(mass_matrix=M)
        assert isinstance(instance, TimeIntegrationStrategy)


# --- QuasiStatic ---


@binds_to(QuasiStaticProcess)
class TestQuasiStaticProcess:
    """QuasiStaticProcess の単体テスト."""

    def test_predict_returns_copy(self):
        proc = QuasiStaticProcess()
        u = np.array([1.0, 2.0, 3.0])
        u_pred = proc.predict(u, dt=0.1)
        np.testing.assert_array_equal(u_pred, u)
        u_pred[0] = 999.0
        assert u[0] == 1.0  # 元の配列は変更されない

    def test_correct_is_noop(self):
        proc = QuasiStaticProcess()
        u = np.array([1.0, 2.0])
        du = np.array([0.5, 0.5])
        proc.correct(u, du, dt=0.1)  # 例外が出ないこと

    def test_effective_stiffness_passthrough(self):
        proc = QuasiStaticProcess()
        K = sp.eye(3, format="csr") * 100.0
        K_eff = proc.effective_stiffness(K, dt=0.1)
        assert K_eff is K

    def test_effective_residual_passthrough(self):
        proc = QuasiStaticProcess()
        R = np.array([1.0, -2.0, 3.0])
        R_eff = proc.effective_residual(R, dt=0.1)
        assert R_eff is R

    def test_process_method(self):
        proc = QuasiStaticProcess()
        inp = TimeIntegrationInput(u=np.zeros(3), du=np.zeros(3), dt=0.1)
        out = proc.process(inp)
        assert out.u.shape == (3,)

    def test_meta(self):
        assert QuasiStaticProcess.meta.name == "QuasiStatic"
        assert not QuasiStaticProcess.meta.deprecated


# --- GeneralizedAlpha ---


@binds_to(GeneralizedAlphaProcess)
class TestGeneralizedAlphaProcess:
    """GeneralizedAlphaProcess の単体テスト."""

    @pytest.fixture()
    def simple_system(self):
        """2DOF バネ-質量系."""
        M = sp.diags([1.0, 1.0], format="csr")
        K = sp.diags([100.0, 100.0], format="csr")
        return M, K

    def test_parameters_rho_inf_1(self):
        """rho_inf=1.0 → Newmark 平均加速度法."""
        M = sp.eye(2, format="csr")
        proc = GeneralizedAlphaProcess(mass_matrix=M, rho_inf=1.0)
        assert proc.alpha_m == pytest.approx(0.5)
        assert proc.alpha_f == pytest.approx(0.5)
        assert proc.gamma == pytest.approx(0.5)
        assert proc.beta == pytest.approx(0.25)

    def test_parameters_rho_inf_0(self):
        """rho_inf=0.0 → 最大数値減衰."""
        M = sp.eye(2, format="csr")
        proc = GeneralizedAlphaProcess(mass_matrix=M, rho_inf=0.0)
        assert proc.alpha_m == pytest.approx(-1.0)
        assert proc.alpha_f == pytest.approx(0.0)
        assert proc.gamma == pytest.approx(1.5)
        assert proc.beta == pytest.approx(1.0)

    def test_predict_at_rest(self):
        """静止状態からの予測子はゼロ変位."""
        M = sp.eye(2, format="csr")
        proc = GeneralizedAlphaProcess(mass_matrix=M)
        u = np.zeros(2)
        u_pred = proc.predict(u, dt=0.01)
        np.testing.assert_array_almost_equal(u_pred, np.zeros(2))

    def test_predict_with_velocity(self):
        """初速ありの予測子: u_pred ≈ dt*v (β小のとき)."""
        M = sp.eye(2, format="csr")
        proc = GeneralizedAlphaProcess(mass_matrix=M, rho_inf=1.0)
        proc.set_initial_state(velocity=np.array([1.0, 0.0]))
        u = np.zeros(2)
        dt = 0.01
        u_pred = proc.predict(u, dt=dt)
        # u_pred = dt*v + 0.5*dt^2*(1-2*0.25)*acc = 0.01 + 0 = 0.01
        assert u_pred[0] == pytest.approx(dt * 1.0, rel=1e-10)
        assert u_pred[1] == pytest.approx(0.0, abs=1e-15)

    def test_effective_stiffness_adds_mass(self, simple_system):
        """有効剛性にMassが加わる."""
        M, K = simple_system
        proc = GeneralizedAlphaProcess(mass_matrix=M, rho_inf=1.0)
        dt = 0.01
        K_eff = proc.effective_stiffness(K, dt=dt)
        # c0 = 1/(beta*dt^2) = 1/(0.25*0.0001) = 40000
        # (1-alpha_m)*c0 = 0.5*40000 = 20000
        diag_eff = K_eff.diagonal()
        expected = 100.0 + 0.5 * (1.0 / (0.25 * dt**2))
        np.testing.assert_array_almost_equal(diag_eff, [expected, expected])

    def test_effective_stiffness_with_damping(self, simple_system):
        """減衰行列もK_effに加算される."""
        M, K = simple_system
        C = sp.diags([10.0, 10.0], format="csr")
        proc = GeneralizedAlphaProcess(mass_matrix=M, damping_matrix=C, rho_inf=1.0)
        dt = 0.01
        K_eff = proc.effective_stiffness(K, dt=dt)
        c0 = 1.0 / (0.25 * dt**2)
        c1 = 0.5 / (0.25 * dt)
        expected = 100.0 + 0.5 * c0 * 1.0 + 0.5 * c1 * 10.0
        np.testing.assert_array_almost_equal(K_eff.diagonal(), [expected, expected])

    def test_correct_updates_vel_acc(self):
        """correct() で速度・加速度が更新される."""
        M = sp.eye(2, format="csr")
        proc = GeneralizedAlphaProcess(mass_matrix=M, rho_inf=1.0)
        u = np.zeros(2)
        dt = 0.01
        proc.predict(u, dt=dt)  # 内部状態を準備
        u_new = np.array([0.001, 0.0])
        proc.correct(u_new, np.zeros(2), dt=dt)
        assert proc.acc[0] != 0.0
        assert proc.vel[0] != 0.0

    def test_effective_residual_zero_motion(self):
        """静止状態では慣性力・減衰力ゼロ."""
        M = sp.eye(2, format="csr")
        proc = GeneralizedAlphaProcess(mass_matrix=M)
        R = np.array([1.0, -1.0])
        R_eff = proc.effective_residual(R, dt=0.01)
        np.testing.assert_array_almost_equal(R_eff, R)

    def test_tiny_dt_passthrough(self):
        """dt→0 では K_eff = K."""
        M = sp.eye(2, format="csr")
        K = sp.eye(2, format="csr") * 100.0
        proc = GeneralizedAlphaProcess(mass_matrix=M)
        K_eff = proc.effective_stiffness(K, dt=0.0)
        np.testing.assert_array_almost_equal(K_eff.toarray(), K.toarray())

    def test_process_method(self, simple_system):
        M, K = simple_system
        proc = GeneralizedAlphaProcess(mass_matrix=M)
        inp = TimeIntegrationInput(u=np.zeros(2), du=np.zeros(2), dt=0.01)
        out = proc.process(inp)
        assert out.u.shape == (2,)

    def test_meta(self):
        assert GeneralizedAlphaProcess.meta.name == "GeneralizedAlpha"
        assert not GeneralizedAlphaProcess.meta.deprecated

    def test_set_initial_state(self):
        M = sp.eye(3, format="csr")
        proc = GeneralizedAlphaProcess(mass_matrix=M)
        v0 = np.array([1.0, 2.0, 3.0])
        a0 = np.array([0.1, 0.2, 0.3])
        proc.set_initial_state(velocity=v0, acceleration=a0)
        np.testing.assert_array_equal(proc.vel, v0)
        np.testing.assert_array_equal(proc.acc, a0)
        # 元配列を変更しても proc の状態は変わらない
        v0[0] = 999.0
        assert proc.vel[0] == 1.0


# --- create_time_integration_strategy ファクトリ ---


class TestCreateTimeIntegrationStrategy:
    """create_time_integration_strategy ファクトリのテスト."""

    def test_quasi_static_default(self):
        """mass_matrix=None → QuasiStaticProcess."""
        strategy = create_time_integration_strategy()
        assert isinstance(strategy, QuasiStaticProcess)

    def test_quasi_static_no_dt(self):
        """dt_physical=0 → QuasiStaticProcess."""
        M = sp.eye(3, format="csr")
        strategy = create_time_integration_strategy(mass_matrix=M, dt_physical=0.0)
        assert isinstance(strategy, QuasiStaticProcess)

    def test_dynamic(self):
        """mass_matrix + dt_physical > 0 → GeneralizedAlphaProcess."""
        M = sp.eye(3, format="csr")
        strategy = create_time_integration_strategy(mass_matrix=M, dt_physical=0.01)
        assert isinstance(strategy, GeneralizedAlphaProcess)

    def test_dynamic_rho_inf(self):
        """rho_inf パラメータの伝播."""
        M = sp.eye(3, format="csr")
        strategy = create_time_integration_strategy(mass_matrix=M, dt_physical=0.01, rho_inf=1.0)
        assert isinstance(strategy, GeneralizedAlphaProcess)
        assert strategy.gamma == pytest.approx(0.5)
        assert strategy.beta == pytest.approx(0.25)

    def test_dynamic_with_initial_state(self):
        """初速・初期加速度の設定."""
        M = sp.eye(3, format="csr")
        v0 = np.array([1.0, 2.0, 3.0])
        a0 = np.array([0.1, 0.2, 0.3])
        strategy = create_time_integration_strategy(
            mass_matrix=M, dt_physical=0.01, velocity=v0, acceleration=a0
        )
        assert isinstance(strategy, GeneralizedAlphaProcess)
        np.testing.assert_array_equal(strategy.vel, v0)
        np.testing.assert_array_equal(strategy.acc, a0)

    def test_dynamic_with_damping(self):
        """減衰行列の設定."""
        M = sp.eye(3, format="csr")
        C = sp.eye(3, format="csr") * 10.0
        strategy = create_time_integration_strategy(
            mass_matrix=M, damping_matrix=C, dt_physical=0.01
        )
        assert isinstance(strategy, GeneralizedAlphaProcess)
        assert strategy.C is not None
