"""TimeIntegrationStrategy 具象実装のテスト.

@binds_to による 1:1 紐付け + TimeIntegrationStrategy Protocol 適合テスト。
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.core.strategies import TimeIntegrationStrategy
from xkep_cae.core.testing import binds_to
from xkep_cae.core.time_integration import (
    GeneralizedAlphaProcess,
    QuasiStaticProcess,
    TimeIntegrationInput,
    TimeIntegrationOutput,
    create_time_integration_strategy,
)

# ── Protocol 準拠 ─────────────────────────────────────────


class TestTimeIntegrationProtocolConformance:
    """全 TimeIntegration 具象が Protocol を満たすことを検証."""

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (QuasiStaticProcess, {}),
            (
                GeneralizedAlphaProcess,
                {"mass_matrix": sp.eye(6, format="csr")},
            ),
        ],
    )
    def test_protocol_conformance(self, cls, kwargs):
        instance = cls(**kwargs)
        assert isinstance(instance, TimeIntegrationStrategy)


# ── QuasiStatic ────────────────────────────────────────────


@binds_to(QuasiStaticProcess)
class TestQuasiStaticProcess:
    """QuasiStaticProcess の単体テスト."""

    def test_is_not_dynamic(self):
        proc = QuasiStaticProcess()
        assert proc.is_dynamic is False

    def test_vel_empty(self):
        proc = QuasiStaticProcess()
        assert len(proc.vel) == 0

    def test_acc_empty(self):
        proc = QuasiStaticProcess()
        assert len(proc.acc) == 0

    def test_predict_returns_copy(self):
        proc = QuasiStaticProcess()
        u = np.array([1.0, 2.0, 3.0])
        u_pred = proc.predict(u, dt=0.1)
        np.testing.assert_array_equal(u_pred, u)
        assert u_pred is not u

    def test_correct_noop(self):
        proc = QuasiStaticProcess()
        u = np.array([1.0, 2.0])
        proc.correct(u, np.zeros(2), dt=0.1)

    def test_effective_stiffness_identity(self):
        proc = QuasiStaticProcess()
        K = sp.eye(3, format="csr")
        K_eff = proc.effective_stiffness(K, dt=0.1)
        assert K_eff is K

    def test_effective_residual_identity(self):
        proc = QuasiStaticProcess()
        R = np.array([1.0, -2.0, 0.5])
        R_eff = proc.effective_residual(R, dt=0.1)
        np.testing.assert_array_equal(R_eff, R)

    def test_checkpoint_restore_noop(self):
        proc = QuasiStaticProcess()
        proc.checkpoint()
        proc.restore_checkpoint()

    def test_process_returns_output(self):
        proc = QuasiStaticProcess()
        inp = TimeIntegrationInput(
            u=np.array([1.0, 2.0]),
            du=np.zeros(2),
            dt=0.1,
        )
        out = proc.process(inp)
        assert isinstance(out, TimeIntegrationOutput)
        np.testing.assert_array_equal(out.u, inp.u)


# ── GeneralizedAlpha ───────────────────────────────────────


@binds_to(GeneralizedAlphaProcess)
class TestGeneralizedAlphaProcess:
    """GeneralizedAlphaProcess の単体テスト."""

    @pytest.fixture()
    def proc6(self):
        """6DOF の GeneralizedAlpha インスタンス."""
        M = sp.eye(6, format="csr") * 2.0
        return GeneralizedAlphaProcess(mass_matrix=M, rho_inf=0.9)

    def test_is_dynamic(self, proc6):
        assert proc6.is_dynamic is True

    def test_initial_state_zeros(self, proc6):
        np.testing.assert_array_equal(proc6.vel, np.zeros(6))
        np.testing.assert_array_equal(proc6.acc, np.zeros(6))

    def test_set_initial_state(self, proc6):
        v0 = np.ones(6) * 0.5
        a0 = np.ones(6) * -0.1
        proc6.set_initial_state(velocity=v0, acceleration=a0)
        np.testing.assert_array_equal(proc6.vel, v0)
        np.testing.assert_array_equal(proc6.acc, a0)

    def test_predict_with_zero_state(self, proc6):
        u = np.ones(6)
        u_pred = proc6.predict(u, dt=0.01)
        np.testing.assert_array_almost_equal(u_pred, u)

    def test_predict_with_velocity(self, proc6):
        proc6.set_initial_state(velocity=np.ones(6))
        u = np.zeros(6)
        dt = 0.01
        u_pred = proc6.predict(u, dt=dt)
        assert np.all(u_pred > 0)

    def test_correct_updates_velocity(self, proc6):
        u = np.zeros(6)
        proc6.predict(u, dt=0.01)
        u_new = np.ones(6) * 0.001
        proc6.correct(u_new, np.zeros(6), dt=0.01)
        assert np.any(proc6.vel != 0.0)

    def test_correct_small_dt_noop(self, proc6):
        vel_before = proc6.vel.copy()
        proc6.correct(np.zeros(6), np.zeros(6), dt=0.0)
        np.testing.assert_array_equal(proc6.vel, vel_before)

    def test_effective_stiffness_adds_mass(self, proc6):
        K = sp.eye(6, format="csr")
        K_eff = proc6.effective_stiffness(K, dt=0.01)
        assert K_eff.nnz >= K.nnz
        diag_K = K.diagonal()
        diag_Keff = K_eff.diagonal()
        assert np.all(diag_Keff > diag_K)

    def test_effective_stiffness_small_dt(self, proc6):
        K = sp.eye(6, format="csr")
        K_eff = proc6.effective_stiffness(K, dt=0.0)
        np.testing.assert_array_equal(K.toarray(), K_eff.toarray())

    def test_effective_stiffness_with_damping(self):
        M = sp.eye(4, format="csr")
        C = sp.eye(4, format="csr") * 0.1
        proc = GeneralizedAlphaProcess(mass_matrix=M, damping_matrix=C)
        K = sp.eye(4, format="csr")
        K_eff = proc.effective_stiffness(K, dt=0.01)
        diag = K_eff.diagonal()
        assert np.all(diag > 1.0)

    def test_effective_residual_inertia(self, proc6):
        R = np.zeros(6)
        proc6.predict(np.zeros(6), dt=0.01)
        R_eff = proc6.effective_residual(R, dt=0.01)
        np.testing.assert_array_almost_equal(R_eff, R)

    def test_effective_residual_with_damping(self):
        M = sp.eye(4, format="csr")
        C = sp.eye(4, format="csr") * 0.5
        proc = GeneralizedAlphaProcess(mass_matrix=M, damping_matrix=C)
        proc.set_initial_state(velocity=np.ones(4))
        proc.predict(np.zeros(4), dt=0.01)
        R = np.zeros(4)
        R_eff = proc.effective_residual(R, dt=0.01)
        assert np.any(R_eff != 0.0)

    def test_checkpoint_restore(self, proc6):
        proc6.set_initial_state(velocity=np.ones(6) * 3.0)
        proc6.checkpoint()
        proc6.set_initial_state(velocity=np.zeros(6))
        proc6.restore_checkpoint()
        np.testing.assert_array_equal(proc6.vel, np.ones(6) * 3.0)

    def test_process_returns_output(self, proc6):
        inp = TimeIntegrationInput(
            u=np.ones(6),
            du=np.zeros(6),
            dt=0.01,
        )
        out = proc6.process(inp)
        assert isinstance(out, TimeIntegrationOutput)

    def test_chung_hulbert_params_rho0(self):
        """rho_inf=0 で最大数値減衰."""
        M = sp.eye(2, format="csr")
        proc = GeneralizedAlphaProcess(mass_matrix=M, rho_inf=0.0)
        assert proc.alpha_m == pytest.approx(-1.0)
        assert proc.alpha_f == pytest.approx(0.0)

    def test_chung_hulbert_params_rho1(self):
        """rho_inf=1 で Newmark 平均加速度法."""
        M = sp.eye(2, format="csr")
        proc = GeneralizedAlphaProcess(mass_matrix=M, rho_inf=1.0)
        assert proc.alpha_m == pytest.approx(0.5)
        assert proc.alpha_f == pytest.approx(0.5)
        assert proc.gamma == pytest.approx(0.5)
        assert proc.beta == pytest.approx(0.25)

    def test_dense_mass_matrix_converted(self):
        """np.ndarray 質量行列が CSR に変換される."""
        M_dense = np.eye(3)
        proc = GeneralizedAlphaProcess(mass_matrix=M_dense)
        assert sp.issparse(proc.M)

    def test_dense_damping_matrix_converted(self):
        """np.ndarray 減衰行列が CSR に変換される."""
        M = sp.eye(3, format="csr")
        C_dense = np.eye(3) * 0.1
        proc = GeneralizedAlphaProcess(mass_matrix=M, damping_matrix=C_dense)
        assert sp.issparse(proc.C)


# ── ファクトリ ─────────────────────────────────────────────


class TestCreateTimeIntegrationStrategy:
    """ファクトリ関数のテスト."""

    def test_no_mass_returns_quasi_static(self):
        s = create_time_integration_strategy()
        assert isinstance(s, QuasiStaticProcess)

    def test_mass_no_dt_returns_quasi_static(self):
        M = sp.eye(4, format="csr")
        s = create_time_integration_strategy(mass_matrix=M, dt_physical=0.0)
        assert isinstance(s, QuasiStaticProcess)

    def test_mass_with_dt_returns_dynamic(self):
        M = sp.eye(4, format="csr")
        s = create_time_integration_strategy(mass_matrix=M, dt_physical=0.01)
        assert isinstance(s, GeneralizedAlphaProcess)

    def test_dynamic_with_initial_state(self):
        M = sp.eye(4, format="csr")
        v0 = np.ones(4)
        s = create_time_integration_strategy(mass_matrix=M, dt_physical=0.01, velocity=v0)
        assert isinstance(s, GeneralizedAlphaProcess)
        np.testing.assert_array_equal(s.vel, v0)

    def test_dynamic_with_damping(self):
        M = sp.eye(4, format="csr")
        C = sp.eye(4, format="csr") * 0.1
        s = create_time_integration_strategy(
            mass_matrix=M,
            damping_matrix=C,
            dt_physical=0.01,
        )
        assert isinstance(s, GeneralizedAlphaProcess)
        assert s.C is not None
