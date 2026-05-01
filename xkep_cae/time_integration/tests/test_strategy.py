"""TimeIntegrationStrategy 具象実装のテスト.

@binds_to による 1:1 紐付け + TimeIntegrationStrategy Protocol 適合テスト。
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.core.strategies import TimeIntegrationStrategy
from xkep_cae.core.testing import binds_to
from xkep_cae.time_integration import (
    ExplicitCentralDifferenceProcess,
    GeneralizedAlphaProcess,
    QuasiStaticProcess,
    TimeIntegrationInput,
    TimeIntegrationOutput,
)
from xkep_cae.time_integration.strategy import (
    _create_time_integration_strategy,
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
            (
                ExplicitCentralDifferenceProcess,
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
        s = _create_time_integration_strategy()
        assert isinstance(s, QuasiStaticProcess)

    def test_mass_no_dt_returns_quasi_static(self):
        M = sp.eye(4, format="csr")
        s = _create_time_integration_strategy(mass_matrix=M, dt_physical=0.0)
        assert isinstance(s, QuasiStaticProcess)

    def test_mass_with_dt_returns_dynamic(self):
        M = sp.eye(4, format="csr")
        s = _create_time_integration_strategy(mass_matrix=M, dt_physical=0.01)
        assert isinstance(s, GeneralizedAlphaProcess)

    def test_dynamic_with_initial_state(self):
        M = sp.eye(4, format="csr")
        v0 = np.ones(4)
        s = _create_time_integration_strategy(mass_matrix=M, dt_physical=0.01, velocity=v0)
        assert isinstance(s, GeneralizedAlphaProcess)
        np.testing.assert_array_equal(s.vel, v0)

    def test_dynamic_with_damping(self):
        M = sp.eye(4, format="csr")
        C = sp.eye(4, format="csr") * 0.1
        s = _create_time_integration_strategy(
            mass_matrix=M,
            damping_matrix=C,
            dt_physical=0.01,
        )
        assert isinstance(s, GeneralizedAlphaProcess)
        assert s.C is not None

    def test_explicit_mode_returns_explicit_process(self):
        M = sp.eye(4, format="csr")
        s = _create_time_integration_strategy(
            mass_matrix=M,
            dt_physical=0.01,
            solver_mode="explicit",
        )
        assert isinstance(s, ExplicitCentralDifferenceProcess)

    def test_explicit_mode_with_initial_state(self):
        M = sp.eye(4, format="csr")
        v0 = np.ones(4) * 0.3
        s = _create_time_integration_strategy(
            mass_matrix=M,
            dt_physical=0.01,
            solver_mode="explicit",
            velocity=v0,
        )
        assert isinstance(s, ExplicitCentralDifferenceProcess)
        np.testing.assert_array_equal(s.vel, v0)

    def test_implicit_mode_default(self):
        M = sp.eye(4, format="csr")
        s = _create_time_integration_strategy(mass_matrix=M, dt_physical=0.01)
        assert isinstance(s, GeneralizedAlphaProcess)


# ── ExplicitCentralDifference ─────────────────────────────


@binds_to(ExplicitCentralDifferenceProcess)
class TestExplicitCentralDifferenceProcess:
    """ExplicitCentralDifferenceProcess の単体テスト（status-377 Phase 1）."""

    @pytest.fixture()
    def proc6(self):
        """6DOF の中央差分インスタンス（質量 2.0 / DOF）."""
        M = sp.eye(6, format="csr") * 2.0
        return ExplicitCentralDifferenceProcess(mass_matrix=M)

    def test_is_dynamic(self, proc6):
        assert proc6.is_dynamic is True

    def test_lumped_mass_row_sum(self):
        M = sp.csr_matrix(np.array([[2.0, 1.0], [1.0, 2.0]]))
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M, mass_lumping="row_sum")
        np.testing.assert_array_almost_equal(proc.M_lump, np.array([3.0, 3.0]))

    def test_lumped_mass_diagonal(self):
        M = sp.csr_matrix(np.array([[2.0, 1.0], [1.0, 4.0]]))
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M, mass_lumping="diagonal")
        np.testing.assert_array_almost_equal(proc.M_lump, np.array([2.0, 4.0]))

    def test_lumped_mass_invalid_raises(self):
        M = sp.eye(3, format="csr")
        with pytest.raises(ValueError, match="unknown mass_lumping"):
            ExplicitCentralDifferenceProcess(mass_matrix=M, mass_lumping="bogus")

    def test_invert_diagonal_zero_safe(self):
        M = sp.csr_matrix(np.diag([2.0, 0.0, 4.0]))
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M, mass_lumping="diagonal")
        np.testing.assert_array_almost_equal(proc.M_lump_inv, np.array([0.5, 0.0, 0.25]))

    def test_critical_dt_basic(self, proc6):
        # ω_max = sqrt(λ_max), dt_c = 2/ω_max
        dt_c = proc6.critical_dt(k_max_eigenvalue=4.0)  # ω_max = 2 → dt_c = 1
        assert dt_c == pytest.approx(1.0)

    def test_critical_dt_zero_returns_inf(self, proc6):
        assert proc6.critical_dt(k_max_eigenvalue=0.0) == float("inf")

    def test_step_zero_dt_returns_copy(self, proc6):
        u = np.ones(6) * 0.5
        u_new = proc6.step(u, np.zeros(6), np.zeros(6), dt=0.0)
        np.testing.assert_array_equal(u_new, u)
        assert u_new is not u

    def test_step_unit_force_advances_velocity(self, proc6):
        # M_lump = 2.0, F_ext = 1.0 → a = 0.5, v_new = 0 + dt·0.5
        u = np.zeros(6)
        f_ext = np.ones(6)
        proc6.step(u, f_ext, np.zeros(6), dt=0.01)
        np.testing.assert_array_almost_equal(proc6.acc, np.ones(6) * 0.5)
        np.testing.assert_array_almost_equal(proc6.vel, np.ones(6) * 0.005)

    def test_step_fixed_dofs_zeroed(self, proc6):
        u = np.zeros(6)
        f_ext = np.ones(6)
        fixed = np.array([0, 3])
        proc6.step(u, f_ext, np.zeros(6), dt=0.01, fixed_dofs=fixed)
        assert proc6.acc[0] == 0.0
        assert proc6.acc[3] == 0.0
        assert proc6.vel[0] == 0.0
        assert proc6.vel[3] == 0.0
        # 自由 DOF は更新される
        assert proc6.acc[1] == pytest.approx(0.5)

    def test_sdof_free_vibration_period(self):
        """SDoF 自由振動: 1 周期後の戻り誤差検証.

        Symplectic Euler (semi-implicit) の位相誤差は O(dt·ω) per step。
        dt = T/100（100 step/period）で 1 周期戻り誤差 < 5%。
        """
        m = 1.0
        k = 100.0  # ω = 10, T ≈ 0.6283
        omega = float(np.sqrt(k / m))
        period = 2.0 * np.pi / omega
        dt = period / 100.0  # 100 step / period（CFL 比 dt/dt_c ≈ 0.16）

        M = sp.csr_matrix([[m]])
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M)
        proc.set_initial_state(velocity=np.array([0.0]), acceleration=np.array([0.0]))

        u = np.array([1.0])
        n_steps = int(round(period / dt))
        for _ in range(n_steps):
            f_int = k * u
            u = proc.step(u, np.zeros(1), f_int, dt=dt)

        assert abs(u[0] - 1.0) < 0.05

    def test_sdof_energy_bounded(self):
        """SDoF 自由振動: symplectic 性で E がドリフトせず有界."""
        m = 1.0
        k = 100.0
        omega = float(np.sqrt(k / m))
        period = 2.0 * np.pi / omega
        dt = period / 100.0

        M = sp.csr_matrix([[m]])
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M)
        proc.set_initial_state(velocity=np.array([0.0]))

        u = np.array([1.0])
        E0 = 0.5 * k * u[0] ** 2  # 初期ポテンシャル
        E_max = E0
        for _ in range(int(round(5.0 * period / dt))):  # 5 周期
            f_int = k * u
            u = proc.step(u, np.zeros(1), f_int, dt=dt)
            E = 0.5 * k * u[0] ** 2 + 0.5 * m * proc.vel[0] ** 2
            E_max = max(E_max, E)

        # symplectic Euler は long-term で energy が有界（10% 以内）
        assert E_max < 1.10 * E0

    def test_sdof_critical_dt_courant(self):
        """Courant 条件越えで発散することを確認."""
        m = 1.0
        k = 100.0
        M = sp.csr_matrix([[m]])
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M)
        # Courant 条件: dt_c = 2/ω = 2/10 = 0.2
        dt_c = proc.critical_dt(k_max_eigenvalue=k)
        assert dt_c == pytest.approx(0.2)

        dt_unstable = 1.5 * dt_c  # 明確に超過
        u = np.array([1.0])
        proc.set_initial_state(velocity=np.array([0.0]))
        for _ in range(20):
            f_int = k * u
            u = proc.step(u, np.zeros(1), f_int, dt=dt_unstable)

        # 不安定で振幅が爆発（初期 1.0 から発散）
        assert abs(u[0]) > 100.0

    def test_damping_term_in_step(self):
        """C·v が step() 内で残差から減算される."""
        M = sp.csr_matrix([[1.0]])
        C = sp.csr_matrix([[10.0]])
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M, damping_matrix=C)
        proc.set_initial_state(velocity=np.array([1.0]))

        # F_ext = 0, F_int = 0, C·v = 10 → a = -10
        proc.step(np.zeros(1), np.zeros(1), np.zeros(1), dt=0.01)
        assert proc.acc[0] == pytest.approx(-10.0)

    def test_set_initial_state(self, proc6):
        v0 = np.ones(6) * 0.5
        a0 = np.ones(6) * -0.1
        proc6.set_initial_state(velocity=v0, acceleration=a0)
        np.testing.assert_array_equal(proc6.vel, v0)
        np.testing.assert_array_equal(proc6.acc, a0)

    def test_checkpoint_restore(self, proc6):
        proc6.set_initial_state(velocity=np.ones(6) * 3.0)
        proc6.checkpoint()
        proc6.set_initial_state(velocity=np.zeros(6))
        proc6.restore_checkpoint()
        np.testing.assert_array_equal(proc6.vel, np.ones(6) * 3.0)

    def test_effective_stiffness_diagonal_mass(self, proc6):
        K = sp.eye(6, format="csr")  # 捨てられる
        K_eff = proc6.effective_stiffness(K, dt=0.01)
        # M_lump=2.0, dt²=1e-4 → diag = 2.0 / 1e-4 = 2e4
        diag = K_eff.diagonal()
        np.testing.assert_array_almost_equal(diag, np.ones(6) * 2.0e4)

    def test_effective_stiffness_small_dt_passthrough(self, proc6):
        K = sp.eye(6, format="csr")
        K_eff = proc6.effective_stiffness(K, dt=0.0)
        np.testing.assert_array_equal(K_eff.toarray(), K.toarray())

    def test_effective_residual_no_damping(self, proc6):
        R = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        R_eff = proc6.effective_residual(R, dt=0.01)
        np.testing.assert_array_equal(R_eff, R)

    def test_effective_residual_with_damping(self):
        M = sp.eye(2, format="csr")
        C = sp.eye(2, format="csr") * 0.5
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M, damping_matrix=C)
        proc.set_initial_state(velocity=np.array([2.0, -1.0]))
        R = np.array([1.0, 1.0])
        R_eff = proc.effective_residual(R, dt=0.01)
        # R_eff = R − C·v = (1, 1) − (1.0, −0.5) = (0, 1.5)
        np.testing.assert_array_almost_equal(R_eff, np.array([0.0, 1.5]))

    def test_dense_mass_matrix_converted(self):
        M_dense = np.eye(3) * 2.0
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M_dense)
        assert sp.issparse(proc.M)
        np.testing.assert_array_almost_equal(proc.M_lump, np.ones(3) * 2.0)

    def test_dense_damping_matrix_converted(self):
        M = sp.eye(3, format="csr")
        C_dense = np.eye(3) * 0.1
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M, damping_matrix=C_dense)
        assert sp.issparse(proc.C)

    def test_process_returns_output(self, proc6):
        inp = TimeIntegrationInput(
            u=np.ones(6),
            du=np.zeros(6),
            dt=0.01,
        )
        out = proc6.process(inp)
        assert isinstance(out, TimeIntegrationOutput)


# ── Mass scaling（status-379 Phase 3 候補 (h1)） ───────────────


class TestExplicitCentralDifferenceMassScaling:
    """質量スケーリング β² · M_lump（Belytschko §6.4.2）の単体テスト.

    `TestExplicitCentralDifferenceProcess` に `@binds_to` が付いているため
    本クラスは binds なしで Process メソッドを直接検証する。
    """

    def test_default_beta_one_unchanged(self):
        """デフォルト β=1.0 で集中質量は raw と一致."""
        M = sp.eye(4, format="csr") * 2.0
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M)
        np.testing.assert_array_almost_equal(proc.M_lump, np.ones(4) * 2.0)
        assert proc.mass_scaling_beta == 1.0

    def test_beta_squared_scales_lumped_mass(self):
        """β=10 で M_lump → 100·M_lump_raw."""
        M = sp.eye(3, format="csr") * 1.5
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M, mass_scaling_beta=10.0)
        np.testing.assert_array_almost_equal(proc.M_lump, np.ones(3) * 150.0)
        np.testing.assert_array_almost_equal(proc.M_lump_inv, np.ones(3) / 150.0)

    def test_beta_below_one_raises(self):
        """β<1 は Δt_c を縮小するだけなので拒否."""
        M = sp.eye(3, format="csr")
        with pytest.raises(ValueError, match="mass_scaling_beta must be >= 1.0"):
            ExplicitCentralDifferenceProcess(mass_matrix=M, mass_scaling_beta=0.5)

    def test_beta_scales_critical_dt(self):
        """β=4 で集中質量を 16 倍に増やすと、同じ K に対する critical_dt が 4 倍化."""
        M = sp.eye(2, format="csr")
        proc_raw = ExplicitCentralDifferenceProcess(mass_matrix=M)
        proc_scaled = ExplicitCentralDifferenceProcess(mass_matrix=M, mass_scaling_beta=4.0)
        K_max_raw = 100.0
        K_max_scaled = K_max_raw / proc_scaled.M_lump[0]
        assert proc_raw.critical_dt(K_max_raw) == pytest.approx(0.2)
        assert proc_scaled.critical_dt(K_max_scaled) == pytest.approx(0.8)

    def test_set_mass_scaling_beta_increases(self):
        """set_mass_scaling_beta() で β を上方更新できる."""
        M = sp.eye(3, format="csr") * 2.0
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M)
        proc.set_mass_scaling_beta(5.0)
        assert proc.mass_scaling_beta == 5.0
        np.testing.assert_array_almost_equal(proc.M_lump, np.ones(3) * 50.0)

    def test_set_mass_scaling_beta_monotone(self):
        """set_mass_scaling_beta() は単調増加のみ（既存 β 以下では何もしない）."""
        M = sp.eye(3, format="csr") * 2.0
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M, mass_scaling_beta=3.0)
        proc.set_mass_scaling_beta(2.0)
        assert proc.mass_scaling_beta == 3.0
        np.testing.assert_array_almost_equal(proc.M_lump, np.ones(3) * 18.0)

    def test_set_mass_scaling_beta_invalid_raises(self):
        """β<1 で set_mass_scaling_beta は ValueError."""
        M = sp.eye(2, format="csr")
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M)
        with pytest.raises(ValueError, match="mass_scaling_beta must be >= 1.0"):
            proc.set_mass_scaling_beta(0.5)

    def test_factory_passes_mass_scaling_beta(self):
        """_create_time_integration_strategy で β が伝搬する."""
        M = sp.eye(4, format="csr")
        s = _create_time_integration_strategy(
            mass_matrix=M,
            dt_physical=0.01,
            solver_mode="explicit",
            mass_scaling_beta=7.0,
        )
        assert isinstance(s, ExplicitCentralDifferenceProcess)
        assert s.mass_scaling_beta == 7.0
        np.testing.assert_array_almost_equal(s.M_lump, np.ones(4) * 49.0)

    def test_set_mass_scaling_beta_preserves_kinetic_energy(self):
        """status-381 h-bug-1 修正: β 上方更新で KE = 0.5·M·v² が保存される.

        β_old → β_new で M_lump が β_new²/β_old² 倍化する一方、v は
        (β_old/β_new) でスケールダウンされ、KE は不変。
        """
        M = sp.eye(3, format="csr") * 2.0
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M, mass_scaling_beta=2.0)
        # 初期 KE: M_lump=8.0, v=[1,1,1] → KE = 0.5·8·3 = 12
        proc.set_initial_state(velocity=np.ones(3))
        ke_before = 0.5 * float(np.sum(proc.M_lump * proc.vel * proc.vel))
        assert ke_before == pytest.approx(12.0)

        proc.set_mass_scaling_beta(10.0)
        ke_after = 0.5 * float(np.sum(proc.M_lump * proc.vel * proc.vel))
        # M_lump=200.0, v=[0.2,0.2,0.2] → KE = 0.5·200·3·0.04 = 12
        assert ke_after == pytest.approx(ke_before, rel=1e-10)
        np.testing.assert_array_almost_equal(proc.vel, np.ones(3) * 0.2)

    def test_set_mass_scaling_beta_rescales_acceleration(self):
        """status-381 h-bug-1 修正: a も (β_old/β_new)² で rescale される."""
        M = sp.eye(2, format="csr")
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M, mass_scaling_beta=3.0)
        proc.set_initial_state(acceleration=np.array([6.0, 9.0]))
        # β: 3 → 12 → ratio = 0.25, a *= 0.0625
        proc.set_mass_scaling_beta(12.0)
        np.testing.assert_array_almost_equal(proc.acc, np.array([6.0, 9.0]) * (3.0 / 12.0) ** 2)

    def test_set_mass_scaling_beta_no_rescale_opt_out(self):
        """rescale_state=False で v/a は変更されない（後方互換）."""
        M = sp.eye(2, format="csr")
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M)
        proc.set_initial_state(velocity=np.array([1.0, 2.0]), acceleration=np.array([3.0, 4.0]))
        proc.set_mass_scaling_beta(5.0, rescale_state=False)
        np.testing.assert_array_equal(proc.vel, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(proc.acc, np.array([3.0, 4.0]))

    def test_set_mass_scaling_beta_zero_velocity_unchanged(self):
        """v=0 / a=0 で β 更新しても状態は依然 0（rescale が安全）."""
        M = sp.eye(3, format="csr") * 2.0
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M)
        proc.set_mass_scaling_beta(50.0)
        np.testing.assert_array_equal(proc.vel, np.zeros(3))
        np.testing.assert_array_equal(proc.acc, np.zeros(3))

    # ── status-382 候補 (p3): mass-proportional Rayleigh damping ───────────

    def test_mass_proportional_damping_default_zero(self):
        """default で α=0、減衰は無効."""
        M = sp.eye(3, format="csr")
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M)
        assert proc.mass_proportional_damping_alpha == 0.0

    def test_mass_proportional_damping_negative_raises(self):
        """α < 0 は拒否（damping 注入は単調非負のみ）."""
        M = sp.eye(3, format="csr")
        with pytest.raises(ValueError, match="mass_proportional_damping_alpha must be >= 0"):
            ExplicitCentralDifferenceProcess(mass_matrix=M, mass_proportional_damping_alpha=-1.0)

    def test_mass_proportional_damping_decays_velocity(self):
        """α > 0 で外力ゼロ時に速度が減衰する.

        SDoF ξ = M·a + α·M·v = 0 → v_{n+1} = v_n · (1 - α·dt) （明示的減衰）.
        K=0、F_ext=0、dt 微小、α=2.0 で 1 step 後に v が概略 (1-α·dt) 倍。
        """
        M = sp.eye(2, format="csr") * 1.0
        proc = ExplicitCentralDifferenceProcess(mass_matrix=M, mass_proportional_damping_alpha=2.0)
        proc.set_initial_state(velocity=np.array([1.0, 1.0]))
        u = np.zeros(2)
        dt = 0.01
        # f_ext = f_int = 0 → a = -α·v = -2·v_old
        u_new = proc.step(u=u, f_ext=np.zeros(2), f_int=np.zeros(2), dt=dt)
        # v_new = v_old + dt·a = v_old·(1 - α·dt) = 1·(1 - 0.02) = 0.98
        np.testing.assert_array_almost_equal(proc.vel, np.array([0.98, 0.98]))
        # u_new = u + dt·v_new = 0 + 0.01·0.98 = 0.0098
        np.testing.assert_array_almost_equal(u_new, np.array([0.0098, 0.0098]))

    def test_mass_proportional_damping_independent_of_beta(self):
        """α は M スケーリングと独立に作用する（β 倍化しても damping rate 不変）."""
        M = sp.eye(2, format="csr") * 1.0
        # β=1
        proc1 = ExplicitCentralDifferenceProcess(
            mass_matrix=M,
            mass_scaling_beta=1.0,
            mass_proportional_damping_alpha=3.0,
        )
        proc1.set_initial_state(velocity=np.array([1.0, 1.0]))
        proc1.step(u=np.zeros(2), f_ext=np.zeros(2), f_int=np.zeros(2), dt=0.01)
        # β=10
        proc10 = ExplicitCentralDifferenceProcess(
            mass_matrix=M,
            mass_scaling_beta=10.0,
            mass_proportional_damping_alpha=3.0,
        )
        proc10.set_initial_state(velocity=np.array([1.0, 1.0]))
        proc10.step(u=np.zeros(2), f_ext=np.zeros(2), f_int=np.zeros(2), dt=0.01)
        # 両者の v は同じ（damping rate α は β 独立）
        np.testing.assert_array_almost_equal(proc1.vel, proc10.vel)

    def test_factory_passes_damping_alpha(self):
        """_create_time_integration_strategy で α が伝搬する."""
        M = sp.eye(3, format="csr")
        s = _create_time_integration_strategy(
            mass_matrix=M,
            dt_physical=0.01,
            solver_mode="explicit",
            mass_proportional_damping_alpha=5.5,
        )
        assert isinstance(s, ExplicitCentralDifferenceProcess)
        assert s.mass_proportional_damping_alpha == 5.5
