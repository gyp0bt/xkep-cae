"""動的解析+幾何学非線形の複合物理テスト.

test_dynamics_physics.py / test_geometric_nonlinear_physics.py を補完し、
動的・非線形が組み合わさった場合の物理的正しさを検証する。

テスト構成:
- TestDynamicLoadOrderPhysics: 動的荷重の反力・変位オーダー妥当性
- TestDynamicCRStressPhysics: CR動的解析後の応力時刻歴の物理性
- TestDynamicDisplacementPhysics: 変位時刻歴の物理的性質
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh
from xkep_cae.dynamics import (
    NonlinearTransientConfig,
    solve_nonlinear_transient,
)
from xkep_cae.elements.beam_timo3d import (
    assemble_cr_beam3d,
    timo_beam3d_ke_global,
)

pytestmark = pytest.mark.slow


# ====================================================================
# ヘルパー
# ====================================================================


def _build_cantilever(n_elems=10, L=0.5, E=2.1e11, nu=0.3, rho=7800.0, r=0.005):
    """カンチレバー梁の構築."""
    G = E / (2.0 * (1.0 + nu))
    A = np.pi * r**2
    Iy = np.pi * r**4 / 4.0
    Iz = Iy
    J = 2.0 * Iy
    kappa = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)

    nodes = np.zeros((n_elems + 1, 3))
    nodes[:, 0] = np.linspace(0, L, n_elems + 1)
    conn = np.array([[i, i + 1] for i in range(n_elems)])

    n_nodes = n_elems + 1
    ndof = 6 * n_nodes

    K = np.zeros((ndof, ndof))
    for e in range(n_elems):
        n1, n2 = conn[e]
        coords = nodes[np.array([n1, n2])]
        ke = timo_beam3d_ke_global(coords, E, G, A, Iy, Iz, J, kappa, kappa)
        edofs = np.array([6 * n1 + d for d in range(6)] + [6 * n2 + d for d in range(6)])
        K[np.ix_(edofs, edofs)] += ke

    Le = L / n_elems
    M = np.zeros((ndof, ndof))
    for i in range(n_nodes):
        m_i = rho * A * Le * (0.5 if (i == 0 or i == n_elems) else 1.0)
        for d in range(3):
            M[6 * i + d, 6 * i + d] = m_i
        I_rot = m_i * r**2 / 2.0
        for d in range(3, 6):
            M[6 * i + d, 6 * i + d] = I_rot

    fixed_dofs = np.arange(6)

    return {
        "nodes": nodes,
        "conn": conn,
        "K": K,
        "M": M,
        "E": E,
        "G": G,
        "A": A,
        "Iy": Iy,
        "Iz": Iz,
        "J": J,
        "kappa_y": kappa,
        "kappa_z": kappa,
        "rho": rho,
        "r": r,
        "L": L,
        "fixed_dofs": fixed_dofs,
        "ndof": ndof,
        "n_nodes": n_nodes,
        "n_elems": n_elems,
    }


def _estimate_period(data):
    """第1固有周期を推定する."""
    fixed = data["fixed_dofs"]
    ndof = data["ndof"]
    free = np.setdiff1d(np.arange(ndof), fixed)
    K_ff = data["K"][np.ix_(free, free)]
    M_ff = data["M"][np.ix_(free, free)]
    eigvals, _ = eigh(K_ff, M_ff)
    omega1 = np.sqrt(max(eigvals[0], 0.0))
    return 2.0 * np.pi / omega1 if omega1 > 0 else 1.0


def _make_cr_assemblers(data):
    """CR-Timo用コールバック."""
    nodes, conn = data["nodes"], data["conn"]
    E, G = data["E"], data["G"]
    A, Iy, Iz, J = data["A"], data["Iy"], data["Iz"], data["J"]
    ky, kz = data["kappa_y"], data["kappa_z"]

    def f_int_fn(u):
        _, f = assemble_cr_beam3d(
            nodes,
            conn,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            ky,
            kz,
            stiffness=False,
            internal_force=True,
            sparse=False,
        )
        return f

    def K_T_fn(u):
        K, _ = assemble_cr_beam3d(
            nodes,
            conn,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            ky,
            kz,
            stiffness=True,
            internal_force=False,
            sparse=False,
        )
        return K

    return f_int_fn, K_T_fn


def _make_linear_assemblers(K):
    """線形梁用コールバック."""
    return lambda u: K @ u, lambda u: K


# ====================================================================
# TestDynamicLoadOrderPhysics: 荷重オーダーの妥当性
# ====================================================================


class TestDynamicLoadOrderPhysics:
    """動的荷重に対する反力・変位のオーダーが物理的に妥当."""

    def test_static_load_converges_to_static_solution(self):
        """ゆっくりランプ荷重 → 静的解に収束.

        慣性力が無視できるほどゆっくり荷重をかければ、
        最終変位は静的解と一致するべき。
        """
        data = _build_cantilever(n_elems=8, L=0.5, r=0.005)
        ndof = data["ndof"]
        fixed = data["fixed_dofs"]
        tip_dof = 6 * data["n_elems"] + 1
        P = 1.0

        # 静的解
        free = np.setdiff1d(np.arange(ndof), fixed)
        f_static = np.zeros(ndof)
        f_static[tip_dof] = P
        u_static = np.zeros(ndof)
        u_static[free] = np.linalg.solve(data["K"][np.ix_(free, free)], f_static[free])
        delta_static = u_static[tip_dof]

        # 動的解（HHT-αで数値減衰あり、長時間ランプ）
        T1 = _estimate_period(data)
        ramp_time = 20.0 * T1  # 十分にゆっくり

        def get_force(t):
            return f_static * min(t / ramp_time, 1.0)

        # Rayleigh減衰を追加して振動を減衰させる
        alpha_R = 0.1 * np.sqrt(data["K"][tip_dof, tip_dof] / data["M"][tip_dof, tip_dof])
        C = alpha_R * data["M"]

        f_int_fn, K_T_fn = _make_linear_assemblers(data["K"])
        dt = T1 / 20.0
        n_steps = int(25.0 * T1 / dt)

        alpha_hht = -0.1
        gamma_hht = 0.5 * (1.0 - 2.0 * alpha_hht)
        beta_hht = 0.25 * (1.0 - alpha_hht) ** 2

        cfg = NonlinearTransientConfig(
            dt=dt,
            n_steps=n_steps,
            tol_force=1e-12,
            alpha_hht=alpha_hht,
            gamma=gamma_hht,
            beta=beta_hht,
        )
        res = solve_nonlinear_transient(
            data["M"],
            get_force,
            np.zeros(ndof),
            np.zeros(ndof),
            cfg,
            f_int_fn,
            K_T_fn,
            C=C,
            fixed_dofs=fixed,
            show_progress=False,
        )
        assert res.converged

        # 最終変位が静的解に近い（減衰で振動消えた後）
        delta_dynamic = res.displacement[-1, tip_dof]
        error = abs(delta_dynamic - delta_static) / abs(delta_static)
        assert error < 0.15, (
            f"動的解が静的解に収束していない: "
            f"δ_static={delta_static:.6e}, δ_dynamic={delta_dynamic:.6e}, error={error:.4f}"
        )

    def test_impulse_response_decays_with_damping(self):
        """減衰あり衝撃荷重 → 振幅が時間とともに減少."""
        data = _build_cantilever(n_elems=8, L=0.5, r=0.005)
        ndof = data["ndof"]
        fixed = data["fixed_dofs"]
        tip_dof = 6 * data["n_elems"] + 1

        # 初速を与えて自由振動
        v0 = np.zeros(ndof)
        v0[tip_dof] = 0.1  # 先端に初速

        # Rayleigh減衰
        T1 = _estimate_period(data)
        omega1 = 2.0 * np.pi / T1
        zeta = 0.05  # 5%減衰
        alpha_R = 2.0 * zeta * omega1
        C = alpha_R * data["M"]

        f_int_fn, K_T_fn = _make_linear_assemblers(data["K"])
        dt = T1 / 30.0
        n_steps = int(15 * T1 / dt)

        cfg = NonlinearTransientConfig(dt=dt, n_steps=n_steps, tol_force=1e-12)
        res = solve_nonlinear_transient(
            data["M"],
            np.zeros(ndof),
            np.zeros(ndof),
            v0,
            cfg,
            f_int_fn,
            K_T_fn,
            C=C,
            fixed_dofs=fixed,
            show_progress=False,
        )
        assert res.converged

        # 先端変位の振幅が減少していることを確認
        tip_disp = res.displacement[:, tip_dof]
        first_quarter = int(n_steps // 4)
        last_quarter = int(3 * n_steps // 4)

        amp_first = np.max(np.abs(tip_disp[:first_quarter]))
        amp_last = np.max(np.abs(tip_disp[last_quarter:]))

        assert amp_last < amp_first, (
            f"減衰で振幅が減少していない: amp_first={amp_first:.6e}, amp_last={amp_last:.6e}"
        )


# ====================================================================
# TestDynamicDisplacementPhysics: 変位時刻歴の物理性
# ====================================================================


class TestDynamicDisplacementPhysics:
    """変位の時刻歴が物理的に妥当であることを検証."""

    def test_free_vibration_oscillatory(self):
        """非減衰自由振動: 先端変位が振動的（符号が反転する）."""
        data = _build_cantilever(n_elems=8, L=0.5, r=0.005)
        ndof = data["ndof"]
        fixed = data["fixed_dofs"]
        tip_dof = 6 * data["n_elems"] + 1

        # 初期変位（静的たわみ）で自由振動
        free = np.setdiff1d(np.arange(ndof), fixed)
        f_static = np.zeros(ndof)
        f_static[tip_dof] = -1.0
        u0 = np.zeros(ndof)
        u0[free] = np.linalg.solve(data["K"][np.ix_(free, free)], f_static[free])

        f_int_fn, K_T_fn = _make_linear_assemblers(data["K"])
        T1 = _estimate_period(data)
        dt = T1 / 40.0
        n_steps = int(5 * T1 / dt)

        cfg = NonlinearTransientConfig(dt=dt, n_steps=n_steps, tol_force=1e-12)
        res = solve_nonlinear_transient(
            data["M"],
            np.zeros(ndof),
            u0,
            np.zeros(ndof),
            cfg,
            f_int_fn,
            K_T_fn,
            fixed_dofs=fixed,
            show_progress=False,
        )
        assert res.converged

        tip_disp = res.displacement[:, tip_dof]
        # 符号の反転を確認（振動的であること）
        sign_changes = np.sum(np.diff(np.sign(tip_disp)) != 0)
        assert sign_changes >= 4, f"振動的でない: 符号変化回数={sign_changes}, 期待>=4 (5周期分)"

    def test_cr_nonlinear_dynamic_bounded(self):
        """CR非線形動的解析: 変位が発散しない（有界性）."""
        data = _build_cantilever(n_elems=10, L=0.5, r=0.005)
        ndof = data["ndof"]
        fixed = data["fixed_dofs"]
        tip_dof = 6 * data["n_elems"] + 1

        # 正弦波荷重
        T1 = _estimate_period(data)
        omega = 2.0 * np.pi / T1
        P = 0.5  # 適度な荷重

        f_base = np.zeros(ndof)
        f_base[tip_dof] = P

        def get_force(t):
            return f_base * np.sin(omega * t)

        # 軽い減衰で安定性確保
        alpha_R = 0.01 * omega
        C = alpha_R * data["M"]

        f_int_fn, K_T_fn = _make_cr_assemblers(data)
        dt = T1 / 30.0
        n_steps = int(3 * T1 / dt)

        cfg = NonlinearTransientConfig(dt=dt, n_steps=n_steps, tol_force=1e-6)
        res = solve_nonlinear_transient(
            data["M"],
            get_force,
            np.zeros(ndof),
            np.zeros(ndof),
            cfg,
            f_int_fn,
            K_T_fn,
            C=C,
            fixed_dofs=fixed,
            show_progress=False,
        )
        assert res.converged, f"非収束: step={res.failed_step}"

        # 変位が有界であること（梁の長さの数倍を超えないこと）
        max_disp = np.max(np.abs(res.displacement))
        assert max_disp < 10.0 * data["L"], f"変位が発散: max_disp={max_disp:.6e}, L={data['L']}"

    def test_velocity_consistent_with_displacement(self):
        """速度が変位の時間微分と整合.

        v ≈ (u_{n+1} - u_{n-1}) / (2Δt) （中心差分近似）
        """
        data = _build_cantilever(n_elems=8, L=0.5, r=0.005)
        ndof = data["ndof"]
        fixed = data["fixed_dofs"]
        tip_dof = 6 * data["n_elems"] + 1

        # 初期変位で自由振動
        free = np.setdiff1d(np.arange(ndof), fixed)
        f_static = np.zeros(ndof)
        f_static[tip_dof] = -1.0
        u0 = np.zeros(ndof)
        u0[free] = np.linalg.solve(data["K"][np.ix_(free, free)], f_static[free])

        f_int_fn, K_T_fn = _make_linear_assemblers(data["K"])
        T1 = _estimate_period(data)
        dt = T1 / 80.0  # 細かいΔtで精度確保
        n_steps = int(2 * T1 / dt)

        cfg = NonlinearTransientConfig(dt=dt, n_steps=n_steps, tol_force=1e-12)
        res = solve_nonlinear_transient(
            data["M"],
            np.zeros(ndof),
            u0,
            np.zeros(ndof),
            cfg,
            f_int_fn,
            K_T_fn,
            fixed_dofs=fixed,
            show_progress=False,
        )
        assert res.converged

        # 先端DOFの速度を中心差分で近似
        u = res.displacement[:, tip_dof]
        v_newmark = res.velocity[:, tip_dof]

        # 中心差分近似（内部点のみ）
        v_fd = np.zeros(len(u))
        for i in range(1, len(u) - 1):
            v_fd[i] = (u[i + 1] - u[i - 1]) / (2.0 * dt)

        # 中間部分で比較（最初と最後は境界効果）
        mid_start = len(u) // 4
        mid_end = 3 * len(u) // 4
        v_nm_mid = v_newmark[mid_start:mid_end]
        v_fd_mid = v_fd[mid_start:mid_end]

        max_v = np.max(np.abs(v_nm_mid))
        if max_v > 1e-15:
            error = np.max(np.abs(v_nm_mid - v_fd_mid)) / max_v
            assert error < 0.05, f"速度が変位の時間微分と不整合: error={error:.4f}"
