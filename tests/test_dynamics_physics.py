"""動的解析の物理テスト — 梁要素の非線形動解析の物理的妥当性を検証.

プログラムテストではなく「物理的に当然の性質」をコード化する。

テスト構成:
- TestDynamicsEnergyPhysics: エネルギー保存（多自由度系、非線形梁）
- TestDynamicsLargeRotationPhysics: 大回転動的挙動の妥当性
- TestDynamicsSymmetryPhysics: 動的応答の対称性
- TestDynamicsFrequencyPhysics: 固有周波数の精度
- TestDynamicsStabilityPhysics: 時間積分の安定性
- TestCRvsCosseratPhysics: CR-Timo vs Cosserat 物理比較
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh

from xkep_cae.dynamics import (
    NonlinearTransientConfig,
    solve_nonlinear_transient,
)

pytestmark = pytest.mark.slow


# ====================================================================
# ヘルパー: 梁メッシュ・アセンブリ
# ====================================================================


def _build_cantilever_3d(
    n_elems: int = 10,
    L: float = 1.0,
    E: float = 2.1e11,
    nu: float = 0.3,
    rho: float = 7800.0,
    r: float = 0.01,
):
    """3Dカンチレバー梁の構造データを構築する."""
    from xkep_cae.elements.beam_timo3d import timo_beam3d_ke_global

    G = E / (2.0 * (1.0 + nu))
    A = np.pi * r**2
    Iy = np.pi * r**4 / 4.0
    Iz = Iy
    J = 2.0 * Iy
    kappa_y = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)
    kappa_z = kappa_y

    nodes = np.zeros((n_elems + 1, 3))
    nodes[:, 0] = np.linspace(0, L, n_elems + 1)
    conn = np.array([[i, i + 1] for i in range(n_elems)])

    n_nodes = n_elems + 1
    ndof = 6 * n_nodes

    K = np.zeros((ndof, ndof))
    for e in range(n_elems):
        n1, n2 = conn[e]
        coords = nodes[np.array([n1, n2])]
        ke = timo_beam3d_ke_global(coords, E, G, A, Iy, Iz, J, kappa_y, kappa_z)
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
        "kappa_y": kappa_y,
        "kappa_z": kappa_z,
        "rho": rho,
        "r": r,
        "L": L,
        "fixed_dofs": fixed_dofs,
        "ndof": ndof,
        "n_nodes": n_nodes,
        "n_elems": n_elems,
    }


def _get_free_dofs(data):
    """自由DOFの情報を返す."""
    ndof = data["ndof"]
    fixed = data["fixed_dofs"]
    free_mask = np.ones(ndof, dtype=bool)
    free_mask[fixed] = False
    free = np.where(free_mask)[0]
    return free, free_mask


def _estimate_period(data):
    """第1固有周期を推定する."""
    free, _ = _get_free_dofs(data)
    K_ff = data["K"][np.ix_(free, free)]
    M_ff = data["M"][np.ix_(free, free)]
    eigvals, _ = eigh(K_ff, M_ff)
    omega1 = np.sqrt(max(eigvals[0], 0.0))
    return 2.0 * np.pi / omega1 if omega1 > 0 else 1.0


def _make_cr_assemblers(data):
    """CR-Timo用の内力/接線剛性コールバック."""
    from xkep_cae.elements.beam_timo3d import assemble_cr_beam3d

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


def _make_cosserat_assemblers(data):
    """非線形Cosserat rod用の内力/接線剛性コールバック."""
    from xkep_cae.elements.beam_cosserat import assemble_cosserat_nonlinear

    nodes, conn = data["nodes"], data["conn"]
    E, G = data["E"], data["G"]
    A, Iy, Iz, J = data["A"], data["Iy"], data["Iz"], data["J"]
    ky, kz = data["kappa_y"], data["kappa_z"]

    def f_int_fn(u):
        _, f = assemble_cosserat_nonlinear(
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
        )
        return f

    def K_T_fn(u):
        K, _ = assemble_cosserat_nonlinear(
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
        )
        return K

    return f_int_fn, K_T_fn


def _make_linear_assemblers(K):
    """線形梁用の内力/接線剛性コールバック."""

    def f_int_fn(u):
        return K @ u

    def K_T_fn(u):
        return K

    return f_int_fn, K_T_fn


def _compute_beam_energies(data, u_hist, v_hist, f_int_fn):
    """梁系のエネルギーを時刻歴で計算する."""
    M = data["M"]
    n_steps = u_hist.shape[0]
    kinetic = np.zeros(n_steps)
    strain = np.zeros(n_steps)

    for i in range(n_steps):
        v = v_hist[i]
        u = u_hist[i]
        kinetic[i] = 0.5 * v @ M @ v
        f_int = f_int_fn(u)
        strain[i] = 0.5 * u @ f_int

    return kinetic, strain, kinetic + strain


def _static_initial_disp(data, load_dof, load_value):
    """静的荷重による初期変位を計算する."""
    free, _ = _get_free_dofs(data)
    F = np.zeros(data["ndof"])
    F[load_dof] = load_value
    K_ff = data["K"][np.ix_(free, free)]
    u0 = np.zeros(data["ndof"])
    u0[free] = np.linalg.solve(K_ff, F[free])
    return u0


# ====================================================================
# TestDynamicsEnergyPhysics: エネルギー保存
# ====================================================================


class TestDynamicsEnergyPhysics:
    """動的解析のエネルギー保存（物理テスト）.

    非減衰系では全エネルギー E = T + U が保存されるべき。
    """

    def test_cantilever_free_vibration_energy_linear(self):
        """線形カンチレバーの自由振動でエネルギーが保存される."""
        data = _build_cantilever_3d(n_elems=8, L=0.5, r=0.005)
        ndof = data["ndof"]
        fixed = data["fixed_dofs"]
        tip_dof = 6 * data["n_elems"] + 1

        u0 = _static_initial_disp(data, tip_dof, -1.0)
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

        _, _, total = _compute_beam_energies(data, res.displacement, res.velocity, f_int_fn)
        E0 = total[0]
        assert E0 > 0
        err_rel = np.max(np.abs(total - E0)) / E0
        assert err_rel < 0.01, f"線形梁エネルギー保存誤差: {err_rel:.4e}"

    def test_cantilever_free_vibration_energy_cr_nonlinear(self):
        """CR-Timo非線形カンチレバーでエネルギーが近似的に保存される."""
        data = _build_cantilever_3d(n_elems=10, L=0.5, r=0.005)
        ndof = data["ndof"]
        fixed = data["fixed_dofs"]
        tip_dof = 6 * data["n_elems"] + 1

        u0 = _static_initial_disp(data, tip_dof, -0.5)
        f_int_fn, K_T_fn = _make_cr_assemblers(data)
        T1 = _estimate_period(data)
        dt = T1 / 40.0
        n_steps = int(3 * T1 / dt)

        # CR/Cosseratは数値微分接線剛性(eps=1e-7)のため1e-6が実用限界
        cfg = NonlinearTransientConfig(dt=dt, n_steps=n_steps, tol_force=1e-6)
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
        assert res.converged, f"非収束: step={res.failed_step}"

        _, _, total = _compute_beam_energies(data, res.displacement, res.velocity, f_int_fn)
        E0 = total[0]
        assert E0 > 0
        # 数値微分起因のNR精度が限定的なため、10%以内で検証
        err_rel = np.max(np.abs(total - E0)) / E0
        assert err_rel < 0.10, f"CR非線形梁エネルギー保存誤差: {err_rel:.4e}"

    def test_hht_alpha_numerical_dissipation_beam(self):
        """HHT-α法でエネルギーが単調減少する（数値減衰）."""
        data = _build_cantilever_3d(n_elems=8, L=0.5, r=0.005)
        ndof = data["ndof"]
        fixed = data["fixed_dofs"]
        tip_dof = 6 * data["n_elems"] + 1

        u0 = _static_initial_disp(data, tip_dof, -1.0)
        f_int_fn, K_T_fn = _make_linear_assemblers(data["K"])
        T1 = _estimate_period(data)
        dt = T1 / 40.0
        n_steps = int(10 * T1 / dt)

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

        _, _, total = _compute_beam_energies(data, res.displacement, res.velocity, f_int_fn)
        assert total[-1] < total[0], (
            f"HHT-αでエネルギー増加: E0={total[0]:.6e}, E_end={total[-1]:.6e}"
        )
        dissipation = (total[0] - total[-1]) / total[0]
        assert dissipation > 0.001, f"HHT-αの減衰が小さすぎる: {dissipation:.6e}"


# ====================================================================
# TestDynamicsLargeRotationPhysics: 大回転動的挙動
# ====================================================================


class TestDynamicsLargeRotationPhysics:
    """大回転を伴う動的解析の物理的妥当性."""

    def test_cantilever_large_deflection_cr_converges(self):
        """CR-Timoカンチレバーの大変形動解析が収束する."""
        data = _build_cantilever_3d(n_elems=20, L=1.0, r=0.01)
        ndof = data["ndof"]
        fixed = data["fixed_dofs"]
        tip_dof = 6 * data["n_elems"] + 1

        target_delta = 0.1 * data["L"]
        P = 3.0 * data["E"] * data["Iy"] * target_delta / data["L"] ** 3
        f_static = np.zeros(ndof)
        f_static[tip_dof] = -P
        ramp_time = 0.01

        def get_force(t):
            return f_static * min(t / ramp_time, 1.0)

        f_int_fn, K_T_fn = _make_cr_assemblers(data)
        T1 = _estimate_period(data)
        dt = T1 / 30.0
        n_steps = int(2 * T1 / dt)

        cfg = NonlinearTransientConfig(dt=dt, n_steps=n_steps, tol_force=1e-8)
        res = solve_nonlinear_transient(
            data["M"],
            get_force,
            np.zeros(ndof),
            np.zeros(ndof),
            cfg,
            f_int_fn,
            K_T_fn,
            fixed_dofs=fixed,
            show_progress=False,
        )
        assert res.converged, f"CR大変形非収束: step={res.failed_step}"

        tip_disp = abs(res.displacement[-1, tip_dof])
        assert tip_disp > 0.01 * target_delta
        assert tip_disp < 10.0 * target_delta

    def test_cantilever_large_deflection_cosserat_converges(self):
        """Cosserat非線形カンチレバーの大変形動解析が収束する."""
        data = _build_cantilever_3d(n_elems=20, L=1.0, r=0.01)
        ndof = data["ndof"]
        fixed = data["fixed_dofs"]
        tip_dof = 6 * data["n_elems"] + 1

        target_delta = 0.1 * data["L"]
        P = 3.0 * data["E"] * data["Iy"] * target_delta / data["L"] ** 3
        f_static = np.zeros(ndof)
        f_static[tip_dof] = -P
        ramp_time = 0.01

        def get_force(t):
            return f_static * min(t / ramp_time, 1.0)

        f_int_fn, K_T_fn = _make_cosserat_assemblers(data)
        T1 = _estimate_period(data)
        dt = T1 / 30.0
        n_steps = int(2 * T1 / dt)

        # Cosserat数値微分の精度限界を考慮
        cfg = NonlinearTransientConfig(dt=dt, n_steps=n_steps, tol_force=1e-6)
        res = solve_nonlinear_transient(
            data["M"],
            get_force,
            np.zeros(ndof),
            np.zeros(ndof),
            cfg,
            f_int_fn,
            K_T_fn,
            fixed_dofs=fixed,
            show_progress=False,
        )
        assert res.converged, f"Cosserat大変形非収束: step={res.failed_step}"

        tip_disp = abs(res.displacement[-1, tip_dof])
        assert tip_disp > 0.01 * target_delta
        assert tip_disp < 10.0 * target_delta


# ====================================================================
# TestCRvsCosseratPhysics: 要素比較
# ====================================================================


class TestCRvsCosseratPhysics:
    """CR-Timo vs Cosserat の物理的等価性テスト."""

    def test_small_deformation_match(self):
        """小変形でCR-TimoとCosseratが同等の結果を返す."""
        data = _build_cantilever_3d(n_elems=20, L=0.5, r=0.005)
        ndof = data["ndof"]
        fixed = data["fixed_dofs"]
        tip_dof = 6 * data["n_elems"] + 1

        f_static = np.zeros(ndof)
        f_static[tip_dof] = -0.1
        ramp_time = 0.005

        def get_force(t):
            return f_static * min(t / ramp_time, 1.0)

        T1 = _estimate_period(data)
        dt = T1 / 30.0
        n_steps = int(T1 / dt)

        # CR-Timo（数値微分精度限界を考慮）
        f_cr, K_cr = _make_cr_assemblers(data)
        cfg_cr = NonlinearTransientConfig(dt=dt, n_steps=n_steps, tol_force=1e-6)
        res_cr = solve_nonlinear_transient(
            data["M"],
            get_force,
            np.zeros(ndof),
            np.zeros(ndof),
            cfg_cr,
            f_cr,
            K_cr,
            fixed_dofs=fixed,
            show_progress=False,
        )
        assert res_cr.converged

        # Cosserat
        f_cos, K_cos = _make_cosserat_assemblers(data)
        cfg_cos = NonlinearTransientConfig(dt=dt, n_steps=n_steps, tol_force=1e-6)
        res_cos = solve_nonlinear_transient(
            data["M"],
            get_force,
            np.zeros(ndof),
            np.zeros(ndof),
            cfg_cos,
            f_cos,
            K_cos,
            fixed_dofs=fixed,
            show_progress=False,
        )
        assert res_cos.converged

        tip_cr = res_cr.displacement[-1, tip_dof]
        tip_cos = res_cos.displacement[-1, tip_dof]

        if abs(tip_cr) > 1e-12:
            rel_diff = abs(tip_cr - tip_cos) / abs(tip_cr)
            assert rel_diff < 0.05, (
                f"CR vs Cosserat差: {rel_diff:.4f} (CR={tip_cr:.6e}, Cos={tip_cos:.6e})"
            )

    def test_large_deformation_convergence_comparison(self):
        """大変形（δ/L≈10%）で両者が同等の先端変位を返す.

        十分な要素数（20要素）で両者の差が1%以内。
        """
        data = _build_cantilever_3d(n_elems=20, L=1.0, r=0.01)
        ndof = data["ndof"]
        fixed = data["fixed_dofs"]
        tip_dof = 6 * data["n_elems"] + 1

        # δ/L ≈ 10%
        target_delta = 0.1 * data["L"]
        P = 3.0 * data["E"] * data["Iy"] * target_delta / data["L"] ** 3
        f_static = np.zeros(ndof)
        f_static[tip_dof] = -P

        # 静的NR解の比較
        from xkep_cae.elements.beam_cosserat import assemble_cosserat_nonlinear
        from xkep_cae.elements.beam_timo3d import assemble_cr_beam3d

        free, _ = _get_free_dofs(data)
        E, G = data["E"], data["G"]
        A, Iy, Iz, J = data["A"], data["Iy"], data["Iz"], data["J"]
        ky, kz = data["kappa_y"], data["kappa_z"]
        nodes, conn = data["nodes"], data["conn"]

        # CR NR
        u_cr = np.zeros(ndof)
        for _ in range(50):
            K_T, f_int = assemble_cr_beam3d(
                nodes,
                conn,
                u_cr,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                ky,
                kz,
                stiffness=True,
                internal_force=True,
                sparse=False,
            )
            R = f_static - f_int
            R[fixed] = 0.0
            if np.linalg.norm(R[free]) / max(np.linalg.norm(f_static[free]), 1.0) < 1e-8:
                break
            K_ff = K_T[np.ix_(free, free)]
            du = np.zeros(ndof)
            du[free] = np.linalg.solve(K_ff, R[free])
            u_cr += du

        # Cosserat NR
        u_cos = np.zeros(ndof)
        for _ in range(50):
            K_T, f_int = assemble_cosserat_nonlinear(
                nodes,
                conn,
                u_cos,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                ky,
                kz,
                stiffness=True,
                internal_force=True,
            )
            R = f_static - f_int
            R[fixed] = 0.0
            if np.linalg.norm(R[free]) / max(np.linalg.norm(f_static[free]), 1.0) < 1e-6:
                break
            K_ff = K_T[np.ix_(free, free)]
            du = np.zeros(ndof)
            du[free] = np.linalg.solve(K_ff, R[free])
            u_cos += du

        delta_cr = abs(u_cr[tip_dof])
        delta_cos = abs(u_cos[tip_dof])

        rel_diff = abs(delta_cr - delta_cos) / delta_cr
        assert rel_diff < 0.01, (
            f"大変形CR vs Cosserat差: {rel_diff:.4f} (CR={delta_cr:.6e}, Cos={delta_cos:.6e})"
        )

    def test_cosserat_nr_convergence_stall_documented(self):
        """Cosseratの数値微分接線剛性によるNR収束ストールの確認.

        数値微分（eps=1e-7）起因で残差が~1e-7〜1e-8でストールする。
        これはCosserat固有の課題であり、解析的接線剛性が必要な根拠。
        """
        data = _build_cantilever_3d(n_elems=16, L=0.5, r=0.005)
        ndof = data["ndof"]
        fixed = data["fixed_dofs"]
        tip_dof = 6 * data["n_elems"] + 1

        from xkep_cae.elements.beam_cosserat import assemble_cosserat_nonlinear

        free, _ = _get_free_dofs(data)
        F = np.zeros(ndof)
        F[tip_dof] = -1.0
        nodes, conn = data["nodes"], data["conn"]
        E, G = data["E"], data["G"]
        A, Iy, Iz, J = data["A"], data["Iy"], data["Iz"], data["J"]
        ky, kz = data["kappa_y"], data["kappa_z"]

        u = np.zeros(ndof)
        residuals = []
        for _nr in range(30):
            K_T, f_int = assemble_cosserat_nonlinear(
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
                internal_force=True,
            )
            R = F - f_int
            R[fixed] = 0.0
            ref = max(np.linalg.norm(F[free]), 1.0)
            res_rel = np.linalg.norm(R[free]) / ref
            residuals.append(res_rel)
            if res_rel < 1e-10:
                break
            K_ff = K_T[np.ix_(free, free)]
            du = np.zeros(ndof)
            du[free] = np.linalg.solve(K_ff, R[free])
            u += du

        # 3反復後に急速低下し、その後ストール
        assert residuals[2] < 1e-5, "3反復目で十分低下すべき"

        # ストール確認: 最後の5反復で残差が1桁も下がらない
        late_residuals = residuals[-5:]
        if len(late_residuals) >= 5:
            ratio = max(late_residuals) / min(late_residuals)
            assert ratio < 100, "ストール確認（100倍以内で変動）"


# ====================================================================
# TestDynamicsSymmetryPhysics: 対称性
# ====================================================================


class TestDynamicsSymmetryPhysics:
    """対称荷重に対する動的応答の対称性."""

    def test_simply_supported_symmetric_response(self):
        """単純支持梁の中央荷重で左右対称の動的応答."""
        from xkep_cae.elements.beam_timo3d import timo_beam3d_ke_global

        n_elems = 10
        L = 1.0
        E = 2.1e11
        nu = 0.3
        rho = 7800.0
        r = 0.01
        G = E / (2.0 * (1.0 + nu))
        A = np.pi * r**2
        Iy = np.pi * r**4 / 4.0
        Iz = Iy
        J = 2.0 * Iy
        kappa_y = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)
        kappa_z = kappa_y

        nodes = np.zeros((n_elems + 1, 3))
        nodes[:, 0] = np.linspace(0, L, n_elems + 1)
        conn = np.array([[i, i + 1] for i in range(n_elems)])
        n_nodes = n_elems + 1
        ndof = 6 * n_nodes

        K = np.zeros((ndof, ndof))
        for e in range(n_elems):
            n1, n2 = conn[e]
            coords = nodes[np.array([n1, n2])]
            ke = timo_beam3d_ke_global(coords, E, G, A, Iy, Iz, J, kappa_y, kappa_z)
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

        fixed = np.array([1, 6 * n_elems + 1])  # uy拘束
        mid = n_elems // 2
        tip_dof = 6 * mid + 1
        f_static = np.zeros(ndof)
        f_static[tip_dof] = -10.0

        f_int_fn, K_T_fn = _make_linear_assemblers(K)

        free_mask = np.ones(ndof, dtype=bool)
        free_mask[fixed] = False
        free = np.where(free_mask)[0]
        K_ff = K[np.ix_(free, free)]
        M_ff = M[np.ix_(free, free)]
        eigvals, _ = eigh(K_ff, M_ff)
        omega1 = np.sqrt(max(eigvals[0], 0.0))
        T1 = 2.0 * np.pi / omega1 if omega1 > 0 else 1.0
        dt = T1 / 30.0
        n_steps = int(T1 / dt)

        cfg = NonlinearTransientConfig(dt=dt, n_steps=n_steps, tol_force=1e-12)
        res = solve_nonlinear_transient(
            M,
            f_static,
            np.zeros(ndof),
            np.zeros(ndof),
            cfg,
            f_int_fn,
            K_T_fn,
            fixed_dofs=fixed,
            show_progress=False,
        )
        assert res.converged

        u_final = res.displacement[-1]
        for i in range(1, mid):
            j = n_elems - i
            uy_left = u_final[6 * i + 1]
            uy_right = u_final[6 * j + 1]
            if abs(uy_left) > 1e-15:
                sym_err = abs(uy_left - uy_right) / abs(uy_left)
                assert sym_err < 0.01, (
                    f"対称性破れ: node {i} uy={uy_left:.6e}, node {j} uy={uy_right:.6e}"
                )


# ====================================================================
# TestDynamicsFrequencyPhysics: 周波数精度
# ====================================================================


class TestDynamicsFrequencyPhysics:
    """動的解析の周波数精度."""

    def test_cantilever_fundamental_frequency(self):
        """カンチレバー梁の第1固有周波数が理論値と一致.

        理論: f₁ = (1.8751²)/(2π) × √(EI/(ρAL⁴))
        """
        data = _build_cantilever_3d(n_elems=20, L=1.0, r=0.01)
        free, _ = _get_free_dofs(data)
        K_ff = data["K"][np.ix_(free, free)]
        M_ff = data["M"][np.ix_(free, free)]
        eigvals, _ = eigh(K_ff, M_ff)
        f1_fem = np.sqrt(max(eigvals[0], 0.0)) / (2.0 * np.pi)

        beta1_L = 1.8751
        f1_theory = (
            beta1_L**2
            / (2.0 * np.pi)
            * np.sqrt(data["E"] * data["Iy"] / (data["rho"] * data["A"] * data["L"] ** 4))
        )

        rel_err = abs(f1_fem - f1_theory) / f1_theory
        assert rel_err < 0.10, (
            f"第1固有周波数誤差: {rel_err:.4f} ({f1_fem:.2f} vs {f1_theory:.2f} Hz)"
        )

    def test_geometric_stiffening_increases_frequency(self):
        """幾何剛性効果: 軸引張力で固有周波数が上昇する.

        接線剛性K_T(u)の固有値が軸引張力下で上昇することを
        静的に確認する。動的FFTは数値微分精度の制約を回避。
        """
        from xkep_cae.elements.beam_timo3d import assemble_cr_beam3d

        data = _build_cantilever_3d(n_elems=10, L=0.5, r=0.005)
        ndof = data["ndof"]
        free, _ = _get_free_dofs(data)
        nodes, conn = data["nodes"], data["conn"]
        E, G = data["E"], data["G"]
        A, Iy, Iz, J = data["A"], data["Iy"], data["Iz"], data["J"]
        ky, kz = data["kappa_y"], data["kappa_z"]
        M_ff = data["M"][np.ix_(free, free)]

        # ケース1: 無変形状態 → 線形K
        K_ff_0 = data["K"][np.ix_(free, free)]
        eigvals_0, _ = eigh(K_ff_0, M_ff)
        omega1_0 = np.sqrt(max(eigvals_0[0], 0.0))

        # ケース2: 軸引張変形状態 → CR接線K
        u_axial = np.zeros(ndof)
        axial_strain = 0.001  # 0.1%歪み
        for i in range(data["n_elems"] + 1):
            u_axial[6 * i] = axial_strain * data["L"] * i / data["n_elems"]

        K_T, _ = assemble_cr_beam3d(
            nodes,
            conn,
            u_axial,
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
        K_ff_axial = K_T[np.ix_(free, free)]
        eigvals_axial, _ = eigh(K_ff_axial, M_ff)
        omega1_axial = np.sqrt(max(eigvals_axial[0], 0.0))

        assert omega1_axial > omega1_0, (
            f"幾何剛性: ω_axial={omega1_axial:.4f} <= ω_0={omega1_0:.4f}"
        )


# ====================================================================
# TestDynamicsStabilityPhysics: 安定性
# ====================================================================


class TestDynamicsStabilityPhysics:
    """時間積分の数値安定性."""

    def test_newmark_unconditional_stability(self):
        """平均加速度法は大きなΔtでも発散しない."""
        data = _build_cantilever_3d(n_elems=5, L=0.5, r=0.005)
        ndof = data["ndof"]
        fixed = data["fixed_dofs"]

        free, _ = _get_free_dofs(data)
        K_ff = data["K"][np.ix_(free, free)]
        M_ff = data["M"][np.ix_(free, free)]
        eigvals, _ = eigh(K_ff, M_ff)
        omega_max = np.sqrt(max(eigvals[-1], 0.0))
        dt_cr = 2.0 / omega_max
        dt = 10.0 * dt_cr
        n_steps = 20

        tip_dof = 6 * data["n_elems"] + 1
        u0 = np.zeros(ndof)
        u0[tip_dof] = -1e-4

        f_int_fn, K_T_fn = _make_linear_assemblers(data["K"])
        cfg = NonlinearTransientConfig(dt=dt, n_steps=n_steps, tol_force=1e-10)
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
        max_disp = np.max(np.abs(res.displacement))
        assert max_disp < 1.0, f"解が発散: {max_disp:.4e}"
        assert max_disp > 1e-10, f"解がゼロ: {max_disp:.4e}"

    def test_central_difference_instability_detection(self):
        """Central Difference法のΔt超過で不安定フラグ."""
        from xkep_cae.dynamics import (
            CentralDifferenceConfig,
            critical_time_step,
            solve_central_difference,
        )

        data = _build_cantilever_3d(n_elems=5, L=0.5, r=0.005)
        dt_cr = critical_time_step(data["M"], data["K"], fixed_dofs=data["fixed_dofs"])
        assert dt_cr > 0

        cfg = CentralDifferenceConfig(dt=1.5 * dt_cr, n_steps=10)
        res = solve_central_difference(
            data["M"],
            np.zeros_like(data["M"]),
            data["K"],
            np.zeros(data["ndof"]),
            np.zeros(data["ndof"]),
            np.zeros(data["ndof"]),
            cfg,
            fixed_dofs=data["fixed_dofs"],
        )
        assert not res.stable
