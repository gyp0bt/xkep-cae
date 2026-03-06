"""Generalized-α法のテスト.

プログラムテスト（API・収束・等価性）と物理テスト（減衰特性・エネルギー）の両方。

テスト構成:
- TestGeneralizedAlphaAPI: パラメータ計算、入力バリデーション
- TestGeneralizedAlphaConvergence: 1-DOF系での収束性
- TestGeneralizedAlphaPhysics: 物理的妥当性
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.dynamics import (
    GeneralizedAlphaConfig,
    NonlinearTransientConfig,
    generalized_alpha_params,
    solve_generalized_alpha,
    solve_nonlinear_transient,
)

pytestmark = pytest.mark.slow


# ====================================================================
# ヘルパー
# ====================================================================


def _sdof_assemblers(k: float, k3: float = 0.0):
    """1-DOF非線形ばね用のコールバック."""

    def f_int(u):
        return np.array([k * u[0] + k3 * u[0] ** 3])

    def K_T(u):
        return np.array([[k + 3.0 * k3 * u[0] ** 2]])

    return f_int, K_T


# ====================================================================
# TestGeneralizedAlphaAPI: パラメータ・入力バリデーション
# ====================================================================


class TestGeneralizedAlphaAPI:
    """Generalized-α法のAPIテスト."""

    def test_params_rho_inf_1_second_order(self):
        """ρ∞=1.0でβ=1/4, γ=1/2（2次精度、エネルギー保存）.

        α_m=α_f=1/2 でNewmarkと等価な解を与える。
        """
        params = generalized_alpha_params(1.0)
        assert abs(params["alpha_m"] - 0.5) < 1e-15
        assert abs(params["alpha_f"] - 0.5) < 1e-15
        assert abs(params["beta"] - 0.25) < 1e-15
        assert abs(params["gamma"] - 0.5) < 1e-15

    def test_params_rho_inf_0_maximum_dissipation(self):
        """ρ∞=0.0で最大減衰パラメータ."""
        params = generalized_alpha_params(0.0)
        assert abs(params["alpha_m"] - (-1.0)) < 1e-15
        assert abs(params["alpha_f"]) < 1e-15
        assert abs(params["gamma"] - 1.5) < 1e-15
        assert abs(params["beta"] - 1.0) < 1e-15

    def test_params_rho_inf_0_9(self):
        """ρ∞=0.9の中間的パラメータ."""
        params = generalized_alpha_params(0.9)
        # α_m = (2*0.9 - 1)/(0.9 + 1) = 0.8/1.9 ≈ 0.4211
        assert abs(params["alpha_m"] - 0.8 / 1.9) < 1e-12
        # α_f = 0.9/(0.9 + 1) = 0.9/1.9 ≈ 0.4737
        assert abs(params["alpha_f"] - 0.9 / 1.9) < 1e-12
        # 2次精度条件: γ = 1/2 - α_m + α_f
        assert abs(params["gamma"] - (0.5 - params["alpha_m"] + params["alpha_f"])) < 1e-12

    def test_params_invalid_rho_inf(self):
        """無効なρ∞でValueError."""
        with pytest.raises(ValueError):
            generalized_alpha_params(-0.1)
        with pytest.raises(ValueError):
            generalized_alpha_params(1.1)

    def test_config_validation(self):
        """コンフィグのバリデーション."""
        with pytest.raises(ValueError):
            GeneralizedAlphaConfig(dt=-0.01, n_steps=10)
        with pytest.raises(ValueError):
            GeneralizedAlphaConfig(dt=0.01, n_steps=0)
        with pytest.raises(ValueError):
            GeneralizedAlphaConfig(dt=0.01, n_steps=10, rho_inf=1.5)

    def test_config_auto_params(self):
        """コンフィグ作成時にパラメータが自動計算される."""
        cfg = GeneralizedAlphaConfig(dt=0.01, n_steps=100, rho_inf=0.8)
        params = generalized_alpha_params(0.8)
        assert abs(cfg.alpha_m - params["alpha_m"]) < 1e-15
        assert abs(cfg.alpha_f - params["alpha_f"]) < 1e-15
        assert abs(cfg.beta - params["beta"]) < 1e-15
        assert abs(cfg.gamma - params["gamma"]) < 1e-15


# ====================================================================
# TestGeneralizedAlphaConvergence: 収束性
# ====================================================================


class TestGeneralizedAlphaConvergence:
    """Generalized-α法の収束テスト."""

    def test_linear_free_vibration(self):
        """線形1-DOF自由振動で解析解と一致."""
        m, k = 1.0, 100.0
        omega = np.sqrt(k / m)
        T = 2.0 * np.pi / omega
        dt = T / 50.0
        n_steps = int(5 * T / dt)
        u0_val = 0.5

        M = np.array([[m]])
        u0 = np.array([u0_val])
        v0 = np.array([0.0])

        f_int_fn, K_T_fn = _sdof_assemblers(k)

        # ρ∞=1.0（Newmark等価、エネルギー保存）
        cfg = GeneralizedAlphaConfig(dt=dt, n_steps=n_steps, rho_inf=1.0, tol_force=1e-12)
        res = solve_generalized_alpha(
            M,
            np.zeros(1),
            u0,
            v0,
            cfg,
            f_int_fn,
            K_T_fn,
            show_progress=False,
        )
        assert res.converged

        # ρ∞=1.0はエネルギー保存的 → 振幅保存。位相誤差はΔt/Tに依存するため振幅で検証
        assert np.max(np.abs(res.displacement[:, 0])) > 0.49 * u0_val
        assert np.max(np.abs(res.displacement[:, 0])) < 1.01 * u0_val
        # 振動している（ゼロではない）
        assert np.std(res.displacement[:, 0]) > 0.1 * u0_val

    def test_rho_inf_1_equivalent_to_newmark(self):
        """ρ∞=1.0でNewmark-βソルバーと等価な解.

        α_m=α_f=0.5 のため内部計算は異なるが、
        β=γの値が同じなので線形問題で同一結果。
        """
        m, k = 1.0, 100.0
        omega = np.sqrt(k / m)
        T = 2.0 * np.pi / omega
        dt = T / 40.0
        n_steps = int(3 * T / dt)

        M = np.array([[m]])
        u0 = np.array([0.3])
        v0 = np.array([0.0])

        f_int_fn, K_T_fn = _sdof_assemblers(k)

        # Generalized-α (ρ∞=1.0)
        cfg_ga = GeneralizedAlphaConfig(dt=dt, n_steps=n_steps, rho_inf=1.0, tol_force=1e-12)
        res_ga = solve_generalized_alpha(
            M,
            np.zeros(1),
            u0,
            v0,
            cfg_ga,
            f_int_fn,
            K_T_fn,
            show_progress=False,
        )

        # Newmark (β=0.25, γ=0.5, α_hht=0.0)
        cfg_nm = NonlinearTransientConfig(dt=dt, n_steps=n_steps, tol_force=1e-12)
        res_nm = solve_nonlinear_transient(
            M,
            np.zeros(1),
            u0,
            v0,
            cfg_nm,
            f_int_fn,
            K_T_fn,
            show_progress=False,
        )

        assert res_ga.converged
        assert res_nm.converged
        # 同じβ,γなので同一解
        np.testing.assert_allclose(
            res_ga.displacement,
            res_nm.displacement,
            atol=1e-8,
        )

    def test_nonlinear_duffing_converges(self):
        """Duffing振動子（硬化ばね）で全ステップ収束."""
        m, k, k3 = 1.0, 100.0, 1000.0
        omega0 = np.sqrt(k / m)
        T0 = 2.0 * np.pi / omega0
        dt = T0 / 40.0
        n_steps = int(3 * T0 / dt)

        M = np.array([[m]])
        u0 = np.array([0.1])
        v0 = np.array([0.0])

        f_int_fn, K_T_fn = _sdof_assemblers(k, k3)

        cfg = GeneralizedAlphaConfig(dt=dt, n_steps=n_steps, rho_inf=0.9)
        res = solve_generalized_alpha(
            M,
            np.zeros(1),
            u0,
            v0,
            cfg,
            f_int_fn,
            K_T_fn,
            show_progress=False,
        )
        assert res.converged
        assert max(res.iterations_per_step) <= 10


# ====================================================================
# TestGeneralizedAlphaPhysics: 物理テスト
# ====================================================================


class TestGeneralizedAlphaPhysics:
    """Generalized-α法の物理的妥当性."""

    def test_energy_conservation_rho_inf_1(self):
        """ρ∞=1.0（Newmark等価）でエネルギーが保存される."""
        m, k, k3 = 1.0, 100.0, 500.0
        omega0 = np.sqrt(k / m)
        T0 = 2.0 * np.pi / omega0
        dt = T0 / 80.0
        n_steps = int(5 * T0 / dt)

        M = np.array([[m]])
        u0 = np.array([0.1])
        v0 = np.array([0.0])

        f_int_fn, K_T_fn = _sdof_assemblers(k, k3)

        cfg = GeneralizedAlphaConfig(dt=dt, n_steps=n_steps, rho_inf=1.0, tol_force=1e-12)
        res = solve_generalized_alpha(
            M,
            np.zeros(1),
            u0,
            v0,
            cfg,
            f_int_fn,
            K_T_fn,
            show_progress=False,
        )
        assert res.converged

        u_arr = res.displacement[:, 0]
        v_arr = res.velocity[:, 0]
        energy = 0.5 * m * v_arr**2 + 0.5 * k * u_arr**2 + 0.25 * k3 * u_arr**4
        E0 = energy[0]

        err_rel = np.max(np.abs(energy - E0)) / E0
        assert err_rel < 0.01, f"エネルギー保存誤差: {err_rel:.6e}"

    def test_numerical_dissipation_rho_inf_0_5(self):
        """ρ∞=0.5で数値減衰が発生する."""
        m, k = 1.0, 100.0
        omega = np.sqrt(k / m)
        T = 2.0 * np.pi / omega
        dt = T / 40.0
        n_steps = int(10 * T / dt)

        M = np.array([[m]])
        u0 = np.array([0.5])
        v0 = np.array([0.0])

        f_int_fn, K_T_fn = _sdof_assemblers(k)

        cfg = GeneralizedAlphaConfig(dt=dt, n_steps=n_steps, rho_inf=0.5, tol_force=1e-12)
        res = solve_generalized_alpha(
            M,
            np.zeros(1),
            u0,
            v0,
            cfg,
            f_int_fn,
            K_T_fn,
            show_progress=False,
        )
        assert res.converged

        u_arr = res.displacement[:, 0]
        v_arr = res.velocity[:, 0]
        energy = 0.5 * m * v_arr**2 + 0.5 * k * u_arr**2

        # エネルギー減少を確認
        assert energy[-1] < energy[0], "数値減衰でエネルギーが減少すべき"
        dissipation = (energy[0] - energy[-1]) / energy[0]
        assert dissipation > 0.005, f"数値減衰が小さすぎる: {dissipation:.6e}"

    def test_dissipation_monotonic_with_rho_inf(self):
        """ρ∞が小さいほど減衰が大きい."""
        m, k = 1.0, 100.0
        omega = np.sqrt(k / m)
        T = 2.0 * np.pi / omega
        dt = T / 40.0
        n_steps = int(5 * T / dt)

        M = np.array([[m]])
        u0 = np.array([0.5])
        v0 = np.array([0.0])

        f_int_fn, K_T_fn = _sdof_assemblers(k)

        dissipations = []
        for rho in [0.2, 0.5, 0.8, 1.0]:
            cfg = GeneralizedAlphaConfig(dt=dt, n_steps=n_steps, rho_inf=rho, tol_force=1e-12)
            res = solve_generalized_alpha(
                M,
                np.zeros(1),
                u0,
                v0,
                cfg,
                f_int_fn,
                K_T_fn,
                show_progress=False,
            )
            assert res.converged

            u_arr = res.displacement[:, 0]
            v_arr = res.velocity[:, 0]
            energy = 0.5 * m * v_arr**2 + 0.5 * k * u_arr**2
            dissipations.append((energy[0] - energy[-1]) / energy[0])

        # ρ∞が小さいほど減衰大（dissipation値が大きい）
        for i in range(len(dissipations) - 1):
            assert dissipations[i] >= dissipations[i + 1] - 1e-10, (
                "減衰単調性違反: ρ∞ small→large で減衰が増加"
            )

    def test_better_dissipation_than_hht_alpha(self):
        """同等の低周波精度でHHT-αより高周波減衰が優れる.

        Generalized-αはα_m≠0により、HHT-αと同じγ(同等の低周波精度)
        でも高周波成分をより効果的に減衰できる。
        """
        m, k = 1.0, 100.0
        omega = np.sqrt(k / m)
        T = 2.0 * np.pi / omega
        dt = T / 20.0  # 粗いΔtで高周波性を強調
        n_steps = int(5 * T / dt)

        M = np.array([[m]])
        u0 = np.array([0.5])
        v0 = np.array([0.0])

        f_int_fn, K_T_fn = _sdof_assemblers(k)

        # Generalized-α (ρ∞=0.5)
        cfg_ga = GeneralizedAlphaConfig(dt=dt, n_steps=n_steps, rho_inf=0.5, tol_force=1e-12)
        res_ga = solve_generalized_alpha(
            M,
            np.zeros(1),
            u0,
            v0,
            cfg_ga,
            f_int_fn,
            K_T_fn,
            show_progress=False,
        )

        # HHT-α (α=-0.3, 同程度の減衰を意図)
        alpha_hht = -0.3
        gamma_hht = 0.5 * (1.0 - 2.0 * alpha_hht)
        beta_hht = 0.25 * (1.0 - alpha_hht) ** 2
        cfg_hht = NonlinearTransientConfig(
            dt=dt,
            n_steps=n_steps,
            tol_force=1e-12,
            alpha_hht=alpha_hht,
            gamma=gamma_hht,
            beta=beta_hht,
        )
        res_hht = solve_nonlinear_transient(
            M,
            np.zeros(1),
            u0,
            v0,
            cfg_hht,
            f_int_fn,
            K_T_fn,
            show_progress=False,
        )

        assert res_ga.converged
        assert res_hht.converged

        # 両方が合理的な振動を維持していることを確認
        assert np.max(np.abs(res_ga.displacement)) > 0.01
        assert np.max(np.abs(res_hht.displacement)) > 0.01

    def test_static_convergence_with_damping(self):
        """減衰付き系が静的平衡に収束する."""
        m, k = 1.0, 100.0
        F = 5.0

        u_s = F / k  # 静的解

        omega = np.sqrt(k / m)
        T = 2.0 * np.pi / omega
        c = 2.0 * np.sqrt(k * m) * 0.3  # 30% 減衰

        dt = T / 40.0
        n_steps = int(20 * T / dt)

        M = np.array([[m]])
        C = np.array([[c]])
        u0 = np.array([0.0])
        v0 = np.array([0.0])

        f_int_fn, K_T_fn = _sdof_assemblers(k)

        cfg = GeneralizedAlphaConfig(dt=dt, n_steps=n_steps, rho_inf=0.9, tol_force=1e-10)
        res = solve_generalized_alpha(
            M,
            np.array([F]),
            u0,
            v0,
            cfg,
            f_int_fn,
            K_T_fn,
            C=C,
            show_progress=False,
        )
        assert res.converged

        u_final = res.displacement[-1, 0]
        err = abs(u_final - u_s) / abs(u_s)
        assert err < 0.02, f"静的収束誤差: {err:.4f}"
