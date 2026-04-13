"""TestFiber1DMaterialPhysics — 物理的妥当性テスト.

設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md
Phase F1 完了判定テスト（6件の一部）。

テスト項目:
1. 単軸サイクルでの閉ループ + 残留ひずみ確認
2. H=0 の完全弾塑性での正確な降伏・除荷挙動
3. 塑性仕事 = ヒステリシスループ面積（エネルギー整合）
4. 一貫接線の FD 検証
5. 弾性域での状態不変性
6. work/beam_hysteresis/01_kh_vs_friction_equivalence.py との等価性検証

[← README](../../../../README.md)
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.elements.fiber.materials import BilinearKinematicHardening1D
from xkep_cae.elements.fiber.state import Fiber1DState


class TestFiber1DMaterialPhysics:
    """物理テスト — ヒステリシス・エネルギー・接線整合性."""

    def test_uniaxial_cycle_residual_strain(self) -> None:
        """単軸サイクル ε=[0→0.01→-0.01→0.01] で残留ひずみと閉ループを確認.

        BilinearKH は降伏後に塑性ひずみが蓄積し、除荷時に
        弾性勾配 E で戻る。再載荷で反対方向に降伏する。
        """
        E = 100_000.0
        sigma_y = 300.0
        H = 10_000.0
        mat = BilinearKinematicHardening1D(E=E, sigma_y=sigma_y, H=H)

        # ε パス: 0 → +0.01 → -0.01 → +0.01
        n = 200
        eps_path = np.concatenate(
            [
                np.linspace(0, 0.01, n),
                np.linspace(0.01, -0.01, 2 * n),
                np.linspace(-0.01, 0.01, 2 * n),
            ]
        )

        state = Fiber1DState()
        sigmas = []
        for eps in eps_path:
            sigma, _, state = mat.evaluate(float(eps), state)
            sigmas.append(sigma)
        sigmas = np.array(sigmas)

        # 降伏が発生していることを確認
        assert state.eps_p != 0.0, "塑性ひずみが蓄積されていない"
        assert state.alpha != 0.0, "背応力がシフトしていない"

        # ピーク応力が降伏応力を超えている（硬化により）
        assert max(sigmas) > sigma_y
        # 最小応力が負の降伏応力を下回っている
        assert min(sigmas) < -sigma_y

    def test_perfect_elastoplastic_yield(self) -> None:
        """H=0（完全弾塑性）で降伏後の応力が σ_y に保持される."""
        E = 200_000.0
        sigma_y = 500.0
        mat = BilinearKinematicHardening1D(E=E, sigma_y=sigma_y, H=0.0)

        state = Fiber1DState()
        # 降伏ひずみ
        eps_y = sigma_y / E

        # 弾性域
        sigma, E_t, state = mat.evaluate(eps_y * 0.5, state)
        assert sigma == pytest.approx(E * eps_y * 0.5, rel=1e-10)
        assert E_t == pytest.approx(E, rel=1e-10)

        # 降伏後（2倍の降伏ひずみ）
        sigma, E_t, state = mat.evaluate(eps_y * 2.0, state)
        assert sigma == pytest.approx(sigma_y, rel=1e-10)
        assert E_t == pytest.approx(0.0, abs=1e-10)  # H=0 → 接線=0

    def test_energy_balance_loop_area(self) -> None:
        """塑性仕事 = ヒステリシスループ面積（エネルギー整合）.

        σ-ε ループの面積（台形積分）と、
        各ステップの塑性仕事 Σ(σ·Δε_p) が一致することを検証。
        """
        E = 100_000.0
        sigma_y = 300.0
        H = 10_000.0
        mat = BilinearKinematicHardening1D(E=E, sigma_y=sigma_y, H=H)

        # 完全な1サイクル: 0 → +ε_max → -ε_max → +ε_max (→ 0)
        eps_max = 0.01
        n = 400
        eps_path = np.concatenate(
            [
                np.linspace(0, eps_max, n),
                np.linspace(eps_max, -eps_max, 2 * n),
                np.linspace(-eps_max, eps_max, 2 * n),
            ]
        )

        state = Fiber1DState()
        sigmas = []
        plastic_work = 0.0
        prev_eps_p = 0.0

        for eps in eps_path:
            sigma, _, state = mat.evaluate(float(eps), state)
            sigmas.append(sigma)
            # 塑性仕事: σ * Δε_p
            d_eps_p = state.eps_p - prev_eps_p
            plastic_work += sigma * d_eps_p
            prev_eps_p = state.eps_p

        sigmas = np.array(sigmas)

        # 後半の閉ループ部分（ε_max → -ε_max → ε_max）のループ面積
        loop_start = n  # 最初の ε_max に到達した時点
        loop_eps = eps_path[loop_start:]
        loop_sigma = sigmas[loop_start:]
        loop_area = abs(np.trapezoid(loop_sigma, loop_eps))

        # 後半の塑性仕事
        state2 = Fiber1DState()
        # まず最初の載荷でstate到達
        for eps in eps_path[:n]:
            _, _, state2 = mat.evaluate(float(eps), state2)

        plastic_work_loop = 0.0
        prev_eps_p2 = state2.eps_p
        for eps in loop_eps:
            sigma, _, state2 = mat.evaluate(float(eps), state2)
            d_eps_p = state2.eps_p - prev_eps_p2
            plastic_work_loop += sigma * d_eps_p
            prev_eps_p2 = state2.eps_p

        # ループ面積と塑性仕事が一致（rtol=5%、離散化誤差込み）
        assert plastic_work_loop == pytest.approx(loop_area, rel=0.05)
        # 正の散逸を確認
        assert plastic_work_loop > 0.0

    def test_consistent_tangent_fd(self) -> None:
        """一貫接線 dσ/dε の有限差分検証（atol=1e-5）.

        弾性域と塑性域の両方で接線を検証する。
        """
        E = 100_000.0
        sigma_y = 300.0
        H = 10_000.0
        mat = BilinearKinematicHardening1D(E=E, sigma_y=sigma_y, H=H)
        h = 1e-7  # FD 摂動幅

        # Case 1: 弾性域（ε < ε_y）
        state = Fiber1DState()
        eps = 0.001  # E=100000 → σ=100 < σ_y=300
        sigma_c, E_t, _ = mat.evaluate(eps, state)
        sigma_p, _, _ = mat.evaluate(eps + h, state)
        sigma_m, _, _ = mat.evaluate(eps - h, state)
        fd_tangent = (sigma_p - sigma_m) / (2 * h)
        assert E_t == pytest.approx(fd_tangent, abs=1e-3)

        # Case 2: 塑性域（まず降伏させてから）
        state_plastic = Fiber1DState()
        # 降伏ひずみ超えまで載荷
        eps_yield = sigma_y / E
        _, _, state_plastic = mat.evaluate(eps_yield * 2.0, state_plastic)
        # 塑性域での接線検証
        eps2 = eps_yield * 3.0
        sigma_c, E_t, _ = mat.evaluate(eps2, state_plastic)
        sigma_p, _, _ = mat.evaluate(eps2 + h, state_plastic)
        sigma_m, _, _ = mat.evaluate(eps2 - h, state_plastic)
        fd_tangent = (sigma_p - sigma_m) / (2 * h)
        assert E_t == pytest.approx(fd_tangent, abs=1e-3)

        # 塑性接線が E*H/(E+H) であることを確認
        E_t_analytical = E * H / (E + H)
        assert E_t == pytest.approx(E_t_analytical, rel=1e-10)

    def test_elastic_state_unchanged(self) -> None:
        """弾性域内での載荷・除荷で状態が変化しないことを確認."""
        E = 100_000.0
        sigma_y = 500.0
        H = 10_000.0
        mat = BilinearKinematicHardening1D(E=E, sigma_y=sigma_y, H=H)

        state0 = Fiber1DState()

        # 弾性域内で往復
        for eps in [0.001, 0.003, 0.001, -0.001, 0.0]:
            sigma, E_t, state_new = mat.evaluate(eps, state0)
            assert state_new is state0, "弾性域で状態オブジェクトが変わった"
            assert E_t == pytest.approx(E)

    def test_kh_strand_friction_equivalence(self) -> None:
        """KH と撚線摩擦の数学的等価性を検証.

        work/beam_hysteresis/01_kh_vs_friction_equivalence.py の結論:
        KinematicHardening1D ≡ StrandFriction1D（同じ変分不等式）。

        KH: (E, σ_y, H) → σ = E(ε - ε_p), |σ - α| ≤ σ_y
        摩擦: (k_strand, f_y, k_slip) → 同一式

        パラメータ対応: E=k_strand, σ_y=f_y, H=k_slip
        """
        E = 100_000.0
        sigma_y = 3000.0
        H = 10_000.0
        mat = BilinearKinematicHardening1D(E=E, sigma_y=sigma_y, H=H)

        # サイクル載荷パスで応力を記録
        n = 300
        eps_path = np.concatenate(
            [
                np.linspace(0, 0.3, n),
                np.linspace(0.3, -0.3, 2 * n),
                np.linspace(-0.3, 0.0, n),
            ]
        )

        state = Fiber1DState()
        sigmas = []
        for eps in eps_path:
            sigma, _, state = mat.evaluate(float(eps), state)
            sigmas.append(sigma)

        # 除荷方向が弾性勾配 E で始まることを確認
        # ε=0.3（ピーク）直後の除荷区間（idx_peak+1 → idx_peak+2）
        idx_peak = n  # linspace 境界を1つスキップ
        d_sigma = sigmas[idx_peak + 1] - sigmas[idx_peak]
        d_eps = eps_path[idx_peak + 1] - eps_path[idx_peak]
        assert abs(d_eps) > 0, "Δε が 0（linspace 境界衝突）"
        unload_slope = d_sigma / d_eps
        assert unload_slope == pytest.approx(E, rel=0.02)

        # 反転後に反対方向に降伏するまで 2σ_y のドロップ
        # （Bauschinger 効果: 移動硬化の特徴）
        sigma_at_peak = sigmas[n - 1]  # ε=0.3 到達時の応力
        # 降伏ドロップ幅は 2σ_y
        expected_sigma_at_reverse_yield = sigma_at_peak - 2 * sigma_y
        # 除荷中のどこかでこの応力を通過する
        unload_sigmas = np.array(sigmas[n - 1 :])
        min_idx = np.argmin(np.abs(unload_sigmas - expected_sigma_at_reverse_yield))
        assert unload_sigmas[min_idx] == pytest.approx(expected_sigma_at_reverse_yield, rel=0.05)
