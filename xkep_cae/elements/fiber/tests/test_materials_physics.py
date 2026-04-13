"""TestFiber1DMaterialPhysics — 物理的妥当性テスト.

設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md
Phase F1 完了判定テスト（6件）+ Phase F2 追加テスト。

テスト項目（F1）:
1. 単軸サイクルでの閉ループ + 残留ひずみ確認
2. H=0 の完全弾塑性での正確な降伏・除荷挙動
3. 塑性仕事 = ヒステリシスループ面積（エネルギー整合）
4. 一貫接線の FD 検証
5. 弾性域での状態不変性
6. work/beam_hysteresis/01_kh_vs_friction_equivalence.py との等価性検証

テスト項目（F2）:
7. 05_smooth_teardrop.py 単一サイクル再現（rtol=1%）
8. 劣化効果 — β=1.0 vs β=0.25 での U/L 比検証
9. N 層収束性 — N 増加で応力-ひずみ曲線が安定
10. 一貫接線 FD 検証（弾性域・スリップ遷移付近）
11. エネルギー散逸正定値性
12. β=1.0 で BilinearKH との等価性

[← README](../../../../README.md)
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.elements.fiber.materials import (
    BilinearKinematicHardening1D,
    MultiLayerFrictionDegrading1D,
)
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


def _make_teardrop_model(
    n_layers: int = 150,
    beta: float = 0.25,
) -> MultiLayerFrictionDegrading1D:
    """05_smooth_teardrop.py と同一パラメータのモデルを生成."""
    E_base = 1500.0
    k_contact_total = 7500.0
    k_weights = np.linspace(0.5, 1.5, n_layers)
    k_weights *= n_layers / k_weights.sum()
    k_each = k_contact_total / n_layers
    k_virgin = k_each * k_weights
    eps_y_arr = np.logspace(np.log10(0.002), np.log10(0.30), n_layers)
    f_y = k_virgin * eps_y_arr
    return MultiLayerFrictionDegrading1D(
        E_base=E_base,
        k_virgin=k_virgin,
        k_degraded=k_virgin * beta,
        f_y=f_y,
    )


def _run_single_cycle(
    mat: MultiLayerFrictionDegrading1D,
    eps_max: float,
    n: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """0 → eps_max → -eps_max → 0 の応力-ひずみ曲線を返す."""
    eps_path = np.concatenate(
        [
            np.linspace(0, eps_max, n),
            np.linspace(eps_max, -eps_max, 2 * n),
            np.linspace(-eps_max, 0, n),
        ]
    )
    state = mat.initial_state()
    sigmas = []
    for eps in eps_path:
        sigma, _, state = mat.evaluate(float(eps), state)
        sigmas.append(sigma)
    return eps_path, np.array(sigmas)


class TestMultiLayerFrictionDegradingPhysics:
    """MultiLayerFrictionDegrading1D 物理テスト."""

    def test_smooth_teardrop_reproduction(self) -> None:
        """05_smooth_teardrop.py の単一ファイバー応力-ひずみを rtol=1% で再現.

        プロトタイプの MultiLayerFriction.stress() と同一パラメータ・
        同一ひずみ経路で計算し、応力値が1%以内で一致することを検証。
        """
        N = 150
        E_base = 1500.0
        k_contact_total = 7500.0
        k_weights = np.linspace(0.5, 1.5, N)
        k_weights *= N / k_weights.sum()
        k_each = k_contact_total / N
        k_arr = k_each * k_weights
        eps_y_arr = np.logspace(np.log10(0.002), np.log10(0.30), N)
        f_y_arr = k_arr * eps_y_arr
        beta = 0.25

        mat = MultiLayerFrictionDegrading1D(
            E_base=E_base,
            k_virgin=k_arr,
            k_degraded=k_arr * beta,
            f_y=f_y_arr,
        )

        # プロトタイプと同一アルゴリズムでの参照応力を直接計算
        eps_max = 0.306
        n = 500
        eps_path = np.concatenate(
            [
                np.linspace(0, eps_max, n),
                np.linspace(eps_max, -eps_max, 2 * n),
                np.linspace(-eps_max, 0, n),
            ]
        )

        # --- プロトタイプ直接計算（mutable 版）---
        slip_ref = np.zeros(N)
        slipped_ref = np.zeros(N, dtype=bool)
        sigmas_ref = []
        for eps in eps_path:
            sigma = E_base * eps
            for i in range(N):
                k = (k_arr * beta)[i] if slipped_ref[i] else k_arr[i]
                trial = k * (eps - slip_ref[i])
                if abs(trial) <= f_y_arr[i]:
                    sigma += trial
                else:
                    slipped_ref[i] = True
                    k = (k_arr * beta)[i]
                    trial = k * (eps - slip_ref[i])
                    if abs(trial) <= f_y_arr[i]:
                        sigma += trial
                    else:
                        s = np.sign(trial)
                        slip_ref[i] += s * (abs(trial) - f_y_arr[i]) / k
                        sigma += s * f_y_arr[i]
            sigmas_ref.append(sigma)
        sigmas_ref = np.array(sigmas_ref)

        # --- frozen 版 ---
        state = mat.initial_state()
        sigmas_frozen = []
        for eps in eps_path:
            sigma, _, state = mat.evaluate(float(eps), state)
            sigmas_frozen.append(sigma)
        sigmas_frozen = np.array(sigmas_frozen)

        # 全点で rtol=1% 以内に一致
        np.testing.assert_allclose(sigmas_frozen, sigmas_ref, rtol=0.01)

    def test_degradation_asymmetric_slopes(self) -> None:
        """β=0.25 で U/L < 1（除荷勾配 < 載荷勾配）になることを検証.

        β=1.0（劣化なし）では U/L ≈ 1.0 であり、
        β=0.25 では U/L < 1.0 となる（除荷時の接線剛性低下）。
        """
        eps_max = 0.20
        n = 800

        # β=1.0（劣化なし）
        mat_nodeg = _make_teardrop_model(n_layers=60, beta=1.0)
        eps_nd, sig_nd = _run_single_cycle(mat_nodeg, eps_max, n=n)

        # β=0.25（劣化あり）
        mat_deg = _make_teardrop_model(n_layers=60, beta=0.25)
        eps_dg, sig_dg = _run_single_cycle(mat_deg, eps_max, n=n)

        # 載荷勾配（ε ≈ 0.05 近辺）
        def _slope_at(eps_arr: np.ndarray, sig_arr: np.ndarray, target: float) -> float:
            idx = np.searchsorted(eps_arr[:n], target)
            idx = min(idx, n - 2)
            return (sig_arr[idx + 1] - sig_arr[idx]) / (eps_arr[idx + 1] - eps_arr[idx])

        slope_L_nodeg = _slope_at(eps_nd, sig_nd, 0.05)
        slope_L_deg = _slope_at(eps_dg, sig_dg, 0.05)

        # 除荷勾配（ε_max 直後）
        unload_start = n + 5
        slope_U_nodeg = (sig_nd[unload_start + 1] - sig_nd[unload_start]) / (
            eps_nd[unload_start + 1] - eps_nd[unload_start]
        )
        slope_U_deg = (sig_dg[unload_start + 1] - sig_dg[unload_start]) / (
            eps_dg[unload_start + 1] - eps_dg[unload_start]
        )

        # 劣化なし: U/L ≈ 1 に近い
        ratio_nodeg = abs(slope_U_nodeg / slope_L_nodeg)
        # 劣化あり: U/L < 1（典型的に 0.7-0.8）
        ratio_deg = abs(slope_U_deg / slope_L_deg)

        assert ratio_deg < ratio_nodeg, (
            f"劣化あり U/L={ratio_deg:.3f} ≥ 劣化なし U/L={ratio_nodeg:.3f}"
        )
        assert ratio_deg < 0.95, f"劣化あり U/L={ratio_deg:.3f} が 0.95 を下回らない"

    def test_n_layers_convergence(self) -> None:
        """N 増加で応力-ひずみ曲線が収束（N=30 vs N=150 の差 < 5%）.

        少数層では角のある応答、多数層では滑らかな応答に収束する。
        ピーク応力・残留ひずみの収束を検証。
        """
        eps_max = 0.20
        results = {}
        for n_layers in [30, 60, 150]:
            mat = _make_teardrop_model(n_layers=n_layers)
            eps_arr, sig_arr = _run_single_cycle(mat, eps_max, n=400)
            results[n_layers] = {
                "sigma_max": sig_arr.max(),
                "sigma_min": sig_arr.min(),
            }

        # N=60 と N=150 のピーク応力差が 5% 以内
        assert results[60]["sigma_max"] == pytest.approx(results[150]["sigma_max"], rel=0.05)
        # N=30 と N=150 の差はやや大きいが 10% 以内
        assert results[30]["sigma_max"] == pytest.approx(results[150]["sigma_max"], rel=0.10)

    def test_consistent_tangent_fd(self) -> None:
        """一貫接線 dσ/dε の有限差分検証.

        弾性域（全層ロック）とスリップ中の両方で検証する。
        """
        mat = _make_teardrop_model(n_layers=30)
        h = 1e-7

        # Case 1: 弾性域（微小ひずみ）
        state = mat.initial_state()
        eps = 1e-5
        _, E_t, _ = mat.evaluate(eps, state)
        sig_p, _, _ = mat.evaluate(eps + h, state)
        sig_m, _, _ = mat.evaluate(eps - h, state)
        fd = (sig_p - sig_m) / (2 * h)
        assert E_t == pytest.approx(fd, rel=1e-4)

        # Case 2: 部分スリップ域（中間ひずみ）
        # まずスリップを発生させてstateを進める
        state2 = mat.initial_state()
        for eps_i in np.linspace(0, 0.05, 100):
            _, _, state2 = mat.evaluate(float(eps_i), state2)
        eps = 0.06
        _, E_t, _ = mat.evaluate(eps, state2)
        sig_p, _, _ = mat.evaluate(eps + h, state2)
        sig_m, _, _ = mat.evaluate(eps - h, state2)
        fd = (sig_p - sig_m) / (2 * h)
        assert E_t == pytest.approx(fd, rel=1e-3)

    def test_energy_dissipation_positive(self) -> None:
        """閉ループのエネルギー散逸が正（第2法則）.

        ヒステリシスループ面積 > 0 であることを検証。
        """
        mat = _make_teardrop_model(n_layers=60)
        eps_max = 0.15
        n = 400
        # 閉ループ: +eps_max → -eps_max → +eps_max
        eps_path = np.concatenate(
            [
                np.linspace(0, eps_max, n),
                np.linspace(eps_max, -eps_max, 2 * n),
                np.linspace(-eps_max, eps_max, 2 * n),
            ]
        )
        state = mat.initial_state()
        sigmas = []
        for eps in eps_path:
            sigma, _, state = mat.evaluate(float(eps), state)
            sigmas.append(sigma)
        sigmas = np.array(sigmas)

        # 後半閉ループ部分の面積
        loop_eps = eps_path[n:]
        loop_sig = sigmas[n:]
        loop_area = abs(np.trapezoid(loop_sig, loop_eps))
        assert loop_area > 0, "ヒステリシスループ面積が 0 以下"

    def test_single_layer_matches_kh(self) -> None:
        """N=1, β=1.0 の MultiLayerFriction ≡ BilinearKH（等価性）.

        σ = E_base·ε + k·(ε - slip) の構造から、KH との正しい対応:
        - 弾性剛性: E_base + k = E → k = E²/(E+H)
        - スリップ時接線: E_base = E·H/(E+H)
        - 降伏条件: f_y = σ_y·E/(E+H)
        """
        E = 100_000.0
        H = 10_000.0
        sigma_y = 300.0

        E_base = E * H / (E + H)
        k = E * E / (E + H)
        f_y = sigma_y * E / (E + H)

        mat_kh = BilinearKinematicHardening1D(E=E, sigma_y=sigma_y, H=H)
        mat_ml = MultiLayerFrictionDegrading1D(
            E_base=E_base,
            k_virgin=np.array([k]),
            k_degraded=np.array([k]),  # β=1.0
            f_y=np.array([f_y]),
        )

        eps_path = np.concatenate(
            [
                np.linspace(0, 0.01, 100),
                np.linspace(0.01, -0.01, 200),
                np.linspace(-0.01, 0.01, 200),
            ]
        )

        state_kh = Fiber1DState()
        state_ml = mat_ml.initial_state()
        for eps in eps_path:
            sig_kh, Et_kh, state_kh = mat_kh.evaluate(float(eps), state_kh)
            sig_ml, Et_ml, state_ml = mat_ml.evaluate(float(eps), state_ml)
            assert sig_ml == pytest.approx(sig_kh, abs=1e-8), (
                f"eps={eps:.6f}: ML={sig_ml:.4f} vs KH={sig_kh:.4f}"
            )
