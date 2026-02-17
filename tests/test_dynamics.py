"""過渡応答解析（Newmark-β / HHT-α）のテスト.

解析解との比較で時間積分の正確性を検証する。

テスト構成:
- TestNewmarkSDOF: 1自由度系の自由振動・ステップ荷重・減衰付き振動
- TestNewmarkBeam: 梁の過渡応答（カンチレバー自由振動・エネルギー保存）
- TestHHTAlpha: HHT-α法の数値減衰特性
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.dynamics import TransientConfig, TransientResult, solve_transient

# ====================================================================
# ヘルパー
# ====================================================================


def _sdof_matrices(m: float, c: float, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1自由度系の M, C, K 行列を返す."""
    M = np.array([[m]])
    C = np.array([[c]])
    K = np.array([[k]])
    return M, C, K


# ====================================================================
# Newmark-β: 1自由度テスト
# ====================================================================


class TestNewmarkSDOF:
    """1自由度系での Newmark-β 検証."""

    def test_free_vibration_undamped(self):
        """非減衰自由振動: u(t) = u0·cos(ωₙt).

        解析解: u(t) = u0·cos(ωₙt), v(t) = -u0·ωₙ·sin(ωₙt)
        """
        m, k = 1.0, 100.0
        omega_n = np.sqrt(k / m)
        T = 2.0 * np.pi / omega_n  # 周期
        u0_val = 0.5

        # 10周期、1周期あたり200ステップ（周期伸び誤差の蓄積を抑制）
        n_periods = 10
        dt = T / 200.0
        n_steps = int(n_periods * T / dt)

        M, C, K = _sdof_matrices(m, 0.0, k)
        cfg = TransientConfig(dt=dt, n_steps=n_steps)

        result = solve_transient(
            M,
            C,
            K,
            f_ext=np.zeros(1),
            u0=np.array([u0_val]),
            v0=np.zeros(1),
            config=cfg,
        )

        assert isinstance(result, TransientResult)
        assert result.time.shape == (n_steps + 1,)
        assert result.displacement.shape == (n_steps + 1, 1)

        # 解析解と比較
        t = result.time
        u_exact = u0_val * np.cos(omega_n * t)
        u_num = result.displacement[:, 0]

        # 平均加速度法は非減衰系で振幅が保存される
        # 最後の5周期での誤差を評価
        idx_start = n_steps // 2
        err = np.max(np.abs(u_num[idx_start:] - u_exact[idx_start:]))
        assert err < 0.01 * u0_val, f"自由振動誤差が大きい: {err:.6e}"

    def test_free_vibration_amplitude_preservation(self):
        """非減衰自由振動で振幅が保存される（平均加速度法の特性）."""
        m, k = 2.0, 200.0
        omega_n = np.sqrt(k / m)
        T = 2.0 * np.pi / omega_n
        u0_val = 1.0

        dt = T / 50.0
        n_steps = int(20 * T / dt)

        M, C, K = _sdof_matrices(m, 0.0, k)
        cfg = TransientConfig(dt=dt, n_steps=n_steps)
        result = solve_transient(M, C, K, np.zeros(1), np.array([u0_val]), np.zeros(1), cfg)

        u = result.displacement[:, 0]
        v = result.velocity[:, 0]

        # エネルギー: E = 0.5*k*u² + 0.5*m*v²
        E = 0.5 * k * u**2 + 0.5 * m * v**2
        E0 = E[0]

        # 全ステップでエネルギーが保存される（平均加速度法は正確に保存はしないが非常に近い）
        err_rel = np.max(np.abs(E - E0)) / E0
        assert err_rel < 1e-3, f"エネルギー保存誤差: {err_rel:.6e}"

    def test_free_vibration_damped(self):
        """減衰付き自由振動: u(t) = u0·e^{-ξωₙt}·cos(ωd·t).

        解析解（減衰比 ξ < 1）:
            ωd = ωₙ√(1-ξ²)
            u(t) = u0·e^{-ξωₙt}·(cos(ωd·t) + ξωₙ/ωd·sin(ωd·t))
        """
        m, k = 1.0, 400.0
        omega_n = np.sqrt(k / m)
        xi = 0.05  # 減衰比 5%
        c = 2.0 * xi * omega_n * m
        omega_d = omega_n * np.sqrt(1.0 - xi**2)
        T_d = 2.0 * np.pi / omega_d
        u0_val = 1.0

        dt = T_d / 100.0
        n_steps = int(5 * T_d / dt)

        M, C, K = _sdof_matrices(m, c, k)
        cfg = TransientConfig(dt=dt, n_steps=n_steps)
        result = solve_transient(M, C, K, np.zeros(1), np.array([u0_val]), np.zeros(1), cfg)

        t = result.time
        u_exact = (
            u0_val
            * np.exp(-xi * omega_n * t)
            * (np.cos(omega_d * t) + xi * omega_n / omega_d * np.sin(omega_d * t))
        )
        u_num = result.displacement[:, 0]

        # 相対誤差（減衰エンベロープに対する）
        envelope = u0_val * np.exp(-xi * omega_n * t)
        mask = envelope > 0.01 * u0_val  # エンベロープが小さすぎる部分を除外
        err = np.max(np.abs(u_num[mask] - u_exact[mask])) / u0_val
        assert err < 0.005, f"減衰自由振動の誤差: {err:.6e}"

    def test_step_load_undamped(self):
        """非減衰ステップ荷重応答: u(t) = (F₀/k)·(1 - cos(ωₙt)).

        t=0 で F₀ の定常力が作用。初期変位・速度ゼロ。
        """
        m, k = 1.0, 100.0
        F0 = 10.0
        omega_n = np.sqrt(k / m)
        T = 2.0 * np.pi / omega_n

        dt = T / 100.0
        n_steps = int(5 * T / dt)

        M, C, K = _sdof_matrices(m, 0.0, k)
        cfg = TransientConfig(dt=dt, n_steps=n_steps)
        result = solve_transient(
            M,
            C,
            K,
            f_ext=np.array([F0]),
            u0=np.zeros(1),
            v0=np.zeros(1),
            config=cfg,
        )

        t = result.time
        u_exact = (F0 / k) * (1.0 - np.cos(omega_n * t))
        u_num = result.displacement[:, 0]

        err = np.max(np.abs(u_num - u_exact))
        u_static = F0 / k
        assert err < 0.01 * u_static, f"ステップ荷重応答の誤差: {err:.6e}"

    def test_step_load_damped(self):
        """減衰ステップ荷重応答.

        解析解: u(t) = (F₀/k)·[1 - e^{-ξωₙt}·(cos(ωd·t) + ξωₙ/ωd·sin(ωd·t))]
        """
        m, k = 1.0, 400.0
        omega_n = np.sqrt(k / m)
        xi = 0.1
        c = 2.0 * xi * omega_n * m
        omega_d = omega_n * np.sqrt(1.0 - xi**2)
        F0 = 20.0

        T_d = 2.0 * np.pi / omega_d
        dt = T_d / 100.0
        n_steps = int(5 * T_d / dt)

        M, C, K = _sdof_matrices(m, c, k)
        cfg = TransientConfig(dt=dt, n_steps=n_steps)
        result = solve_transient(M, C, K, np.array([F0]), np.zeros(1), np.zeros(1), cfg)

        t = result.time
        u_static = F0 / k
        u_exact = u_static * (
            1.0
            - np.exp(-xi * omega_n * t)
            * (np.cos(omega_d * t) + xi * omega_n / omega_d * np.sin(omega_d * t))
        )
        u_num = result.displacement[:, 0]

        err = np.max(np.abs(u_num - u_exact)) / u_static
        assert err < 0.01, f"減衰ステップ荷重の誤差: {err:.6e}"

    def test_harmonic_load_steady_state(self):
        """調和荷重の定常応答.

        F(t) = F₀·sin(Ωt), Ω/ωₙ = 0.5（共振遠方）
        定常応答振幅: |H(Ω)|·F₀/k
        """
        m, k = 1.0, 100.0
        omega_n = np.sqrt(k / m)
        xi = 0.02
        c = 2.0 * xi * omega_n * m
        F0 = 5.0
        freq_ratio = 0.5
        Omega = freq_ratio * omega_n

        T_exc = 2.0 * np.pi / Omega
        dt = T_exc / 80.0
        n_periods = 30  # 十分に定常状態に達する
        n_steps = int(n_periods * T_exc / dt)

        M, C, K = _sdof_matrices(m, c, k)
        cfg = TransientConfig(dt=dt, n_steps=n_steps)

        def f_harmonic(t: float) -> np.ndarray:
            return np.array([F0 * np.sin(Omega * t)])

        result = solve_transient(M, C, K, f_harmonic, np.zeros(1), np.zeros(1), cfg)

        # 解析解: 定常応答振幅
        r = freq_ratio
        H_mag = 1.0 / np.sqrt((1 - r**2) ** 2 + (2 * xi * r) ** 2)
        u_amp_exact = F0 / k * H_mag

        # 最後の3周期の振幅を取得
        idx_start = n_steps - int(3 * T_exc / dt)
        u_tail = result.displacement[idx_start:, 0]
        u_amp_num = (np.max(u_tail) - np.min(u_tail)) / 2.0

        err = abs(u_amp_num - u_amp_exact) / u_amp_exact
        assert err < 0.05, f"調和荷重の定常振幅誤差: {err:.3f}"

    def test_initial_velocity(self):
        """初速度のみの自由振動: u(t) = (v0/ωₙ)·sin(ωₙt)."""
        m, k = 1.0, 100.0
        omega_n = np.sqrt(k / m)
        T = 2.0 * np.pi / omega_n
        v0_val = 5.0

        dt = T / 200.0
        n_steps = int(5 * T / dt)

        M, C, K = _sdof_matrices(m, 0.0, k)
        cfg = TransientConfig(dt=dt, n_steps=n_steps)
        result = solve_transient(M, C, K, np.zeros(1), np.zeros(1), np.array([v0_val]), cfg)

        t = result.time
        u_exact = (v0_val / omega_n) * np.sin(omega_n * t)
        u_num = result.displacement[:, 0]

        err = np.max(np.abs(u_num - u_exact))
        u_max = v0_val / omega_n
        assert err < 0.01 * u_max, f"初速度応答の誤差: {err:.6e}"

    def test_fixed_dofs(self):
        """拘束DOFが正しく処理される（2DOF系で1DOFを固定）."""
        # 2DOF 直列ばね: m1-k1-m2-k2-壁
        # DOF 1（壁側）を固定
        m1, m2 = 1.0, 1.0
        k1, k2 = 100.0, 200.0

        M = np.diag([m1, m2])
        K = np.array([[k1 + k2, -k2], [-k2, k2]])
        C = np.zeros((2, 2))

        # DOF 1 を固定 → 実質 1DOF: m1·ä + (k1+k2)·u = 0
        omega_n = np.sqrt((k1 + k2) / m1)
        T = 2.0 * np.pi / omega_n
        u0_val = 0.3

        dt = T / 100.0
        n_steps = int(5 * T / dt)

        cfg = TransientConfig(dt=dt, n_steps=n_steps)
        result = solve_transient(
            M,
            C,
            K,
            np.zeros(2),
            np.array([u0_val, 0.0]),
            np.zeros(2),
            cfg,
            fixed_dofs=np.array([1]),
        )

        # DOF 1 はゼロのまま
        assert np.allclose(result.displacement[:, 1], 0.0)

        # DOF 0 は自由振動
        t = result.time
        u_exact = u0_val * np.cos(omega_n * t)
        err = np.max(np.abs(result.displacement[:, 0] - u_exact))
        assert err < 0.01 * u0_val, f"固定DOFテストの誤差: {err:.6e}"


# ====================================================================
# Newmark-β: コンフィグのバリデーション
# ====================================================================


class TestTransientConfig:
    """TransientConfig のバリデーション."""

    def test_negative_dt_raises(self):
        with pytest.raises(ValueError, match="dt"):
            TransientConfig(dt=-0.01, n_steps=10)

    def test_zero_steps_raises(self):
        with pytest.raises(ValueError, match="n_steps"):
            TransientConfig(dt=0.01, n_steps=0)

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError, match="alpha_hht"):
            TransientConfig(dt=0.01, n_steps=10, alpha_hht=-0.5)

    def test_negative_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            TransientConfig(dt=0.01, n_steps=10, beta=-0.1)

    def test_gamma_too_small_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            TransientConfig(dt=0.01, n_steps=10, gamma=0.3)
