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


# ====================================================================
# HHT-α 法テスト
# ====================================================================


class TestHHTAlpha:
    """HHT-α法の数値減衰特性を検証."""

    def test_hht_reduces_to_newmark(self):
        """α=0 で標準 Newmark（平均加速度法）と同一結果."""
        m, k = 1.0, 100.0
        omega_n = np.sqrt(k / m)
        T = 2.0 * np.pi / omega_n
        u0_val = 1.0

        dt = T / 50.0
        n_steps = int(3 * T / dt)

        M, C, K = _sdof_matrices(m, 0.0, k)

        # 標準 Newmark
        cfg_nm = TransientConfig(dt=dt, n_steps=n_steps, alpha_hht=0.0)
        res_nm = solve_transient(M, C, K, np.zeros(1), np.array([u0_val]), np.zeros(1), cfg_nm)

        # HHT α=0 明示指定
        cfg_hht = TransientConfig(dt=dt, n_steps=n_steps, beta=0.25, gamma=0.5, alpha_hht=0.0)
        res_hht = solve_transient(M, C, K, np.zeros(1), np.array([u0_val]), np.zeros(1), cfg_hht)

        np.testing.assert_allclose(res_hht.displacement, res_nm.displacement, atol=1e-14)
        np.testing.assert_allclose(res_hht.velocity, res_nm.velocity, atol=1e-14)

    def test_hht_numerical_damping(self):
        """HHT-α (α=-0.1) は高周波成分を数値的に減衰させる.

        非減衰 SDOF で α<0 のとき、振幅が徐々に減少する（数値散逸）。
        α=0（標準 Newmark）と比較して振幅が小さくなることを確認。
        """
        m, k = 1.0, 100.0
        omega_n = np.sqrt(k / m)
        T = 2.0 * np.pi / omega_n
        u0_val = 1.0

        dt = T / 20.0  # 粗い刻みで数値減衰が顕著に出る
        n_steps = int(10 * T / dt)

        M, C, K = _sdof_matrices(m, 0.0, k)

        # 標準 Newmark（数値減衰なし）
        alpha_hht = -0.1
        gamma_hht = (1.0 - 2.0 * alpha_hht) / 2.0
        beta_hht = (1.0 - alpha_hht) ** 2 / 4.0

        cfg_nm = TransientConfig(dt=dt, n_steps=n_steps, beta=0.25, gamma=0.5)
        res_nm = solve_transient(M, C, K, np.zeros(1), np.array([u0_val]), np.zeros(1), cfg_nm)

        # HHT-α（数値減衰あり）
        cfg_hht = TransientConfig(
            dt=dt, n_steps=n_steps, beta=beta_hht, gamma=gamma_hht, alpha_hht=alpha_hht
        )
        res_hht = solve_transient(M, C, K, np.zeros(1), np.array([u0_val]), np.zeros(1), cfg_hht)

        # 後半の振幅を比較
        idx_half = n_steps // 2
        amp_nm = np.max(np.abs(res_nm.displacement[idx_half:, 0]))
        amp_hht = np.max(np.abs(res_hht.displacement[idx_half:, 0]))

        # HHT の振幅は Newmark より小さい（数値減衰のため）
        assert amp_hht < amp_nm, f"HHT振幅 ({amp_hht:.6f}) が Newmark振幅 ({amp_nm:.6f}) より大きい"

    def test_hht_step_response_converges(self):
        """HHT-α でもステップ荷重の静的変位に収束する."""
        m, k = 1.0, 400.0
        omega_n = np.sqrt(k / m)
        T = 2.0 * np.pi / omega_n
        F0 = 20.0
        u_static = F0 / k

        # 減衰を入れて速く収束させる
        xi = 0.1
        c = 2.0 * xi * omega_n * m

        alpha_hht = -0.05
        gamma_hht = (1.0 - 2.0 * alpha_hht) / 2.0
        beta_hht = (1.0 - alpha_hht) ** 2 / 4.0

        dt = T / 50.0
        n_steps = int(20 * T / dt)

        M, C, K = _sdof_matrices(m, c, k)
        cfg = TransientConfig(
            dt=dt, n_steps=n_steps, beta=beta_hht, gamma=gamma_hht, alpha_hht=alpha_hht
        )
        result = solve_transient(M, C, K, np.array([F0]), np.zeros(1), np.zeros(1), cfg)

        # 最終変位が静的変位に収束
        u_final = result.displacement[-1, 0]
        err = abs(u_final - u_static) / u_static
        assert err < 0.01, f"HHTステップ応答の定常偏差: {err:.6e}"


# ====================================================================
# 梁の過渡応答テスト
# ====================================================================


def _build_cantilever_beam_matrices(
    n_elems: int,
    L: float,
    E: float,
    rho: float,
    A: float,
    I: float,  # noqa: E741
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2D カンチレバー梁の M, C, K 行列と固定 DOF を構築する.

    Euler-Bernoulli 梁。節点0を固定。

    Returns:
        M, C, K, fixed_dofs
    """
    from xkep_cae.elements.beam_eb2d import eb_beam2d_ke_global
    from xkep_cae.numerical_tests.core import generate_beam_mesh_2d
    from xkep_cae.numerical_tests.frequency import _assemble_mass_2d
    from xkep_cae.numerical_tests.runner import _assemble_2d

    nodes, conn = generate_beam_mesh_2d(n_elems, L)

    ke_func = lambda coords: eb_beam2d_ke_global(coords, E, A, I)  # noqa: E731

    K_full, _ = _assemble_2d(nodes, conn, ke_func)
    M_full = _assemble_mass_2d(nodes, conn, rho, A)

    # 固定端: 節点0 の全DOF (ux, uy, θz)
    fixed_dofs = np.array([0, 1, 2], dtype=int)

    # 減衰なし
    ndof = K_full.shape[0]
    C_full = np.zeros((ndof, ndof))

    return M_full, C_full, K_full, fixed_dofs


class TestNewmarkBeam:
    """梁の過渡応答テスト."""

    def test_cantilever_free_vibration_frequency(self):
        """カンチレバー梁の自由振動周波数が解析解と一致.

        解析解: f₁ = (β₁L)² / (2πL²) · √(EI/ρA)
        β₁L = 1.8751（1次モード）

        第1固有モードの形状で初期変位を与え、FFTでピーク周波数を検出。
        """
        from scipy.linalg import eigh

        L = 1.0
        E = 2.0e11  # 鋼
        rho = 7800.0
        b, h = 0.02, 0.04
        A = b * h
        Iz = b * h**3 / 12.0
        n_elems = 20

        # 解析解
        beta1L = 1.8751
        f1_exact = beta1L**2 / (2.0 * np.pi * L**2) * np.sqrt(E * Iz / (rho * A))
        T1 = 1.0 / f1_exact

        M, C, K, fixed = _build_cantilever_beam_matrices(n_elems, L, E, rho, A, Iz)
        ndof = K.shape[0]

        # 第1固有モード形状を初期変位に使用（高次モード励起を回避）
        free = np.array([i for i in range(ndof) if i not in fixed])
        K_ff = K[np.ix_(free, free)]
        M_ff = M[np.ix_(free, free)]
        eigvals, eigvecs = eigh(K_ff, M_ff)
        phi1 = eigvecs[:, 0]

        u0 = np.zeros(ndof)
        u0[free] = phi1 * 0.001 / np.max(np.abs(phi1))  # 正規化して 1mm スケール

        tip_uy_dof = 3 * n_elems + 1

        dt = T1 / 80.0  # Nyquist: f_s = 80/T1 → 十分高い
        n_steps = int(10 * T1 / dt)

        cfg = TransientConfig(dt=dt, n_steps=n_steps)
        result = solve_transient(M, C, K, np.zeros(ndof), u0, np.zeros(ndof), cfg, fixed_dofs=fixed)

        # FFT でピーク周波数を検出
        u_tip = result.displacement[:, tip_uy_dof]
        N = len(u_tip)
        freqs = np.fft.rfftfreq(N, d=dt)
        fft_mag = np.abs(np.fft.rfft(u_tip))
        # DC 成分をスキップ
        peak_idx = np.argmax(fft_mag[1:]) + 1
        f1_num = freqs[peak_idx]

        err = abs(f1_num - f1_exact) / f1_exact
        assert err < 0.03, f"固有振動数の誤差: {err:.3f} ({f1_num:.2f} vs {f1_exact:.2f} Hz)"

    def test_cantilever_energy_conservation(self):
        """非減衰カンチレバー梁で全エネルギーが保存される.

        E_total = 0.5·u^T·K·u + 0.5·v^T·M·v = const
        """
        L = 0.5
        E = 2.0e11
        rho = 7800.0
        b, h = 0.01, 0.02
        A = b * h
        Iz = b * h**3 / 12.0
        n_elems = 10

        beta1L = 1.8751
        f1 = beta1L**2 / (2.0 * np.pi * L**2) * np.sqrt(E * Iz / (rho * A))
        T1 = 1.0 / f1

        M, C, K, fixed = _build_cantilever_beam_matrices(n_elems, L, E, rho, A, Iz)
        ndof = K.shape[0]

        tip_uy_dof = 3 * n_elems + 1
        u0 = np.zeros(ndof)
        u0[tip_uy_dof] = 0.0005

        dt = T1 / 40.0
        n_steps = int(5 * T1 / dt)

        cfg = TransientConfig(dt=dt, n_steps=n_steps)
        result = solve_transient(M, C, K, np.zeros(ndof), u0, np.zeros(ndof), cfg, fixed_dofs=fixed)

        # 自由DOFでのエネルギー計算
        free = np.array([i for i in range(ndof) if i not in fixed])
        K_ff = K[np.ix_(free, free)]
        M_ff = M[np.ix_(free, free)]

        energy = np.zeros(n_steps + 1)
        for i in range(n_steps + 1):
            u_f = result.displacement[i, free]
            v_f = result.velocity[i, free]
            energy[i] = 0.5 * u_f @ K_ff @ u_f + 0.5 * v_f @ M_ff @ v_f

        E0 = energy[0]
        assert E0 > 0, "初期エネルギーがゼロ"

        err_rel = np.max(np.abs(energy - E0)) / E0
        assert err_rel < 0.01, f"梁のエネルギー保存誤差: {err_rel:.6e}"

    def test_cantilever_tip_load_static_limit(self):
        """先端集中荷重のステップ応答が静的解に収束する.

        減衰付きで十分長い時間経過後、u → u_static = PL³/(3EI)
        """
        L = 0.5
        E = 2.0e11
        rho = 7800.0
        b, h = 0.01, 0.02
        A = b * h
        Iz = b * h**3 / 12.0
        P = 100.0  # 先端荷重 [N]
        n_elems = 10

        u_static = P * L**3 / (3.0 * E * Iz)

        beta1L = 1.8751
        f1 = beta1L**2 / (2.0 * np.pi * L**2) * np.sqrt(E * Iz / (rho * A))
        omega1 = 2.0 * np.pi * f1
        T1 = 1.0 / f1

        M, _, K, fixed = _build_cantilever_beam_matrices(n_elems, L, E, rho, A, Iz)
        ndof = K.shape[0]

        # Rayleigh 減衰（ξ₁ ≈ 10%）
        xi1 = 0.1
        alpha_r = 0.0
        beta_r = 2.0 * xi1 / omega1
        C = alpha_r * M + beta_r * K

        # 先端荷重ベクトル
        tip_uy_dof = 3 * n_elems + 1
        f = np.zeros(ndof)
        f[tip_uy_dof] = -P  # 下向き

        dt = T1 / 40.0
        n_steps = int(20 * T1 / dt)

        cfg = TransientConfig(dt=dt, n_steps=n_steps)
        result = solve_transient(M, C, K, f, np.zeros(ndof), np.zeros(ndof), cfg, fixed_dofs=fixed)

        # 最終変位が静的解に収束（下向き = 負）
        u_final = -result.displacement[-1, tip_uy_dof]  # 符号反転して正に
        err = abs(u_final - u_static) / u_static
        assert err < 0.05, f"先端変位の静的収束誤差: {err:.3f} ({u_final:.6e} vs {u_static:.6e})"


# ====================================================================
# 集中質量行列テスト
# ====================================================================


class TestLumpedMass:
    """集中質量行列（HRZ法）の検証."""

    def test_lumped_mass_2d_is_diagonal(self):
        """2D集中質量行列が対角行列."""
        from xkep_cae.numerical_tests.frequency import _beam2d_lumped_mass_local

        Me = _beam2d_lumped_mass_local(rho=7800.0, A=0.001, L=0.1)
        # 非対角要素がゼロ
        assert np.allclose(Me, np.diag(np.diag(Me)))
        # 全対角要素が正
        assert np.all(np.diag(Me) > 0)

    def test_lumped_mass_3d_is_diagonal(self):
        """3D集中質量行列が対角行列."""
        from xkep_cae.numerical_tests.frequency import _beam3d_lumped_mass_local

        Me = _beam3d_lumped_mass_local(rho=7800.0, A=0.001, Iy=1e-8, Iz=1e-8, L=0.1)
        assert np.allclose(Me, np.diag(np.diag(Me)))
        assert np.all(np.diag(Me) > 0)

    def test_lumped_mass_2d_total_mass_preserved(self):
        """2D集中質量の並進方向合計が要素質量と一致."""
        from xkep_cae.numerical_tests.frequency import _beam2d_lumped_mass_local

        rho, A, L = 7800.0, 0.002, 0.5
        m_elem = rho * A * L
        Me = _beam2d_lumped_mass_local(rho, A, L)

        # 軸方向（DOF 0, 3）の合計 = m
        assert np.isclose(Me[0, 0] + Me[3, 3], m_elem)
        # 横方向（DOF 1, 4）の合計 = m
        assert np.isclose(Me[1, 1] + Me[4, 4], m_elem)

    def test_lumped_mass_3d_total_mass_preserved(self):
        """3D集中質量の並進方向合計が要素質量と一致."""
        from xkep_cae.numerical_tests.frequency import _beam3d_lumped_mass_local

        rho, A, L = 7800.0, 0.002, 0.5
        Iy, Iz = 1e-8, 2e-8
        m_elem = rho * A * L
        Me = _beam3d_lumped_mass_local(rho, A, Iy, Iz, L)

        # 各並進方向の合計 = m
        for d in [0, 1, 2]:  # ux, uy, uz
            assert np.isclose(Me[d, d] + Me[d + 6, d + 6], m_elem)

    def test_lumped_mass_2d_global_total_mass(self):
        """2Dグローバル集中質量の全体質量が正しい."""
        from xkep_cae.numerical_tests.core import generate_beam_mesh_2d
        from xkep_cae.numerical_tests.frequency import _assemble_lumped_mass_2d

        n_elems = 10
        L = 2.0
        rho, A = 7800.0, 0.001
        m_total = rho * A * L

        nodes, conn = generate_beam_mesh_2d(n_elems, L)
        M = _assemble_lumped_mass_2d(nodes, conn, rho, A)

        # 対角行列
        assert np.allclose(M, np.diag(np.diag(M)))

        # ux 方向（DOF 0, 3, 6, ...）の全合計 = m_total
        ux_dofs = np.arange(0, M.shape[0], 3)
        m_sum = np.sum(M[ux_dofs, ux_dofs])
        assert np.isclose(m_sum, m_total, rtol=1e-12)

    def test_lumped_mass_3d_global_total_mass(self):
        """3Dグローバル集中質量の全体質量が正しい."""
        from xkep_cae.numerical_tests.core import generate_beam_mesh_3d
        from xkep_cae.numerical_tests.frequency import _assemble_lumped_mass_3d

        n_elems = 8
        L = 1.5
        rho, A = 7800.0, 0.002
        Iy, Iz = 1e-8, 2e-8
        m_total = rho * A * L

        nodes, conn = generate_beam_mesh_3d(n_elems, L)
        M = _assemble_lumped_mass_3d(nodes, conn, rho, A, Iy, Iz)

        # 対角行列
        assert np.allclose(M, np.diag(np.diag(M)))

        # ux 方向の合計 = m_total
        ux_dofs = np.arange(0, M.shape[0], 6)
        m_sum = np.sum(M[ux_dofs, ux_dofs])
        assert np.isclose(m_sum, m_total, rtol=1e-12)

    def test_lumped_vs_consistent_eigenfreq_convergence(self):
        """集中質量でも固有振動数が解析解に収束する.

        メッシュ細分化で集中・整合の差が小さくなることを確認。
        """
        from scipy.linalg import eigh

        from xkep_cae.elements.beam_eb2d import eb_beam2d_ke_global
        from xkep_cae.numerical_tests.core import generate_beam_mesh_2d
        from xkep_cae.numerical_tests.frequency import (
            _assemble_lumped_mass_2d,
            _assemble_mass_2d,
        )
        from xkep_cae.numerical_tests.runner import _assemble_2d

        L = 1.0
        E_mod = 2.0e11
        rho = 7800.0
        b, h = 0.02, 0.04
        A = b * h
        Iz = b * h**3 / 12.0

        beta1L = 1.8751
        f1_exact = beta1L**2 / (2.0 * np.pi * L**2) * np.sqrt(E_mod * Iz / (rho * A))

        fixed_dofs = np.array([0, 1, 2])

        for n_elems in [10, 20]:
            nodes, conn = generate_beam_mesh_2d(n_elems, L)
            ke_func = lambda coords: eb_beam2d_ke_global(coords, E_mod, A, Iz)  # noqa: E731
            K_full, _ = _assemble_2d(nodes, conn, ke_func)

            M_con = _assemble_mass_2d(nodes, conn, rho, A)
            M_lum = _assemble_lumped_mass_2d(nodes, conn, rho, A)

            ndof = K_full.shape[0]
            free = np.array([i for i in range(ndof) if i not in fixed_dofs])
            K_ff = K_full[np.ix_(free, free)]

            # 整合質量行列の第1固有振動数
            eigvals_c, _ = eigh(K_ff, M_con[np.ix_(free, free)])
            f1_con = np.sqrt(eigvals_c[0]) / (2.0 * np.pi)

            # 集中質量行列の第1固有振動数
            eigvals_l, _ = eigh(K_ff, M_lum[np.ix_(free, free)])
            f1_lum = np.sqrt(eigvals_l[0]) / (2.0 * np.pi)

            err_con = abs(f1_con - f1_exact) / f1_exact
            err_lum = abs(f1_lum - f1_exact) / f1_exact

            # 両方 5% 未満
            assert err_con < 0.05, f"整合: n={n_elems}, err={err_con:.4f}"
            assert err_lum < 0.05, f"集中: n={n_elems}, err={err_lum:.4f}"

    def test_lumped_mass_beam_transient(self):
        """集中質量でカンチレバー先端荷重が静的解に収束する."""
        from xkep_cae.elements.beam_eb2d import eb_beam2d_ke_global
        from xkep_cae.numerical_tests.core import generate_beam_mesh_2d
        from xkep_cae.numerical_tests.frequency import _assemble_lumped_mass_2d
        from xkep_cae.numerical_tests.runner import _assemble_2d

        L = 0.5
        E_mod = 2.0e11
        rho = 7800.0
        b, h = 0.01, 0.02
        A = b * h
        Iz = b * h**3 / 12.0
        P = 100.0
        n_elems = 10

        u_static = P * L**3 / (3.0 * E_mod * Iz)

        beta1L = 1.8751
        f1 = beta1L**2 / (2.0 * np.pi * L**2) * np.sqrt(E_mod * Iz / (rho * A))
        omega1 = 2.0 * np.pi * f1
        T1 = 1.0 / f1

        nodes, conn = generate_beam_mesh_2d(n_elems, L)
        ke_func = lambda coords: eb_beam2d_ke_global(coords, E_mod, A, Iz)  # noqa: E731
        K, _ = _assemble_2d(nodes, conn, ke_func)
        M = _assemble_lumped_mass_2d(nodes, conn, rho, A)
        ndof = K.shape[0]
        fixed = np.array([0, 1, 2])

        # Rayleigh 減衰
        xi1 = 0.1
        beta_r = 2.0 * xi1 / omega1
        C = beta_r * K

        tip_uy_dof = 3 * n_elems + 1
        f = np.zeros(ndof)
        f[tip_uy_dof] = -P

        dt = T1 / 40.0
        n_steps = int(20 * T1 / dt)

        cfg = TransientConfig(dt=dt, n_steps=n_steps)
        result = solve_transient(M, C, K, f, np.zeros(ndof), np.zeros(ndof), cfg, fixed_dofs=fixed)

        u_final = -result.displacement[-1, tip_uy_dof]
        err = abs(u_final - u_static) / u_static
        assert err < 0.05, f"集中質量の静的収束誤差: {err:.3f} ({u_final:.6e} vs {u_static:.6e})"
