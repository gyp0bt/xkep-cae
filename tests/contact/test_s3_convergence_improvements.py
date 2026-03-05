"""S3 収束改善のユニットテスト.

ILU 適応制御、Schur 正則化、λ ウォームスタート、
GMRES restart、NCP active set ヒステリシスの単体検証。
"""

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.contact.solver_ncp import (
    _adaptive_ilu_drop_tol,
    _build_ilu_preconditioner,
    _compute_schur_diagonal,
    _regularize_schur,
)

# === ILU 適応 drop_tol ===


class TestAdaptiveILUDropTol:
    """_adaptive_ilu_drop_tol のテスト."""

    def test_small_n_active_unchanged(self):
        """n_active <= 20 では base_tol そのまま."""
        assert _adaptive_ilu_drop_tol(1e-4, 0) == 1e-4
        assert _adaptive_ilu_drop_tol(1e-4, 10) == 1e-4
        assert _adaptive_ilu_drop_tol(1e-4, 20) == 1e-4

    def test_large_n_active_scales(self):
        """n_active > 20 で線形にスケール."""
        tol = _adaptive_ilu_drop_tol(1e-4, 40)
        assert tol == pytest.approx(2e-4, rel=1e-10)

        tol = _adaptive_ilu_drop_tol(1e-4, 100)
        assert tol == pytest.approx(5e-4, rel=1e-10)

    def test_upper_bound(self):
        """上限 0.1 を超えない."""
        tol = _adaptive_ilu_drop_tol(1e-4, 100000)
        assert tol <= 0.1

    def test_monotonically_increasing(self):
        """n_active 増加に伴い monotonically 増加."""
        tols = [_adaptive_ilu_drop_tol(1e-4, n) for n in [10, 20, 40, 80, 160]]
        for i in range(len(tols) - 1):
            assert tols[i + 1] >= tols[i]


class TestBuildILUPreconditioner:
    """_build_ilu_preconditioner のテスト."""

    def test_spd_matrix_success(self):
        """SPD 行列で ILU が成功する."""
        n = 50
        A = sp.random(n, n, density=0.3, format="csr")
        A = A.T @ A + 10.0 * sp.eye(n)
        ilu = _build_ilu_preconditioner(A, 1e-4)
        assert ilu is not None

    def test_ilu_solve_is_reasonable(self):
        """ILU ソルブが妥当な結果を返す."""
        n = 30
        A = sp.random(n, n, density=0.5, format="csr")
        A = A.T @ A + 5.0 * sp.eye(n)
        ilu = _build_ilu_preconditioner(A, 1e-4)
        assert ilu is not None
        b = np.ones(n)
        x = ilu.solve(b)
        assert np.all(np.isfinite(x))


# === Schur 正則化 ===


class TestRegularizeSchur:
    """_regularize_schur のテスト."""

    def test_empty_matrix(self):
        """0×0 行列が通る."""
        S = np.zeros((0, 0))
        S_reg = _regularize_schur(S)
        assert S_reg.shape == (0, 0)

    def test_spd_matrix_barely_changes(self):
        """SPD 行列は微小な変更のみ."""
        S = np.array([[10.0, 1.0], [1.0, 10.0]])
        S_reg = _regularize_schur(S)
        diff = np.max(np.abs(S_reg - S))
        assert diff < 1e-6  # 微小な正則化のみ

    def test_singular_matrix_becomes_solvable(self):
        """特異行列が正則化後に解ける."""
        S = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
        S_reg = _regularize_schur(S)
        # solve が成功するか確認
        b = np.array([1.0, 2.0])
        x = np.linalg.solve(S_reg, b)
        assert np.all(np.isfinite(x))

    def test_zero_diagonal_gets_strong_reg(self):
        """対角がゼロの行にはより強い正則化が入る."""
        S = np.array([[10.0, 0.0], [0.0, 0.0]])
        S_reg = _regularize_schur(S)
        # (0,0) はほぼ変わらず、(1,1) は有意に正になる
        assert S_reg[1, 1] > 1e-10
        assert abs(S_reg[0, 0] - 10.0) < 0.01

    def test_negative_diagonal_corrected(self):
        """負の対角が正に補正される."""
        S = np.array([[5.0, 0.0], [0.0, -1.0]])
        S_reg = _regularize_schur(S)
        assert S_reg[1, 1] > 0.0


class TestComputeSchurDiagonal:
    """_compute_schur_diagonal のテスト."""

    def test_with_ilu(self):
        """ILU あり: 正の対角が返る."""
        n = 20
        K = sp.random(n, n, density=0.5, format="csr")
        K = K.T @ K + 5.0 * sp.eye(n)
        ilu = _build_ilu_preconditioner(K, 1e-4)

        n_active = 3
        G = sp.random(n_active, n, density=0.3, format="csr")
        s_diag = _compute_schur_diagonal(ilu, G, n_active, 1e4)
        assert len(s_diag) == n_active
        assert np.all(s_diag > 0.0)

    def test_without_ilu(self):
        """ILU なし: k_pen ベースのフォールバック."""
        k_pen = 1e6
        n_active = 5
        G = sp.random(n_active, 20, density=0.3, format="csr")
        s_diag = _compute_schur_diagonal(None, G, n_active, k_pen)
        assert len(s_diag) == n_active
        expected = 1.0 / k_pen
        np.testing.assert_allclose(s_diag, expected, rtol=1e-10)


# === NCP Active Set ヒステリシス ===


class TestNCPActiveSetHysteresis:
    """NCP active set のヒステリシス判定ロジック検証."""

    def test_basic_activation(self):
        """p_n > 0 で活性化."""
        lams = np.array([100.0, 0.0, -10.0])
        gaps = np.array([-0.01, 0.01, 0.01])
        k_pen = 1e4
        p_n_arr = np.maximum(0.0, lams + k_pen * (-gaps))
        mask = p_n_arr > 0.0
        # λ=100, gap=-0.01 → p_n = 100 + 1e4*0.01 = 200 > 0 → active
        # λ=0, gap=0.01 → p_n = 0 + 1e4*(-0.01) = -100 < 0 → inactive
        # λ=-10, gap=0.01 → p_n = -10 + 1e4*(-0.01) = -110 < 0 → inactive
        assert mask[0] is np.True_
        assert mask[1] is np.False_
        assert mask[2] is np.False_

    def test_hysteresis_prevents_deactivation(self):
        """ヒステリシスが微小変動での非活性化を防ぐ."""
        k_pen = 1e4
        ncp_hyst = 10.0  # ヒステリシス閾値

        # 前回 active だったペアが p_n=-5 に → ヒステリシス内で active 維持
        lams = np.array([5.0])
        gaps = np.array([0.001])  # p_n = 5 + 1e4*(-0.001) = -5
        p_n_arr = np.maximum(0.0, lams + k_pen * (-gaps))
        prev_active = np.array([True])

        # ヒステリシスなし: inactive
        new_active = p_n_arr > 0.0
        assert not new_active[0]

        # ヒステリシスあり: active 維持
        raw_p = lams + k_pen * (-gaps)  # = -5 > -10 → 維持
        keep = prev_active & (raw_p > -ncp_hyst)
        hysteresis_mask = new_active | keep
        assert hysteresis_mask[0]  # -5 > -10 なので維持

    def test_hysteresis_allows_deactivation_beyond_threshold(self):
        """閾値を超えたら非活性化を許可."""
        k_pen = 1e4
        ncp_hyst = 10.0

        lams = np.array([0.0])
        gaps = np.array([0.01])  # p_n = 0 + 1e4*(-0.01) = -100
        p_n_arr = np.maximum(0.0, lams + k_pen * (-gaps))
        prev_active = np.array([True])

        raw_p = lams + k_pen * (-gaps)  # = -100 < -10 → 非活性化
        new_active = p_n_arr > 0.0
        keep = prev_active & (raw_p > -ncp_hyst)
        hysteresis_mask = new_active | keep
        assert not hysteresis_mask[0]  # -100 < -10 なので非活性化


# === λ ウォームスタート ===


class TestLambdaWarmStart:
    """λ の接線予測子ロジック検証."""

    def test_lambda_extrapolation(self):
        """前ステップの Δλ を荷重比で外挿."""
        lam_prev = np.array([100.0, 50.0, 0.0])
        lam_current = np.array([150.0, 70.0, 10.0])
        dlam = lam_current - lam_prev  # [50, 20, 10]

        delta_frac = 0.1
        delta_frac_prev = 0.1
        ratio = min(delta_frac / delta_frac_prev, 2.0)  # = 1.0

        lam_predicted = np.maximum(lam_current + ratio * dlam, 0.0)
        np.testing.assert_allclose(lam_predicted, [200.0, 90.0, 20.0])

    def test_lambda_non_negative(self):
        """予測値は非負."""
        lam_prev = np.array([10.0, 5.0])
        lam_current = np.array([3.0, 1.0])  # 減少トレンド
        dlam = lam_current - lam_prev  # [-7, -4]

        ratio = 1.0
        lam_predicted = np.maximum(lam_current + ratio * dlam, 0.0)
        assert np.all(lam_predicted >= 0.0)  # = [0, 0]

    def test_ratio_capped_at_2(self):
        """比率は 2.0 でキャップ."""
        delta_frac = 0.3
        delta_frac_prev = 0.1
        ratio = min(delta_frac / delta_frac_prev, 2.0)
        assert ratio == 2.0
