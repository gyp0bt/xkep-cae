"""Levenberg-Marquardt 正則化の単体テスト.

LinearSolveProcess の lm_lambda パラメータによる正則化が
不定値行列の線形ソルブを安定化することを検証する。

status-239 で追加。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.solver._newton_steps import (
    LinearSolveInput,
    LinearSolveProcess,
)


class TestLMRegularizationAPI:
    """LM 正則化の API テスト."""

    def test_lambda_zero_no_effect(self):
        """λ=0 は正則化なし（従来動作）."""
        K = sp.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
        R = np.array([1.0, 2.0])
        fixed = np.array([], dtype=int)

        proc = LinearSolveProcess()
        out = proc.process(LinearSolveInput(K_T=K, R_u=R, fixed_dofs=fixed, lm_lambda=0.0))

        assert out.success
        # K @ du = -R
        expected = np.linalg.solve(K.toarray(), -R)
        np.testing.assert_allclose(out.du, expected, atol=1e-12)

    def test_lambda_positive_modifies_solution(self):
        """λ>0 で解が変化する."""
        K = sp.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
        R = np.array([1.0, 2.0])
        fixed = np.array([], dtype=int)

        proc = LinearSolveProcess()
        out0 = proc.process(LinearSolveInput(K_T=K, R_u=R, fixed_dofs=fixed, lm_lambda=0.0))
        out1 = proc.process(LinearSolveInput(K_T=K, R_u=R, fixed_dofs=fixed, lm_lambda=1.0))

        assert out0.success and out1.success
        # 解が異なることを確認
        assert not np.allclose(out0.du, out1.du)
        # 正則化で変位が小さくなる（信頼領域制約）
        assert np.linalg.norm(out1.du) < np.linalg.norm(out0.du)

    def test_default_lambda_zero(self):
        """デフォルト lm_lambda=0.0."""
        inp = LinearSolveInput(
            K_T=sp.eye(2),
            R_u=np.array([1.0, 1.0]),
            fixed_dofs=np.array([], dtype=int),
        )
        assert inp.lm_lambda == 0.0


class TestLMRegularizationStability:
    """LM 正則化の安定性テスト."""

    def test_indefinite_matrix_without_lm_may_succeed(self):
        """不定値行列でも spsolve は数学的には解ける（対称正定値不要）."""
        # 不定値行列（固有値 -1, 5）
        K = sp.csc_matrix(np.array([[2.0, 3.0], [3.0, 2.0]]))
        R = np.array([1.0, 0.0])
        fixed = np.array([], dtype=int)

        proc = LinearSolveProcess()
        out = proc.process(LinearSolveInput(K_T=K, R_u=R, fixed_dofs=fixed, lm_lambda=0.0))
        # spsolve は不定値でも解ける
        assert out.success

    def test_indefinite_matrix_with_lm_produces_descent(self):
        """不定値行列 + LM 正則化で降下方向を生成."""
        # 不定値行列（固有値 -1, 5）
        K = sp.csc_matrix(np.array([[2.0, 3.0], [3.0, 2.0]]))
        R = np.array([1.0, 0.5])
        fixed = np.array([], dtype=int)

        proc = LinearSolveProcess()
        # λ=2: 十分な正則化で正定値化（全固有値 > 0）
        out_lm = proc.process(LinearSolveInput(K_T=K, R_u=R, fixed_dofs=fixed, lm_lambda=2.0))

        assert out_lm.success
        # LM 正則化後の du は降下方向: du · R < 0（du = -K_reg^{-1} R）
        descent = float(np.dot(out_lm.du, R))
        assert descent < 0, f"降下方向でない: du·R = {descent}"

    def test_singular_matrix_with_lm_succeeds(self):
        """特異行列 + LM 正則化で正則化."""
        # 特異行列（ランク1）
        K = sp.csc_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
        R = np.array([1.0, 0.0])
        fixed = np.array([], dtype=int)

        proc = LinearSolveProcess()
        out = proc.process(LinearSolveInput(K_T=K, R_u=R, fixed_dofs=fixed, lm_lambda=1.0))
        assert out.success
        assert out.du is not None

    def test_marquardt_scaling(self):
        """Marquardt 型スケーリング: λ·diag(|K|) がDOFスケール差を吸収."""
        # 並進DOF（大剛性）vs 回転DOF（小剛性）
        K = sp.csc_matrix(np.diag([1e6, 1e6, 1e6, 30.0, 30.0, 30.0]))
        R = np.array([100.0, 100.0, 100.0, 0.01, 0.01, 0.01])
        fixed = np.array([], dtype=int)

        proc = LinearSolveProcess()
        out = proc.process(LinearSolveInput(K_T=K, R_u=R, fixed_dofs=fixed, lm_lambda=0.1))

        assert out.success
        # 正則化後の対角: K_ii + λ * |K_ii|
        # 並進: 1e6 + 0.1*1e6 = 1.1e6
        # 回転: 30 + 0.1*30 = 33
        # Marquardt 型なのでスケール差を自動吸収
        du_trans = np.linalg.norm(out.du[:3])
        du_rot = np.linalg.norm(out.du[3:])
        # 両方とも有限値であること
        assert du_trans > 0 and du_rot > 0

    def test_bc_applied_after_regularization(self):
        """境界条件はLM正則化後に適用."""
        K = sp.csc_matrix(np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]]))
        R = np.array([1.0, 2.0, 3.0])
        fixed = np.array([0])

        proc = LinearSolveProcess()
        out = proc.process(LinearSolveInput(K_T=K, R_u=R, fixed_dofs=fixed, lm_lambda=0.5))

        assert out.success
        # 固定DOFは変位ゼロ
        assert abs(out.du[0]) < 1e-12

    def test_large_lambda_approaches_steepest_descent(self):
        """大きな λ で最急降下方向に近づく."""
        K = sp.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
        R = np.array([1.0, 2.0])
        fixed = np.array([], dtype=int)

        proc = LinearSolveProcess()
        out = proc.process(LinearSolveInput(K_T=K, R_u=R, fixed_dofs=fixed, lm_lambda=1e6))

        assert out.success
        # λ→∞ で du → -R / (λ·diag(|K|))（スケール付き最急降下）
        # du と -R の方向が近い
        cos_angle = np.dot(out.du, -R) / (np.linalg.norm(out.du) * np.linalg.norm(R))
        assert cos_angle > 0.9, f"最急降下方向との角度が大きい: cos={cos_angle}"
