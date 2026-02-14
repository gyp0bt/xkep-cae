"""弧長法テスト.

テスト方針:
  1. ArcLengthResult データクラス
  2. 線形問題で NR と同じ結果を返すこと
  3. 非線形スプリングのスナップスルー（解析解あり）
  4. 片持ち梁の大変形で NR と弧長法の結果が一致
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.elements.beam_cosserat import (
    CosseratRod,
    assemble_cosserat_beam,
)
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection
from xkep_cae.solver import ArcLengthResult, arc_length, newton_raphson


# --- 共通パラメータ ---
E = 200_000.0   # MPa
NU = 0.3


class TestArcLengthResult:
    """ArcLengthResult データクラスのテスト."""

    def test_creation(self):
        result = ArcLengthResult(
            u=np.zeros(12),
            lam=1.0,
            converged=True,
            n_steps=10,
            total_iterations=25,
        )
        assert result.converged
        assert result.lam == 1.0
        assert result.n_steps == 10


class TestArcLengthLinear:
    """弧長法で線形問題を解くテスト."""

    def test_axial_cantilever(self):
        """軸引張: 弧長法と NR の結果が一致."""
        sec = BeamSection.rectangle(10.0, 20.0)
        mat = BeamElastic1D(E=E, nu=NU)
        n_elems = 5
        L = 100.0
        P = 1000.0

        rod = CosseratRod(section=sec)
        n_nodes = n_elems + 1
        total_dof = n_nodes * 6

        def _assemble_tangent(u):
            K, _ = assemble_cosserat_beam(
                n_elems, L, rod, mat, u,
                stiffness=True, internal_force=False,
            )
            return sp.csr_matrix(K)

        def _assemble_fint(u):
            _, f_int = assemble_cosserat_beam(
                n_elems, L, rod, mat, u,
                stiffness=False, internal_force=True,
            )
            return f_int

        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems] = P
        fixed_dofs = np.arange(6)

        result_nr = newton_raphson(
            f_ext, fixed_dofs,
            _assemble_tangent, _assemble_fint,
            n_load_steps=1, show_progress=False,
        )

        result_al = arc_length(
            f_ext, fixed_dofs,
            _assemble_tangent, _assemble_fint,
            show_progress=False,
            lambda_max=1.0,
        )

        assert result_nr.converged
        assert result_al.converged
        assert result_al.lam >= 1.0 - 1e-3

        u_al_normalized = result_al.u / result_al.lam
        np.testing.assert_array_almost_equal(
            u_al_normalized, result_nr.u, decimal=3,
        )

    def test_bending_cantilever(self):
        """曲げ荷重: 弧長法と NR が一致."""
        sec = BeamSection.rectangle(10.0, 20.0)
        mat = BeamElastic1D(E=E, nu=NU)
        n_elems = 10
        L = 100.0
        P = 100.0

        rod = CosseratRod(section=sec)
        n_nodes = n_elems + 1
        total_dof = n_nodes * 6

        def _assemble_tangent(u):
            K, _ = assemble_cosserat_beam(
                n_elems, L, rod, mat, u,
                stiffness=True, internal_force=False,
            )
            return sp.csr_matrix(K)

        def _assemble_fint(u):
            _, f_int = assemble_cosserat_beam(
                n_elems, L, rod, mat, u,
                stiffness=False, internal_force=True,
            )
            return f_int

        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems + 1] = P
        fixed_dofs = np.arange(6)

        result_nr = newton_raphson(
            f_ext, fixed_dofs,
            _assemble_tangent, _assemble_fint,
            n_load_steps=1, show_progress=False,
        )

        result_al = arc_length(
            f_ext, fixed_dofs,
            _assemble_tangent, _assemble_fint,
            show_progress=False,
            lambda_max=1.0,
        )

        assert result_nr.converged
        assert result_al.converged

        u_al_normalized = result_al.u / result_al.lam
        np.testing.assert_array_almost_equal(
            u_al_normalized, result_nr.u, decimal=3,
        )


class TestSnapThroughSpring:
    """非線形スプリングのスナップスルー問題.

    1-DOF 非線形スプリング: f_int(u) = k₁·u + k₃·u³
    k₁ > 0, k₃ < 0 の場合、荷重-変位曲線にリミットポイントが存在。

    荷重-変位関係: P = k₁·u + k₃·u³
    リミットポイント: dP/du = k₁ + 3·k₃·u² = 0
      → u_lim = √(-k₁/(3·k₃))
      → P_lim = k₁·u_lim + k₃·u_lim³ = (2/3)·k₁·√(-k₁/(3·k₃))
    """

    def test_snap_through_traces_limit_point(self):
        """弧長法がスナップスルーのリミットポイントを通過する."""
        k1 = 1.0
        k3 = -0.01

        # リミットポイントの解析値
        u_lim = np.sqrt(-k1 / (3.0 * k3))  # = √(100/3) ≈ 5.774
        P_lim = k1 * u_lim + k3 * u_lim**3   # ≈ 3.849

        def assemble_tangent(u_vec):
            u = u_vec[0]
            K_val = k1 + 3.0 * k3 * u**2
            return sp.csr_matrix(np.array([[K_val]]))

        def assemble_fint(u_vec):
            u = u_vec[0]
            return np.array([k1 * u + k3 * u**3])

        f_ext = np.array([1.0])
        fixed_dofs = np.array([], dtype=int)

        result = arc_length(
            f_ext, fixed_dofs,
            assemble_tangent, assemble_fint,
            n_steps=100,
            delta_l=0.5,
            max_iter=30,
            show_progress=False,
            lambda_max=None,
        )

        assert result.n_steps >= 10, f"弧長法が {result.n_steps} ステップしか進まなかった"

        lam_hist = np.array(result.load_history)

        # 荷重が増加→減少に転じるリミットポイントの確認
        lam_max_computed = float(np.max(lam_hist))
        idx_max = int(np.argmax(lam_hist))

        # リミットポイントの荷重が解析値と近いこと
        rel_err = abs(lam_max_computed - P_lim) / P_lim
        assert rel_err < 0.15, (
            f"P_lim: 数値={lam_max_computed:.4f}, 解析={P_lim:.4f}, "
            f"相対誤差={rel_err:.3f}"
        )

        # リミットポイント後も計算が続いていること（スナップバック追跡）
        assert result.n_steps > idx_max + 1, (
            "リミットポイント後にステップが進んでいない"
        )

    def test_pre_limit_agrees_with_analytical(self):
        """リミットポイント前の荷重-変位曲線が解析解と一致する."""
        k1 = 1.0
        k3 = -0.01
        u_lim = np.sqrt(-k1 / (3.0 * k3))

        def assemble_tangent(u_vec):
            u = u_vec[0]
            K_val = k1 + 3.0 * k3 * u**2
            return sp.csr_matrix(np.array([[K_val]]))

        def assemble_fint(u_vec):
            u = u_vec[0]
            return np.array([k1 * u + k3 * u**3])

        f_ext = np.array([1.0])
        fixed_dofs = np.array([], dtype=int)

        result = arc_length(
            f_ext, fixed_dofs,
            assemble_tangent, assemble_fint,
            n_steps=50,
            delta_l=0.3,
            max_iter=30,
            show_progress=False,
            lambda_max=None,
        )

        # リミットポイント前のステップを抽出し、解析解と比較
        for i, (lam_i, u_i) in enumerate(
            zip(result.load_history, result.displacement_history)
        ):
            u_val = u_i[0]
            if abs(u_val) > u_lim * 0.8:
                break  # リミット近くでは精度が落ちる
            # 解析解: P = k1*u + k3*u^3
            P_exact = k1 * u_val + k3 * u_val**3
            if abs(P_exact) > 1e-10:
                rel_err = abs(lam_i - P_exact) / abs(P_exact)
                assert rel_err < 0.01, (
                    f"Step {i}: u={u_val:.4f}, "
                    f"λ={lam_i:.6f}, P_exact={P_exact:.6f}, "
                    f"err={rel_err:.4f}"
                )
