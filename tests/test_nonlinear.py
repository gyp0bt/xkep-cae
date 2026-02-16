"""非線形解析テスト.

テスト方針:
  1. Newton-Raphson ソルバーの基本動作
  2. 線形問題での収束確認（線形→非線形の一致）
  3. 大変形片持ち梁（Euler elastica ベンチマーク）
  4. assemble_cosserat_beam のヘルパー関数テスト
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae.elements.beam_cosserat import (
    CosseratRod,
    assemble_cosserat_beam,
)
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection
from xkep_cae.solver import NonlinearResult, newton_raphson

# --- テスト用パラメータ ---
E = 200_000.0  # MPa
NU = 0.3
G = E / (2.0 * (1.0 + NU))


def _make_section() -> BeamSection:
    return BeamSection.rectangle(10.0, 20.0)


def _make_material() -> BeamElastic1D:
    return BeamElastic1D(E=E, nu=NU)


class TestNonlinearResult:
    """NonlinearResult データクラスのテスト."""

    def test_creation(self):
        result = NonlinearResult(
            u=np.zeros(12),
            converged=True,
            n_load_steps=10,
            total_iterations=25,
        )
        assert result.converged
        assert result.n_load_steps == 10
        assert result.total_iterations == 25


class TestNewtonRaphsonLinear:
    """Newton-Raphson で線形問題を解くテスト.

    線形問題では1回の反復で収束し、直接法と同じ結果を返すべき。
    """

    def test_axial_cantilever(self):
        """軸引張: 線形問題を NR で解く."""
        sec = _make_section()
        mat = _make_material()
        n_elems = 5
        L = 100.0
        P = 1000.0

        rod = CosseratRod(section=sec)
        n_nodes = n_elems + 1
        total_dof = n_nodes * 6

        def _assemble_tangent(u):
            K, _ = assemble_cosserat_beam(
                n_elems,
                L,
                rod,
                mat,
                u,
                stiffness=True,
                internal_force=False,
            )
            return sp.csr_matrix(K)

        def _assemble_fint(u):
            _, f_int = assemble_cosserat_beam(
                n_elems,
                L,
                rod,
                mat,
                u,
                stiffness=False,
                internal_force=True,
            )
            return f_int

        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems] = P  # 軸方向

        fixed_dofs = np.arange(6)  # 節点0を全固定

        result = newton_raphson(
            f_ext,
            fixed_dofs,
            _assemble_tangent,
            _assemble_fint,
            n_load_steps=1,
            max_iter=10,
            show_progress=False,
        )

        assert result.converged
        delta_expected = P * L / (E * sec.A)
        np.testing.assert_almost_equal(
            result.u[6 * n_elems],
            delta_expected,
            decimal=6,
        )

    def test_bending_cantilever(self):
        """曲げ荷重: 線形問題を NR で解く."""
        sec = _make_section()
        mat = _make_material()
        n_elems = 20
        L = 100.0
        P = 100.0

        rod = CosseratRod(section=sec)
        n_nodes = n_elems + 1
        total_dof = n_nodes * 6

        def _assemble_tangent(u):
            K, _ = assemble_cosserat_beam(
                n_elems,
                L,
                rod,
                mat,
                u,
                stiffness=True,
                internal_force=False,
            )
            return sp.csr_matrix(K)

        def _assemble_fint(u):
            _, f_int = assemble_cosserat_beam(
                n_elems,
                L,
                rod,
                mat,
                u,
                stiffness=False,
                internal_force=True,
            )
            return f_int

        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems + 1] = P  # y方向

        fixed_dofs = np.arange(6)

        result = newton_raphson(
            f_ext,
            fixed_dofs,
            _assemble_tangent,
            _assemble_fint,
            n_load_steps=1,
            max_iter=10,
            show_progress=False,
        )

        assert result.converged
        # 解析解 (Timoshenko): δ = PL³/(3EI) + PL/(κGA)
        kappa = 5.0 / 6.0
        delta_bending = P * L**3 / (3.0 * E * sec.Iz)
        delta_shear = P * L / (kappa * G * sec.A)
        delta_exact = delta_bending + delta_shear

        # 20要素のメッシュ精度範囲で一致
        delta_nr = result.u[6 * n_elems + 1]
        rel_error = abs(delta_nr - delta_exact) / abs(delta_exact)
        assert rel_error < 0.01, f"相対誤差 {rel_error:.4f} > 1%"

    def test_load_stepping(self):
        """荷重増分で線形問題を解く（最終結果は1ステップと一致）."""
        sec = _make_section()
        mat = _make_material()
        n_elems = 5
        L = 100.0
        P = 1000.0

        rod = CosseratRod(section=sec)
        n_nodes = n_elems + 1
        total_dof = n_nodes * 6

        def _assemble_tangent(u):
            K, _ = assemble_cosserat_beam(
                n_elems,
                L,
                rod,
                mat,
                u,
                stiffness=True,
                internal_force=False,
            )
            return sp.csr_matrix(K)

        def _assemble_fint(u):
            _, f_int = assemble_cosserat_beam(
                n_elems,
                L,
                rod,
                mat,
                u,
                stiffness=False,
                internal_force=True,
            )
            return f_int

        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems] = P

        fixed_dofs = np.arange(6)

        # 1ステップ
        result_1 = newton_raphson(
            f_ext,
            fixed_dofs,
            _assemble_tangent,
            _assemble_fint,
            n_load_steps=1,
            show_progress=False,
        )

        # 5ステップ
        result_5 = newton_raphson(
            f_ext,
            fixed_dofs,
            _assemble_tangent,
            _assemble_fint,
            n_load_steps=5,
            show_progress=False,
        )

        assert result_1.converged
        assert result_5.converged
        np.testing.assert_array_almost_equal(
            result_1.u,
            result_5.u,
            decimal=6,
        )


class TestAssembleCosseratBeam:
    """assemble_cosserat_beam ヘルパー関数のテスト."""

    def test_stiffness_only(self):
        sec = _make_section()
        mat = _make_material()
        rod = CosseratRod(section=sec)
        u = np.zeros(66)  # 11 nodes * 6 DOF
        K, f = assemble_cosserat_beam(
            10,
            100.0,
            rod,
            mat,
            u,
            stiffness=True,
            internal_force=False,
        )
        assert K is not None
        assert f is None
        assert K.shape == (66, 66)

    def test_internal_force_only(self):
        sec = _make_section()
        mat = _make_material()
        rod = CosseratRod(section=sec)
        u = np.zeros(66)
        K, f = assemble_cosserat_beam(
            10,
            100.0,
            rod,
            mat,
            u,
            stiffness=False,
            internal_force=True,
        )
        assert K is None
        assert f is not None
        assert f.shape == (66,)

    def test_zero_disp_gives_zero_force(self):
        sec = _make_section()
        mat = _make_material()
        rod = CosseratRod(section=sec)
        u = np.zeros(66)
        _, f = assemble_cosserat_beam(
            10,
            100.0,
            rod,
            mat,
            u,
            stiffness=False,
            internal_force=True,
        )
        np.testing.assert_array_almost_equal(f, np.zeros(66))


class TestLargeDeformationCantilever:
    """大変形片持ち梁テスト.

    片持ち梁の先端に集中荷重 P を作用させ、大変形応答を確認する。
    幾何学的非線形効果（接線剛性 = 材料剛性 + 幾何剛性）により、
    線形解析よりも変位が小さくなる（剛性増加効果）。

    厳密には Euler elastica の解と比較すべきだが、
    ここでは定性的な確認（収束性、変位の方向、非線形効果の出現）を行う。
    """

    def test_large_deflection_converges(self):
        """大変位荷重で NR が収束すること."""
        sec = BeamSection.rectangle(10.0, 10.0)
        mat = _make_material()
        n_elems = 20
        L = 200.0
        # 大きな荷重: 線形解で L/10 程度のたわみが出る
        # δ_lin = PL³/(3EI) → P = 3EI·δ/(L³)
        Iz = sec.Iz
        delta_target = L / 5.0  # 40mm たわみ
        P = 3.0 * E * Iz * delta_target / L**3

        rod = CosseratRod(section=sec)
        n_nodes = n_elems + 1
        total_dof = n_nodes * 6

        def _assemble_tangent(u):
            K, _ = assemble_cosserat_beam(
                n_elems,
                L,
                rod,
                mat,
                u,
                stiffness=True,
                internal_force=False,
            )
            return sp.csr_matrix(K)

        def _assemble_fint(u):
            _, f_int = assemble_cosserat_beam(
                n_elems,
                L,
                rod,
                mat,
                u,
                stiffness=False,
                internal_force=True,
            )
            return f_int

        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems + 1] = P  # y方向

        fixed_dofs = np.arange(6)

        result = newton_raphson(
            f_ext,
            fixed_dofs,
            _assemble_tangent,
            _assemble_fint,
            n_load_steps=10,
            max_iter=30,
            show_progress=False,
        )

        assert result.converged, "大変形解析が収束しない"
        # 変位がゼロでないこと
        delta_tip = result.u[6 * n_elems + 1]
        assert abs(delta_tip) > 0.01 * L, "先端変位が小さすぎる"
        # 変位の向きが正しい（荷重方向と同じ）
        assert delta_tip > 0, "先端変位の向きが反転している"

    def test_nonlinear_stiffer_than_linear(self):
        """幾何学的非線形効果: 引張時の曲げは線形解より小さい.

        軸力が作用している状態で曲げ荷重を加えると、
        幾何剛性による剛性増加効果で変位が小さくなる。
        """
        sec = BeamSection.rectangle(10.0, 10.0)
        mat = _make_material()
        n_elems = 20
        L = 200.0
        P_axial = 10000.0  # 軸力（引張）
        P_bending = 100.0  # 曲げ荷重

        rod = CosseratRod(section=sec)
        n_nodes = n_elems + 1
        total_dof = n_nodes * 6

        def _assemble_tangent(u):
            K, _ = assemble_cosserat_beam(
                n_elems,
                L,
                rod,
                mat,
                u,
                stiffness=True,
                internal_force=False,
            )
            return sp.csr_matrix(K)

        def _assemble_fint(u):
            _, f_int = assemble_cosserat_beam(
                n_elems,
                L,
                rod,
                mat,
                u,
                stiffness=False,
                internal_force=True,
            )
            return f_int

        # 非線形: 軸力 + 曲げ
        f_ext_nl = np.zeros(total_dof)
        f_ext_nl[6 * n_elems] = P_axial
        f_ext_nl[6 * n_elems + 1] = P_bending

        fixed_dofs = np.arange(6)

        result_nl = newton_raphson(
            f_ext_nl,
            fixed_dofs,
            _assemble_tangent,
            _assemble_fint,
            n_load_steps=5,
            max_iter=30,
            show_progress=False,
        )
        assert result_nl.converged

        # 線形: 曲げのみ（接線剛性 = 材料剛性のみ）
        def _assemble_tangent_linear(u):
            """材料剛性のみ（幾何剛性なし）."""
            K = np.zeros((total_dof, total_dof))
            elem_len = L / n_elems
            for i in range(n_elems):
                coords = np.array(
                    [
                        [i * elem_len, 0.0, 0.0],
                        [(i + 1) * elem_len, 0.0, 0.0],
                    ]
                )
                Ke = rod.local_stiffness(coords, mat)
                K[6 * i : 6 * (i + 2), 6 * i : 6 * (i + 2)] += Ke
            return sp.csr_matrix(K)

        f_ext_lin = np.zeros(total_dof)
        f_ext_lin[6 * n_elems + 1] = P_bending

        result_lin = newton_raphson(
            f_ext_lin,
            fixed_dofs,
            _assemble_tangent_linear,
            _assemble_fint,
            n_load_steps=1,
            max_iter=10,
            show_progress=False,
        )
        assert result_lin.converged

        delta_nl = abs(result_nl.u[6 * n_elems + 1])
        delta_lin = abs(result_lin.u[6 * n_elems + 1])

        # 引張下では非線形の方が変位が小さい（剛性増加）
        assert delta_nl < delta_lin, (
            f"非線形 {delta_nl:.6f} >= 線形 {delta_lin:.6f}: 引張下の幾何剛性効果が効いていない"
        )

    def test_load_history_recorded(self):
        """荷重履歴が正しく記録されること."""
        sec = _make_section()
        mat = _make_material()
        n_elems = 5
        L = 100.0
        P = 1000.0

        rod = CosseratRod(section=sec)
        n_nodes = n_elems + 1
        total_dof = n_nodes * 6

        def _assemble_tangent(u):
            K, _ = assemble_cosserat_beam(
                n_elems,
                L,
                rod,
                mat,
                u,
                stiffness=True,
                internal_force=False,
            )
            return sp.csr_matrix(K)

        def _assemble_fint(u):
            _, f_int = assemble_cosserat_beam(
                n_elems,
                L,
                rod,
                mat,
                u,
                stiffness=False,
                internal_force=True,
            )
            return f_int

        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems] = P

        fixed_dofs = np.arange(6)

        result = newton_raphson(
            f_ext,
            fixed_dofs,
            _assemble_tangent,
            _assemble_fint,
            n_load_steps=5,
            show_progress=False,
        )

        assert len(result.load_history) == 5
        np.testing.assert_array_almost_equal(
            result.load_history,
            [0.2, 0.4, 0.6, 0.8, 1.0],
        )
        assert len(result.displacement_history) == 5

    def test_sri_large_deflection(self):
        """SRI スキームでも大変形解析が収束すること."""
        sec = BeamSection.rectangle(10.0, 10.0)
        mat = _make_material()
        n_elems = 20
        L = 200.0
        Iz = sec.Iz
        delta_target = L / 10.0
        P = 3.0 * E * Iz * delta_target / L**3

        rod = CosseratRod(section=sec, integration_scheme="sri")
        n_nodes = n_elems + 1
        total_dof = n_nodes * 6

        def _assemble_tangent(u):
            K, _ = assemble_cosserat_beam(
                n_elems,
                L,
                rod,
                mat,
                u,
                stiffness=True,
                internal_force=False,
            )
            return sp.csr_matrix(K)

        def _assemble_fint(u):
            _, f_int = assemble_cosserat_beam(
                n_elems,
                L,
                rod,
                mat,
                u,
                stiffness=False,
                internal_force=True,
            )
            return f_int

        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems + 1] = P

        fixed_dofs = np.arange(6)

        result = newton_raphson(
            f_ext,
            fixed_dofs,
            _assemble_tangent,
            _assemble_fint,
            n_load_steps=5,
            max_iter=30,
            show_progress=False,
        )

        assert result.converged, "SRI大変形解析が収束しない"
        delta_tip = result.u[6 * n_elems + 1]
        assert abs(delta_tip) > 0.001 * L
