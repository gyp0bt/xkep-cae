"""ビームアセンブラ移植テスト（Phase 13）.

新 xkep_cae.elements パッケージに移植された CR 梁要素・アセンブラの
機能検証テスト。deprecated パッケージを一切使用しない。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xkep_cae.elements import BeamSection, BeamSection2D, ULCRBeamAssembler
from xkep_cae.elements._beam_assembly import assemble_cr_beam3d
from xkep_cae.elements._beam_cr import (
    timo_beam3d_cr_internal_force,
    timo_beam3d_cr_tangent,
    timo_beam3d_cr_tangent_analytical,
    timo_beam3d_ke_global,
    timo_beam3d_ke_local,
    timo_beam3d_lumped_mass_local,
    timo_beam3d_mass_global,
    timo_beam3d_mass_local,
)

# =====================================================================
# BeamSection テスト
# =====================================================================


class TestBeamSectionAPI:
    """BeamSection/BeamSection2D の API テスト."""

    def test_circle_section(self) -> None:
        d = 0.01
        sec = BeamSection.circle(d)
        assert sec.A == pytest.approx(math.pi * (d / 2) ** 2)
        assert sec.Iy == pytest.approx(math.pi * d**4 / 64)
        assert sec.Iz == pytest.approx(math.pi * d**4 / 64)
        assert sec.J == pytest.approx(math.pi * d**4 / 32)
        assert sec.shape == "circle"

    def test_rectangle_section(self) -> None:
        sec = BeamSection.rectangle(0.02, 0.03)
        assert sec.A == pytest.approx(0.02 * 0.03)
        assert sec.Iy == pytest.approx(0.02 * 0.03**3 / 12)
        assert sec.Iz == pytest.approx(0.03 * 0.02**3 / 12)

    def test_pipe_section(self) -> None:
        sec = BeamSection.pipe(0.02, 0.01)
        assert sec.A > 0
        assert sec.Iy > 0
        assert sec.J > 0

    def test_pipe_invalid(self) -> None:
        with pytest.raises(ValueError):
            BeamSection.pipe(0.01, 0.02)

    def test_cowper_kappa(self) -> None:
        sec = BeamSection.circle(0.01)
        kappa = sec.cowper_kappa_y(0.3)
        assert 0.8 < kappa < 0.95

    def test_to_2d(self) -> None:
        sec = BeamSection.circle(0.01)
        sec2d = sec.to_2d()
        assert isinstance(sec2d, BeamSection2D)
        assert sec2d.A == sec.A
        assert sec2d.I == sec.Iz

    def test_frozen(self) -> None:
        sec = BeamSection.circle(0.01)
        with pytest.raises(AttributeError):
            sec.A = 1.0  # type: ignore[misc]

    def test_2d_circle(self) -> None:
        sec2d = BeamSection2D.circle(0.01)
        assert sec2d.A > 0
        assert sec2d.I > 0

    def test_2d_rectangle(self) -> None:
        sec2d = BeamSection2D.rectangle(0.02, 0.03)
        assert sec2d.A == pytest.approx(0.02 * 0.03)

    def test_invalid_section(self) -> None:
        with pytest.raises(ValueError):
            BeamSection(A=-1, Iy=1, Iz=1, J=1)
        with pytest.raises(ValueError):
            BeamSection2D(A=-1, I=1)


# =====================================================================
# 剛性行列テスト
# =====================================================================


def _cantilever_beam(n_elems: int = 10, L: float = 1.0) -> tuple:
    """片持梁のメッシュを生成."""
    n_nodes = n_elems + 1
    coords = np.zeros((n_nodes, 3))
    coords[:, 0] = np.linspace(0, L, n_nodes)
    conn = np.array([[i, i + 1] for i in range(n_elems)])
    return coords, conn


class TestStiffnessAPI:
    """剛性行列の基本テスト."""

    def test_ke_local_symmetric(self) -> None:
        Ke = timo_beam3d_ke_local(200e9, 80e9, 1e-4, 1e-8, 1e-8, 2e-8, 0.1, 5 / 6, 5 / 6)
        assert Ke.shape == (12, 12)
        np.testing.assert_allclose(Ke, Ke.T, atol=1e-10)

    def test_ke_local_positive_definite(self) -> None:
        Ke = timo_beam3d_ke_local(200e9, 80e9, 1e-4, 1e-8, 1e-8, 2e-8, 0.1, 5 / 6, 5 / 6)
        eigvals = np.linalg.eigvalsh(Ke)
        # 剛体モードで 0 固有値があるので非負
        assert np.all(eigvals > -1e-5)

    def test_ke_global_symmetric(self) -> None:
        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        Ke = timo_beam3d_ke_global(coords, 200e9, 80e9, 1e-4, 1e-8, 1e-8, 2e-8, 5 / 6, 5 / 6)
        np.testing.assert_allclose(Ke, Ke.T, atol=1e-5)


# =====================================================================
# 質量行列テスト
# =====================================================================


class TestMassAPI:
    """質量行列の基本テスト."""

    def test_lumped_mass_diagonal(self) -> None:
        Me = timo_beam3d_lumped_mass_local(7850, 1e-4, 1e-8, 1e-8, 0.1)
        assert Me.shape == (12, 12)
        # 対角行列であること
        off_diag = Me - np.diag(np.diag(Me))
        assert np.allclose(off_diag, 0)
        # 全対角要素が正
        assert np.all(np.diag(Me) > 0)

    def test_consistent_mass_symmetric(self) -> None:
        Me = timo_beam3d_mass_local(7850, 1e-4, 1e-8, 1e-8, 0.1)
        np.testing.assert_allclose(Me, Me.T, atol=1e-15)

    def test_mass_global_lumped(self) -> None:
        coords = np.array([[0, 0, 0], [0.1, 0, 0]], dtype=float)
        Me = timo_beam3d_mass_global(coords, 7850, 1e-4, 1e-8, 1e-8, lumped=True)
        assert Me.shape == (12, 12)

    def test_mass_global_consistent(self) -> None:
        coords = np.array([[0, 0, 0], [0.1, 0, 0]], dtype=float)
        Me = timo_beam3d_mass_global(coords, 7850, 1e-4, 1e-8, 1e-8, lumped=False)
        np.testing.assert_allclose(Me, Me.T, atol=1e-15)


# =====================================================================
# CR 内力・接線剛性テスト
# =====================================================================


class TestCRBeamAPI:
    """CR 梁要素の基本テスト."""

    def test_zero_displacement_zero_force(self) -> None:
        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        u = np.zeros(12)
        f = timo_beam3d_cr_internal_force(
            coords, u, 200e9, 80e9, 1e-4, 1e-8, 1e-8, 2e-8, 5 / 6, 5 / 6
        )
        np.testing.assert_allclose(f, 0, atol=1e-10)

    def test_tangent_symmetric(self) -> None:
        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        u = np.zeros(12)
        u[7] = 0.001  # y方向微小変位
        K = timo_beam3d_cr_tangent_analytical(
            coords, u, 200e9, 80e9, 1e-4, 1e-8, 1e-8, 2e-8, 5 / 6, 5 / 6
        )
        np.testing.assert_allclose(K, K.T, atol=1e-3)

    def test_analytical_vs_numerical_tangent(self) -> None:
        coords = np.array([[0, 0, 0], [0.1, 0, 0]], dtype=float)
        u = np.zeros(12)
        u[7] = 0.001
        K_a = timo_beam3d_cr_tangent_analytical(
            coords, u, 200e9, 80e9, 1e-4, 1e-8, 1e-8, 2e-8, 5 / 6, 5 / 6
        )
        K_n = timo_beam3d_cr_tangent(coords, u, 200e9, 80e9, 1e-4, 1e-8, 1e-8, 2e-8, 5 / 6, 5 / 6)
        np.testing.assert_allclose(K_a, K_n, rtol=1e-4)


# =====================================================================
# アセンブリテスト
# =====================================================================


class TestAssemblyAPI:
    """グローバルアセンブリの基本テスト."""

    def test_assemble_sparse(self) -> None:
        coords, conn = _cantilever_beam(5)
        ndof = len(coords) * 6
        u = np.zeros(ndof)
        K, f = assemble_cr_beam3d(
            coords,
            conn,
            u,
            200e9,
            80e9,
            1e-4,
            1e-8,
            1e-8,
            2e-8,
            5 / 6,
            5 / 6,
            sparse=True,
            analytical_tangent=True,
        )
        assert K is not None
        assert K.shape == (ndof, ndof)
        assert f is not None
        np.testing.assert_allclose(f, 0, atol=1e-10)

    def test_assemble_dense(self) -> None:
        coords, conn = _cantilever_beam(3)
        ndof = len(coords) * 6
        u = np.zeros(ndof)
        K, f = assemble_cr_beam3d(
            coords,
            conn,
            u,
            200e9,
            80e9,
            1e-4,
            1e-8,
            1e-8,
            2e-8,
            5 / 6,
            5 / 6,
            sparse=False,
        )
        assert isinstance(K, np.ndarray)
        assert K.shape == (ndof, ndof)

    def test_assemble_numerical_tangent(self) -> None:
        coords, conn = _cantilever_beam(3)
        ndof = len(coords) * 6
        u = np.zeros(ndof)
        K, f = assemble_cr_beam3d(
            coords,
            conn,
            u,
            200e9,
            80e9,
            1e-4,
            1e-8,
            1e-8,
            2e-8,
            5 / 6,
            5 / 6,
            sparse=True,
            analytical_tangent=False,
        )
        assert K is not None

    def test_force_only(self) -> None:
        coords, conn = _cantilever_beam(3)
        ndof = len(coords) * 6
        u = np.zeros(ndof)
        K, f = assemble_cr_beam3d(
            coords,
            conn,
            u,
            200e9,
            80e9,
            1e-4,
            1e-8,
            1e-8,
            2e-8,
            5 / 6,
            5 / 6,
            stiffness=False,
            internal_force=True,
        )
        assert K is None
        assert f is not None


# =====================================================================
# ULCRBeamAssembler テスト
# =====================================================================


class TestULCRBeamAssemblerAPI:
    """ULCRBeamAssembler の基本テスト."""

    def _make_assembler(self, n_elems: int = 5) -> ULCRBeamAssembler:
        coords, conn = _cantilever_beam(n_elems)
        return ULCRBeamAssembler(
            coords,
            conn,
            200e9,
            80e9,
            1e-4,
            1e-8,
            1e-8,
            2e-8,
            5 / 6,
            5 / 6,
        )

    def test_zero_displacement(self) -> None:
        asm = self._make_assembler()
        u = np.zeros(asm.ndof)
        K = asm.assemble_tangent(u)
        f = asm.assemble_internal_force(u)
        assert K is not None
        np.testing.assert_allclose(f, 0, atol=1e-10)

    def test_update_reference(self) -> None:
        asm = self._make_assembler()
        u = np.zeros(asm.ndof)
        u[6] = 0.001  # x方向微小変位
        asm.update_reference(u)
        np.testing.assert_allclose(asm.u_total_accum, u)
        assert asm.coords_ref[1, 0] == pytest.approx(0.2 + 0.001)

    def test_checkpoint_rollback(self) -> None:
        asm = self._make_assembler()
        coords_before = asm.coords_ref.copy()
        asm.checkpoint()
        u = np.zeros(asm.ndof)
        u[6] = 0.1
        asm.update_reference(u)
        assert asm.coords_ref[1, 0] != coords_before[1, 0]
        asm.rollback()
        np.testing.assert_allclose(asm.coords_ref, coords_before)

    def test_get_total_displacement(self) -> None:
        asm = self._make_assembler()
        u = np.zeros(asm.ndof)
        u[6] = 0.001
        asm.update_reference(u)
        u_incr = np.zeros(asm.ndof)
        u_incr[6] = 0.002
        u_total = asm.get_total_displacement(u_incr)
        assert u_total[6] == pytest.approx(0.003)

    def test_mass_lumped(self) -> None:
        asm = self._make_assembler()
        M = asm.assemble_mass(7850, lumped=True)
        assert M.shape == (asm.ndof, asm.ndof)

    def test_mass_consistent(self) -> None:
        asm = self._make_assembler()
        M = asm.assemble_mass(7850, lumped=False)
        assert M.shape == (asm.ndof, asm.ndof)


# =====================================================================
# 物理的妥当性テスト
# =====================================================================


class TestBeamPhysics:
    """物理的妥当性の検証テスト."""

    def test_cantilever_tip_load_convergence(self) -> None:
        """片持梁先端荷重の Newton-Raphson 収束."""
        n_elems = 16
        L = 1.0
        E = 200e9
        nu = 0.3
        G = E / (2 * (1 + nu))
        d = 0.01
        sec = BeamSection.circle(d)
        kappa = sec.cowper_kappa_y(nu)

        coords, conn = _cantilever_beam(n_elems, L)
        asm = ULCRBeamAssembler(
            coords,
            conn,
            E,
            G,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.J,
            kappa,
        )

        ndof = asm.ndof
        free_dofs = np.arange(6, ndof)  # 固定端を除外
        P = -10.0  # y方向荷重

        u = np.zeros(ndof)
        f_ext = np.zeros(ndof)
        f_ext[ndof - 5] = P  # 先端 y方向

        for _ in range(20):
            K = asm.assemble_tangent(u)
            f_int = asm.assemble_internal_force(u)
            r = f_ext - f_int
            r_free = r[free_dofs]

            if np.linalg.norm(r_free) < 1e-6:
                break

            K_dense = K.toarray() if hasattr(K, "toarray") else K
            K_ff = K_dense[np.ix_(free_dofs, free_dofs)]
            du = np.linalg.solve(K_ff, r_free)
            u[free_dofs] += du

        # 収束後、先端たわみを確認
        tip_y = u[ndof - 5]
        # 解析解: δ = PL³/(3EI)
        delta_exact = P * L**3 / (3 * E * sec.Iz)
        assert tip_y == pytest.approx(delta_exact, rel=0.05)

    def test_axial_extension(self) -> None:
        """軸方向引張の線形応答."""
        n_elems = 5
        L = 1.0
        E = 200e9
        G = 80e9
        A = 1e-4
        Iy = 1e-8
        Iz = 1e-8
        J = 2e-8

        coords, conn = _cantilever_beam(n_elems, L)
        ndof = (n_elems + 1) * 6
        u = np.zeros(ndof)

        # 先端に軸力
        P = 1e3
        f_ext = np.zeros(ndof)
        f_ext[ndof - 6] = P  # 先端 x方向

        free_dofs = np.arange(6, ndof)
        K, _ = assemble_cr_beam3d(
            coords,
            conn,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            5 / 6,
            5 / 6,
            sparse=False,
        )
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        r_free = f_ext[free_dofs]
        du = np.linalg.solve(K_ff, r_free)
        u[free_dofs] = du

        # 解析解: δ = PL/(EA)
        delta_exact = P * L / (E * A)
        tip_x = u[ndof - 6]
        assert tip_x == pytest.approx(delta_exact, rel=1e-6)
