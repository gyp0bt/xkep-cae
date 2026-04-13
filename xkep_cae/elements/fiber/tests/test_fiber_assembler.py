"""ULCRFiberBeamAssembler テスト.

ファイバー梁アセンブラが弾性材料で線形梁アセンブラと一致することを検証。
設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md Phase F4
[← README](../../../../README.md)
"""

from __future__ import annotations

import math

import numpy as np

from xkep_cae.core import binds_to
from xkep_cae.elements._beam_assembler import (
    ULCRBeamAssembler,
    ULCRFiberBeamAssembler,
    ULCRFiberBeamAssemblerInput,
    ULCRFiberBeamAssemblerProcess,
)
from xkep_cae.elements.fiber.materials import Elastic1D
from xkep_cae.elements.fiber.section import CircularFiberSection


def _make_cantilever(
    n_elems: int = 5,
    L: float = 50.0,
    d: float = 17.0,
    E: float = 200_000.0,
    G: float = 80_000.0,
) -> tuple:
    """片持ち梁のメッシュとパラメータを生成.

    Returns:
        (coords, conn, p) where p は梁パラメータの辞書
    """
    r = d / 2.0
    A = math.pi * r**2
    Iy = math.pi * r**4 / 4.0
    Iz = Iy
    J = Iy + Iz
    kappa = 6.0 / 7.0

    n_nodes = n_elems + 1
    coords = np.zeros((n_nodes, 3))
    coords[:, 0] = np.linspace(0, L, n_nodes)
    conn = np.array([[i, i + 1] for i in range(n_elems)])

    p = {
        "E": E,
        "G": G,
        "A": A,
        "Iy": Iy,
        "Iz": Iz,
        "J": J,
        "kappa": kappa,
        "d": d,
        "L": L,
    }
    return coords, conn, p


@binds_to(ULCRFiberBeamAssemblerProcess)
class TestULCRFiberBeamAssemblerAPI:
    """ファイバー梁アセンブラの API テスト."""

    def test_assembler_creation(self) -> None:
        """ULCRFiberBeamAssembler が正常に生成される."""
        coords, conn, p = _make_cantilever()
        sec = CircularFiberSection.polar(diameter=p["d"], n_radial=10, n_theta=20)
        mat = Elastic1D(E=p["E"])

        asm = ULCRFiberBeamAssembler(
            node_coords=coords,
            connectivity=conn,
            section=sec,
            material=mat,
            G=p["G"],
            J=p["J"],
            kappa_y=p["kappa"],
        )
        assert asm.n_nodes == len(coords)
        assert asm.ndof == 6 * len(coords)

    def test_zero_displacement_zero_force(self) -> None:
        """変位ゼロで内力ゼロ."""
        coords, conn, p = _make_cantilever()
        sec = CircularFiberSection.polar(diameter=p["d"], n_radial=10, n_theta=20)
        mat = Elastic1D(E=p["E"])

        asm = ULCRFiberBeamAssembler(
            coords,
            conn,
            sec,
            mat,
            p["G"],
            p["J"],
            p["kappa"],
        )
        u = np.zeros(asm.ndof)
        f = asm.assemble_internal_force(u)
        np.testing.assert_allclose(f, 0.0, atol=1e-10)

    def test_tangent_returns_sparse(self) -> None:
        """接線剛性が sparse 行列."""
        import scipy.sparse as sp

        coords, conn, p = _make_cantilever(n_elems=3)
        sec = CircularFiberSection.polar(diameter=p["d"], n_radial=8, n_theta=16)
        mat = Elastic1D(E=p["E"])

        asm = ULCRFiberBeamAssembler(
            coords,
            conn,
            sec,
            mat,
            p["G"],
            p["J"],
            p["kappa"],
        )
        u = np.zeros(asm.ndof)
        u[6 * 3 + 1] = 0.01  # 先端横変位
        K = asm.assemble_tangent(u)
        assert sp.issparse(K)
        assert K.shape == (asm.ndof, asm.ndof)

    def test_process_creation(self) -> None:
        """ULCRFiberBeamAssemblerProcess が正常に動作."""
        coords, conn, p = _make_cantilever(n_elems=3)
        sec = CircularFiberSection.polar(diameter=p["d"], n_radial=8, n_theta=16)
        mat = Elastic1D(E=p["E"])

        proc = ULCRFiberBeamAssemblerProcess()
        inp = ULCRFiberBeamAssemblerInput(
            node_coords=coords,
            connectivity=conn,
            section=sec,
            material=mat,
            G=p["G"],
            J=p["J"],
            kappa_y=p["kappa"],
        )
        result = proc.process(inp)
        assert result.n_nodes == len(coords)
        assert result.ndof == 6 * len(coords)

    def test_checkpoint_rollback(self) -> None:
        """checkpoint/rollback でセクション状態が復元される."""
        coords, conn, p = _make_cantilever(n_elems=3)
        sec = CircularFiberSection.polar(diameter=p["d"], n_radial=8, n_theta=16)
        mat = Elastic1D(E=p["E"])

        asm = ULCRFiberBeamAssembler(
            coords,
            conn,
            sec,
            mat,
            p["G"],
            p["J"],
            p["kappa"],
        )

        asm.checkpoint()
        coords_before = asm.coords_ref.copy()

        u = np.zeros(asm.ndof)
        u[6 * 3 + 1] = 0.5  # 先端横変位
        asm.assemble_internal_force(u)
        asm.update_reference(u)

        assert not np.allclose(asm.coords_ref, coords_before)

        asm.rollback()
        np.testing.assert_allclose(asm.coords_ref, coords_before)


class TestULCRFiberBeamAssemblerPhysics:
    """ファイバー梁アセンブラの物理テスト."""

    def test_internal_force_matches_linear(self) -> None:
        """弾性材料: 内力ベクトルが線形梁アセンブラと一致（< 1%）."""
        coords, conn, p = _make_cantilever(n_elems=5)
        sec = CircularFiberSection.polar(diameter=p["d"], n_radial=12, n_theta=24)
        mat = Elastic1D(E=p["E"])

        asm_fiber = ULCRFiberBeamAssembler(
            coords.copy(),
            conn,
            sec,
            mat,
            p["G"],
            p["J"],
            p["kappa"],
        )
        asm_linear = ULCRBeamAssembler(
            coords.copy(),
            conn,
            p["E"],
            p["G"],
            p["A"],
            p["Iy"],
            p["Iz"],
            p["J"],
            p["kappa"],
        )

        u = np.zeros(asm_fiber.ndof)
        u[6 * 5 + 4] = 0.01  # 先端 θ_y 回転

        f_fiber = asm_fiber.assemble_internal_force(u)
        f_linear = asm_linear.assemble_internal_force(u)

        nonzero = np.abs(f_linear) > 1e-6
        if np.any(nonzero):
            rel_err = np.max(np.abs(f_fiber[nonzero] - f_linear[nonzero])) / np.max(
                np.abs(f_linear[nonzero])
            )
            assert rel_err < 0.01, f"内力相対誤差 {rel_err:.4%} > 1%"

    def test_tangent_diagonal_matches_linear(self) -> None:
        """弾性材料: 接線剛性の対角成分が線形梁と一致（< 1%）."""
        coords, conn, p = _make_cantilever(n_elems=3)
        sec = CircularFiberSection.polar(diameter=p["d"], n_radial=12, n_theta=24)
        mat = Elastic1D(E=p["E"])

        asm_fiber = ULCRFiberBeamAssembler(
            coords.copy(),
            conn,
            sec,
            mat,
            p["G"],
            p["J"],
            p["kappa"],
        )
        asm_linear = ULCRBeamAssembler(
            coords.copy(),
            conn,
            p["E"],
            p["G"],
            p["A"],
            p["Iy"],
            p["Iz"],
            p["J"],
            p["kappa"],
        )

        u = np.zeros(asm_fiber.ndof)

        K_fiber = asm_fiber.assemble_tangent(u)
        K_linear = asm_linear.assemble_tangent(u)

        diag_fiber = np.array(K_fiber.diagonal()).flatten()
        diag_linear = np.array(K_linear.diagonal()).flatten()

        nonzero = np.abs(diag_linear) > 1.0
        if np.any(nonzero):
            rel_err = np.max(
                np.abs(diag_fiber[nonzero] - diag_linear[nonzero]) / np.abs(diag_linear[nonzero])
            )
            assert rel_err < 0.01, f"接線対角相対誤差 {rel_err:.4%} > 1%"

    def test_mass_matrix_matches_linear(self) -> None:
        """質量行列が線形梁アセンブラと一致."""
        coords, conn, p = _make_cantilever(n_elems=3)
        sec = CircularFiberSection.polar(diameter=p["d"], n_radial=10, n_theta=20)
        mat = Elastic1D(E=p["E"])

        asm_fiber = ULCRFiberBeamAssembler(
            coords.copy(),
            conn,
            sec,
            mat,
            p["G"],
            p["J"],
            p["kappa"],
        )
        asm_linear = ULCRBeamAssembler(
            coords.copy(),
            conn,
            p["E"],
            p["G"],
            p["A"],
            p["Iy"],
            p["Iz"],
            p["J"],
            p["kappa"],
        )

        rho = 7850.0
        M_fiber = asm_fiber.assemble_mass(rho, lumped=True)
        M_linear = asm_linear.assemble_mass(rho, lumped=True)

        diag_fiber = np.array(M_fiber.diagonal()).flatten()
        diag_linear = np.array(M_linear.diagonal()).flatten()

        np.testing.assert_allclose(diag_fiber, diag_linear, rtol=1e-10)
