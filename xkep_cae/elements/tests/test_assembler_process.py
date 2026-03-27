"""アセンブラ Process のテスト.

ULCRBeamAssemblerProcess, Hex8AssemblerProcess, MixedAssemblerProcess の
Process API テスト。

status-250: 脱法修正 A1-A3 のテスト。

[← README](../../../README.md)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae.core import binds_to
from xkep_cae.elements._beam_assembler import (
    ULCRBeamAssemblerInput,
    ULCRBeamAssemblerOutput,
    ULCRBeamAssemblerProcess,
)
from xkep_cae.elements._hex8_assembler import (
    Hex8AssemblerInput,
    Hex8AssemblerOutput,
    Hex8AssemblerProcess,
)
from xkep_cae.elements._mixed_assembler import (
    MixedAssemblerInput,
    MixedAssemblerOutput,
    MixedAssemblerProcess,
)


def _beam_input(n_elems: int = 5) -> ULCRBeamAssemblerInput:
    """テスト用梁アセンブラ入力を生成."""
    L = 1.0
    coords = np.column_stack(
        [np.linspace(0, L, n_elems + 1), np.zeros(n_elems + 1), np.zeros(n_elems + 1)]
    )
    conn = np.column_stack([np.arange(n_elems), np.arange(1, n_elems + 1)])
    return ULCRBeamAssemblerInput(
        node_coords=coords,
        connectivity=conn,
        E=200e9,
        G=80e9,
        A=1e-4,
        Iy=1e-8,
        Iz=1e-8,
        J=2e-8,
        kappa_y=5 / 6,
        kappa_z=5 / 6,
    )


# ====================================================================
# A1: ULCRBeamAssemblerProcess
# ====================================================================


@binds_to(ULCRBeamAssemblerProcess)
class TestULCRBeamAssemblerProcessAPI:
    """ULCRBeamAssemblerProcess の API テスト."""

    def test_process_creates_assembler(self) -> None:
        """アセンブラが正しく生成される."""
        inp = _beam_input()
        result = ULCRBeamAssemblerProcess().process(inp)

        assert isinstance(result, ULCRBeamAssemblerOutput)
        assert result.n_nodes == 6
        assert result.ndof == 36
        assert result.assembler is not None

    def test_assembler_tangent(self) -> None:
        """接線剛性行列が正しい形状."""
        result = ULCRBeamAssemblerProcess().process(_beam_input())
        asm = result.assembler
        K = asm.assemble_tangent(np.zeros(result.ndof))
        assert K is not None
        assert K.shape == (result.ndof, result.ndof)

    def test_assembler_internal_force_zero(self) -> None:
        """ゼロ変位で内力がゼロ."""
        result = ULCRBeamAssemblerProcess().process(_beam_input())
        asm = result.assembler
        f = asm.assemble_internal_force(np.zeros(result.ndof))
        np.testing.assert_allclose(f, 0, atol=1e-10)

    def test_assembler_mass(self) -> None:
        """質量行列が生成される."""
        result = ULCRBeamAssemblerProcess().process(_beam_input())
        asm = result.assembler
        M = asm.assemble_mass(7850, lumped=True)
        assert M.shape == (result.ndof, result.ndof)

    def test_frozen_input(self) -> None:
        """Input は frozen dataclass."""
        inp = _beam_input()
        try:
            inp.E = 999.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow mutation")
        except AttributeError:
            pass

    def test_frozen_output(self) -> None:
        """Output は frozen dataclass."""
        result = ULCRBeamAssemblerProcess().process(_beam_input())
        try:
            result.ndof = 999  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow mutation")
        except AttributeError:
            pass


# ====================================================================
# A2: Hex8AssemblerProcess
# ====================================================================


def _hex8_input() -> Hex8AssemblerInput:
    """テスト用 HEX8 アセンブラ入力（1要素キューブ）."""
    coords = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    return Hex8AssemblerInput(
        node_coords=coords,
        connectivity=conn,
        E=200e3,
        nu=0.3,
    )


@binds_to(Hex8AssemblerProcess)
class TestHex8AssemblerProcessAPI:
    """Hex8AssemblerProcess の API テスト."""

    def test_process_creates_assembler(self) -> None:
        """アセンブラが正しく生成される."""
        result = Hex8AssemblerProcess().process(_hex8_input())

        assert isinstance(result, Hex8AssemblerOutput)
        assert result.n_nodes == 8
        assert len(result.rotational_dofs) == 24  # 8 nodes * 3 rot DOF
        assert result.assembler is not None

    def test_assembler_tangent(self) -> None:
        """接線剛性行列が正しい形状."""
        result = Hex8AssemblerProcess().process(_hex8_input())
        asm = result.assembler
        ndof = 8 * 6
        K = asm.assemble_tangent(np.zeros(ndof))
        assert isinstance(K, sp.csr_matrix)
        assert K.shape == (ndof, ndof)

    def test_assembler_internal_force_zero(self) -> None:
        """ゼロ変位で内力がゼロ."""
        result = Hex8AssemblerProcess().process(_hex8_input())
        asm = result.assembler
        ndof = 8 * 6
        f = asm.assemble_internal_force(np.zeros(ndof))
        np.testing.assert_allclose(f, 0, atol=1e-10)

    def test_rotational_dofs_are_tuple(self) -> None:
        """rotational_dofs が tuple（frozen 対応）."""
        result = Hex8AssemblerProcess().process(_hex8_input())
        assert isinstance(result.rotational_dofs, tuple)

    def test_frozen_input(self) -> None:
        """Input は frozen dataclass."""
        inp = _hex8_input()
        try:
            inp.E = 999.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow mutation")
        except AttributeError:
            pass

    def test_frozen_output(self) -> None:
        """Output は frozen dataclass."""
        result = Hex8AssemblerProcess().process(_hex8_input())
        try:
            result.n_nodes = 999  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow mutation")
        except AttributeError:
            pass


# ====================================================================
# A3: MixedAssemblerProcess
# ====================================================================


@binds_to(MixedAssemblerProcess)
class TestMixedAssemblerProcessAPI:
    """MixedAssemblerProcess の API テスト."""

    def test_process_creates_assembler(self) -> None:
        """混合アセンブラが正しく生成される."""
        beam_result = ULCRBeamAssemblerProcess().process(_beam_input())
        beam_asm = beam_result.assembler

        hex_result = Hex8AssemblerProcess().process(
            Hex8AssemblerInput(
                node_coords=np.zeros((8, 3)),
                connectivity=np.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
                E=200e3,
                nu=0.3,
                global_node_offset=6,
                total_ndof=84,  # (6+8)*6
            )
        )
        hex_asm = hex_result.assembler

        total_ndof = 84
        result = MixedAssemblerProcess().process(
            MixedAssemblerInput(
                beam_assembler=beam_asm,
                hex_assembler=hex_asm,
                total_ndof=total_ndof,
            )
        )

        assert isinstance(result, MixedAssemblerOutput)
        assert result.total_ndof == total_ndof
        assert result.assembler is not None

    def test_mixed_tangent_shape(self) -> None:
        """混合接線剛性行列の形状."""
        beam_result = ULCRBeamAssemblerProcess().process(_beam_input())
        total_ndof = beam_result.ndof
        result = MixedAssemblerProcess().process(
            MixedAssemblerInput(
                beam_assembler=beam_result.assembler,
                hex_assembler=beam_result.assembler,  # 同じでOK（形状テスト用）
                total_ndof=total_ndof,
            )
        )
        asm = result.assembler
        K = asm.assemble_tangent(np.zeros(total_ndof))
        assert K.shape == (total_ndof, total_ndof)

    def test_frozen_input(self) -> None:
        """Input は frozen dataclass."""
        inp = MixedAssemblerInput(
            beam_assembler=None,
            hex_assembler=None,
            total_ndof=6,
        )
        try:
            inp.total_ndof = 999  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow mutation")
        except AttributeError:
            pass

    def test_frozen_output(self) -> None:
        """Output は frozen dataclass."""
        beam_result = ULCRBeamAssemblerProcess().process(_beam_input())
        result = MixedAssemblerProcess().process(
            MixedAssemblerInput(
                beam_assembler=beam_result.assembler,
                hex_assembler=beam_result.assembler,
                total_ndof=beam_result.ndof,
            )
        )
        try:
            result.total_ndof = 999  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow mutation")
        except AttributeError:
            pass
