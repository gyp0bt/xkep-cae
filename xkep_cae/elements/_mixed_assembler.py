"""梁 + HEX8 ソリッド混合アセンブラ.

ULCRBeamAssembler と Hex8Assembler を統合し、
ContactFrictionProcess が要求する AssembleCallbacks を提供する。

[← README](../../README.md)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _resize_csr(mat: sp.spmatrix, n: int) -> sp.csr_matrix:
    """疎行列を (n, n) にリサイズ（ゼロ行/列を追加）."""
    if mat.shape == (n, n):
        return mat.tocsr()
    coo = mat.tocoo()
    return sp.coo_matrix(
        (coo.data, (coo.row, coo.col)),
        shape=(n, n),
    ).tocsr()


class MixedAssembler:
    """梁 + ソリッド混合アセンブラ.

    全節点 6 DOF/node レイアウト。
    ソリッド節点の回転 DOF は外部で固定する。

    使い方::

        mixed = MixedAssembler(beam_asm, hex_asm, total_ndof)
        K = mixed.assemble_tangent(u)
        f = mixed.assemble_internal_force(u)
        mixed.update_reference(u_converged)
    """

    def __init__(
        self,
        beam_assembler: object,
        hex_assembler: object,
        total_ndof: int,
    ) -> None:
        self._beam = beam_assembler
        self._hex = hex_assembler
        self._total_ndof = total_ndof

    @property
    def coords_ref(self) -> np.ndarray:
        """全節点の参照座標 (n_total_nodes, 3)."""
        return np.vstack([self._beam.coords_ref, self._hex.coords_ref])

    @property
    def ndof(self) -> int:
        """全体 DOF 数."""
        return self._total_ndof

    def assemble_tangent(self, u: np.ndarray) -> sp.csr_matrix:
        """全体接線剛性行列（梁 + ソリッド）."""
        n = self._total_ndof
        K_beam = _resize_csr(self._beam.assemble_tangent(u), n)
        K_hex = _resize_csr(self._hex.assemble_tangent(u), n)
        return K_beam + K_hex

    def assemble_internal_force(self, u: np.ndarray) -> np.ndarray:
        """全体内力ベクトル（梁 + ソリッド）."""
        n = self._total_ndof
        f_beam = self._beam.assemble_internal_force(u)
        f_hex = self._hex.assemble_internal_force(u)

        f = np.zeros(n)
        f[: len(f_beam)] += f_beam
        f[: len(f_hex)] += f_hex
        return f

    def update_reference(self, u_incr: np.ndarray) -> None:
        """参照配置更新を両アセンブラに委譲.

        梁アセンブラには梁 DOF 範囲のみ渡す（サイズ不一致回避）。
        HEX8 アセンブラは全体 DOF を受け取り、offset で自分の範囲を参照。
        """
        beam_ndof = self._beam.ndof
        self._beam.update_reference(u_incr[:beam_ndof])
        self._hex.update_reference(u_incr)

    def checkpoint(self) -> None:
        """チェックポイント保存."""
        self._beam.checkpoint()
        self._hex.checkpoint()

    def rollback(self) -> None:
        """チェックポイント復元."""
        self._beam.rollback()
        self._hex.rollback()

    @property
    def u_total_accum(self) -> np.ndarray:
        """初期配置からの累積変位（全DOF）."""
        n = self._total_ndof
        beam_u = self._beam.u_total_accum
        hex_u = self._hex.u_total_accum
        u = np.zeros(n)
        u[: len(beam_u)] += beam_u
        u[: len(hex_u)] += hex_u
        return u

    def get_total_displacement(self, u_incr: np.ndarray) -> np.ndarray:
        """増分変位を初期配置からの total 変位に変換."""
        return self.u_total_accum + u_incr
