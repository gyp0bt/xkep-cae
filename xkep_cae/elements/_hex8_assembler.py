"""HEX8 ソリッドアセンブラ.

ULCRBeamAssembler と同じインターフェースで HEX8 ソリッド要素の
全体剛性行列・内力ベクトルを組み立てる。

全節点 6 DOF/node レイアウト（梁要素との混合組立用）。
回転 DOF (d=3,4,5) はゼロ剛性。

[← README](../../README.md)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae.elements._hex8 import hex8_internal_force_6dof, hex8_stiffness_6dof


class Hex8Assembler:
    """HEX8 ソリッドメッシュのアセンブラ（6 DOF/node）.

    使い方::

        asm = Hex8Assembler(node_coords, connectivity, E, nu,
                            global_node_offset=n_beam_nodes)
        K = asm.assemble_tangent(u)
        f = asm.assemble_internal_force(u)
        asm.update_reference(u_converged)
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        E: float,
        nu: float,
        *,
        global_node_offset: int = 0,
        total_ndof: int = 0,
    ) -> None:
        """初期化.

        Args:
            node_coords: (n_nodes, 3) ソリッド節点の初期座標
            connectivity: (n_elems, 8) 要素接続（ローカル節点番号）
            E: ヤング率
            nu: ポアソン比
            global_node_offset: 全体節点番号でのオフセット
                                （梁節点数を指定 → DOF = 6*(offset+i)+d）
            total_ndof: 全体DOF数（指定しなければ自動計算）
        """
        self.coords_ref = node_coords.copy()
        self.connectivity = connectivity
        self.n_nodes = len(node_coords)
        self.E = E
        self.nu = nu
        self._offset = global_node_offset
        n_total = self._offset + self.n_nodes
        self._total_ndof = total_ndof if total_ndof > 0 else n_total * 6
        self._u_total_accum = np.zeros(self._total_ndof)
        self._ckpt_coords_ref: np.ndarray | None = None
        self._ckpt_u_total_accum: np.ndarray | None = None

    def _global_dofs(self, local_nodes: np.ndarray) -> np.ndarray:
        """ローカル節点番号 → 全体DOFインデックス (8*6=48)."""
        global_nodes = local_nodes + self._offset
        dofs = np.empty(len(local_nodes) * 6, dtype=int)
        for i, gn in enumerate(global_nodes):
            for d in range(6):
                dofs[i * 6 + d] = 6 * gn + d
        return dofs

    def assemble_tangent(self, u: np.ndarray) -> sp.csr_matrix:
        """全体接線剛性行列."""
        ndof = self._total_ndof
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for elem in self.connectivity:
            local_nodes = np.asarray(elem, dtype=int)
            coords_e = self.coords_ref[local_nodes]
            Ke = hex8_stiffness_6dof(coords_e, self.E, self.nu)
            dofs = self._global_dofs(local_nodes)

            for ii in range(48):
                for jj in range(48):
                    if abs(Ke[ii, jj]) > 1e-30:
                        rows.append(dofs[ii])
                        cols.append(dofs[jj])
                        vals.append(Ke[ii, jj])

        if not vals:
            return sp.csr_matrix((ndof, ndof))
        return sp.csr_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))),
            shape=(ndof, ndof),
        )

    def assemble_internal_force(self, u: np.ndarray) -> np.ndarray:
        """全体内力ベクトル."""
        ndof = self._total_ndof
        f_int = np.zeros(ndof)

        for elem in self.connectivity:
            local_nodes = np.asarray(elem, dtype=int)
            coords_e = self.coords_ref[local_nodes]
            dofs = self._global_dofs(local_nodes)
            u_elem = u[dofs]
            fe = hex8_internal_force_6dof(coords_e, u_elem, self.E, self.nu)
            for k in range(48):
                f_int[dofs[k]] += fe[k]

        return f_int

    def update_reference(self, u_incr: np.ndarray) -> None:
        """収束後に参照配置を更新（UL用）."""
        for i in range(self.n_nodes):
            gn = self._offset + i
            self.coords_ref[i] += u_incr[6 * gn : 6 * gn + 3]
        self._u_total_accum += u_incr

    def checkpoint(self) -> None:
        """参照配置のチェックポイントを保存."""
        self._ckpt_coords_ref = self.coords_ref.copy()
        self._ckpt_u_total_accum = self._u_total_accum.copy()

    def rollback(self) -> None:
        """チェックポイントから参照配置を復元."""
        if self._ckpt_coords_ref is not None:
            self.coords_ref = self._ckpt_coords_ref.copy()
            self._u_total_accum = self._ckpt_u_total_accum.copy()

    @property
    def u_total_accum(self) -> np.ndarray:
        """初期配置からの累積変位（出力用）."""
        return self._u_total_accum

    def rotational_dofs(self) -> list[int]:
        """ソリッド節点の回転DOFリスト（固定すべきDOF）."""
        rot_dofs = []
        for i in range(self.n_nodes):
            gn = self._offset + i
            for d in [3, 4, 5]:
                rot_dofs.append(6 * gn + d)
        return rot_dofs
