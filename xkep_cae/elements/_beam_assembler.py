"""Updated Lagrangian CR 梁アセンブラ.

ULCRBeamAssembler: Updated Lagrangian + Corotational 定式化の梁アセンブラ。
各収束ステップ後に参照配置を更新し、ヘリカル梁の大回転に対応。
"""

from __future__ import annotations

import numpy as np

from xkep_cae.elements._beam_assembly import assemble_cr_beam3d
from xkep_cae.elements._beam_cr import (
    _rotvec_to_rotmat,
    timo_beam3d_lumped_mass_local,
    timo_beam3d_mass_global,
)


class ULCRBeamAssembler:
    """Updated Lagrangian CR 梁アセンブラ.

    標準 CR 定式化はヘリカル梁で ~13° 以上の累積回転に対し収束劣化する。
    UL では各収束ステップ後に参照配置を更新し、ステップ内変形を小さく保つ。

    使い方::

        assembler = ULCRBeamAssembler(coords, conn, E, G, A, Iy, Iz, J, kappa)
        K = assembler.assemble_tangent(u_incr)
        f = assembler.assemble_internal_force(u_incr)
        assembler.update_reference(u_converged)
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        E: float,
        G: float,
        A: float,
        Iy: float,
        Iz: float,
        J: float,
        kappa_y: float,
        kappa_z: float = 0.0,
        *,
        v_ref: np.ndarray | None = None,
        scf: float | None = None,
    ):
        self.coords_ref = node_coords.copy()
        self.connectivity = connectivity
        self.n_nodes = len(node_coords)
        self.ndof = self.n_nodes * 6
        self.E = E
        self.G = G
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.kappa_y = kappa_y
        self.kappa_z = kappa_z if kappa_z > 0 else kappa_y
        self.v_ref = v_ref
        self.scf = scf
        self.R_ref = np.tile(np.eye(3), (self.n_nodes, 1, 1))
        self._u_total_accum = np.zeros(self.ndof)
        self._ckpt_coords_ref: np.ndarray | None = None
        self._ckpt_R_ref: np.ndarray | None = None
        self._ckpt_u_total_accum: np.ndarray | None = None

    def _to_total_u(self, u_incr: np.ndarray) -> np.ndarray:
        """増分変位を CR 要素用の変位に変換."""
        return u_incr.copy()

    def assemble_tangent(self, u_incr: np.ndarray) -> object:
        """増分変位から接線剛性行列を計算."""
        K_T, _ = assemble_cr_beam3d(
            self.coords_ref,
            self.connectivity,
            u_incr,
            self.E,
            self.G,
            self.A,
            self.Iy,
            self.Iz,
            self.J,
            self.kappa_y,
            self.kappa_z,
            v_ref=self.v_ref,
            scf=self.scf,
            stiffness=True,
            internal_force=False,
            analytical_tangent=True,
        )
        return K_T

    def assemble_internal_force(self, u_incr: np.ndarray) -> np.ndarray:
        """増分変位から内力ベクトルを計算."""
        u_total = self._to_total_u(u_incr)
        _, f_int = assemble_cr_beam3d(
            self.coords_ref,
            self.connectivity,
            u_total,
            self.E,
            self.G,
            self.A,
            self.Iy,
            self.Iz,
            self.J,
            self.kappa_y,
            self.kappa_z,
            v_ref=self.v_ref,
            scf=self.scf,
            stiffness=False,
            internal_force=True,
        )
        return f_int

    def update_reference(self, u_incr: np.ndarray) -> None:
        """収束後に参照配置を更新."""
        for i in range(self.n_nodes):
            self.coords_ref[i] += u_incr[6 * i : 6 * i + 3]
            theta_incr = u_incr[6 * i + 3 : 6 * i + 6]
            R_incr = _rotvec_to_rotmat(theta_incr)
            self.R_ref[i] = self.R_ref[i] @ R_incr
        self._u_total_accum += u_incr

    def checkpoint(self) -> None:
        """参照配置のチェックポイントを保存."""
        self._ckpt_coords_ref = self.coords_ref.copy()
        self._ckpt_R_ref = self.R_ref.copy()
        self._ckpt_u_total_accum = self._u_total_accum.copy()

    def rollback(self) -> None:
        """チェックポイントから参照配置を復元."""
        if self._ckpt_coords_ref is not None:
            self.coords_ref = self._ckpt_coords_ref.copy()
            self.R_ref = self._ckpt_R_ref.copy()
            self._u_total_accum = self._ckpt_u_total_accum.copy()

    @property
    def u_total_accum(self) -> np.ndarray:
        """初期配置からの累積変位（出力用）."""
        return self._u_total_accum

    def get_total_displacement(self, u_incr: np.ndarray) -> np.ndarray:
        """増分変位を初期配置からの total 変位に変換."""
        return self._u_total_accum + u_incr

    def assemble_mass(self, rho: float, *, lumped: bool = True) -> object:
        """全体質量行列をアセンブリ."""
        import scipy.sparse as sp_mod

        ndof = self.ndof
        n_elems = len(self.connectivity)
        if lumped:
            diag = np.zeros(ndof)
            for e in range(n_elems):
                n1, n2 = self.connectivity[e]
                coords_e = self.coords_ref[[n1, n2]]
                Me_diag = timo_beam3d_lumped_mass_local(
                    rho,
                    self.A,
                    self.Iy,
                    self.Iz,
                    float(np.linalg.norm(coords_e[1] - coords_e[0])),
                )
                edofs = np.array([6 * n1 + i for i in range(6)] + [6 * n2 + i for i in range(6)])
                for k in range(12):
                    diag[edofs[k]] += Me_diag[k, k]
            return sp_mod.diags(diag, format="csr")
        else:
            rows: list[int] = []
            cols: list[int] = []
            vals: list[float] = []
            for e in range(n_elems):
                n1, n2 = self.connectivity[e]
                coords_e = self.coords_ref[[n1, n2]]
                Me = timo_beam3d_mass_global(
                    coords_e,
                    rho,
                    self.A,
                    self.Iy,
                    self.Iz,
                    v_ref=self.v_ref,
                    lumped=False,
                )
                edofs = [6 * n1 + i for i in range(6)] + [6 * n2 + i for i in range(6)]
                for ii in range(12):
                    for jj in range(12):
                        if abs(Me[ii, jj]) > 1e-30:
                            rows.append(edofs[ii])
                            cols.append(edofs[jj])
                            vals.append(Me[ii, jj])
            return sp_mod.csr_matrix(
                (np.array(vals), (np.array(rows), np.array(cols))),
                shape=(ndof, ndof),
            )
