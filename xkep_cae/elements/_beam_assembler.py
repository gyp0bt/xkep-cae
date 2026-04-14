"""Updated Lagrangian CR 梁アセンブラ.

ULCRBeamAssembler: Updated Lagrangian + Corotational 定式化の梁アセンブラ。
各収束ステップ後に参照配置を更新し、ヘリカル梁の大回転に対応。

ULCRFiberBeamAssembler: ファイバー断面積分による材料非線形版。
弾性材料では ULCRBeamAssembler と一致する。

status-250: ULCRBeamAssemblerProcess 追加（脱法修正 A1）。
status-329: ULCRFiberBeamAssembler 追加（Phase F4）。

[← README](../../README.md)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp_mod

from xkep_cae.core import PreProcess, ProcessMeta
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


# ====================================================================
# Process I/O データ
# ====================================================================


@dataclass(frozen=True)
class ULCRBeamAssemblerInput:
    """UL+CR 梁アセンブラ生成の入力.

    Attributes:
        node_coords: 節点座標 (n_nodes, 3)
        connectivity: 要素接続 (n_elems, 2)
        E: ヤング率
        G: せん断弾性係数
        A: 断面積
        Iy: 断面二次モーメント (y軸)
        Iz: 断面二次モーメント (z軸)
        J: ねじり定数
        kappa_y: せん断補正係数 (y)
        kappa_z: せん断補正係数 (z), 0なら kappa_y と同値
        v_ref: 参照ベクトル (省略可)
        scf: 応力集中係数 (省略可)
    """

    node_coords: np.ndarray
    connectivity: np.ndarray
    E: float
    G: float
    A: float
    Iy: float
    Iz: float
    J: float
    kappa_y: float
    kappa_z: float = 0.0
    v_ref: np.ndarray | None = None
    scf: float | None = None


@dataclass(frozen=True)
class ULCRBeamAssemblerOutput:
    """UL+CR 梁アセンブラ生成の出力.

    Attributes:
        assembler: 生成された ULCRBeamAssembler インスタンス
        n_nodes: 節点数
        ndof: 自由度数
    """

    assembler: object  # ULCRBeamAssembler
    n_nodes: int
    ndof: int


# ====================================================================
# Process
# ====================================================================


class ULCRBeamAssemblerProcess(
    PreProcess[ULCRBeamAssemblerInput, ULCRBeamAssemblerOutput],
):
    """UL+CR 梁アセンブラ生成 Process.

    Updated Lagrangian + Corotational 定式化の梁アセンブラを生成する。
    """

    meta = ProcessMeta(
        name="ULCRBeamAssembler",
        module="pre",
        version="1.0.0",
        document_path="../../docs/elements.md",
    )

    def process(self, input_data: ULCRBeamAssemblerInput) -> ULCRBeamAssemblerOutput:
        """UL+CR 梁アセンブラを生成."""
        assembler = ULCRBeamAssembler(
            node_coords=input_data.node_coords,
            connectivity=input_data.connectivity,
            E=input_data.E,
            G=input_data.G,
            A=input_data.A,
            Iy=input_data.Iy,
            Iz=input_data.Iz,
            J=input_data.J,
            kappa_y=input_data.kappa_y,
            kappa_z=input_data.kappa_z,
            v_ref=input_data.v_ref,
            scf=input_data.scf,
        )
        return ULCRBeamAssemblerOutput(
            assembler=assembler,
            n_nodes=assembler.n_nodes,
            ndof=assembler.ndof,
        )


# ====================================================================
# ファイバー梁アセンブラ（Phase F4）
# ====================================================================


class ULCRFiberBeamAssembler:
    """Updated Lagrangian CR 梁アセンブラ（ファイバー断面積分版）.

    ULCRBeamAssembler と同一の CR 運動学を使用しつつ、
    材料応答をファイバー断面積分で評価する。
    弾性材料では ULCRBeamAssembler と一致する。

    使い方::

        assembler = ULCRFiberBeamAssembler(
            coords, conn, section, material, G, J, kappa_y,
        )
        K = assembler.assemble_tangent(u_incr)
        f = assembler.assemble_internal_force(u_incr)
        assembler.update_reference(u_converged)

    設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md Phase F4
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        section: object,  # CircularFiberSection
        material: object,  # Fiber1DMaterialStrategy
        G: float,
        J: float,
        kappa_y: float,
        kappa_z: float = 0.0,
        *,
        v_ref: np.ndarray | None = None,
    ):
        from xkep_cae.elements.fiber.state import Fiber1DState, SectionState

        self.coords_ref = node_coords.copy()
        self.connectivity = connectivity
        self.n_nodes = len(node_coords)
        self.ndof = self.n_nodes * 6
        self.section = section
        self.material = material
        self.G = G
        self.J = J
        self.kappa_y = kappa_y
        self.kappa_z = kappa_z if kappa_z > 0 else kappa_y
        self.v_ref = v_ref
        self.R_ref = np.tile(np.eye(3), (self.n_nodes, 1, 1))
        self._u_total_accum = np.zeros(self.ndof)

        # 要素ごとのセクション状態（1積分点/要素）
        n_elems = len(connectivity)
        n_fiber = section.n_fiber
        # material に initial_state() があれば使用（MultiLayerFriction 等）
        if hasattr(material, "initial_state"):
            fiber_init = material.initial_state()
        else:
            fiber_init = Fiber1DState()
        init_state = SectionState(
            fibers=tuple(fiber_init for _ in range(n_fiber)),
        )
        self._section_states: list[SectionState] = [init_state] * n_elems
        self._section_states_trial: list[SectionState] = list(self._section_states)

        # チェックポイント
        self._ckpt_coords_ref: np.ndarray | None = None
        self._ckpt_R_ref: np.ndarray | None = None
        self._ckpt_u_total_accum: np.ndarray | None = None
        self._ckpt_section_states: list | None = None

    def _assemble(
        self,
        u_incr: np.ndarray,
        *,
        stiffness: bool = True,
        internal_force: bool = True,
    ) -> tuple[object | None, np.ndarray | None]:
        """ファイバー梁の全体アセンブリ."""
        from xkep_cae.elements.fiber.strand_beam import (
            StrandFiberBeamConfig,
            StrandFiberBeamProcess,
        )

        n_elems = len(self.connectivity)
        ndof = self.ndof
        proc = StrandFiberBeamProcess()

        f_int_global = np.zeros(ndof) if internal_force else None

        rows_list: list[np.ndarray] = []
        cols_list: list[np.ndarray] = []
        vals_list: list[np.ndarray] = []

        trial_states: list = list(self._section_states)

        for e in range(n_elems):
            n1, n2 = int(self.connectivity[e, 0]), int(self.connectivity[e, 1])
            coords_e = self.coords_ref[[n1, n2]]
            edofs = np.array([6 * n1 + i for i in range(6)] + [6 * n2 + i for i in range(6)])
            u_e = u_incr[edofs]

            cfg = StrandFiberBeamConfig(
                coords_init=coords_e,
                u_elem=u_e,
                section=self.section,
                material=self.material,
                section_state=self._section_states[e],
                G=self.G,
                J=self.J,
                kappa_y=self.kappa_y,
                kappa_z=self.kappa_z,
                v_ref=self.v_ref,
            )
            result = proc.process(cfg)
            trial_states[e] = result.section_state_new

            if internal_force:
                for k in range(12):
                    f_int_global[edofs[k]] += result.f_int[k]

            if stiffness:
                Ke = result.K_elem
                ii = np.repeat(edofs, 12)
                jj = np.tile(edofs, 12)
                rows_list.append(ii)
                cols_list.append(jj)
                vals_list.append(Ke.ravel())

        self._section_states_trial = trial_states

        K_T = None
        if stiffness and rows_list:
            rows = np.concatenate(rows_list)
            cols = np.concatenate(cols_list)
            vals = np.concatenate(vals_list)
            K_T = sp_mod.csr_matrix((vals, (rows, cols)), shape=(ndof, ndof))

        return K_T, f_int_global

    def assemble_tangent(self, u_incr: np.ndarray) -> object:
        """増分変位から接線剛性行列を計算."""
        K_T, _ = self._assemble(u_incr, stiffness=True, internal_force=False)
        return K_T

    def assemble_internal_force(self, u_incr: np.ndarray) -> np.ndarray:
        """増分変位から内力ベクトルを計算."""
        _, f_int = self._assemble(u_incr, stiffness=False, internal_force=True)
        return f_int

    def assemble_tangent_and_force(self, u_incr: np.ndarray) -> tuple[object, np.ndarray]:
        """接線剛性と内力を同時に計算（効率化）."""
        K_T, f_int = self._assemble(u_incr, stiffness=True, internal_force=True)
        return K_T, f_int

    def update_reference(self, u_incr: np.ndarray) -> None:
        """収束後に参照配置とセクション状態を更新."""
        for i in range(self.n_nodes):
            self.coords_ref[i] += u_incr[6 * i : 6 * i + 3]
            theta_incr = u_incr[6 * i + 3 : 6 * i + 6]
            R_incr = _rotvec_to_rotmat(theta_incr)
            self.R_ref[i] = self.R_ref[i] @ R_incr
        self._u_total_accum += u_incr
        self._section_states = list(self._section_states_trial)

    def checkpoint(self) -> None:
        """参照配置とセクション状態のチェックポイントを保存.

        status-331: TL モード（update_reference なし）でも
        セクション状態を確定するため、checkpoint 時に
        trial 状態をコミットする。UL モードでは update_reference で
        既にコミット済みなので二重更新だが実害なし。
        """
        # Trial 状態をコミット（TL モードでの状態確定に必要）
        self._section_states = list(self._section_states_trial)
        self._ckpt_coords_ref = self.coords_ref.copy()
        self._ckpt_R_ref = self.R_ref.copy()
        self._ckpt_u_total_accum = self._u_total_accum.copy()
        self._ckpt_section_states = list(self._section_states)

    def rollback(self) -> None:
        """チェックポイントから復元."""
        if self._ckpt_coords_ref is not None:
            self.coords_ref = self._ckpt_coords_ref.copy()
            self.R_ref = self._ckpt_R_ref.copy()
            self._u_total_accum = self._ckpt_u_total_accum.copy()
            self._section_states = list(self._ckpt_section_states)

    @property
    def u_total_accum(self) -> np.ndarray:
        """初期配置からの累積変位."""
        return self._u_total_accum

    def get_total_displacement(self, u_incr: np.ndarray) -> np.ndarray:
        """増分変位を初期配置からの total 変位に変換."""
        return self._u_total_accum + u_incr

    def assemble_mass(self, rho: float, *, lumped: bool = True) -> object:
        """全体質量行列をアセンブリ（断面積は diameter から計算）."""
        d = self.section.diameter
        r = d / 2.0
        A = math.pi * r**2
        Iy = math.pi * r**4 / 4.0
        Iz = Iy

        ndof = self.ndof
        n_elems = len(self.connectivity)
        if lumped:
            diag = np.zeros(ndof)
            for e in range(n_elems):
                n1, n2 = self.connectivity[e]
                coords_e = self.coords_ref[[n1, n2]]
                Me_diag = timo_beam3d_lumped_mass_local(
                    rho,
                    A,
                    Iy,
                    Iz,
                    float(np.linalg.norm(coords_e[1] - coords_e[0])),
                )
                edofs = np.array([6 * n1 + i for i in range(6)] + [6 * n2 + i for i in range(6)])
                for k in range(12):
                    diag[edofs[k]] += Me_diag[k, k]
            return sp_mod.diags(diag, format="csr")
        else:
            rows_l: list[int] = []
            cols_l: list[int] = []
            vals_l: list[float] = []
            for e in range(n_elems):
                n1, n2 = self.connectivity[e]
                coords_e = self.coords_ref[[n1, n2]]
                Me = timo_beam3d_mass_global(
                    coords_e,
                    rho,
                    A,
                    Iy,
                    Iz,
                    v_ref=self.v_ref,
                    lumped=False,
                )
                edofs = [6 * n1 + i for i in range(6)] + [6 * n2 + i for i in range(6)]
                for ii in range(12):
                    for jj in range(12):
                        if abs(Me[ii, jj]) > 1e-30:
                            rows_l.append(edofs[ii])
                            cols_l.append(edofs[jj])
                            vals_l.append(Me[ii, jj])
            return sp_mod.csr_matrix(
                (np.array(vals_l), (np.array(rows_l), np.array(cols_l))),
                shape=(ndof, ndof),
            )


# ====================================================================
# ファイバー梁アセンブラ Process I/O
# ====================================================================


@dataclass(frozen=True)
class ULCRFiberBeamAssemblerInput:
    """UL+CR ファイバー梁アセンブラ生成の入力.

    Attributes:
        node_coords: 節点座標 (n_nodes, 3)
        connectivity: 要素接続 (n_elems, 2)
        section: ファイバー断面定義（CircularFiberSection）
        material: 1D 材料則（Fiber1DMaterialStrategy 準拠）
        G: せん断弾性係数
        J: ねじり定数
        kappa_y: せん断補正係数 (y)
        kappa_z: せん断補正係数 (z)
        v_ref: 参照ベクトル
    """

    node_coords: np.ndarray
    connectivity: np.ndarray
    section: object
    material: object
    G: float
    J: float
    kappa_y: float
    kappa_z: float = 0.0
    v_ref: np.ndarray | None = None


@dataclass(frozen=True)
class ULCRFiberBeamAssemblerOutput:
    """UL+CR ファイバー梁アセンブラ生成の出力.

    Attributes:
        assembler: 生成された ULCRFiberBeamAssembler インスタンス
        n_nodes: 節点数
        ndof: 自由度数
    """

    assembler: object
    n_nodes: int
    ndof: int


class ULCRFiberBeamAssemblerProcess(
    PreProcess[ULCRFiberBeamAssemblerInput, ULCRFiberBeamAssemblerOutput],
):
    """UL+CR ファイバー梁アセンブラ生成 Process.

    ファイバー断面積分による材料非線形版の梁アセンブラを生成する。
    弾性材料では ULCRBeamAssemblerProcess と同等の結果を返す。

    設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md Phase F4
    """

    meta = ProcessMeta(
        name="ULCRFiberBeamAssembler",
        module="pre",
        version="1.0.0",
        document_path="../../docs/elements.md",
    )

    def process(self, input_data: ULCRFiberBeamAssemblerInput) -> ULCRFiberBeamAssemblerOutput:
        """UL+CR ファイバー梁アセンブラを生成."""
        assembler = ULCRFiberBeamAssembler(
            node_coords=input_data.node_coords,
            connectivity=input_data.connectivity,
            section=input_data.section,
            material=input_data.material,
            G=input_data.G,
            J=input_data.J,
            kappa_y=input_data.kappa_y,
            kappa_z=input_data.kappa_z,
            v_ref=input_data.v_ref,
        )
        return ULCRFiberBeamAssemblerOutput(
            assembler=assembler,
            n_nodes=assembler.n_nodes,
            ndof=assembler.ndof,
        )
