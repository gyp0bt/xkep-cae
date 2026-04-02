"""StrandBendingOscillationProcess — 7本撚線曲げ揺動 Process.

端部剛体結合（MPC DOF消去）+ 曲げ処方変位 + 揺動サイクルの
撚線曲げ揺動解析を実行する BatchProcess。

物理モデル:
  - 7本撚線メッシュ（StrandMeshProcess）
  - 端部: MPC剛体結合（MPCEliminationProcess）
    - 各端面の全素線端部節点 → 参照点に結合
  - 境界条件:
    - 左端参照点: 全DOF固定（固定端）
    - 右端参照点: 処方回転（曲げ揺動）
  - 接触: smooth_penalty + Coulomb摩擦
  - 動的ソルバー（GeneralizedAlpha）

status-253: DOF消去MPC + 端部剛体結合 → 7本撚線曲げ揺動。

[← README](../../README.md)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.constraints.mpc_elimination import (
    MPCEliminationConfig,
    MPCEliminationProcess,
    MPCGroup,
)
from xkep_cae.contact._contact_pair import _ContactConfigInput, _ContactManagerInput
from xkep_cae.contact.solver.process import ContactFrictionProcess
from xkep_cae.core import (
    AssembleCallbacks,
    BatchProcess,
    BoundaryData,
    ContactFrictionInputData,
    ContactSetupData,
    MeshData,
    ProcessMeta,
    SolverResultData,
)
from xkep_cae.elements._beam_assembler import (
    ULCRBeamAssemblerInput,
    ULCRBeamAssemblerProcess,
)
from xkep_cae.mesh.process import StrandMeshConfig, StrandMeshProcess
from xkep_cae.numerical_tests.three_point_bend_jig import _circle_section

# ====================================================================
# 拡張系 UL アセンブララッパー
# ====================================================================


class _ExtendedULAssemblerWrapper:
    """ULアセンブラを拡張DOF系にラップする.

    ul_assembler は梁ノードのみ (ndof_beam) を扱うが、
    MPC参照点ノードを含む拡張系 (ndof_total) との整合性が必要。
    u_total_accum / coords_ref をゼロパディングで拡張し、
    checkpoint / rollback を委譲する。
    """

    def __init__(self, assembler: object, ndof_beam: int, ndof_total: int) -> None:
        self._asm = assembler
        self._ndof_beam = ndof_beam
        self._ndof_total = ndof_total

    @property
    def u_total_accum(self) -> np.ndarray:
        u = np.zeros(self._ndof_total)
        u[: self._ndof_beam] = self._asm.u_total_accum
        return u

    @property
    def coords_ref(self) -> np.ndarray:
        return self._asm.coords_ref

    def checkpoint(self) -> None:
        self._asm.checkpoint()

    def rollback(self) -> None:
        self._asm.rollback()


# ====================================================================
# 入出力データ
# ====================================================================


@dataclass(frozen=True)
class StrandBendingOscillationConfig:
    """7本撚線曲げ揺動の構成.

    Attributes:
        n_strands: 素線本数
        wire_radius: 素線半径 [mm]
        pitch_length: ピッチ長 [mm]
        gap: 素線間ギャップ [mm]
        n_elements_per_pitch: ピッチあたりの要素数
        n_pitches: ピッチ数
        E: ヤング率 [MPa]
        nu: ポアソン比
        rho: 密度 [ton/mm³]
        bending_curvature: 曲げ曲率 κ [1/mm]
        n_cycles: 揺動サイクル数
        n_increments_per_cycle: 1サイクルあたりのインクリメント数
        rho_inf: Generalized-α 数値減衰パラメータ
        mu: 摩擦係数
        k_pen: ペナルティ剛性（0=自動）
        max_nr_attempts: NR最大反復数
        tol_force: NR力収束判定
        max_increments: 最大インクリメント数
    """

    n_strands: int = 7
    wire_radius: float = 0.5  # mm (R=0.5mm → d=1mm)
    pitch_length: float = 100.0  # mm
    gap: float = 0.0  # mm (自動引き上げ)
    n_elements_per_pitch: int = 16
    n_pitches: float = 1.0
    E: float = 130.0e3  # MPa (銅)
    nu: float = 0.3
    rho: float = 8.96e-9  # ton/mm³ (銅)
    bending_curvature: float = 0.001  # 1/mm (曲率)
    n_cycles: int = 1
    n_increments_per_cycle: int = 20
    rho_inf: float = 0.9
    mu: float = 0.15
    k_pen: float = 0.0  # 0 = 自動
    max_nr_attempts: int = 50
    tol_force: float = 1e-8
    max_increments: int = 10000
    lumped_mass: bool = True
    exclude_same_strand: bool = True
    tangent_fd_diagnostic: bool = False  # ストール時FD接線診断（status-257）
    smoothing_delta: float = 0.0  # 0=自動推定（1000/wire_radius）, >0=手動指定
    huber_delta_h: float = 0.0  # >0: Huber遷移幅を直接指定（k_penスケール非依存, status-261）
    du_norm_cap: float = 0.0  # NR更新キャップ（0=制限なし）
    # チェックポイント復元（status-278: 中盤からの対策効果検証用）
    # pickle ファイルパスを指定すると、保存された u0/vel/acc から再開。
    # load_frac_start 以降の荷重増分のみ実行される。
    resume_checkpoint: str = ""  # チェックポイントファイルパス（空=通常実行）
    # free_end_mode: MPC端部剛体結合を使わず、各素線端部ノードに直接
    # 処方変位（θ_z）を与えるモード。並進DOFは自由。(status-280)
    free_end_mode: bool = False


@dataclass(frozen=True)
class StrandBendingOscillationResult:
    """7本撚線曲げ揺動の結果.

    Attributes:
        solver_result: ソルバー結果
        mesh: メッシュデータ（参照点ノード含む）
        mpc_config: MPC構成（デバッグ用）
        n_ref_nodes: 追加された参照点ノード数
        n_strand_nodes: 元の撚線ノード数
        total_ndof: 全体DOF数
        bending_angle: 処方曲げ角度 [rad]
    """

    solver_result: SolverResultData
    mesh: MeshData
    n_ref_nodes: int
    n_strand_nodes: int
    total_ndof: int
    bending_angle: float


# ====================================================================
# 端部節点収集
# ====================================================================


def _collect_end_nodes(
    connectivity: np.ndarray,
    n_strands: int,
    strand_ids: np.ndarray,
) -> tuple[list[int], list[int]]:
    """各素線の左端/右端節点を収集する.

    Returns:
        left_nodes: 左端（最小x方向）節点のリスト
        right_nodes: 右端（最大x方向）節点のリスト
    """
    left_nodes = []
    right_nodes = []

    for s in range(n_strands):
        # この素線の要素を取得
        elem_mask = strand_ids == s
        strand_elems = connectivity[elem_mask]
        if len(strand_elems) == 0:
            continue
        # 要素に1回だけ出現する節点 = 端部節点
        node_count: dict[int, int] = {}
        for e in strand_elems:
            for n in e:
                node_count[int(n)] = node_count.get(int(n), 0) + 1
        end_nodes = [n for n, c in node_count.items() if c == 1]
        end_nodes.sort()
        if len(end_nodes) >= 2:
            left_nodes.append(end_nodes[0])
            right_nodes.append(end_nodes[-1])
        elif len(end_nodes) == 1:
            left_nodes.append(end_nodes[0])

    return left_nodes, right_nodes


# ====================================================================
# Process
# ====================================================================


class StrandBendingOscillationProcess(
    BatchProcess[StrandBendingOscillationConfig, StrandBendingOscillationResult],
):
    """7本撚線曲げ揺動 Process.

    パイプライン:
    1. StrandMeshProcess でメッシュ生成
    2. 端部参照点ノードを追加
    3. MPCEliminationProcess で端部剛体結合
    4. ULCRBeamAssemblerProcess でアセンブラ構築
    5. 曲げ処方変位を境界条件に設定
    6. ContactFrictionProcess で求解
    """

    meta = ProcessMeta(
        name="StrandBendingOscillation",
        module="batch",
        version="1.0.0",
        document_path="docs/strand_bending_oscillation.md",
    )
    uses = [
        StrandMeshProcess,
        MPCEliminationProcess,
        ULCRBeamAssemblerProcess,
        ContactFrictionProcess,
    ]

    def process(  # noqa: C901, PLR0912, PLR0915
        self,
        input_data: StrandBendingOscillationConfig,
    ) -> StrandBendingOscillationResult:
        """撚線曲げ揺動を実行."""
        cfg = input_data

        # ── 1. メッシュ生成 ──
        mesh_result = StrandMeshProcess().process(
            StrandMeshConfig(
                n_strands=cfg.n_strands,
                wire_radius=cfg.wire_radius,
                pitch_length=cfg.pitch_length,
                gap=cfg.gap,
                n_elements_per_pitch=cfg.n_elements_per_pitch,
                n_pitches=cfg.n_pitches,
            )
        )
        mesh = mesh_result.mesh
        strand_coords = mesh.node_coords
        strand_conn = mesh.connectivity
        n_strand_nodes = len(strand_coords)

        # ── 2. 端部節点の収集 ──
        left_nodes, right_nodes = _collect_end_nodes(strand_conn, cfg.n_strands, mesh.strand_ids)

        if cfg.free_end_mode:
            return self._process_free_end(
                cfg, mesh, strand_coords, strand_conn, n_strand_nodes, left_nodes, right_nodes
            )

        # ── 3. 参照点ノードの追加（MPCモード） ──
        # 左端参照点 = 左端節点群の重心
        left_coords = strand_coords[left_nodes]
        left_ref_coord = np.mean(left_coords, axis=0)
        # 右端参照点 = 右端節点群の重心
        right_coords = strand_coords[right_nodes]
        right_ref_coord = np.mean(right_coords, axis=0)

        # 拡張座標: 撚線ノード + 2参照点
        n_ref_nodes = 2
        ref_left_node = n_strand_nodes
        ref_right_node = n_strand_nodes + 1
        extended_coords = np.vstack(
            [strand_coords, left_ref_coord.reshape(1, 3), right_ref_coord.reshape(1, 3)]
        )
        n_total_nodes = len(extended_coords)
        ndof = n_total_nodes * 6

        # 拡張メッシュ（参照点はアセンブラに含めないが、座標系に含める）
        extended_mesh = MeshData(
            node_coords=extended_coords,
            connectivity=strand_conn,
            radii=mesh.radii,
            n_strands=cfg.n_strands,
            strand_ids=mesh.strand_ids,
        )

        # ── 4. MPC構築 ──
        mpc_groups = []
        # 左端MPC
        if left_nodes:
            mpc_groups.append(
                MPCGroup(
                    master_node=ref_left_node,
                    slave_nodes=np.array(left_nodes, dtype=int),
                    slave_coords=strand_coords[left_nodes],
                    master_coord=left_ref_coord,
                )
            )
        # 右端MPC
        if right_nodes:
            mpc_groups.append(
                MPCGroup(
                    master_node=ref_right_node,
                    slave_nodes=np.array(right_nodes, dtype=int),
                    slave_coords=strand_coords[right_nodes],
                    master_coord=right_ref_coord,
                )
            )

        mpc_result = MPCEliminationProcess().process(
            MPCEliminationConfig(
                mpc_groups=mpc_groups,
                ndof_total=ndof,
                ndof_per_node=6,
            )
        )

        # ── 5. アセンブラ構築 ──
        sec = _circle_section(cfg.wire_radius * 2.0, cfg.nu)
        G = cfg.E / (2.0 * (1.0 + cfg.nu))

        beam_result = ULCRBeamAssemblerProcess().process(
            ULCRBeamAssemblerInput(
                node_coords=strand_coords,  # 撚線ノードのみ
                connectivity=strand_conn,
                E=cfg.E,
                G=G,
                A=sec["A"],
                Iy=sec["Iy"],
                Iz=sec["Iz"],
                J=sec["J"],
                kappa_y=sec["kappa"],
                kappa_z=sec["kappa"],
            )
        )
        assembler = beam_result.assembler
        ndof_beam = n_strand_nodes * 6

        # 剛性/内力のラッパー: 参照点DOFを含む拡張系にゼロパディング
        def _assemble_tangent_extended(u: np.ndarray) -> sp.csr_matrix:
            u_beam = u[:ndof_beam]
            K_beam = assembler.assemble_tangent(u_beam)
            # 拡張系: 参照点DOFの行列はゼロ
            K_ext = sp.lil_matrix((ndof, ndof))
            K_ext[:ndof_beam, :ndof_beam] = K_beam
            return K_ext.tocsr()

        def _assemble_internal_force_extended(u: np.ndarray) -> np.ndarray:
            u_beam = u[:ndof_beam]
            f_beam = assembler.assemble_internal_force(u_beam)
            f_ext_padded = np.zeros(ndof)
            f_ext_padded[:ndof_beam] = f_beam
            return f_ext_padded

        # 質量行列
        M_beam = assembler.assemble_mass(cfg.rho, lumped=cfg.lumped_mass)
        M_ext = sp.lil_matrix((ndof, ndof))
        M_ext[:ndof_beam, :ndof_beam] = M_beam

        # MPC参照点の質量補強（status-278: 回転慣性NR収束不良修正）
        # lumped質量行列では参照点ノードの質量がゼロ。MPC変換 T^T M T で
        # slave ノードの質量が参照点に集約されるが、回転慣性が ~10^-7 と
        # 極めて小さく、effective_stiffness の回転対角項がほぼゼロになる。
        # → NRの回転DOF更新が発散し、残差が収束しない。
        # 対策: 参照点に slave ノードの質量を直接加算し、
        # 平行軸定理で回転慣性を計算する。
        for ref_node, end_nodes in [(ref_left_node, left_nodes), (ref_right_node, right_nodes)]:
            ref_coord = (
                strand_coords[ref_node]
                if ref_node < len(strand_coords)
                else np.mean(strand_coords[end_nodes], axis=0)
            )
            m_total = 0.0  # 並進質量の和
            I_xx, I_yy, I_zz = 0.0, 0.0, 0.0  # 回転慣性（平行軸定理）
            for en in end_nodes:
                m_n = float(M_beam[en * 6, en * 6])  # ノード並進質量
                m_total += m_n
                # 平行軸定理: I += m * r²
                if en < len(strand_coords):
                    dr = strand_coords[en] - ref_coord
                    I_xx += m_n * (dr[1] ** 2 + dr[2] ** 2)
                    I_yy += m_n * (dr[0] ** 2 + dr[2] ** 2)
                    I_zz += m_n * (dr[0] ** 2 + dr[1] ** 2)
                # ノード自身の回転慣性も加算
                for d in range(3):
                    rot_dof = en * 6 + 3 + d
                    if rot_dof < M_beam.shape[0]:
                        i_n = float(M_beam[rot_dof, rot_dof])
                        if d == 0:
                            I_xx += i_n
                        elif d == 1:
                            I_yy += i_n
                        else:
                            I_zz += i_n
            # 参照点に質量を設定
            for d in range(3):
                M_ext[ref_node * 6 + d, ref_node * 6 + d] = m_total
            M_ext[ref_node * 6 + 3, ref_node * 6 + 3] = I_xx
            M_ext[ref_node * 6 + 4, ref_node * 6 + 4] = I_yy
            M_ext[ref_node * 6 + 5, ref_node * 6 + 5] = I_zz

        M_ext = M_ext.tocsr()

        # ── 6. 境界条件 ──
        # 左端参照点: 全DOF固定
        fixed_dofs = set()
        for k in range(6):
            fixed_dofs.add(ref_left_node * 6 + k)

        # 右端参照点: xyz固定 + θ_x,θ_y固定、θ_z のみ処方変位
        # status-278: θ_x,θ_y を自由にすると、lumped質量行列の回転慣性が
        # 極めて小さく（~10^-7）、動的残差がNRで収束しない。
        # 撚線曲げ試験では θ_x,θ_y は物理的に不要（曲げ面内回転のみ）。
        for k in range(5):  # x,y,z,θ_x,θ_y を固定
            fixed_dofs.add(ref_right_node * 6 + k)

        # 曲げ角度 = κ * L
        strand_length = cfg.pitch_length * cfg.n_pitches
        bending_angle = cfg.bending_curvature * strand_length

        # 処方変位: 右端参照点の θ_z （曲げ回転）
        # 揺動: 0 → +θ → 0 → -θ → 0 を n_cycles 回
        prescribed_dof = ref_right_node * 6 + 5  # θ_z
        prescribed_dofs = np.array([prescribed_dof], dtype=int)
        prescribed_values = np.array([bending_angle])

        fixed_dofs_arr = np.array(sorted(fixed_dofs), dtype=int)

        # 時間パラメータ: 揺動周期を動的解析のt_totalに設定
        # 梁の固有振動数から概算
        sec_Iy = sec["Iy"]
        sec_A = sec["A"]
        f1 = (math.pi / (2.0 * strand_length**2)) * math.sqrt(
            cfg.E * sec_Iy * cfg.n_strands / (cfg.rho * sec_A * cfg.n_strands)
        )
        T1 = 1.0 / f1 if f1 > 1e-30 else 1.0
        # 揺動周期 = 少なくとも固有周期の10倍（準静的挙動）
        t_cycle = max(10.0 * T1, 1.0)
        t_total = t_cycle * cfg.n_cycles

        dt_initial = t_total / (cfg.n_increments_per_cycle * cfg.n_cycles)

        boundary = BoundaryData(
            fixed_dofs=fixed_dofs_arr,
            prescribed_dofs=prescribed_dofs,
            prescribed_values=prescribed_values,
            f_ext_total=np.zeros(ndof),
            mpc_transform=mpc_result,
        )

        # ── 7. 接触設定 ──
        # smoothing_delta 自動推定: δ = 1000 / r_min（status-260: 5000→1000に変更）
        _smoothing_delta = (
            cfg.smoothing_delta if cfg.smoothing_delta > 0.0 else 1000.0 / cfg.wire_radius
        )
        contact_config = _ContactConfigInput(
            beam_E=cfg.E,
            beam_I=sec_Iy,
            mu=cfg.mu,
            adaptive_timestepping=True,
            dt_min_fraction=dt_initial / (t_total * 64.0),
            dt_max_fraction=dt_initial / t_total,
            exclude_same_strand=cfg.exclude_same_strand,
            smoothing_delta=_smoothing_delta,
            huber_delta_h=cfg.huber_delta_h,
        )
        manager = _ContactManagerInput(config=contact_config)
        contact_setup = ContactSetupData(
            manager=manager,
            k_pen=cfg.k_pen,
            mu=cfg.mu,
        )

        # ── 8. ソルバー実行 ──
        # ULアセンブラを拡張DOF系にラップ（参照点DOFのゼロパディング）
        extended_assembler = _ExtendedULAssemblerWrapper(assembler, ndof_beam, ndof)

        # チェックポイント復元（status-278, status-279で途中再開対応）
        _u0 = None
        _vel0 = None
        _acc0 = None
        _frac_start = 0.0
        if cfg.resume_checkpoint:
            import pickle as _pickle

            with open(cfg.resume_checkpoint, "rb") as _f:
                _ckpt = _pickle.load(_f)
            _u0 = _ckpt["state"].u.copy()
            _vel0 = _ckpt["time_vel"]
            _acc0 = _ckpt["time_acc"]
            _frac_start = _ckpt["load_frac"]
            # ULアセンブラの累積変位を復元
            if hasattr(extended_assembler._asm, "_u_total_accum"):
                extended_assembler._asm._u_total_accum[:] = _u0[:ndof_beam]
            print(f"  [RESUME] frac={_frac_start:.4f}, ||u||={np.linalg.norm(_u0):.4e}")

        solver_input = ContactFrictionInputData(
            mesh=extended_mesh,
            boundary=boundary,
            contact=contact_setup,
            callbacks=AssembleCallbacks(
                assemble_tangent=_assemble_tangent_extended,
                assemble_internal_force=_assemble_internal_force_extended,
                ul_assembler=extended_assembler,
            ),
            u0=_u0,
            mass_matrix=M_ext,
            dt_physical=t_total,
            rho_inf=cfg.rho_inf,
            velocity=_vel0,
            acceleration=_acc0,
            max_nr_attempts=cfg.max_nr_attempts,
            tol_force=cfg.tol_force,
            max_increments=cfg.max_increments,
            tangent_fd_diagnostic=cfg.tangent_fd_diagnostic,
            du_norm_cap=cfg.du_norm_cap,
            load_frac_start=_frac_start,
        )
        solver = ContactFrictionProcess()
        solver_result = solver.process(solver_input)

        return StrandBendingOscillationResult(
            solver_result=solver_result,
            mesh=extended_mesh,
            n_ref_nodes=n_ref_nodes,
            n_strand_nodes=n_strand_nodes,
            total_ndof=ndof,
            bending_angle=bending_angle,
        )

    def _process_free_end(  # noqa: PLR0912, PLR0915
        self,
        cfg: StrandBendingOscillationConfig,
        mesh: MeshData,
        strand_coords: np.ndarray,
        strand_conn: np.ndarray,
        n_strand_nodes: int,
        left_nodes: list[int],
        right_nodes: list[int],
    ) -> StrandBendingOscillationResult:
        """MPC不使用・端部直接処方モードで撚線曲げ揺動を実行.

        status-280: MPC端部剛体結合の代わりに、各素線端部ノードの
        θ_z を直接処方し、並進DOFは自由にする。
        - 左端: 全素線端部ノードの全6DOF固定
        - 右端: θ_z処方、θ_x/θ_y固定、u_x/u_y/u_z自由
        - 参照点ノード不要 → 拡張系不要 → MPC不要
        """
        ndof = n_strand_nodes * 6

        # ── アセンブラ構築 ──
        sec = _circle_section(cfg.wire_radius * 2.0, cfg.nu)
        G = cfg.E / (2.0 * (1.0 + cfg.nu))

        beam_result = ULCRBeamAssemblerProcess().process(
            ULCRBeamAssemblerInput(
                node_coords=strand_coords,
                connectivity=strand_conn,
                E=cfg.E,
                G=G,
                A=sec["A"],
                Iy=sec["Iy"],
                Iz=sec["Iz"],
                J=sec["J"],
                kappa_y=sec["kappa"],
                kappa_z=sec["kappa"],
            )
        )
        assembler = beam_result.assembler

        # 質量行列（直接使用、拡張不要）
        M = assembler.assemble_mass(cfg.rho, lumped=cfg.lumped_mass)

        # ── 境界条件 ──
        fixed_dofs: set[int] = set()
        prescribed_dofs_list: list[int] = []
        prescribed_values_list: list[float] = []

        # 左端: 全素線端部ノードの全6DOF固定
        for n in left_nodes:
            for k in range(6):
                fixed_dofs.add(n * 6 + k)

        # 右端: θ_z処方, θ_x/θ_y固定, u_x/u_y/u_z自由
        strand_length = cfg.pitch_length * cfg.n_pitches
        bending_angle = cfg.bending_curvature * strand_length

        for n in right_nodes:
            # θ_x, θ_y を固定（曲げ面内回転のみ許可）
            fixed_dofs.add(n * 6 + 3)  # θ_x
            fixed_dofs.add(n * 6 + 4)  # θ_y
            # θ_z を処方
            prescribed_dofs_list.append(n * 6 + 5)
            prescribed_values_list.append(bending_angle)
            # u_x, u_y, u_z は自由 → 断面が自然に変位

        fixed_dofs_arr = np.array(sorted(fixed_dofs), dtype=int)
        prescribed_dofs_arr = np.array(prescribed_dofs_list, dtype=int)
        prescribed_values_arr = np.array(prescribed_values_list)

        # ── 時間パラメータ ──
        sec_Iy = sec["Iy"]
        sec_A = sec["A"]
        f1 = (math.pi / (2.0 * strand_length**2)) * math.sqrt(
            cfg.E * sec_Iy * cfg.n_strands / (cfg.rho * sec_A * cfg.n_strands)
        )
        T1 = 1.0 / f1 if f1 > 1e-30 else 1.0
        t_cycle = max(10.0 * T1, 1.0)
        t_total = t_cycle * cfg.n_cycles
        dt_initial = t_total / (cfg.n_increments_per_cycle * cfg.n_cycles)

        boundary = BoundaryData(
            fixed_dofs=fixed_dofs_arr,
            prescribed_dofs=prescribed_dofs_arr,
            prescribed_values=prescribed_values_arr,
            f_ext_total=np.zeros(ndof),
            mpc_transform=None,  # MPC不使用
        )

        # ── 接触設定 ──
        _smoothing_delta = (
            cfg.smoothing_delta if cfg.smoothing_delta > 0.0 else 1000.0 / cfg.wire_radius
        )
        contact_config = _ContactConfigInput(
            beam_E=cfg.E,
            beam_I=sec_Iy,
            mu=cfg.mu,
            adaptive_timestepping=True,
            dt_min_fraction=dt_initial / (t_total * 64.0),
            dt_max_fraction=dt_initial / t_total,
            exclude_same_strand=cfg.exclude_same_strand,
            smoothing_delta=_smoothing_delta,
            huber_delta_h=cfg.huber_delta_h,
        )
        manager = _ContactManagerInput(config=contact_config)
        contact_setup = ContactSetupData(
            manager=manager,
            k_pen=cfg.k_pen,
            mu=cfg.mu,
        )

        # ── ソルバー実行 ──
        solver_input = ContactFrictionInputData(
            mesh=mesh,  # 元のメッシュ（参照点なし）
            boundary=boundary,
            contact=contact_setup,
            callbacks=AssembleCallbacks(
                assemble_tangent=assembler.assemble_tangent,
                assemble_internal_force=assembler.assemble_internal_force,
                ul_assembler=assembler,
            ),
            mass_matrix=M,
            dt_physical=t_total,
            rho_inf=cfg.rho_inf,
            max_nr_attempts=cfg.max_nr_attempts,
            tol_force=cfg.tol_force,
            max_increments=cfg.max_increments,
            tangent_fd_diagnostic=cfg.tangent_fd_diagnostic,
            du_norm_cap=cfg.du_norm_cap,
        )
        solver_result = ContactFrictionProcess().process(solver_input)

        return StrandBendingOscillationResult(
            solver_result=solver_result,
            mesh=mesh,
            n_ref_nodes=0,
            n_strand_nodes=n_strand_nodes,
            total_ndof=ndof,
            bending_angle=bending_angle,
        )
