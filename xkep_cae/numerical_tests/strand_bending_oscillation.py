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
import scipy.sparse.linalg as spla

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

    status-283: update_reference / coords_ref を拡張系でサポート。
    """

    def __init__(
        self,
        assembler: object,
        ndof_beam: int,
        ndof_total: int,
        ref_node_coords: np.ndarray | None = None,
    ) -> None:
        self._asm = assembler
        self._ndof_beam = ndof_beam
        self._ndof_total = ndof_total
        # 参照点ノードの座標（梁ノード以降の追加ノード）
        self._ref_node_coords = ref_node_coords.copy() if ref_node_coords is not None else None
        self._ref_node_coords_ckpt = ref_node_coords.copy() if ref_node_coords is not None else None

    @property
    def u_total_accum(self) -> np.ndarray:
        u = np.zeros(self._ndof_total)
        u[: self._ndof_beam] = self._asm.u_total_accum
        return u

    @property
    def coords_ref(self) -> np.ndarray:
        beam_coords = self._asm.coords_ref
        if self._ref_node_coords is not None:
            return np.vstack([beam_coords, self._ref_node_coords])
        return beam_coords

    def checkpoint(self) -> None:
        self._asm.checkpoint()
        if self._ref_node_coords is not None:
            self._ref_node_coords_ckpt = self._ref_node_coords.copy()

    def rollback(self) -> None:
        self._asm.rollback()
        if self._ref_node_coords_ckpt is not None:
            self._ref_node_coords = self._ref_node_coords_ckpt.copy()

    def update_reference(self, u_incr: np.ndarray) -> None:
        """参照配置を増分変位で更新する."""
        self._asm.update_reference(u_incr[: self._ndof_beam])
        # 参照点ノードの座標も並進変位で更新
        if self._ref_node_coords is not None:
            n_beam_nodes = self._ndof_beam // 6
            n_ref = len(self._ref_node_coords)
            for i in range(n_ref):
                node_idx = n_beam_nodes + i
                self._ref_node_coords[i] += u_incr[node_idx * 6 : node_idx * 6 + 3]


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
    exclude_end_elements: int = (
        0  # >0: 各素線の端部N要素を接触候補から除外（status-296: MPC安定化）
    )
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
    # contact_enabled: Falseで接触計算を無効化（radii=0で検出スキップ）
    contact_enabled: bool = True
    # loading_mode: "rotation"=端部θ_y処方（従来）, "moment"=端部M_y荷重（status-281）
    # moment モードでは EI*θ/L の曲げモーメントを各素線右端に印加。
    # エラスティカ理論により大変形でも M-θ 線形 → NR安定。
    loading_mode: str = "rotation"
    # Hertz型非線形ペナルティ（status-285）
    # 1.0=線形ペナルティ（従来）, 1.5=Hertz型（p_n ∝ δ^1.5）
    penalty_exponent: float = 1.0
    # チェックポイント保存（status-286: 曲げ完了時点のpickle保存）
    checkpoint_path: str = ""  # 非空で曲げ完了時にpickle保存
    # 揺動サイクル（status-286: sin波処方変位）
    # n_oscillation_cycles > 0 で揺動フェーズを有効化。
    # 曲げフェーズ（frac=0→1.0）完了後、揺動フェーズ（frac=0→1.0）を実行。
    # 揺動の処方変位: θ(frac) = θ_max * sin(2π * n_oscillation_cycles * frac)
    # resume_checkpoint が設定されている場合、曲げスキップで揺動のみ実行。
    n_oscillation_cycles: int = 0  # 0=曲げのみ, >0=曲げ後に揺動
    # 揺動振幅 [mm]（status-299: 先端横変位揺動）
    # >0: 揺動フェーズでθ_y回転処方の代わりに先端u_z横変位±amplitude[mm]で処方。
    # 曲げ完了位置を中心にsin波振動: u_z(frac) = u_z_bend + amplitude * sin(2π*n_osc*frac)
    # 0: 従来のθ_y回転cos波揺動（status-286互換）
    oscillation_amplitude: float = 0.0  # [mm], 0=θ回転揺動
    # 被膜パラメータ（status-301: 1000本モデルの縮小モデル用）
    coating_stiffness: float = 0.0  # 被膜剛性 [N/mm], 0=被膜なし
    coating_damping: float = 0.0  # 被膜減衰 [N·s/mm]
    coating_mu: float = 0.0  # 被膜摩擦係数
    coating_k_t_ratio: float = 0.5  # 接線剛性比
    coating_thickness: float = 0.0  # 被膜厚さ [mm], >0: core_radius = wire_radius - thickness
    coating_barrier: bool = True  # バリア関数被膜モデル（status-303）


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


def _collect_adjacent_nodes(
    connectivity: np.ndarray,
    strand_ids: np.ndarray,
    end_nodes: list[int],
) -> list[int]:
    """各端部節点の隣接ノード（1要素内側）を返す.

    力カップル方式でモーメント荷重を生成するために使用。
    """
    adj_nodes = []
    for end_node in end_nodes:
        found = False
        for _i, e in enumerate(connectivity):
            if end_node in e:
                # 端部ノードと同じ要素の他方のノード
                other = int(e[1]) if int(e[0]) == end_node else int(e[0])
                adj_nodes.append(other)
                found = True
                break
        if not found:
            adj_nodes.append(end_node)  # fallback
    return adj_nodes


# ====================================================================
# 静的 Newton-Raphson ソルバー（接触なし問題用）
# ====================================================================


def _static_nr_solve(  # noqa: PLR0912, PLR0915
    assembler: object,
    ndof: int,
    fixed_dofs: np.ndarray,
    prescribed_dofs: np.ndarray,
    prescribed_values: np.ndarray,
    f_ext_total: np.ndarray,
    n_increments: int,
    max_nr: int = 50,
    tol: float = 1e-8,
    show_progress: bool = True,
) -> SolverResultData:
    """接触なし問題用の静的NRソルバー（UL増分形）.

    status-281: 動的ソルバー（Generalized-α）は慣性項による残差連成で
    ヘリカル複数素線の曲げ収束が困難。接触なし問題では純粋な
    静的NR法で直接求解する。

    UL（Updated Lagrangian）フォーマレーション:
    - 各収束ステップ後に参照配置を更新（update_reference）
    - NR反復中は増分変位 u_incr をアセンブラに渡す
    - 累積変位 u_total は出力用に追跡

    Returns:
        SolverResultData: ソルバー結果
    """
    import time

    t0 = time.time()
    load_history: list[float] = []
    n_cutbacks = 0
    total_attempts = 0

    # BC用マスク
    all_constrained = set(fixed_dofs.tolist()) | set(prescribed_dofs.tolist())
    free_mask = np.ones(ndof, dtype=bool)
    for d in all_constrained:
        free_mask[d] = False

    frac = 0.0
    frac_prev = 0.0  # 前回収束時のfrac
    dt_frac = 1.0 / n_increments
    dt_min = dt_frac / 64.0
    incr = 0

    while frac < 1.0 - 1e-12:
        dt_try = min(dt_frac, 1.0 - frac)
        frac_target = frac + dt_try

        # UL増分: 前回収束状態からの増分変位
        u_incr = np.zeros(ndof)
        converged = False

        # 処方変位の増分（前回収束からの差分）
        delta_presc = (frac_target - frac_prev) * prescribed_values
        u_incr[prescribed_dofs] = delta_presc
        f_ref = 1.0  # 初期値（att==0で更新される）

        for att in range(max_nr):
            total_attempts += 1
            f_int = assembler.assemble_internal_force(u_incr)
            R = f_int - f_ext_total * frac_target
            R[fixed_dofs] = 0.0
            R[prescribed_dofs] = 0.0

            res_norm = float(np.linalg.norm(R[free_mask]))
            # 参照ノルム: 外力があればその大きさ、なければ初回残差
            f_ext_norm = float(np.linalg.norm(f_ext_total * frac_target))
            if f_ext_norm > 1e-30:
                f_ref = f_ext_norm
            elif att == 0:
                f_ref = max(res_norm, 1.0)
            # else: f_ref は前回の値を維持

            if show_progress and att % 5 == 0:
                print(
                    f"  Static Incr {incr + 1} (frac={frac_target:.4f}), "
                    f"att {att}, ||R||/||f||={res_norm / f_ref:.3e}"
                )

            if res_norm / f_ref < tol:
                converged = True
                break
            if att > 3 and res_norm / f_ref > 1e6:
                break  # 発散

            K = assembler.assemble_tangent(u_incr)
            # BC適用: 拘束DOFの行/列をゼロ化、対角=1
            K_lil = K.tolil()
            for d in all_constrained:
                K_lil[d, :] = 0.0
                K_lil[:, d] = 0.0
                K_lil[d, d] = 1.0
            K_csr = K_lil.tocsr()

            du = spla.spsolve(K_csr, -R)
            u_incr += du

        if converged:
            # UL参照配置を更新
            assembler.update_reference(u_incr)
            assembler.checkpoint()
            frac_prev = frac_target
            incr += 1
            frac = frac_target
            load_history.append(frac)
            # dt成長
            dt_frac = min(dt_frac * 1.5, 1.0 / n_increments)
        else:
            # カットバック
            n_cutbacks += 1
            assembler.rollback()
            dt_frac *= 0.5
            if dt_frac < dt_min:
                if show_progress:
                    print(f"  Static solver: dt_min到達 (frac={frac:.4f})")
                break

    elapsed = time.time() - t0
    # 累積変位を取得
    u_total = assembler.u_total_accum
    return SolverResultData(
        u=u_total,
        converged=frac >= 1.0 - 1e-10,
        n_increments=incr,
        total_attempts=total_attempts,
        load_history=tuple(load_history),
        elapsed_seconds=elapsed,
        n_cutbacks=n_cutbacks,
    )


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
        # 接触無効時: radii=0 で接触ペア検出をスキップ（status-280）
        if not cfg.contact_enabled:
            mesh = MeshData(
                node_coords=mesh.node_coords,
                connectivity=mesh.connectivity,
                radii=0.0,
                n_strands=mesh.n_strands,
                strand_ids=mesh.strand_ids,
            )
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

        # 右端参照点: u_y固定 + θ_x,θ_z固定、θ_y のみ処方変位
        # status-280: x-z面カンチレバー曲げ。
        # u_x, u_z は自由（梁端の横移動+軸短縮を許容）。
        # u_y を固定（面外変位を拘束）。
        for k in [1, 3, 5]:  # u_y, θ_x, θ_z を固定
            fixed_dofs.add(ref_right_node * 6 + k)

        # 曲げ角度 = κ * L
        strand_length = cfg.pitch_length * cfg.n_pitches
        bending_angle = cfg.bending_curvature * strand_length

        # 処方変位: 右端参照点の θ_y （x-z面曲げ回転）
        prescribed_dof = ref_right_node * 6 + 4  # θ_y
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
            mpc_groups=mpc_groups,  # UL更新時のT再構築用（status-283）
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
            exclude_end_elements=cfg.exclude_end_elements,
            smoothing_delta=_smoothing_delta,
            huber_delta_h=cfg.huber_delta_h,
            coating_stiffness=cfg.coating_stiffness,
            coating_damping=cfg.coating_damping,
            coating_mu=cfg.coating_mu,
            coating_k_t_ratio=cfg.coating_k_t_ratio,
            coating_thickness=cfg.coating_thickness,
            coating_barrier=cfg.coating_barrier,
        )
        manager = _ContactManagerInput(config=contact_config)
        contact_setup = ContactSetupData(
            manager=manager,
            k_pen=cfg.k_pen,
            mu=cfg.mu,
        )

        # ── 8. ソルバー実行 ──
        # ULアセンブラを拡張DOF系にラップ（参照点DOFのゼロパディング）
        # 参照点ノード座標（梁ノード以降の追加ノード）
        _ref_coords = extended_coords[n_strand_nodes:]
        extended_assembler = _ExtendedULAssemblerWrapper(
            assembler, ndof_beam, ndof, ref_node_coords=_ref_coords
        )

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
            penalty_exponent=cfg.penalty_exponent,
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
        f_ext = np.zeros(ndof)

        # 左端: 全素線端部ノードの全6DOF固定
        for n in left_nodes:
            for k in range(6):
                fixed_dofs.add(n * 6 + k)

        strand_length = cfg.pitch_length * cfg.n_pitches
        bending_angle = cfg.bending_curvature * strand_length

        if cfg.loading_mode == "moment":
            # status-281: 力カップル方式モーメント荷重
            # エラスティカ理論: M = EI * θ / L（大変形でも M-θ 線形）
            # NR収束判定は並進残差のみ使用するため、純モーメントは偽収束する。
            # 対策: 端部2ノードに逆向き力 (F, -F) で力カップルを構成。
            # M_y = F_x * Δz → F_x = M_y / Δz
            sec_Iy = sec["Iy"]
            m_target = cfg.E * sec_Iy * bending_angle / strand_length
            adj_nodes = _collect_adjacent_nodes(strand_conn, mesh.strand_ids, right_nodes)
            for n_end, n_adj in zip(right_nodes, adj_nodes, strict=True):
                # 端部ノードと隣接ノードの距離（z方向投影）
                dz = abs(strand_coords[n_end][2] - strand_coords[n_adj][2])
                if dz < 1e-10:
                    dz = np.linalg.norm(strand_coords[n_end] - strand_coords[n_adj])
                f_couple = m_target / dz
                # x方向の力カップルで y軸周りモーメントを生成
                # M_y = F_x * Δz (右手系: +x力 × +z位置 = +y回転)
                f_ext[n_end * 6 + 0] = f_couple  # +F_x at tip
                f_ext[n_adj * 6 + 0] = -f_couple  # -F_x at adjacent
                # u_y を固定（面外変位を拘束）
                fixed_dofs.add(n_end * 6 + 1)
                # u_x, u_z, θ_x, θ_y, θ_z は全て自由
        else:
            # 従来: θ_y処方（x-z面曲げ）, θ_x/θ_z自由, u自由
            for n in right_nodes:
                prescribed_dofs_list.append(n * 6 + 4)
                prescribed_values_list.append(bending_angle)

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

        # status-299: 曲げ+揺動統合モード
        # n_oscillation_cycles > 0 の場合、曲げ+揺動を1回のソルバーで実行。
        # frac_bend = 曲げフェーズが全体のどこまでか
        _n_osc = cfg.n_oscillation_cycles
        # u_z直接揺動（oscillation_amplitude > 0）は prescribed_dofs が変わるため
        # 2フェーズ方式。θ_y揺動（amplitude=0）のみ統合モード。
        # 統合モード: 曲げ+揺動を1回のソルバーで実行。
        # 2フェーズ方式はUL参照配置不整合（CR梁のf_int=0問題）で使用不可。
        _combined_mode = _n_osc > 0 and cfg.loading_mode != "moment"
        if _combined_mode:
            _frac_bend = cfg.n_cycles / (cfg.n_cycles + _n_osc)
            t_total = t_cycle * (cfg.n_cycles + _n_osc)
            # 揺動振幅: oscillation_amplitude > 0 なら先端変位→θ変換
            # 先端変位 δ ≈ R * Δθ where R = 1/κ = strand_length / (κ*strand_length)
            if cfg.oscillation_amplitude > 0.0:
                _R_bend = strand_length / bending_angle if bending_angle > 1e-10 else strand_length
                _theta_osc_amp = cfg.oscillation_amplitude / _R_bend
            else:
                _theta_osc_amp = bending_angle  # 従来: 全曲げ角度で揺動

            def _combined_prescribed_func(frac: float) -> np.ndarray:
                if frac <= _frac_bend:
                    # 曲げフェーズ: θ = bending_angle * (frac / frac_bend)
                    _theta = bending_angle * (frac / _frac_bend)
                else:
                    # 揺動フェーズ: 1-cos 波形でC1連続（transition微分=0）
                    # sin(2πnt) は transition で dθ/dt ≠ 0 → 不連続。
                    # 代わりに (1-cos(2πnt))/2 を使い、最初の半周期で振幅まで
                    # 滑らかに立ち上がる。
                    _osc_frac = (frac - _frac_bend) / (1.0 - _frac_bend)
                    _phase = 2.0 * math.pi * _n_osc * _osc_frac
                    _delta_theta = _theta_osc_amp * math.sin(_phase)
                    # 最初の1/4周期にランプ適用（C1連続化）
                    _ramp_end = 0.25 / _n_osc  # 最初のn_oscの1/4周期
                    if _osc_frac < _ramp_end:
                        # sin波にハーフcos窓: 0→1の滑らかなランプ
                        _w = 0.5 * (1.0 - math.cos(math.pi * _osc_frac / _ramp_end))
                        _delta_theta *= _w
                    _theta = bending_angle + _delta_theta
                return np.full(len(prescribed_dofs_arr), _theta)

            print(
                f"  統合モード: 曲げ(frac<{_frac_bend:.3f}) + "
                f"θ揺動±{math.degrees(_theta_osc_amp):.1f}°×{_n_osc}cyc"
            )
            if cfg.oscillation_amplitude > 0.0:
                print(
                    f"  先端変位±{cfg.oscillation_amplitude:.1f}mm → "
                    f"θ振幅±{math.degrees(_theta_osc_amp):.1f}°"
                )
        else:
            _frac_bend = 1.0
            _combined_prescribed_func = None
            t_total = t_cycle * cfg.n_cycles

        _total_cycles = (cfg.n_cycles + _n_osc) if _combined_mode else cfg.n_cycles
        dt_initial = t_total / (cfg.n_increments_per_cycle * _total_cycles)

        boundary = BoundaryData(
            fixed_dofs=fixed_dofs_arr,
            prescribed_dofs=prescribed_dofs_arr,
            prescribed_values=prescribed_values_arr,
            f_ext_total=f_ext,
            mpc_transform=None,  # MPC不使用
            prescribed_func=_combined_prescribed_func,
        )

        # ── 接触設定 ──
        _smoothing_delta = (
            cfg.smoothing_delta if cfg.smoothing_delta > 0.0 else 1000.0 / cfg.wire_radius
        )
        _cutback_depth = 256.0 if _combined_mode else 64.0
        contact_config = _ContactConfigInput(
            beam_E=cfg.E,
            beam_I=sec_Iy,
            mu=cfg.mu,
            adaptive_timestepping=True,
            dt_min_fraction=dt_initial / (t_total * _cutback_depth),
            dt_max_fraction=dt_initial / t_total,
            exclude_same_strand=cfg.exclude_same_strand,
            exclude_end_elements=cfg.exclude_end_elements,
            smoothing_delta=_smoothing_delta,
            huber_delta_h=cfg.huber_delta_h,
            coating_stiffness=cfg.coating_stiffness,
            coating_damping=cfg.coating_damping,
            coating_mu=cfg.coating_mu,
            coating_k_t_ratio=cfg.coating_k_t_ratio,
            coating_thickness=cfg.coating_thickness,
            coating_barrier=cfg.coating_barrier,
        )
        manager = _ContactManagerInput(config=contact_config)
        contact_setup = ContactSetupData(
            manager=manager,
            k_pen=cfg.k_pen,
            mu=cfg.mu,
        )

        # ── ソルバー実行: 曲げフェーズ ──
        _callbacks = AssembleCallbacks(
            assemble_tangent=assembler.assemble_tangent,
            assemble_internal_force=assembler.assemble_internal_force,
            ul_assembler=assembler,
        )

        # チェックポイント復元時は曲げフェーズをスキップ
        if cfg.resume_checkpoint and cfg.n_oscillation_cycles > 0:
            import pickle as _pickle

            with open(cfg.resume_checkpoint, "rb") as _f:
                _ckpt = _pickle.load(_f)
            _u_bend = _ckpt["state"].u.copy()
            _vel_bend = _ckpt["time_vel"]
            _acc_bend = _ckpt["time_acc"]
            # ULアセンブラの完全状態復元（自工程保証:
            # coords_ref + R_ref + _u_total_accum の3点セットで参照配置を正確復元）
            if "ul_coords_ref" in _ckpt:
                if hasattr(assembler, "coords_ref"):
                    assembler.coords_ref[:] = _ckpt["ul_coords_ref"]
                if hasattr(assembler, "R_ref"):
                    assembler.R_ref[:] = _ckpt["ul_R_ref"]
                if hasattr(assembler, "_u_total_accum"):
                    assembler._u_total_accum[:] = _ckpt["ul_u_total_accum"]
                print("  [RESUME] ULアセンブラ完全復元: coords_ref + R_ref + u_accum")
            elif "ul_u_total_accum" in _ckpt:
                # 後方互換: coords_ref/R_ref なしの旧checkpoint
                _u_accum = _ckpt["ul_u_total_accum"]
                if hasattr(assembler, "_u_total_accum"):
                    assembler._u_total_accum[:] = _u_accum
                print(
                    f"  [RESUME] UL累積変位のみ復元（旧形式）: "
                    f"||u_accum||={np.linalg.norm(_u_accum):.4e}"
                )
            elif hasattr(assembler, "_u_total_accum"):
                assembler._u_total_accum[:] = _u_bend
            # 接触マネージャ状態の復元（自工程保証:
            # 各インクリメント完了時の保存状態で次のインクリメントをクリーンに開始）
            if "manager_pairs" in _ckpt:
                _restored_pairs = _ckpt["manager_pairs"]
                _restored_config = _ckpt.get("manager_config", contact_config)
                _restored_conn = _ckpt.get("connectivity", None)
                manager = _ContactManagerInput(
                    pairs=_restored_pairs,
                    config=_restored_config,
                    connectivity=_restored_conn,
                )
                contact_setup = ContactSetupData(
                    manager=manager,
                    k_pen=cfg.k_pen,
                    mu=cfg.mu,
                )
                _n_active = sum(
                    1 for p in _restored_pairs if hasattr(p, "state") and p.state.p_n > 0
                )
                print(
                    f"  [RESUME] 接触マネージャ復元: "
                    f"{len(_restored_pairs)} pairs ({_n_active} active)"
                )
            print(f"  [RESUME] 曲げcheckpointロード: ||u||={np.linalg.norm(_u_bend):.4e}")
            solver_result_bend = None
        else:
            # 曲げフェーズ実行
            solver_input = ContactFrictionInputData(
                mesh=mesh,
                boundary=boundary,
                contact=contact_setup,
                callbacks=_callbacks,
                mass_matrix=M,
                dt_physical=t_total,
                rho_inf=cfg.rho_inf,
                max_nr_attempts=cfg.max_nr_attempts,
                tol_force=cfg.tol_force,
                max_increments=cfg.max_increments,
                tangent_fd_diagnostic=cfg.tangent_fd_diagnostic,
                du_norm_cap=cfg.du_norm_cap,
                penalty_exponent=cfg.penalty_exponent,
                checkpoint_path=cfg.checkpoint_path,
                checkpoint_frac=0.99,
            )
            solver_result_bend = ContactFrictionProcess().process(solver_input)
            _u_bend = solver_result_bend.u
            # status-299: 速度・加速度を揺動フェーズに引き継ぎ（慣性力不整合防止）
            _vel_bend = solver_result_bend.final_velocity
            _acc_bend = solver_result_bend.final_acceleration
            _frac_bend = (
                solver_result_bend.load_history[-1] if solver_result_bend.load_history else 0.0
            )
            print(
                f"  曲げフェーズ完了: frac={_frac_bend:.4f}, "
                f"incr={solver_result_bend.n_increments}, "
                f"cutback={solver_result_bend.n_cutbacks}"
            )

            # status-299: 接触マネージャ状態を揺動フェーズに引き継ぎ（旧2フェーズ用）
            if (
                not _combined_mode
                and solver_result_bend.final_contact_manager is not None
                and cfg.n_oscillation_cycles > 0
            ):
                _bend_mgr = solver_result_bend.final_contact_manager
                contact_setup = ContactSetupData(
                    manager=_bend_mgr,
                    k_pen=cfg.k_pen,
                    mu=cfg.mu,
                )
                _n_active = sum(
                    1
                    for p in getattr(_bend_mgr, "pairs", [])
                    if hasattr(p, "state") and p.state.p_n > 0
                )
                print(
                    f"  接触マネージャ引き継ぎ: "
                    f"{len(getattr(_bend_mgr, 'pairs', []))} pairs ({_n_active} active)"
                )

        # ── 揺動フェーズ ──
        # 統合モード（_combined_mode）の場合、曲げ+揺動は prescribed_func で
        # 1回のソルバーで実行済み。旧2フェーズ方式はフォールバック。
        if _combined_mode:
            solver_result = solver_result_bend
        elif cfg.n_oscillation_cycles > 0:
            _n_osc = cfg.n_oscillation_cycles

            if cfg.oscillation_amplitude > 0.0:
                # status-299: 先端横変位揺動（u_z ±amplitude）
                # 曲げ完了位置を中心にsin波で往復。
                # prescribed_dofs: u_z（揺動）+ θ_y（曲げ完了値で固定）
                _uz_dofs = [n * 6 + 2 for n in right_nodes]
                _theta_dofs = [n * 6 + 4 for n in right_nodes]
                _osc_prescribed_dofs = np.array(_uz_dofs + _theta_dofs, dtype=int)
                _uz_at_bend = np.array([float(_u_bend[d]) for d in _uz_dofs])
                _theta_at_bend = np.array([float(_u_bend[d]) for d in _theta_dofs])
                _n_uz = len(_uz_dofs)
                _osc_amplitude = cfg.oscillation_amplitude
                # prescribed_values: u_z=曲げ完了値（frac=1.0でsin=0）, θ_y=曲げ完了値
                _osc_prescribed_values = np.concatenate([_uz_at_bend, _theta_at_bend])

                def _oscillation_func(frac: float) -> np.ndarray:
                    # prescribed_func は state.u[prescribed_dofs] に書き込む絶対値。
                    # u_z: 曲げ完了位置 + sin波変位
                    # θ_y: 曲げ完了値で固定（安定性確保）
                    delta_uz = _osc_amplitude * math.sin(2.0 * math.pi * _n_osc * frac)
                    uz_vals = _uz_at_bend + delta_uz
                    return np.concatenate([uz_vals, _theta_at_bend])

                print(
                    f"  揺動フェーズ: 先端u_z横変位±{_osc_amplitude:.1f}mm, "
                    f"{_n_osc}サイクル（θ_y固定）"
                )
                _uz_str = ", ".join(f"{uz:.2f}" for uz in _uz_at_bend[:3])
                _uz_suffix = "..." if len(_uz_at_bend) > 3 else ""
                print(f"  曲げ完了時u_z: {_uz_str}{_uz_suffix}")
            else:
                # 従来: θ_y回転cos波揺動（status-286互換）
                _osc_prescribed_dofs = prescribed_dofs_arr
                _osc_prescribed_values = prescribed_values_arr
                _theta_at_ckpt = float(_u_bend[prescribed_dofs_arr[0]])
                _theta_amplitude = _theta_at_ckpt

                def _oscillation_func(frac: float) -> np.ndarray:
                    # cos(0)=1 → Δθ=0, cos(π)=-1 → Δθ=-2θ_ckpt
                    delta_theta = _theta_amplitude * (math.cos(2.0 * math.pi * _n_osc * frac) - 1.0)
                    return np.full(len(_osc_prescribed_dofs), delta_theta)

            # 揺動フェーズの時間パラメータ
            t_osc = t_cycle * cfg.n_oscillation_cycles

            boundary_osc = BoundaryData(
                fixed_dofs=fixed_dofs_arr,
                prescribed_dofs=_osc_prescribed_dofs,
                prescribed_values=_osc_prescribed_values,
                f_ext_total=f_ext,
                mpc_transform=None,
                prescribed_func=_oscillation_func,
            )

            # status-299: 揺動フェーズ用に新しいアセンブラを作成。
            # 曲げ中のupdate_reference()で座標が変わったアセンブラを再利用すると
            # u_incr=0 → f_int=0 になり応力状態が消失する。
            # 新品アセンブラ（原点メッシュ）+ u0=_u_bend（全累積変位）で
            # f_int(u_bend) が正しく計算される。
            _beam_result_osc = ULCRBeamAssemblerProcess().process(
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
            _asm_osc = _beam_result_osc.assembler
            _callbacks_osc = AssembleCallbacks(
                assemble_tangent=_asm_osc.assemble_tangent,
                assemble_internal_force=_asm_osc.assemble_internal_force,
                ul_assembler=_asm_osc,
            )

            solver_input_osc = ContactFrictionInputData(
                mesh=mesh,
                boundary=boundary_osc,
                contact=contact_setup,
                callbacks=_callbacks_osc,
                u0=_u_bend,
                mass_matrix=M,
                dt_physical=t_osc,
                rho_inf=cfg.rho_inf,
                velocity=_vel_bend,
                acceleration=_acc_bend,
                max_nr_attempts=cfg.max_nr_attempts,
                tol_force=cfg.tol_force,
                max_increments=cfg.max_increments,
                tangent_fd_diagnostic=cfg.tangent_fd_diagnostic,
                du_norm_cap=cfg.du_norm_cap,
                penalty_exponent=cfg.penalty_exponent,
                skip_initial_detection=False,
            )
            solver_result = ContactFrictionProcess().process(solver_input_osc)
        else:
            solver_result = solver_result_bend

        return StrandBendingOscillationResult(
            solver_result=solver_result,
            mesh=mesh,
            n_ref_nodes=0,
            n_strand_nodes=n_strand_nodes,
            total_ndof=ndof,
            bending_angle=bending_angle,
        )
