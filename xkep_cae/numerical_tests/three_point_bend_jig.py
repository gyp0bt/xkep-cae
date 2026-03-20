"""単線の剛体支えと押しジグによる三点曲げ試験 Process.

剛体支持点（ピン + ローラー）で支持した単線ワイヤを、
剛体押しジグ（変位制御）で中央から押し下げる三点曲げ。
変位–荷重応答を Euler-Bernoulli / Timoshenko 解析解と比較する。

3つの Process を提供:
  - ThreePointBendJigProcess: 直接変位制御（理想剛体ジグ、準静的）
  - DynamicThreePointBendJigProcess: 直接変位制御（動的、質量行列付き）
  - ThreePointBendContactJigProcess: HEX8 連続体要素ジグ + 接触

物理モデル:
  - ワイヤ: x軸方向直線梁（Timoshenko CR 3D）
  - 支持: 左端=ピン（xyz+rx固定）、右端=ローラー（yz固定）
  - ジグ: 変位制御（直接 or HEX8 接触）

解析解:
  EB:   δ = PL³/(48EI)
  Timo: δ = PL³/(48EI) + PL/(4κGA)

[← README](../../README.md)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

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
from xkep_cae.elements._beam_assembler import ULCRBeamAssembler
from xkep_cae.elements._beam_cr import timo_beam3d_ke_global
from xkep_cae.elements._hex8_assembler import Hex8Assembler
from xkep_cae.elements._mixed_assembler import MixedAssembler

# ====================================================================
# 入出力データ
# ====================================================================


@dataclass(frozen=True)
class ThreePointBendJigConfig:
    """三点曲げジグ試験の構成."""

    wire_length: float = 100.0  # mm
    wire_diameter: float = 2.0  # mm
    n_elems_wire: int = 20
    E: float = 200e3  # MPa
    nu: float = 0.3
    jig_push: float = 0.1  # mm（ジグ下方変位量）


@dataclass(frozen=True)
class ThreePointBendJigResult:
    """三点曲げジグ試験の結果."""

    solver_result: SolverResultData
    wire_midpoint_deflection: float
    reaction_force: float
    analytical_deflection_eb: float
    analytical_deflection_timo: float
    analytical_stiffness_eb: float
    analytical_stiffness_timo: float
    relative_error_eb: float
    relative_error_timo: float
    config: ThreePointBendJigConfig
    mesh: MeshData
    wire_mid_node: int
    n_wire_nodes: int


# ====================================================================
# 断面定数
# ====================================================================


def _circle_section(d: float, nu: float) -> dict:
    """円形断面の断面定数."""
    r = d / 2.0
    A = math.pi * r**2
    Iy = math.pi * d**4 / 64.0
    J = math.pi * d**4 / 32.0
    kappa = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)
    return {"A": A, "Iy": Iy, "Iz": Iy, "J": J, "kappa": kappa}


# ====================================================================
# 解析解
# ====================================================================


def _analytical_three_point_bend(
    P: float,
    L: float,
    E: float,
    I: float,  # noqa: E741
    kappa: float,
    G: float,
    A: float,
) -> dict:
    """三点曲げ解析解（EB + Timoshenko）.

    Returns:
        dict: delta_eb, delta_timo, stiffness_eb, stiffness_timo
    """
    delta_eb = P * L**3 / (48.0 * E * I)
    delta_shear = P * L / (4.0 * kappa * G * A)
    delta_timo = delta_eb + delta_shear
    stiffness_eb = 48.0 * E * I / L**3
    stiffness_timo = P / delta_timo if delta_timo > 0 else stiffness_eb
    return {
        "delta_eb": delta_eb,
        "delta_timo": delta_timo,
        "delta_shear": delta_shear,
        "stiffness_eb": stiffness_eb,
        "stiffness_timo": stiffness_timo,
    }


# ====================================================================
# メッシュ生成
# ====================================================================


def _build_wire_mesh(cfg: ThreePointBendJigConfig) -> tuple[MeshData, int]:
    """単線ワイヤメッシュを生成する.

    Returns:
        mesh_data: ワイヤメッシュ
        wire_mid_node: 中央節点インデックス
    """
    n_elems = cfg.n_elems_wire
    if n_elems % 2 != 0:
        n_elems += 1

    L = cfg.wire_length
    R = cfg.wire_diameter / 2.0
    n_nodes = n_elems + 1
    x = np.linspace(0, L, n_nodes)
    nodes = np.column_stack([x, np.zeros(n_nodes), np.zeros(n_nodes)])
    conn = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])
    radii = np.full(n_elems, R)

    mesh_data = MeshData(
        node_coords=nodes,
        connectivity=conn,
        radii=radii,
        n_strands=1,
    )
    wire_mid_node = n_elems // 2
    return mesh_data, wire_mid_node


# ====================================================================
# 剛性行列直接アセンブリ（線形 Timoshenko 3D）
# ====================================================================


def _assemble_linear_stiffness(mesh: MeshData, E: float, G: float, sec: dict) -> sp.csr_matrix:
    """線形 Timoshenko 3D 梁の全体剛性行列."""
    nc = mesh.node_coords
    conn = mesh.connectivity
    n_nodes = len(nc)
    ndof = n_nodes * 6

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for elem in conn:
        n1, n2 = int(elem[0]), int(elem[1])
        coords = nc[np.array([n1, n2])]
        Ke = timo_beam3d_ke_global(
            coords,
            E,
            G,
            sec["A"],
            sec["Iy"],
            sec["Iz"],
            sec["J"],
            sec["kappa"],
            sec["kappa"],
        )
        edofs = np.array(
            [6 * n1 + d for d in range(6)] + [6 * n2 + d for d in range(6)],
            dtype=int,
        )
        for ii in range(12):
            for jj in range(12):
                if abs(Ke[ii, jj]) > 1e-30:
                    rows.append(edofs[ii])
                    cols.append(edofs[jj])
                    vals.append(Ke[ii, jj])

    return sp.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(ndof, ndof),
    )


# ====================================================================
# Process
# ====================================================================


class ThreePointBendJigProcess(BatchProcess[ThreePointBendJigConfig, ThreePointBendJigResult]):
    """単線の剛体支え＋押しジグ三点曲げ Process.

    理想剛体ジグ: ワイヤ中央節点に直接変位制御を適用。
    接触コンプライアンスなし → 解析解と厳密に一致可能。

    パイプライン:
    1. ワイヤメッシュ生成
    2. UL CR 梁アセンブラ構築
    3. 境界条件設定（支持 + 中央変位制御）
    4. ContactFrictionProcess で求解（接触なし、準静的）
    5. 反力計算 + 解析解比較
    """

    meta = ProcessMeta(
        name="ThreePointBendJig",
        module="batch",
        version="1.0.0",
        document_path="docs/three_point_bend_jig.md",
    )
    uses = [ContactFrictionProcess]

    def process(self, input_data: ThreePointBendJigConfig) -> ThreePointBendJigResult:
        """三点曲げジグ試験を実行."""
        cfg = input_data
        sec = _circle_section(cfg.wire_diameter, cfg.nu)
        G = cfg.E / (2.0 * (1.0 + cfg.nu))

        # 1. メッシュ
        mesh_data, wire_mid_node = _build_wire_mesh(cfg)
        n_nodes = len(mesh_data.node_coords)
        n_wire_nodes = n_nodes
        ndof = n_nodes * 6

        # 2. アセンブラ（UL CR 梁）
        assembler = ULCRBeamAssembler(
            node_coords=mesh_data.node_coords,
            connectivity=mesh_data.connectivity,
            E=cfg.E,
            G=G,
            A=sec["A"],
            Iy=sec["Iy"],
            Iz=sec["Iz"],
            J=sec["J"],
            kappa_y=sec["kappa"],
            kappa_z=sec["kappa"],
        )

        # 3. 境界条件
        #    左端（node 0）: x, y, z 並進 + rx 固定（ピン + ねじり拘束）
        #    右端（node n-1）: y, z 並進固定（ローラー）
        #    中央（wire_mid_node）: y 変位処方（ジグ押し）
        fixed_dofs = set()
        for d in [0, 1, 2, 3]:
            fixed_dofs.add(d)
        right_node = n_wire_nodes - 1
        fixed_dofs.add(6 * right_node + 1)
        fixed_dofs.add(6 * right_node + 2)
        fixed_dofs = np.array(sorted(fixed_dofs), dtype=int)

        # ジグ = 中央節点の y 変位を直接制御
        prescribed_dofs = np.array([6 * wire_mid_node + 1], dtype=int)
        prescribed_values = np.array([-cfg.jig_push])  # 負 = 下方

        boundary = BoundaryData(
            fixed_dofs=fixed_dofs,
            prescribed_dofs=prescribed_dofs,
            prescribed_values=prescribed_values,
            f_ext_total=np.zeros(ndof),
        )

        # 4. 接触設定（ダミー — 接触なし）
        contact_config = _ContactConfigInput(
            contact_mode="smooth_penalty",
        )
        manager = _ContactManagerInput(config=contact_config)
        contact_setup = ContactSetupData(
            manager=manager,
            k_pen=0.0,
            use_friction=False,
            mu=None,
            contact_mode="smooth_penalty",
        )

        # 5. ソルバー実行
        solver_input = ContactFrictionInputData(
            mesh=mesh_data,
            boundary=boundary,
            contact=contact_setup,
            callbacks=AssembleCallbacks(
                assemble_tangent=assembler.assemble_tangent,
                assemble_internal_force=assembler.assemble_internal_force,
                ul_assembler=assembler,
            ),
        )
        solver = ContactFrictionProcess()
        solver_result = solver.process(solver_input)

        # 6. 結果評価
        u = solver_result.u
        wire_mid_y_dof = 6 * wire_mid_node + 1
        wire_mid_deflection = abs(u[wire_mid_y_dof])

        # 反力計算: K*u から中央節点の y 方向内力を取得
        K = _assemble_linear_stiffness(mesh_data, cfg.E, G, sec)
        f_int = K.dot(u)
        reaction_force = abs(f_int[wire_mid_y_dof])

        # 解析解
        ana = _analytical_three_point_bend(
            P=reaction_force,
            L=cfg.wire_length,
            E=cfg.E,
            I=sec["Iy"],
            kappa=sec["kappa"],
            G=G,
            A=sec["A"],
        )

        # 相対誤差: 処方変位 vs 解析解からの期待変位
        expected_eb = ana["delta_eb"]
        expected_timo = ana["delta_timo"]
        rel_err_eb = (
            abs(wire_mid_deflection - expected_eb) / expected_eb
            if expected_eb > 0
            else float("inf")
        )
        rel_err_timo = (
            abs(wire_mid_deflection - expected_timo) / expected_timo
            if expected_timo > 0
            else float("inf")
        )

        return ThreePointBendJigResult(
            solver_result=solver_result,
            wire_midpoint_deflection=wire_mid_deflection,
            reaction_force=reaction_force,
            analytical_deflection_eb=expected_eb,
            analytical_deflection_timo=expected_timo,
            analytical_stiffness_eb=ana["stiffness_eb"],
            analytical_stiffness_timo=ana["stiffness_timo"],
            relative_error_eb=rel_err_eb,
            relative_error_timo=rel_err_timo,
            config=cfg,
            mesh=mesh_data,
            wire_mid_node=wire_mid_node,
            n_wire_nodes=n_wire_nodes,
        )


# ====================================================================
# HEX8 連続体要素ジグ版 — 接触ベースの三点曲げ
# ====================================================================


@dataclass(frozen=True)
class ThreePointBendContactJigConfig:
    """HEX8 接触ジグ版三点曲げの構成."""

    wire_length: float = 100.0  # mm
    wire_diameter: float = 2.0  # mm
    n_elems_wire: int = 20
    E: float = 200e3  # MPa（ワイヤ）
    nu: float = 0.3
    jig_push: float = 0.1  # mm（ジグ下方変位量）
    jig_E_factor: float = 1000.0  # ジグ/ワイヤ剛性比
    jig_width: float = 1.0  # mm（x方向）
    jig_depth: float = 4.0  # mm（z方向、ワイヤ径の2倍）
    jig_height: float = 2.0  # mm（y方向）
    initial_gap: float = 0.0  # mm（初期ギャップ、0=接触面一致）
    k_pen: float = 0.0  # ペナルティ剛性（0=自動推定）
    smoothing_delta: float = 200.0  # softplus の平滑化パラメータ
    n_uzawa_max: int = 20  # Uzawa 最大反復回数


@dataclass(frozen=True)
class ThreePointBendContactJigResult:
    """HEX8 接触ジグ版三点曲げの結果."""

    solver_result: SolverResultData
    wire_midpoint_deflection: float
    contact_force_norm: float
    analytical_stiffness_eb: float
    analytical_stiffness_timo: float
    effective_stiffness: float
    stiffness_error_eb: float
    config: ThreePointBendContactJigConfig
    wire_mid_node: int
    n_wire_nodes: int
    n_hex_nodes: int


def _build_hex8_jig_mesh(
    cfg: ThreePointBendContactJigConfig,
    n_wire_nodes: int,
    wire_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    """HEX8 ジグメッシュを生成する.

    Returns:
        jig_coords: (8, 3) ジグ節点座標
        jig_hex_conn: (1, 8) HEX8 接続
        jig_edge_conn: (2, 2) 接触エッジ接続（全体節点番号）
        jig_top_nodes: 上面節点の全体インデックス
        jig_rot_dofs: ソリッド節点の回転DOFリスト
    """
    L = cfg.wire_length
    w = cfg.jig_width
    d = cfg.jig_depth
    h = cfg.jig_height
    gap_y = wire_radius + cfg.initial_gap

    cx = L / 2.0  # ジグ中心 x
    # ジグ底面は y=gap_y、上面は y=gap_y+h
    # z方向は [-d/2, +d/2]
    jig_coords = np.array(
        [
            [cx - w / 2, gap_y, -d / 2],  # 0: 底面左前
            [cx + w / 2, gap_y, -d / 2],  # 1: 底面右前
            [cx + w / 2, gap_y, +d / 2],  # 2: 底面右後
            [cx - w / 2, gap_y, +d / 2],  # 3: 底面左後
            [cx - w / 2, gap_y + h, -d / 2],  # 4: 上面左前
            [cx + w / 2, gap_y + h, -d / 2],  # 5: 上面右前
            [cx + w / 2, gap_y + h, +d / 2],  # 6: 上面右後
            [cx - w / 2, gap_y + h, +d / 2],  # 7: 上面左後
        ]
    )

    # HEX8 接続（ローカル節点番号）
    jig_hex_conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])

    # 接触エッジ: 底面の z 方向エッジ（ワイヤ軸に垂直）
    # 全体節点番号 = n_wire_nodes + ローカル番号
    off = n_wire_nodes
    jig_edge_conn = np.array(
        [
            [off + 0, off + 3],  # 左側 z-edge
            [off + 1, off + 2],  # 右側 z-edge
        ]
    )

    # 上面節点（全体番号）— 変位処方
    jig_top_nodes = [off + 4, off + 5, off + 6, off + 7]

    # ソリッド節点の回転 DOF（固定すべき）
    jig_rot_dofs = []
    for i in range(8):
        gn = off + i
        for d_idx in [3, 4, 5]:
            jig_rot_dofs.append(6 * gn + d_idx)

    return jig_coords, jig_hex_conn, jig_edge_conn, jig_top_nodes, jig_rot_dofs


class ThreePointBendContactJigProcess(
    BatchProcess[ThreePointBendContactJigConfig, ThreePointBendContactJigResult],
):
    """HEX8 連続体要素ジグ + 接触による三点曲げ Process.

    物理モデル:
        ┌──────────────┐  ← HEX8 ジグ（高剛性）
        └──────┬───────┘     top: 変位処方
               ↓ 接触          bottom: smooth penalty 接触
     ─────────●─────────  ← ワイヤ（CR 梁）
     △                 ○
     ピン             ローラー

    パイプライン:
    1. ワイヤメッシュ + HEX8 ジグメッシュ生成
    2. MixedAssembler（梁 + HEX8）構築
    3. 境界条件（支持 + ジグ上面変位制御 + ソリッド回転DOF固定）
    4. ContactFrictionProcess で求解（smooth_penalty 接触）
    5. 結果評価 + 解析解比較
    """

    meta = ProcessMeta(
        name="ThreePointBendContactJig",
        module="batch",
        version="1.0.0",
        document_path="docs/three_point_bend_jig.md",
    )
    uses = [ContactFrictionProcess]

    def process(self, input_data: ThreePointBendContactJigConfig) -> ThreePointBendContactJigResult:
        """接触ジグ三点曲げ試験を実行."""
        cfg = input_data
        sec = _circle_section(cfg.wire_diameter, cfg.nu)
        G = cfg.E / (2.0 * (1.0 + cfg.nu))
        wire_radius = cfg.wire_diameter / 2.0

        # 1. ワイヤメッシュ
        wire_mesh, wire_mid_node = _build_wire_mesh(
            ThreePointBendJigConfig(
                wire_length=cfg.wire_length,
                wire_diameter=cfg.wire_diameter,
                n_elems_wire=cfg.n_elems_wire,
                E=cfg.E,
                nu=cfg.nu,
                jig_push=cfg.jig_push,
            )
        )
        n_wire_nodes = len(wire_mesh.node_coords)
        n_wire_elems = len(wire_mesh.connectivity)

        # 2. HEX8 ジグメッシュ
        jig_coords, jig_hex_conn, jig_edge_conn, jig_top_nodes, jig_rot_dofs = _build_hex8_jig_mesh(
            cfg, n_wire_nodes, wire_radius
        )
        n_hex_nodes = 8

        # 3. 統合メッシュ（接触検出用）
        all_nodes = np.vstack([wire_mesh.node_coords, jig_coords])
        n_total_nodes = len(all_nodes)
        ndof = n_total_nodes * 6

        # 接触セグメント: 梁要素 + ジグ底面 z-edge
        all_conn = np.vstack([wire_mesh.connectivity, jig_edge_conn])
        n_total_segments = len(all_conn)

        # 半径: ワイヤ要素は wire_radius、ジグエッジは 0
        wire_radii = np.asarray(wire_mesh.radii)
        if wire_radii.ndim == 0:
            wire_radii = np.full(n_wire_elems, float(wire_radii))
        jig_radii = np.zeros(len(jig_edge_conn))
        all_radii = np.concatenate([wire_radii, jig_radii])

        # レイヤー ID（同層除外用）: ワイヤ=0、ジグ=1
        layer_ids = np.zeros(n_total_segments, dtype=int)
        layer_ids[n_wire_elems:] = 1
        elem_layer_map = {i: int(layer_ids[i]) for i in range(n_total_segments)}

        mesh_data = MeshData(
            node_coords=all_nodes,
            connectivity=all_conn,
            radii=all_radii,
            n_strands=1,
            layer_ids=layer_ids,
        )

        # 4. アセンブラ
        beam_asm = ULCRBeamAssembler(
            node_coords=wire_mesh.node_coords,
            connectivity=wire_mesh.connectivity,
            E=cfg.E,
            G=G,
            A=sec["A"],
            Iy=sec["Iy"],
            Iz=sec["Iz"],
            J=sec["J"],
            kappa_y=sec["kappa"],
            kappa_z=sec["kappa"],
        )

        E_jig = cfg.E * cfg.jig_E_factor
        hex_asm = Hex8Assembler(
            node_coords=jig_coords,
            connectivity=jig_hex_conn,
            E=E_jig,
            nu=cfg.nu,
            global_node_offset=n_wire_nodes,
            total_ndof=ndof,
        )

        mixed_asm = MixedAssembler(beam_asm, hex_asm, ndof)

        # 5. 境界条件
        fixed_dofs = set()
        # 左端ピン（xyz + rx）
        for d_idx in [0, 1, 2, 3]:
            fixed_dofs.add(d_idx)
        # 右端ローラー（yz）
        right_node = n_wire_nodes - 1
        fixed_dofs.add(6 * right_node + 1)
        fixed_dofs.add(6 * right_node + 2)
        # ソリッド節点の回転 DOF 固定
        for rd in jig_rot_dofs:
            fixed_dofs.add(rd)
        # ジグ全節点の x, z 並進を固定（y 方向のみ変位処方 + 接触）
        for i in range(8):
            gn = n_wire_nodes + i
            fixed_dofs.add(6 * gn + 0)  # x
            fixed_dofs.add(6 * gn + 2)  # z

        # ジグ全 8 節点の y 変位を処方（剛体ジグ → 全ノード同一変位）
        # HEX8 内で上面のみ処方すると内部応力が巨大になり NR 発散。
        prescribed_dofs = []
        for i in range(8):
            gn = n_wire_nodes + i
            prescribed_dofs.append(6 * gn + 1)  # y-dof

        fixed_dofs_arr = np.array(sorted(fixed_dofs), dtype=int)
        prescribed_dofs_arr = np.array(prescribed_dofs, dtype=int)
        push_total = cfg.jig_push + cfg.initial_gap
        prescribed_values = np.full(len(prescribed_dofs), -push_total)

        boundary = BoundaryData(
            fixed_dofs=fixed_dofs_arr,
            prescribed_dofs=prescribed_dofs_arr,
            prescribed_values=prescribed_values,
            f_ext_total=np.zeros(ndof),
        )

        # 6. 接触設定
        #    k_pen 推定: 梁グローバル剛性ベース
        #    auto（12EI/L_elem³ベース）は梁–梁接触向け設計。
        #    梁–剛体ジグ接触では接触点の有効剛性 = 48EI/L³ が支配的。
        k_pen = cfg.k_pen
        if k_pen <= 0.0:
            k_beam_global = 48.0 * cfg.E * sec["Iy"] / cfg.wire_length**3
            k_pen = 0.5 * k_beam_global

        contact_config = _ContactConfigInput(
            contact_mode="smooth_penalty",
            smoothing_delta=cfg.smoothing_delta,
            n_uzawa_max=cfg.n_uzawa_max,
            beam_E=cfg.E,
            beam_I=sec["Iy"],
            beam_A=sec["A"],
            exclude_same_layer=True,
            elem_layer_map=elem_layer_map,
            adjust_initial_penetration=False,
            adaptive_timestepping=True,
            dt_grow_factor=1.5,
            dt_shrink_factor=0.5,
        )
        manager = _ContactManagerInput(config=contact_config)
        contact_setup = ContactSetupData(
            manager=manager,
            k_pen=k_pen,
            use_friction=False,
            mu=None,
            contact_mode="smooth_penalty",
        )

        # 7. ソルバー実行
        solver_input = ContactFrictionInputData(
            mesh=mesh_data,
            boundary=boundary,
            contact=contact_setup,
            callbacks=AssembleCallbacks(
                assemble_tangent=mixed_asm.assemble_tangent,
                assemble_internal_force=mixed_asm.assemble_internal_force,
                ul_assembler=mixed_asm,
            ),
        )
        solver = ContactFrictionProcess()
        solver_result = solver.process(solver_input)

        # 8. 結果評価
        u = solver_result.u
        wire_mid_y_dof = 6 * wire_mid_node + 1
        wire_mid_deflection = abs(u[wire_mid_y_dof])

        # 接触力ノルム
        fc_hist = solver_result.contact_force_history
        contact_force_norm = fc_hist[-1] if fc_hist else 0.0

        # 解析解
        ana = _analytical_three_point_bend(
            P=1.0,
            L=cfg.wire_length,
            E=cfg.E,
            I=sec["Iy"],
            kappa=sec["kappa"],
            G=G,
            A=sec["A"],
        )

        # 有効剛性 = 反力 / 変位
        effective_stiffness = (
            contact_force_norm / wire_mid_deflection if wire_mid_deflection > 1e-15 else 0.0
        )
        stiffness_error_eb = (
            abs(effective_stiffness - ana["stiffness_eb"]) / ana["stiffness_eb"]
            if ana["stiffness_eb"] > 0
            else float("inf")
        )

        return ThreePointBendContactJigResult(
            solver_result=solver_result,
            wire_midpoint_deflection=wire_mid_deflection,
            contact_force_norm=contact_force_norm,
            analytical_stiffness_eb=ana["stiffness_eb"],
            analytical_stiffness_timo=ana["stiffness_timo"],
            effective_stiffness=effective_stiffness,
            stiffness_error_eb=stiffness_error_eb,
            config=cfg,
            wire_mid_node=wire_mid_node,
            n_wire_nodes=n_wire_nodes,
            n_hex_nodes=n_hex_nodes,
        )


# ====================================================================
# 動的三点曲げ Process（初速度制御 + Generalized-α 時間積分）
# ====================================================================


@dataclass(frozen=True)
class DynamicThreePointBendJigConfig:
    """動的三点曲げジグ試験の構成.

    時間増分は dt_initial / dt_min / max_increments で制御。
    適応時間増分が収束性に応じて自動調整する。
    """

    wire_length: float = 100.0  # mm
    wire_diameter: float = 2.0  # mm
    n_elems_wire: int = 20
    E: float = 200e3  # MPa
    nu: float = 0.3
    rho: float = 7.85e-9  # ton/mm³ (鉄鋼)
    jig_push: float = 0.1  # mm（等価静的変位量 → 初速度に変換）
    n_periods: float = 3.0  # 固有周期の何周期分を計算するか
    rho_inf: float = 0.9  # Generalized-α の数値減衰パラメータ
    lumped_mass: bool = True  # 集中質量行列
    # 時間増分制御
    dt_initial: float = 0.0  # 初期時間増分 [s]（0=自動: T1/20）
    dt_min: float = 0.0  # 許容最低時間増分 [s]（0=自動: dt_initial/32）
    max_increments: int = 10000  # 最大インクリメント数


@dataclass(frozen=True)
class DynamicThreePointBendJigResult:
    """動的三点曲げジグ試験の結果."""

    solver_result: SolverResultData
    wire_midpoint_deflection: float
    reaction_force: float
    analytical_deflection_static: float
    analytical_frequency_hz: float
    analytical_period: float
    dynamic_amplification: float
    max_deflection: float
    initial_velocity: float
    config: DynamicThreePointBendJigConfig
    mesh: MeshData
    wire_mid_node: int
    n_wire_nodes: int
    time_history: np.ndarray
    deflection_history: np.ndarray


def _beam_fundamental_frequency(
    L: float,
    E: float,
    I: float,  # noqa: E741
    rho: float,
    A: float,  # noqa: E741
) -> float:
    """単純支持梁の1次固有振動数 [Hz].

    f_1 = (π²/2π) * sqrt(EI / (ρAL⁴)) = π/(2L²) * sqrt(EI/(ρA))
    """
    return (math.pi / (2.0 * L**2)) * math.sqrt(E * I / (rho * A))


class DynamicThreePointBendJigProcess(
    BatchProcess[DynamicThreePointBendJigConfig, DynamicThreePointBendJigResult],
):
    """単線の剛体支え＋押しジグ動的三点曲げ Process.

    初速度制御: ワイヤ中央節点に初速度 v₀ を与え、自由振動応答を計算。
    Generalized-α 時間積分で動的応答を計算し、解析解と比較する。

    初速度の決定:
      v₀ = ω₁ * δ_s（1次固有角振動数 × 静的等価変位）
      → 最大変位 ≈ δ_s（自由振動の振幅が静的変位に一致）

    解析解（初速度 v₀ による自由振動）:
      - 静的等価変位: δ_s = jig_push
      - 1次固有振動数: f₁ = π/(2L²) √(EI/ρA)
      - 応答: δ(t) = (v₀/ω₁) * sin(ω₁t)  （モード1のみ）
      - 最大変位: δ_max = v₀/ω₁ = δ_s

    時間増分制御:
      dt_initial, dt_min, max_increments で適応時間増分を制御。

    パイプライン:
    1. ワイヤメッシュ生成
    2. UL CR 梁アセンブラ + 質量行列構築
    3. 初速度設定 + 境界条件（支持のみ、外力なし）
    4. ContactFrictionProcess（動的モード）で求解
    5. 時刻歴応答 + 解析解比較
    """

    meta = ProcessMeta(
        name="DynamicThreePointBendJig",
        module="batch",
        version="2.0.0",
        document_path="docs/three_point_bend_jig.md",
    )
    uses = [ContactFrictionProcess]

    def process(self, input_data: DynamicThreePointBendJigConfig) -> DynamicThreePointBendJigResult:
        """動的三点曲げジグ試験を実行."""
        cfg = input_data
        sec = _circle_section(cfg.wire_diameter, cfg.nu)
        G = cfg.E / (2.0 * (1.0 + cfg.nu))

        # 1. メッシュ
        mesh_data, wire_mid_node = _build_wire_mesh(
            ThreePointBendJigConfig(
                wire_length=cfg.wire_length,
                wire_diameter=cfg.wire_diameter,
                n_elems_wire=cfg.n_elems_wire,
                E=cfg.E,
                nu=cfg.nu,
                jig_push=cfg.jig_push,
            )
        )
        n_nodes = len(mesh_data.node_coords)
        n_wire_nodes = n_nodes
        ndof = n_nodes * 6

        # 2. アセンブラ + 質量行列
        assembler = ULCRBeamAssembler(
            node_coords=mesh_data.node_coords,
            connectivity=mesh_data.connectivity,
            E=cfg.E,
            G=G,
            A=sec["A"],
            Iy=sec["Iy"],
            Iz=sec["Iz"],
            J=sec["J"],
            kappa_y=sec["kappa"],
            kappa_z=sec["kappa"],
        )
        mass_matrix = assembler.assemble_mass(cfg.rho, lumped=cfg.lumped_mass)

        # 3. 固有振動数 → 時間制御パラメータ
        f1 = _beam_fundamental_frequency(cfg.wire_length, cfg.E, sec["Iy"], cfg.rho, sec["A"])
        T1 = 1.0 / f1
        omega1 = 2.0 * math.pi * f1
        t_total = cfg.n_periods * T1

        # 初速度: モーダル質量補正付き
        # 梁の集中質量では節点質量 m_mid ≠ モーダル質量 M₁
        wire_mid_y_dof = 6 * wire_mid_node + 1
        phi1 = np.zeros(ndof)
        for i in range(n_nodes):
            x_i = mesh_data.node_coords[i, 0]
            phi1[6 * i + 1] = math.sin(math.pi * x_i / cfg.wire_length)
        M_modal = float(phi1 @ mass_matrix @ phi1)
        m_mid = float(mass_matrix[wire_mid_y_dof, wire_mid_y_dof])
        modal_ratio = M_modal / m_mid if m_mid > 1e-30 else 1.0
        v0 = omega1 * cfg.jig_push * modal_ratio

        # 時間増分パラメータ
        dt_initial = cfg.dt_initial if cfg.dt_initial > 0 else T1 / 20.0
        dt_min = cfg.dt_min if cfg.dt_min > 0 else dt_initial / 32.0

        # load_frac ベースの時間増分設定
        dt_initial_frac = dt_initial / t_total
        dt_min_frac = dt_min / t_total

        # 4. 初速度ベクトル
        velocity = np.zeros(ndof)
        velocity[6 * wire_mid_node + 1] = -v0  # 下向き初速度

        # 5. 境界条件（支持のみ、外力なし）
        fixed_dofs = set()
        for d in [0, 1, 2, 3]:
            fixed_dofs.add(d)
        right_node = n_wire_nodes - 1
        fixed_dofs.add(6 * right_node + 1)
        fixed_dofs.add(6 * right_node + 2)
        fixed_dofs_arr = np.array(sorted(fixed_dofs), dtype=int)

        boundary = BoundaryData(
            fixed_dofs=fixed_dofs_arr,
            f_ext_total=np.zeros(ndof),
        )

        # 6. 接触設定（ダミー、適応時間増分で制御）
        contact_config = _ContactConfigInput(
            contact_mode="smooth_penalty",
            adaptive_timestepping=True,
            dt_grow_factor=1.5,
            dt_shrink_factor=0.5,
            dt_min_fraction=dt_min_frac,
            dt_max_fraction=dt_initial_frac,
        )
        manager = _ContactManagerInput(config=contact_config)
        contact_setup = ContactSetupData(
            manager=manager,
            k_pen=0.0,
            use_friction=False,
            mu=None,
            contact_mode="smooth_penalty",
        )

        # 7. ソルバー実行（動的モード、初速度付き）
        solver_input = ContactFrictionInputData(
            mesh=mesh_data,
            boundary=boundary,
            contact=contact_setup,
            callbacks=AssembleCallbacks(
                assemble_tangent=assembler.assemble_tangent,
                assemble_internal_force=assembler.assemble_internal_force,
                ul_assembler=assembler,
            ),
            mass_matrix=mass_matrix,
            dt_physical=t_total,
            rho_inf=cfg.rho_inf,
            velocity=velocity,
        )
        solver = ContactFrictionProcess()
        solver_result = solver.process(solver_input)

        # 8. 結果評価
        u = solver_result.u
        wire_mid_y_dof = 6 * wire_mid_node + 1
        wire_mid_deflection = abs(u[wire_mid_y_dof])

        # 時刻歴
        disp_hist = solver_result.displacement_history
        n_hist = len(disp_hist)
        time_arr = np.linspace(0, t_total, n_hist) if n_hist > 0 else np.array([0.0])
        defl_arr = np.array([abs(d[wire_mid_y_dof]) for d in disp_hist] if n_hist > 0 else [0.0])
        max_deflection = float(np.max(defl_arr)) if len(defl_arr) > 0 else 0.0

        # 反力計算
        K = _assemble_linear_stiffness(mesh_data, cfg.E, G, sec)
        f_int = K.dot(u)
        reaction_force = abs(f_int[wire_mid_y_dof])

        # 解析解
        static_deflection = cfg.jig_push  # v₀ = ω₁*δ_s → 振幅 = δ_s

        # 動的増幅率（max_deflection / static_deflection）
        dynamic_amplification = max_deflection / static_deflection if static_deflection > 0 else 0.0

        return DynamicThreePointBendJigResult(
            solver_result=solver_result,
            wire_midpoint_deflection=wire_mid_deflection,
            reaction_force=reaction_force,
            analytical_deflection_static=static_deflection,
            analytical_frequency_hz=f1,
            analytical_period=T1,
            dynamic_amplification=dynamic_amplification,
            max_deflection=max_deflection,
            initial_velocity=v0,
            config=cfg,
            mesh=mesh_data,
            wire_mid_node=wire_mid_node,
            n_wire_nodes=n_wire_nodes,
            time_history=time_arr,
            deflection_history=defl_arr,
        )
