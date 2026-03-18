"""単線の剛体支えと押しジグによる三点曲げ試験 Process.

剛体支持点（ピン + ローラー）で支持した単線ワイヤを、
剛体押しジグ（変位制御）で中央から押し下げる三点曲げ。
変位–荷重応答を Euler-Bernoulli / Timoshenko 解析解と比較する。

物理モデル:
  - ワイヤ: x軸方向直線梁（Timoshenko CR 3D）
  - 支持: 左端=ピン（xyz+rx固定）、右端=ローラー（yz固定）
  - ジグ: ワイヤ中央節点への直接変位制御（理想剛体ジグ）

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
