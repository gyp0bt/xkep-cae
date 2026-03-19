"""接触なし梁揺動解析 Process.

単純支持梁に初速度を与え、接触なしで自由振動させる。
三点曲げの前準備として、非線形域の動的挙動を検証する。

物理モデル:
  - ワイヤ: x軸方向直線梁（Timoshenko CR 3D）
  - 支持: 左端=ピン（xyz+rx固定）、右端=ローラー（yz固定）
  - 加振: 中央節点に初速度 v₀（y方向下向き）

検証項目:
  - 動的ソルバー（GeneralizedAlpha）の収束
  - 非線形域（大変形）での物理的妥当性
  - 数値粘性（高周波減衰率）の評価
  - 3D応力コンター可視化

解析解（小振幅線形の場合）:
  f₁ = π/(2L²) √(EI/ρA)
  δ(t) = (v₀/ω₁) sin(ω₁t)
  非線形域では解析解から乖離するが、エネルギー保存が妥当性指標。

[← README](../../README.md)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
from xkep_cae.numerical_tests.three_point_bend_jig import (
    ThreePointBendJigConfig,
    _beam_fundamental_frequency,
    _build_wire_mesh,
    _circle_section,
)

# ====================================================================
# 入出力データ
# ====================================================================


@dataclass(frozen=True)
class BeamOscillationConfig:
    """梁揺動解析の構成.

    メッシュサイズ ≈ 半径を目安とする（n_elems_wire = wire_length / (wire_diameter/2)）。
    初期時間増分は小さめに設定（T₁/100）。
    """

    wire_length: float = 100.0  # mm
    wire_diameter: float = 2.0  # mm
    n_elems_wire: int = 100  # メッシュサイズ ≈ 半径（1mm）
    E: float = 200e3  # MPa
    nu: float = 0.3
    rho: float = 7.85e-9  # ton/mm³ (鉄鋼)
    amplitude: float = 5.0  # mm（初速度等価振幅、非線形域）
    n_periods: float = 3.0  # 固有周期の何周期分を計算するか
    rho_inf: float = 0.9  # Generalized-α の数値減衰パラメータ
    lumped_mass: bool = True  # 集中質量行列
    # 時間増分制御
    dt_initial: float = 0.0  # 初期時間増分 [s]（0=自動: T₁/100）
    dt_min: float = 0.0  # 許容最低時間増分 [s]（0=自動: dt_initial/64）
    max_increments: int = 50000  # 最大インクリメント数


@dataclass(frozen=True)
class BeamOscillationResult:
    """梁揺動解析の結果."""

    solver_result: SolverResultData
    mesh: MeshData
    wire_mid_node: int
    n_wire_nodes: int
    config: BeamOscillationConfig
    # 物理量
    analytical_frequency_hz: float
    analytical_period: float
    initial_velocity: float
    # 時刻歴
    time_history: np.ndarray
    deflection_history: np.ndarray  # 中央節点 y 変位の符号付き時刻歴
    # 評価指標
    max_deflection: float
    amplitude_ratio: float  # max_deflection / config.amplitude
    # 数値粘性評価
    energy_history: np.ndarray  # 各ステップの運動+ポテンシャルエネルギー
    energy_decay_ratio: float  # 最終エネルギー / 初期エネルギー
    # 応力情報（要素ごと最大曲げ応力）
    element_stress_history: list[np.ndarray] = field(default_factory=list)


# ====================================================================
# 応力計算ユーティリティ
# ====================================================================


def compute_element_bending_stress(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    u: np.ndarray,
    E: float,
    wire_radius: float,
) -> np.ndarray:
    """変位場から要素ごとの最大曲げ応力を計算.

    曲率 κ を中心差分で近似し、σ_max = E * κ * R。
    両端要素は片側差分で近似。

    Args:
        node_coords: 初期節点座標 (n_nodes, 3)
        connectivity: 要素接続 (n_elems, 2)
        u: 変位ベクトル (ndof,)
        E: ヤング率
        wire_radius: ワイヤ半径

    Returns:
        element_stress: 各要素の最大曲げ応力 (n_elems,)
    """
    n_nodes = len(node_coords)
    n_elems = len(connectivity)

    # 変形後の y 変位
    uy = np.array([u[6 * i + 1] for i in range(n_nodes)])
    # x 座標
    x = node_coords[:, 0]

    # 節点ごとの曲率（中心差分）
    kappa_node = np.zeros(n_nodes)
    for i in range(1, n_nodes - 1):
        dx_m = x[i] - x[i - 1]
        dx_p = x[i + 1] - x[i]
        if dx_m > 0 and dx_p > 0:
            kappa_node[i] = 2.0 * (
                uy[i + 1] / (dx_p * (dx_m + dx_p))
                - uy[i] / (dx_m * dx_p)
                + uy[i - 1] / (dx_m * (dx_m + dx_p))
            )
    # 端部: 片側差分
    if n_nodes >= 3:
        dx = x[1] - x[0]
        if dx > 0:
            kappa_node[0] = kappa_node[1]
        dx = x[-1] - x[-2]
        if dx > 0:
            kappa_node[-1] = kappa_node[-2]

    # 要素ごと: 両端節点の曲率平均
    element_stress = np.zeros(n_elems)
    for e in range(n_elems):
        n0, n1 = connectivity[e]
        kappa_avg = 0.5 * (abs(kappa_node[n0]) + abs(kappa_node[n1]))
        element_stress[e] = E * kappa_avg * wire_radius

    return element_stress


# ====================================================================
# エネルギー計算
# ====================================================================


def _compute_kinetic_energy(
    mass_matrix: sp.spmatrix,
    velocity: np.ndarray,
) -> float:
    """運動エネルギー T = 0.5 * v^T M v."""
    return 0.5 * float(velocity @ mass_matrix @ velocity)


def _compute_strain_energy(
    assembler: ULCRBeamAssembler,
    u_incr: np.ndarray,
) -> float:
    """ひずみエネルギー U ≈ 0.5 * u^T f_int."""
    f_int = assembler.assemble_internal_force(u_incr)
    return 0.5 * float(np.dot(u_incr, f_int))


# ====================================================================
# Process
# ====================================================================


class BeamOscillationProcess(
    BatchProcess[BeamOscillationConfig, BeamOscillationResult],
):
    """接触なし梁揺動解析 Process.

    単純支持梁に初速度を与え、動的ソルバー（GeneralizedAlpha）で
    自由振動応答を計算する。接触なし。

    パイプライン:
    1. ワイヤメッシュ生成（メッシュサイズ ≈ 半径）
    2. UL CR 梁アセンブラ + 質量行列構築
    3. 初速度設定 + 境界条件（支持のみ）
    4. ContactFrictionProcess（動的モード、接触なし）で求解
    5. 時刻歴・応力・数値粘性の評価
    """

    meta = ProcessMeta(
        name="BeamOscillation",
        module="batch",
        version="1.0.0",
        document_path="docs/beam_oscillation.md",
    )
    uses = [ContactFrictionProcess]

    def process(self, input_data: BeamOscillationConfig) -> BeamOscillationResult:
        """梁揺動解析を実行."""
        cfg = input_data
        sec = _circle_section(cfg.wire_diameter, cfg.nu)
        G = cfg.E / (2.0 * (1.0 + cfg.nu))
        wire_radius = cfg.wire_diameter / 2.0

        # 1. メッシュ
        mesh_data, wire_mid_node = _build_wire_mesh(
            ThreePointBendJigConfig(
                wire_length=cfg.wire_length,
                wire_diameter=cfg.wire_diameter,
                n_elems_wire=cfg.n_elems_wire,
                E=cfg.E,
                nu=cfg.nu,
                jig_push=cfg.amplitude,
            )
        )
        n_nodes = len(mesh_data.node_coords)
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

        # 初速度: v₀ = ω₁ * δ_s → 小振幅での最大変位 ≈ amplitude
        v0 = omega1 * cfg.amplitude

        # 時間増分パラメータ（小さめ）
        dt_initial = cfg.dt_initial if cfg.dt_initial > 0 else T1 / 100.0
        dt_min = cfg.dt_min if cfg.dt_min > 0 else dt_initial / 64.0

        # load_frac ベースの時間増分設定
        dt_initial_frac = dt_initial / t_total
        dt_min_frac = dt_min / t_total

        # 4. 初速度ベクトル
        velocity = np.zeros(ndof)
        velocity[6 * wire_mid_node + 1] = -v0  # 下向き初速度

        # 5. 境界条件（支持のみ、外力なし）
        fixed_dofs = set()
        # 左端（node 0）: x, y, z + rx 固定（ピン）
        for d in [0, 1, 2, 3]:
            fixed_dofs.add(d)
        # 右端: y, z 固定（ローラー）
        right_node = n_nodes - 1
        fixed_dofs.add(6 * right_node + 1)
        fixed_dofs.add(6 * right_node + 2)
        fixed_dofs_arr = np.array(sorted(fixed_dofs), dtype=int)

        boundary = BoundaryData(
            fixed_dofs=fixed_dofs_arr,
            f_ext_total=np.zeros(ndof),
        )

        # 6. 接触設定（ダミー — 接触なし）
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
        wire_mid_y_dof = 6 * wire_mid_node + 1
        disp_hist = solver_result.displacement_history
        n_hist = len(disp_hist)
        time_arr = np.linspace(0, t_total, n_hist) if n_hist > 0 else np.array([0.0])

        # 符号付き変位（下向きが負）
        defl_arr = np.array([d[wire_mid_y_dof] for d in disp_hist] if n_hist > 0 else [0.0])
        max_deflection = float(np.max(np.abs(defl_arr)))

        # 振幅比
        amplitude_ratio = max_deflection / cfg.amplitude if cfg.amplitude > 0 else 0.0

        # 数値粘性評価: 速度近似 → 運動エネルギー推定
        energy_history = np.zeros(n_hist)
        if n_hist >= 2:
            for i in range(n_hist):
                # ひずみエネルギー（近似: 0.5 * k_eff * δ² — 線形近似）
                u_i = disp_hist[i]
                E_strain = 0.5 * float(np.dot(u_i, assembler.assemble_internal_force(u_i)))
                # 運動エネルギー（中心差分速度近似）
                if 0 < i < n_hist - 1:
                    dt = time_arr[i + 1] - time_arr[i - 1]
                    if dt > 0:
                        v_approx = (disp_hist[i + 1] - disp_hist[i - 1]) / dt
                        E_kinetic = 0.5 * float(v_approx @ mass_matrix @ v_approx)
                    else:
                        E_kinetic = 0.0
                elif i == 0:
                    # 初期: 運動エネルギー = 0.5 * m * v0²
                    E_kinetic = 0.5 * float(velocity @ mass_matrix @ velocity)
                else:
                    E_kinetic = 0.0
                energy_history[i] = abs(E_strain) + abs(E_kinetic)

        # エネルギー減衰率
        E_init = energy_history[0] if len(energy_history) > 0 else 1.0
        E_final = energy_history[-1] if len(energy_history) > 0 else 0.0
        energy_decay_ratio = E_final / E_init if E_init > 1e-30 else 0.0

        # 要素応力時刻歴（最大曲げ応力）
        element_stress_history = []
        for u_snap in disp_hist:
            stress = compute_element_bending_stress(
                mesh_data.node_coords,
                mesh_data.connectivity,
                u_snap,
                cfg.E,
                wire_radius,
            )
            element_stress_history.append(stress)

        return BeamOscillationResult(
            solver_result=solver_result,
            mesh=mesh_data,
            wire_mid_node=wire_mid_node,
            n_wire_nodes=n_nodes,
            config=cfg,
            analytical_frequency_hz=f1,
            analytical_period=T1,
            initial_velocity=v0,
            time_history=time_arr,
            deflection_history=defl_arr,
            max_deflection=max_deflection,
            amplitude_ratio=amplitude_ratio,
            energy_history=energy_history,
            energy_decay_ratio=energy_decay_ratio,
            element_stress_history=element_stress_history,
        )
