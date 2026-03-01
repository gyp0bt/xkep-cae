"""撚線曲げ + サイクル変位ベンチマーク.

z軸上に配置した撚線の一端を固定し、他端にモーメントを与えて ~90° 曲げ、
曲がった状態で z 方向にサイクリック力荷重を2周期与える。

目的:
- 1000本撚線の速度ベンチマーク（仮目標: 2時間、緩和: 6時間）
- 少数素線（7/19/37本）でのプロファイリング

ロードパス:
  Phase 1: 0 → M_bend （曲げモーメント、n_bending_steps 分割）
  Phase 2: ±F_cycle を2周期（4半周期 × n_steps_per_quarter）

[← README](../../README.md)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.pair import ContactConfig, ContactManager, ContactStatus
from xkep_cae.contact.solver_hooks import (
    BenchmarkTimingCollector,
    ContactSolveResult,
    newton_raphson_with_contact,
)
from xkep_cae.elements.beam_timo3d import assemble_cr_beam3d
from xkep_cae.mesh.twisted_wire import TwistedWireMesh, make_twisted_wire_mesh
from xkep_cae.sections.beam import BeamSection

# ====================================================================
# 物理パラメータ（鋼線デフォルト）
# ====================================================================

_E = 200e9  # Pa
_NU = 0.3
_G = _E / (2.0 * (1.0 + _NU))
_WIRE_D = 0.002  # 2 mm 直径
_KAPPA = 6.0 * (1.0 + _NU) / (7.0 + 6.0 * _NU)
_NDOF_PER_NODE = 6


# ====================================================================
# 結果データクラス
# ====================================================================


@dataclass
class WireBendingBenchmarkResult:
    """撚線曲げベンチマーク結果.

    Attributes:
        n_strands: 素線本数
        n_elems: 総要素数
        n_nodes: 総節点数
        ndof: 総自由度数
        mesh_length: モデル長さ [m]
        phase1_converged: Phase 1（曲げ）収束フラグ
        phase2_converged: Phase 2（サイクル）収束フラグ
        phase1_result: Phase 1 のソルバー結果
        phase2_results: Phase 2 の各半周期のソルバー結果
        timing: 工程別タイミングデータ
        total_time_s: 総計算時間 [s]
        tip_displacement_final: 先端変位 (x, y, z) [m]
        max_penetration_ratio: 最大貫入比
        n_active_contacts: 最終活性接触ペア数
    """

    n_strands: int
    n_elems: int
    n_nodes: int
    ndof: int
    mesh_length: float
    phase1_converged: bool
    phase2_converged: bool
    phase1_result: ContactSolveResult
    phase2_results: list[ContactSolveResult] = field(default_factory=list)
    timing: BenchmarkTimingCollector | None = None
    total_time_s: float = 0.0
    tip_displacement_final: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_penetration_ratio: float = 0.0
    n_active_contacts: int = 0


# ====================================================================
# ヘルパー
# ====================================================================


def _make_cr_assemblers(mesh: TwistedWireMesh, E: float, G: float, section: BeamSection):
    """CR (corotational) 梁アセンブラを構築."""
    node_coords = mesh.node_coords
    connectivity = mesh.connectivity
    ndof_total = mesh.n_nodes * _NDOF_PER_NODE
    kappa = _KAPPA

    def assemble_tangent(u):
        K_T, _ = assemble_cr_beam3d(
            node_coords,
            connectivity,
            u,
            E,
            G,
            section.A,
            section.Iy,
            section.Iz,
            section.J,
            kappa,
            kappa,
            stiffness=True,
            internal_force=False,
        )
        return sp.csr_matrix(K_T)

    def assemble_internal_force(u):
        _, f_int = assemble_cr_beam3d(
            node_coords,
            connectivity,
            u,
            E,
            G,
            section.A,
            section.Iy,
            section.Iz,
            section.J,
            kappa,
            kappa,
            stiffness=False,
            internal_force=True,
        )
        return f_int

    return assemble_tangent, assemble_internal_force, ndof_total


def _fix_strand_starts(mesh: TwistedWireMesh) -> np.ndarray:
    """全素線の開始端（z=0）を全拘束する DOF セットを返す."""
    fixed = set()
    for sid in range(mesh.n_strands):
        nodes = mesh.strand_nodes(sid)
        node = nodes[0]
        for d in range(_NDOF_PER_NODE):
            fixed.add(_NDOF_PER_NODE * node + d)
    return np.array(sorted(fixed), dtype=int)


def _get_strand_end_dofs(mesh: TwistedWireMesh, strand_id: int, end: str) -> np.ndarray:
    """素線の端点の DOF インデックスを取得."""
    nodes = mesh.strand_nodes(strand_id)
    node = nodes[0] if end == "start" else nodes[-1]
    return np.array([_NDOF_PER_NODE * node + d for d in range(_NDOF_PER_NODE)], dtype=int)


def _max_penetration_ratio(mgr: ContactManager) -> float:
    """最大貫入比を計算."""
    max_pen = 0.0
    for p in mgr.pairs:
        if p.state.status == ContactStatus.INACTIVE:
            continue
        if p.state.gap < 0:
            pen = abs(p.state.gap) / (p.radius_a + p.radius_b)
            if pen > max_pen:
                max_pen = pen
    return max_pen


def _count_active_pairs(mgr: ContactManager) -> int:
    """有効な接触ペア数をカウント."""
    return sum(1 for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE)


def _build_contact_manager(
    mesh: TwistedWireMesh,
    section: BeamSection,
    *,
    n_outer_max: int = 5,
    auto_kpen: bool = True,
    use_friction: bool = False,
    mu: float = 0.0,
) -> ContactManager:
    """接触マネージャを構築."""
    elem_layer_map = mesh.build_elem_layer_map()
    kpen_mode = "beam_ei" if auto_kpen else "manual"
    kpen_scale = 0.1 if auto_kpen else 1e5

    return ContactManager(
        config=ContactConfig(
            k_pen_scale=kpen_scale,
            k_pen_mode=kpen_mode,
            beam_E=_E if auto_kpen else 0.0,
            beam_I=section.Iy if auto_kpen else 0.0,
            k_t_ratio=0.1,
            mu=mu,
            g_on=0.0,
            g_off=1e-5,
            n_outer_max=n_outer_max,
            use_friction=use_friction,
            use_line_search=True,
            line_search_max_steps=5,
            use_geometric_stiffness=True,
            tol_penetration_ratio=0.02,
            penalty_growth_factor=2.0,
            k_pen_max=1e12,
            elem_layer_map=elem_layer_map,
            exclude_same_layer=True,
            midpoint_prescreening=True,
            linear_solver="auto",
        ),
    )


# ====================================================================
# メイン: 撚線曲げ + サイクル変位ベンチマーク
# ====================================================================


def run_wire_bending_benchmark(
    n_strands: int = 7,
    *,
    wire_diameter: float = _WIRE_D,
    pitch: float = 0.040,
    n_elems_per_strand: int = 8,
    n_pitches: float = 1.0,
    # 曲げパラメータ
    bend_angle_deg: float = 90.0,
    n_bending_steps: int = 20,
    # サイクルパラメータ
    cyclic_amplitude_mm: float = 5.0,
    n_cycles: int = 2,
    n_steps_per_quarter: int = 5,
    # NR パラメータ
    max_iter: int = 30,
    n_outer_max: int = 5,
    tol_force: float = 1e-6,
    # 接触パラメータ
    auto_kpen: bool = True,
    use_friction: bool = False,
    mu: float = 0.0,
    show_progress: bool = True,
) -> WireBendingBenchmarkResult:
    """撚線曲げ + サイクル変位ベンチマークを実行.

    ロードパス:
      Phase 1: z=0端を固定、z=L端にモーメント Mx を n_bending_steps で負荷
               → 曲げ角度 ~bend_angle_deg°
      Phase 2: 曲がった状態から z 方向にサイクリック力 ±F を2周期

    Args:
        n_strands: 素線本数（3, 7, 19, 37, 61, 91, ...）
        wire_diameter: 素線直径 [m]
        pitch: 撚ピッチ [m]
        n_elems_per_strand: 素線あたり要素数
        n_pitches: ピッチ数
        bend_angle_deg: 目標曲げ角度 [°]
        n_bending_steps: 曲げ荷重ステップ数
        cyclic_amplitude_mm: サイクル変位振幅 [mm]
        n_cycles: サイクル数
        n_steps_per_quarter: 1/4 周期あたりのステップ数
        max_iter: NR 最大反復数
        n_outer_max: Outer loop 最大反復数
        tol_force: 力残差収束判定値
        auto_kpen: ペナルティ剛性自動推定
        use_friction: 摩擦を有効にするか
        mu: 摩擦係数
        show_progress: 進捗表示

    Returns:
        WireBendingBenchmarkResult
    """
    t_total_start = time.perf_counter()
    timing = BenchmarkTimingCollector()

    # ------------------------------------------------------------------
    # 1. メッシュ生成
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    mesh = make_twisted_wire_mesh(
        n_strands,
        wire_diameter,
        pitch,
        length=0.0,
        n_elems_per_strand=n_elems_per_strand,
        n_pitches=n_pitches,
    )
    timing.record(0, 0, -1, "mesh_generation", time.perf_counter() - t0)

    section = BeamSection.circle(wire_diameter)
    ndof_total = mesh.n_nodes * _NDOF_PER_NODE
    L = mesh.length

    if show_progress:
        print(f"\n{'=' * 70}")
        print(f"  撚線曲げベンチマーク: {n_strands}本撚線")
        print(f"{'=' * 70}")
        print(f"  要素数:     {mesh.n_elems}")
        print(f"  節点数:     {mesh.n_nodes}")
        print(f"  自由度数:   {ndof_total}")
        print(f"  モデル長さ: {L * 1000:.1f} mm")
        print(f"  目標曲げ角: {bend_angle_deg}°")
        print(f"  サイクル振幅: ±{cyclic_amplitude_mm} mm, {n_cycles}周期")

    # ------------------------------------------------------------------
    # 2. CR 梁アセンブラ構築
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    assemble_tangent, assemble_internal_force, ndof = _make_cr_assemblers(mesh, _E, _G, section)
    timing.record(0, 0, -1, "assembler_setup", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # 3. 境界条件 — z=0 端を全固定
    # ------------------------------------------------------------------
    fixed_dofs = _fix_strand_starts(mesh)

    # ------------------------------------------------------------------
    # 4. 接触マネージャ
    # ------------------------------------------------------------------
    mgr = _build_contact_manager(
        mesh,
        section,
        n_outer_max=n_outer_max,
        auto_kpen=auto_kpen,
        use_friction=use_friction,
        mu=mu,
    )

    # ------------------------------------------------------------------
    # Phase 1: 曲げ（モーメント荷重）
    # ------------------------------------------------------------------
    # M = E*I*θ/L（小変形の目安; 大変形では NR で自動調整）
    bend_angle_rad = np.deg2rad(bend_angle_deg)
    M_per_strand = _E * section.Iy * bend_angle_rad / L

    f_ext_bend = np.zeros(ndof)
    for sid in range(mesh.n_strands):
        end_dofs = _get_strand_end_dofs(mesh, sid, "end")
        # DOF 3 = 回転 rx（x 軸まわりモーメント）
        f_ext_bend[end_dofs[3]] = M_per_strand

    if show_progress:
        print(f"\n--- Phase 1: 曲げ（M={M_per_strand:.4e} N·m/strand, {n_bending_steps} steps）---")

    result_bend = newton_raphson_with_contact(
        f_ext_bend,
        fixed_dofs,
        assemble_tangent,
        assemble_internal_force,
        mgr,
        mesh.node_coords,
        mesh.connectivity,
        mesh.radii,
        n_load_steps=n_bending_steps,
        max_iter=max_iter,
        tol_force=tol_force,
        show_progress=show_progress,
        broadphase_margin=0.01,
        timing=timing,
    )

    u_after_bend = result_bend.u.copy()

    if show_progress:
        # 先端変位を表示
        tip_node = mesh.strand_nodes(0)[-1]
        tip_disp = u_after_bend[_NDOF_PER_NODE * tip_node : _NDOF_PER_NODE * tip_node + 3]
        print(
            f"  Phase 1 完了: converged={result_bend.converged}, "
            f"NR={result_bend.total_newton_iterations}"
        )
        print(
            f"  先端変位: dx={tip_disp[0] * 1000:.3f} mm, "
            f"dy={tip_disp[1] * 1000:.3f} mm, dz={tip_disp[2] * 1000:.3f} mm"
        )

    # ------------------------------------------------------------------
    # Phase 2: サイクリック z 方向力荷重
    # ------------------------------------------------------------------
    # F ≈ 3EI·δ/L³（片持ち梁たわみ; 曲がった状態での目安）
    delta_m = cyclic_amplitude_mm * 1e-3
    F_total = 3.0 * _E * section.Iy * delta_m / L**3
    F_per_strand = F_total / n_strands

    # 単位荷重ベクトル: 全素線の z=L 端に z 方向力
    f_ext_cycle_unit = np.zeros(ndof)
    for sid in range(mesh.n_strands):
        end_dofs = _get_strand_end_dofs(mesh, sid, "end")
        f_ext_cycle_unit[end_dofs[2]] = F_per_strand  # DOF 2 = z 並進

    # サイクリック荷重: 0 → +1 → -1 → +1 → -1（2周期）
    # 各半周期は n_steps_per_quarter ステップ
    phase2_results: list[ContactSolveResult] = []
    u = u_after_bend.copy()
    current_amp = 0.0
    phase2_all_converged = True

    # 半周期の振幅列: +1, -1, +1, -1 (2周期 = 4半周期)
    half_cycle_amps: list[float] = []
    for _ in range(n_cycles):
        half_cycle_amps.extend([1.0, -1.0])

    for hc_idx, target_amp in enumerate(half_cycle_amps):
        delta_amp = target_amp - current_amp
        # f_ext_total: 増分荷重（サイクリック分のみ）
        f_ext_total_hc = delta_amp * f_ext_cycle_unit
        # f_ext_base: ベース荷重（曲げモーメント + 現在のサイクリック力）
        f_ext_base_hc = f_ext_bend + current_amp * f_ext_cycle_unit

        if show_progress:
            print(
                f"\n--- Phase 2-{hc_idx + 1}: サイクル amp "
                f"{current_amp:.1f} → {target_amp:.1f} "
                f"({n_steps_per_quarter} steps) ---"
            )

        result_hc = newton_raphson_with_contact(
            f_ext_total_hc,
            fixed_dofs,
            assemble_tangent,
            assemble_internal_force,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=n_steps_per_quarter,
            max_iter=max_iter,
            tol_force=tol_force,
            show_progress=show_progress,
            u0=u,
            broadphase_margin=0.01,
            f_ext_base=f_ext_base_hc,
            timing=timing,
        )

        u = result_hc.u.copy()
        phase2_results.append(result_hc)
        if not result_hc.converged:
            phase2_all_converged = False
        current_amp = target_amp

        if show_progress:
            tip_node = mesh.strand_nodes(0)[-1]
            tip_disp = u[_NDOF_PER_NODE * tip_node : _NDOF_PER_NODE * tip_node + 3]
            print(
                f"  半周期{hc_idx + 1} 完了: converged={result_hc.converged}, "
                f"NR={result_hc.total_newton_iterations}"
            )
            print(f"  先端変位: dz={tip_disp[2] * 1000:.3f} mm")

    # ------------------------------------------------------------------
    # 結果集約
    # ------------------------------------------------------------------
    total_time = time.perf_counter() - t_total_start

    tip_node = mesh.strand_nodes(0)[-1]
    tip_disp = u[_NDOF_PER_NODE * tip_node : _NDOF_PER_NODE * tip_node + 3]

    max_pen = _max_penetration_ratio(mgr)
    n_active = _count_active_pairs(mgr)

    if show_progress:
        print(f"\n{'=' * 70}")
        print(f"  ベンチマーク完了: {total_time:.2f} s")
        print(f"{'=' * 70}")
        print(f"  Phase 1 収束: {result_bend.converged}")
        print(f"  Phase 2 収束: {phase2_all_converged}")
        print(f"  活性接触ペア: {n_active}")
        print(f"  最大貫入比:   {max_pen:.6f}")
        print(f"  総計算時間:   {total_time:.2f} s")

        if timing is not None:
            print(f"\n{timing.summary_table()}")

    return WireBendingBenchmarkResult(
        n_strands=n_strands,
        n_elems=mesh.n_elems,
        n_nodes=mesh.n_nodes,
        ndof=ndof_total,
        mesh_length=L,
        phase1_converged=result_bend.converged,
        phase2_converged=phase2_all_converged,
        phase1_result=result_bend,
        phase2_results=phase2_results,
        timing=timing,
        total_time_s=total_time,
        tip_displacement_final=(float(tip_disp[0]), float(tip_disp[1]), float(tip_disp[2])),
        max_penetration_ratio=max_pen,
        n_active_contacts=n_active,
    )


# ====================================================================
# レポート出力
# ====================================================================


def print_benchmark_report(result: WireBendingBenchmarkResult) -> str:
    """ベンチマーク結果のフォーマット済みレポートを返す."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"  撚線曲げベンチマーク結果: {result.n_strands}本")
    lines.append(f"{'=' * 70}")
    lines.append(f"  要素数:       {result.n_elems}")
    lines.append(f"  節点数:       {result.n_nodes}")
    lines.append(f"  自由度数:     {result.ndof}")
    lines.append(f"  モデル長さ:   {result.mesh_length * 1000:.1f} mm")
    lines.append(f"  Phase 1 収束: {result.phase1_converged}")
    lines.append(f"  Phase 2 収束: {result.phase2_converged}")
    lines.append(f"  活性接触:     {result.n_active_contacts}")
    lines.append(f"  最大貫入比:   {result.max_penetration_ratio:.6f}")
    lines.append(f"  総計算時間:   {result.total_time_s:.2f} s")

    dx, dy, dz = result.tip_displacement_final
    lines.append(
        f"  先端変位:     dx={dx * 1000:.3f} mm, dy={dy * 1000:.3f} mm, dz={dz * 1000:.3f} mm"
    )

    # Phase 2 詳細
    total_nr = result.phase1_result.total_newton_iterations
    for r in result.phase2_results:
        total_nr += r.total_newton_iterations
    lines.append(f"  NR反復合計:   {total_nr}")

    if result.timing is not None:
        lines.append("")
        lines.append(result.timing.summary_table())

    report = "\n".join(lines)
    return report


# ====================================================================
# スケーリング: 複数素線本数でのベンチマーク
# ====================================================================


def run_scaling_benchmark(
    strand_counts: list[int] | None = None,
    **kwargs,
) -> list[WireBendingBenchmarkResult]:
    """複数の素線本数で曲げベンチマークを実行しスケーリングを分析.

    Args:
        strand_counts: 素線本数リスト（デフォルト [7, 19, 37]）
        **kwargs: run_wire_bending_benchmark に渡す追加引数

    Returns:
        list of WireBendingBenchmarkResult
    """
    if strand_counts is None:
        strand_counts = [7, 19, 37]

    results = []
    for n in strand_counts:
        result = run_wire_bending_benchmark(n_strands=n, **kwargs)
        results.append(result)

    # スケーリングレポート
    print(f"\n{'=' * 80}")
    print("  曲げベンチマーク スケーリングレポート")
    print(f"{'=' * 80}")
    header = (
        f"{'n_strands':>10} {'n_elems':>8} {'ndof':>8} "
        f"{'P1_conv':>8} {'P2_conv':>8} {'active':>7} "
        f"{'pen_ratio':>10} {'time(s)':>10}"
    )
    print(header)
    print("-" * 80)
    for r in results:
        print(
            f"{r.n_strands:>10} {r.n_elems:>8} {r.ndof:>8} "
            f"{'Y' if r.phase1_converged else 'N':>8} "
            f"{'Y' if r.phase2_converged else 'N':>8} "
            f"{r.n_active_contacts:>7} "
            f"{r.max_penetration_ratio:>10.6f} "
            f"{r.total_time_s:>10.2f}"
        )
    print()

    return results
