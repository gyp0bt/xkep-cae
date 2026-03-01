"""Phase S3: 撚線接触NR収束ベンチマーク + 工程別処理時間計測.

撚線モデル（7→19→37→61→91本）に対して接触NRソルバーを実行し、
収束性と各工程の処理時間を計測する。ボトルネック特定と
スケーラビリティ分析が目的。

計測対象の工程:
  - broadphase: AABB候補検出
  - geometry_update: 最近接点更新
  - friction_mapping: 摩擦 return mapping
  - structural_internal_force: 構造内力計算
  - contact_force: 接触内力計算
  - structural_tangent: 構造接線剛性組み立て
  - contact_stiffness: 接触接線剛性組み立て
  - bc_apply: 境界条件適用
  - linear_solve: 線形ソルバー
  - line_search: merit line search
  - outer_convergence_check: Outer収束判定 + AL更新
"""

import time

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactStatus,
)
from xkep_cae.contact.solver_hooks import (
    BenchmarkTimingCollector,
    newton_raphson_with_contact,
)
from xkep_cae.elements.beam_timo3d import (
    assemble_cr_beam3d,
    timo_beam3d_ke_global,
)
from xkep_cae.mesh.twisted_wire import (
    TwistedWireMesh,
    make_twisted_wire_mesh,
)
from xkep_cae.sections.beam import BeamSection

pytestmark = pytest.mark.slow

# ====================================================================
# 共通パラメータ
# ====================================================================

_NDOF_PER_NODE = 6

# 鋼線パラメータ
_E = 200e9  # Pa
_NU = 0.3
_G = _E / (2.0 * (1.0 + _NU))
_WIRE_D = 0.002  # 直径 2mm
_SECTION = BeamSection.circle(_WIRE_D)
_KAPPA = 6.0 * (1.0 + _NU) / (7.0 + 6.0 * _NU)

_PITCH = 0.040  # 40mm ピッチ
_N_ELEM_PER_STRAND = 8  # ベンチマーク用（軽量化）


# ====================================================================
# ヘルパー: アセンブラ構築
# ====================================================================


def _make_timo3d_assemblers(mesh: TwistedWireMesh):
    """Timoshenko 3D線形梁のアセンブリコールバックを構築."""
    node_coords = mesh.node_coords
    connectivity = mesh.connectivity
    ndof_total = mesh.n_nodes * _NDOF_PER_NODE

    K = np.zeros((ndof_total, ndof_total))
    for elem in connectivity:
        n1, n2 = int(elem[0]), int(elem[1])
        coords = node_coords[np.array([n1, n2])]
        Ke = timo_beam3d_ke_global(
            coords,
            _E,
            _G,
            _SECTION.A,
            _SECTION.Iy,
            _SECTION.Iz,
            _SECTION.J,
            _KAPPA,
            _KAPPA,
        )
        edofs = np.array(
            [6 * n1 + d for d in range(6)] + [6 * n2 + d for d in range(6)],
            dtype=int,
        )
        K[np.ix_(edofs, edofs)] += Ke
    K_sp = sp.csr_matrix(K)

    def assemble_tangent(u):
        return K_sp

    def assemble_internal_force(u):
        return K_sp.dot(u)

    return assemble_tangent, assemble_internal_force, ndof_total


def _make_cr_assemblers(mesh: TwistedWireMesh):
    """CR梁のアセンブリコールバックを構築."""
    node_coords = mesh.node_coords
    connectivity = mesh.connectivity
    ndof_total = mesh.n_nodes * _NDOF_PER_NODE

    def assemble_tangent(u):
        K_T, _ = assemble_cr_beam3d(
            node_coords,
            connectivity,
            u,
            _E,
            _G,
            _SECTION.A,
            _SECTION.Iy,
            _SECTION.Iz,
            _SECTION.J,
            _KAPPA,
            _KAPPA,
            stiffness=True,
            internal_force=False,
        )
        return K_T

    def assemble_internal_force(u):
        _, f_int = assemble_cr_beam3d(
            node_coords,
            connectivity,
            u,
            _E,
            _G,
            _SECTION.A,
            _SECTION.Iy,
            _SECTION.Iz,
            _SECTION.J,
            _KAPPA,
            _KAPPA,
            stiffness=False,
            internal_force=True,
        )
        return f_int

    return assemble_tangent, assemble_internal_force, ndof_total


# ====================================================================
# ヘルパー: 境界条件・荷重
# ====================================================================


def _get_strand_end_dofs(mesh: TwistedWireMesh, strand_id: int, end: str):
    """素線の端点のDOFインデックスを取得."""
    nodes = mesh.strand_nodes(strand_id)
    node = nodes[0] if end == "start" else nodes[-1]
    return np.array([_NDOF_PER_NODE * node + d for d in range(_NDOF_PER_NODE)], dtype=int)


def _fix_all_strand_starts(mesh: TwistedWireMesh) -> np.ndarray:
    """全素線の開始端を全固定するDOFセットを返す."""
    fixed = set()
    for sid in range(mesh.n_strands):
        dofs = _get_strand_end_dofs(mesh, sid, "start")
        fixed.update(dofs.tolist())
    return np.array(sorted(fixed), dtype=int)


def _count_active_pairs(mgr: ContactManager) -> int:
    """有効な接触ペア数をカウント."""
    return sum(1 for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE)


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


# ====================================================================
# ベンチマーク実行: 撚線接触NRソルバー + タイミング計測
# ====================================================================


def _run_benchmark(
    n_strands: int,
    load_type: str = "tension",
    load_value: float = 100.0,
    *,
    assembler_type: str = "timo3d",
    n_load_steps: int = 5,
    max_iter: int = 30,
    n_outer_max: int = 5,
    n_elems_per_strand: int = _N_ELEM_PER_STRAND,
    auto_kpen: bool = True,
):
    """撚線ベンチマークを実行し、タイミングデータを返す.

    Returns:
        (result, mgr, mesh, timing, setup_time_s)
    """
    # --- メッシュ生成 ---
    t0_mesh = time.perf_counter()
    mesh = make_twisted_wire_mesh(
        n_strands,
        _WIRE_D,
        _PITCH,
        length=0.0,
        n_elems_per_strand=n_elems_per_strand,
        n_pitches=1.0,
    )
    mesh_time = time.perf_counter() - t0_mesh

    # --- アセンブラ構築 ---
    t0_setup = time.perf_counter()
    if assembler_type == "cr":
        assemble_tangent, assemble_internal_force, ndof_total = _make_cr_assemblers(mesh)
    else:
        assemble_tangent, assemble_internal_force, ndof_total = _make_timo3d_assemblers(mesh)
    setup_time = time.perf_counter() - t0_setup

    # --- 境界条件 ---
    fixed_dofs = _fix_all_strand_starts(mesh)

    # --- 外力ベクトル ---
    f_ext = np.zeros(ndof_total)
    if load_type == "tension":
        f_per_strand = load_value / n_strands
        for sid in range(mesh.n_strands):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext[end_dofs[2]] = f_per_strand
    elif load_type == "bending":
        m_per_strand = load_value / n_strands
        for sid in range(mesh.n_strands):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext[end_dofs[4]] = m_per_strand

    # --- 接触マネージャ ---
    elem_layer_map = mesh.build_elem_layer_map()
    kpen_mode = "manual"
    beam_E = 0.0
    beam_I = 0.0
    kpen_scale = 1e5
    if auto_kpen:
        kpen_mode = "beam_ei"
        beam_E = _E
        beam_I = _SECTION.Iy
        kpen_scale = 0.1

    mgr = ContactManager(
        config=ContactConfig(
            k_pen_scale=kpen_scale,
            k_pen_mode=kpen_mode,
            beam_E=beam_E,
            beam_I=beam_I,
            k_t_ratio=0.1,
            mu=0.0,
            g_on=0.0,
            g_off=1e-5,
            n_outer_max=n_outer_max,
            use_friction=False,
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

    # --- NRソルバー実行 + タイミング計測 ---
    timing = BenchmarkTimingCollector()
    timing.record(0, 0, -1, "mesh_generation", mesh_time)
    timing.record(0, 0, -1, "assembler_setup", setup_time)

    result = newton_raphson_with_contact(
        f_ext,
        fixed_dofs,
        assemble_tangent,
        assemble_internal_force,
        mgr,
        mesh.node_coords,
        mesh.connectivity,
        mesh.radii,
        n_load_steps=n_load_steps,
        max_iter=max_iter,
        show_progress=False,
        broadphase_margin=0.01,
        timing=timing,
    )

    return result, mgr, mesh, timing


def _print_benchmark_report(
    n_strands: int,
    result,
    mgr: ContactManager,
    mesh: TwistedWireMesh,
    timing: BenchmarkTimingCollector,
):
    """ベンチマーク結果レポートを表示."""
    n_active = _count_active_pairs(mgr)
    max_pen = _max_penetration_ratio(mgr)

    print(f"\n{'=' * 70}")
    print(f"  撚線ベンチマーク: {n_strands}本")
    print(f"{'=' * 70}")
    print(f"  素線数:     {n_strands}")
    print(f"  要素数:     {mesh.n_elems}")
    print(f"  節点数:     {mesh.n_nodes}")
    print(f"  自由度数:   {mesh.n_nodes * _NDOF_PER_NODE}")
    print(f"  収束:       {'Yes' if result.converged else 'No'}")
    print(f"  荷重ステップ: {result.n_load_steps}")
    print(f"  NR反復合計: {result.total_newton_iterations}")
    print(f"  Outer合計:  {result.total_outer_iterations}")
    print(f"  活性ペア:   {n_active}")
    print(f"  最大貫入比: {max_pen:.6f}")
    print()

    # 工程別タイミング
    print(timing.summary_table())

    # ステップ別時間
    step_t = timing.step_times()
    if step_t:
        print(f"\n{'Step':>6} {'Time(s)':>10}")
        print("-" * 18)
        for s in sorted(step_t):
            print(f"{s:>6} {step_t[s]:>10.4f}")

    print()


# ====================================================================
# テストクラス
# ====================================================================


class TestBenchmarkTiming:
    """撚線接触NR収束 + 工程別処理時間ベンチマーク."""

    @pytest.mark.parametrize("n_strands", [7, 19])
    def test_small_strand_timing(self, n_strands):
        """小規模（7/19本）のタイミング計測（収束は報告のみ）."""
        result, mgr, mesh, timing = _run_benchmark(
            n_strands,
            load_type="tension",
            load_value=100.0,
            n_load_steps=5,
            max_iter=30,
        )

        _print_benchmark_report(n_strands, result, mgr, mesh, timing)

        # タイミングデータの検証（収束は保証しない - 接触問題では発散もありうる）
        assert timing.total_time() > 0, "タイミングデータが記録されていない"
        totals = timing.phase_totals()
        assert len(totals) > 0, "工程別データが空"

    @pytest.mark.parametrize("n_strands", [37, 61])
    def test_medium_strand_timing(self, n_strands):
        """中規模（37/61本）の収束性 + タイミング計測."""
        result, mgr, mesh, timing = _run_benchmark(
            n_strands,
            load_type="tension",
            load_value=100.0,
            n_load_steps=3,
            max_iter=30,
        )

        _print_benchmark_report(n_strands, result, mgr, mesh, timing)

        # タイミングデータの記録を検証（収束は保証しない）
        assert timing.total_time() > 0
        totals = timing.phase_totals()
        assert "broadphase" in totals or "structural_tangent" in totals

    def test_91_strand_timing(self):
        """大規模（91本）の収束性 + タイミング計測."""
        result, mgr, mesh, timing = _run_benchmark(
            91,
            load_type="tension",
            load_value=100.0,
            n_load_steps=2,
            max_iter=20,
            n_outer_max=3,
        )

        _print_benchmark_report(91, result, mgr, mesh, timing)

        # タイミングデータの記録のみ検証
        assert timing.total_time() > 0

    def test_timing_collector_phases(self):
        """BenchmarkTimingCollector が全主要工程を記録することを検証."""
        result, mgr, mesh, timing = _run_benchmark(
            7,
            load_type="tension",
            load_value=50.0,
            n_load_steps=3,
            max_iter=20,
        )

        totals = timing.phase_totals()
        counts = timing.phase_counts()

        # 主要工程が記録されていること
        expected_phases = {
            "mesh_generation",
            "assembler_setup",
            "broadphase",
            "geometry_update",
            "structural_internal_force",
            "contact_force",
            "structural_tangent",
            "bc_apply",
            "linear_solve",
        }

        recorded_phases = set(totals.keys())
        missing = expected_phases - recorded_phases
        assert not missing, f"未記録の工程: {missing}"

        # 各工程が正の時間を持つこと
        for phase in expected_phases:
            assert totals[phase] > 0, f"{phase} の合計時間が 0"
            assert counts[phase] > 0, f"{phase} の呼び出し回数が 0"

        # summary_table が正常生成されること
        table = timing.summary_table()
        assert "TOTAL" in table
        assert "broadphase" in table

    def test_step_times(self):
        """ステップ別タイミングが正しく集計されること."""
        result, mgr, mesh, timing = _run_benchmark(
            7,
            load_type="tension",
            load_value=50.0,
            n_load_steps=3,
        )

        step_t = timing.step_times()
        # mesh_generation と assembler_setup は step=0 で記録
        assert 0 in step_t, "step=0（セットアップ）が含まれていない"
        # 荷重ステップが含まれること
        assert len(step_t) >= 2, f"ステップ数が少ない: {len(step_t)}"

    def test_bending_load(self):
        """曲げ荷重での収束性 + タイミング計測."""
        result, mgr, mesh, timing = _run_benchmark(
            7,
            load_type="bending",
            load_value=0.01,
            n_load_steps=3,
        )

        _print_benchmark_report(7, result, mgr, mesh, timing)
        assert timing.total_time() > 0


class TestBenchmarkScaling:
    """撚線本数に対する処理時間スケーリング分析."""

    def test_timing_scaling_report(self):
        """7→19→37本のスケーリングレポート."""
        results = {}
        for n_strands in [7, 19, 37]:
            result, mgr, mesh, timing = _run_benchmark(
                n_strands,
                load_type="tension",
                load_value=100.0,
                n_load_steps=3,
                max_iter=20,
            )
            n_active = _count_active_pairs(mgr)
            results[n_strands] = {
                "n_elems": mesh.n_elems,
                "ndof": mesh.n_nodes * _NDOF_PER_NODE,
                "converged": result.converged,
                "n_newton": result.total_newton_iterations,
                "n_outer": result.total_outer_iterations,
                "n_active": n_active,
                "total_time_s": timing.total_time(),
                "phase_totals": timing.phase_totals(),
            }

        # スケーリングレポート表示
        print(f"\n{'=' * 80}")
        print("  撚線スケーリングレポート")
        print(f"{'=' * 80}")
        header = (
            f"{'n_strands':>10} {'n_elems':>8} {'ndof':>8} {'conv':>5} "
            f"{'n_NR':>6} {'n_outer':>8} {'active':>7} {'total(s)':>10}"
        )
        print(header)
        print("-" * 75)
        for ns, r in results.items():
            print(
                f"{ns:>10} {r['n_elems']:>8} {r['ndof']:>8} "
                f"{'Y' if r['converged'] else 'N':>5} "
                f"{r['n_newton']:>6} {r['n_outer']:>8} "
                f"{r['n_active']:>7} {r['total_time_s']:>10.4f}"
            )

        # 工程別スケーリング
        all_phases = set()
        for r in results.values():
            all_phases.update(r["phase_totals"].keys())
        all_phases = sorted(all_phases)

        print(f"\n{'Phase':<30}", end="")
        for ns in results:
            print(f" {ns:>10}本", end="")
        print()
        print("-" * (30 + 12 * len(results)))
        for phase in all_phases:
            print(f"{phase:<30}", end="")
            for ns in results:
                t = results[ns]["phase_totals"].get(phase, 0.0)
                print(f" {t:>10.4f}s", end="")
            print()
        print()

        # 基本検証
        for ns, r in results.items():
            assert r["total_time_s"] > 0, f"{ns}本: タイミングデータが空"
