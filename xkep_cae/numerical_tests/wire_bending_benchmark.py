"""撚線曲げ揺動ベンチマーク.

z軸上に配置した撚線の一端を固定し、他端にモーメントを与えて ~90° 曲げ、
曲がった状態で z 方向にサイクル変位を2周期与える。

名称:
  曲げ揺動（まげようどう）= 曲げ + サイクル変位（揺動）

目的:
- 1000本撚線の速度ベンチマーク（仮目標: 2時間、緩和: 6時間）
- 少数素線（7/19/37本）でのプロファイリング

ロードパス:
  Phase 1（曲げ）: 0 → M_bend（モーメント荷重、n_bending_steps 分割）
  Phase 2（揺動）: z方向サイクル変位 ±Δz を n_cycles 周期
    - 変位制御: 自由端の z-DOF を拘束し、正弦波で処方
    - 曲げモーメントは外力として維持

GIF出力:
  各撚線種類について経時変化GIFを保存（matplotlib + PIL）。

[← README](../../README.md)
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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
class BendingOscillationResult:
    """曲げ揺動ベンチマーク結果.

    Attributes:
        n_strands: 素線本数
        n_elems: 総要素数
        n_nodes: 総節点数
        ndof: 総自由度数
        mesh_length: モデル長さ [m]
        phase1_converged: Phase 1（曲げ）収束フラグ
        phase2_converged: Phase 2（揺動）収束フラグ
        phase1_result: Phase 1 のソルバー結果
        phase2_results: Phase 2 の各変位ステップのソルバー結果
        timing: 工程別タイミングデータ
        total_time_s: 総計算時間 [s]
        tip_displacement_final: 先端変位 (x, y, z) [m]
        max_penetration_ratio: 最大貫入比
        n_active_contacts: 最終活性接触ペア数
        displacement_snapshots: GIF用の各ステップ変形座標
        snapshot_labels: 各スナップショットのラベル
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
    displacement_snapshots: list[np.ndarray] = field(default_factory=list)
    snapshot_labels: list[str] = field(default_factory=list)


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
        return K_T

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


def _deformed_coords(node_coords_ref: np.ndarray, u: np.ndarray) -> np.ndarray:
    """変形座標を計算 (n_nodes, 3)."""
    n_nodes = node_coords_ref.shape[0]
    coords = node_coords_ref.copy()
    for i in range(n_nodes):
        coords[i, 0] += u[_NDOF_PER_NODE * i]
        coords[i, 1] += u[_NDOF_PER_NODE * i + 1]
        coords[i, 2] += u[_NDOF_PER_NODE * i + 2]
    return coords


# ====================================================================
# GIF 出力
# ====================================================================


def export_bending_oscillation_gif(
    mesh: TwistedWireMesh,
    displacement_snapshots: list[np.ndarray],
    snapshot_labels: list[str],
    output_dir: str | Path,
    *,
    prefix: str = "bending_oscillation",
    views: list[str] | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
    dpi: int = 80,
    duration: int = 200,
) -> list[Path]:
    """曲げ揺動の経時変化GIFを生成.

    Args:
        mesh: 撚線メッシュ
        displacement_snapshots: 各ステップの変位ベクトル (ndof,)
        snapshot_labels: 各ステップのラベル
        output_dir: 出力ディレクトリ
        prefix: ファイル名プレフィックス
        views: 描画ビュー方向 (デフォルト ["yz", "xz"])
        figsize: 図サイズ
        dpi: 解像度
        duration: フレーム間ミリ秒

    Returns:
        生成されたGIFファイルパスのリスト
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage
    except ImportError:
        return []

    if views is None:
        views = ["yz", "xz"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _VIEW_AXES = {
        "xy": (0, 1, "X [mm]", "Y [mm]"),
        "xz": (0, 2, "X [mm]", "Z [mm]"),
        "yz": (1, 2, "Y [mm]", "Z [mm]"),
    }

    # 素線ごとの色パレット
    _COLORS = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    node_coords_ref = mesh.node_coords
    connectivity = mesh.connectivity

    # 全フレームの変形座標を収集
    deformed_frames = []
    for u in displacement_snapshots:
        coords_def = _deformed_coords(node_coords_ref, u)
        deformed_frames.append(coords_def * 1000.0)  # m → mm

    output_files: list[Path] = []

    for view in views:
        if view not in _VIEW_AXES:
            continue
        ax_h, ax_v, label_h, label_v = _VIEW_AXES[view]

        # 全フレームの描画範囲を計算
        all_h: list[float] = []
        all_v: list[float] = []
        for coords in deformed_frames:
            for elem in connectivity:
                n1, n2 = int(elem[0]), int(elem[1])
                all_h.extend([coords[n1, ax_h], coords[n2, ax_h]])
                all_v.extend([coords[n1, ax_v], coords[n2, ax_v]])

        if not all_h:
            continue

        h_min, h_max = min(all_h), max(all_h)
        v_min, v_max = min(all_v), max(all_v)
        h_range = max(h_max - h_min, 0.1)
        v_range = max(v_max - v_min, 0.1)
        margin = 0.1
        xlim = (h_min - h_range * margin, h_max + h_range * margin)
        ylim = (v_min - v_range * margin, v_max + v_range * margin)

        # 素線→要素のマッピング
        strand_elems: dict[int, list[int]] = {}
        for sid in range(mesh.n_strands):
            strand_elems[sid] = mesh.strand_elems(sid).tolist()

        # 各フレームを描画
        pil_images: list[PILImage.Image] = []
        for frame_idx, (coords, label) in enumerate(
            zip(deformed_frames, snapshot_labels, strict=True)
        ):
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

            for sid in range(mesh.n_strands):
                color = _COLORS[sid % len(_COLORS)]
                for ei, elem_idx in enumerate(strand_elems[sid]):
                    elem = connectivity[elem_idx]
                    n1, n2 = int(elem[0]), int(elem[1])
                    ax.plot(
                        [coords[n1, ax_h], coords[n2, ax_h]],
                        [coords[n1, ax_v], coords[n2, ax_v]],
                        color=color,
                        linewidth=1.5,
                        label=f"strand {sid}" if ei == 0 and frame_idx == 0 else None,
                    )

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(label_h)
            ax.set_ylabel(label_v)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{label} — {view.upper()}")
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img = PILImage.open(buf).convert("RGB")
            pil_images.append(img)

        if not pil_images:
            continue

        gif_path = out_path / f"{prefix}_{mesh.n_strands}strand_{view}.gif"
        if len(pil_images) == 1:
            pil_images[0].save(gif_path, format="GIF")
        else:
            pil_images[0].save(
                gif_path,
                format="GIF",
                save_all=True,
                append_images=pil_images[1:],
                duration=duration,
                loop=0,
            )
        output_files.append(gif_path)

    return output_files


# ====================================================================
# メイン: 曲げ揺動ベンチマーク
# ====================================================================


def run_bending_oscillation(
    n_strands: int = 7,
    *,
    wire_diameter: float = _WIRE_D,
    pitch: float = 0.040,
    n_elems_per_strand: int = 8,
    n_pitches: float = 1.0,
    # 曲げパラメータ
    bend_angle_deg: float = 90.0,
    n_bending_steps: int = 20,
    # 揺動パラメータ（サイクル変位）
    oscillation_amplitude_mm: float = 5.0,
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
    # GIF 出力
    gif_output_dir: str | Path | None = None,
    gif_snapshot_interval: int = 1,
) -> BendingOscillationResult:
    """曲げ揺動ベンチマークを実行.

    ロードパス:
      Phase 1（曲げ）: z=0 端を固定、z=L 端にモーメント Mx を n_bending_steps で負荷
      Phase 2（揺動）: 自由端 z-DOF を拘束し、正弦波変位 ±Δz を n_cycles 周期
        - 変位制御: 自由端の z-DOF を処方（固定拘束追加）
        - 曲げモーメントは外力 f_ext_bend として維持

    Args:
        n_strands: 素線本数（3, 7, 19, 37, 61, 91, ...）
        wire_diameter: 素線直径 [m]
        pitch: 撚ピッチ [m]
        n_elems_per_strand: 素線あたり要素数
        n_pitches: ピッチ数
        bend_angle_deg: 目標曲げ角度 [°]
        n_bending_steps: 曲げ荷重ステップ数
        oscillation_amplitude_mm: 揺動変位振幅 [mm]
        n_cycles: サイクル数
        n_steps_per_quarter: 1/4 周期あたりのステップ数
        max_iter: NR 最大反復数
        n_outer_max: Outer loop 最大反復数
        tol_force: 力残差収束判定値
        auto_kpen: ペナルティ剛性自動推定
        use_friction: 摩擦を有効にするか
        mu: 摩擦係数
        show_progress: 進捗表示
        gif_output_dir: GIF出力先ディレクトリ（None で GIF 無し）
        gif_snapshot_interval: GIF スナップショット間隔（1=全ステップ）

    Returns:
        BendingOscillationResult
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
        print(f"  曲げ揺動ベンチマーク: {n_strands}本撚線")
        print(f"{'=' * 70}")
        print(f"  要素数:     {mesh.n_elems}")
        print(f"  節点数:     {mesh.n_nodes}")
        print(f"  自由度数:   {ndof_total}")
        print(f"  モデル長さ: {L * 1000:.1f} mm")
        print(f"  目標曲げ角: {bend_angle_deg}°")
        print(f"  揺動振幅:   ±{oscillation_amplitude_mm} mm, {n_cycles}周期")

    # ------------------------------------------------------------------
    # 2. CR 梁アセンブラ構築
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    assemble_tangent, assemble_internal_force, ndof = _make_cr_assemblers(mesh, _E, _G, section)
    timing.record(0, 0, -1, "assembler_setup", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # 3. 境界条件 — z=0 端を全固定
    # ------------------------------------------------------------------
    fixed_dofs_base = _fix_strand_starts(mesh)

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
    # GIF 用スナップショット収集
    # ------------------------------------------------------------------
    snapshots: list[np.ndarray] = []
    snap_labels: list[str] = []
    snap_counter = 0

    def _record_snapshot(u: np.ndarray, label: str):
        nonlocal snap_counter
        snap_counter += 1
        if snap_counter % gif_snapshot_interval == 0 or snap_counter == 1:
            snapshots.append(u.copy())
            snap_labels.append(label)

    # 初期状態
    _record_snapshot(np.zeros(ndof), "初期")

    # ------------------------------------------------------------------
    # Phase 1: 曲げ（モーメント荷重 — 力制御）
    # ------------------------------------------------------------------
    bend_angle_rad = np.deg2rad(bend_angle_deg)
    M_per_strand = _E * section.Iy * bend_angle_rad / L

    f_ext_bend = np.zeros(ndof)
    for sid in range(mesh.n_strands):
        end_dofs = _get_strand_end_dofs(mesh, sid, "end")
        f_ext_bend[end_dofs[3]] = M_per_strand  # DOF 3 = rx

    if show_progress:
        print(f"\n--- Phase 1: 曲げ（M={M_per_strand:.4e} N·m/strand, {n_bending_steps} steps）---")

    result_bend = newton_raphson_with_contact(
        f_ext_bend,
        fixed_dofs_base,
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
    _record_snapshot(u_after_bend, f"曲げ完了 ({bend_angle_deg}°)")

    if show_progress:
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
    # Phase 2: 揺動（z方向サイクル変位 — 変位制御）
    # ------------------------------------------------------------------
    # 自由端の z-DOF を拘束に追加
    z_dofs_end: list[int] = []
    for sid in range(mesh.n_strands):
        end_dofs = _get_strand_end_dofs(mesh, sid, "end")
        z_dofs_end.append(int(end_dofs[2]))  # DOF 2 = z 並進
    z_dofs_end_arr = np.array(z_dofs_end, dtype=int)

    # Phase 2 用の拘束 DOF = 基本拘束 + 自由端 z-DOF
    fixed_dofs_phase2 = np.unique(np.concatenate([fixed_dofs_base, z_dofs_end_arr]))

    # Phase 1 完了時の自由端 z 変位（基準値）
    z_base = u_after_bend[z_dofs_end_arr].copy()

    amplitude_m = oscillation_amplitude_mm * 1e-3
    total_osc_steps = n_cycles * 4 * n_steps_per_quarter

    phase2_results: list[ContactSolveResult] = []
    u = u_after_bend.copy()
    phase2_all_converged = True

    if show_progress:
        print(
            f"\n--- Phase 2: 揺動（±{oscillation_amplitude_mm} mm, "
            f"{n_cycles}周期, {total_osc_steps} steps, 変位制御）---"
        )

    for osc_step in range(total_osc_steps):
        # 正弦波変位: sin(2π * step / (4 * n_steps_per_quarter))
        phase_frac = (osc_step + 1) / (4 * n_steps_per_quarter)
        delta_z = amplitude_m * np.sin(2.0 * np.pi * phase_frac)

        # 自由端 z-DOF に変位を処方
        u[z_dofs_end_arr] = z_base + delta_z

        # NR ソルバー実行（n_load_steps=1, 曲げモーメント維持）
        result_step = newton_raphson_with_contact(
            f_ext_bend,
            fixed_dofs_phase2,
            assemble_tangent,
            assemble_internal_force,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=1,
            max_iter=max_iter,
            tol_force=tol_force,
            show_progress=False,
            u0=u,
            broadphase_margin=0.01,
            timing=timing,
        )

        u = result_step.u.copy()
        phase2_results.append(result_step)

        if not result_step.converged:
            phase2_all_converged = False

        # スナップショット
        cycle_num = osc_step // (4 * n_steps_per_quarter) + 1
        quarter = (osc_step % (4 * n_steps_per_quarter)) // n_steps_per_quarter + 1
        _record_snapshot(u, f"C{cycle_num} Q{quarter} dz={delta_z * 1000:.2f}mm")

        if show_progress and (osc_step + 1) % n_steps_per_quarter == 0:
            tip_node = mesh.strand_nodes(0)[-1]
            tip_dz = u[_NDOF_PER_NODE * tip_node + 2]
            tip_dz_ref = mesh.node_coords[tip_node, 2]
            print(
                f"  揺動 step {osc_step + 1}/{total_osc_steps}: "
                f"Δz={delta_z * 1000:.2f} mm, "
                f"tip_z={tip_dz * 1000:.2f} mm (ref={tip_dz_ref * 1000:.1f} mm), "
                f"conv={result_step.converged}, "
                f"NR={result_step.total_newton_iterations}"
            )

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
        print(f"  曲げ揺動ベンチマーク完了: {total_time:.2f} s")
        print(f"{'=' * 70}")
        print(f"  Phase 1 収束: {result_bend.converged}")
        print(f"  Phase 2 収束: {phase2_all_converged}")
        print(f"  活性接触ペア: {n_active}")
        print(f"  最大貫入比:   {max_pen:.6f}")
        print(f"  総計算時間:   {total_time:.2f} s")

        if timing is not None:
            print(f"\n{timing.summary_table()}")

    result = BendingOscillationResult(
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
        displacement_snapshots=snapshots,
        snapshot_labels=snap_labels,
    )

    # GIF 出力
    if gif_output_dir is not None and snapshots:
        gif_files = export_bending_oscillation_gif(
            mesh, snapshots, snap_labels, gif_output_dir, prefix="bending_oscillation"
        )
        if show_progress and gif_files:
            print(f"  GIF出力: {[str(f) for f in gif_files]}")

    return result


# ====================================================================
# レポート出力
# ====================================================================


def print_benchmark_report(result: BendingOscillationResult) -> str:
    """ベンチマーク結果のフォーマット済みレポートを返す."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"  曲げ揺動ベンチマーク結果: {result.n_strands}本")
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

    total_nr = result.phase1_result.total_newton_iterations
    for r in result.phase2_results:
        total_nr += r.total_newton_iterations
    lines.append(f"  NR反復合計:   {total_nr}")
    lines.append(f"  Phase2ステップ数: {len(result.phase2_results)}")

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
) -> list[BendingOscillationResult]:
    """複数の素線本数で曲げ揺動ベンチマークを実行しスケーリングを分析.

    Args:
        strand_counts: 素線本数リスト（デフォルト [7, 19, 37]）
        **kwargs: run_bending_oscillation に渡す追加引数

    Returns:
        list of BendingOscillationResult
    """
    if strand_counts is None:
        strand_counts = [7, 19, 37]

    results = []
    for n in strand_counts:
        result = run_bending_oscillation(n_strands=n, **kwargs)
        results.append(result)

    # スケーリングレポート
    print(f"\n{'=' * 80}")
    print("  曲げ揺動ベンチマーク スケーリングレポート")
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


# 旧名との互換性
WireBendingBenchmarkResult = BendingOscillationResult
run_wire_bending_benchmark = run_bending_oscillation
