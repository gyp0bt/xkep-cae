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
from xkep_cae.contact.solver_ncp import newton_raphson_contact_ncp
from xkep_cae.elements.beam_timo3d import assemble_cr_beam3d
from xkep_cae.mesh.twisted_wire import TwistedWireMesh, make_twisted_wire_mesh
from xkep_cae.sections.beam import BeamSection

# ====================================================================
# 物理パラメータ（鋼線デフォルト）
# ====================================================================

_DEFAULT_E = 200e9  # Pa
_DEFAULT_NU = 0.3
_WIRE_D = 0.002  # 2 mm 直径
_NDOF_PER_NODE = 6


def _compute_G(E: float, nu: float) -> float:
    """せん断弾性係数を計算."""
    return E / (2.0 * (1.0 + nu))


def _compute_kappa(nu: float) -> float:
    """Cowper (1966) 円形断面のせん断補正係数を計算."""
    return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)


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


def _make_cr_assemblers(
    mesh: TwistedWireMesh, E: float, G: float, section: BeamSection, kappa: float
):
    """CR (corotational) 梁アセンブラを構築."""
    node_coords = mesh.node_coords
    connectivity = mesh.connectivity
    ndof_total = mesh.n_nodes * _NDOF_PER_NODE

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
    E: float,
    *,
    n_outer_max: int = 5,
    auto_kpen: bool = True,
    use_friction: bool = False,
    mu: float = 0.0,
    k_pen_scaling: str = "sqrt",
    penalty_growth_factor: float = 4.0,
    use_mortar: bool = False,
    n_gauss: int = 2,
    augmented_threshold: int = 20,
    saddle_regularization: float = 0.0,
    ncp_active_threshold: float = 0.0,
    lambda_relaxation: float = 1.0,
    # カテゴリD: 接触パラメータ（従来ハードコード）
    k_t_ratio: float = 0.1,
    g_on: float = 0.0,
    g_off: float = 1e-5,
    use_line_search: bool = True,
    line_search_max_steps: int = 5,
    use_geometric_stiffness: bool = True,
    tol_penetration_ratio: float = 0.02,
    k_pen_max: float = 1e12,
    exclude_same_layer: bool = True,
    midpoint_prescreening: bool = True,
    linear_solver: str = "auto",
    line_contact: bool = True,
) -> ContactManager:
    """接触マネージャを構築."""
    elem_layer_map = mesh.build_elem_layer_map()
    kpen_mode = "beam_ei" if auto_kpen else "manual"
    kpen_scale = 0.1 if auto_kpen else 1e5

    return ContactManager(
        config=ContactConfig(
            k_pen_scale=kpen_scale,
            k_pen_mode=kpen_mode,
            beam_E=E if auto_kpen else 0.0,
            beam_I=section.Iy if auto_kpen else 0.0,
            beam_A=section.A if auto_kpen else 0.0,
            k_t_ratio=k_t_ratio,
            mu=mu,
            g_on=g_on,
            g_off=g_off,
            n_outer_max=n_outer_max,
            use_friction=use_friction,
            use_line_search=use_line_search,
            line_search_max_steps=line_search_max_steps,
            use_geometric_stiffness=use_geometric_stiffness,
            tol_penetration_ratio=tol_penetration_ratio,
            penalty_growth_factor=penalty_growth_factor,
            k_pen_max=k_pen_max,
            elem_layer_map=elem_layer_map,
            exclude_same_layer=exclude_same_layer,
            midpoint_prescreening=midpoint_prescreening,
            linear_solver=linear_solver,
            k_pen_scaling=k_pen_scaling,
            use_mortar=use_mortar,
            line_contact=line_contact,
            n_gauss=n_gauss,
            augmented_threshold=augmented_threshold,
            saddle_regularization=saddle_regularization,
            ncp_active_threshold=ncp_active_threshold,
            lambda_relaxation=lambda_relaxation,
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
    n_elems_per_pitch: int = 16,
    n_elems_per_strand: int | None = None,  # 後方互換: 指定時は n_elems_per_pitch を上書き
    n_pitches: float = 1.0,
    strand_diameter: float | None = None,  # 撚線外径 [m]（非貫入配置）
    # 材料パラメータ（カテゴリB+C: ハードコード除去）
    E: float = _DEFAULT_E,
    nu: float = _DEFAULT_NU,
    # 曲げパラメータ
    bend_angle_deg: float = 90.0,
    n_bending_steps: int = 45,
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
    k_pen_scaling: str = "sqrt",
    penalty_growth_factor: float = 4.0,
    show_progress: bool = True,
    # GIF 出力
    gif_output_dir: str | Path | None = None,
    gif_snapshot_interval: int = 1,
    # S3: NCP ソルバーパラメータ
    use_ncp: bool = False,
    use_mortar: bool = True,
    n_gauss: int = 2,
    ncp_k_pen: float = 0.0,
    augmented_threshold: int = 20,
    saddle_regularization: float = 0.0,
    ncp_active_threshold: float = 0.0,
    lambda_relaxation: float = 1.0,
    adaptive_timestepping: bool = True,
    modified_nr_threshold: int = 5,
    # カテゴリD: 接触パラメータ（従来ハードコード）
    k_t_ratio: float = 0.1,
    g_on: float = 0.0,
    g_off: float = 1e-5,
    use_line_search: bool = True,
    line_search_max_steps: int = 5,
    use_geometric_stiffness: bool = True,
    tol_penetration_ratio: float = 0.02,
    k_pen_max: float = 1e12,
    exclude_same_layer: bool = True,
    midpoint_prescreening: bool = True,
    linear_solver: str = "auto",
    line_contact: bool = True,
    # カテゴリE: 数値パラメータ（従来ハードコード）
    broadphase_margin: float = 0.01,
    # メッシュ密度検査
    min_elems_per_pitch: int = 16,
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
        n_elems_per_pitch: 1ピッチあたり要素数（n_pitchesでスケーリング）
        n_pitches: ピッチ数
        E: ヤング率 [Pa]
        nu: ポアソン比
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
        k_t_ratio: 接線/法線ペナルティ比
        g_on: 接触活性化ギャップ [m]
        g_off: 接触非活性化ギャップ [m]
        use_line_search: ライン探索有効
        line_search_max_steps: ライン探索最大ステップ
        use_geometric_stiffness: 幾何学的剛性行列を使用
        tol_penetration_ratio: 貫入比閾値
        k_pen_max: ペナルティ剛性上限
        exclude_same_layer: 同層除外
        midpoint_prescreening: 中点プレスクリーニング
        linear_solver: 線形ソルバー選択
        line_contact: ライン接触
        broadphase_margin: 広域探索マージン [m]

    Returns:
        BendingOscillationResult
    """
    t_total_start = time.perf_counter()
    timing = BenchmarkTimingCollector()

    # ------------------------------------------------------------------
    # 1. メッシュ生成
    # ------------------------------------------------------------------
    # ピッチあたりの要素数 → 素線あたりの要素数にスケーリング
    # n_elems_per_strand が明示指定された場合はそちらを優先（後方互換）
    if n_elems_per_strand is not None:
        _n_elems_per_strand = n_elems_per_strand
    else:
        _n_elems_per_strand = max(4, int(round(n_elems_per_pitch * n_pitches)))

    t0 = time.perf_counter()
    mesh = make_twisted_wire_mesh(
        n_strands,
        wire_diameter,
        pitch,
        length=0.0,
        n_elems_per_strand=_n_elems_per_strand,
        n_pitches=n_pitches,
        strand_diameter=strand_diameter,
        min_elems_per_pitch=min_elems_per_pitch,
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
    # 2. CR 梁アセンブラ構築（E, nu から G, kappa を導出）
    # ------------------------------------------------------------------
    G = _compute_G(E, nu)
    kappa = _compute_kappa(nu)
    t0 = time.perf_counter()
    assemble_tangent, assemble_internal_force, ndof = _make_cr_assemblers(
        mesh, E, G, section, kappa
    )
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
        E,
        n_outer_max=n_outer_max,
        auto_kpen=auto_kpen,
        use_friction=use_friction,
        mu=mu,
        k_pen_scaling=k_pen_scaling,
        penalty_growth_factor=penalty_growth_factor,
        use_mortar=use_mortar if use_ncp else False,
        n_gauss=n_gauss,
        augmented_threshold=augmented_threshold,
        saddle_regularization=saddle_regularization,
        ncp_active_threshold=ncp_active_threshold,
        lambda_relaxation=lambda_relaxation,
        k_t_ratio=k_t_ratio,
        g_on=g_on,
        g_off=g_off,
        use_line_search=use_line_search,
        line_search_max_steps=line_search_max_steps,
        use_geometric_stiffness=use_geometric_stiffness,
        tol_penetration_ratio=tol_penetration_ratio,
        k_pen_max=k_pen_max,
        exclude_same_layer=exclude_same_layer,
        midpoint_prescreening=midpoint_prescreening,
        linear_solver=linear_solver,
        line_contact=line_contact,
    )

    # ------------------------------------------------------------------
    # 4b. 初期貫入オフセット（LS-DYNA IGNORE=1 相当）
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    mgr.detect_candidates(
        mesh.node_coords,
        mesh.connectivity,
        mesh.radii,
        margin=broadphase_margin,
    )
    n_initial_pen = mgr.store_initial_offsets(mesh.node_coords)
    timing.record(0, 0, -1, "initial_penetration_offset", time.perf_counter() - t0)

    if show_progress and n_initial_pen > 0:
        max_offset = min(p.gap_offset for p in mgr.pairs if p.gap_offset < 0.0)
        print(f"  初期貫入オフセット: {n_initial_pen}ペア（最大 {max_offset * 1000:.4f} mm）")

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
    # Phase 1: 曲げ
    # ------------------------------------------------------------------
    bend_angle_rad = np.deg2rad(bend_angle_deg)
    M_per_strand = E * section.Iy * bend_angle_rad / L

    f_ext_bend = np.zeros(ndof)
    rx_dofs_end: list[int] = []
    for sid in range(mesh.n_strands):
        end_dofs = _get_strand_end_dofs(mesh, sid, "end")
        f_ext_bend[end_dofs[3]] = M_per_strand  # DOF 3 = rx
        rx_dofs_end.append(int(end_dofs[3]))
    rx_dofs_end_arr = np.array(rx_dofs_end, dtype=int)

    if use_ncp:
        # --- 変位制御: 自由端 rx DOF に回転角を処方 ---
        # 力制御の限界点(frac≈0.178)を回避するため、
        # 回転角を直接処方して各ステップで平衡解を求める
        # NCPソルバーの内部機構（二分法、Modified NR、適応LS、接線予測子）を活用

        # 処方変位: load_frac=1.0 で θ = bend_angle_rad
        prescribed_vals = np.full(len(rx_dofs_end), bend_angle_rad)

        if show_progress:
            print(
                f"\n--- Phase 1: 曲げ（変位制御 θ={bend_angle_deg}°, {n_bending_steps} steps）---"
            )

        _ncp_result = newton_raphson_contact_ncp(
            np.zeros(ndof),  # 変位制御: 外力なし
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
            tol_ncp=tol_force,
            show_progress=show_progress,
            broadphase_margin=broadphase_margin,
            line_contact=line_contact,
            use_mortar=use_mortar,
            n_gauss=n_gauss,
            k_pen=ncp_k_pen,
            adaptive_timestepping=adaptive_timestepping,
            modified_nr_threshold=modified_nr_threshold,
            prescribed_dofs=rx_dofs_end_arr,
            prescribed_values=prescribed_vals,
        )
        result_bend = ContactSolveResult(
            u=_ncp_result.u,
            converged=_ncp_result.converged,
            n_load_steps=_ncp_result.n_load_steps,
            total_newton_iterations=_ncp_result.total_newton_iterations,
            total_outer_iterations=0,
            n_active_final=_ncp_result.n_active_final,
            load_history=_ncp_result.load_history,
            displacement_history=_ncp_result.displacement_history,
            contact_force_history=_ncp_result.contact_force_history,
            graph_history=_ncp_result.graph_history,
        )
    else:
        if show_progress:
            print(
                f"\n--- Phase 1: 曲げ（M={M_per_strand:.4e} N·m/strand, "
                f"{n_bending_steps} steps）---"
            )
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
            broadphase_margin=broadphase_margin,
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
    # NCP変位制御時は rx DOFs も拘束に含める（曲げ角維持）
    if use_ncp:
        fixed_dofs_phase2 = np.unique(
            np.concatenate([fixed_dofs_base, z_dofs_end_arr, rx_dofs_end_arr])
        )
    else:
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

        # NR ソルバー実行（n_load_steps=1, 曲げ維持）
        # NCP変位制御: 外力0、rx DOFで曲げ角維持
        # AL力制御: f_ext_bend で曲げモーメント維持
        if use_ncp:
            _ncp_step = newton_raphson_contact_ncp(
                np.zeros(ndof),
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
                tol_ncp=tol_force,
                show_progress=False,
                u0=u,
                broadphase_margin=broadphase_margin,
                line_contact=line_contact,
                use_mortar=use_mortar,
                n_gauss=n_gauss,
                k_pen=ncp_k_pen,
            )
            result_step = ContactSolveResult(
                u=_ncp_step.u,
                converged=_ncp_step.converged,
                n_load_steps=_ncp_step.n_load_steps,
                total_newton_iterations=_ncp_step.total_newton_iterations,
                total_outer_iterations=0,
                n_active_final=_ncp_step.n_active_final,
                load_history=_ncp_step.load_history,
                displacement_history=_ncp_step.displacement_history,
                contact_force_history=_ncp_step.contact_force_history,
                graph_history=_ncp_step.graph_history,
            )
        else:
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
                broadphase_margin=broadphase_margin,
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
