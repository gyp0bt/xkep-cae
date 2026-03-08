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
import math
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
from xkep_cae.elements.beam_timo3d import ULCRBeamAssembler, assemble_cr_beam3d
from xkep_cae.mesh.twisted_wire import TwistedWireMesh, make_twisted_wire_mesh
from xkep_cae.sections.beam import BeamSection

# ====================================================================
# 物理パラメータ（鋼線デフォルト）
# ====================================================================

_DEFAULT_E = 200e3  # MPa（鋼、mm-ton-MPa単位系）
_DEFAULT_NU = 0.3
_WIRE_D = 2.0  # mm 直径（mm-ton-MPa単位系）
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
    mesh: TwistedWireMesh,
    E: float,
    G: float,
    section: BeamSection,
    kappa: float,
    use_ul: bool = False,
):
    """CR (corotational) 梁アセンブラを構築.

    Args:
        use_ul: True の場合 Updated Lagrangian アセンブラを返す。
            ヘリカル梁の大回転収束問題を解消する。
    """
    node_coords = mesh.node_coords
    connectivity = mesh.connectivity
    ndof_total = mesh.n_nodes * _NDOF_PER_NODE

    if use_ul:
        ul_asm = ULCRBeamAssembler(
            node_coords,
            connectivity,
            E,
            G,
            section.A,
            section.Iy,
            section.Iz,
            section.J,
            kappa,
            kappa,
        )
        return ul_asm.assemble_tangent, ul_asm.assemble_internal_force, ndof_total, ul_asm

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

    return assemble_tangent, assemble_internal_force, ndof_total, None


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
    g_off: float = 0.01,  # mm
    use_line_search: bool = True,
    line_search_max_steps: int = 5,
    use_geometric_stiffness: bool = True,
    tol_penetration_ratio: float = 0.02,
    k_pen_max: float = 1e12,
    exclude_same_layer: bool = True,
    midpoint_prescreening: bool = True,
    linear_solver: str = "auto",
    line_contact: bool = True,
    coating_stiffness: float = 0.0,
    coating_damping: float = 0.0,
    core_radii: np.ndarray | None = None,
) -> ContactManager:
    """接触マネージャを構築."""
    elem_layer_map = mesh.build_elem_layer_map()
    # 常に材料ベースでk_penを推定（status-140: 手動モード廃止）
    kpen_scale = 0.1

    return ContactManager(
        config=ContactConfig(
            k_pen_scale=kpen_scale,
            k_pen_mode="beam_ei",
            beam_E=E,
            beam_I=section.Iy,
            beam_A=section.A,
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
            coating_stiffness=coating_stiffness,
            coating_damping=coating_damping,
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
# GIF 出力（3Dチューブレンダリング）
# ====================================================================

# 2D→3D視角の対応マッピング（後方互換性）
_VIEW_2D_TO_3D = {
    "xy": "front_xy",
    "xz": "side_xz",
    "yz": "end_yz",
}


def export_bending_oscillation_gif(
    mesh: TwistedWireMesh,
    displacement_snapshots: list[np.ndarray],
    snapshot_labels: list[str],
    output_dir: str | Path,
    *,
    prefix: str = "bending_oscillation",
    views: list[str] | None = None,
    figsize: tuple[float, float] = (12.0, 10.0),
    dpi: int = 80,
    duration: int = 200,
) -> list[Path]:
    """曲げ揺動の経時変化GIFを3Dチューブレンダリングで生成.

    Args:
        mesh: 撚線メッシュ
        displacement_snapshots: 各ステップの変位ベクトル (ndof,)
        snapshot_labels: 各ステップのラベル
        output_dir: 出力ディレクトリ
        prefix: ファイル名プレフィックス
        views: 描画ビュー方向。旧2D名("xy","xz","yz")または
            3Dプリセット名("isometric","front_xy"等)。デフォルト ["isometric", "end_yz"]
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

        from xkep_cae.output.render_beam_3d import VIEW_PRESETS, render_twisted_wire_3d
    except ImportError:
        return []

    if views is None:
        views = ["isometric", "end_yz"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    node_coords_ref = mesh.node_coords

    # 全フレームの変形座標を収集
    deformed_frames = []
    for u in displacement_snapshots:
        coords_def = _deformed_coords(node_coords_ref, u)
        deformed_frames.append(coords_def)

    output_files: list[Path] = []

    for view in views:
        # 旧2D名 → 3Dプリセット名に変換（後方互換性）
        view_3d = _VIEW_2D_TO_3D.get(view, view)
        if view_3d not in VIEW_PRESETS:
            continue

        preset = VIEW_PRESETS[view_3d]

        pil_images: list[PILImage.Image] = []
        for coords, label in zip(deformed_frames, snapshot_labels, strict=True):
            title = f"{label} — {preset['label']}"
            fig, _ax = render_twisted_wire_3d(
                mesh,
                node_coords=coords,
                elev=float(preset["elev"]),
                azim=float(preset["azim"]),
                title=title,
                figsize=figsize,
                dpi=dpi,
                n_circ=10,
            )

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
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
    pitch: float = 40.0,  # mm
    n_elems_per_pitch: int = 16,
    n_elems_per_strand: int | None = None,  # 後方互換: 指定時は n_elems_per_pitch を上書き
    n_pitches: float = 1.0,
    strand_diameter: float | None = None,  # 撚線外径 [mm]（非貫入配置）
    # 材料パラメータ（カテゴリB+C: ハードコード除去）
    E: float = _DEFAULT_E,
    nu: float = _DEFAULT_NU,
    # 曲げパラメータ
    bend_angle_deg: float = 90.0,
    n_bending_steps: int | None = None,
    max_angle_per_step_deg: float = 3.0,
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
    g_off: float = 0.01,  # mm
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
    broadphase_margin: float = 10.0,  # mm
    # メッシュ密度検査
    min_elems_per_pitch: int = 16,
    # Updated Lagrangian
    use_updated_lagrangian: bool = False,
    # 被膜モデル
    coating_thickness: float = 0.0,
    coating_stiffness: float = 0.0,
    coating_youngs: float = 0.0,
    coating_nu: float = 0.4,
    coating_damping: float = 0.0,
    # メッシュギャップ（弦近似誤差による初期貫入防止）
    mesh_gap: float = 0.0,
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
        n_bending_steps: 曲げ荷重ステップ数（Noneで自動推定: ceil(angle/max_angle_per_step_deg)）
        max_angle_per_step_deg: 1ステップあたり最大角度 [°]（デフォルト3°、自動推定時に使用）
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
    # 0. n_bending_steps 自動推定
    # ------------------------------------------------------------------
    if n_bending_steps is None:
        n_bending_steps = max(1, math.ceil(bend_angle_deg / max_angle_per_step_deg))

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
        coating_thickness=coating_thickness,
        gap=mesh_gap,
    )
    timing.record(0, 0, -1, "mesh_generation", time.perf_counter() - t0)

    # 被膜ありの場合、接触半径を被膜込み / 芯線半径を分離
    if coating_thickness > 0.0:
        from xkep_cae.mesh.twisted_wire import CoatingModel, coated_radii

        _coating_youngs = coating_youngs if coating_youngs > 0.0 else 100.0  # MPa（PE被膜）
        _coat_model = CoatingModel(thickness=coating_thickness, E=_coating_youngs, nu=coating_nu)
        contact_radii = coated_radii(mesh, _coat_model)
        # k_coat 自動導出: Winkler基盤 k = E_coat / t_coat [Pa/m]
        if coating_stiffness <= 0.0 and _coating_youngs > 0.0:
            coating_stiffness = _coating_youngs / coating_thickness
        # 粘性減衰の自動推定（ζ=1.0 臨界減衰、面密度ρ*t近似）
        if coating_damping <= 0.0 and coating_stiffness > 0.0:
            coating_damping = 0.0  # デフォルト無減衰（ユーザ指定時のみ有効）
    else:
        contact_radii = mesh.radii

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
        print(f"  モデル長さ: {L:.1f} mm")
        print(f"  目標曲げ角: {bend_angle_deg}°")
        print(f"  揺動振幅:   ±{oscillation_amplitude_mm} mm, {n_cycles}周期")

    # ------------------------------------------------------------------
    # 2. CR 梁アセンブラ構築（E, nu から G, kappa を導出）
    # ------------------------------------------------------------------
    G = _compute_G(E, nu)
    kappa = _compute_kappa(nu)
    t0 = time.perf_counter()
    assemble_tangent, assemble_internal_force, ndof, ul_asm = _make_cr_assemblers(
        mesh, E, G, section, kappa, use_ul=use_updated_lagrangian
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
        coating_stiffness=coating_stiffness,
        coating_damping=coating_damping,
    )

    # ------------------------------------------------------------------
    # 4b. 初期貫入チェック・修正
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    mgr.detect_candidates(
        mesh.node_coords,
        mesh.connectivity,
        contact_radii,
        margin=broadphase_margin,
    )
    n_initial_pen = mgr.check_initial_penetration(mesh.node_coords)
    timing.record(0, 0, -1, "initial_penetration_check", time.perf_counter() - t0)

    if show_progress and n_initial_pen > 0:
        print(f"  初期貫入検出: {n_initial_pen}ペア")

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
    _record_snapshot(np.zeros(ndof), "initial")

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

    if use_ncp and use_updated_lagrangian and ul_asm is not None:
        # --- UL + NCP 統合: adaptive_timestepping と UL 参照更新を一体化 ---
        # NCP ソルバ内部で各ステップ収束後に参照配置を更新。
        # adaptive_timestepping が自然に UL と連動（不収束時に角度増分を自動縮小）。
        prescribed_vals_total = np.full(len(rx_dofs_end), bend_angle_rad)

        # 適応時間増分: 物理ベースの初期Δt推定
        # n_load_steps=1 でも max_angle_per_step_deg から適切な初期増分を計算
        _physics_n_steps = max(1, math.ceil(bend_angle_deg / max_angle_per_step_deg))
        _dt_init_frac = 1.0 / _physics_n_steps if adaptive_timestepping else 0.0

        if show_progress:
            print(
                f"\n--- Phase 1: 曲げ（UL+NCP統合 θ={bend_angle_deg}°, "
                f"{n_bending_steps} steps, adaptive={adaptive_timestepping}"
                f"{f', dt_init={_dt_init_frac:.4f}' if _dt_init_frac > 0 else ''}）---"
            )

        _ncp_result = newton_raphson_contact_ncp(
            np.zeros(ndof),  # 外力なし（変位制御）
            fixed_dofs_base,
            assemble_tangent,
            assemble_internal_force,
            mgr,
            ul_asm.coords_ref,
            mesh.connectivity,
            contact_radii,
            max_iter=max_iter,
            tol_force=tol_force,
            tol_ncp=tol_force,
            show_progress=show_progress,
            broadphase_margin=broadphase_margin,
            line_contact=line_contact,
            use_mortar=use_mortar,
            n_gauss=n_gauss,
            k_pen=ncp_k_pen,
            prescribed_dofs=rx_dofs_end_arr,
            prescribed_values=prescribed_vals_total,
            adaptive_timestepping=adaptive_timestepping,
            dt_initial_fraction=_dt_init_frac,
            dt_grow_iter_threshold=8,  # Phase1は接触なし→積極成長
            ul_assembler=ul_asm,
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
    elif use_ncp:
        # --- TL + NCP: 従来の一括ロードステッピング ---
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
            contact_radii,
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
            dt_initial_fraction=_dt_init_frac,
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
            contact_radii,
            n_load_steps=n_bending_steps,
            max_iter=max_iter,
            tol_force=tol_force,
            show_progress=show_progress,
            broadphase_margin=broadphase_margin,
            timing=timing,
        )

    u_after_bend = result_bend.u.copy()
    _record_snapshot(u_after_bend, f"bend done ({bend_angle_deg}deg)")

    if show_progress:
        tip_node = mesh.strand_nodes(0)[-1]
        tip_disp = u_after_bend[_NDOF_PER_NODE * tip_node : _NDOF_PER_NODE * tip_node + 3]
        print(
            f"  Phase 1 完了: converged={result_bend.converged}, "
            f"NR={result_bend.total_newton_iterations}"
        )
        print(
            f"  先端変位: dx={tip_disp[0]:.3f} mm, dy={tip_disp[1]:.3f} mm, dz={tip_disp[2]:.3f} mm"
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
    # UL使用時: 参照配置が更新済みなのでz_base=0、uも0からスタート
    if use_updated_lagrangian and ul_asm is not None:
        z_base = np.zeros(len(z_dofs_end))
        u = np.zeros(ndof)
    else:
        z_base = u_after_bend[z_dofs_end_arr].copy()
        u = u_after_bend.copy()

    amplitude_m = oscillation_amplitude_mm  # mm-ton-MPa系: 振幅はmm単位
    total_osc_steps = n_cycles * 4 * n_steps_per_quarter

    phase2_results: list[ContactSolveResult] = []
    phase2_all_converged = True

    if show_progress:
        print(
            f"\n--- Phase 2: 揺動（±{oscillation_amplitude_mm} mm, "
            f"{n_cycles}周期, {total_osc_steps} steps, 変位制御）---"
        )

    # --- Phase 2: NCP prescribed_dofs + adaptive_timestepping 方式 ---
    # 各四半周期をNCP内部の荷重ステッピングで処理（特異行列回避）
    if use_ncp and use_updated_lagrangian and ul_asm is not None:
        # UL+NCP統合方式: 各四半周期の目標z変位をprescribed_valuesで処方
        # prescribed_dofs = z_dofs_end + rx_dofs_end (曲げ角維持)
        osc_prescribed_dofs = np.concatenate([z_dofs_end_arr, rx_dofs_end_arr])
        n_z = len(z_dofs_end_arr)
        n_rx = len(rx_dofs_end_arr)

        # 四半周期ごとのウェイポイント生成
        waypoints: list[float] = []
        for osc_step in range(total_osc_steps):
            phase_frac = (osc_step + 1) / (4 * n_steps_per_quarter)
            delta_z = amplitude_m * np.sin(2.0 * np.pi * phase_frac)
            waypoints.append(delta_z)

        # Phase2の初期Δt: Phase1の物理ベース推定から一貫して使用
        # n_load_steps=1で統一し、dt_initial_fractionで初期増分を制御
        # 揺動は方向転換があるため、Phase1より保守的な初期増分が必要
        _osc_base_n = max(5, _physics_n_steps // 6)  # Phase1物理推定の1/6
        _osc_dt_init_base = 1.0 / _osc_base_n if adaptive_timestepping else 0.0

        prev_delta_z = 0.0
        for osc_step in range(total_osc_steps):
            delta_z = waypoints[osc_step]
            # 増分: 前ステップからの差分
            incr_z = delta_z - prev_delta_z

            # prescribed_values: [z_dofs の増分, rx_dofs は 0（曲げ角維持）]
            prescribed_vals = np.zeros(n_z + n_rx)
            prescribed_vals[:n_z] = incr_z  # z方向増分変位

            # UL参照の一時チェックポイント
            ul_asm.checkpoint()

            _ncp_step = newton_raphson_contact_ncp(
                np.zeros(ndof),
                fixed_dofs_base,  # z=0端のみ固定（prescribed_dofsで残りを制御）
                assemble_tangent,
                assemble_internal_force,
                mgr,
                ul_asm.coords_ref,
                mesh.connectivity,
                contact_radii,
                max_iter=max_iter,
                tol_force=tol_force,
                tol_ncp=tol_force,
                show_progress=show_progress,
                broadphase_margin=broadphase_margin,
                line_contact=line_contact,
                use_mortar=use_mortar,
                n_gauss=n_gauss,
                k_pen=ncp_k_pen,
                prescribed_dofs=osc_prescribed_dofs,
                prescribed_values=prescribed_vals,
                adaptive_timestepping=adaptive_timestepping,
                dt_initial_fraction=_osc_dt_init_base,
                ul_assembler=ul_asm,
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

            # UL更新は NCP 内部で実施済み → u=0（リセット済み）
            u = np.zeros(ndof)
            phase2_results.append(result_step)

            if not result_step.converged:
                phase2_all_converged = False
                if show_progress:
                    print(f"  揺動 step {osc_step + 1}/{total_osc_steps}: 不収束, break")
                break

            prev_delta_z = delta_z

            # スナップショット（UL時はaccum変位を使用）
            cycle_num = osc_step // (4 * n_steps_per_quarter) + 1
            quarter = (osc_step % (4 * n_steps_per_quarter)) // n_steps_per_quarter + 1
            _snap_u = ul_asm.get_total_displacement(np.zeros(ndof))
            _record_snapshot(_snap_u, f"C{cycle_num} Q{quarter} dz={delta_z:.2f}mm")

            if show_progress and (osc_step + 1) % n_steps_per_quarter == 0:
                print(
                    f"  揺動 step {osc_step + 1}/{total_osc_steps}: "
                    f"Δz={delta_z:.2f} mm, "
                    f"conv={result_step.converged}, "
                    f"NR={result_step.total_newton_iterations}"
                )

    else:
        # --- 従来方式: 個別NRステップ ---
        for osc_step in range(total_osc_steps):
            phase_frac = (osc_step + 1) / (4 * n_steps_per_quarter)
            delta_z = amplitude_m * np.sin(2.0 * np.pi * phase_frac)

            u[z_dofs_end_arr] = z_base + delta_z

            if use_ncp:
                _phase2_node_coords = (
                    ul_asm.coords_ref
                    if (use_updated_lagrangian and ul_asm is not None)
                    else mesh.node_coords
                )
                _ncp_step = newton_raphson_contact_ncp(
                    np.zeros(ndof),
                    fixed_dofs_phase2,
                    assemble_tangent,
                    assemble_internal_force,
                    mgr,
                    _phase2_node_coords,
                    mesh.connectivity,
                    contact_radii,
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
                    contact_radii,
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

            cycle_num = osc_step // (4 * n_steps_per_quarter) + 1
            quarter = (osc_step % (4 * n_steps_per_quarter)) // n_steps_per_quarter + 1
            _snap_u = (u_after_bend + u) if (use_updated_lagrangian and ul_asm is not None) else u
            _record_snapshot(_snap_u, f"C{cycle_num} Q{quarter} dz={delta_z:.2f}mm")

            if show_progress and (osc_step + 1) % n_steps_per_quarter == 0:
                tip_node = mesh.strand_nodes(0)[-1]
                tip_dz = u[_NDOF_PER_NODE * tip_node + 2]
                tip_dz_ref = mesh.node_coords[tip_node, 2]
                print(
                    f"  揺動 step {osc_step + 1}/{total_osc_steps}: "
                    f"Δz={delta_z:.2f} mm, "
                    f"tip_z={tip_dz:.2f} mm (ref={tip_dz_ref:.1f} mm), "
                    f"conv={result_step.converged}, "
                    f"NR={result_step.total_newton_iterations}"
                )

    # ------------------------------------------------------------------
    # 結果集約
    # ------------------------------------------------------------------
    total_time = time.perf_counter() - t_total_start

    # UL使用時: 累積変位はul_asm.get_total_displacement()で取得
    tip_node = mesh.strand_nodes(0)[-1]
    if use_updated_lagrangian and ul_asm is not None:
        u_total_final = ul_asm.get_total_displacement(np.zeros(ndof))
        tip_disp = u_total_final[_NDOF_PER_NODE * tip_node : _NDOF_PER_NODE * tip_node + 3]
    else:
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
    lines.append(f"  モデル長さ:   {result.mesh_length:.1f} mm")
    lines.append(f"  Phase 1 収束: {result.phase1_converged}")
    lines.append(f"  Phase 2 収束: {result.phase2_converged}")
    lines.append(f"  活性接触:     {result.n_active_contacts}")
    lines.append(f"  最大貫入比:   {result.max_penetration_ratio:.6f}")
    lines.append(f"  総計算時間:   {result.total_time_s:.2f} s")

    dx, dy, dz = result.tip_displacement_final
    lines.append(f"  先端変位:     dx={dx:.3f} mm, dy={dy:.3f} mm, dz={dz:.3f} mm")

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
