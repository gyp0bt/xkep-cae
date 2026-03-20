"""ContourField3DProcess — 梁3Dコンターレンダリング.

変形後の梁をチューブ状に3Dレンダリングし、
要素ごとのスカラーフィールド（S11/LE11/SK1等）をカラーマッピングで可視化する。

フィールドラベル規約（Abaqus準拠）:
  - S: 応力 (S11=曲げ応力 [MPa])
  - LE: 対数ひずみ (LE11=曲げひずみ [-])
  - SK: 曲率 (SK1=曲率 [1/mm])

出力先は毎回上書き。アスペクト比・軸範囲は初期形状基準で固定し、
変形倍率（deformation_scale）で変形モードを強調表示する。

出力:
  - 各フィールドの3Dビュー PNG（正面・斜視）× フレーム数
  - 全フィールドの時刻歴プロット PNG

[← README](../../README.md)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from xkep_cae.core import MeshData, PostProcess, ProcessMeta

# フィールドラベル定義
_FIELD_LABELS: dict[str, str] = {
    "S11": "Bending Stress S11 [MPa]",
    "LE11": "Bending Strain LE11 [-]",
    "SK1": "Curvature SK1 [1/mm]",
    "Mises": "von Mises Stress [MPa]",
    "MaxPrincipal": "Max Principal Stress [MPa]",
}

# フィールドの表示フォーマット
_FIELD_FORMATS: dict[str, str] = {
    "S11": ".1f",
    "LE11": ".2e",
    "SK1": ".2e",
    "Mises": ".1f",
    "MaxPrincipal": ".1f",
}


@dataclass(frozen=True)
class ContourFieldInput:
    """コンターフィールド仕様."""

    name: str  # e.g. "S11", "LE11", "SK1"
    snapshots: list[np.ndarray]  # 各スナップショットの要素値


@dataclass(frozen=True)
class StressContour3DConfig:
    """3Dコンターの設定."""

    mesh: MeshData
    node_coords_initial: np.ndarray  # 初期節点座標
    displacement_snapshots: list[np.ndarray]  # 各スナップショットの変位
    contour_fields: list[ContourFieldInput]  # コンターフィールド群
    time_values: np.ndarray  # 各スナップショットの時刻
    wire_radius: float = 1.0  # mm
    output_dir: str = "tmp/oscillation"
    prefix: str = "beam_oscillation"
    tube_segments: int = 12  # チューブ断面の分割数
    n_render_frames: int = 6  # レンダリングするフレーム数
    ndof_per_node: int = 6
    deformation_scale: float = 0.0  # 変形倍率（0=自動）


@dataclass(frozen=True)
class StressContour3DResult:
    """3Dコンターの結果."""

    image_paths: list[str] = field(default_factory=list)
    field_max_values: dict[str, float] = field(default_factory=dict)


class StressContour3DProcess(PostProcess[StressContour3DConfig, StressContour3DResult]):
    """梁3Dコンターレンダリング Process.

    変形後の梁をチューブ状に3Dレンダリングし、
    各フィールド（S11/LE11/SK1等）をカラーマッピングで表示する。
    アスペクト比は初期形状基準で固定し、変形倍率で調整する。
    """

    meta = ProcessMeta(
        name="StressContour3D",
        module="post",
        version="3.0.0",
        document_path="docs/stress_contour.md",
    )

    def process(self, input_data: StressContour3DConfig) -> StressContour3DResult:
        """3Dコンターレンダリングを実行."""
        cfg = input_data
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_snapshots = len(cfg.displacement_snapshots)
        if n_snapshots == 0 or len(cfg.contour_fields) == 0:
            return StressContour3DResult()

        # レンダリングフレーム選択（等間隔）
        n_frames = min(cfg.n_render_frames, n_snapshots)
        frame_indices = np.linspace(0, n_snapshots - 1, n_frames, dtype=int)

        # 変形倍率の自動計算
        deformation_scale = _auto_deformation_scale(cfg)

        # 初期形状の軸範囲を固定（アスペクト比維持）
        axis_ranges = _compute_fixed_axis_ranges(
            cfg.node_coords_initial, cfg.wire_radius
        )

        # 各フレームの変形後座標を事前計算
        deformed_coords_cache: dict[int, np.ndarray] = {}
        for snap_idx in frame_indices:
            u = cfg.displacement_snapshots[snap_idx]
            coords = cfg.node_coords_initial.copy()
            n_nodes = len(coords)
            for i in range(n_nodes):
                for d in range(3):
                    idx = i * cfg.ndof_per_node + d
                    if idx < len(u):
                        coords[i, d] += u[idx] * deformation_scale
            deformed_coords_cache[int(snap_idx)] = coords

        image_paths: list[str] = []
        field_max_values: dict[str, float] = {}

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize

            cmap = plt.cm.jet

            # フィールドごとにレンダリング
            for cf in cfg.contour_fields:
                fname = cf.name
                label = _FIELD_LABELS.get(fname, fname)
                fmt = _FIELD_FORMATS.get(fname, ".2e")

                # 全スナップショットの最大値（カラーバー統一用）
                vmax = 0.0
                for snap in cf.snapshots:
                    if len(snap) > 0:
                        vmax = max(vmax, float(np.max(np.abs(snap))))
                field_max_values[fname] = vmax

                norm = Normalize(vmin=0, vmax=vmax if vmax > 0 else 1e-10)

                for fi, snap_idx in enumerate(frame_indices):
                    data = cf.snapshots[snap_idx]
                    t_val = (
                        cfg.time_values[snap_idx]
                        if snap_idx < len(cfg.time_values)
                        else 0.0
                    )
                    coords = deformed_coords_cache[int(snap_idx)]
                    conn = cfg.mesh.connectivity

                    fig = plt.figure(figsize=(14, 6))

                    # 正面ビュー（XY平面）
                    ax1 = fig.add_subplot(121, projection="3d")
                    _render_tube_3d(
                        ax1, coords, conn, data,
                        cfg.wire_radius, cfg.tube_segments, cmap, norm,
                    )
                    _apply_fixed_axes(ax1, axis_ranges)
                    ax1.set_xlabel("X [mm]")
                    ax1.set_ylabel("Y [mm]")
                    ax1.set_zlabel("Z [mm]")
                    ax1.set_title(f"Side view XY (t={t_val * 1000:.3f} ms)")
                    ax1.view_init(elev=0, azim=-90)

                    # 斜視ビュー
                    ax2 = fig.add_subplot(122, projection="3d")
                    _render_tube_3d(
                        ax2, coords, conn, data,
                        cfg.wire_radius, cfg.tube_segments, cmap, norm,
                    )
                    _apply_fixed_axes(ax2, axis_ranges)
                    ax2.set_xlabel("X [mm]")
                    ax2.set_ylabel("Y [mm]")
                    ax2.set_zlabel("Z [mm]")
                    ax2.set_title(f"Perspective (t={t_val * 1000:.3f} ms)")
                    ax2.view_init(elev=25, azim=-60)

                    # カラーバー
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    fig.colorbar(sm, ax=[ax1, ax2], label=label, shrink=0.6)

                    snap_max = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
                    fig.suptitle(
                        f"Beam Oscillation — {label}\n"
                        f"t = {t_val * 1000:.3f} ms, "
                        f"max = {snap_max:{fmt}}, "
                        f"deform x{deformation_scale:.0f}",
                        fontsize=11,
                    )
                    fig.tight_layout(rect=[0, 0, 1, 0.93])

                    img_path = str(
                        output_dir / f"{cfg.prefix}_{fname}_{fi:03d}.png"
                    )
                    fig.savefig(img_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    image_paths.append(img_path)

            # 時刻歴プロット（全フィールド）
            _path = _render_time_history(cfg, output_dir, plt, field_max_values)
            if _path:
                image_paths.append(_path)

        except ImportError:
            pass

        return StressContour3DResult(
            image_paths=image_paths,
            field_max_values=field_max_values,
        )


def _auto_deformation_scale(cfg: StressContour3DConfig) -> float:
    """変形倍率を自動計算."""
    if cfg.deformation_scale > 0.0:
        return cfg.deformation_scale
    max_disp = 0.0
    for u in cfg.displacement_snapshots:
        n_nodes = len(cfg.node_coords_initial)
        for i in range(n_nodes):
            for d in range(3):
                idx = i * cfg.ndof_per_node + d
                if idx < len(u):
                    max_disp = max(max_disp, abs(u[idx]))
    beam_length = np.ptp(cfg.node_coords_initial[:, 0])
    target_visual = beam_length * 0.05
    if max_disp > 1e-15:
        scale = target_visual / max_disp
    else:
        scale = 1.0
    return max(scale, 1.0)


def _compute_fixed_axis_ranges(
    node_coords: np.ndarray,
    radius: float,
) -> dict[str, tuple[float, float]]:
    """初期形状からxyz軸範囲を計算（固定用）."""
    margin = radius * 3
    x_min = float(node_coords[:, 0].min()) - margin
    x_max = float(node_coords[:, 0].max()) + margin

    # 梁長の10%をy方向に余裕として確保（変形表示用）
    beam_length = x_max - x_min
    y_margin = beam_length * 0.1
    z_margin = margin

    return {
        "x": (x_min, x_max),
        "y": (-y_margin, y_margin),
        "z": (-z_margin, z_margin),
    }


def _apply_fixed_axes(
    ax: object,
    ranges: dict[str, tuple[float, float]],
) -> None:
    """軸範囲とアスペクト比を固定."""
    ax.set_xlim(ranges["x"])
    ax.set_ylim(ranges["y"])
    ax.set_zlim(ranges["z"])
    rx = ranges["x"][1] - ranges["x"][0]
    ry = ranges["y"][1] - ranges["y"][0]
    rz = ranges["z"][1] - ranges["z"][0]
    ax.set_box_aspect([rx, ry, rz])


def _render_tube_3d(
    ax: object,
    coords: np.ndarray,
    connectivity: np.ndarray,
    element_values: np.ndarray,
    radius: float,
    n_seg: int,
    cmap: object,
    norm: object,
) -> None:
    """要素ごとにチューブを3D描画しフィールド値でカラーリング."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    theta = np.linspace(0, 2 * np.pi, n_seg + 1)

    for e in range(len(connectivity)):
        n0, n1 = int(connectivity[e, 0]), int(connectivity[e, 1])
        p0 = coords[n0]
        p1 = coords[n1]

        axis = p1 - p0
        length = np.linalg.norm(axis)
        if length < 1e-15:
            continue
        axis_dir = axis / length

        if abs(axis_dir[2]) < 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        else:
            ref = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(axis_dir, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis_dir, e1)

        circle = np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2)
        ring0 = p0 + radius * circle
        ring1 = p1 + radius * circle

        s_val = element_values[e] if e < len(element_values) else 0.0
        color = cmap(norm(abs(s_val)))

        for j in range(n_seg):
            verts = [
                [ring0[j], ring0[j + 1], ring1[j + 1], ring1[j]],
            ]
            poly = Poly3DCollection(verts, alpha=0.8)
            poly.set_facecolor(color)
            poly.set_edgecolor("none")
            ax.add_collection3d(poly)


def _render_time_history(
    cfg: StressContour3DConfig,
    output_dir: Path,
    plt: object,
    field_max_values: dict[str, float],
) -> str | None:
    """変位 + 全フィールド時刻歴プロットを出力."""
    n_snapshots = len(cfg.displacement_snapshots)
    if n_snapshots < 2:
        return None

    n_fields = len(cfg.contour_fields)
    n_plots = 1 + n_fields  # 変位 + 各フィールド

    n_nodes = len(cfg.node_coords_initial)
    mid_node = n_nodes // 2
    mid_y_dof = mid_node * cfg.ndof_per_node + 1

    t_ms = cfg.time_values * 1000

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    # 変位時刻歴
    defl = np.array([u[mid_y_dof] for u in cfg.displacement_snapshots])
    axes[0].plot(t_ms, defl, "b-", linewidth=0.8)
    axes[0].set_ylabel("Mid-span deflection [mm]")
    axes[0].set_title("Beam Oscillation Time History")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color="k", linewidth=0.5)

    # フィールド時刻歴
    colors = ["r", "g", "m", "c", "orange"]
    for i, cf in enumerate(cfg.contour_fields):
        label = _FIELD_LABELS.get(cf.name, cf.name)
        max_vals = np.array(
            [float(np.max(np.abs(s))) if len(s) > 0 else 0.0 for s in cf.snapshots]
        )
        ax = axes[i + 1]
        ax.plot(t_ms, max_vals, f"{colors[i % len(colors)]}-", linewidth=0.8)
        ax.set_ylabel(f"Max {cf.name}")
        ax.grid(True, alpha=0.3)
        # y軸にフィールドラベル
        ax.set_title(label, fontsize=9, loc="left")

    axes[-1].set_xlabel("Time [ms]")

    fig.tight_layout()
    path = str(output_dir / f"{cfg.prefix}_time_history.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
