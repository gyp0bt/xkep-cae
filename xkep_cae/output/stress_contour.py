"""StressContour3DProcess — 梁3D応力コンターレンダリング.

変形後の梁をチューブ状に3Dレンダリングし、
要素ごとの最大曲げ応力をカラーマッピングで可視化する。

出力:
  - 3Dビュー PNG（正面・斜視）
  - 時刻歴アニメーション用の連番 PNG

[← README](../../README.md)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from xkep_cae.core import MeshData, PostProcess, ProcessMeta


@dataclass(frozen=True)
class StressContour3DConfig:
    """3D応力コンターの設定."""

    mesh: MeshData
    node_coords_initial: np.ndarray  # 初期節点座標
    displacement_snapshots: list[np.ndarray]  # 各スナップショットの変位
    element_stress_snapshots: list[np.ndarray]  # 各スナップショットの要素応力
    time_values: np.ndarray  # 各スナップショットの時刻
    wire_radius: float = 1.0  # mm
    output_dir: str = "docs/verification"
    prefix: str = "beam_oscillation"
    tube_segments: int = 12  # チューブ断面の分割数
    n_render_frames: int = 6  # レンダリングするフレーム数
    ndof_per_node: int = 6
    stress_label: str = "Max Bending Stress [MPa]"


@dataclass(frozen=True)
class StressContour3DResult:
    """3D応力コンターの結果."""

    image_paths: list[str] = field(default_factory=list)
    max_stress_global: float = 0.0


class StressContour3DProcess(PostProcess[StressContour3DConfig, StressContour3DResult]):
    """梁3D応力コンターレンダリング Process.

    変形後の梁をチューブ状に3Dレンダリングし、
    曲げ応力をカラーマッピングで表示する。
    """

    meta = ProcessMeta(
        name="StressContour3D",
        module="post",
        version="1.0.0",
        document_path="docs/stress_contour.md",
    )

    def process(self, input_data: StressContour3DConfig) -> StressContour3DResult:
        """3D応力コンターレンダリングを実行."""
        cfg = input_data
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_snapshots = len(cfg.displacement_snapshots)
        if n_snapshots == 0:
            return StressContour3DResult()

        # レンダリングフレーム選択（等間隔）
        n_frames = min(cfg.n_render_frames, n_snapshots)
        frame_indices = np.linspace(0, n_snapshots - 1, n_frames, dtype=int)

        # 全スナップショットの最大応力（カラーバー統一用）
        max_stress_global = 0.0
        for stress in cfg.element_stress_snapshots:
            if len(stress) > 0:
                max_stress_global = max(max_stress_global, float(np.max(stress)))

        image_paths: list[str] = []

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize

            cmap = plt.cm.jet
            norm = Normalize(vmin=0, vmax=max_stress_global if max_stress_global > 0 else 1.0)

            for fi, snap_idx in enumerate(frame_indices):
                u = cfg.displacement_snapshots[snap_idx]
                stress = cfg.element_stress_snapshots[snap_idx]
                t_val = cfg.time_values[snap_idx] if snap_idx < len(cfg.time_values) else 0.0

                # 変形後座標
                coords = cfg.node_coords_initial.copy()
                n_nodes = len(coords)
                for i in range(n_nodes):
                    for d in range(3):
                        idx = i * cfg.ndof_per_node + d
                        if idx < len(u):
                            coords[i, d] += u[idx]

                conn = cfg.mesh.connectivity

                # 3Dチューブ生成
                fig = plt.figure(figsize=(14, 6))

                # 正面ビュー（XY平面）
                ax1 = fig.add_subplot(121, projection="3d")
                _render_tube_3d(
                    ax1,
                    coords,
                    conn,
                    stress,
                    cfg.wire_radius,
                    cfg.tube_segments,
                    cmap,
                    norm,
                )
                ax1.set_xlabel("X [mm]")
                ax1.set_ylabel("Y [mm]")
                ax1.set_zlabel("Z [mm]")
                ax1.set_title(f"Side view XY (t={t_val * 1000:.3f} ms)")
                ax1.view_init(elev=0, azim=-90)

                # 斜視ビュー
                ax2 = fig.add_subplot(122, projection="3d")
                _render_tube_3d(
                    ax2,
                    coords,
                    conn,
                    stress,
                    cfg.wire_radius,
                    cfg.tube_segments,
                    cmap,
                    norm,
                )
                ax2.set_xlabel("X [mm]")
                ax2.set_ylabel("Y [mm]")
                ax2.set_zlabel("Z [mm]")
                ax2.set_title(f"Perspective (t={t_val * 1000:.3f} ms)")
                ax2.view_init(elev=25, azim=-60)

                # カラーバー
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                fig.colorbar(sm, ax=[ax1, ax2], label=cfg.stress_label, shrink=0.6)

                fig.suptitle(
                    f"Beam Oscillation — {cfg.stress_label}\n"
                    f"t = {t_val * 1000:.3f} ms, "
                    f"σ_max = {float(np.max(stress)) if len(stress) > 0 else 0:.1f} MPa",
                    fontsize=11,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.93])

                img_path = str(output_dir / f"{cfg.prefix}_stress3d_{fi:03d}.png")
                fig.savefig(img_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                image_paths.append(img_path)

            # 時刻歴プロット（変位 + エネルギー）も追加
            _path = _render_time_history(
                cfg,
                output_dir,
                plt,
                max_stress_global,
            )
            if _path:
                image_paths.append(_path)

        except ImportError:
            pass

        return StressContour3DResult(
            image_paths=image_paths,
            max_stress_global=max_stress_global,
        )


def _render_tube_3d(
    ax: object,
    coords: np.ndarray,
    connectivity: np.ndarray,
    element_stress: np.ndarray,
    radius: float,
    n_seg: int,
    cmap: object,
    norm: object,
) -> None:
    """要素ごとにチューブを3D描画し応力でカラーリング."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    theta = np.linspace(0, 2 * np.pi, n_seg + 1)

    for e in range(len(connectivity)):
        n0, n1 = int(connectivity[e, 0]), int(connectivity[e, 1])
        p0 = coords[n0]
        p1 = coords[n1]

        # 要素軸方向
        axis = p1 - p0
        length = np.linalg.norm(axis)
        if length < 1e-15:
            continue
        axis_dir = axis / length

        # 断面法線ベクトル（軸方向に垂直な2方向）
        if abs(axis_dir[2]) < 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        else:
            ref = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(axis_dir, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis_dir, e1)

        # チューブ頂点
        circle = np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2)
        ring0 = p0 + radius * circle
        ring1 = p1 + radius * circle

        # 応力に基づく色
        s_val = element_stress[e] if e < len(element_stress) else 0.0
        color = cmap(norm(s_val))

        # 側面パッチ
        for j in range(n_seg):
            verts = [
                [ring0[j], ring0[j + 1], ring1[j + 1], ring1[j]],
            ]
            poly = Poly3DCollection(verts, alpha=0.8)
            poly.set_facecolor(color)
            poly.set_edgecolor("none")
            ax.add_collection3d(poly)

    # 軸スケーリング
    all_x = coords[:, 0]
    all_y = coords[:, 1]
    margin = radius * 3
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(
        min(all_y.min() - margin, -margin),
        max(all_y.max() + margin, margin),
    )
    ax.set_zlim(-margin, margin)


def _render_time_history(
    cfg: StressContour3DConfig,
    output_dir: Path,
    plt: object,
    max_stress: float,
) -> str | None:
    """変位時刻歴 + 応力時刻歴プロットを出力."""
    n_snapshots = len(cfg.displacement_snapshots)
    if n_snapshots < 2:
        return None

    # 中央節点の y 変位
    n_nodes = len(cfg.node_coords_initial)
    mid_node = n_nodes // 2
    mid_y_dof = mid_node * cfg.ndof_per_node + 1

    t_ms = cfg.time_values * 1000  # ms 変換
    defl = np.array([u[mid_y_dof] for u in cfg.displacement_snapshots])
    max_s = np.array(
        [float(np.max(s)) if len(s) > 0 else 0.0 for s in cfg.element_stress_snapshots]
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(t_ms, defl, "b-", linewidth=0.8)
    ax1.set_ylabel("Mid-span deflection [mm]")
    ax1.set_title("Beam Oscillation Time History")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", linewidth=0.5)

    ax2.plot(t_ms, max_s, "r-", linewidth=0.8)
    ax2.set_ylabel("Max bending stress [MPa]")
    ax2.set_xlabel("Time [ms]")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = str(output_dir / f"{cfg.prefix}_time_history.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
