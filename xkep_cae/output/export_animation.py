"""FIELD ANIMATION エクスポート（梁要素のアニメーション出力）.

xkep-cae独自のアニメーション出力機能。
梁要素をx,y,z軸方向から見た二次元プロットを生成する。

対応範囲（現行バージョン）:
  - 梁要素（B21, B22, B31, B32）の線図描画
  - 要素セット（ELSET）ごとの色分け・凡例表示
  - xy, xz, yz の3ビュー方向

出力:
  - PNG画像ファイル（各ビュー方向ごと、各フレームごと）
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xkep_cae.io.abaqus_inp import AbaqusMesh

# 梁要素タイプの判定セット
_BEAM_TYPES = {"B21", "B22", "B31", "B32"}

# ビュー方向 → 座標軸インデックスのマッピング
_VIEW_AXES: dict[str, tuple[int, int, str, str]] = {
    "xy": (0, 1, "X", "Y"),
    "xz": (0, 2, "X", "Z"),
    "yz": (1, 2, "Y", "Z"),
}

# 要素セットの描画色パレット
_COLORS = [
    "#1f77b4",  # 青
    "#ff7f0e",  # 橙
    "#2ca02c",  # 緑
    "#d62728",  # 赤
    "#9467bd",  # 紫
    "#8c564b",  # 茶
    "#e377c2",  # ピンク
    "#7f7f7f",  # 灰
    "#bcbd22",  # 黄緑
    "#17becf",  # 水色
]


def _collect_beam_segments(
    mesh: AbaqusMesh,
    node_coords: np.ndarray | None = None,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """梁要素のセグメントを要素セット名ごとに収集する.

    Args:
        mesh: AbaqusMesh オブジェクト
        node_coords: 変形後の節点座標 (n_nodes, ndim)。Noneの場合はmeshの節点座標を使用

    Returns:
        {elset_name: [(start_xyz, end_xyz), ...]} の辞書
    """
    # ノードラベル → インデックス & 座標のマッピング構築
    label_to_idx: dict[int, int] = {}
    if node_coords is None:
        coords = np.zeros((len(mesh.nodes), 3))
        for i, node in enumerate(mesh.nodes):
            label_to_idx[node.label] = i
            coords[i] = [node.x, node.y, node.z]
    else:
        for i, node in enumerate(mesh.nodes):
            label_to_idx[node.label] = i
        # node_coords が 2D の場合は3Dに拡張
        if node_coords.shape[1] == 2:
            coords = np.zeros((node_coords.shape[0], 3))
            coords[:, :2] = node_coords
        else:
            coords = node_coords

    segments: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    for group in mesh.element_groups:
        if group.elem_type.upper() not in _BEAM_TYPES:
            continue

        elset_name = group.elset or "default"

        if elset_name not in segments:
            segments[elset_name] = []

        for _label, node_ids in group.elements:
            # 梁要素は少なくとも2節点（B21/B31は2節点、B22/B32は3節点）
            # 端点のみ使用（中間節点は無視して直線描画）
            n0 = label_to_idx.get(node_ids[0])
            n1 = label_to_idx.get(node_ids[-1])
            if n0 is not None and n1 is not None:
                segments[elset_name].append((coords[n0], coords[n1]))

    return segments


def render_beam_animation_frame(
    mesh: AbaqusMesh,
    view: str = "xy",
    *,
    node_coords: np.ndarray | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
    dpi: int = 100,
    margin_ratio: float = 0.1,
) -> tuple:
    """梁要素の1フレームを描画する.

    Args:
        mesh: AbaqusMesh オブジェクト
        view: ビュー方向 ("xy", "xz", "yz")
        node_coords: 変形後の節点座標。Noneの場合はmeshの初期座標を使用
        title: 図のタイトル
        figsize: 図のサイズ (幅, 高さ) インチ
        dpi: 解像度
        margin_ratio: 全要素が画面に収まるようにするマージン比率

    Returns:
        (fig, ax) matplotlib のFigureとAxesオブジェクト
    """
    import matplotlib.pyplot as plt

    if view not in _VIEW_AXES:
        raise ValueError(f"未対応のビュー方向: {view}。対応: {list(_VIEW_AXES.keys())}")

    ax_h, ax_v, label_h, label_v = _VIEW_AXES[view]

    segments = _collect_beam_segments(mesh, node_coords)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    color_idx = 0
    all_h: list[float] = []
    all_v: list[float] = []

    for elset_name, segs in segments.items():
        color = _COLORS[color_idx % len(_COLORS)]
        color_idx += 1

        for i, (p0, p1) in enumerate(segs):
            h_vals = [p0[ax_h], p1[ax_h]]
            v_vals = [p0[ax_v], p1[ax_v]]
            all_h.extend(h_vals)
            all_v.extend(v_vals)
            ax.plot(
                h_vals,
                v_vals,
                color=color,
                linewidth=2.0,
                label=elset_name if i == 0 else None,
            )

    # 全要素が画面に収まるようにビュー設定
    if all_h and all_v:
        h_min, h_max = min(all_h), max(all_h)
        v_min, v_max = min(all_v), max(all_v)

        h_range = h_max - h_min if h_max > h_min else 1.0
        v_range = v_max - v_min if v_max > v_min else 1.0

        margin_h = h_range * margin_ratio
        margin_v = v_range * margin_ratio

        ax.set_xlim(h_min - margin_h, h_max + margin_h)
        ax.set_ylim(v_min - margin_v, v_max + margin_v)

    ax.set_xlabel(label_h)
    ax.set_ylabel(label_v)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Beam elements — {view.upper()} view")

    if segments:
        ax.legend(loc="best", framealpha=0.8)

    fig.tight_layout()
    return fig, ax


def export_field_animation(
    mesh: AbaqusMesh,
    output_dir: str | Path = "animation",
    views: list[str] | None = None,
    *,
    node_coords_frames: list[np.ndarray] | None = None,
    frame_labels: list[str] | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
    dpi: int = 100,
) -> list[Path]:
    """梁要素のアニメーションフレームをPNG画像として出力する.

    Args:
        mesh: AbaqusMesh オブジェクト
        output_dir: 出力ディレクトリパス
        views: 描画するビュー方向リスト。Noneの場合 ["xy", "xz", "yz"]
        node_coords_frames: フレームごとの節点座標リスト。
            Noneの場合は初期配置のみ1フレーム出力
        frame_labels: フレームのラベルリスト（タイトルに使用）
        figsize: 図のサイズ
        dpi: 解像度

    Returns:
        出力されたPNGファイルのパスリスト
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if views is None:
        views = ["xy", "xz", "yz"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    output_files: list[Path] = []

    if node_coords_frames is None:
        # 初期配置のみ1フレーム
        node_coords_frames = [None]
        if frame_labels is None:
            frame_labels = ["Initial"]

    if frame_labels is None:
        frame_labels = [f"Frame {i}" for i in range(len(node_coords_frames))]

    for frame_idx, (coords, label) in enumerate(zip(node_coords_frames, frame_labels, strict=True)):
        for view in views:
            title = f"{label} — {view.upper()} view"
            fig, _ax = render_beam_animation_frame(
                mesh,
                view=view,
                node_coords=coords,
                title=title,
                figsize=figsize,
                dpi=dpi,
            )

            fname = f"frame_{frame_idx:04d}_{view}.png"
            fpath = out_path / fname
            fig.savefig(fpath, bbox_inches="tight")
            plt.close(fig)
            output_files.append(fpath)

    return output_files


__all__ = [
    "export_field_animation",
    "render_beam_animation_frame",
]
