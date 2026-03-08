"""3D梁レンダリング — 円形断面付きチューブ表面描画.

梁要素の中心線から円筒チューブメッシュを生成し、
matplotlib mplot3d の plot_surface でレンダリングする。
2D投影の線図ではなく、3D表面として断面形状が視認できる描画を提供する。

対応:
  - TwistedWireMesh の各素線を円筒チューブとして描画
  - 任意視角（elev/azim）でのレンダリング
  - 素線ごとの色分け
  - 端面キャップ（円形断面の閉鎖）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# 素線の描画色パレット（10色サイクル）
_STRAND_COLORS = [
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


def _make_tube_mesh(
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    n_circ: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2点間の円筒チューブメッシュを生成する.

    Args:
        p0: 始点 (3,)
        p1: 終点 (3,)
        radius: 円筒半径
        n_circ: 円周方向の分割数

    Returns:
        (X, Y, Z) 各 (2, n_circ+1) のメッシュグリッド
    """
    # 軸方向ベクトル
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-15:
        # 退化要素（長さゼロ）→ 空メッシュ
        empty = np.zeros((2, n_circ + 1))
        return empty, empty, empty

    axis_unit = axis / length

    # 軸に垂直な2ベクトルを構築
    # axis_unit と最も非平行な座標軸を選択
    abs_ax = np.abs(axis_unit)
    if abs_ax[0] <= abs_ax[1] and abs_ax[0] <= abs_ax[2]:
        ref = np.array([1.0, 0.0, 0.0])
    elif abs_ax[1] <= abs_ax[2]:
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])

    e1 = np.cross(axis_unit, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis_unit, e1)

    # 円周方向の角度（閉じたループ）
    theta = np.linspace(0, 2 * np.pi, n_circ + 1)

    # 2断面（始点・終点）× 円周方向
    X = np.zeros((2, n_circ + 1))
    Y = np.zeros((2, n_circ + 1))
    Z = np.zeros((2, n_circ + 1))

    for i, center in enumerate([p0, p1]):
        for j, t in enumerate(theta):
            pt = center + radius * (np.cos(t) * e1 + np.sin(t) * e2)
            X[i, j] = pt[0]
            Y[i, j] = pt[1]
            Z[i, j] = pt[2]

    return X, Y, Z


def _make_cap_mesh(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    n_circ: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """端面キャップ（円盤）のメッシュを生成する.

    Args:
        center: 中心点 (3,)
        normal: 法線方向 (3,)
        radius: 半径
        n_circ: 円周方向の分割数

    Returns:
        (X, Y, Z) 各 (2, n_circ+1) のメッシュグリッド
    """
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-15:
        empty = np.zeros((2, n_circ + 1))
        return empty, empty, empty

    axis_unit = normal / norm_len

    abs_ax = np.abs(axis_unit)
    if abs_ax[0] <= abs_ax[1] and abs_ax[0] <= abs_ax[2]:
        ref = np.array([1.0, 0.0, 0.0])
    elif abs_ax[1] <= abs_ax[2]:
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])

    e1 = np.cross(axis_unit, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis_unit, e1)

    theta = np.linspace(0, 2 * np.pi, n_circ + 1)

    X = np.zeros((2, n_circ + 1))
    Y = np.zeros((2, n_circ + 1))
    Z = np.zeros((2, n_circ + 1))

    # 中心点（第0行）
    X[0, :] = center[0]
    Y[0, :] = center[1]
    Z[0, :] = center[2]

    # 円周上の点（第1行）
    for j, t in enumerate(theta):
        pt = center + radius * (np.cos(t) * e1 + np.sin(t) * e2)
        X[1, j] = pt[0]
        Y[1, j] = pt[1]
        Z[1, j] = pt[2]

    return X, Y, Z


def render_beam_3d(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    wire_radius: float,
    *,
    strand_node_ranges: list[tuple[int, int]] | None = None,
    n_strands: int = 1,
    elev: float = 25.0,
    azim: float = -60.0,
    title: str = "",
    figsize: tuple[float, float] = (12.0, 10.0),
    dpi: int = 150,
    n_circ: int = 12,
    show_caps: bool = True,
    scale_mm: bool = True,
    bg_color: str = "#f0f0f0",
) -> tuple[Figure, Axes]:
    """梁要素を3D円筒チューブとしてレンダリングする.

    Args:
        node_coords: (n_nodes, 3) 節点座標 [m]
        connectivity: (n_elems, 2) 要素接続
        wire_radius: 素線断面半径 [m]
        strand_node_ranges: 各素線の節点範囲 [(start, end), ...]
        n_strands: 素線数
        elev: カメラ仰角 [deg]
        azim: カメラ方位角 [deg]
        title: 図タイトル
        figsize: 図サイズ
        dpi: 解像度
        n_circ: 円周方向分割数（大きいほど滑らか）
        show_caps: 端面キャップを描画するか
        scale_mm: Trueならmm表示、Falseならm表示
        bg_color: 背景色

    Returns:
        (fig, ax) matplotlibのFigure と 3D Axes
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    scale = 1000.0 if scale_mm else 1.0
    unit = "mm" if scale_mm else "m"
    r = wire_radius * scale

    coords = node_coords * scale

    # 素線ごとの要素分類
    if strand_node_ranges is not None:
        # strand_node_ranges から各要素がどの素線に属するか判定
        elem_strand = np.full(len(connectivity), -1, dtype=int)
        for sid, (ns, ne) in enumerate(strand_node_ranges):
            for eidx, (n0, n1) in enumerate(connectivity):
                if ns <= n0 < ne and ns <= n1 < ne:
                    elem_strand[eidx] = sid
    else:
        # 全要素を均等に分配
        elems_per = max(1, len(connectivity) // n_strands)
        elem_strand = np.array(
            [min(i // elems_per, n_strands - 1) for i in range(len(connectivity))]
        )

    # 素線ごとにレンダリング
    for sid in range(n_strands):
        color = _STRAND_COLORS[sid % len(_STRAND_COLORS)]
        mask = elem_strand == sid
        elem_indices = np.where(mask)[0]

        for eidx in elem_indices:
            n0, n1 = connectivity[eidx]
            p0 = coords[n0]
            p1 = coords[n1]

            X, Y, Z = _make_tube_mesh(p0, p1, r, n_circ)
            ax.plot_surface(
                X,
                Y,
                Z,
                color=color,
                alpha=0.85,
                shade=True,
                linewidth=0,
                antialiased=True,
            )

        # 端面キャップ（最初と最後の要素端点）
        if show_caps and len(elem_indices) > 0:
            # 最初の要素の始点
            first_elem = elem_indices[0]
            n0_first = connectivity[first_elem, 0]
            n1_first = connectivity[first_elem, 1]
            normal_first = coords[n1_first] - coords[n0_first]
            Xc, Yc, Zc = _make_cap_mesh(coords[n0_first], normal_first, r, n_circ)
            ax.plot_surface(
                Xc,
                Yc,
                Zc,
                color=color,
                alpha=0.9,
                shade=True,
                linewidth=0,
                antialiased=True,
            )

            # 最後の要素の終点
            last_elem = elem_indices[-1]
            n0_last = connectivity[last_elem, 0]
            n1_last = connectivity[last_elem, 1]
            normal_last = coords[n1_last] - coords[n0_last]
            Xc, Yc, Zc = _make_cap_mesh(coords[n1_last], normal_last, r, n_circ)
            ax.plot_surface(
                Xc,
                Yc,
                Zc,
                color=color,
                alpha=0.9,
                shade=True,
                linewidth=0,
                antialiased=True,
            )

    ax.set_xlabel(f"X [{unit}]")
    ax.set_ylabel(f"Y [{unit}]")
    ax.set_zlabel(f"Z [{unit}]")
    ax.view_init(elev=elev, azim=azim)

    if title:
        ax.set_title(title, fontsize=12, pad=15)

    # アスペクト比を等軸に
    _set_equal_aspect_3d(ax, coords)

    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    return fig, ax


def _set_equal_aspect_3d(ax: Axes, coords: np.ndarray) -> None:
    """3D軸のアスペクト比を等軸にする."""
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    max_range = max(x_range, y_range, z_range)
    if max_range < 1e-15:
        max_range = 1.0

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)


def render_twisted_wire_3d(
    mesh,
    *,
    node_coords: np.ndarray | None = None,
    elev: float = 25.0,
    azim: float = -60.0,
    title: str = "",
    figsize: tuple[float, float] = (12.0, 10.0),
    dpi: int = 150,
    n_circ: int = 12,
    show_caps: bool = True,
) -> tuple[Figure, Axes]:
    """TwistedWireMeshを3Dチューブとしてレンダリングする（便利関数）.

    Args:
        mesh: TwistedWireMesh オブジェクト
        node_coords: 変形後の節点座標。Noneの場合はmeshの初期座標を使用
        elev: カメラ仰角 [deg]
        azim: カメラ方位角 [deg]
        title: 図タイトル
        figsize: 図サイズ
        dpi: 解像度
        n_circ: 円周方向分割数
        show_caps: 端面キャップを描画するか

    Returns:
        (fig, ax) matplotlibのFigure と 3D Axes
    """
    coords = node_coords if node_coords is not None else mesh.node_coords

    return render_beam_3d(
        node_coords=coords,
        connectivity=mesh.connectivity,
        wire_radius=mesh.wire_radius,
        strand_node_ranges=mesh.strand_node_ranges,
        n_strands=mesh.n_strands,
        elev=elev,
        azim=azim,
        title=title,
        figsize=figsize,
        dpi=dpi,
        n_circ=n_circ,
        show_caps=show_caps,
    )


# 標準視角プリセット
VIEW_PRESETS: dict[str, dict[str, float | str]] = {
    "isometric": {"elev": 25.0, "azim": -60.0, "label": "Isometric"},
    "front_xy": {"elev": 0.0, "azim": -90.0, "label": "Front (XY)"},
    "side_xz": {"elev": 0.0, "azim": 0.0, "label": "Side (XZ)"},
    "end_yz": {"elev": 0.0, "azim": -180.0, "label": "End (YZ)"},
    "top_down": {"elev": 90.0, "azim": -90.0, "label": "Top-down"},
    "oblique_30": {"elev": 20.0, "azim": -30.0, "label": "Oblique 30deg"},
    "oblique_60": {"elev": 20.0, "azim": -60.0, "label": "Oblique 60deg"},
    "bird_eye": {"elev": 45.0, "azim": -45.0, "label": "Bird's eye"},
}


def render_multiview_3d(
    mesh,
    *,
    node_coords: np.ndarray | None = None,
    views: list[str] | None = None,
    title_prefix: str = "",
    figsize_per_view: tuple[float, float] = (8.0, 7.0),
    dpi: int = 150,
    n_circ: int = 12,
) -> list[tuple[str, Figure, Axes]]:
    """複数視角から3Dレンダリングを生成する.

    Args:
        mesh: TwistedWireMesh オブジェクト
        node_coords: 変形後の節点座標。Noneの場合はmeshの初期座標を使用
        views: 視角名リスト (VIEW_PRESETS のキー)。Noneの場合は全プリセット
        title_prefix: タイトルの接頭辞
        figsize_per_view: 各ビューの図サイズ
        dpi: 解像度
        n_circ: 円周方向分割数

    Returns:
        [(view_name, fig, ax), ...] のリスト
    """
    if views is None:
        views = list(VIEW_PRESETS.keys())

    results = []
    for vname in views:
        preset = VIEW_PRESETS[vname]
        title = f"{title_prefix} — {preset['label']}" if title_prefix else str(preset["label"])
        fig, ax = render_twisted_wire_3d(
            mesh,
            node_coords=node_coords,
            elev=float(preset["elev"]),
            azim=float(preset["azim"]),
            title=title,
            figsize=figsize_per_view,
            dpi=dpi,
            n_circ=n_circ,
        )
        results.append((vname, fig, ax))

    return results


__all__ = [
    "VIEW_PRESETS",
    "render_beam_3d",
    "render_multiview_3d",
    "render_twisted_wire_3d",
]
