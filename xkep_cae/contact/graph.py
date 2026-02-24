"""接触グラフ表現.

多点接触の状態を無向グラフとして表現し、
時間ステップごとのトポロジー変遷を追跡する。

== グラフ表現 ==

ノード: 梁要素（セグメント）
エッジ: 活性接触ペア（ACTIVE or SLIDING）
エッジ属性:
  - gap: 法線方向ギャップ
  - p_n: 法線反力
  - status: 接触状態（ACTIVE/SLIDING）
  - stick: stick/slip フラグ
  - dissipation: 散逸エネルギー増分

== 時系列グラフ ==

ContactGraphHistory は各ステップのスナップショットを保持し、
接触トポロジーの時間発展を分析する機能を提供する。

用途:
  - 撚線モデルの素線間接触パターン分析
  - 接触ペア数の時間推移
  - 接触力分布の可視化データ生成
  - 接触ネットワークの連結性分析
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from xkep_cae.contact.pair import ContactManager, ContactStatus


@dataclass
class ContactEdge:
    """接触グラフのエッジ（1接触ペア）.

    Attributes:
        elem_a: 要素Aインデックス
        elem_b: 要素Bインデックス
        gap: 法線方向ギャップ
        p_n: 法線反力
        status: 接触状態
        stick: stick/slip
        dissipation: 散逸エネルギー増分
        s: セグメントA上の最近接パラメータ
        t: セグメントB上の最近接パラメータ
    """

    elem_a: int
    elem_b: int
    gap: float
    p_n: float
    status: str  # "ACTIVE", "SLIDING"
    stick: bool
    dissipation: float
    s: float
    t: float


@dataclass
class ContactGraph:
    """接触グラフの1スナップショット.

    Attributes:
        step: ステップ番号
        load_factor: 荷重係数 (0~1)
        nodes: アクティブな要素インデックスの集合
        edges: 接触エッジのリスト
        n_total_pairs: 全接触ペア数（非活性含む）
    """

    step: int
    load_factor: float
    nodes: set[int]
    edges: list[ContactEdge]
    n_total_pairs: int

    @property
    def n_nodes(self) -> int:
        """アクティブノード数."""
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        """アクティブエッジ数."""
        return len(self.edges)

    @property
    def n_active(self) -> int:
        """ACTIVE 状態のエッジ数."""
        return sum(1 for e in self.edges if e.status == "ACTIVE")

    @property
    def n_sliding(self) -> int:
        """SLIDING 状態のエッジ数."""
        return sum(1 for e in self.edges if e.status == "SLIDING")

    @property
    def total_normal_force(self) -> float:
        """法線反力の合計."""
        return sum(e.p_n for e in self.edges)

    @property
    def total_dissipation(self) -> float:
        """散逸エネルギー増分の合計."""
        return sum(e.dissipation for e in self.edges)

    def degree_map(self) -> dict[int, int]:
        """各ノードの次数（接触ペア数）を返す."""
        deg: dict[int, int] = {}
        for e in self.edges:
            deg[e.elem_a] = deg.get(e.elem_a, 0) + 1
            deg[e.elem_b] = deg.get(e.elem_b, 0) + 1
        return deg

    def adjacency_list(self) -> dict[int, list[int]]:
        """隣接リストを返す."""
        adj: dict[int, list[int]] = {}
        for e in self.edges:
            adj.setdefault(e.elem_a, []).append(e.elem_b)
            adj.setdefault(e.elem_b, []).append(e.elem_a)
        return adj

    def connected_components(self) -> list[set[int]]:
        """連結成分を返す（BFS）."""
        visited: set[int] = set()
        components: list[set[int]] = []
        adj = self.adjacency_list()

        for node in self.nodes:
            if node in visited:
                continue
            # BFS
            component: set[int] = set()
            queue = [node]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in adj.get(current, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if component:
                components.append(component)

        return components

    def to_adjacency_matrix(self, n_elems: int | None = None) -> np.ndarray:
        """隣接行列を返す.

        Args:
            n_elems: 全要素数。None の場合は max(nodes)+1 を使用。

        Returns:
            (n_elems, n_elems) の隣接行列。値は法線反力。
        """
        if n_elems is None:
            n_elems = max(self.nodes) + 1 if self.nodes else 0
        A = np.zeros((n_elems, n_elems))
        for e in self.edges:
            A[e.elem_a, e.elem_b] = e.p_n
            A[e.elem_b, e.elem_a] = e.p_n
        return A

    def to_dict(self) -> dict:
        """辞書表現を返す（JSON出力用）."""
        return {
            "step": self.step,
            "load_factor": self.load_factor,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_active": self.n_active,
            "n_sliding": self.n_sliding,
            "total_normal_force": self.total_normal_force,
            "total_dissipation": self.total_dissipation,
            "edges": [
                {
                    "elem_a": e.elem_a,
                    "elem_b": e.elem_b,
                    "gap": e.gap,
                    "p_n": e.p_n,
                    "status": e.status,
                    "stick": e.stick,
                    "dissipation": e.dissipation,
                    "s": e.s,
                    "t": e.t,
                }
                for e in self.edges
            ],
        }


def snapshot_contact_graph(
    manager: ContactManager,
    step: int = 0,
    load_factor: float = 0.0,
) -> ContactGraph:
    """ContactManager の現在の状態からグラフのスナップショットを生成する.

    Args:
        manager: 接触マネージャ
        step: ステップ番号
        load_factor: 荷重係数

    Returns:
        ContactGraph インスタンス
    """
    nodes: set[int] = set()
    edges: list[ContactEdge] = []

    for pair in manager.pairs:
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        status_str = "SLIDING" if pair.state.status == ContactStatus.SLIDING else "ACTIVE"

        nodes.add(pair.elem_a)
        nodes.add(pair.elem_b)
        edges.append(
            ContactEdge(
                elem_a=pair.elem_a,
                elem_b=pair.elem_b,
                gap=pair.state.gap,
                p_n=pair.state.p_n,
                status=status_str,
                stick=pair.state.stick,
                dissipation=pair.state.dissipation,
                s=pair.state.s,
                t=pair.state.t,
            )
        )

    return ContactGraph(
        step=step,
        load_factor=load_factor,
        nodes=nodes,
        edges=edges,
        n_total_pairs=manager.n_pairs,
    )


@dataclass
class ContactGraphHistory:
    """接触グラフの時系列.

    各ステップのスナップショットを保持し、
    接触トポロジーの時間発展を分析する。

    Attributes:
        snapshots: ステップごとのグラフスナップショット
    """

    snapshots: list[ContactGraph] = field(default_factory=list)

    def add(self, graph: ContactGraph) -> None:
        """スナップショットを追加する."""
        self.snapshots.append(graph)

    @property
    def n_steps(self) -> int:
        """ステップ数."""
        return len(self.snapshots)

    def edge_count_series(self) -> np.ndarray:
        """各ステップのエッジ数の時系列."""
        return np.array([g.n_edges for g in self.snapshots])

    def node_count_series(self) -> np.ndarray:
        """各ステップのノード数の時系列."""
        return np.array([g.n_nodes for g in self.snapshots])

    def total_force_series(self) -> np.ndarray:
        """各ステップの法線反力合計の時系列."""
        return np.array([g.total_normal_force for g in self.snapshots])

    def dissipation_series(self) -> np.ndarray:
        """各ステップの散逸エネルギーの時系列."""
        return np.array([g.total_dissipation for g in self.snapshots])

    def load_factor_series(self) -> np.ndarray:
        """荷重係数の時系列."""
        return np.array([g.load_factor for g in self.snapshots])

    def topology_change_steps(self) -> list[int]:
        """接触トポロジーが変化したステップのリスト.

        エッジ集合（elem_a, elem_b のペア）が前ステップと異なるステップを返す。
        """
        changes: list[int] = []
        prev_edge_set: set[tuple[int, int]] | None = None

        for g in self.snapshots:
            edge_set = {(e.elem_a, e.elem_b) for e in g.edges}
            if prev_edge_set is not None and edge_set != prev_edge_set:
                changes.append(g.step)
            prev_edge_set = edge_set

        return changes

    def to_dict_list(self) -> list[dict]:
        """全スナップショットの辞書リスト（JSON出力用）."""
        return [g.to_dict() for g in self.snapshots]

    # -- 統計分析メソッド --

    def stick_slip_ratio_series(self) -> np.ndarray:
        """各ステップの stick/(stick+slip) 比率の時系列.

        全エッジが stick の場合 1.0、全 slip の場合 0.0。
        エッジがない場合は 1.0（デフォルト）。
        """
        ratios = []
        for g in self.snapshots:
            n_total = len(g.edges)
            if n_total == 0:
                ratios.append(1.0)
                continue
            n_stick = sum(1 for e in g.edges if e.stick)
            ratios.append(n_stick / n_total)
        return np.array(ratios)

    def mean_normal_force_series(self) -> np.ndarray:
        """各ステップの平均法線反力の時系列.

        アクティブエッジがない場合は 0.0。
        """
        means = []
        for g in self.snapshots:
            if not g.edges:
                means.append(0.0)
            else:
                means.append(sum(e.p_n for e in g.edges) / len(g.edges))
        return np.array(means)

    def max_normal_force_series(self) -> np.ndarray:
        """各ステップの最大法線反力の時系列."""
        maxes = []
        for g in self.snapshots:
            if not g.edges:
                maxes.append(0.0)
            else:
                maxes.append(max(e.p_n for e in g.edges))
        return np.array(maxes)

    def connected_component_count_series(self) -> np.ndarray:
        """各ステップの連結成分数の時系列."""
        return np.array([len(g.connected_components()) for g in self.snapshots])

    def contact_duration_map(self) -> dict[tuple[int, int], int]:
        """各エッジ (elem_a, elem_b) が接触していたステップ数.

        全ステップを走査し、エッジが出現したステップ数をカウントする。
        """
        durations: dict[tuple[int, int], int] = {}
        for g in self.snapshots:
            for e in g.edges:
                key = (min(e.elem_a, e.elem_b), max(e.elem_a, e.elem_b))
                durations[key] = durations.get(key, 0) + 1
        return durations

    def cumulative_dissipation_series(self) -> np.ndarray:
        """散逸エネルギーの累積時系列."""
        diss = self.dissipation_series()
        return np.cumsum(diss)

    def summary(self) -> dict:
        """時系列の要約統計を辞書で返す.

        Returns:
            dict: 各キーに統計値を格納
                - n_steps: ステップ数
                - max_edges: 最大エッジ数
                - max_nodes: 最大ノード数
                - max_normal_force: 全ステップ中の最大法線反力
                - total_dissipation: 散逸エネルギー合計
                - n_topology_changes: トポロジー変化回数
                - unique_contacts: ユニークな接触ペア数
                - mean_stick_ratio: 平均 stick 比率
        """
        if not self.snapshots:
            return {
                "n_steps": 0,
                "max_edges": 0,
                "max_nodes": 0,
                "max_normal_force": 0.0,
                "total_dissipation": 0.0,
                "n_topology_changes": 0,
                "unique_contacts": 0,
                "mean_stick_ratio": 1.0,
            }

        edges = self.edge_count_series()
        nodes = self.node_count_series()
        forces = self.max_normal_force_series()
        diss = self.dissipation_series()
        stick_ratios = self.stick_slip_ratio_series()
        topo_changes = self.topology_change_steps()
        durations = self.contact_duration_map()

        return {
            "n_steps": self.n_steps,
            "max_edges": int(edges.max()) if len(edges) > 0 else 0,
            "max_nodes": int(nodes.max()) if len(nodes) > 0 else 0,
            "max_normal_force": float(forces.max()) if len(forces) > 0 else 0.0,
            "total_dissipation": float(diss.sum()),
            "n_topology_changes": len(topo_changes),
            "unique_contacts": len(durations),
            "mean_stick_ratio": float(stick_ratios.mean()) if len(stick_ratios) > 0 else 1.0,
        }


# ====================================================================
# 可視化関数
# ====================================================================


def plot_contact_graph(
    graph: ContactGraph,
    *,
    ax: object | None = None,
    node_positions: dict[int, tuple[float, float]] | None = None,
    title: str | None = None,
    show_force: bool = True,
    node_size: float = 300,
    edge_width_scale: float = 2.0,
    figsize: tuple[float, float] = (8, 6),
) -> object:
    """接触グラフを matplotlib で描画する.

    ノードを2D平面に配置し、接触エッジを法線反力の大きさに応じた
    太さで描画する。

    Args:
        graph: 接触グラフ
        ax: matplotlib Axes（None なら新規作成）
        node_positions: {elem_id: (x, y)} の辞書（None なら自動配置）
        title: タイトル（None なら自動生成）
        show_force: エッジに法線反力を表示
        node_size: ノードのマーカーサイズ
        edge_width_scale: エッジ太さの基本スケール
        figsize: 図のサイズ

    Returns:
        matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("matplotlib が必要です: pip install matplotlib") from err

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if not graph.nodes:
        ax.text(0.5, 0.5, "接触なし", ha="center", va="center", transform=ax.transAxes)
        if title:
            ax.set_title(title)
        return ax

    # ノード位置の自動配置（円形レイアウト）
    if node_positions is None:
        node_positions = _circular_layout(graph.nodes)

    # エッジ描画
    max_pn = max((e.p_n for e in graph.edges), default=1.0)
    if max_pn < 1e-30:
        max_pn = 1.0

    for edge in graph.edges:
        if edge.elem_a not in node_positions or edge.elem_b not in node_positions:
            continue
        x1, y1 = node_positions[edge.elem_a]
        x2, y2 = node_positions[edge.elem_b]

        width = edge_width_scale * (edge.p_n / max_pn + 0.1)
        color = "tab:red" if edge.status == "SLIDING" else "tab:blue"
        alpha = 0.7

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=alpha, zorder=1)

        if show_force and edge.p_n > 0:
            mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            ax.text(mx, my, f"{edge.p_n:.1e}", fontsize=6, ha="center", va="center", zorder=3)

    # ノード描画
    xs = [node_positions[n][0] for n in graph.nodes if n in node_positions]
    ys = [node_positions[n][1] for n in graph.nodes if n in node_positions]
    labels = [str(n) for n in graph.nodes if n in node_positions]

    ax.scatter(xs, ys, s=node_size, c="tab:green", edgecolors="black", linewidths=1, zorder=2)
    for x, y, lbl in zip(xs, ys, labels, strict=True):
        ax.text(x, y, lbl, fontsize=7, ha="center", va="center", zorder=4)

    if title is None:
        title = f"Step {graph.step}, λ={graph.load_factor:.3f}, edges={graph.n_edges}"
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def plot_contact_graph_history(
    history: ContactGraphHistory,
    *,
    figsize: tuple[float, float] = (12, 8),
) -> object:
    """接触グラフ時系列の要約プロットを描画する.

    4パネル: エッジ数、ノード数、法線反力合計、散逸エネルギー。

    Args:
        history: 接触グラフ時系列
        figsize: 図のサイズ

    Returns:
        matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("matplotlib が必要です: pip install matplotlib") from err

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    lf = history.load_factor_series()

    # エッジ数
    ax = axes[0, 0]
    ax.plot(lf, history.edge_count_series(), "o-", markersize=3)
    ax.set_ylabel("接触エッジ数")
    ax.set_xlabel("荷重係数 λ")
    ax.set_title("接触ペア数の推移")
    ax.grid(True, alpha=0.3)

    # ノード数
    ax = axes[0, 1]
    ax.plot(lf, history.node_count_series(), "s-", markersize=3, color="tab:orange")
    ax.set_ylabel("接触ノード数")
    ax.set_xlabel("荷重係数 λ")
    ax.set_title("接触要素数の推移")
    ax.grid(True, alpha=0.3)

    # 法線反力合計
    ax = axes[1, 0]
    ax.plot(lf, history.total_force_series(), "^-", markersize=3, color="tab:red")
    ax.set_ylabel("法線反力合計 [N]")
    ax.set_xlabel("荷重係数 λ")
    ax.set_title("接触力の推移")
    ax.grid(True, alpha=0.3)

    # 散逸エネルギー
    ax = axes[1, 1]
    ax.plot(lf, history.dissipation_series(), "d-", markersize=3, color="tab:purple")
    ax.set_ylabel("散逸エネルギー [J]")
    ax.set_xlabel("荷重係数 λ")
    ax.set_title("摩擦散逸の推移")
    ax.grid(True, alpha=0.3)

    # トポロジー変化ステップを縦線で表示
    change_steps = set(history.topology_change_steps())
    for g in history.snapshots:
        if g.step in change_steps:
            for a in axes.flat:
                a.axvline(g.load_factor, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)

    fig.tight_layout()
    return fig


def save_contact_graph_gif(
    history: ContactGraphHistory,
    filepath: str,
    *,
    node_positions: dict[int, tuple[float, float]] | None = None,
    fps: int = 2,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 80,
) -> None:
    """接触グラフ時系列を GIF アニメーションとして保存する.

    Args:
        history: 接触グラフ時系列
        filepath: 出力ファイルパス (.gif)
        node_positions: 固定ノード位置（None なら全ステップで統一した自動配置）
        fps: フレームレート
        figsize: 1フレームの図サイズ
        dpi: 解像度
    """
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as err:
        raise ImportError("matplotlib と Pillow が必要です") from err

    if history.n_steps == 0:
        return

    # 全ステップのノードを統合して位置を決定
    if node_positions is None:
        all_nodes: set[int] = set()
        for g in history.snapshots:
            all_nodes |= g.nodes
        node_positions = _circular_layout(all_nodes)

    frames: list[Image.Image] = []
    for g in history.snapshots:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_contact_graph(
            g,
            ax=ax,
            node_positions=node_positions,
            show_force=True,
        )
        fig.canvas.draw()

        # Figure → PIL Image
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        img = Image.frombytes("RGBA", (w, h), buf)
        frames.append(img.convert("RGB"))
        plt.close(fig)

    if frames:
        duration = int(1000 / fps)
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )


def _circular_layout(nodes: set[int]) -> dict[int, tuple[float, float]]:
    """ノードを円形に配置する.

    Args:
        nodes: ノード集合

    Returns:
        {node_id: (x, y)} の辞書
    """
    import math

    sorted_nodes = sorted(nodes)
    n = len(sorted_nodes)
    if n == 0:
        return {}
    if n == 1:
        return {sorted_nodes[0]: (0.0, 0.0)}

    positions: dict[int, tuple[float, float]] = {}
    for i, node in enumerate(sorted_nodes):
        angle = 2 * math.pi * i / n
        positions[node] = (math.cos(angle), math.sin(angle))
    return positions
