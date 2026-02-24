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
