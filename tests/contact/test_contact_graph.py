"""接触グラフ表現のテスト.

ContactGraph / ContactGraphHistory のデータ構造と集約機能を検証する。
"""

import numpy as np

from xkep_cae.contact.graph import (
    ContactGraphHistory,
    snapshot_contact_graph,
)
from xkep_cae.contact.pair import (
    ContactManager,
    ContactPair,
    ContactState,
    ContactStatus,
)

# ====================================================================
# ヘルパー: テスト用のContactManagerを構築
# ====================================================================


def _make_test_manager(n_active=3, n_inactive=2):
    """テスト用のContactManagerを構築.

    Args:
        n_active: アクティブペア数
        n_inactive: 非アクティブペア数
    """
    mgr = ContactManager()

    for i in range(n_active):
        pair = ContactPair(
            elem_a=i * 2,
            elem_b=i * 2 + 1,
            nodes_a=np.array([i * 4, i * 4 + 1]),
            nodes_b=np.array([i * 4 + 2, i * 4 + 3]),
            radius_a=0.001,
            radius_b=0.001,
        )
        pair.state = ContactState(
            gap=-0.0001 * (i + 1),
            p_n=100.0 * (i + 1),
            status=ContactStatus.ACTIVE if i % 2 == 0 else ContactStatus.SLIDING,
            stick=i % 2 == 0,
            dissipation=0.01 * (i + 1),
            s=0.5,
            t=0.5,
        )
        mgr.pairs.append(pair)

    for i in range(n_inactive):
        pair = ContactPair(
            elem_a=100 + i * 2,
            elem_b=100 + i * 2 + 1,
            nodes_a=np.array([100 + i * 4, 100 + i * 4 + 1]),
            nodes_b=np.array([100 + i * 4 + 2, 100 + i * 4 + 3]),
            radius_a=0.001,
            radius_b=0.001,
        )
        pair.state = ContactState(status=ContactStatus.INACTIVE)
        mgr.pairs.append(pair)

    return mgr


# ====================================================================
# snapshot_contact_graph テスト
# ====================================================================


class TestSnapshotContactGraph:
    """グラフスナップショット生成のテスト."""

    def test_basic_snapshot(self):
        """基本的なスナップショット生成."""
        mgr = _make_test_manager(n_active=3, n_inactive=2)
        graph = snapshot_contact_graph(mgr, step=5, load_factor=0.5)

        assert graph.step == 5
        assert graph.load_factor == 0.5
        assert graph.n_edges == 3
        assert graph.n_total_pairs == 5

    def test_nodes_from_active_pairs(self):
        """アクティブペアからノード集合が生成される."""
        mgr = _make_test_manager(n_active=2)
        graph = snapshot_contact_graph(mgr)
        # 2ペア → 各ペアで2ノード → 最大4ノード
        assert graph.n_nodes == 4
        assert 0 in graph.nodes
        assert 1 in graph.nodes
        assert 2 in graph.nodes
        assert 3 in graph.nodes

    def test_inactive_pairs_excluded(self):
        """非アクティブペアはエッジに含まれない."""
        mgr = _make_test_manager(n_active=1, n_inactive=5)
        graph = snapshot_contact_graph(mgr)
        assert graph.n_edges == 1
        assert graph.n_total_pairs == 6

    def test_empty_manager(self):
        """ペアなしのマネージャ."""
        mgr = ContactManager()
        graph = snapshot_contact_graph(mgr)
        assert graph.n_edges == 0
        assert graph.n_nodes == 0

    def test_edge_attributes(self):
        """エッジ属性が正しく転写される."""
        mgr = _make_test_manager(n_active=1)
        graph = snapshot_contact_graph(mgr)
        assert len(graph.edges) == 1
        e = graph.edges[0]
        assert e.elem_a == 0
        assert e.elem_b == 1
        assert e.p_n == 100.0
        assert e.gap == -0.0001
        assert e.status == "ACTIVE"
        assert e.stick is True

    def test_sliding_status(self):
        """SLIDING 状態が正しく反映される."""
        mgr = _make_test_manager(n_active=2)
        graph = snapshot_contact_graph(mgr)
        statuses = [e.status for e in graph.edges]
        assert "ACTIVE" in statuses
        assert "SLIDING" in statuses


# ====================================================================
# ContactGraph メソッドテスト
# ====================================================================


class TestContactGraphMethods:
    """ContactGraph の分析メソッドテスト."""

    def test_n_active_and_sliding(self):
        """ACTIVE/SLIDING のカウント."""
        mgr = _make_test_manager(n_active=3)
        graph = snapshot_contact_graph(mgr)
        # i=0: ACTIVE, i=1: SLIDING, i=2: ACTIVE
        assert graph.n_active == 2
        assert graph.n_sliding == 1

    def test_total_normal_force(self):
        """法線反力合計."""
        mgr = _make_test_manager(n_active=3)
        graph = snapshot_contact_graph(mgr)
        assert abs(graph.total_normal_force - (100.0 + 200.0 + 300.0)) < 1e-10

    def test_total_dissipation(self):
        """散逸エネルギー合計."""
        mgr = _make_test_manager(n_active=3)
        graph = snapshot_contact_graph(mgr)
        assert abs(graph.total_dissipation - (0.01 + 0.02 + 0.03)) < 1e-10

    def test_degree_map(self):
        """次数マップ."""
        mgr = _make_test_manager(n_active=2)
        graph = snapshot_contact_graph(mgr)
        deg = graph.degree_map()
        # 各ノードの次数は1（独立した2ペア）
        for node in graph.nodes:
            assert deg[node] == 1

    def test_adjacency_list(self):
        """隣接リスト."""
        mgr = _make_test_manager(n_active=2)
        graph = snapshot_contact_graph(mgr)
        adj = graph.adjacency_list()
        assert 1 in adj[0]
        assert 0 in adj[1]

    def test_connected_components_separate(self):
        """独立したペアは別の連結成分."""
        mgr = _make_test_manager(n_active=2)
        graph = snapshot_contact_graph(mgr)
        comps = graph.connected_components()
        assert len(comps) == 2

    def test_connected_components_connected(self):
        """共有ノードがある場合は1つの連結成分."""
        mgr = ContactManager()
        # ペア1: elem 0-1
        p1 = ContactPair(0, 1, np.array([0, 1]), np.array([2, 3]))
        p1.state = ContactState(status=ContactStatus.ACTIVE, p_n=100.0)
        mgr.pairs.append(p1)
        # ペア2: elem 1-2（elem 1を共有）
        p2 = ContactPair(1, 2, np.array([2, 3]), np.array([4, 5]))
        p2.state = ContactState(status=ContactStatus.ACTIVE, p_n=200.0)
        mgr.pairs.append(p2)

        graph = snapshot_contact_graph(mgr)
        comps = graph.connected_components()
        assert len(comps) == 1
        assert graph.n_nodes == 3

    def test_adjacency_matrix(self):
        """隣接行列."""
        mgr = _make_test_manager(n_active=1)
        graph = snapshot_contact_graph(mgr)
        A = graph.to_adjacency_matrix(n_elems=4)
        assert A.shape == (4, 4)
        assert A[0, 1] == 100.0
        assert A[1, 0] == 100.0
        assert A[0, 0] == 0.0

    def test_to_dict(self):
        """辞書表現."""
        mgr = _make_test_manager(n_active=2)
        graph = snapshot_contact_graph(mgr, step=3, load_factor=0.3)
        d = graph.to_dict()
        assert d["step"] == 3
        assert d["load_factor"] == 0.3
        assert d["n_edges"] == 2
        assert len(d["edges"]) == 2
        assert "elem_a" in d["edges"][0]
        assert "p_n" in d["edges"][0]


# ====================================================================
# ContactGraphHistory テスト
# ====================================================================


class TestContactGraphHistory:
    """接触グラフ時系列のテスト."""

    def _make_history(self) -> ContactGraphHistory:
        """3ステップの時系列を構築."""
        history = ContactGraphHistory()

        # ステップ1: 2ペアアクティブ
        mgr1 = _make_test_manager(n_active=2, n_inactive=1)
        history.add(snapshot_contact_graph(mgr1, step=1, load_factor=0.33))

        # ステップ2: 3ペアアクティブ（トポロジー変化）
        mgr2 = _make_test_manager(n_active=3, n_inactive=0)
        history.add(snapshot_contact_graph(mgr2, step=2, load_factor=0.67))

        # ステップ3: 3ペアアクティブ（同じトポロジー）
        mgr3 = _make_test_manager(n_active=3, n_inactive=0)
        history.add(snapshot_contact_graph(mgr3, step=3, load_factor=1.0))

        return history

    def test_n_steps(self):
        """ステップ数."""
        history = self._make_history()
        assert history.n_steps == 3

    def test_edge_count_series(self):
        """エッジ数時系列."""
        history = self._make_history()
        series = history.edge_count_series()
        assert len(series) == 3
        assert series[0] == 2
        assert series[1] == 3
        assert series[2] == 3

    def test_node_count_series(self):
        """ノード数時系列."""
        history = self._make_history()
        series = history.node_count_series()
        assert len(series) == 3
        assert series[0] == 4  # 2ペア × 2ノード
        assert series[1] == 6  # 3ペア × 2ノード

    def test_total_force_series(self):
        """法線反力合計時系列."""
        history = self._make_history()
        series = history.total_force_series()
        assert len(series) == 3
        assert abs(series[0] - 300.0) < 1e-10  # 100+200
        assert abs(series[1] - 600.0) < 1e-10  # 100+200+300

    def test_load_factor_series(self):
        """荷重係数時系列."""
        history = self._make_history()
        series = history.load_factor_series()
        assert abs(series[0] - 0.33) < 1e-10
        assert abs(series[2] - 1.0) < 1e-10

    def test_topology_change_steps(self):
        """トポロジー変化ステップの検出."""
        history = self._make_history()
        changes = history.topology_change_steps()
        # ステップ1→2でエッジ数が変化
        assert 2 in changes
        # ステップ2→3はトポロジー同じ
        assert 3 not in changes

    def test_dissipation_series(self):
        """散逸エネルギー時系列."""
        history = self._make_history()
        series = history.dissipation_series()
        assert len(series) == 3
        assert series[0] > 0

    def test_to_dict_list(self):
        """辞書リスト出力."""
        history = self._make_history()
        dicts = history.to_dict_list()
        assert len(dicts) == 3
        assert dicts[0]["step"] == 1
        assert dicts[2]["step"] == 3

    def test_empty_history(self):
        """空の時系列."""
        history = ContactGraphHistory()
        assert history.n_steps == 0
        assert len(history.edge_count_series()) == 0
        assert len(history.topology_change_steps()) == 0
