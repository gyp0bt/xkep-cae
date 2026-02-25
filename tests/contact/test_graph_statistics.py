"""接触グラフ統計分析テスト.

ContactGraphHistory の統計分析メソッドを検証する。

テスト対象:
  - stick_slip_ratio_series
  - mean_normal_force_series
  - max_normal_force_series
  - connected_component_count_series
  - contact_duration_map
  - cumulative_dissipation_series
  - summary
"""

import pytest

from xkep_cae.contact.graph import (
    ContactEdge,
    ContactGraph,
    ContactGraphHistory,
)


def _make_edge(elem_a, elem_b, p_n=1.0, stick=True, dissipation=0.0):
    """テスト用のContactEdgeを作成."""
    return ContactEdge(
        elem_a=elem_a,
        elem_b=elem_b,
        gap=-0.001,
        p_n=p_n,
        status="ACTIVE" if stick else "SLIDING",
        stick=stick,
        dissipation=dissipation,
        s=0.5,
        t=0.5,
    )


def _make_graph(step, load_factor, edges, n_total_pairs=10):
    """テスト用のContactGraphを作成."""
    nodes = set()
    for e in edges:
        nodes.add(e.elem_a)
        nodes.add(e.elem_b)
    return ContactGraph(
        step=step,
        load_factor=load_factor,
        nodes=nodes,
        edges=edges,
        n_total_pairs=n_total_pairs,
    )


class TestStickSlipRatio:
    """stick_slip_ratio_series のテスト."""

    def test_all_stick(self):
        """全エッジがstickなら比率=1.0."""
        hist = ContactGraphHistory()
        edges = [_make_edge(0, 1, stick=True), _make_edge(2, 3, stick=True)]
        hist.add(_make_graph(1, 0.5, edges))
        ratios = hist.stick_slip_ratio_series()
        assert ratios[0] == pytest.approx(1.0)

    def test_all_slip(self):
        """全エッジがslipなら比率=0.0."""
        hist = ContactGraphHistory()
        edges = [_make_edge(0, 1, stick=False), _make_edge(2, 3, stick=False)]
        hist.add(_make_graph(1, 0.5, edges))
        ratios = hist.stick_slip_ratio_series()
        assert ratios[0] == pytest.approx(0.0)

    def test_mixed(self):
        """mixed stick/slip."""
        hist = ContactGraphHistory()
        edges = [
            _make_edge(0, 1, stick=True),
            _make_edge(2, 3, stick=False),
            _make_edge(4, 5, stick=True),
        ]
        hist.add(_make_graph(1, 0.5, edges))
        ratios = hist.stick_slip_ratio_series()
        assert ratios[0] == pytest.approx(2.0 / 3.0)

    def test_empty(self):
        """エッジなしは1.0."""
        hist = ContactGraphHistory()
        hist.add(_make_graph(1, 0.5, []))
        ratios = hist.stick_slip_ratio_series()
        assert ratios[0] == pytest.approx(1.0)


class TestForceStatistics:
    """mean/max_normal_force_series のテスト."""

    def test_mean_force(self):
        """平均法線反力."""
        hist = ContactGraphHistory()
        edges = [_make_edge(0, 1, p_n=10.0), _make_edge(2, 3, p_n=20.0)]
        hist.add(_make_graph(1, 0.5, edges))
        means = hist.mean_normal_force_series()
        assert means[0] == pytest.approx(15.0)

    def test_max_force(self):
        """最大法線反力."""
        hist = ContactGraphHistory()
        edges = [_make_edge(0, 1, p_n=10.0), _make_edge(2, 3, p_n=20.0)]
        hist.add(_make_graph(1, 0.5, edges))
        maxes = hist.max_normal_force_series()
        assert maxes[0] == pytest.approx(20.0)

    def test_empty_graph(self):
        """エッジなしは0.0."""
        hist = ContactGraphHistory()
        hist.add(_make_graph(1, 0.5, []))
        assert hist.mean_normal_force_series()[0] == 0.0
        assert hist.max_normal_force_series()[0] == 0.0


class TestConnectedComponents:
    """connected_component_count_series のテスト."""

    def test_single_component(self):
        """全ノードが連結なら1成分."""
        hist = ContactGraphHistory()
        edges = [_make_edge(0, 1), _make_edge(1, 2)]
        hist.add(_make_graph(1, 0.5, edges))
        cc = hist.connected_component_count_series()
        assert cc[0] == 1

    def test_two_components(self):
        """2つの独立グループなら2成分."""
        hist = ContactGraphHistory()
        edges = [_make_edge(0, 1), _make_edge(2, 3)]
        hist.add(_make_graph(1, 0.5, edges))
        cc = hist.connected_component_count_series()
        assert cc[0] == 2

    def test_empty(self):
        """エッジなしは0成分."""
        hist = ContactGraphHistory()
        hist.add(_make_graph(1, 0.5, []))
        cc = hist.connected_component_count_series()
        assert cc[0] == 0


class TestContactDuration:
    """contact_duration_map のテスト."""

    def test_persistent_contact(self):
        """全ステップで接触するエッジのduration."""
        hist = ContactGraphHistory()
        for s in range(1, 6):
            hist.add(_make_graph(s, s / 5.0, [_make_edge(0, 1)]))
        durations = hist.contact_duration_map()
        assert durations[(0, 1)] == 5

    def test_intermittent_contact(self):
        """断続的に接触するエッジ."""
        hist = ContactGraphHistory()
        hist.add(_make_graph(1, 0.2, [_make_edge(0, 1)]))
        hist.add(_make_graph(2, 0.4, []))  # 一時離間
        hist.add(_make_graph(3, 0.6, [_make_edge(0, 1)]))
        durations = hist.contact_duration_map()
        assert durations[(0, 1)] == 2

    def test_order_independence(self):
        """(a,b) と (b,a) は同一ペアとして集計."""
        hist = ContactGraphHistory()
        hist.add(_make_graph(1, 0.5, [_make_edge(3, 1)]))
        hist.add(_make_graph(2, 1.0, [_make_edge(1, 3)]))
        durations = hist.contact_duration_map()
        assert durations[(1, 3)] == 2


class TestCumulativeDissipation:
    """cumulative_dissipation_series のテスト."""

    def test_cumulative(self):
        """散逸エネルギーの累積."""
        hist = ContactGraphHistory()
        hist.add(_make_graph(1, 0.5, [_make_edge(0, 1, dissipation=1.0)]))
        hist.add(_make_graph(2, 1.0, [_make_edge(0, 1, dissipation=2.0)]))
        hist.add(_make_graph(3, 1.0, [_make_edge(0, 1, dissipation=0.5)]))
        cum = hist.cumulative_dissipation_series()
        assert cum[0] == pytest.approx(1.0)
        assert cum[1] == pytest.approx(3.0)
        assert cum[2] == pytest.approx(3.5)


class TestSummary:
    """summary() のテスト."""

    def test_empty_history(self):
        """空の時系列."""
        hist = ContactGraphHistory()
        s = hist.summary()
        assert s["n_steps"] == 0
        assert s["max_edges"] == 0

    def test_basic_summary(self):
        """基本的な要約."""
        hist = ContactGraphHistory()
        hist.add(
            _make_graph(
                1,
                0.5,
                [
                    _make_edge(0, 1, p_n=10.0, stick=True, dissipation=0.1),
                    _make_edge(2, 3, p_n=5.0, stick=False, dissipation=0.2),
                ],
            )
        )
        hist.add(
            _make_graph(
                2,
                1.0,
                [
                    _make_edge(0, 1, p_n=15.0, stick=True, dissipation=0.3),
                ],
            )
        )
        s = hist.summary()
        assert s["n_steps"] == 2
        assert s["max_edges"] == 2
        assert s["max_nodes"] == 4  # step 1: 4 nodes
        assert s["max_normal_force"] == pytest.approx(15.0)
        assert s["total_dissipation"] == pytest.approx(0.6)  # 0.1+0.2+0.3
        assert s["unique_contacts"] == 2  # (0,1), (2,3)
        assert 0.0 < s["mean_stick_ratio"] < 1.0

    def test_summary_returns_dict(self):
        """summary が dict を返す."""
        hist = ContactGraphHistory()
        hist.add(_make_graph(1, 0.5, [_make_edge(0, 1)]))
        s = hist.summary()
        assert isinstance(s, dict)
        expected_keys = {
            "n_steps",
            "max_edges",
            "max_nodes",
            "max_normal_force",
            "total_dissipation",
            "n_topology_changes",
            "unique_contacts",
            "mean_stick_ratio",
        }
        assert set(s.keys()) == expected_keys
