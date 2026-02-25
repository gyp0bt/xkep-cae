"""接触グラフ統計ダッシュボード可視化のテスト.

plot_statistics_dashboard() のテスト。
"""

from __future__ import annotations

import pytest

from xkep_cae.contact.graph import (
    ContactEdge,
    ContactGraph,
    ContactGraphHistory,
    plot_statistics_dashboard,
)


def _make_history(n_steps: int = 5) -> ContactGraphHistory:
    """テスト用の ContactGraphHistory を作成."""
    history = ContactGraphHistory()
    for step in range(n_steps):
        lf = (step + 1) / n_steps
        edges = []
        nodes: set[int] = set()
        # ステップが進むにつれて接触ペアが増加
        for i in range(min(step + 1, 3)):
            edges.append(
                ContactEdge(
                    elem_a=i,
                    elem_b=i + 3,
                    gap=-0.001 * (step + 1),
                    p_n=10.0 * (step + 1) * (i + 1),
                    status="ACTIVE" if i % 2 == 0 else "SLIDING",
                    stick=i % 2 == 0,
                    dissipation=0.1 * step * (i + 1) if i % 2 == 1 else 0.0,
                    s=0.5,
                    t=0.5,
                )
            )
            nodes.add(i)
            nodes.add(i + 3)

        graph = ContactGraph(
            step=step,
            load_factor=lf,
            nodes=nodes,
            edges=edges,
            n_total_pairs=6,
        )
        history.add(graph)
    return history


class TestPlotStatisticsDashboard:
    """plot_statistics_dashboard のテスト."""

    @pytest.fixture(autouse=True)
    def _skip_no_matplotlib(self):
        """matplotlib がなければスキップ."""
        pytest.importorskip("matplotlib")

    def test_basic_dashboard(self):
        """基本描画: エラーなく Figure を返す."""
        import matplotlib.pyplot as plt

        history = _make_history(5)
        fig = plot_statistics_dashboard(history)
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 6
        plt.close("all")

    def test_empty_history(self):
        """空の時系列でもエラーなし."""
        import matplotlib.pyplot as plt

        history = ContactGraphHistory()
        fig = plot_statistics_dashboard(history)
        assert fig is not None
        plt.close("all")

    def test_single_step(self):
        """1ステップの時系列."""
        import matplotlib.pyplot as plt

        history = _make_history(1)
        fig = plot_statistics_dashboard(history)
        assert fig is not None
        plt.close("all")

    def test_custom_figsize(self):
        """カスタム figsize."""
        import matplotlib.pyplot as plt

        history = _make_history(3)
        fig = plot_statistics_dashboard(history, figsize=(16, 12))
        assert fig is not None
        plt.close("all")

    def test_no_contact_history(self):
        """接触なしの時系列."""
        import matplotlib.pyplot as plt

        history = ContactGraphHistory()
        for step in range(3):
            history.add(
                ContactGraph(
                    step=step,
                    load_factor=step / 3.0,
                    nodes=set(),
                    edges=[],
                    n_total_pairs=0,
                )
            )
        fig = plot_statistics_dashboard(history)
        assert fig is not None
        plt.close("all")

    def test_panels_have_data(self):
        """各パネルにデータが描画されている."""
        import matplotlib.pyplot as plt

        history = _make_history(5)
        fig = plot_statistics_dashboard(history)
        axes = fig.get_axes()

        # 各パネルに line または bar がある
        for i, ax in enumerate(axes):
            has_content = len(ax.lines) > 0 or len(ax.patches) > 0 or len(ax.texts) > 0
            assert has_content, f"パネル {i} にデータがない"

        plt.close("all")
