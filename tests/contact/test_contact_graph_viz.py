"""接触グラフ可視化テスト.

plot_contact_graph, plot_contact_graph_history, save_contact_graph_gif のテスト。
matplotlib が利用できない環境ではスキップする。
"""

import os
import tempfile

import pytest

from xkep_cae.contact.graph import (
    ContactEdge,
    ContactGraph,
    ContactGraphHistory,
    _circular_layout,
    plot_contact_graph,
    plot_contact_graph_history,
    save_contact_graph_gif,
)

# matplotlib が利用できない場合はスキップ
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Pillow が利用できない場合
try:
    from PIL import Image  # noqa: F401

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

pytestmark = pytest.mark.skipif(not HAS_MPL, reason="matplotlib が必要")


def _make_sample_graph(step=0, load_factor=0.5, n_edges=3):
    """テスト用の接触グラフを生成."""
    nodes = set()
    edges = []
    for i in range(n_edges):
        a, b = i * 2, i * 2 + 1
        nodes.add(a)
        nodes.add(b)
        edges.append(
            ContactEdge(
                elem_a=a,
                elem_b=b,
                gap=-0.001 * (i + 1),
                p_n=100.0 * (i + 1),
                status="SLIDING" if i % 2 == 0 else "ACTIVE",
                stick=i % 2 != 0,
                dissipation=0.01 * i,
                s=0.5,
                t=0.5,
            )
        )
    return ContactGraph(
        step=step,
        load_factor=load_factor,
        nodes=nodes,
        edges=edges,
        n_total_pairs=n_edges + 2,
    )


def _make_sample_history(n_steps=5):
    """テスト用の時系列データを生成."""
    history = ContactGraphHistory()
    for i in range(n_steps):
        g = _make_sample_graph(step=i, load_factor=i / n_steps, n_edges=i + 1)
        history.add(g)
    return history


class TestCircularLayout:
    """_circular_layout() のテスト."""

    def test_empty(self):
        pos = _circular_layout(set())
        assert pos == {}

    def test_single_node(self):
        pos = _circular_layout({5})
        assert 5 in pos
        assert pos[5] == (0.0, 0.0)

    def test_multiple_nodes(self):
        pos = _circular_layout({0, 1, 2, 3})
        assert len(pos) == 4
        # 全ノードが単位円上
        for x, y in pos.values():
            r = (x**2 + y**2) ** 0.5
            assert abs(r - 1.0) < 1e-10


class TestPlotContactGraph:
    """plot_contact_graph() のテスト."""

    def test_basic_plot(self):
        """基本的なグラフ描画."""
        g = _make_sample_graph()
        ax = plot_contact_graph(g)
        assert ax is not None
        plt.close("all")

    def test_empty_graph(self):
        """空グラフの描画."""
        g = ContactGraph(step=0, load_factor=0.0, nodes=set(), edges=[], n_total_pairs=0)
        ax = plot_contact_graph(g)
        assert ax is not None
        plt.close("all")

    def test_custom_positions(self):
        """カスタム位置指定."""
        g = _make_sample_graph(n_edges=2)
        positions = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.5, 1.0), 3: (1.5, 1.0)}
        ax = plot_contact_graph(g, node_positions=positions)
        assert ax is not None
        plt.close("all")

    def test_custom_title(self):
        """カスタムタイトル."""
        g = _make_sample_graph()
        ax = plot_contact_graph(g, title="テスト接触グラフ")
        assert ax.get_title() == "テスト接触グラフ"
        plt.close("all")

    def test_no_force_labels(self):
        """反力ラベルなし."""
        g = _make_sample_graph()
        ax = plot_contact_graph(g, show_force=False)
        assert ax is not None
        plt.close("all")

    def test_on_existing_axes(self):
        """既存の Axes に描画."""
        fig, ax = plt.subplots()
        g = _make_sample_graph()
        returned_ax = plot_contact_graph(g, ax=ax)
        assert returned_ax is ax
        plt.close("all")


class TestPlotContactGraphHistory:
    """plot_contact_graph_history() のテスト."""

    def test_basic_history_plot(self):
        """時系列プロットの基本動作."""
        history = _make_sample_history()
        fig = plot_contact_graph_history(history)
        assert fig is not None
        plt.close("all")

    def test_empty_history(self):
        """空の時系列."""
        history = ContactGraphHistory()
        fig = plot_contact_graph_history(history)
        assert fig is not None
        plt.close("all")

    def test_single_step(self):
        """1ステップの時系列."""
        history = ContactGraphHistory()
        history.add(_make_sample_graph())
        fig = plot_contact_graph_history(history)
        assert fig is not None
        plt.close("all")


class TestSaveContactGraphGif:
    """save_contact_graph_gif() のテスト."""

    @pytest.mark.skipif(not HAS_PIL, reason="Pillow が必要")
    def test_save_gif(self):
        """GIF保存."""
        history = _make_sample_history(n_steps=3)
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            filepath = f.name
        try:
            save_contact_graph_gif(history, filepath, fps=2)
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            os.unlink(filepath)

    @pytest.mark.skipif(not HAS_PIL, reason="Pillow が必要")
    def test_empty_history_no_error(self):
        """空時系列でエラーが出ない."""
        history = ContactGraphHistory()
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            filepath = f.name
        try:
            save_contact_graph_gif(history, filepath)
            # 空なのでファイルは作成されない（か、空ファイル）
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    @pytest.mark.skipif(not HAS_PIL, reason="Pillow が必要")
    def test_custom_positions(self):
        """カスタムノード位置でGIF保存."""
        history = _make_sample_history(n_steps=2)
        positions = {i: (float(i % 4), float(i // 4)) for i in range(10)}
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            filepath = f.name
        try:
            save_contact_graph_gif(history, filepath, node_positions=positions)
            assert os.path.exists(filepath)
        finally:
            os.unlink(filepath)
