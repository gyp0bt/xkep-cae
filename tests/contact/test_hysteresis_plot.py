"""ヒステリシス可視化・ループ面積・統計ダッシュボードのテスト."""

import numpy as np
import pytest

from xkep_cae.contact.graph import (
    ContactEdge,
    ContactGraph,
    ContactGraphHistory,
    compute_hysteresis_area,
    plot_hysteresis_curve,
    plot_statistics_dashboard,
)

# matplotlib が利用不可の場合はスキップ
plt = pytest.importorskip("matplotlib.pyplot")


# ====================================================================
# ヘルパー
# ====================================================================


def _make_sample_history(n_steps: int = 10) -> ContactGraphHistory:
    """テスト用接触グラフ時系列を生成する."""
    history = ContactGraphHistory()
    for i in range(n_steps):
        lf = (i + 1) / n_steps
        edges = [
            ContactEdge(
                elem_a=0,
                elem_b=1,
                gap=-0.001 * (i + 1),
                p_n=100.0 * (i + 1),
                status="ACTIVE" if i < 5 else "SLIDING",
                stick=i < 5,
                dissipation=10.0 * i if i >= 5 else 0.0,
                s=0.5,
                t=0.5,
            ),
            ContactEdge(
                elem_a=1,
                elem_b=2,
                gap=-0.0005 * (i + 1),
                p_n=50.0 * (i + 1),
                status="ACTIVE",
                stick=True,
                dissipation=0.0,
                s=0.3,
                t=0.7,
            ),
        ]
        nodes = {0, 1, 2}
        graph = ContactGraph(
            step=i + 1,
            load_factor=lf,
            nodes=nodes,
            edges=edges,
            n_total_pairs=3,
        )
        history.add(graph)
    return history


# ====================================================================
# plot_hysteresis_curve テスト
# ====================================================================


class TestPlotHysteresisCurve:
    """ヒステリシス曲線描画のテスト."""

    def test_basic_scalar_displacements(self):
        """スカラー変位リストでヒステリシス曲線を描画できる."""
        load_factors = [0.0, 0.5, 1.0, 0.5, 0.0]
        displacements = [0.0, 0.3, 0.8, 0.5, 0.1]
        ax = plot_hysteresis_curve(load_factors, displacements)
        assert ax is not None
        plt.close("all")

    def test_ndarray_displacements_with_dof_index(self):
        """ndarray 変位から DOF インデックスを指定して描画できる."""
        n_steps = 10
        ndof = 12
        load_factors = list(np.linspace(0, 1, n_steps))
        displacements = [np.random.randn(ndof) * 0.01 * i for i in range(n_steps)]
        ax = plot_hysteresis_curve(load_factors, displacements, dof_index=3)
        assert ax is not None
        plt.close("all")

    def test_ndarray_displacements_norm(self):
        """DOF 指定なしの場合はノルムで描画する."""
        n_steps = 8
        ndof = 6
        load_factors = list(np.linspace(0, 1, n_steps))
        displacements = [np.ones(ndof) * 0.01 * i for i in range(n_steps)]
        ax = plot_hysteresis_curve(load_factors, displacements, dof_index=None)
        assert ax is not None
        plt.close("all")

    def test_custom_labels(self):
        """カスタムラベルが設定できる."""
        ax = plot_hysteresis_curve(
            [0, 1, 0],
            [0.0, 0.5, 0.1],
            xlabel="先端変位 [mm]",
            ylabel="荷重 [N]",
            title="3本撚りヒステリシス",
        )
        assert ax.get_xlabel() == "先端変位 [mm]"
        assert ax.get_ylabel() == "荷重 [N]"
        assert ax.get_title() == "3本撚りヒステリシス"
        plt.close("all")

    def test_existing_axes(self):
        """既存 Axes に描画できる."""
        fig, ax = plt.subplots()
        ret_ax = plot_hysteresis_curve([0, 1, 0], [0.0, 0.5, 0.1], ax=ax)
        assert ret_ax is ax
        plt.close("all")


# ====================================================================
# compute_hysteresis_area テスト
# ====================================================================


class TestComputeHysteresisArea:
    """ヒステリシスループ面積計算のテスト."""

    def test_zero_area_linear(self):
        """直線上の点列は面積0."""
        lf = [0.0, 0.5, 1.0]
        disp = [0.0, 0.5, 1.0]
        area = compute_hysteresis_area(lf, disp)
        assert area == pytest.approx(0.0, abs=1e-12)

    def test_unit_square(self):
        """単位正方形の面積は1.0."""
        # 反時計回りの正方形
        disp = [0.0, 1.0, 1.0, 0.0]
        lf = [0.0, 0.0, 1.0, 1.0]
        area = compute_hysteresis_area(lf, disp)
        assert area == pytest.approx(1.0, abs=1e-12)

    def test_triangle_area(self):
        """三角形の面積."""
        disp = [0.0, 2.0, 0.0]
        lf = [0.0, 0.0, 1.0]
        area = compute_hysteresis_area(lf, disp)
        assert area == pytest.approx(1.0, abs=1e-12)

    def test_hysteresis_loop_nonzero(self):
        """往復荷重でループ面積が非ゼロ."""
        # 正方向: 0→1 で変位 0→1
        # 逆方向: 1→0 で変位 1→0.5 (残留)
        disp = [0.0, 0.5, 1.0, 0.7, 0.5]
        lf = [0.0, 0.5, 1.0, 0.5, 0.0]
        area = compute_hysteresis_area(lf, disp)
        assert area > 0.0

    def test_too_few_points(self):
        """2点以下は面積0."""
        assert compute_hysteresis_area([0], [0]) == 0.0
        assert compute_hysteresis_area([0, 1], [0, 1]) == 0.0

    def test_ndarray_input(self):
        """numpy 配列入力に対応."""
        disp = np.array([0.0, 1.0, 1.0, 0.0])
        lf = np.array([0.0, 0.0, 1.0, 1.0])
        area = compute_hysteresis_area(lf, disp)
        assert area == pytest.approx(1.0, abs=1e-12)

    def test_with_dof_index(self):
        """ndarray 変位と dof_index で面積を計算できる."""
        displacements = [
            np.array([0.0, 0.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([0.0, 0.0]),
        ]
        lf = [0.0, 0.0, 1.0, 1.0]
        area = compute_hysteresis_area(lf, displacements, dof_index=0)
        assert area == pytest.approx(1.0, abs=1e-12)


# ====================================================================
# plot_statistics_dashboard テスト
# ====================================================================


class TestPlotStatisticsDashboard:
    """統計ダッシュボードのテスト."""

    def test_basic_dashboard(self):
        """基本的なダッシュボードが描画できる."""
        history = _make_sample_history(10)
        fig = plot_statistics_dashboard(history)
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 6
        plt.close("all")

    def test_empty_history(self):
        """空の時系列でもエラーにならない."""
        history = ContactGraphHistory()
        fig = plot_statistics_dashboard(history)
        assert fig is not None
        plt.close("all")

    def test_single_step(self):
        """1ステップの場合でも描画できる."""
        history = _make_sample_history(1)
        fig = plot_statistics_dashboard(history)
        assert fig is not None
        plt.close("all")

    def test_dashboard_panels_content(self):
        """各パネルが適切なタイトルを持つ."""
        history = _make_sample_history(5)
        fig = plot_statistics_dashboard(history)
        axes = fig.get_axes()
        titles = [ax.get_title() for ax in axes]
        assert "stick/slip 比率" in titles
        assert "法線反力の推移" in titles
        assert "接触ネットワーク連結性" in titles
        assert "累積散逸エネルギー" in titles
        assert "接触持続マップ" in titles
        assert "エッジ数・ノード数の推移" in titles
        plt.close("all")

    def test_custom_figsize(self):
        """カスタム figsize が反映される."""
        history = _make_sample_history(5)
        fig = plot_statistics_dashboard(history, figsize=(16, 12))
        assert fig is not None
        plt.close("all")
