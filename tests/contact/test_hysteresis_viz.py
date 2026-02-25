"""撚線ヒステリシス可視化 + ループ面積計算のテスト.

plot_hysteresis_curve() と compute_hysteresis_area() のテスト。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xkep_cae.contact.graph import (
    compute_hysteresis_area,
    plot_hysteresis_curve,
)

# ====================================================================
# compute_hysteresis_area テスト
# ====================================================================


class TestComputeHysteresisArea:
    """ヒステリシスループ面積計算のテスト."""

    def test_triangle_area(self):
        """三角形 (0,0)→(1,0)→(0,1) の面積 = 0.5."""
        load_factors = [0.0, 0.0, 1.0]
        displacements = [0.0, 1.0, 0.0]
        area = compute_hysteresis_area(load_factors, displacements)
        assert abs(area - 0.5) < 1e-12

    def test_square_area(self):
        """正方形 (0,0)→(1,0)→(1,1)→(0,1) の面積 = 1.0."""
        load_factors = [0.0, 0.0, 1.0, 1.0]
        displacements = [0.0, 1.0, 1.0, 0.0]
        area = compute_hysteresis_area(load_factors, displacements)
        assert abs(area - 1.0) < 1e-12

    def test_zero_area_for_line(self):
        """直線上の点列は面積 0."""
        load_factors = [0.0, 0.5, 1.0]
        displacements = [0.0, 0.5, 1.0]
        area = compute_hysteresis_area(load_factors, displacements)
        assert abs(area) < 1e-12

    def test_too_few_points(self):
        """2点以下は面積 0."""
        assert compute_hysteresis_area([0.0, 1.0], [0.0, 1.0]) == 0.0
        assert compute_hysteresis_area([0.0], [0.0]) == 0.0
        assert compute_hysteresis_area([], []) == 0.0

    def test_ndarray_displacements_with_dof_index(self):
        """ndarray 変位で dof_index 指定."""
        load_factors = [0.0, 0.0, 1.0]
        displacements = [
            np.array([0.0, 10.0]),
            np.array([1.0, 20.0]),
            np.array([0.0, 30.0]),
        ]
        # dof_index=0: (0,0)→(1,0)→(0,1) → 面積 0.5
        area = compute_hysteresis_area(load_factors, displacements, dof_index=0)
        assert abs(area - 0.5) < 1e-12

    def test_ndarray_displacements_norm(self):
        """dof_index=None ではノルムを使用."""
        load_factors = [0.0, 0.0, 1.0]
        displacements = [
            np.array([0.0, 0.0]),
            np.array([0.6, 0.8]),  # norm = 1.0
            np.array([0.0, 0.0]),
        ]
        area = compute_hysteresis_area(load_factors, displacements)
        # 三角形 (0,0)→(1,0)→(0,1) → 面積 0.5
        assert abs(area - 0.5) < 1e-12

    def test_symmetric_hysteresis(self):
        """対称ヒステリシスループの面積."""
        # 楕円近似: 上側 y = sin(t), 下側 y = -sin(t), x = cos(t)
        n = 100
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = np.cos(t)
        y = np.sin(t)
        # 円の面積 = π
        area = compute_hysteresis_area(y.tolist(), x.tolist())
        assert abs(area - math.pi) < 0.1  # 離散化誤差許容


# ====================================================================
# plot_hysteresis_curve テスト
# ====================================================================


class TestPlotHysteresisCurve:
    """ヒステリシス曲線描画のテスト."""

    @pytest.fixture(autouse=True)
    def _skip_no_matplotlib(self):
        """matplotlib がなければスキップ."""
        pytest.importorskip("matplotlib")

    def test_basic_plot(self):
        """基本描画: エラーなく Axes を返す."""
        import matplotlib.pyplot as plt

        load_factors = [0.0, 0.5, 1.0, 0.5, 0.0]
        displacements = [0.0, 0.1, 0.3, 0.2, 0.05]
        ax = plot_hysteresis_curve(load_factors, displacements)
        assert ax is not None
        plt.close("all")

    def test_with_dof_index(self):
        """dof_index 指定で描画."""
        import matplotlib.pyplot as plt

        load_factors = [0.0, 0.5, 1.0]
        displacements = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.2, 0.3]),
            np.array([0.2, 0.4, 0.6]),
        ]
        ax = plot_hysteresis_curve(load_factors, displacements, dof_index=2)
        assert ax is not None
        plt.close("all")

    def test_custom_labels(self):
        """カスタムラベル."""
        import matplotlib.pyplot as plt

        ax = plot_hysteresis_curve(
            [0.0, 1.0],
            [0.0, 0.1],
            xlabel="変位 [mm]",
            ylabel="荷重 [N]",
            title="テストカーブ",
        )
        assert ax.get_xlabel() == "変位 [mm]"
        assert ax.get_ylabel() == "荷重 [N]"
        assert ax.get_title() == "テストカーブ"
        plt.close("all")

    def test_existing_axes(self):
        """既存 Axes に描画."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        returned_ax = plot_hysteresis_curve([0.0, 1.0], [0.0, 0.1], ax=ax)
        assert returned_ax is ax
        plt.close("all")
