"""ファイバー断面ジェネレータ.

円形断面をファイバー離散化する frozen dataclass。
平面曲げ（y方向ストリップ）と3D二軸曲げ（極座標格子）の
2つの生成メソッドを提供する。

面積計算は work/beam_hysteresis/05_smooth_teardrop.py::FiberSection._area
と同一アルゴリズム（台形近似）。

設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md Phase F3
[← README](../../../README.md)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CircularFiberSection:
    """円形断面のファイバー離散化.

    Attributes:
        diameter: 断面直径
        n_fiber: ファイバー数
        y: ファイバー y 座標タプル (len=n_fiber)
        z: ファイバー z 座標タプル (len=n_fiber)
        area: ファイバー面積タプル (len=n_fiber)
    """

    diameter: float
    n_fiber: int
    y: tuple[float, ...]
    z: tuple[float, ...]
    area: tuple[float, ...]

    @classmethod
    def strip(cls, diameter: float, n_fiber: int = 60) -> CircularFiberSection:
        """平面曲げ用の y 方向ストリップ分割.

        05_smooth_teardrop.py::FiberSection と同一の離散化:
        - y 座標: [-R + R/n, R - R/n] を n 等分
        - 面積: 台形近似 (wl+wh)/2 * (yh-yl)

        Args:
            diameter: 断面直径
            n_fiber: ファイバー数（デフォルト 60）

        Returns:
            CircularFiberSection
        """
        r = diameter / 2.0
        dy = 2.0 * r / n_fiber

        y_list: list[float] = []
        area_list: list[float] = []

        for i in range(n_fiber):
            yi = -r + r / n_fiber + i * dy
            y_list.append(yi)
            area_list.append(_strip_area(yi, dy, r))

        return cls(
            diameter=diameter,
            n_fiber=n_fiber,
            y=tuple(y_list),
            z=tuple(0.0 for _ in range(n_fiber)),
            area=tuple(area_list),
        )

    @classmethod
    def polar(
        cls,
        diameter: float,
        n_radial: int = 8,
        n_theta: int = 16,
    ) -> CircularFiberSection:
        """3D 二軸曲げ用の極座標格子分割.

        Args:
            diameter: 断面直径
            n_radial: 半径方向分割数
            n_theta: 周方向分割数

        Returns:
            CircularFiberSection
        """
        r = diameter / 2.0
        y_list: list[float] = []
        z_list: list[float] = []
        area_list: list[float] = []

        for ir in range(n_radial):
            r_inner = ir * r / n_radial
            r_outer = (ir + 1) * r / n_radial
            ring_area = math.pi * (r_outer**2 - r_inner**2)
            r_mid = (r_inner + r_outer) / 2.0
            a_fiber = ring_area / n_theta

            for it in range(n_theta):
                theta = 2.0 * math.pi * it / n_theta
                y_list.append(r_mid * math.cos(theta))
                z_list.append(r_mid * math.sin(theta))
                area_list.append(a_fiber)

        n_fiber = n_radial * n_theta
        return cls(
            diameter=diameter,
            n_fiber=n_fiber,
            y=tuple(y_list),
            z=tuple(z_list),
            area=tuple(area_list),
        )


def _strip_area(y: float, dy: float, r: float) -> float:
    """y 方向ストリップ1本の面積（台形近似）.

    05_smooth_teardrop.py::FiberSection._area と同一アルゴリズム。

    Args:
        y: ストリップ中心 y 座標
        dy: ストリップ幅
        r: 断面半径

    Returns:
        ストリップ面積
    """
    yl = max(y - dy / 2.0, -r)
    yh = min(y + dy / 2.0, r)
    if yh <= yl:
        return 0.0
    wl = 2.0 * math.sqrt(max(r**2 - yl**2, 0.0))
    wh = 2.0 * math.sqrt(max(r**2 - yh**2, 0.0))
    return (wl + wh) / 2.0 * (yh - yl)
