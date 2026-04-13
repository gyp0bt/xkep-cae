"""CircularFiberSection API テスト.

設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md Phase F3
[← README](../../../../README.md)
"""

from __future__ import annotations

import math

import pytest

from xkep_cae.elements.fiber.section import CircularFiberSection


class TestCircularFiberSectionAPI:
    """CircularFiberSection の API 仕様テスト."""

    def test_strip_returns_frozen_dataclass(self) -> None:
        """strip() が frozen dataclass を返す."""
        sec = CircularFiberSection.strip(diameter=10.0, n_fiber=20)
        assert isinstance(sec, CircularFiberSection)
        with pytest.raises(AttributeError):
            sec.diameter = 5.0  # type: ignore[misc]

    def test_strip_lengths(self) -> None:
        """strip() の y/z/area がすべて n_fiber 長."""
        n = 40
        sec = CircularFiberSection.strip(diameter=12.0, n_fiber=n)
        assert sec.n_fiber == n
        assert len(sec.y) == n
        assert len(sec.z) == n
        assert len(sec.area) == n

    def test_strip_z_all_zero(self) -> None:
        """strip() は平面曲げ用なので z 座標は全て 0."""
        sec = CircularFiberSection.strip(diameter=10.0, n_fiber=30)
        assert all(z == 0.0 for z in sec.z)

    def test_strip_y_symmetry(self) -> None:
        """strip() の y 座標は原点対称."""
        sec = CircularFiberSection.strip(diameter=10.0, n_fiber=60)
        for i in range(sec.n_fiber):
            j = sec.n_fiber - 1 - i
            assert abs(sec.y[i] + sec.y[j]) < 1e-12

    def test_strip_area_positive(self) -> None:
        """全ファイバーの面積が正."""
        sec = CircularFiberSection.strip(diameter=10.0, n_fiber=60)
        assert all(a > 0 for a in sec.area)

    def test_strip_total_area_convergence(self) -> None:
        """ファイバー面積の総和が円の面積に収束する（誤差 < 1%）."""
        d = 17.0
        sec = CircularFiberSection.strip(diameter=d, n_fiber=60)
        total = sum(sec.area)
        exact = math.pi * (d / 2.0) ** 2
        assert abs(total - exact) / exact < 0.01

    def test_polar_returns_correct_count(self) -> None:
        """polar() が n_radial * n_theta 個のファイバーを返す."""
        sec = CircularFiberSection.polar(diameter=10.0, n_radial=6, n_theta=12)
        assert sec.n_fiber == 6 * 12
        assert len(sec.y) == 72
        assert len(sec.z) == 72
        assert len(sec.area) == 72

    def test_polar_total_area_convergence(self) -> None:
        """polar() の面積総和が円の面積に収束する（誤差 < 1%）."""
        d = 17.0
        sec = CircularFiberSection.polar(diameter=d, n_radial=8, n_theta=16)
        total = sum(sec.area)
        exact = math.pi * (d / 2.0) ** 2
        assert abs(total - exact) / exact < 0.01

    def test_strip_matches_prototype(self) -> None:
        """strip() の面積計算が 05_smooth_teardrop.py と一致."""
        import numpy as np

        d = 17.0
        n = 60
        sec = CircularFiberSection.strip(diameter=d, n_fiber=n)

        # プロトタイプの面積計算を再現
        r = d / 2.0
        y_proto = np.linspace(-r + r / n, r - r / n, n)
        dy = 2 * r / n

        for i in range(n):
            assert abs(sec.y[i] - y_proto[i]) < 1e-12
            yl = max(y_proto[i] - dy / 2, -r)
            yh = min(y_proto[i] + dy / 2, r)
            wl = 2 * np.sqrt(max(r**2 - yl**2, 0))
            wh = 2 * np.sqrt(max(r**2 - yh**2, 0))
            a_expected = (wl + wh) / 2 * (yh - yl)
            assert abs(sec.area[i] - a_expected) < 1e-12
