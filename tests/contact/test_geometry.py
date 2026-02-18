"""接触幾何モジュールのテスト."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.contact.geometry import (
    ClosestPointResult,
    build_contact_frame,
    closest_point_segments,
    compute_gap,
)


# ---------------------------------------------------------------------------
# closest_point_segments
# ---------------------------------------------------------------------------
class TestClosestPointSegments:
    """closest_point_segments のテスト."""

    def test_perpendicular_midpoints(self):
        """直交する2線分: 中点が最近接点."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([2.0, 0.0, 0.0])
        xB0 = np.array([1.0, 1.0, 0.0])
        xB1 = np.array([1.0, -1.0, 0.0])

        result = closest_point_segments(xA0, xA1, xB0, xB1)

        assert isinstance(result, ClosestPointResult)
        np.testing.assert_allclose(result.s, 0.5)
        np.testing.assert_allclose(result.t, 0.5)
        np.testing.assert_allclose(result.distance, 0.0, atol=1e-14)
        assert not result.parallel

    def test_parallel_segments(self):
        """平行な2線分: parallel=True."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 1.0, 0.0])
        xB1 = np.array([1.0, 1.0, 0.0])

        result = closest_point_segments(xA0, xA1, xB0, xB1)

        assert result.parallel
        np.testing.assert_allclose(result.distance, 1.0, atol=1e-12)

    def test_separated_segments(self):
        """離れた2線分の距離."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.0, 3.0])
        xB1 = np.array([1.0, 0.0, 3.0])

        result = closest_point_segments(xA0, xA1, xB0, xB1)

        np.testing.assert_allclose(result.distance, 3.0, atol=1e-12)

    def test_endpoint_clamping(self):
        """端点クランプ: 最近接点が線分外 → 端点に射影."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([2.0, 1.0, 0.0])
        xB1 = np.array([2.0, 2.0, 0.0])

        result = closest_point_segments(xA0, xA1, xB0, xB1)

        # A上の最近接点は s=1.0 (xA1)
        np.testing.assert_allclose(result.s, 1.0)
        # B上の最近接点は t=0.0 (xB0)
        np.testing.assert_allclose(result.t, 0.0)
        expected_dist = np.sqrt(1.0**2 + 1.0**2)  # sqrt(2)
        np.testing.assert_allclose(result.distance, expected_dist, atol=1e-12)

    def test_point_a_on_segment(self):
        """点A（セグメントA上の点）を返す."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([4.0, 0.0, 0.0])
        xB0 = np.array([2.0, 3.0, 0.0])
        xB1 = np.array([2.0, 5.0, 0.0])

        result = closest_point_segments(xA0, xA1, xB0, xB1)

        # point_a = xA0 + s * dA
        expected_a = xA0 + result.s * (xA1 - xA0)
        np.testing.assert_allclose(result.point_a, expected_a, atol=1e-14)

    def test_normal_direction(self):
        """法線は A→B の方向（diff / |diff|）."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.5, 2.0, 0.0])
        xB1 = np.array([0.5, 4.0, 0.0])

        result = closest_point_segments(xA0, xA1, xB0, xB1)

        # A→B 方向は +y
        expected_normal = np.array([0.0, -1.0, 0.0])  # diff = A - B → -y
        np.testing.assert_allclose(np.abs(result.normal @ expected_normal), 1.0, atol=1e-12)

    def test_zero_length_segment(self):
        """縮退セグメント（長さゼロ）でも動作."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([0.0, 0.0, 0.0])  # 点
        xB0 = np.array([1.0, 0.0, 0.0])
        xB1 = np.array([1.0, 0.0, 0.0])  # 点

        result = closest_point_segments(xA0, xA1, xB0, xB1)

        np.testing.assert_allclose(result.distance, 1.0, atol=1e-12)
        assert result.parallel  # 縮退は平行判定

    def test_skew_segments_3d(self):
        """3次元ねじれ配置の2線分."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.5, 0.0, 1.0])
        xB1 = np.array([0.5, 1.0, 1.0])

        result = closest_point_segments(xA0, xA1, xB0, xB1)

        np.testing.assert_allclose(result.s, 0.5, atol=1e-12)
        np.testing.assert_allclose(result.t, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.distance, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# compute_gap
# ---------------------------------------------------------------------------
class TestComputeGap:
    """compute_gap のテスト."""

    def test_positive_gap(self):
        """離間: g > 0."""
        g = compute_gap(5.0, 1.0, 2.0)
        assert g == pytest.approx(2.0)

    def test_zero_gap(self):
        """ちょうど接触: g = 0."""
        g = compute_gap(3.0, 1.5, 1.5)
        assert g == pytest.approx(0.0)

    def test_negative_gap(self):
        """貫通: g < 0."""
        g = compute_gap(1.0, 1.0, 1.0)
        assert g == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# build_contact_frame
# ---------------------------------------------------------------------------
class TestBuildContactFrame:
    """build_contact_frame のテスト."""

    def test_orthonormal(self):
        """(n, t1, t2) が正規直交基底を構成."""
        normal = np.array([0.0, 0.0, 1.0])
        n, t1, t2 = build_contact_frame(normal)

        # 正規化
        np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(t1), 1.0, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(t2), 1.0, atol=1e-14)

        # 直交性
        np.testing.assert_allclose(n @ t1, 0.0, atol=1e-14)
        np.testing.assert_allclose(n @ t2, 0.0, atol=1e-14)
        np.testing.assert_allclose(t1 @ t2, 0.0, atol=1e-14)

    def test_various_normals(self):
        """複数の法線方向で正規直交性を確認."""
        normals = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
            np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
        ]
        for normal in normals:
            n, t1, t2 = build_contact_frame(normal)

            np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-14)
            np.testing.assert_allclose(n @ t1, 0.0, atol=1e-14)
            np.testing.assert_allclose(n @ t2, 0.0, atol=1e-14)
            np.testing.assert_allclose(t1 @ t2, 0.0, atol=1e-14)

    def test_continuity_with_prev_tangent(self):
        """前ステップの t1 を与えた場合の連続性."""
        normal = np.array([0.0, 0.0, 1.0])
        prev_t1 = np.array([1.0, 0.0, 0.0])

        n, t1, t2 = build_contact_frame(normal, prev_tangent1=prev_t1)

        # t1 は prev_t1 に近い方向
        np.testing.assert_allclose(t1, prev_t1, atol=1e-14)
        # 直交性
        np.testing.assert_allclose(n @ t1, 0.0, atol=1e-14)
        np.testing.assert_allclose(n @ t2, 0.0, atol=1e-14)

    def test_prev_tangent_slightly_off(self):
        """前ステップの t1 が若干ずれている場合の修正."""
        normal = np.array([0.0, 0.0, 1.0])
        prev_t1 = np.array([1.0, 0.1, 0.05])  # わずかにずれ

        n, t1, t2 = build_contact_frame(normal, prev_tangent1=prev_t1)

        # 正規直交性を確認
        np.testing.assert_allclose(np.linalg.norm(t1), 1.0, atol=1e-14)
        np.testing.assert_allclose(n @ t1, 0.0, atol=1e-14)
        np.testing.assert_allclose(n @ t2, 0.0, atol=1e-14)
        np.testing.assert_allclose(t1 @ t2, 0.0, atol=1e-14)

    def test_right_hand_system(self):
        """t2 = n × t1 で右手系."""
        normal = np.array([0.0, 0.0, 1.0])
        n, t1, t2 = build_contact_frame(normal)

        expected_t2 = np.cross(n, t1)
        np.testing.assert_allclose(t2, expected_t2, atol=1e-14)
