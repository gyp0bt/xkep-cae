"""解析的剛体円柱表面の射影テスト.

RigidCylinderSurfaceInput + project_beam_to_cylinder_batch の
幾何計算精度を検証する。

[← README](../../README.md)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xkep_cae.contact.geometry._rigid_surface import (
    RigidCylinderSurfaceInput,
    project_beam_to_cylinder_batch,
)


class TestRigidCylinderProjectionAPI:
    """RigidCylinderSurfaceInput の基本 API テスト."""

    def test_dataclass_frozen(self):
        """RigidCylinderSurfaceInput は frozen dataclass."""
        surf = RigidCylinderSurfaceInput(
            center_ref=np.array([50.0, 10.0, 0.0]),
            radius=5.0,
            axis=np.array([0.0, 0.0, 1.0]),
        )
        with pytest.raises(AttributeError):
            surf.radius = 10.0  # type: ignore[misc]

    def test_single_beam_directly_below(self):
        """円柱直下のビームセグメントへの射影."""
        center = np.array([50.0, 20.0, 0.0])
        radius = 10.0
        axis = np.array([0.0, 0.0, 1.0])

        # ビームは x=40..60, y=0 に配置（円柱直下）
        xA0 = np.array([[40.0, 0.0, 0.0]])
        xA1 = np.array([[60.0, 0.0, 0.0]])

        s, dist, normal = project_beam_to_cylinder_batch(xA0, xA1, center, radius, axis)

        # s=0.5 で円柱中心の真下
        assert abs(s[0] - 0.5) < 1e-10
        # 距離 = 20 - 10 = 10（中心線から円柱表面まで）
        assert abs(dist[0] - 10.0) < 1e-10
        # 法線は下向き (0, -1, 0)（円柱→ビーム方向）
        assert abs(normal[0, 1] - (-1.0)) < 1e-10

    def test_gap_with_beam_radius(self):
        """gap = dist - beam_radius の計算確認."""
        center = np.array([50.0, 15.0, 0.0])
        radius = 5.0
        axis = np.array([0.0, 0.0, 1.0])

        xA0 = np.array([[40.0, 0.0, 0.0]])
        xA1 = np.array([[60.0, 0.0, 0.0]])

        s, dist, normal = project_beam_to_cylinder_batch(xA0, xA1, center, radius, axis)

        beam_radius = 8.5
        gap = dist[0] - beam_radius
        # dist = 15 - 5 = 10, gap = 10 - 8.5 = 1.5
        assert abs(gap - 1.5) < 1e-10

    def test_contact_penetration(self):
        """接触（貫入）時の負のギャップ."""
        center = np.array([50.0, 12.0, 0.0])
        radius = 5.0
        axis = np.array([0.0, 0.0, 1.0])

        xA0 = np.array([[40.0, 0.0, 0.0]])
        xA1 = np.array([[60.0, 0.0, 0.0]])

        s, dist, normal = project_beam_to_cylinder_batch(xA0, xA1, center, radius, axis)

        beam_radius = 8.5
        gap = dist[0] - beam_radius
        # dist = 12 - 5 = 7, gap = 7 - 8.5 = -1.5（貫入）
        assert gap < 0
        assert abs(gap - (-1.5)) < 1e-10


class TestRigidCylinderProjectionPhysics:
    """射影の物理的妥当性テスト."""

    def test_batch_multiple_segments(self):
        """複数セグメントの一括射影."""
        center = np.array([50.0, 20.0, 0.0])
        radius = 10.0
        axis = np.array([0.0, 0.0, 1.0])

        # 3 セグメント
        xA0 = np.array([[20.0, 0.0, 0.0], [40.0, 0.0, 0.0], [60.0, 0.0, 0.0]])
        xA1 = np.array([[40.0, 0.0, 0.0], [60.0, 0.0, 0.0], [80.0, 0.0, 0.0]])

        s, dist, normal = project_beam_to_cylinder_batch(xA0, xA1, center, radius, axis)

        assert len(s) == 3
        # 中央セグメント: s=0.5, dist=10
        assert abs(s[1] - 0.5) < 1e-10
        assert abs(dist[1] - 10.0) < 1e-10

        # 左セグメント: s=1.0（右端が円柱に最も近い）
        assert abs(s[0] - 1.0) < 1e-10
        # 右セグメント: s=0.0（左端が円柱に最も近い）
        assert abs(s[2] - 0.0) < 1e-10

    def test_oblique_beam(self):
        """斜めビームの射影精度."""
        center = np.array([0.0, 10.0, 0.0])
        radius = 5.0
        axis = np.array([0.0, 0.0, 1.0])

        # 45° 斜めビーム
        xA0 = np.array([[-10.0, -5.0, 0.0]])
        xA1 = np.array([[10.0, 5.0, 0.0]])

        s, dist, normal = project_beam_to_cylinder_batch(xA0, xA1, center, radius, axis)

        # 最近接点を手計算
        # p(s) = (-10+20s, -5+10s, 0) - (0, 10, 0) = (-10+20s, -15+10s, 0)
        # |d_perp|² = (-10+20s)² + (-15+10s)²
        # 微分: 2*(-10+20s)*20 + 2*(-15+10s)*10 = 0
        # → -200+400s -150+100s = 0 → 500s = 350 → s = 0.7
        assert abs(s[0] - 0.7) < 1e-10

        # p(0.7) = (4, 2, 0), d_perp = (4, -8, 0), |d| = sqrt(16+64) = sqrt(80)
        expected_dist = math.sqrt(80) - 5.0
        assert abs(dist[0] - expected_dist) < 1e-10

    def test_normal_continuity_across_segments(self):
        """隣接セグメント境界での法線連続性.

        解析的円柱表面の主要利点: 離散セグメント境界での法線不連続がない。
        """
        center = np.array([50.0, 20.0, 0.0])
        radius = 10.0
        axis = np.array([0.0, 0.0, 1.0])

        # 境界点を共有する2セグメント
        xA0 = np.array([[45.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        xA1 = np.array([[50.0, 0.0, 0.0], [55.0, 0.0, 0.0]])

        s, dist, normal = project_beam_to_cylinder_batch(xA0, xA1, center, radius, axis)

        # 左セグメント s=1.0 と右セグメント s=0.0 は同じ点（50, 0, 0）
        # → 法線が完全一致（離散化ジャンプなし）
        np.testing.assert_allclose(normal[0], normal[1], atol=1e-10)
        np.testing.assert_allclose(dist[0], dist[1], atol=1e-10)

    def test_z_axis_independence(self):
        """z 方向に離れたビームでも結果同一（z 軸円柱）."""
        center = np.array([50.0, 20.0, 0.0])
        radius = 10.0
        axis = np.array([0.0, 0.0, 1.0])

        xA0 = np.array([[40.0, 0.0, 0.0], [40.0, 0.0, 100.0]])
        xA1 = np.array([[60.0, 0.0, 0.0], [60.0, 0.0, 100.0]])

        s, dist, normal = project_beam_to_cylinder_batch(xA0, xA1, center, radius, axis)

        assert abs(s[0] - s[1]) < 1e-10
        assert abs(dist[0] - dist[1]) < 1e-10
        np.testing.assert_allclose(normal[0], normal[1], atol=1e-10)
