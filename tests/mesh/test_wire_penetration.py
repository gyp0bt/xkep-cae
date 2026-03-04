"""撚線メッシュの非貫入制約テスト.

各撚構成（3/7/19/37/61/91本）で、以下を検証する:
  1. minimum_strand_diameter が正しい最小外径を返す
  2. strand_diameter 指定時に非貫入配置が生成される
  3. 不可能な組み合わせで ValueError が発生する
  4. 3D中心間距離が全ペアで素線直径以上（貫入なし）
"""

import math

import numpy as np
import pytest

from xkep_cae.mesh.twisted_wire import (
    make_strand_layout,
    make_twisted_wire_mesh,
    minimum_strand_diameter,
    validate_strand_geometry,
)

_WIRE_D = 0.002  # 2mm
_WIRE_R = _WIRE_D / 2.0
_PITCH = 0.040
_LENGTH = 0.080
_N_ELEM = 20


def _check_no_penetration_layout(layout, wire_radius, tol=1e-10):
    """配置の全ペアで中心間距離 >= 2*wire_radius を検証."""
    d = 2.0 * wire_radius
    for i, a in enumerate(layout):
        for j, b in enumerate(layout):
            if j <= i:
                continue
            if a.lay_radius < 1e-15 and b.lay_radius < 1e-15:
                continue  # 両方中心（同一素線）
            # 同一配置半径上: 弦距離
            if abs(a.lay_radius - b.lay_radius) < 1e-15:
                dtheta = abs(a.angle_offset - b.angle_offset)
                dist = 2.0 * a.lay_radius * math.sin(dtheta / 2.0)
            else:
                # 異なる配置半径: 最悪ケース = |r_a - r_b|（位相一致時）
                dist = abs(a.lay_radius - b.lay_radius)
            assert dist >= d - tol, f"素線{a.strand_id}-{b.strand_id} 距離 {dist:.6e} < d={d:.6e}"


def _check_no_penetration_mesh(mesh, tol=1e-10):
    """メッシュの全素線ペアで3D中心間距離 >= 2*wire_radius を検証."""
    d = 2.0 * mesh.wire_radius
    n_pts = mesh.n_elems_per_strand + 1
    for i in range(mesh.n_strands):
        ci = mesh.node_coords[mesh.strand_nodes(i)]
        for j in range(i + 1, mesh.n_strands):
            cj = mesh.node_coords[mesh.strand_nodes(j)]
            for k in range(n_pts):
                dist = np.linalg.norm(ci[k] - cj[k])
                assert dist >= d - tol, f"素線{i}-{j} z={ci[k, 2]:.4f} 距離={dist:.6e} < d={d:.6e}"


class TestMinimumStrandDiameter:
    """minimum_strand_diameter のテスト."""

    def test_1_strand(self):
        """1本: 最小外径 = 素線径."""
        assert abs(minimum_strand_diameter(1, _WIRE_D) - _WIRE_D) < 1e-15

    def test_3_strand(self):
        """3本: 最小外径 = 2*(d/√3 + r)."""
        min_d = minimum_strand_diameter(3, _WIRE_D)
        expected = 2.0 * (_WIRE_D / math.sqrt(3.0) + _WIRE_R)
        assert abs(min_d - expected) < 1e-12

    def test_7_strand(self):
        """7本: 最小外径 = 2*(d + r) = 3d."""
        min_d = minimum_strand_diameter(7, _WIRE_D)
        # layer 1: r_lay = d, envelope = d + r = 3r
        expected = 2.0 * (_WIRE_D + _WIRE_R)
        assert abs(min_d - expected) < 1e-12

    def test_19_strand(self):
        """19本: 最小外径 = 2*(2d + r) = 5d."""
        min_d = minimum_strand_diameter(19, _WIRE_D)
        expected = 2.0 * (2.0 * _WIRE_D + _WIRE_R)
        assert abs(min_d - expected) < 1e-12

    def test_monotonic(self):
        """素線数が増えると最小外径も増える."""
        prev = 0.0
        for n in [1, 3, 7, 19, 37, 61, 91]:
            d = minimum_strand_diameter(n, _WIRE_D)
            assert d > prev, f"n={n}: {d} <= {prev}"
            prev = d


class TestValidateStrandGeometry:
    """validate_strand_geometry のテスト."""

    def test_valid_7_strand(self):
        """実現可能な組み合わせでエラーなし."""
        min_d = minimum_strand_diameter(7, _WIRE_D)
        validate_strand_geometry(7, _WIRE_D, min_d * 1.5)  # 余裕あり

    def test_exact_minimum(self):
        """最小外径ちょうどでエラーなし."""
        min_d = minimum_strand_diameter(7, _WIRE_D)
        validate_strand_geometry(7, _WIRE_D, min_d)  # ちょうど

    def test_too_small_raises(self):
        """外径が小さすぎてValueError."""
        min_d = minimum_strand_diameter(7, _WIRE_D)
        with pytest.raises(ValueError, match="小さすぎ"):
            validate_strand_geometry(7, _WIRE_D, min_d * 0.5)

    @pytest.mark.parametrize("n", [3, 7, 19, 37, 61, 91])
    def test_all_configs_too_small(self, n):
        """全構成で外径が小さすぎる場合にValueError."""
        with pytest.raises(ValueError):
            validate_strand_geometry(n, _WIRE_D, _WIRE_D)  # 必ず不足


class TestStrandDiameterLayout:
    """strand_diameter 指定時の配置テスト."""

    @pytest.mark.parametrize("n", [3, 7, 19, 37, 61, 91])
    def test_no_penetration_minimum(self, n):
        """最小外径で非貫入."""
        min_d = minimum_strand_diameter(n, _WIRE_D)
        layout = make_strand_layout(n, _WIRE_R, strand_diameter=min_d)
        _check_no_penetration_layout(layout, _WIRE_R)

    @pytest.mark.parametrize("n", [3, 7, 19, 37, 61, 91])
    def test_no_penetration_with_margin(self, n):
        """外径に余裕がある場合も非貫入."""
        min_d = minimum_strand_diameter(n, _WIRE_D)
        layout = make_strand_layout(n, _WIRE_R, strand_diameter=min_d * 1.2)
        _check_no_penetration_layout(layout, _WIRE_R)

    @pytest.mark.parametrize("n", [3, 7, 19, 37, 61, 91])
    def test_envelope_within_strand_diameter(self, n):
        """全素線が外径内に収まる."""
        min_d = minimum_strand_diameter(n, _WIRE_D)
        sd = min_d * 1.3
        layout = make_strand_layout(n, _WIRE_R, strand_diameter=sd)
        outer_r = sd / 2.0
        for info in layout:
            assert info.lay_radius + info.wire_radius <= outer_r + 1e-12, (
                f"素線{info.strand_id}: lay={info.lay_radius}+r={info.wire_radius} > {outer_r}"
            )


class TestStrandDiameterMesh:
    """strand_diameter 指定時のメッシュ非貫入テスト."""

    @pytest.mark.parametrize("n", [3, 7, 19, 37])
    def test_no_penetration_3d(self, n):
        """3D中心間距離で非貫入を検証."""
        min_d = minimum_strand_diameter(n, _WIRE_D)
        mesh = make_twisted_wire_mesh(
            n,
            _WIRE_D,
            _PITCH,
            _LENGTH,
            _N_ELEM,
            strand_diameter=min_d * 1.1,
        )
        _check_no_penetration_mesh(mesh)

    def test_no_penetration_7_exact_min(self):
        """7本: 最小外径での3D非貫入."""
        min_d = minimum_strand_diameter(7, _WIRE_D)
        mesh = make_twisted_wire_mesh(
            7,
            _WIRE_D,
            _PITCH,
            _LENGTH,
            _N_ELEM,
            strand_diameter=min_d,
        )
        _check_no_penetration_mesh(mesh)

    def test_too_small_raises(self):
        """外径不足でValueError."""
        with pytest.raises(ValueError):
            make_twisted_wire_mesh(
                7,
                _WIRE_D,
                _PITCH,
                _LENGTH,
                _N_ELEM,
                strand_diameter=_WIRE_D * 2.0,
            )


class TestLegacyCompatibility:
    """従来API（gap指定）の後方互換性テスト."""

    @pytest.mark.parametrize("n", [3, 7, 19, 37])
    def test_gap_zero_same_as_before(self, n):
        """gap=0 が従来と同じ配置."""
        layout = make_strand_layout(n, _WIRE_R, gap=0.0)
        for info in layout:
            if info.layer == 0:
                assert info.lay_radius == 0.0
            elif n == 3:
                expected = _WIRE_D / math.sqrt(3.0)
                assert abs(info.lay_radius - expected) < 1e-12
            else:
                expected = info.layer * _WIRE_D
                assert abs(info.lay_radius - expected) < 1e-12

    def test_gap_positive(self):
        """gap > 0 でも非貫入."""
        layout = make_strand_layout(19, _WIRE_R, gap=0.001)
        _check_no_penetration_layout(layout, _WIRE_R)
