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
            min_elems_per_pitch=0,
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
            min_elems_per_pitch=0,
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
                min_elems_per_pitch=0,
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


class TestMeshDensityValidation:
    """要素密度バリデーションのテスト（恒久対策）.

    粗すぎるメッシュは弦近似による初期貫入を引き起こすため、
    デフォルトで ValueError を発生させる。
    """

    def test_coarse_mesh_raises_by_default(self):
        """4要素/ピッチはデフォルトでValueError."""
        with pytest.raises(ValueError, match="要素密度が不足"):
            make_twisted_wire_mesh(7, _WIRE_D, _PITCH, 0.0, 4, n_pitches=1.0)

    def test_adequate_mesh_passes(self):
        """16要素/ピッチはデフォルトで通過."""
        mesh = make_twisted_wire_mesh(7, _WIRE_D, _PITCH, 0.0, 16, n_pitches=1.0)
        assert mesh.n_strands == 7

    def test_explicit_opt_out(self):
        """min_elems_per_pitch=0で検査スキップ."""
        mesh = make_twisted_wire_mesh(
            7,
            _WIRE_D,
            _PITCH,
            0.0,
            4,
            n_pitches=1.0,
            min_elems_per_pitch=0,
        )
        assert mesh.n_strands == 7

    def test_custom_threshold(self):
        """カスタム閾値: 8要素/ピッチ."""
        # 8要素/ピッチで閾値8 → ちょうどOK
        mesh = make_twisted_wire_mesh(
            7,
            _WIRE_D,
            _PITCH,
            0.0,
            8,
            n_pitches=1.0,
            min_elems_per_pitch=8,
        )
        assert mesh.n_strands == 7
        # 7要素/ピッチで閾値8 → NG
        with pytest.raises(ValueError):
            make_twisted_wire_mesh(
                7,
                _WIRE_D,
                _PITCH,
                0.0,
                7,
                n_pitches=1.0,
                min_elems_per_pitch=8,
            )

    def test_single_strand_no_check(self):
        """1本撚り(中心線のみ)は密度チェック不要."""
        mesh = make_twisted_wire_mesh(1, _WIRE_D, _PITCH, 0.0, 4, n_pitches=1.0)
        assert mesh.n_strands == 1


class TestChordApproximationPenetration:
    """弦近似による初期貫入量の物理テスト.

    16要素/ピッチ以上で初期貫入がワイヤ直径の2%以下であることを検証。
    これはプログラムの正しさではなく、物理的に当然の性質の検証。
    """

    def test_penetration_below_2_percent_at_16_elems(self):
        """16要素/ピッチで最大初期貫入 < 2% wire_diameter."""
        mesh = make_twisted_wire_mesh(
            7,
            _WIRE_D,
            _PITCH,
            0.0,
            16,
            n_pitches=1.0,
        )
        max_penetration = _max_chord_penetration(mesh)
        assert max_penetration < 0.02 * _WIRE_D, (
            f"初期貫入 {max_penetration:.2e} > 2% of d={_WIRE_D:.2e}"
        )

    def test_penetration_below_1_percent_at_32_elems(self):
        """32要素/ピッチで最大初期貫入 < 1% wire_diameter."""
        mesh = make_twisted_wire_mesh(
            7,
            _WIRE_D,
            _PITCH,
            0.0,
            32,
            n_pitches=1.0,
        )
        max_penetration = _max_chord_penetration(mesh)
        assert max_penetration < 0.01 * _WIRE_D, (
            f"初期貫入 {max_penetration:.2e} > 1% of d={_WIRE_D:.2e}"
        )

    def test_coarse_mesh_has_large_penetration(self):
        """4要素/ピッチでは大きな初期貫入が生じる（物理的に当然）."""
        mesh = make_twisted_wire_mesh(
            7,
            _WIRE_D,
            _PITCH,
            0.0,
            4,
            n_pitches=1.0,
            min_elems_per_pitch=0,
        )
        max_penetration = _max_chord_penetration(mesh)
        # 4要素/ピッチ → θ=90° → cos(45°)≈0.707 → 貫入≈29%*r_lay
        # 物理的に意味のある貫入量（>5% wire_d）が存在するはず
        assert max_penetration > 0.05 * _WIRE_D, (
            f"粗メッシュなのに貫入量が小さい: {max_penetration:.2e}"
        )


def _max_chord_penetration(mesh, n_subdiv=10):
    """メッシュの最大弦近似貫入量を計算.

    弦近似誤差は要素の中間点で最大になる（節点はヘリカル曲線上にある）。
    各要素を n_subdiv 分割して線形補間し、素線ペア間の最小距離と
    理論上の最小距離(2*wire_radius)の差を求める。
    """
    d = 2.0 * mesh.wire_radius
    max_pen = 0.0
    n_elems = mesh.n_elems_per_strand
    for i in range(mesh.n_strands):
        ci = mesh.node_coords[mesh.strand_nodes(i)]
        for j in range(i + 1, mesh.n_strands):
            cj = mesh.node_coords[mesh.strand_nodes(j)]
            for k in range(n_elems):
                for s in range(1, n_subdiv):
                    t = s / n_subdiv
                    pi = ci[k] * (1 - t) + ci[k + 1] * t
                    pj = cj[k] * (1 - t) + cj[k + 1] * t
                    dist = np.linalg.norm(pi - pj)
                    pen = max(0.0, d - dist)
                    if pen > max_pen:
                        max_pen = pen
    return max_pen
