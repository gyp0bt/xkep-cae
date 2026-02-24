"""撚線メッシュファクトリのテスト.

TwistedWireMesh の幾何整合性・素線配置・ヘリカル形状を検証する。
"""

import math

import numpy as np
import pytest

from xkep_cae.mesh.twisted_wire import (
    StrandInfo,
    TwistedWireMesh,
    compute_helix_angle,
    compute_strand_length_per_pitch,
    make_strand_layout,
    make_twisted_wire_mesh,
)

# ====================================================================
# パラメータ
# ====================================================================

_WIRE_D = 0.002  # 2mm 直径
_WIRE_R = _WIRE_D / 2.0
_PITCH = 0.040  # 40mm ピッチ
_LENGTH = 0.080  # 80mm（2ピッチ分）
_N_ELEM = 20  # 1素線あたり要素数


# ====================================================================
# make_strand_layout テスト
# ====================================================================


class TestMakeStrandLayout:
    """素線配置パターンのテスト."""

    def test_3_strand_layout(self):
        """3本撚り: 中心なし、120°配置."""
        layout = make_strand_layout(3, _WIRE_R)
        assert len(layout) == 3
        # 全素線が layer 1
        for info in layout:
            assert info.layer == 1
            assert info.lay_radius > 0
        # 位相角が120°間隔
        angles = sorted([info.angle_offset for info in layout])
        for i in range(3):
            expected = 2.0 * math.pi * i / 3.0
            assert abs(angles[i] - expected) < 1e-10

    def test_7_strand_layout(self):
        """7本: 中心1本 + 第1層6本."""
        layout = make_strand_layout(7, _WIRE_R)
        assert len(layout) == 7
        # 中心素線
        center = layout[0]
        assert center.layer == 0
        assert center.lay_radius == 0.0
        # 第1層: 6本
        layer1 = [info for info in layout if info.layer == 1]
        assert len(layer1) == 6
        for info in layer1:
            assert info.lay_radius > 0

    def test_19_strand_layout(self):
        """19本: 中心1本 + 第1層6本 + 第2層12本."""
        layout = make_strand_layout(19, _WIRE_R)
        assert len(layout) == 19
        center = [info for info in layout if info.layer == 0]
        layer1 = [info for info in layout if info.layer == 1]
        layer2 = [info for info in layout if info.layer == 2]
        assert len(center) == 1
        assert len(layer1) == 6
        assert len(layer2) == 12

    def test_37_strand_layout(self):
        """37本: 1+6+12+18."""
        layout = make_strand_layout(37, _WIRE_R)
        assert len(layout) == 37
        layers = {}
        for info in layout:
            layers.setdefault(info.layer, []).append(info)
        assert len(layers[0]) == 1
        assert len(layers[1]) == 6
        assert len(layers[2]) == 12
        assert len(layers[3]) == 18

    def test_layer_radii_monotonic(self):
        """外側の層ほど配置半径が大きい."""
        layout = make_strand_layout(19, _WIRE_R)
        by_layer: dict[int, list[StrandInfo]] = {}
        for info in layout:
            by_layer.setdefault(info.layer, []).append(info)
        prev_r = -1.0
        for layer_idx in sorted(by_layer.keys()):
            if layer_idx == 0:
                continue
            r = by_layer[layer_idx][0].lay_radius
            assert r > prev_r, f"層{layer_idx}の半径{r}が前の層{prev_r}以下"
            prev_r = r

    def test_alternating_lay_direction(self):
        """隣接層の撚り方向が交互."""
        layout = make_strand_layout(19, _WIRE_R, lay_direction=1)
        by_layer: dict[int, list[StrandInfo]] = {}
        for info in layout:
            by_layer.setdefault(info.layer, []).append(info)
        if 1 in by_layer and 2 in by_layer:
            dir1 = by_layer[1][0].lay_direction
            dir2 = by_layer[2][0].lay_direction
            assert dir1 == -dir2, "隣接層の撚り方向が同じ"

    def test_3_strand_mutual_contact(self):
        """3本撚りの配置半径: 3素線が互いに接する."""
        layout = make_strand_layout(3, _WIRE_R, gap=0.0)
        r_lay = layout[0].lay_radius
        # 配置半径 = d/√3
        expected = _WIRE_D / math.sqrt(3.0)
        assert abs(r_lay - expected) < 1e-12

    def test_gap_increases_lay_radius(self):
        """ギャップありでは配置半径が大きくなる."""
        layout_no_gap = make_strand_layout(7, _WIRE_R, gap=0.0)
        layout_with_gap = make_strand_layout(7, _WIRE_R, gap=0.001)
        r_no = layout_no_gap[1].lay_radius
        r_gap = layout_with_gap[1].lay_radius
        assert r_gap > r_no

    def test_strand_ids_unique(self):
        """全素線IDが一意."""
        for n in [3, 7, 19, 37]:
            layout = make_strand_layout(n, _WIRE_R)
            ids = [info.strand_id for info in layout]
            assert len(set(ids)) == len(ids)


# ====================================================================
# make_twisted_wire_mesh テスト
# ====================================================================


class TestMakeTwistedWireMesh:
    """撚線メッシュ生成のテスト."""

    def test_basic_3_strand(self):
        """3本撚り基本生成."""
        mesh = make_twisted_wire_mesh(3, _WIRE_D, _PITCH, _LENGTH, _N_ELEM)
        assert isinstance(mesh, TwistedWireMesh)
        assert mesh.n_strands == 3
        assert mesh.n_nodes == 3 * (_N_ELEM + 1)
        assert mesh.n_elems == 3 * _N_ELEM
        assert mesh.node_coords.shape == (mesh.n_nodes, 3)
        assert mesh.connectivity.shape == (mesh.n_elems, 2)

    def test_basic_7_strand(self):
        """7本撚り基本生成."""
        mesh = make_twisted_wire_mesh(7, _WIRE_D, _PITCH, _LENGTH, _N_ELEM)
        assert mesh.n_strands == 7
        assert mesh.n_nodes == 7 * (_N_ELEM + 1)
        assert mesh.n_elems == 7 * _N_ELEM

    def test_basic_19_strand(self):
        """19本撚り基本生成."""
        mesh = make_twisted_wire_mesh(19, _WIRE_D, _PITCH, _LENGTH, _N_ELEM)
        assert mesh.n_strands == 19
        assert mesh.n_nodes == 19 * (_N_ELEM + 1)

    def test_center_strand_is_straight(self):
        """7本撚りの中心素線は直線（z軸上）."""
        mesh = make_twisted_wire_mesh(7, _WIRE_D, _PITCH, _LENGTH, _N_ELEM)
        center_nodes = mesh.strand_nodes(0)
        coords = mesh.node_coords[center_nodes]
        # x, y が 0
        assert np.allclose(coords[:, 0], 0.0, atol=1e-15)
        assert np.allclose(coords[:, 1], 0.0, atol=1e-15)
        # z が 0 → length
        assert abs(coords[0, 2]) < 1e-15
        assert abs(coords[-1, 2] - _LENGTH) < 1e-15

    def test_outer_strand_is_helix(self):
        """7本撚りの外層素線がヘリカル."""
        mesh = make_twisted_wire_mesh(7, _WIRE_D, _PITCH, _LENGTH, _N_ELEM)
        # strand 1 (第1層)
        outer_nodes = mesh.strand_nodes(1)
        coords = mesh.node_coords[outer_nodes]
        # x² + y² ≈ r_lay²
        r_xy = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
        r_lay = mesh.strand_infos[1].lay_radius
        assert np.allclose(r_xy, r_lay, rtol=1e-10)
        # z は 0 → length の範囲
        assert abs(coords[0, 2]) < 1e-15
        assert abs(coords[-1, 2] - _LENGTH) < 1e-15

    def test_connectivity_valid(self):
        """接続行列のインデックスが有効範囲内."""
        mesh = make_twisted_wire_mesh(7, _WIRE_D, _PITCH, _LENGTH, _N_ELEM)
        assert np.all(mesh.connectivity >= 0)
        assert np.all(mesh.connectivity < mesh.n_nodes)

    def test_connectivity_sequential(self):
        """各素線の要素は連続する節点を結ぶ."""
        mesh = make_twisted_wire_mesh(7, _WIRE_D, _PITCH, _LENGTH, _N_ELEM)
        for sid in range(mesh.n_strands):
            elems = mesh.strand_elems(sid)
            for eid in elems:
                n1, n2 = mesh.connectivity[eid]
                assert n2 == n1 + 1

    def test_radii_uniform(self):
        """全要素の断面半径が均一."""
        mesh = make_twisted_wire_mesh(7, _WIRE_D, _PITCH, _LENGTH, _N_ELEM)
        assert np.allclose(mesh.radii, _WIRE_R)

    def test_n_pitches_parameter(self):
        """n_pitchesで長さ指定."""
        mesh = make_twisted_wire_mesh(3, _WIRE_D, _PITCH, 0.0, _N_ELEM, n_pitches=2.5)
        assert abs(mesh.length - 2.5 * _PITCH) < 1e-15

    def test_strand_node_ranges_partition(self):
        """素線ごとの節点範囲が全体をカバーする."""
        mesh = make_twisted_wire_mesh(7, _WIRE_D, _PITCH, _LENGTH, _N_ELEM)
        all_nodes = set()
        for start, end in mesh.strand_node_ranges:
            for n in range(start, end):
                assert n not in all_nodes, f"節点 {n} が重複"
                all_nodes.add(n)
        assert len(all_nodes) == mesh.n_nodes

    def test_strand_elem_ranges_partition(self):
        """素線ごとの要素範囲が全体をカバーする."""
        mesh = make_twisted_wire_mesh(7, _WIRE_D, _PITCH, _LENGTH, _N_ELEM)
        all_elems = set()
        for start, end in mesh.strand_elem_ranges:
            for e in range(start, end):
                assert e not in all_elems, f"要素 {e} が重複"
                all_elems.add(e)
        assert len(all_elems) == mesh.n_elems

    def test_element_lengths_positive(self):
        """全要素の長さが正."""
        mesh = make_twisted_wire_mesh(7, _WIRE_D, _PITCH, _LENGTH, _N_ELEM)
        for eid in range(mesh.n_elems):
            n1, n2 = mesh.connectivity[eid]
            diff = mesh.node_coords[n2] - mesh.node_coords[n1]
            elem_len = np.linalg.norm(diff)
            assert elem_len > 0.0, f"要素{eid}の長さがゼロ"

    def test_no_strand_intersection(self):
        """隣接素線の初期配置で交差がない（中心間距離 > 素線直径）.

        全素線ペアの各z位置での中心間距離が素線直径以上であることを確認。
        """
        mesh = make_twisted_wire_mesh(7, _WIRE_D, _PITCH, _LENGTH, _N_ELEM)
        n_pts = _N_ELEM + 1
        for i in range(mesh.n_strands):
            nodes_i = mesh.strand_nodes(i)
            coords_i = mesh.node_coords[nodes_i]
            for j in range(i + 1, mesh.n_strands):
                nodes_j = mesh.strand_nodes(j)
                coords_j = mesh.node_coords[nodes_j]
                for k in range(n_pts):
                    dist = np.linalg.norm(coords_i[k] - coords_j[k])
                    # 距離 >= 2*radius（接触許容の微小マージン含む）
                    assert dist >= 2.0 * _WIRE_R - 1e-10, (
                        f"素線{i}-{j}がz={coords_i[k, 2]:.4f}で交差: dist={dist:.6f}"
                    )


# ====================================================================
# ヘリカル幾何ユーティリティのテスト
# ====================================================================


class TestHelixGeometryUtils:
    """ヘリックス幾何ユーティリティのテスト."""

    def test_helix_angle_zero_radius(self):
        """配置半径0のヘリックス角は0."""
        alpha = compute_helix_angle(0.0, _PITCH)
        assert abs(alpha) < 1e-15

    def test_helix_angle_large_radius(self):
        """配置半径大のヘリックス角はπ/2に近づく."""
        alpha = compute_helix_angle(1e6, _PITCH)
        assert abs(alpha - math.pi / 2) < 0.01

    def test_helix_angle_known_value(self):
        """既知のヘリックス角: α = arctan(2πR/pitch)."""
        R = 0.005
        alpha = compute_helix_angle(R, _PITCH)
        expected = math.atan2(2 * math.pi * R, _PITCH)
        assert abs(alpha - expected) < 1e-15

    def test_strand_length_per_pitch_center(self):
        """中心素線の1ピッチ長はピッチに等しい."""
        L = compute_strand_length_per_pitch(0.0, _PITCH)
        assert abs(L - _PITCH) < 1e-15

    def test_strand_length_per_pitch_helix(self):
        """ヘリカル素線の1ピッチ長 > ピッチ."""
        L = compute_strand_length_per_pitch(0.005, _PITCH)
        assert L > _PITCH

    def test_strand_length_formula(self):
        """弧長公式: √((2πR)² + pitch²)."""
        R = 0.005
        L = compute_strand_length_per_pitch(R, _PITCH)
        expected = math.sqrt((2 * math.pi * R) ** 2 + _PITCH**2)
        assert abs(L - expected) < 1e-15

    def test_invalid_pitch_raises(self):
        """ピッチ <= 0 でエラー."""
        with pytest.raises(ValueError):
            compute_helix_angle(0.005, 0.0)
        with pytest.raises(ValueError):
            compute_helix_angle(0.005, -1.0)


# ====================================================================
# 大規模メッシュ生成テスト
# ====================================================================


class TestLargeScaleMesh:
    """大規模撚線メッシュの生成テスト."""

    def test_61_strand_mesh(self):
        """61本（1+6+12+18+24）メッシュ生成."""
        mesh = make_twisted_wire_mesh(61, _WIRE_D, _PITCH, _LENGTH, 10)
        assert mesh.n_strands == 61
        assert mesh.n_nodes == 61 * 11
        assert mesh.n_elems == 61 * 10

    def test_91_strand_mesh(self):
        """91本（1+6+12+18+24+30）メッシュ生成."""
        mesh = make_twisted_wire_mesh(91, _WIRE_D, _PITCH, _LENGTH, 10)
        assert mesh.n_strands == 91

    def test_element_quality(self):
        """要素長のばらつきが小さい（最大/最小 < 3）.

        ヘリカル素線は直線より長くなるが、
        要素分割の均一性を確認する。
        """
        mesh = make_twisted_wire_mesh(7, _WIRE_D, _PITCH, _LENGTH, _N_ELEM)
        lengths = []
        for eid in range(mesh.n_elems):
            n1, n2 = mesh.connectivity[eid]
            diff = mesh.node_coords[n2] - mesh.node_coords[n1]
            lengths.append(float(np.linalg.norm(diff)))
        lengths = np.array(lengths)
        ratio = lengths.max() / lengths.min()
        assert ratio < 3.0, f"要素長の最大/最小比 {ratio:.2f} > 3.0"
