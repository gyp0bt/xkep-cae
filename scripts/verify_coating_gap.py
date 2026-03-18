"""被膜厚考慮メッシュ配置の物理的妥当性検証.

status-137: gap_offset廃止 → メッシュ生成時に被膜厚分のgapを確保する
アプローチの検証スクリプト。

検証内容:
1. 被膜なしメッシュ: gap=0で密着配置 → 初期貫入なし（弦近似誤差のみ）
2. 被膜ありメッシュ: coating_thickness指定 → 被膜表面が密着、初期貫入なし
3. 被膜ありメッシュ: coating_thickness未指定（旧動作） → 初期貫入あり
4. 配置半径の検証: r_lay = layer * (d_eff + gap) の確認
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from xkep_cae.contact._manager_process import (
    DetectCandidatesInput,
    DetectCandidatesProcess,
)
from xkep_cae.contact.pair import ContactConfig, ContactManager
from xkep_cae.mesh.twisted_wire import (
    CoatingModel,
    coated_radii,
    make_strand_layout,
    make_twisted_wire_mesh,
)


def check_initial_penetration(mesh, radii_arr, label: str) -> tuple[int, float]:
    """メッシュの初期貫入をContactManagerでチェックする."""
    # 同層除外マップを構築
    elem_layer_map = {}
    for info in mesh.strand_infos:
        start, end = mesh.strand_elem_ranges[info.strand_id]
        for e in range(start, end):
            elem_layer_map[e] = info.layer

    mgr = ContactManager(
        config=ContactConfig(
            elem_layer_map=elem_layer_map,
            exclude_same_layer=True,
            midpoint_prescreening=True,
        )
    )
    det_out = DetectCandidatesProcess().process(DetectCandidatesInput(
        manager=mgr, node_coords=mesh.node_coords,
        connectivity=mesh.connectivity, radii=radii_arr, margin=0.005,
    ))
    mgr = det_out.manager
    n_pen = mgr.check_initial_penetration(mesh.node_coords)

    # 最大貫入量を計算
    max_pen = 0.0
    if n_pen > 0:
        from xkep_cae.contact.geometry import closest_point_segments, compute_gap

        for pair in mgr.pairs:
            xA0 = mesh.node_coords[pair.nodes_a[0]]
            xA1 = mesh.node_coords[pair.nodes_a[1]]
            xB0 = mesh.node_coords[pair.nodes_b[0]]
            xB1 = mesh.node_coords[pair.nodes_b[1]]
            result = closest_point_segments(xA0, xA1, xB0, xB1)
            gap = compute_gap(result.distance, pair.radius_a, pair.radius_b)
            if gap < 0:
                max_pen = max(max_pen, abs(gap))

    print(f"  [{label}] 候補ペア: {len(mgr.pairs)}, 貫入ペア: {n_pen}, "
          f"最大貫入: {max_pen * 1000:.6f} mm")
    return n_pen, max_pen


def test_case_1_no_coating():
    """検証1: 被膜なし、gap=0で密着配置."""
    print("\n=== 検証1: 被膜なし（gap=0, 密着配置） ===")
    mesh = make_twisted_wire_mesh(
        7,
        wire_diameter=0.004,
        pitch=0.050,
        length=0.0,
        n_elems_per_strand=32,
        n_pitches=1.0,
    )
    radii = np.full(mesh.n_elems, mesh.wire_radius)
    n_pen, max_pen = check_initial_penetration(mesh, radii, "被膜なし gap=0")

    d = 2 * mesh.wire_radius
    layout = make_strand_layout(7, mesh.wire_radius)
    r_lay = layout[1].lay_radius
    print(f"  配置半径: {r_lay * 1000:.4f} mm, 素線直径: {d * 1000:.4f} mm")

    # 弦近似による微小貫入（16要素/ピッチで<2%、32要素/ピッチで<0.5%）
    if n_pen > 0:
        pen_ratio = max_pen / d
        print(f"  弦近似貫入率: {pen_ratio:.4%}")
        assert pen_ratio < 0.02, f"被膜なしで2%超の貫入は異常: {pen_ratio:.4%}"
    print("  => OK")


def test_case_2_coating_with_thickness():
    """検証2: 被膜あり、coating_thickness指定でメッシュ生成."""
    print("\n=== 検証2: 被膜あり（coating_thickness指定） ===")
    wire_d = 0.004
    wire_r = wire_d / 2.0
    coat_t = 0.0002  # 200μm 被膜

    coating = CoatingModel(thickness=coat_t, E=1e8, nu=0.4)

    mesh = make_twisted_wire_mesh(
        7,
        wire_diameter=wire_d,
        pitch=0.050,
        length=0.0,
        n_elems_per_strand=32,
        n_pitches=1.0,
        coating_thickness=coat_t,
    )

    radii = coated_radii(mesh, coating)
    n_pen, max_pen = check_initial_penetration(mesh, radii, "被膜あり coat指定")

    layout = make_strand_layout(7, wire_r, coating_thickness=coat_t)
    r_lay = layout[1].lay_radius
    d_eff = 2 * (wire_r + coat_t)
    print(f"  配置半径: {r_lay * 1000:.4f} mm, 有効直径: {d_eff * 1000:.4f} mm")

    assert abs(r_lay - d_eff) < 1e-12, f"配置半径不一致: {r_lay} vs {d_eff}"
    if n_pen > 0:
        pen_ratio = max_pen / d_eff
        print(f"  弦近似貫入率: {pen_ratio:.4%}")
        assert pen_ratio < 0.02, f"被膜考慮済みで2%超の貫入は異常: {pen_ratio:.4%}"
    print("  => OK: 被膜厚分のgapが確保されている")


def test_case_3_coating_without_thickness():
    """検証3: 被膜あり、coating_thickness未指定（旧動作）→ 初期貫入発生."""
    print("\n=== 検証3: 被膜あり（coating_thickness未指定 = 旧動作） ===")
    wire_d = 0.004
    coat_t = 0.0002

    coating = CoatingModel(thickness=coat_t, E=1e8, nu=0.4)

    mesh = make_twisted_wire_mesh(
        7,
        wire_diameter=wire_d,
        pitch=0.050,
        length=0.0,
        n_elems_per_strand=32,
        n_pitches=1.0,
        # coating_thickness 未指定 → 被膜厚は考慮されない
    )

    radii = coated_radii(mesh, coating)
    n_pen, max_pen = check_initial_penetration(mesh, radii, "被膜あり coat未指定")

    expected_pen = 2 * coat_t
    print(f"  期待される初期貫入量: ~{expected_pen * 1000:.4f} mm")
    if n_pen > 0:
        print(f"  => 旧動作の問題を確認: 被膜厚分({max_pen * 1000:.4f} mm)の貫入あり")
    else:
        print("  => 貫入なし（弦近似で偶然回避した可能性）")


def test_case_4_layout_radius_comparison():
    """検証4: 被膜あり/なしの配置半径比較."""
    print("\n=== 検証4: 配置半径の比較 ===")
    wire_r = 0.002
    coat_t = 0.0002

    layout_no_coat = make_strand_layout(7, wire_r)
    layout_with_coat = make_strand_layout(7, wire_r, coating_thickness=coat_t)

    r_no = layout_no_coat[1].lay_radius
    r_with = layout_with_coat[1].lay_radius

    d = 2 * wire_r
    d_eff = 2 * (wire_r + coat_t)

    print(f"  被膜なし: r_lay = {r_no * 1000:.4f} mm (期待: d={d * 1000:.4f} mm)")
    print(f"  被膜あり: r_lay = {r_with * 1000:.4f} mm (期待: d_eff={d_eff * 1000:.4f} mm)")
    print(f"  差分: {(r_with - r_no) * 1000:.4f} mm "
          f"(期待: 2*coat_t={2 * coat_t * 1000:.4f} mm)")

    assert abs(r_no - d) < 1e-12
    assert abs(r_with - d_eff) < 1e-12
    assert abs((r_with - r_no) - 2 * coat_t) < 1e-12
    print("  => OK: 被膜厚分だけ配置半径が増大")


if __name__ == "__main__":
    print("=" * 60)
    print("被膜厚考慮メッシュ配置の物理的妥当性検証")
    print("status-137: gap_offset廃止 → メッシュ側でgap確保")
    print("=" * 60)

    test_case_1_no_coating()
    test_case_2_coating_with_thickness()
    test_case_3_coating_without_thickness()
    test_case_4_layout_radius_comparison()

    print("\n" + "=" * 60)
    print("全検証完了")
    print("=" * 60)
