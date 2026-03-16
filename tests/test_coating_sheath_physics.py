"""被覆線・シースモデルの物理テスト.

CLAUDE.md の物理テスト思想に基づく:
  「物理的に当然の性質」をコード化して検証する。

テスト方針:
  1. 被膜は剛性を増加させる（同材質なら中実断面と一致）
  2. 被膜の剛性寄与は材料比率に比例する
  3. シースは撚線束を囲み、幾何学的に整合する
  4. 層が増えるほどシースが大きくなる
  5. エネルギー的に整合: 複合断面のひずみエネルギー = 素線 + 被膜
"""

from __future__ import annotations

import math

import numpy as np
from xkep_cae.mesh.twisted_wire import (
    CoatingModel,
    SheathModel,
    coated_beam_section,
    coated_contact_radius,
    coating_section_properties,
    compute_envelope_radius,
    make_twisted_wire_mesh,
    sheath_equivalent_stiffness,
    sheath_inner_radius,
    sheath_radial_gap,
    sheath_section_properties,
)


def _make_mesh(n_strands: int, n_elems: int = 16):
    return make_twisted_wire_mesh(
        n_strands=n_strands,
        wire_diameter=2.0e-3,
        pitch=40.0e-3,
        length=40.0e-3,
        n_elems_per_strand=n_elems,
    )


# ====================================================================
# 被膜モデル物理テスト
# ====================================================================


class TestCoatingPhysics:
    """被膜の物理的性質テスト."""

    def test_coating_always_increases_stiffness(self):
        """被膜は常に剛性を増加させる（どんな材料でも）."""
        r_wire = 1.0e-3
        E_wire = 200e9
        nu_wire = 0.3

        # いろいろな被膜材料
        for E_coat in [0.1e9, 1.0e9, 10.0e9, 200e9]:
            coat = CoatingModel(thickness=0.05e-3, E=E_coat, nu=0.35)
            props = coated_beam_section(r_wire, E_wire, nu_wire, coat)

            EA_wire = E_wire * math.pi * r_wire**2
            EI_wire = E_wire * math.pi / 4.0 * r_wire**4

            assert props["EA"] > EA_wire, f"EA should increase with E_coat={E_coat}"
            assert props["EIy"] > EI_wire, f"EIy should increase with E_coat={E_coat}"

    def test_thicker_coating_stiffer(self):
        """厚い被膜ほど剛性が大きい."""
        r_wire = 1.0e-3
        E_wire = 200e9
        nu_wire = 0.3
        coat_E = 3.0e9

        EA_prev = 0.0
        for t in [0.01e-3, 0.05e-3, 0.1e-3, 0.2e-3, 0.5e-3]:
            coat = CoatingModel(thickness=t, E=coat_E, nu=0.35)
            props = coated_beam_section(r_wire, E_wire, nu_wire, coat)
            assert props["EA"] > EA_prev, f"EA should increase with thickness={t}"
            EA_prev = props["EA"]

    def test_coating_stiffness_scales_with_modulus_ratio(self):
        """被膜寄与は E_coat/E_wire 比に概ね比例する."""
        r_wire = 1.0e-3
        E_wire = 200e9
        nu_wire = 0.3
        t = 0.1e-3

        # E_coat = E_wire/100 vs E_coat = E_wire/10
        coat_soft = CoatingModel(thickness=t, E=E_wire / 100, nu=0.35)
        coat_mid = CoatingModel(thickness=t, E=E_wire / 10, nu=0.35)

        p_soft = coated_beam_section(r_wire, E_wire, nu_wire, coat_soft)
        p_mid = coated_beam_section(r_wire, E_wire, nu_wire, coat_mid)

        EA_wire = E_wire * math.pi * r_wire**2
        delta_soft = p_soft["EA"] - EA_wire
        delta_mid = p_mid["EA"] - EA_wire

        # 10倍硬い被膜 → 寄与も約10倍（同じ断面形状）
        ratio = delta_mid / delta_soft
        assert 8.0 < ratio < 12.0, f"Stiffness contribution ratio should be ~10: {ratio:.2f}"

    def test_coating_contact_radius_geometry(self):
        """被膜込み接触半径 = 素線半径 + 被膜厚（厳密）."""
        r_wire = 1.5e-3
        for t in [0.01e-3, 0.1e-3, 0.5e-3]:
            coat = CoatingModel(thickness=t, E=3e9, nu=0.35)
            r_contact = coated_contact_radius(r_wire, coat)
            assert abs(r_contact - (r_wire + t)) < 1e-15

    def test_annular_area_positive_and_bounded(self):
        """環状断面積は正で、外接円の面積より小さい."""
        r_wire = 1.0e-3
        t = 0.2e-3
        coat = CoatingModel(thickness=t, E=3e9, nu=0.35)
        props = coating_section_properties(r_wire, coat)

        assert props["A"] > 0
        r_out = r_wire + t
        A_full_circle = math.pi * r_out**2
        assert props["A"] < A_full_circle

    def test_strain_energy_additivity(self):
        """複合断面の軸ひずみエネルギー = 素線 + 被膜の和.

        U = (1/2) * EA * ε² * L → EA_total = EA_wire + EA_coat
        """
        r_wire = 1.0e-3
        E_wire = 200e9
        nu_wire = 0.3
        t = 0.1e-3
        E_coat = 3.0e9

        coat = CoatingModel(thickness=t, E=E_coat, nu=0.35)
        props = coated_beam_section(r_wire, E_wire, nu_wire, coat)
        cp = coating_section_properties(r_wire, coat)

        # 直接計算
        EA_wire = E_wire * math.pi * r_wire**2
        EA_coat = E_coat * cp["A"]
        EA_expected = EA_wire + EA_coat

        assert abs(props["EA"] - EA_expected) / EA_expected < 1e-12


# ====================================================================
# シースモデル物理テスト
# ====================================================================


class TestSheathPhysics:
    """シースの物理的性質テスト."""

    def test_sheath_encloses_all_wires(self):
        """シース内径は全素線外表面より外にある（ギャップ≥0）."""
        for n in [3, 7, 19]:
            mesh = _make_mesh(n, n_elems=16)
            sheath = SheathModel(thickness=0.5e-3, E=70e9, nu=0.33)
            gaps = sheath_radial_gap(mesh, sheath)
            assert np.all(gaps >= -1e-10), (
                f"{n}本撚り: シース内に貫入あり, min gap={gaps.min():.6e}"
            )

    def test_sheath_encloses_coated_wires(self):
        """被膜付きでもシースが全ワイヤを囲む."""
        mesh = _make_mesh(7, n_elems=16)
        coat = CoatingModel(thickness=0.1e-3, E=3e9, nu=0.35)
        sheath = SheathModel(thickness=0.5e-3, E=70e9, nu=0.33)
        gaps = sheath_radial_gap(mesh, sheath, coating=coat)
        assert np.all(gaps >= -1e-10)

    def test_larger_strand_needs_larger_sheath(self):
        """素線本数が多いほどシースが大きい."""
        sheath = SheathModel(thickness=0.5e-3, E=70e9, nu=0.33)
        r_prev = 0.0
        for n in [3, 7, 19]:
            mesh = _make_mesh(n)
            r = sheath_inner_radius(mesh, sheath)
            assert r > r_prev, f"{n}本: r_inner={r:.6e} should > {r_prev:.6e}"
            r_prev = r

    def test_sheath_radius_geometry_consistency(self):
        """シース内径 = エンベロープ半径 + クリアランス（厳密）."""
        mesh = _make_mesh(7)
        clearance = 0.2e-3
        sheath = SheathModel(thickness=0.5e-3, E=70e9, nu=0.33, clearance=clearance)
        r_env = compute_envelope_radius(mesh)
        r_in = sheath_inner_radius(mesh, sheath)
        assert abs(r_in - (r_env + clearance)) < 1e-15

    def test_sheath_stiffness_proportional_to_modulus(self):
        """シース剛性はヤング率に比例する."""
        mesh = _make_mesh(7)
        s1 = SheathModel(thickness=0.5e-3, E=70e9, nu=0.33)
        s2 = SheathModel(thickness=0.5e-3, E=140e9, nu=0.33)
        stiff1 = sheath_equivalent_stiffness(mesh, s1)
        stiff2 = sheath_equivalent_stiffness(mesh, s2)

        ratio = stiff2["EA"] / stiff1["EA"]
        assert abs(ratio - 2.0) < 1e-10, f"EA ratio should be 2.0: {ratio:.6f}"

    def test_sheath_thicker_is_stiffer(self):
        """厚いシースほど剛性が大きい."""
        mesh = _make_mesh(7)
        thin = SheathModel(thickness=0.2e-3, E=70e9, nu=0.33)
        thick = SheathModel(thickness=1.0e-3, E=70e9, nu=0.33)
        s_thin = sheath_equivalent_stiffness(mesh, thin)
        s_thick = sheath_equivalent_stiffness(mesh, thick)

        assert s_thick["EA"] > s_thin["EA"]
        assert s_thick["EIy"] > s_thin["EIy"]
        assert s_thick["GJ"] > s_thin["GJ"]

    def test_clearance_increases_sheath_radius(self):
        """クリアランスが大きいほどシース内径が大きい."""
        mesh = _make_mesh(7)
        s0 = SheathModel(thickness=0.5e-3, E=70e9, nu=0.33, clearance=0.0)
        s1 = SheathModel(thickness=0.5e-3, E=70e9, nu=0.33, clearance=0.5e-3)
        r0 = sheath_inner_radius(mesh, s0)
        r1 = sheath_inner_radius(mesh, s1)
        assert r1 > r0
        assert abs(r1 - r0 - 0.5e-3) < 1e-15

    def test_sheath_section_area_ring_formula(self):
        """シース断面積 = π(r_out² - r_in²) で幾何学的に正確."""
        mesh = _make_mesh(7)
        sheath = SheathModel(thickness=0.5e-3, E=70e9, nu=0.33)
        props = sheath_section_properties(mesh, sheath)

        r_in = props["r_inner"]
        r_out = props["r_outer"]
        A_expected = math.pi * (r_out**2 - r_in**2)
        assert abs(props["A"] - A_expected) / A_expected < 1e-12


# ====================================================================
# 被膜+シース複合物理テスト
# ====================================================================


class TestCoatingSheathCombinedPhysics:
    """被膜とシースを組み合わせた物理テスト."""

    def test_coating_shifts_envelope_outward(self):
        """被膜によりエンベロープ半径が被膜厚さ分だけ大きくなる."""
        mesh = _make_mesh(7)
        coat = CoatingModel(thickness=0.15e-3, E=3e9, nu=0.35)
        r_bare = compute_envelope_radius(mesh)
        r_coated = compute_envelope_radius(mesh, coating=coat)
        assert abs(r_coated - r_bare - coat.thickness) < 1e-15

    def test_sheath_with_coating_larger_than_without(self):
        """被膜ありのシースは被膜なしより大きい."""
        mesh = _make_mesh(7)
        coat = CoatingModel(thickness=0.1e-3, E=3e9, nu=0.35)
        sheath = SheathModel(thickness=0.5e-3, E=70e9, nu=0.33)

        r_bare = sheath_inner_radius(mesh, sheath)
        r_coated = sheath_inner_radius(mesh, sheath, coating=coat)
        assert r_coated > r_bare

    def test_sheath_stiffness_with_coating_differs(self):
        """被膜ありのシース剛性は被膜なしと異なる（内径が変わるため）."""
        mesh = _make_mesh(7)
        coat = CoatingModel(thickness=0.1e-3, E=3e9, nu=0.35)
        sheath = SheathModel(thickness=0.5e-3, E=70e9, nu=0.33)

        s_bare = sheath_equivalent_stiffness(mesh, sheath)
        s_coated = sheath_equivalent_stiffness(mesh, sheath, coating=coat)

        # 被膜によりシース内径が大きくなる → EI が増加
        assert s_coated["EIy"] > s_bare["EIy"]

    def test_19wire_sheath_stiffer_than_7wire(self):
        """19本撚りのシースは7本より内径が大きい → 同肉厚でEIが大きい."""
        sheath = SheathModel(thickness=0.5e-3, E=70e9, nu=0.33)
        mesh7 = _make_mesh(7)
        mesh19 = _make_mesh(19)

        s7 = sheath_equivalent_stiffness(mesh7, sheath)
        s19 = sheath_equivalent_stiffness(mesh19, sheath)

        assert s19["EIy"] > s7["EIy"]
        assert s19["EA"] > s7["EA"]
