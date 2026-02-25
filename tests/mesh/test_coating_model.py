"""被膜モデル（CoatingModel）のテスト.

被膜の断面特性計算、等価断面剛性、接触半径の検証。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xkep_cae.mesh.twisted_wire import (
    CoatingModel,
    coated_beam_section,
    coated_contact_radius,
    coated_radii,
    coating_section_properties,
    make_twisted_wire_mesh,
)

# ====================================================================
# CoatingModel 基本テスト
# ====================================================================


class TestCoatingModel:
    """CoatingModel の基本テスト."""

    def test_basic_creation(self):
        """基本的なインスタンス生成."""
        cm = CoatingModel(thickness=0.1e-3, E=3.0e9, nu=0.35, mu=0.2)
        assert cm.thickness == 0.1e-3
        assert cm.E == 3.0e9
        assert cm.nu == 0.35
        assert cm.mu == 0.2

    def test_default_mu(self):
        """摩擦係数のデフォルト値."""
        cm = CoatingModel(thickness=0.1e-3, E=3.0e9, nu=0.35)
        assert cm.mu == 0.3

    def test_shear_modulus(self):
        """せん断弾性係数 G = E / 2(1+ν)."""
        cm = CoatingModel(thickness=0.1e-3, E=3.0e9, nu=0.25)
        expected_G = 3.0e9 / (2.0 * 1.25)
        assert abs(cm.G - expected_G) < 1e-6

    def test_invalid_thickness(self):
        """厚さが非正ならエラー."""
        with pytest.raises(ValueError, match="被膜厚さ"):
            CoatingModel(thickness=0.0, E=3.0e9, nu=0.35)
        with pytest.raises(ValueError, match="被膜厚さ"):
            CoatingModel(thickness=-0.1e-3, E=3.0e9, nu=0.35)

    def test_invalid_E(self):
        """ヤング率が非正ならエラー."""
        with pytest.raises(ValueError, match="ヤング率"):
            CoatingModel(thickness=0.1e-3, E=0.0, nu=0.35)

    def test_invalid_nu(self):
        """ポアソン比が範囲外ならエラー."""
        with pytest.raises(ValueError, match="ポアソン比"):
            CoatingModel(thickness=0.1e-3, E=3.0e9, nu=0.5)
        with pytest.raises(ValueError, match="ポアソン比"):
            CoatingModel(thickness=0.1e-3, E=3.0e9, nu=-1.0)

    def test_invalid_mu(self):
        """摩擦係数が負ならエラー."""
        with pytest.raises(ValueError, match="摩擦係数"):
            CoatingModel(thickness=0.1e-3, E=3.0e9, nu=0.35, mu=-0.1)


# ====================================================================
# coating_section_properties テスト
# ====================================================================


class TestCoatingSectionProperties:
    """被膜断面特性計算のテスト."""

    def test_annular_area(self):
        """環状断面積 A = π(r_out² - r_in²)."""
        r_wire = 1.0e-3  # 1mm
        coating = CoatingModel(thickness=0.1e-3, E=3.0e9, nu=0.35)
        props = coating_section_properties(r_wire, coating)

        r_out = r_wire + coating.thickness
        expected_A = math.pi * (r_out**2 - r_wire**2)
        assert abs(props["A"] - expected_A) / expected_A < 1e-12

    def test_annular_inertia(self):
        """環状断面二次モーメント I = π/4 (r_out⁴ - r_in⁴)."""
        r_wire = 1.0e-3
        coating = CoatingModel(thickness=0.2e-3, E=3.0e9, nu=0.35)
        props = coating_section_properties(r_wire, coating)

        r_out = r_wire + coating.thickness
        expected_I = math.pi / 4.0 * (r_out**4 - r_wire**4)
        assert abs(props["Iy"] - expected_I) / expected_I < 1e-12
        assert abs(props["Iz"] - expected_I) / expected_I < 1e-12

    def test_annular_torsion(self):
        """ねじり定数 J = π/2 (r_out⁴ - r_in⁴)."""
        r_wire = 1.0e-3
        coating = CoatingModel(thickness=0.15e-3, E=3.0e9, nu=0.35)
        props = coating_section_properties(r_wire, coating)

        r_out = r_wire + coating.thickness
        expected_J = math.pi / 2.0 * (r_out**4 - r_wire**4)
        assert abs(props["J"] - expected_J) / expected_J < 1e-12

    def test_Iy_equals_Iz(self):
        """円環対称なので Iy = Iz."""
        r_wire = 2.0e-3
        coating = CoatingModel(thickness=0.3e-3, E=1.0e9, nu=0.4)
        props = coating_section_properties(r_wire, coating)
        assert abs(props["Iy"] - props["Iz"]) < 1e-20

    def test_thin_coating_approximation(self):
        """薄肉近似: A ≈ 2πrt, I ≈ πr³t."""
        r_wire = 5.0e-3  # 5mm（大きい半径で薄肉近似が良好）
        t = 0.01e-3  # 0.01mm（非常に薄い）
        coating = CoatingModel(thickness=t, E=3.0e9, nu=0.35)
        props = coating_section_properties(r_wire, coating)

        # 薄肉近似
        A_approx = 2.0 * math.pi * r_wire * t
        I_approx = math.pi * r_wire**3 * t

        assert abs(props["A"] - A_approx) / A_approx < 0.01  # 1%以内
        assert abs(props["Iy"] - I_approx) / I_approx < 0.01


# ====================================================================
# coated_beam_section テスト
# ====================================================================


class TestCoatedBeamSection:
    """等価断面剛性テスト."""

    def test_stiffness_increases(self):
        """被膜により軸・曲げ・ねじり剛性が増加."""
        r_wire = 1.0e-3
        E_wire = 200.0e9
        nu_wire = 0.3
        coating = CoatingModel(thickness=0.1e-3, E=3.0e9, nu=0.35)

        props = coated_beam_section(r_wire, E_wire, nu_wire, coating)

        # 素線のみの剛性
        A_wire = math.pi * r_wire**2
        I_wire = math.pi / 4.0 * r_wire**4
        J_wire = math.pi / 2.0 * r_wire**4
        G_wire = E_wire / (2.0 * (1.0 + nu_wire))

        EA_wire = E_wire * A_wire
        EI_wire = E_wire * I_wire
        GJ_wire = G_wire * J_wire

        assert props["EA"] > EA_wire
        assert props["EIy"] > EI_wire
        assert props["EIz"] > EI_wire
        assert props["GJ"] > GJ_wire

    def test_stiffness_ratio(self):
        """ヤング率比が正しい."""
        E_wire = 200.0e9
        coating = CoatingModel(thickness=0.1e-3, E=4.0e9, nu=0.35)
        props = coated_beam_section(1.0e-3, E_wire, 0.3, coating)
        assert abs(props["n_axial"] - 4.0e9 / 200.0e9) < 1e-15

    def test_same_material_equals_solid(self):
        """素線と同じ材料の被膜 → 被膜込み半径の中実円断面と一致."""
        r_wire = 1.0e-3
        t = 0.5e-3
        E = 200.0e9
        nu = 0.3
        coating = CoatingModel(thickness=t, E=E, nu=nu)

        props = coated_beam_section(r_wire, E, nu, coating)

        r_total = r_wire + t
        EA_solid = E * math.pi * r_total**2
        EI_solid = E * math.pi / 4.0 * r_total**4
        G = E / (2.0 * (1.0 + nu))
        GJ_solid = G * math.pi / 2.0 * r_total**4

        assert abs(props["EA"] - EA_solid) / EA_solid < 1e-12
        assert abs(props["EIy"] - EI_solid) / EI_solid < 1e-12
        assert abs(props["GJ"] - GJ_solid) / GJ_solid < 1e-12

    def test_soft_coating_small_contribution(self):
        """柔らかい被膜は剛性寄与が小さい."""
        r_wire = 1.0e-3
        E_wire = 200.0e9
        nu_wire = 0.3
        # 被膜ヤング率が素線の1/1000
        coating = CoatingModel(thickness=0.1e-3, E=0.2e9, nu=0.4)

        props = coated_beam_section(r_wire, E_wire, nu_wire, coating)

        EA_wire = E_wire * math.pi * r_wire**2
        EI_wire = E_wire * math.pi / 4.0 * r_wire**4

        # 被膜寄与は 0.1% 未満
        assert (props["EA"] - EA_wire) / EA_wire < 0.001
        assert (props["EIy"] - EI_wire) / EI_wire < 0.001


# ====================================================================
# coated_contact_radius / coated_radii テスト
# ====================================================================


class TestCoatedRadius:
    """被膜込み接触半径のテスト."""

    def test_coated_contact_radius(self):
        """接触半径 = 素線半径 + 被膜厚さ."""
        r_wire = 1.0e-3
        coating = CoatingModel(thickness=0.1e-3, E=3.0e9, nu=0.35)
        assert coated_contact_radius(r_wire, coating) == r_wire + 0.1e-3

    def test_coated_radii_vector(self):
        """メッシュの全要素に被膜込み半径を適用."""
        mesh = make_twisted_wire_mesh(
            n_strands=3,
            wire_diameter=2.0e-3,
            pitch=40.0e-3,
            length=40.0e-3,
            n_elems_per_strand=4,
        )
        coating = CoatingModel(thickness=0.05e-3, E=3.0e9, nu=0.35)
        radii = coated_radii(mesh, coating)

        assert radii.shape == (mesh.n_elems,)
        expected = mesh.wire_radius + coating.thickness
        np.testing.assert_allclose(radii, expected)

    def test_coated_radii_larger_than_wire(self):
        """被膜込み半径 > 素線半径."""
        mesh = make_twisted_wire_mesh(
            n_strands=7,
            wire_diameter=2.0e-3,
            pitch=40.0e-3,
            length=40.0e-3,
            n_elems_per_strand=4,
        )
        coating = CoatingModel(thickness=0.1e-3, E=3.0e9, nu=0.35)
        r_coated = coated_radii(mesh, coating)
        r_bare = mesh.radii

        assert np.all(r_coated > r_bare)
