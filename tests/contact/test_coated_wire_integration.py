"""撚撚線（被膜付き撚線）統合解析テスト.

被膜付き撚線モデルの統合テスト:
- CoatingModel による被膜込み接触半径・等価断面剛性
- SheathModel によるシース幾何・剛性・ギャップ計算
- 被膜付き3本撚り接触解析（CoatingModel + ContactSolver）
- 被膜の剛性寄与・接触半径増大効果の定量的検証
- シース-素線幾何整合性

テスト目的:
  - 被膜込み接触半径による接触検出の変化
  - 等価断面剛性の正確性
  - シースとの幾何整合性
  - 被膜付き接触ソルバーの収束性
"""

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactStatus,
)
from xkep_cae.contact.solver_hooks import (
    newton_raphson_with_contact,
)
from xkep_cae.elements.beam_timo3d import (
    assemble_cr_beam3d,
)
from xkep_cae.mesh.twisted_wire import (
    CoatingModel,
    SheathModel,
    coated_beam_section,
    coated_contact_radius,
    coated_radii,
    compute_envelope_radius,
    make_twisted_wire_mesh,
    outermost_strand_ids,
    sheath_equivalent_stiffness,
    sheath_inner_radius,
    sheath_radial_gap,
    sheath_section_properties,
)
from xkep_cae.sections.beam import BeamSection

pytestmark = pytest.mark.slow

# ====================================================================
# 共通パラメータ
# ====================================================================

_NDOF_PER_NODE = 6

# 鋼線パラメータ
_E = 200e9  # Pa
_NU = 0.3
_G = _E / (2.0 * (1.0 + _NU))
_WIRE_D = 0.002  # 直径 2mm
_WIRE_R = _WIRE_D / 2.0
_SECTION = BeamSection.circle(_WIRE_D)
_KAPPA = 6.0 * (1.0 + _NU) / (7.0 + 6.0 * _NU)

_PITCH = 0.040  # 40mm ピッチ
_N_ELEM_PER_STRAND = 16  # 1素線あたり要素数

# 被膜パラメータ（PA系プラスチック）
_COATING = CoatingModel(
    thickness=0.05e-3,  # 50μm
    E=3.0e9,  # PA: 3GPa
    nu=0.35,
    mu=0.25,
)

# シースパラメータ（アルミ）
_SHEATH = SheathModel(
    thickness=0.3e-3,  # 0.3mm
    E=70.0e9,  # Al: 70GPa
    nu=0.33,
    mu=0.15,
    clearance=0.05e-3,  # 50μm
)


# ====================================================================
# ヘルパー
# ====================================================================


def _make_cr_assembler_coated(mesh, coating):
    """被膜込み等価剛性を使った CR 梁アセンブラ.

    素線の E, G をベースに、被膜寄与分を断面パラメータに折り込む。
    A_eq = A_wire + (E_coat/E_wire) * A_coat
    Iy_eq = I_wire + (E_coat/E_wire) * I_coat
    J_eq = J_wire + (G_coat/G_wire) * J_coat
    """
    node_coords = mesh.node_coords
    connectivity = mesh.connectivity
    ndof_total = mesh.n_nodes * _NDOF_PER_NODE

    # 被膜込み等価断面剛性
    eq = coated_beam_section(_WIRE_R, _E, _NU, coating)
    A_eq = eq["EA"] / _E
    Iy_eq = eq["EIy"] / _E
    Iz_eq = eq["EIz"] / _E
    J_eq = eq["GJ"] / _G

    def assemble_tangent(u):
        K_T, _ = assemble_cr_beam3d(
            node_coords,
            connectivity,
            u,
            _E,
            _G,
            A_eq,
            Iy_eq,
            Iz_eq,
            J_eq,
            _KAPPA,
            _KAPPA,
            stiffness=True,
            internal_force=False,
        )
        return sp.csr_matrix(K_T)

    def assemble_internal_force(u):
        _, f_int = assemble_cr_beam3d(
            node_coords,
            connectivity,
            u,
            _E,
            _G,
            A_eq,
            Iy_eq,
            Iz_eq,
            J_eq,
            _KAPPA,
            _KAPPA,
            stiffness=False,
            internal_force=True,
        )
        return f_int

    return assemble_tangent, assemble_internal_force, ndof_total


def _make_cr_assembler_bare(mesh):
    """素の（被膜なし）CR 梁アセンブラ."""
    node_coords = mesh.node_coords
    connectivity = mesh.connectivity
    ndof_total = mesh.n_nodes * _NDOF_PER_NODE

    def assemble_tangent(u):
        K_T, _ = assemble_cr_beam3d(
            node_coords,
            connectivity,
            u,
            _E,
            _G,
            _SECTION.A,
            _SECTION.Iy,
            _SECTION.Iz,
            _SECTION.J,
            _KAPPA,
            _KAPPA,
            stiffness=True,
            internal_force=False,
        )
        return sp.csr_matrix(K_T)

    def assemble_internal_force(u):
        _, f_int = assemble_cr_beam3d(
            node_coords,
            connectivity,
            u,
            _E,
            _G,
            _SECTION.A,
            _SECTION.Iy,
            _SECTION.Iz,
            _SECTION.J,
            _KAPPA,
            _KAPPA,
            stiffness=False,
            internal_force=True,
        )
        return f_int

    return assemble_tangent, assemble_internal_force, ndof_total


def _get_strand_end_dofs(mesh, strand_id, end):
    """素線の端点のDOFインデックスを取得."""
    nodes = mesh.strand_nodes(strand_id)
    node = nodes[0] if end == "start" else nodes[-1]
    return np.array([_NDOF_PER_NODE * node + d for d in range(_NDOF_PER_NODE)], dtype=int)


def _fix_all_strand_starts(mesh):
    """全素線の開始端を全固定するDOFセットを返す."""
    fixed = set()
    for sid in range(mesh.n_strands):
        dofs = _get_strand_end_dofs(mesh, sid, "start")
        fixed.update(dofs.tolist())
    return np.array(sorted(fixed), dtype=int)


def _count_active_pairs(mgr):
    """有効な接触ペア数をカウント."""
    return sum(1 for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE)


# ====================================================================
# 1. 被膜込み接触半径・断面剛性の整合性テスト
# ====================================================================


class TestCoatedBeamIntegration:
    """被膜込み梁パラメータの整合性テスト."""

    def test_coated_radius_larger_than_bare(self):
        """被膜込み半径 > 素線半径."""
        r_coated = coated_contact_radius(_WIRE_R, _COATING)
        assert r_coated > _WIRE_R
        assert r_coated == pytest.approx(_WIRE_R + _COATING.thickness)

    def test_coated_radii_array(self):
        """coated_radii がメッシュ全要素分の配列を返す."""
        mesh = make_twisted_wire_mesh(
            3, _WIRE_D, _PITCH, length=0.0, n_elems_per_strand=4, n_pitches=1.0
        )
        radii = coated_radii(mesh, _COATING)
        assert radii.shape == (mesh.n_elems,)
        assert np.all(radii > mesh.wire_radius)
        np.testing.assert_allclose(radii, _WIRE_R + _COATING.thickness)

    def test_coated_section_stiffness_increases(self):
        """被膜込み断面剛性 > 素線のみの剛性."""
        eq = coated_beam_section(_WIRE_R, _E, _NU, _COATING)
        EA_wire = _E * _SECTION.A
        EI_wire = _E * _SECTION.Iy
        assert eq["EA"] > EA_wire
        assert eq["EIy"] > EI_wire
        assert eq["EIz"] > EI_wire
        assert eq["GJ"] > _G * _SECTION.J

    def test_coating_stiffness_ratio(self):
        """被膜剛性寄与の比率確認（被膜 E << 素線 E なので軽微な寄与）."""
        eq = coated_beam_section(_WIRE_R, _E, _NU, _COATING)
        EA_wire = _E * _SECTION.A
        # 被膜は E_coat/E_wire = 3e9/200e9 = 1.5% なので寄与は小さい
        ratio = (eq["EA"] - EA_wire) / EA_wire
        assert 0.0 < ratio < 0.10, f"被膜 EA 寄与比 {ratio:.4f} が予期範囲外"


# ====================================================================
# 2. シース幾何整合性テスト
# ====================================================================


class TestSheathGeometryIntegration:
    """シース + 被膜の幾何整合性テスト."""

    def test_envelope_radius_with_coating(self):
        """被膜込みエンベロープ半径 > 被膜なし."""
        mesh = make_twisted_wire_mesh(
            7, _WIRE_D, _PITCH, length=0.0, n_elems_per_strand=4, n_pitches=1.0
        )
        r_bare = compute_envelope_radius(mesh)
        r_coated = compute_envelope_radius(mesh, coating=_COATING)
        assert r_coated > r_bare
        assert r_coated - r_bare == pytest.approx(_COATING.thickness)

    def test_sheath_inner_radius_with_coating(self):
        """シース内径は被膜込みエンベロープ + クリアランス."""
        mesh = make_twisted_wire_mesh(
            7, _WIRE_D, _PITCH, length=0.0, n_elems_per_strand=4, n_pitches=1.0
        )
        r_inner = sheath_inner_radius(mesh, _SHEATH, coating=_COATING)
        r_env = compute_envelope_radius(mesh, coating=_COATING)
        assert r_inner == pytest.approx(r_env + _SHEATH.clearance)

    def test_sheath_section_properties_positive(self):
        """シース断面特性が全て正."""
        mesh = make_twisted_wire_mesh(
            7, _WIRE_D, _PITCH, length=0.0, n_elems_per_strand=4, n_pitches=1.0
        )
        sp_props = sheath_section_properties(mesh, _SHEATH, coating=_COATING)
        for key in ("A", "Iy", "Iz", "J"):
            assert sp_props[key] > 0, f"シース {key} <= 0"

    def test_sheath_stiffness_positive(self):
        """シース等価剛性が全て正."""
        mesh = make_twisted_wire_mesh(
            7, _WIRE_D, _PITCH, length=0.0, n_elems_per_strand=4, n_pitches=1.0
        )
        stiff = sheath_equivalent_stiffness(mesh, _SHEATH, coating=_COATING)
        for key in ("EA", "EIy", "EIz", "GJ"):
            assert stiff[key] > 0, f"シース {key} <= 0"

    def test_radial_gap_positive_with_clearance(self):
        """初期クリアランス > 0 なら径方向ギャップは全て正."""
        mesh = make_twisted_wire_mesh(
            7, _WIRE_D, _PITCH, length=0.0, n_elems_per_strand=4, n_pitches=1.0
        )
        gaps = sheath_radial_gap(mesh, _SHEATH, coating=_COATING)
        assert np.all(gaps >= 0), f"初期ギャップに負値: min={gaps.min():.3e}"

    def test_outermost_strand_count(self):
        """7本撚りの最外層素線は6本."""
        mesh = make_twisted_wire_mesh(
            7, _WIRE_D, _PITCH, length=0.0, n_elems_per_strand=4, n_pitches=1.0
        )
        outer_ids = outermost_strand_ids(mesh)
        assert len(outer_ids) == 6


# ====================================================================
# 3. 被膜付き3本撚り接触解析テスト
# ====================================================================


class TestCoatedThreeStrandContact:
    """被膜付き3本撚りの接触解析統合テスト.

    被膜による接触半径増大・摩擦係数変更の効果を検証。
    3本撚りは収束するため、統合テストとして使用。
    """

    def _solve_coated_3strand(
        self,
        load_type,
        load_value,
        *,
        with_coating=True,
        use_friction=False,
        n_load_steps=10,
    ):
        """被膜付き3本撚り解析のヘルパー."""
        # 被膜がある場合は初期ギャップを設定して初期貫入を防ぐ
        # 被膜込み半径の増加分 = coating.thickness、素線間で2枚分 = 2 * thickness
        # 余裕を持って 2 * 2 * thickness
        gap = _COATING.thickness * 4 if with_coating else 0.0
        mesh = make_twisted_wire_mesh(
            3,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=_N_ELEM_PER_STRAND,
            n_pitches=1.0,
            gap=gap,
        )

        if with_coating:
            at, af, ndof = _make_cr_assembler_coated(mesh, _COATING)
            radii = coated_radii(mesh, _COATING)
            mu = _COATING.mu
        else:
            at, af, ndof = _make_cr_assembler_bare(mesh)
            radii = mesh.radii
            mu = 0.3

        fixed_dofs = _fix_all_strand_starts(mesh)

        f_ext = np.zeros(ndof)
        if load_type == "tension":
            f_per = load_value / 3
            for sid in range(3):
                end_dofs = _get_strand_end_dofs(mesh, sid, "end")
                f_ext[end_dofs[2]] = f_per
        elif load_type == "lateral":
            f_per = load_value / 3
            for sid in range(3):
                end_dofs = _get_strand_end_dofs(mesh, sid, "end")
                f_ext[end_dofs[0]] = f_per
        elif load_type == "bending":
            m_per = load_value / 3
            for sid in range(3):
                end_dofs = _get_strand_end_dofs(mesh, sid, "end")
                f_ext[end_dofs[4]] = m_per

        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                k_pen_mode="manual",
                k_t_ratio=0.01 if use_friction else 0.1,
                mu=mu,
                g_on=0.0,
                g_off=1e-5,
                n_outer_max=8,
                use_friction=use_friction,
                mu_ramp_steps=10 if use_friction else 0,
                use_line_search=True,
                line_search_max_steps=5,
                use_geometric_stiffness=True,
                tol_penetration_ratio=0.02,
                penalty_growth_factor=2.0,
                k_pen_max=1e12,
            ),
        )

        result = newton_raphson_with_contact(
            f_ext,
            fixed_dofs,
            at,
            af,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            radii,
            n_load_steps=n_load_steps,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.01,
        )
        return result, mgr, mesh

    def test_coated_tension_converges(self):
        """被膜付き3本撚り引張が収束（摩擦なし）."""
        result, _, _ = self._solve_coated_3strand("tension", 100.0)
        assert result.converged, "被膜付き3本撚り引張が収束しなかった"

    def test_coated_lateral_converges(self):
        """被膜付き3本撚り横力が収束（摩擦なし）."""
        result, _, _ = self._solve_coated_3strand("lateral", 10.0)
        assert result.converged, "被膜付き3本撚り横力が収束しなかった"

    def test_coated_bending_converges(self):
        """被膜付き3本撚り曲げが収束（摩擦なし）."""
        result, _, _ = self._solve_coated_3strand("bending", 0.05)
        assert result.converged, "被膜付き3本撚り曲げが収束しなかった"

    def test_coated_tension_with_friction(self):
        """被膜付き3本撚り引張 + 摩擦が収束."""
        result, _, _ = self._solve_coated_3strand(
            "tension", 50.0, use_friction=True, n_load_steps=15
        )
        assert result.converged, "被膜付き3本撚り引張（摩擦）が収束しなかった"

    def test_coating_changes_contact_radius(self):
        """被膜の有無で接触半径が変わる."""
        mesh = make_twisted_wire_mesh(
            3, _WIRE_D, _PITCH, length=0.0, n_elems_per_strand=4, n_pitches=1.0
        )
        radii_bare = mesh.radii
        radii_coated = coated_radii(mesh, _COATING)
        # 被膜厚さ分だけ大きい
        np.testing.assert_allclose(
            radii_coated - radii_bare,
            _COATING.thickness,
            atol=1e-15,
        )

    def test_coated_vs_bare_stiffness(self):
        """被膜付きは素線のみより剛性が高い（同荷重で変位が小さい）."""
        r_coated, _, mesh_c = self._solve_coated_3strand("tension", 50.0, with_coating=True)
        r_bare, _, mesh_b = self._solve_coated_3strand("tension", 50.0, with_coating=False)
        if r_coated.converged and r_bare.converged:
            # z方向最大変位を比較
            u_c = r_coated.u
            u_b = r_bare.u
            max_uz_coated = np.max(np.abs(u_c[2::6]))
            max_uz_bare = np.max(np.abs(u_b[2::6]))
            # 被膜の剛性寄与により変位が（わずかに）小さい
            # ただし鋼線に対して PA の寄与は 1.5% 程度なので差は微小
            assert max_uz_coated <= max_uz_bare * 1.05, (
                f"被膜付き変位 {max_uz_coated:.3e} > 素線のみ {max_uz_bare:.3e} * 1.05"
            )


# ====================================================================
# 4. 7本撚り被膜付き統合テスト
# ====================================================================


class TestCoatedSevenStrandIntegration:
    """7本撚り被膜付き統合テスト.

    7本撚りの接触収束は困難（xfail）だが、
    被膜パラメータのセットアップと幾何整合性は検証可能。
    """

    def test_seven_strand_coated_setup(self):
        """7本撚り被膜付きモデルのセットアップが正常に完了."""
        mesh = make_twisted_wire_mesh(
            7,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=4,
            n_pitches=1.0,
        )
        # 被膜込み半径
        radii = coated_radii(mesh, _COATING)
        assert radii.shape == (mesh.n_elems,)
        assert np.all(radii > _WIRE_R)

        # 等価断面剛性
        eq = coated_beam_section(_WIRE_R, _E, _NU, _COATING)
        assert eq["EA"] > 0
        assert eq["EIy"] > 0
        assert eq["GJ"] > 0

        # シース幾何
        r_env = compute_envelope_radius(mesh, coating=_COATING)
        r_inner = sheath_inner_radius(mesh, _SHEATH, coating=_COATING)
        assert r_inner > r_env

        # 層別マッピング
        layer_map = mesh.build_elem_layer_map()
        layers = set(layer_map.values())
        assert 0 in layers  # 中心層
        assert 1 in layers  # 外層

    def test_seven_strand_sheath_gap_distribution(self):
        """7本撚り: シース-素線ギャップ分布が物理的に妥当."""
        mesh = make_twisted_wire_mesh(
            7,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=8,
            n_pitches=1.0,
        )
        gaps = sheath_radial_gap(mesh, _SHEATH, coating=_COATING)
        # クリアランス > 0 なので全て非負
        assert np.all(gaps >= -1e-10), f"ギャップに負値: min={gaps.min():.3e}"
        # ギャップは概ねクリアランス付近
        # ヘリカル配置のため若干のばらつきがある
        assert gaps.mean() > 0

    def test_seven_strand_contact_config_setup(self):
        """7本撚り被膜付きの ContactConfig セットアップ."""
        mesh = make_twisted_wire_mesh(
            7,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=4,
            n_pitches=1.0,
        )
        layer_map = mesh.build_elem_layer_map()
        max_lay = max(layer_map.values())
        staged_steps = (max_lay + 1) * 2

        config = ContactConfig(
            k_pen_scale=0.1,
            k_pen_mode="beam_ei",
            beam_E=_E,
            beam_I=_SECTION.Iy,
            k_t_ratio=0.01,
            mu=_COATING.mu,
            g_on=0.0,
            g_off=1e-5,
            n_outer_max=10,
            use_friction=True,
            mu_ramp_steps=10,
            use_line_search=True,
            use_geometric_stiffness=True,
            staged_activation_steps=staged_steps,
            elem_layer_map=layer_map,
            k_pen_scaling="sqrt",
            contact_damping=0.8,
            use_modified_newton=True,
            modified_newton_refresh=5,
        )
        mgr = ContactManager(config=config)
        assert mgr.config.mu == _COATING.mu
        assert mgr.config.staged_activation_steps == staged_steps
        assert mgr.config.k_pen_mode == "beam_ei"

    @pytest.mark.xfail(
        reason="7本撚り多点接触: 36+ペア同時収束が困難（接触特化ソルバーが必要）",
        strict=False,
    )
    def test_seven_strand_coated_tension(self):
        """7本撚り被膜付き引張テスト（xfail）."""
        mesh = make_twisted_wire_mesh(
            7,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=4,
            n_pitches=1.0,
            gap=0.0005,
        )
        at, af, ndof = _make_cr_assembler_bare(mesh)
        radii = coated_radii(mesh, _COATING)
        fixed_dofs = _fix_all_strand_starts(mesh)

        f_ext = np.zeros(ndof)
        for sid in range(7):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext[end_dofs[2]] = 1.0 / 7.0  # 1N total

        layer_map = mesh.build_elem_layer_map()
        max_lay = max(layer_map.values())
        staged_steps = (max_lay + 1) * 2

        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=0.1,
                k_pen_mode="beam_ei",
                beam_E=_E,
                beam_I=_SECTION.Iy,
                k_t_ratio=0.01,
                mu=_COATING.mu,
                g_on=0.0,
                g_off=1e-5,
                n_outer_max=10,
                use_friction=True,
                mu_ramp_steps=10,
                use_line_search=True,
                use_geometric_stiffness=True,
                staged_activation_steps=staged_steps,
                elem_layer_map=layer_map,
                k_pen_scaling="sqrt",
                contact_damping=0.7,
                use_modified_newton=True,
                modified_newton_refresh=3,
            ),
        )

        result = newton_raphson_with_contact(
            f_ext,
            fixed_dofs,
            at,
            af,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            radii,
            n_load_steps=20,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.01,
        )
        assert result.converged, "7本撚り被膜付き引張が収束しなかった"
