"""実梁要素（Timoshenko 3D / CR梁）での接触テスト.

ばねモデルではなく、実際のFEM梁要素を使用した梁梁接触の貫入テスト。
Timoshenko 3D梁（線形）とCR梁（幾何学的非線形）の両方で接触挙動を検証する。

梁パラメータ: アルミ合金 (E=70GPa, ν=0.33), 円形断面 d=20mm, L=0.5m
初期ギャップ: 0.5mm (search_radius=20mm に対して 2.5%)
4セグメント分割で FEM 解の精度を確保

セットアップ:
  梁A: x軸方向 (0,0,0)→(L,0,0), 一端固定（カンチレバー）
  梁B: y軸方向 (L/2,-L/2,h)→(L/2,L/2,h), 一端固定 + 他端にz方向押し下げ
  交差点 (s≈0.5, t≈0.5) 付近で接触発生

テスト項目:
1. Timoshenko 3D線形梁の接触検出・貫入制限
2. CR梁（非線形）の接触検出・貫入制限
3. マルチセグメント（8分割）での接触
4. EI/L³ベース自動k_pen推定
5. Timo3D vs CR梁の小変位一致性
6. 摩擦付き接触
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
    timo_beam3d_ke_global,
)
from xkep_cae.sections.beam import BeamSection

pytestmark = pytest.mark.slow

# ====================================================================
# 共通パラメータ
# ====================================================================

_NDOF_PER_NODE = 6
_TOL_PEN_RATIO = 0.01

# 梁のパラメータ（アルミニウム、円形断面 d=20mm）
_E = 70e9  # Pa
_NU = 0.33
_G = _E / (2.0 * (1.0 + _NU))
_D = 0.02  # 直径 20mm
_RADIUS = _D / 2.0
_SECTION = BeamSection.circle(_D)
_KAPPA = 6.0 * (1.0 + _NU) / (7.0 + 6.0 * _NU)  # Cowper

_L = 0.5  # 梁長さ 0.5m
_DEFAULT_N_SEG = 4  # デフォルト分割数
_DEFAULT_Z_TOP = 2 * _RADIUS + 0.0005  # 初期ギャップ 0.5mm
_SEARCH_RADIUS = 2 * _RADIUS  # r_a + r_b = 0.02m

# 収束実績のある基本パラメータ
_DEFAULT_K_PEN = 1e4
_DEFAULT_F_Z = 500.0  # N
_DEFAULT_N_STEPS = 20


# ====================================================================
# ヘルパー: モデル構築
# ====================================================================


def _make_crossing_beam_model(
    n_seg_a=_DEFAULT_N_SEG,
    n_seg_b=_DEFAULT_N_SEG,
    z_top=_DEFAULT_Z_TOP,
):
    """交差梁モデルの共通メッシュ・拘束を構築.

    Returns:
        node_coords_ref, connectivity, ndof_total, n_nodes_a, n_nodes_b, fixed_dofs
    """
    n_nodes_a = n_seg_a + 1
    n_nodes_b = n_seg_b + 1
    n_nodes = n_nodes_a + n_nodes_b
    ndof_total = n_nodes * _NDOF_PER_NODE

    # 梁A: x軸方向
    coords_a = np.zeros((n_nodes_a, 3))
    for i in range(n_nodes_a):
        coords_a[i, 0] = i * _L / n_seg_a

    # 梁B: y軸方向、z=z_top
    coords_b = np.zeros((n_nodes_b, 3))
    for i in range(n_nodes_b):
        coords_b[i, 0] = _L / 2.0
        coords_b[i, 1] = -_L / 2.0 + i * _L / n_seg_b
        coords_b[i, 2] = z_top

    node_coords_ref = np.vstack([coords_a, coords_b])

    conn_a = np.array([[i, i + 1] for i in range(n_seg_a)])
    conn_b = np.array([[n_nodes_a + i, n_nodes_a + i + 1] for i in range(n_seg_b)])
    connectivity = np.vstack([conn_a, conn_b])

    # 拘束: 各梁の根元を全固定（カンチレバー）
    fixed = set()
    for d in range(_NDOF_PER_NODE):
        fixed.add(d)  # 梁A node 0
    for d in range(_NDOF_PER_NODE):
        fixed.add(n_nodes_a * _NDOF_PER_NODE + d)  # 梁B node 0
    fixed_dofs = np.array(sorted(fixed), dtype=int)

    return (
        node_coords_ref,
        connectivity,
        ndof_total,
        n_nodes_a,
        n_nodes_b,
        fixed_dofs,
    )


def _make_timo3d_assemblers(node_coords_ref, connectivity, ndof_total):
    """Timoshenko 3D 線形梁のアセンブリコールバックを構築."""
    # 剛性行列を事前構築（線形なのでu非依存）
    K = np.zeros((ndof_total, ndof_total))
    for elem in connectivity:
        n1, n2 = int(elem[0]), int(elem[1])
        coords = node_coords_ref[np.array([n1, n2])]
        Ke = timo_beam3d_ke_global(
            coords,
            _E,
            _G,
            _SECTION.A,
            _SECTION.Iy,
            _SECTION.Iz,
            _SECTION.J,
            _KAPPA,
            _KAPPA,
        )
        edofs = np.array(
            [6 * n1 + dd for dd in range(6)] + [6 * n2 + dd for dd in range(6)],
            dtype=int,
        )
        K[np.ix_(edofs, edofs)] += Ke
    K_sp = sp.csr_matrix(K)

    def assemble_tangent(u):
        return K_sp

    def assemble_internal_force(u):
        return K_sp.dot(u)

    return assemble_tangent, assemble_internal_force


def _make_cr_assemblers(node_coords_ref, connectivity, ndof_total):
    """CR梁のアセンブリコールバックを構築."""

    def assemble_tangent(u):
        K_T, _ = assemble_cr_beam3d(
            node_coords_ref,
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
            node_coords_ref,
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

    return assemble_tangent, assemble_internal_force


# ====================================================================
# ヘルパー: EA/L ベース k_pen 推定
# ====================================================================


def _estimate_k_pen_from_beam(E, I_val, L_elem):  # noqa: E741
    """EI/L³ ベースのペナルティ剛性推定.

    梁要素の曲げ剛性 12EI/L³ を基準としてスケーリング。
    梁の接触問題では曲げ剛性が支配的であり、軸剛性 EA/L は過大評価になる。

    Args:
        E: ヤング率
        I: 断面二次モーメント
        L_elem: 要素長さ

    Returns:
        k_pen: 推定ペナルティ剛性
    """
    k_bend = 12.0 * E * I_val / L_elem**3
    return 10.0 * k_bend


# ====================================================================
# ヘルパー: ソルバ実行
# ====================================================================


def _solve_contact(
    assembler_type,
    n_seg_a=_DEFAULT_N_SEG,
    n_seg_b=_DEFAULT_N_SEG,
    f_z=_DEFAULT_F_Z,
    k_pen_scale=_DEFAULT_K_PEN,
    n_load_steps=_DEFAULT_N_STEPS,
    max_iter=50,
    use_friction=False,
    mu=0.3,
    tol_penetration_ratio=_TOL_PEN_RATIO,
    n_outer_max=8,
    z_top=_DEFAULT_Z_TOP,
):
    """実梁要素の接触問題を解く.

    Args:
        assembler_type: "timo3d" or "cr"
    """
    (
        node_coords_ref,
        connectivity,
        ndof_total,
        n_nodes_a,
        n_nodes_b,
        fixed_dofs,
    ) = _make_crossing_beam_model(n_seg_a, n_seg_b, z_top)

    if assembler_type == "timo3d":
        assemble_tangent, assemble_internal_force = _make_timo3d_assemblers(
            node_coords_ref,
            connectivity,
            ndof_total,
        )
    else:
        assemble_tangent, assemble_internal_force = _make_cr_assemblers(
            node_coords_ref,
            connectivity,
            ndof_total,
        )

    # 外力: 梁B端部にz方向押し下げ
    f_ext = np.zeros(ndof_total)
    node_b_last = n_nodes_a + n_nodes_b - 1
    f_ext[node_b_last * _NDOF_PER_NODE + 2] = -f_z

    mgr = ContactManager(
        config=ContactConfig(
            k_pen_scale=k_pen_scale,
            k_t_ratio=0.1,
            mu=mu,
            g_on=0.0,
            g_off=1e-4,
            n_outer_max=n_outer_max,
            use_friction=use_friction,
            mu_ramp_steps=3 if use_friction else 0,
            use_line_search=True,
            line_search_max_steps=5,
            use_geometric_stiffness=True,
            tol_penetration_ratio=tol_penetration_ratio,
            penalty_growth_factor=2.0,
            k_pen_max=1e12,
        ),
    )

    result = newton_raphson_with_contact(
        f_ext,
        fixed_dofs,
        assemble_tangent,
        assemble_internal_force,
        mgr,
        node_coords_ref,
        connectivity,
        _RADIUS,
        n_load_steps=n_load_steps,
        max_iter=max_iter,
        show_progress=False,
        broadphase_margin=0.05,
    )

    return result, mgr, ndof_total, node_coords_ref


# ====================================================================
# テスト: Timoshenko 3D 線形梁 + 接触
# ====================================================================


class TestTimo3DContactDetection:
    """Timoshenko 3D 梁での接触検出テスト."""

    def test_contact_detected(self):
        """z方向押し下げで接触が検出される."""
        result, mgr, _, _ = _solve_contact("timo3d", f_z=500.0)
        assert result.converged
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) > 0, "接触が検出されていない"

    def test_no_contact_without_force(self):
        """押し下げ力なしでは接触しない."""
        result, mgr, _, _ = _solve_contact("timo3d", f_z=0.0)
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) == 0, "力なしで接触が発生した"


class TestTimo3DPenetrationBound:
    """Timoshenko 3D 梁での貫入量制限テスト."""

    def test_penetration_bounded(self):
        """適応的ペナルティにより貫入が制限される."""
        result, mgr, _, _ = _solve_contact("timo3d", f_z=500.0)
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            if pair.state.gap < 0:
                pen_ratio = abs(pair.state.gap) / _SEARCH_RADIUS
                assert pen_ratio < 0.02, f"貫入比 {pen_ratio:.4f} > 2%"

    def test_higher_penalty_reduces_penetration(self):
        """高いペナルティ剛性でより小さい貫入."""
        _, mgr_low, _, _ = _solve_contact("timo3d", f_z=200.0, k_pen_scale=1e4)
        _, mgr_high, _, _ = _solve_contact("timo3d", f_z=200.0, k_pen_scale=1e5)

        def max_penetration(mgr):
            pens = []
            for p in mgr.pairs:
                if p.state.gap < 0:
                    pens.append(abs(p.state.gap))
            return max(pens) if pens else 0.0

        pen_low = max_penetration(mgr_low)
        pen_high = max_penetration(mgr_high)
        assert pen_high <= pen_low + 1e-10, (
            f"高ペナルティで貫入が減少していない: high={pen_high:.6e} vs low={pen_low:.6e}"
        )


class TestTimo3DMultiSegment:
    """Timoshenko 3D マルチセグメント梁での接触テスト."""

    def test_8_segment_contact(self):
        """8分割梁での接触検出と貫入制限."""
        result, mgr, _, _ = _solve_contact(
            "timo3d",
            n_seg_a=8,
            n_seg_b=8,
            f_z=200.0,
        )
        assert result.converged
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) > 0

        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            if pair.state.gap < 0:
                pen_ratio = abs(pair.state.gap) / _SEARCH_RADIUS
                assert pen_ratio < 0.02


# ====================================================================
# テスト: CR梁（幾何学的非線形）+ 接触
# ====================================================================


class TestCRBeamContactDetection:
    """CR梁での接触検出テスト."""

    def test_contact_detected(self):
        """CR梁でz方向押し下げにより接触が検出される."""
        result, mgr, _, _ = _solve_contact("cr", f_z=500.0)
        assert result.converged
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) > 0, "CR梁で接触が検出されていない"

    def test_no_contact_without_force(self):
        """CR梁で押し下げ力なしでは接触しない."""
        result, mgr, _, _ = _solve_contact("cr", f_z=0.0)
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) == 0


class TestCRBeamPenetrationBound:
    """CR梁での貫入量制限テスト."""

    def test_penetration_bounded(self):
        """CR梁で適応的ペナルティにより貫入が制限される."""
        result, mgr, _, _ = _solve_contact("cr", f_z=500.0)
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            if pair.state.gap < 0:
                pen_ratio = abs(pair.state.gap) / _SEARCH_RADIUS
                assert pen_ratio < 0.02, f"CR梁: 貫入比 {pen_ratio:.4f} > 2%"


class TestCRBeamMultiSegment:
    """CR梁マルチセグメント接触テスト."""

    def test_8_segment_contact(self):
        """CR梁8分割での接触検出と貫入制限."""
        result, mgr, _, _ = _solve_contact(
            "cr",
            n_seg_a=8,
            n_seg_b=8,
            f_z=200.0,
        )
        assert result.converged
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) > 0

        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            if pair.state.gap < 0:
                pen_ratio = abs(pair.state.gap) / _SEARCH_RADIUS
                assert pen_ratio < 0.02


# ====================================================================
# テスト: Timo3D vs CR梁の一致性
# ====================================================================


class TestTimo3DVsCRConsistency:
    """Timoshenko 3DとCR梁の小変位時の接触応答一致性."""

    def test_small_load_response_similar(self):
        """小荷重では Timo3D と CR梁の接触応答が類似する."""
        f_z = 200.0
        result_t, _, _, _ = _solve_contact("timo3d", f_z=f_z)
        result_c, _, _, _ = _solve_contact("cr", f_z=f_z)
        assert result_t.converged
        assert result_c.converged

        # 最終変位の比較（z方向の最大変位が近い）
        uz_t = result_t.u[2::_NDOF_PER_NODE]
        uz_c = result_c.u[2::_NDOF_PER_NODE]

        max_uz_t = np.min(uz_t)
        max_uz_c = np.min(uz_c)

        if abs(max_uz_t) > 1e-10:
            rel_diff = abs(max_uz_t - max_uz_c) / abs(max_uz_t)
            assert rel_diff < 0.20, f"Timo3D vs CR: z変位の相対差 {rel_diff:.4f} > 20%"


# ====================================================================
# テスト: EI/L³ベース k_pen 推定
# ====================================================================


class TestAutoKPenEstimation:
    """EI/L³ ベースの自動ペナルティ剛性推定テスト."""

    def test_estimate_k_pen_reasonable(self):
        """推定値が曲げ剛性の妥当な倍数."""
        L_elem = _L / 4
        k_pen = _estimate_k_pen_from_beam(_E, _SECTION.Iy, L_elem)
        k_bend = 12.0 * _E * _SECTION.Iy / L_elem**3
        ratio = k_pen / k_bend
        assert 1.0 < ratio < 10000.0, f"k_pen/k_bend = {ratio:.1f}; 1〜10000 の範囲を期待"

    def test_auto_k_pen_converges(self):
        """自動推定 k_pen で接触が収束する."""
        L_elem = _L / _DEFAULT_N_SEG
        k_pen = _estimate_k_pen_from_beam(_E, _SECTION.Iy, L_elem)
        result, mgr, _, _ = _solve_contact(
            "timo3d",
            f_z=200.0,
            k_pen_scale=k_pen,
        )
        assert result.converged


# ====================================================================
# テスト: 摩擦付き実梁接触
# ====================================================================


class TestRealBeamFrictionContact:
    """実梁要素での摩擦接触テスト."""

    def test_timo3d_friction_converges(self):
        """Timo3D梁の摩擦接触が収束する."""
        result, _, _, _ = _solve_contact(
            "timo3d",
            f_z=200.0,
            use_friction=True,
            mu=0.3,
        )
        assert result.converged

    def test_cr_beam_friction_converges(self):
        """CR梁の摩擦接触が収束する."""
        result, _, _, _ = _solve_contact(
            "cr",
            f_z=200.0,
            use_friction=True,
            mu=0.3,
        )
        assert result.converged

    def test_friction_does_not_worsen_penetration(self):
        """摩擦が法線方向の貫入を悪化させない."""
        _, mgr_nf, _, _ = _solve_contact("timo3d", f_z=200.0, use_friction=False)
        _, mgr_f, _, _ = _solve_contact(
            "timo3d",
            f_z=200.0,
            use_friction=True,
            mu=0.3,
        )

        def max_penetration(mgr):
            pens = []
            for p in mgr.pairs:
                if p.state.status != ContactStatus.INACTIVE and p.state.gap < 0:
                    pens.append(abs(p.state.gap))
            return max(pens) if pens else 0.0

        pen_nf = max_penetration(mgr_nf)
        pen_f = max_penetration(mgr_f)

        if pen_nf > 1e-12:
            assert pen_f < 3.0 * pen_nf + 1e-10, f"摩擦による貫入悪化: {pen_f:.6e} vs {pen_nf:.6e}"


# ====================================================================
# テスト: 長距離スライド（複数セグメントを跨ぐ接触点移動）
# ====================================================================


def _make_offset_crossing_beam_model(
    n_seg_a=8,
    n_seg_b=_DEFAULT_N_SEG,
    x_cross=None,
    z_top=_DEFAULT_Z_TOP,
):
    """交差位置オフセット付きモデル.

    梁Bの交差位置を任意に設定可能。セグメント境界付近に配置して
    接触点のセグメント間移動をテストする。

    Args:
        n_seg_a: 梁Aの分割数
        n_seg_b: 梁Bの分割数
        x_cross: 梁Bが梁Aと交差するx座標 (None → L/2)
        z_top: 梁Bのz座標
    """
    if x_cross is None:
        x_cross = _L / 2.0

    n_nodes_a = n_seg_a + 1
    n_nodes_b = n_seg_b + 1
    n_nodes = n_nodes_a + n_nodes_b
    ndof_total = n_nodes * _NDOF_PER_NODE

    # 梁A: x軸方向 (0,0,0)→(L,0,0)
    coords_a = np.zeros((n_nodes_a, 3))
    for i in range(n_nodes_a):
        coords_a[i, 0] = i * _L / n_seg_a

    # 梁B: y軸方向, x=x_cross, z=z_top
    coords_b = np.zeros((n_nodes_b, 3))
    for i in range(n_nodes_b):
        coords_b[i, 0] = x_cross
        coords_b[i, 1] = -_L / 2.0 + i * _L / n_seg_b
        coords_b[i, 2] = z_top

    node_coords_ref = np.vstack([coords_a, coords_b])

    conn_a = np.array([[i, i + 1] for i in range(n_seg_a)])
    conn_b = np.array([[n_nodes_a + i, n_nodes_a + i + 1] for i in range(n_seg_b)])
    connectivity = np.vstack([conn_a, conn_b])

    # 拘束: 各梁の根元を全固定
    fixed = set()
    for d in range(_NDOF_PER_NODE):
        fixed.add(d)  # 梁A node 0
    for d in range(_NDOF_PER_NODE):
        fixed.add(n_nodes_a * _NDOF_PER_NODE + d)  # 梁B node 0
    fixed_dofs = np.array(sorted(fixed), dtype=int)

    return (
        node_coords_ref,
        connectivity,
        ndof_total,
        n_nodes_a,
        n_nodes_b,
        fixed_dofs,
    )


def _solve_slide_contact(
    assembler_type="timo3d",
    n_seg_a=8,
    n_seg_b=_DEFAULT_N_SEG,
    f_z=200.0,
    f_x=300.0,
    x_cross=None,
    k_pen_scale=_DEFAULT_K_PEN,
    n_load_steps=30,
    use_friction=False,
    mu=0.3,
    z_top=_DEFAULT_Z_TOP,
):
    """横スライド付き実梁接触問題を解く.

    梁Bの自由端に z方向押し下げ + x方向スライド力を付与。
    梁Aは8セグメントで、接触点がセグメント間を移動する。
    """
    (
        node_coords_ref,
        connectivity,
        ndof_total,
        n_nodes_a,
        n_nodes_b,
        fixed_dofs,
    ) = _make_offset_crossing_beam_model(n_seg_a, n_seg_b, x_cross, z_top)

    if assembler_type == "timo3d":
        assemble_tangent, assemble_internal_force = _make_timo3d_assemblers(
            node_coords_ref, connectivity, ndof_total
        )
    else:
        assemble_tangent, assemble_internal_force = _make_cr_assemblers(
            node_coords_ref, connectivity, ndof_total
        )

    # 外力: 梁B端部にz方向押し下げ + x方向スライド
    f_ext = np.zeros(ndof_total)
    node_b_last = n_nodes_a + n_nodes_b - 1
    f_ext[node_b_last * _NDOF_PER_NODE + 0] = f_x  # x方向スライド
    f_ext[node_b_last * _NDOF_PER_NODE + 2] = -f_z  # z方向↓

    mgr = ContactManager(
        config=ContactConfig(
            k_pen_scale=k_pen_scale,
            k_t_ratio=0.1,
            mu=mu,
            g_on=0.0,
            g_off=1e-4,
            n_outer_max=8,
            use_friction=use_friction,
            mu_ramp_steps=3 if use_friction else 0,
            use_line_search=True,
            line_search_max_steps=5,
            use_geometric_stiffness=True,
            tol_penetration_ratio=_TOL_PEN_RATIO,
            penalty_growth_factor=2.0,
            k_pen_max=1e12,
        ),
    )

    result = newton_raphson_with_contact(
        f_ext,
        fixed_dofs,
        assemble_tangent,
        assemble_internal_force,
        mgr,
        node_coords_ref,
        connectivity,
        _RADIUS,
        n_load_steps=n_load_steps,
        max_iter=50,
        show_progress=False,
        broadphase_margin=0.05,
    )

    return result, mgr, ndof_total, n_nodes_a, n_nodes_b


class TestLongRangeSlide:
    """長距離スライド: 接触点が複数セグメントを跨いで移動するテスト.

    8セグメント梁Aの上を梁Bがスライドする。
    セグメント境界付近に初期交差点を配置して、
    接触ペアの遷移を検証する。
    """

    def test_slide_contact_detected(self):
        """8セグメント梁 + 横スライド力で接触検出."""
        result, mgr, _, _, _ = _solve_slide_contact(
            f_z=200.0,
            f_x=100.0,
            n_load_steps=20,
        )
        assert result.converged
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) > 0, "スライド時に接触が検出されなかった"

    def test_slide_penetration_bounded(self):
        """スライド中も貫入量がsearch_radiusの2%以下."""
        result, mgr, _, _, _ = _solve_slide_contact(
            f_z=200.0,
            f_x=100.0,
            n_load_steps=20,
        )
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            if pair.state.gap < 0:
                pen_ratio = abs(pair.state.gap) / _SEARCH_RADIUS
                assert pen_ratio < 0.02, f"スライド貫入超過: {pen_ratio:.4f} > 2%"

    def test_slide_x_displacement_positive(self):
        """スライド力によりx方向変位が生じる."""
        result, _, _, n_nodes_a, n_nodes_b = _solve_slide_contact(
            f_z=200.0,
            f_x=200.0,
            n_load_steps=20,
        )
        assert result.converged

        # 梁B先端のx方向変位
        node_b_last = n_nodes_a + n_nodes_b - 1
        ux_tip = result.u[node_b_last * _NDOF_PER_NODE + 0]
        assert ux_tip > 0.0, f"スライド方向のx変位が非正: {ux_tip:.6e}"

    def test_segment_boundary_crossing(self):
        """セグメント境界付近の交差でも接触が正しく検出される.

        梁Aの8セグメント分割でセグメント長 = L/8 = 62.5mm。
        交差位置をセグメント境界（x=L/8）付近に配置。
        """
        seg_len = _L / 8
        # セグメント1と2の境界付近
        x_cross = seg_len + 0.001  # 境界から1mm先
        result, mgr, _, _, _ = _solve_slide_contact(
            f_z=200.0,
            f_x=0.0,
            x_cross=x_cross,
            n_load_steps=20,
        )
        assert result.converged
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) > 0, "セグメント境界付近で接触が検出されなかった"

    def test_slide_with_friction(self):
        """摩擦ありスライドが収束しペネトレーション制限される."""
        result, mgr, _, _, _ = _solve_slide_contact(
            f_z=200.0,
            f_x=100.0,
            use_friction=True,
            mu=0.3,
            n_load_steps=20,
        )
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            if pair.state.gap < 0:
                pen_ratio = abs(pair.state.gap) / _SEARCH_RADIUS
                assert pen_ratio < 0.02, f"摩擦スライド貫入超過: {pen_ratio:.4f}"

    def test_cr_beam_slide(self):
        """CR梁（非線形）での横スライド接触."""
        result, mgr, _, _, _ = _solve_slide_contact(
            assembler_type="cr",
            f_z=200.0,
            f_x=100.0,
            n_load_steps=20,
        )
        assert result.converged
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) > 0, "CR梁スライドで接触が検出されなかった"
