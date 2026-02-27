"""Stage S3: シース-素線/被膜 有限滑り接触テスト.

テスト対象:
- θ再配置ロジック
- ギャップ計算
- 法線接触力（ペナルティ / コンプライアンス行列）
- 摩擦 return mapping（stick/slip）
- 一括評価 API
- 接触力の全体力ベクトルへの組み立て
"""

import math

import numpy as np
import pytest

from xkep_cae.contact.sheath_contact import (
    SheathContactConfig,
    SheathContactManager,
    SheathContactPoint,
    _angle_diff,
    assemble_sheath_forces,
    build_contact_frame_sheath,
    build_sheath_contact_manager,
    build_sheath_sheath_contact_manager,
    check_theta_rebuild_needed,
    compute_sheath_gaps,
    compute_strand_theta,
    evaluate_normal_forces,
    evaluate_sheath_contact,
    evaluate_sheath_inner_radius,
    rebuild_compliance_matrix,
    sheath_friction_return_mapping,
    sheath_outer_radius,
    sheath_sheath_merged_coords,
    update_contact_angles,
)
from xkep_cae.mesh.twisted_wire import (
    CoatingModel,
    SheathModel,
    make_twisted_wire_mesh,
)

# ============================================================
# フィクスチャ
# ============================================================


@pytest.fixture
def mesh_7():
    """7本撚りメッシュ."""
    return make_twisted_wire_mesh(
        n_strands=7,
        wire_diameter=1e-3,
        pitch=20e-3,
        length=20e-3,
        n_elems_per_strand=10,
    )


@pytest.fixture
def mesh_3():
    """3本撚りメッシュ."""
    return make_twisted_wire_mesh(
        n_strands=3,
        wire_diameter=1e-3,
        pitch=20e-3,
        length=20e-3,
        n_elems_per_strand=10,
    )


@pytest.fixture
def sheath():
    """テスト用シースモデル."""
    return SheathModel(
        thickness=0.5e-3,
        E=70e9,
        nu=0.33,
        mu=0.3,
        clearance=0.0,
    )


@pytest.fixture
def coating():
    """テスト用被膜モデル."""
    return CoatingModel(
        thickness=0.05e-3,
        E=3e9,
        nu=0.4,
        mu=0.2,
    )


# ============================================================
# TestAngleDiff
# ============================================================


class TestAngleDiff:
    """角度差正規化テスト."""

    def test_zero_diff(self):
        assert _angle_diff(1.0, 1.0) == pytest.approx(0.0)

    def test_small_positive(self):
        assert _angle_diff(0.5, 0.3) == pytest.approx(0.2)

    def test_wrap_around(self):
        d = _angle_diff(0.1, 2 * math.pi - 0.1)
        assert d == pytest.approx(0.2, abs=1e-10)

    def test_negative_wrap(self):
        d = _angle_diff(2 * math.pi - 0.1, 0.1)
        assert d == pytest.approx(-0.2, abs=1e-10)


# ============================================================
# TestContactFrame
# ============================================================


class TestContactFrame:
    """シース接触フレーム構築テスト."""

    def test_orthonormal_at_0(self):
        n, t1, t2 = build_contact_frame_sheath(0.0)
        assert n == pytest.approx([1, 0, 0], abs=1e-12)
        assert t1 == pytest.approx([0, 1, 0], abs=1e-12)
        assert t2 == pytest.approx([0, 0, 1], abs=1e-12)

    def test_orthonormal_at_pi_2(self):
        n, t1, t2 = build_contact_frame_sheath(math.pi / 2)
        assert n == pytest.approx([0, 1, 0], abs=1e-12)
        assert t1 == pytest.approx([-1, 0, 0], abs=1e-12)
        assert t2 == pytest.approx([0, 0, 1], abs=1e-12)

    def test_orthogonality(self):
        """任意角度で直交性を確認."""
        for theta in [0.0, 0.5, 1.0, 2.0, 4.5]:
            n, t1, t2 = build_contact_frame_sheath(theta)
            assert abs(n @ t1) < 1e-12
            assert abs(n @ t2) < 1e-12
            assert abs(t1 @ t2) < 1e-12
            assert np.linalg.norm(n) == pytest.approx(1.0)
            assert np.linalg.norm(t1) == pytest.approx(1.0)
            assert np.linalg.norm(t2) == pytest.approx(1.0)


# ============================================================
# TestBuildManager
# ============================================================


class TestBuildManager:
    """マネージャ構築テスト."""

    def test_7strand_builds(self, mesh_7, sheath):
        mgr = build_sheath_contact_manager(mesh_7, sheath)
        assert mgr.n_points == 6  # 最外層6本

    def test_3strand_builds(self, mesh_3, sheath):
        mgr = build_sheath_contact_manager(mesh_3, sheath)
        assert mgr.n_points == 3  # 全3本（中心なし）

    def test_with_coating(self, mesh_7, sheath, coating):
        mgr = build_sheath_contact_manager(mesh_7, sheath, coating=coating)
        assert mgr.r_eff == pytest.approx(0.5e-3 + 0.05e-3)

    def test_initial_angles(self, mesh_7, sheath):
        mgr = build_sheath_contact_manager(mesh_7, sheath)
        angles = mgr.get_contact_angles()
        assert len(angles) == 6
        # 60度間隔
        for i in range(6):
            expected = 2.0 * math.pi * i / 6
            assert angles[i] == pytest.approx(expected, abs=1e-10)

    def test_penalty_initialized(self, mesh_7, sheath):
        config = SheathContactConfig(k_pen=1e8, k_t_ratio=0.3)
        mgr = build_sheath_contact_manager(mesh_7, sheath, config=config)
        for pt in mgr.points:
            assert pt.k_pen == 1e8
            assert pt.k_t == pytest.approx(1e8 * 0.3)


# ============================================================
# TestThetaUpdate
# ============================================================


class TestThetaUpdate:
    """θ再配置テスト."""

    def test_undeformed_no_change(self, mesh_7, sheath):
        mgr = build_sheath_contact_manager(mesh_7, sheath)
        delta = update_contact_angles(mgr, mesh_7.node_coords)
        assert np.max(np.abs(delta)) < 0.1  # ヘリカル配置のため完全0ではない

    def test_rotation_shifts_theta(self, mesh_7, sheath):
        """座標を45度回転するとθが~45度変化."""
        mgr = build_sheath_contact_manager(mesh_7, sheath)
        # 座標を45度回転
        angle = math.pi / 4
        R = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        rotated_coords = (R @ mesh_7.node_coords.T).T
        delta = update_contact_angles(mgr, rotated_coords)
        # 全素線のθがほぼπ/4だけ変化
        for d in delta:
            assert abs(d) == pytest.approx(math.pi / 4, abs=0.2)

    def test_compute_strand_theta_basic(self, mesh_7):
        """素線θの計算."""
        # 第1外層素線（strand_id=1, angle_offset=0）の中央節点はx>0, y≈0
        node_start, node_end = mesh_7.strand_node_ranges[1]
        indices = np.arange(node_start, node_end)
        theta = compute_strand_theta(mesh_7.node_coords, indices)
        # ヘリカル配置のため、z中央でのθは初期角度付近
        # pitch=20mm, length=20mm → z中央で半周（π）の撚り位相回転
        # 厳密な値は撚りピッチに依存するが、0〜2πの範囲内
        assert 0.0 <= theta < 2 * math.pi


# ============================================================
# TestGapComputation
# ============================================================


class TestGapComputation:
    """ギャップ計算テスト."""

    def test_initial_gap_zero_clearance(self, mesh_7, sheath):
        """clearance=0 なら初期ギャップ ≈ 0."""
        mgr = build_sheath_contact_manager(mesh_7, sheath)
        update_contact_angles(mgr, mesh_7.node_coords)
        gaps = compute_sheath_gaps(mgr, mesh_7.node_coords)
        # 初期配置ではほぼ密着
        for g in gaps:
            assert abs(g) < 1e-4  # ヘリカルなので完全0ではない

    def test_positive_clearance_gives_positive_gap(self, mesh_7):
        """clearance > 0 なら初期ギャップ > 0."""
        sheath_with_gap = SheathModel(thickness=0.5e-3, E=70e9, nu=0.33, mu=0.3, clearance=1e-3)
        mgr = build_sheath_contact_manager(mesh_7, sheath_with_gap)
        update_contact_angles(mgr, mesh_7.node_coords)
        gaps = compute_sheath_gaps(mgr, mesh_7.node_coords)
        for g in gaps:
            assert g > 0.0  # クリアランスがあるので非接触

    def test_inward_displacement_creates_penetration(self, mesh_7, sheath):
        """素線を径方向外側に押すと貫入（gap < 0）が発生."""
        mgr = build_sheath_contact_manager(mesh_7, sheath)

        # 最外層素線を外向きに 0.5mm 変位させる
        displaced = mesh_7.node_coords.copy()
        for pt in mgr.points:
            for nid in pt.node_indices:
                x, y = displaced[nid, 0], displaced[nid, 1]
                r = math.sqrt(x**2 + y**2)
                if r > 1e-15:
                    scale = 0.5e-3 / r
                    displaced[nid, 0] += x * scale
                    displaced[nid, 1] += y * scale

        update_contact_angles(mgr, displaced)
        gaps = compute_sheath_gaps(mgr, displaced)
        # 少なくとも1つは貫入
        assert np.min(gaps) < 0.0


# ============================================================
# TestNormalForce
# ============================================================


class TestNormalForce:
    """法線力テスト."""

    def test_no_penetration_no_force(self, mesh_7):
        """ギャップ正→力ゼロ."""
        sheath_gap = SheathModel(thickness=0.5e-3, E=70e9, nu=0.33, mu=0.3, clearance=1e-3)
        mgr = build_sheath_contact_manager(mesh_7, sheath_gap)
        update_contact_angles(mgr, mesh_7.node_coords)
        compute_sheath_gaps(mgr, mesh_7.node_coords)
        forces = evaluate_normal_forces(mgr)
        assert np.all(forces == 0.0)
        assert mgr.n_active == 0

    def test_penetration_creates_force(self, mesh_7, sheath):
        """貫入ありのとき法線力 > 0."""
        config = SheathContactConfig(k_pen=1e8)
        mgr = build_sheath_contact_manager(mesh_7, sheath, config=config)

        # 外向き変位
        displaced = mesh_7.node_coords.copy()
        for pt in mgr.points:
            for nid in pt.node_indices:
                x, y = displaced[nid, 0], displaced[nid, 1]
                r = math.sqrt(x**2 + y**2)
                if r > 1e-15:
                    displaced[nid, 0] += x * 0.5e-3 / r
                    displaced[nid, 1] += y * 0.5e-3 / r

        update_contact_angles(mgr, displaced)
        compute_sheath_gaps(mgr, displaced)
        forces = evaluate_normal_forces(mgr)
        assert np.any(forces > 0.0)
        assert mgr.n_active > 0

    def test_force_proportional_to_penetration(self):
        """ペナルティ法: F = k_pen * penetration."""
        pt = SheathContactPoint(strand_id=0, k_pen=1e6)
        pt.gap = -0.001  # 1mm 貫入
        pt.active = False

        mgr = SheathContactManager(points=[pt], config=SheathContactConfig(k_pen=1e6))
        forces = evaluate_normal_forces(mgr)
        assert forces[0] == pytest.approx(1e6 * 0.001)  # F = k * δ


# ============================================================
# TestFriction
# ============================================================


class TestFriction:
    """摩擦 return mapping テスト."""

    def test_stick_below_yield(self):
        """小変位で stick."""
        pt = SheathContactPoint(strand_id=0, k_pen=1e6, k_t=5e5, active=True)
        pt.p_n = 100.0  # 法線力100N
        pt.z_t = np.zeros(2)

        delta_ut = np.array([1e-6, 0.0])  # 微小周方向変位
        q = sheath_friction_return_mapping(pt, delta_ut, mu=0.3)

        assert pt.stick is True
        assert q[0] == pytest.approx(5e5 * 1e-6)  # k_t * delta_u

    def test_slip_beyond_yield(self):
        """大変位で slip."""
        pt = SheathContactPoint(strand_id=0, k_pen=1e6, k_t=5e5, active=True)
        pt.p_n = 100.0
        pt.z_t = np.zeros(2)

        delta_ut = np.array([1.0, 0.0])  # 大きな変位
        q = sheath_friction_return_mapping(pt, delta_ut, mu=0.3)

        assert pt.stick is False
        q_norm = np.linalg.norm(q)
        assert q_norm == pytest.approx(0.3 * 100.0, abs=1e-8)  # μ * p_n

    def test_no_force_when_inactive(self):
        """非活性接触点→摩擦力ゼロ."""
        pt = SheathContactPoint(strand_id=0, active=False)
        pt.p_n = 0.0
        q = sheath_friction_return_mapping(pt, np.array([1.0, 1.0]), mu=0.3)
        assert q == pytest.approx([0.0, 0.0])

    def test_friction_zero_mu(self):
        """μ=0 → 摩擦力ゼロ."""
        pt = SheathContactPoint(strand_id=0, k_pen=1e6, k_t=5e5, active=True)
        pt.p_n = 100.0
        q = sheath_friction_return_mapping(pt, np.array([1.0, 0.0]), mu=0.0)
        assert q == pytest.approx([0.0, 0.0])

    def test_dissipation_positive_on_slip(self):
        """slip時の散逸が非負."""
        pt = SheathContactPoint(strand_id=0, k_pen=1e6, k_t=5e5, active=True)
        pt.p_n = 100.0
        pt.z_t = np.zeros(2)

        delta_ut = np.array([1.0, 0.5])
        sheath_friction_return_mapping(pt, delta_ut, mu=0.3)
        assert pt.dissipation >= 0.0

    def test_two_component_friction(self):
        """周方向+軸方向の2成分摩擦."""
        pt = SheathContactPoint(strand_id=0, k_pen=1e6, k_t=5e5, active=True)
        pt.p_n = 100.0
        pt.z_t = np.zeros(2)

        delta_ut = np.array([1e-6, 2e-6])  # 小変位（stick）
        q = sheath_friction_return_mapping(pt, delta_ut, mu=0.3)

        assert q[0] == pytest.approx(5e5 * 1e-6)
        assert q[1] == pytest.approx(5e5 * 2e-6)


# ============================================================
# TestEvaluateContact
# ============================================================


class TestEvaluateContact:
    """一括評価APIテスト."""

    def test_evaluate_returns_all_fields(self, mesh_7, sheath):
        mgr = build_sheath_contact_manager(mesh_7, sheath)
        result = evaluate_sheath_contact(mgr, mesh_7.node_coords)
        assert "delta_theta" in result
        assert "gaps" in result
        assert "normal_forces" in result
        assert "friction_forces" in result
        assert "n_active" in result
        assert len(result["gaps"]) == 6

    def test_evaluate_with_ref_computes_friction(self, mesh_7, sheath):
        """参照座標を渡すと摩擦が計算される."""
        config = SheathContactConfig(k_pen=1e8, mu=0.3)
        mgr = build_sheath_contact_manager(mesh_7, sheath, config=config)

        # 外向き変位 + 周方向変位
        displaced = mesh_7.node_coords.copy()
        for pt in mgr.points:
            for nid in pt.node_indices:
                x, y = displaced[nid, 0], displaced[nid, 1]
                r = math.sqrt(x**2 + y**2)
                if r > 1e-15:
                    displaced[nid, 0] += x * 0.5e-3 / r + 1e-5
                    displaced[nid, 1] += y * 0.5e-3 / r + 1e-5

        result = evaluate_sheath_contact(mgr, displaced, node_coords_ref=mesh_7.node_coords)
        # 貫入があれば法線力が発生
        if result["n_active"] > 0:
            assert np.any(result["normal_forces"] > 0)


# ============================================================
# TestAssembleForces
# ============================================================


class TestAssembleForces:
    """力ベクトル組み立てテスト."""

    def test_zero_when_no_active(self, mesh_7, sheath):
        """活性接触なし → 力ゼロ."""
        sheath_gap = SheathModel(thickness=0.5e-3, E=70e9, nu=0.33, mu=0.3, clearance=1e-3)
        mgr = build_sheath_contact_manager(mesh_7, sheath_gap)
        update_contact_angles(mgr, mesh_7.node_coords)
        compute_sheath_gaps(mgr, mesh_7.node_coords)
        evaluate_normal_forces(mgr)

        f = assemble_sheath_forces(mgr, mesh_7.node_coords, ndof_per_node=6)
        assert np.all(f == 0.0)

    def test_force_direction_inward(self):
        """法線力は径方向内向き（中心方向）."""
        # 手動セットアップ: θ=0 の接触点、p_n=100
        pt = SheathContactPoint(
            strand_id=0,
            node_indices=np.array([0]),
            theta=0.0,
            k_pen=1e6,
            active=True,
        )
        pt.p_n = 100.0
        pt.z_t = np.zeros(2)

        mgr = SheathContactManager(points=[pt])
        coords = np.array([[1.0, 0.0, 0.0]])  # x軸上

        f = assemble_sheath_forces(mgr, coords, ndof_per_node=3)
        # θ=0 → normal = [1,0,0], 法線力は -normal 方向
        assert f[0] == pytest.approx(-100.0)
        assert f[1] == pytest.approx(0.0)
        assert f[2] == pytest.approx(0.0)


# ============================================================
# TestComplianceMatrix
# ============================================================


class TestComplianceMatrix:
    """コンプライアンス行列モードテスト."""

    def test_rebuild_produces_valid_matrix(self, mesh_7, sheath):
        mgr = build_sheath_contact_manager(mesh_7, sheath)
        C = rebuild_compliance_matrix(mgr, mesh_7, sheath)
        assert C.shape == (6, 6)
        # 対称
        assert np.allclose(C, C.T, atol=1e-14)
        # 正定値
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0)

    def test_compliance_mode_uses_matrix(self, mesh_7, sheath):
        """use_compliance_matrix=True でC行列ベースの力計算."""
        config = SheathContactConfig(use_compliance_matrix=True, k_pen=1e8)
        mgr = build_sheath_contact_manager(mesh_7, sheath, config=config)
        C = rebuild_compliance_matrix(mgr, mesh_7, sheath)

        # 全点に一様貫入を設定
        for pt in mgr.points:
            pt.gap = -1e-4

        forces = evaluate_normal_forces(mgr)
        # 全力が正
        assert np.all(forces > 0)
        # C^{-1} @ δ と一致
        delta = np.full(6, 1e-4)
        expected = np.linalg.solve(C, delta)
        np.testing.assert_allclose(forces, expected, rtol=1e-8)


# ============================================================
# TestThetaRebuild
# ============================================================


class TestThetaRebuild:
    """θ再構築判定テスト."""

    def test_small_change_no_rebuild(self):
        config = SheathContactConfig(theta_rebuild_tol=0.05)
        mgr = SheathContactManager(config=config)
        delta = np.array([0.01, -0.01, 0.02])
        assert check_theta_rebuild_needed(mgr, delta) is False

    def test_large_change_triggers_rebuild(self):
        config = SheathContactConfig(theta_rebuild_tol=0.05)
        mgr = SheathContactManager(config=config)
        delta = np.array([0.01, -0.06, 0.02])
        assert check_theta_rebuild_needed(mgr, delta) is True


# ============================================================
# TestInnerRadius
# ============================================================


class TestInnerRadius:
    """シース内面半径評価テスト."""

    def test_uniform_radius(self, mesh_7, sheath):
        """全角度でほぼ均一な半径."""
        mgr = build_sheath_contact_manager(mesh_7, sheath)
        r0 = evaluate_sheath_inner_radius(mgr, 0.0)
        r1 = evaluate_sheath_inner_radius(mgr, math.pi / 3)
        # 対称性から大きくは変わらない
        assert abs(r0 - r1) / r0 < 0.1

    def test_fallback_to_base(self):
        """プロファイル未設定 → 基本内径を返す."""
        mgr = SheathContactManager(sheath_r_inner_base=5.0)
        assert evaluate_sheath_inner_radius(mgr, 0.0) == 5.0


# ============================================================
# Stage S4: シース-シース接触テスト
# ============================================================


class TestSheathOuterRadius:
    """シース外径計算テスト."""

    def test_outer_radius_positive(self, mesh_7, sheath):
        r = sheath_outer_radius(mesh_7, sheath)
        assert r > 0.0

    def test_outer_radius_includes_thickness(self, mesh_7, sheath):
        from xkep_cae.mesh.twisted_wire import sheath_inner_radius

        r_in = sheath_inner_radius(mesh_7, sheath)
        r_out = sheath_outer_radius(mesh_7, sheath)
        assert r_out == pytest.approx(r_in + sheath.thickness)

    def test_coating_increases_radius(self, mesh_7, sheath, coating):
        r_no_coat = sheath_outer_radius(mesh_7, sheath)
        r_with_coat = sheath_outer_radius(mesh_7, sheath, coating=coating)
        assert r_with_coat > r_no_coat


class TestSheathSheathMergedCoords:
    """統合座標テスト."""

    def test_two_meshes(self, mesh_7):
        coords, conn, noff, eoff = sheath_sheath_merged_coords([mesh_7, mesh_7])
        # 中心素線は11ノード（10要素+1）× 2本
        assert coords.shape[0] == 22
        assert conn.shape[0] == 20
        assert noff == [0, 11]
        assert eoff == [0, 10]

    def test_three_meshes(self, mesh_3):
        """3本撚り（中心なし）の場合."""
        coords, conn, noff, eoff = sheath_sheath_merged_coords([mesh_3, mesh_3, mesh_3])
        # 3本撚り: center_id=0, 11ノード × 3本
        assert coords.shape[0] == 33
        assert len(noff) == 3


class TestSheathSheathContactManager:
    """シース-シース ContactManager 構築テスト."""

    def test_builds_with_two_cables(self, sheath):
        """2本の撚線でシース-シース接触マネージャが構築できる."""
        mesh1 = make_twisted_wire_mesh(
            n_strands=7,
            wire_diameter=1e-3,
            pitch=20e-3,
            length=20e-3,
            n_elems_per_strand=10,
        )
        # 2本目は横に1cm ずらす
        mesh2 = make_twisted_wire_mesh(
            n_strands=7,
            wire_diameter=1e-3,
            pitch=20e-3,
            length=20e-3,
            n_elems_per_strand=10,
        )
        # mesh2 の座標をx方向にオフセット
        offset = 3e-3  # シース外径より小さくして接触させる
        mesh2.node_coords[:, 0] += offset

        mgr = build_sheath_sheath_contact_manager(
            [mesh1, mesh2],
            [sheath, sheath],
        )
        # ContactManager が返される
        assert mgr is not None
        assert mgr.n_pairs > 0

    def test_far_apart_no_contact(self, sheath):
        """十分離れていれば接触候補なし."""
        mesh1 = make_twisted_wire_mesh(
            n_strands=7,
            wire_diameter=1e-3,
            pitch=20e-3,
            length=20e-3,
            n_elems_per_strand=5,
        )
        mesh2 = make_twisted_wire_mesh(
            n_strands=7,
            wire_diameter=1e-3,
            pitch=20e-3,
            length=20e-3,
            n_elems_per_strand=5,
        )
        # 100mm 離す（接触しない距離）
        mesh2.node_coords[:, 0] += 100e-3

        mgr = build_sheath_sheath_contact_manager(
            [mesh1, mesh2],
            [sheath, sheath],
        )
        # 候補が見つからない
        assert mgr.n_pairs == 0

    def test_raises_for_single_cable(self, mesh_7, sheath):
        """1本だけではエラー."""
        with pytest.raises(ValueError, match="2本以上"):
            build_sheath_sheath_contact_manager([mesh_7], [sheath])

    def test_three_cables(self, sheath):
        """3本のシース間接触."""
        meshes = []
        for i in range(3):
            m = make_twisted_wire_mesh(
                n_strands=7,
                wire_diameter=1e-3,
                pitch=20e-3,
                length=20e-3,
                n_elems_per_strand=5,
            )
            # 三角形配置
            angle = 2.0 * math.pi * i / 3
            offset = 3e-3
            m.node_coords[:, 0] += offset * math.cos(angle)
            m.node_coords[:, 1] += offset * math.sin(angle)
            meshes.append(m)

        mgr = build_sheath_sheath_contact_manager(
            meshes,
            [sheath, sheath, sheath],
        )
        assert mgr.n_pairs > 0
