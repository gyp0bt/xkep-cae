"""接触ペア・状態管理のテスト."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactPair,
    ContactState,
    ContactStatus,
)


# ---------------------------------------------------------------------------
# ContactState
# ---------------------------------------------------------------------------
class TestContactState:
    """ContactState のテスト."""

    def test_default_values(self):
        """デフォルト値の確認."""
        state = ContactState()
        assert state.s == 0.0
        assert state.t == 0.0
        assert state.gap == 0.0
        assert state.lambda_n == 0.0
        assert state.status == ContactStatus.INACTIVE
        assert state.stick is True
        assert state.dissipation == 0.0

    def test_copy_independence(self):
        """copy が深いコピー: 変更が伝播しない."""
        state = ContactState(
            s=0.5,
            t=0.3,
            gap=-0.1,
            normal=np.array([1.0, 0.0, 0.0]),
            lambda_n=100.0,
            status=ContactStatus.ACTIVE,
        )
        copied = state.copy()

        # 値が等しい
        assert copied.s == 0.5
        assert copied.lambda_n == 100.0
        np.testing.assert_array_equal(copied.normal, state.normal)

        # 独立性: コピーを変更しても元に影響しない
        copied.normal[0] = -1.0
        assert state.normal[0] == 1.0

        copied.lambda_n = 999.0
        assert state.lambda_n == 100.0

    def test_z_t_default(self):
        """z_t のデフォルトは零ベクトル (2,)."""
        state = ContactState()
        np.testing.assert_array_equal(state.z_t, np.zeros(2))


# ---------------------------------------------------------------------------
# ContactPair
# ---------------------------------------------------------------------------
class TestContactPair:
    """ContactPair のテスト."""

    def test_search_radius(self):
        """search_radius = radius_a + radius_b."""
        pair = ContactPair(
            elem_a=0,
            elem_b=1,
            nodes_a=np.array([0, 1]),
            nodes_b=np.array([2, 3]),
            radius_a=0.5,
            radius_b=0.3,
        )
        assert pair.search_radius == pytest.approx(0.8)

    def test_is_active_default(self):
        """デフォルトでは非活性."""
        pair = ContactPair(
            elem_a=0,
            elem_b=1,
            nodes_a=np.array([0, 1]),
            nodes_b=np.array([2, 3]),
        )
        assert not pair.is_active()

    def test_is_active_when_active(self):
        """状態が ACTIVE なら活性."""
        pair = ContactPair(
            elem_a=0,
            elem_b=1,
            nodes_a=np.array([0, 1]),
            nodes_b=np.array([2, 3]),
            state=ContactState(status=ContactStatus.ACTIVE),
        )
        assert pair.is_active()


# ---------------------------------------------------------------------------
# ContactConfig
# ---------------------------------------------------------------------------
class TestContactConfig:
    """ContactConfig のテスト."""

    def test_defaults(self):
        """デフォルト設定値の確認."""
        cfg = ContactConfig()
        assert cfg.k_pen_scale == 1.0
        assert cfg.k_t_ratio == 0.5
        assert cfg.mu == pytest.approx(0.3)
        assert cfg.g_on == 0.0
        assert cfg.g_off == pytest.approx(1e-6)
        assert cfg.n_outer_max == 5
        assert cfg.use_friction is False

    def test_custom_values(self):
        """カスタム設定."""
        cfg = ContactConfig(k_pen_scale=2.0, mu=0.5, use_friction=True)
        assert cfg.k_pen_scale == 2.0
        assert cfg.mu == 0.5
        assert cfg.use_friction is True


# ---------------------------------------------------------------------------
# ContactManager
# ---------------------------------------------------------------------------
class TestContactManager:
    """ContactManager のテスト."""

    def test_empty_manager(self):
        """空のマネージャ."""
        mgr = ContactManager()
        assert mgr.n_pairs == 0
        assert mgr.n_active == 0
        assert mgr.get_active_pairs() == []

    def test_add_pair(self):
        """ペア追加."""
        mgr = ContactManager()
        pair = mgr.add_pair(
            elem_a=0,
            elem_b=1,
            nodes_a=np.array([0, 1]),
            nodes_b=np.array([2, 3]),
            radius_a=0.5,
            radius_b=0.5,
        )

        assert mgr.n_pairs == 1
        assert pair.elem_a == 0
        assert pair.elem_b == 1
        assert pair.radius_a == 0.5

    def test_active_count(self):
        """活性ペア数のカウント."""
        mgr = ContactManager()

        # 2ペア追加
        p1 = mgr.add_pair(0, 1, np.array([0, 1]), np.array([2, 3]))
        p2 = mgr.add_pair(2, 3, np.array([4, 5]), np.array([6, 7]))

        assert mgr.n_active == 0

        # p1 を活性化
        p1.state.status = ContactStatus.ACTIVE
        assert mgr.n_active == 1

        # p2 も活性化
        p2.state.status = ContactStatus.SLIDING
        assert mgr.n_active == 2

    def test_get_active_pairs(self):
        """get_active_pairs は活性ペアのみ返す."""
        mgr = ContactManager()
        p1 = mgr.add_pair(0, 1, np.array([0, 1]), np.array([2, 3]))
        p2 = mgr.add_pair(2, 3, np.array([4, 5]), np.array([6, 7]))
        mgr.add_pair(4, 5, np.array([8, 9]), np.array([10, 11]))

        p1.state.status = ContactStatus.ACTIVE
        p2.state.status = ContactStatus.SLIDING

        active = mgr.get_active_pairs()
        assert len(active) == 2
        assert p1 in active
        assert p2 in active

    def test_reset_all(self):
        """reset_all で全ペアが非活性化."""
        mgr = ContactManager()
        p1 = mgr.add_pair(0, 1, np.array([0, 1]), np.array([2, 3]))
        p1.state.status = ContactStatus.ACTIVE
        p1.state.lambda_n = 100.0
        p1.state.gap = -0.5

        mgr.reset_all()

        assert p1.state.status == ContactStatus.INACTIVE
        assert p1.state.lambda_n == 0.0
        assert p1.state.gap == 0.0

    def test_custom_config(self):
        """カスタム設定のマネージャ."""
        cfg = ContactConfig(mu=0.5, use_friction=True)
        mgr = ContactManager(config=cfg)
        assert mgr.config.mu == 0.5
        assert mgr.config.use_friction is True


# ---------------------------------------------------------------------------
# Phase C1: detect_candidates / update_geometry / Active-set
# ---------------------------------------------------------------------------
class TestDetectCandidates:
    """detect_candidates (broadphase候補探索) のテスト."""

    @staticmethod
    def _make_two_beam_mesh():
        """2本の梁が交差する簡単なメッシュ."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, -1.0, 0.5],
                [1.0, 1.0, 0.5],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        return coords, conn

    def test_crossing_beams_detected(self):
        """交差する2梁 → 候補として検出."""
        coords, conn = self._make_two_beam_mesh()
        mgr = ContactManager()
        candidates = mgr.detect_candidates(coords, conn, radii=0.5, margin=0.5)

        assert len(candidates) >= 1
        assert (0, 1) in candidates
        assert mgr.n_pairs >= 1

    def test_distant_beams_not_detected(self):
        """離れた2梁 → 候補なし."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [100.0, 100.0, 100.0],
                [101.0, 100.0, 100.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        mgr = ContactManager()
        candidates = mgr.detect_candidates(coords, conn, radii=0.1)

        assert len(candidates) == 0

    def test_existing_pairs_deactivated(self):
        """候補から外れた既存ペアは INACTIVE になる."""
        coords, conn = self._make_two_beam_mesh()
        mgr = ContactManager()
        mgr.detect_candidates(coords, conn, radii=0.5, margin=1.0)
        assert mgr.n_pairs >= 1
        for p in mgr.pairs:
            p.state.status = ContactStatus.ACTIVE

        coords_far = coords.copy()
        coords_far[2] = [100.0, 100.0, 100.0]
        coords_far[3] = [101.0, 100.0, 100.0]
        mgr.detect_candidates(coords_far, conn, radii=0.5, margin=1.0)

        for p in mgr.pairs:
            if p.elem_a == 0 and p.elem_b == 1:
                assert p.state.status == ContactStatus.INACTIVE

    def test_new_candidates_added(self):
        """新しい候補がペアリストに追加される."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 1.5, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3], [4, 5]])
        mgr = ContactManager()

        candidates = mgr.detect_candidates(coords, conn, radii=0.0, margin=1.0)

        assert (0, 1) in candidates
        pair_keys = {(min(p.elem_a, p.elem_b), max(p.elem_a, p.elem_b)) for p in mgr.pairs}
        assert (0, 1) in pair_keys

    def test_per_element_radii(self):
        """要素ごとの半径でbroadphaseが動作."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [1.0, 5.0, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        radii = np.array([1.0, 4.5])
        mgr = ContactManager()

        candidates = mgr.detect_candidates(coords, conn, radii=radii)
        assert (0, 1) in candidates

        pair = mgr.pairs[0]
        assert pair.radius_a == pytest.approx(1.0)
        assert pair.radius_b == pytest.approx(4.5)


class TestUpdateGeometry:
    """update_geometry (narrowphase + Active-set) のテスト."""

    def test_updates_gap_and_closest_point(self):
        """幾何更新でギャップ・最近接パラメータが設定される."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, -1.0, 0.5],
                [1.0, 1.0, 0.5],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        mgr = ContactManager()
        mgr.add_pair(0, 1, conn[0], conn[1], radius_a=0.0, radius_b=0.0)

        mgr.update_geometry(coords)

        pair = mgr.pairs[0]
        assert 0.0 <= pair.state.s <= 1.0
        assert 0.0 <= pair.state.t <= 1.0
        assert pair.state.gap > 0.0

    def test_contact_frame_orthonormal(self):
        """更新後の接触フレームが正規直交基底."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, -1.0, 0.5],
                [1.0, 1.0, 0.5],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        mgr = ContactManager()
        mgr.add_pair(0, 1, conn[0], conn[1])

        mgr.update_geometry(coords)

        st = mgr.pairs[0].state
        n, t1, t2 = st.normal, st.tangent1, st.tangent2
        np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(t1), 1.0, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(t2), 1.0, atol=1e-14)
        np.testing.assert_allclose(n @ t1, 0.0, atol=1e-14)
        np.testing.assert_allclose(n @ t2, 0.0, atol=1e-14)
        np.testing.assert_allclose(t1 @ t2, 0.0, atol=1e-14)

    def test_gap_with_radius(self):
        """半径を考慮したギャップ計算."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        mgr = ContactManager()
        mgr.add_pair(0, 1, conn[0], conn[1], radius_a=0.5, radius_b=0.5)

        mgr.update_geometry(coords)

        pair = mgr.pairs[0]
        assert pair.state.gap == pytest.approx(1.0)

    def test_penetration_gap(self):
        """貫通時のギャップ（負値）."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        mgr = ContactManager()
        mgr.add_pair(0, 1, conn[0], conn[1], radius_a=0.5, radius_b=0.5)

        mgr.update_geometry(coords)

        pair = mgr.pairs[0]
        assert pair.state.gap == pytest.approx(-0.5)

    def test_frame_continuity(self):
        """法線フレームの連続性（2回の update_geometry）."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 2.0, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        mgr = ContactManager()
        mgr.add_pair(0, 1, conn[0], conn[1])

        mgr.update_geometry(coords)
        t1_first = mgr.pairs[0].state.tangent1.copy()

        coords2 = coords.copy()
        coords2[2] = [0.5, 1.01, 0.01]
        mgr.update_geometry(coords2)
        t1_second = mgr.pairs[0].state.tangent1

        dot = abs(float(t1_first @ t1_second))
        assert dot > 0.9


class TestActiveSetHysteresis:
    """Active-set ヒステリシスのテスト."""

    def test_activate_on_contact(self):
        """gap <= g_on で活性化."""
        mgr = ContactManager(config=ContactConfig(g_on=0.0, g_off=1e-6))
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        mgr.add_pair(0, 1, conn[0], conn[1], radius_a=0.5, radius_b=0.5)

        mgr.update_geometry(coords)

        assert mgr.pairs[0].state.status == ContactStatus.ACTIVE

    def test_stays_active_in_hysteresis_band(self):
        """活性状態で gap が g_on < gap < g_off のとき活性のまま."""
        cfg = ContactConfig(g_on=0.0, g_off=0.1)
        mgr = ContactManager(config=cfg)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.05, 0.0],
                [1.0, 1.05, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        mgr.add_pair(0, 1, conn[0], conn[1], radius_a=0.5, radius_b=0.5)
        mgr.pairs[0].state.status = ContactStatus.ACTIVE

        mgr.update_geometry(coords)

        assert mgr.pairs[0].state.gap == pytest.approx(0.05)
        assert mgr.pairs[0].state.status == ContactStatus.ACTIVE

    def test_deactivate_beyond_g_off(self):
        """gap >= g_off で非活性化."""
        cfg = ContactConfig(g_on=0.0, g_off=0.1)
        mgr = ContactManager(config=cfg)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        mgr.add_pair(0, 1, conn[0], conn[1], radius_a=0.5, radius_b=0.5)
        mgr.pairs[0].state.status = ContactStatus.ACTIVE

        mgr.update_geometry(coords)

        assert mgr.pairs[0].state.status == ContactStatus.INACTIVE

    def test_inactive_stays_inactive_in_band(self):
        """非活性状態で g_on < gap < g_off → 非活性のまま（ヒステリシス）."""
        cfg = ContactConfig(g_on=0.0, g_off=0.1)
        mgr = ContactManager(config=cfg)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.05, 0.0],
                [1.0, 1.05, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        mgr.add_pair(0, 1, conn[0], conn[1], radius_a=0.5, radius_b=0.5)

        mgr.update_geometry(coords)

        assert mgr.pairs[0].state.status == ContactStatus.INACTIVE

    def test_sliding_deactivates_beyond_g_off(self):
        """SLIDING 状態でも gap >= g_off で非活性化."""
        cfg = ContactConfig(g_on=0.0, g_off=0.1)
        mgr = ContactManager(config=cfg)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [1.0, 5.0, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        mgr.add_pair(0, 1, conn[0], conn[1], radius_a=0.5, radius_b=0.5)
        mgr.pairs[0].state.status = ContactStatus.SLIDING

        mgr.update_geometry(coords)

        assert mgr.pairs[0].state.status == ContactStatus.INACTIVE
