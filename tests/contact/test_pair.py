"""接触ペア・状態管理のテスト."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.contact._contact_pair import (
    _ContactConfigInput,
    _ContactManagerInput,
    _ContactPairOutput,
    _ContactStateOutput,
    _copy_state,
    _evolve_pair,
    _evolve_state,
    _get_active_pairs,
    _is_active_pair,
    _n_active,
    _n_pairs,
    _pair_search_radius,
)
from xkep_cae.contact._manager_process import (
    AddPairInput,
    AddPairProcess,
    DetectCandidatesInput,
    DetectCandidatesProcess,
    ResetAllPairsInput,
    ResetAllPairsProcess,
    UpdateGeometryInput,
    UpdateGeometryProcess,
)
from xkep_cae.contact._types import ContactStatus


# ---------------------------------------------------------------------------
# _ContactStateOutput
# ---------------------------------------------------------------------------
class TestContactState:
    """_ContactStateOutput のテスト."""

    def test_default_values(self):
        """デフォルト値の確認."""
        state = _ContactStateOutput()
        assert state.s == 0.0
        assert state.t == 0.0
        assert state.gap == 0.0
        assert state.lambda_n == 0.0
        assert state.status == ContactStatus.INACTIVE
        assert state.stick is True
        assert state.dissipation == 0.0

    def test_copy_independence(self):
        """_copy_state が深いコピー: 変更が伝播しない."""
        state = _ContactStateOutput(
            s=0.5,
            t=0.3,
            gap=-0.1,
            normal=np.array([1.0, 0.0, 0.0]),
            lambda_n=100.0,
            status=ContactStatus.ACTIVE,
        )
        copied = _copy_state(state)

        # 値が等しい
        assert copied.s == 0.5
        assert copied.lambda_n == 100.0
        np.testing.assert_array_equal(copied.normal, state.normal)

        # 独立性: コピーの配列を変更しても元に影響しない
        copied.normal[0] = -1.0
        assert state.normal[0] == 1.0

        # frozen なので scalar フィールドは変更不可（独立性は保証される）

    def test_z_t_default(self):
        """z_t のデフォルトは零ベクトル (2,)."""
        state = _ContactStateOutput()
        np.testing.assert_array_equal(state.z_t, np.zeros(2))


# ---------------------------------------------------------------------------
# _ContactPairOutput
# ---------------------------------------------------------------------------
class TestContactPair:
    """_ContactPairOutput のテスト."""

    def test_search_radius(self):
        """_pair_search_radius = radius_a + radius_b."""
        pair = _ContactPairOutput(
            elem_a=0,
            elem_b=1,
            nodes_a=np.array([0, 1]),
            nodes_b=np.array([2, 3]),
            radius_a=0.5,
            radius_b=0.3,
        )
        assert _pair_search_radius(pair) == pytest.approx(0.8)

    def test_is_active_default(self):
        """デフォルトでは非活性."""
        pair = _ContactPairOutput(
            elem_a=0,
            elem_b=1,
            nodes_a=np.array([0, 1]),
            nodes_b=np.array([2, 3]),
        )
        assert not _is_active_pair(pair)

    def test_is_active_when_active(self):
        """状態が ACTIVE なら活性."""
        pair = _ContactPairOutput(
            elem_a=0,
            elem_b=1,
            nodes_a=np.array([0, 1]),
            nodes_b=np.array([2, 3]),
            state=_ContactStateOutput(status=ContactStatus.ACTIVE),
        )
        assert _is_active_pair(pair)


# ---------------------------------------------------------------------------
# _ContactConfigInput
# ---------------------------------------------------------------------------
class TestContactConfig:
    """_ContactConfigInput のテスト."""

    def test_defaults(self):
        """デフォルト設定値の確認."""
        cfg = _ContactConfigInput()
        assert cfg.k_pen_scale == 0.1
        assert cfg.k_t_ratio == 0.5
        assert cfg.mu == pytest.approx(0.3)
        assert cfg.g_on == 0.0
        assert cfg.g_off == pytest.approx(1e-6)
        assert cfg.n_outer_max == 5
        assert cfg.use_friction is False

    def test_custom_values(self):
        """カスタム設定."""
        cfg = _ContactConfigInput(k_pen_scale=2.0, mu=0.5, use_friction=True)
        assert cfg.k_pen_scale == 2.0
        assert cfg.mu == 0.5
        assert cfg.use_friction is True


# ---------------------------------------------------------------------------
# _ContactManagerInput
# ---------------------------------------------------------------------------
class TestContactManager:
    """_ContactManagerInput のテスト."""

    def test_empty_manager(self):
        """空のマネージャ."""
        mgr = _ContactManagerInput()
        assert _n_pairs(mgr) == 0
        assert _n_active(mgr) == 0
        assert _get_active_pairs(mgr) == []

    def test_add_pair(self):
        """ペア追加."""
        mgr = _ContactManagerInput()
        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=0,
                elem_b=1,
                nodes_a=np.array([0, 1]),
                nodes_b=np.array([2, 3]),
                radius_a=0.5,
                radius_b=0.5,
            )
        )
        mgr = _ap_out.manager
        pair = _ap_out.pair

        assert _n_pairs(mgr) == 1
        assert pair.elem_a == 0
        assert pair.elem_b == 1
        assert pair.radius_a == 0.5

    def test_active_count(self):
        """活性ペア数のカウント."""
        mgr = _ContactManagerInput()

        # 2ペア追加
        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=0,
                elem_b=1,
                nodes_a=np.array([0, 1]),
                nodes_b=np.array([2, 3]),
            )
        )
        mgr = _ap_out.manager
        p1 = _ap_out.pair

        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=2,
                elem_b=3,
                nodes_a=np.array([4, 5]),
                nodes_b=np.array([6, 7]),
            )
        )
        mgr = _ap_out.manager
        p2 = _ap_out.pair

        assert _n_active(mgr) == 0

        # p1 を活性化
        p1 = _evolve_pair(p1, state=_evolve_state(p1.state, status=ContactStatus.ACTIVE))
        mgr = _ContactManagerInput(pairs=[p1, p2], config=mgr.config)
        assert _n_active(mgr) == 1

        # p2 も活性化
        p2 = _evolve_pair(p2, state=_evolve_state(p2.state, status=ContactStatus.SLIDING))
        mgr = _ContactManagerInput(pairs=[p1, p2], config=mgr.config)
        assert _n_active(mgr) == 2

    def test_get_active_pairs(self):
        """_get_active_pairs は活性ペアのみ返す."""
        mgr = _ContactManagerInput()
        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=0,
                elem_b=1,
                nodes_a=np.array([0, 1]),
                nodes_b=np.array([2, 3]),
            )
        )
        mgr = _ap_out.manager
        p1 = _ap_out.pair

        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=2,
                elem_b=3,
                nodes_a=np.array([4, 5]),
                nodes_b=np.array([6, 7]),
            )
        )
        mgr = _ap_out.manager
        p2 = _ap_out.pair

        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=4,
                elem_b=5,
                nodes_a=np.array([8, 9]),
                nodes_b=np.array([10, 11]),
            )
        )
        mgr = _ap_out.manager
        p3 = _ap_out.pair

        p1 = _evolve_pair(p1, state=_evolve_state(p1.state, status=ContactStatus.ACTIVE))
        p2 = _evolve_pair(p2, state=_evolve_state(p2.state, status=ContactStatus.SLIDING))
        mgr = _ContactManagerInput(pairs=[p1, p2, p3], config=mgr.config)

        active = _get_active_pairs(mgr)
        assert len(active) == 2
        assert p1 in active
        assert p2 in active

    def test_reset_all(self):
        """ResetAllPairsProcess で全ペアが非活性化."""
        mgr = _ContactManagerInput()
        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=0,
                elem_b=1,
                nodes_a=np.array([0, 1]),
                nodes_b=np.array([2, 3]),
            )
        )
        mgr = _ap_out.manager
        p1 = _ap_out.pair

        p1 = _evolve_pair(
            p1,
            state=_evolve_state(
                p1.state,
                status=ContactStatus.ACTIVE,
                lambda_n=100.0,
                gap=-0.5,
            ),
        )
        mgr = _ContactManagerInput(pairs=[p1], config=mgr.config)

        _ra_out = ResetAllPairsProcess().process(ResetAllPairsInput(manager=mgr))
        mgr = _ra_out.manager

        assert mgr.pairs[0].state.status == ContactStatus.INACTIVE
        assert mgr.pairs[0].state.lambda_n == 0.0
        assert mgr.pairs[0].state.gap == 0.0

    def test_custom_config(self):
        """カスタム設定のマネージャ."""
        cfg = _ContactConfigInput(mu=0.5, use_friction=True)
        mgr = _ContactManagerInput(config=cfg)
        assert mgr.config.mu == 0.5
        assert mgr.config.use_friction is True


# ---------------------------------------------------------------------------
# Phase C1: detect_candidates / update_geometry / Active-set
# ---------------------------------------------------------------------------
class TestDetectCandidates:
    """DetectCandidatesProcess (broadphase候補探索) のテスト."""

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
        mgr = _ContactManagerInput()
        _dc_out = DetectCandidatesProcess().process(
            DetectCandidatesInput(
                manager=mgr,
                node_coords=coords,
                connectivity=conn,
                radii=0.5,
                margin=0.5,
            )
        )
        mgr = _dc_out.manager
        candidates = _dc_out.candidates

        assert len(candidates) >= 1
        assert (0, 1) in candidates
        assert _n_pairs(mgr) >= 1

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
        mgr = _ContactManagerInput()
        _dc_out = DetectCandidatesProcess().process(
            DetectCandidatesInput(
                manager=mgr,
                node_coords=coords,
                connectivity=conn,
                radii=0.1,
            )
        )
        candidates = _dc_out.candidates

        assert len(candidates) == 0

    def test_existing_pairs_deactivated(self):
        """候補から外れた既存ペアは INACTIVE になる."""
        coords, conn = self._make_two_beam_mesh()
        mgr = _ContactManagerInput()
        _dc_out = DetectCandidatesProcess().process(
            DetectCandidatesInput(
                manager=mgr,
                node_coords=coords,
                connectivity=conn,
                radii=0.5,
                margin=1.0,
            )
        )
        mgr = _dc_out.manager
        assert _n_pairs(mgr) >= 1

        # 全ペアを ACTIVE に
        new_pairs = [
            _evolve_pair(p, state=_evolve_state(p.state, status=ContactStatus.ACTIVE))
            for p in mgr.pairs
        ]
        mgr = _ContactManagerInput(pairs=new_pairs, config=mgr.config)

        coords_far = coords.copy()
        coords_far[2] = [100.0, 100.0, 100.0]
        coords_far[3] = [101.0, 100.0, 100.0]
        _dc_out = DetectCandidatesProcess().process(
            DetectCandidatesInput(
                manager=mgr,
                node_coords=coords_far,
                connectivity=conn,
                radii=0.5,
                margin=1.0,
            )
        )
        mgr = _dc_out.manager

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
        mgr = _ContactManagerInput()

        _dc_out = DetectCandidatesProcess().process(
            DetectCandidatesInput(
                manager=mgr,
                node_coords=coords,
                connectivity=conn,
                radii=0.0,
                margin=1.0,
            )
        )
        mgr = _dc_out.manager
        candidates = _dc_out.candidates

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
        mgr = _ContactManagerInput()

        _dc_out = DetectCandidatesProcess().process(
            DetectCandidatesInput(
                manager=mgr,
                node_coords=coords,
                connectivity=conn,
                radii=radii,
            )
        )
        mgr = _dc_out.manager
        candidates = _dc_out.candidates
        assert (0, 1) in candidates

        pair = mgr.pairs[0]
        assert pair.radius_a == pytest.approx(1.0)
        assert pair.radius_b == pytest.approx(4.5)


class TestUpdateGeometry:
    """UpdateGeometryProcess (narrowphase + Active-set) のテスト."""

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
        mgr = _ContactManagerInput()
        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=0,
                elem_b=1,
                nodes_a=conn[0],
                nodes_b=conn[1],
                radius_a=0.0,
                radius_b=0.0,
            )
        )
        mgr = _ap_out.manager

        _ug_out = UpdateGeometryProcess().process(
            UpdateGeometryInput(
                manager=mgr,
                node_coords=coords,
            )
        )
        mgr = _ug_out.manager

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
        mgr = _ContactManagerInput()
        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=0,
                elem_b=1,
                nodes_a=conn[0],
                nodes_b=conn[1],
            )
        )
        mgr = _ap_out.manager

        _ug_out = UpdateGeometryProcess().process(
            UpdateGeometryInput(
                manager=mgr,
                node_coords=coords,
            )
        )
        mgr = _ug_out.manager

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
        mgr = _ContactManagerInput()
        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=0,
                elem_b=1,
                nodes_a=conn[0],
                nodes_b=conn[1],
                radius_a=0.5,
                radius_b=0.5,
            )
        )
        mgr = _ap_out.manager

        _ug_out = UpdateGeometryProcess().process(
            UpdateGeometryInput(
                manager=mgr,
                node_coords=coords,
            )
        )
        mgr = _ug_out.manager

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
        mgr = _ContactManagerInput()
        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=0,
                elem_b=1,
                nodes_a=conn[0],
                nodes_b=conn[1],
                radius_a=0.5,
                radius_b=0.5,
            )
        )
        mgr = _ap_out.manager

        _ug_out = UpdateGeometryProcess().process(
            UpdateGeometryInput(
                manager=mgr,
                node_coords=coords,
            )
        )
        mgr = _ug_out.manager

        pair = mgr.pairs[0]
        assert pair.state.gap == pytest.approx(-0.5)

    def test_frame_continuity(self):
        """法線フレームの連続性（2回の UpdateGeometryProcess）."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 2.0, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        mgr = _ContactManagerInput()
        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=0,
                elem_b=1,
                nodes_a=conn[0],
                nodes_b=conn[1],
            )
        )
        mgr = _ap_out.manager

        _ug_out = UpdateGeometryProcess().process(
            UpdateGeometryInput(
                manager=mgr,
                node_coords=coords,
            )
        )
        mgr = _ug_out.manager
        t1_first = mgr.pairs[0].state.tangent1.copy()

        coords2 = coords.copy()
        coords2[2] = [0.5, 1.01, 0.01]
        _ug_out = UpdateGeometryProcess().process(
            UpdateGeometryInput(
                manager=mgr,
                node_coords=coords2,
            )
        )
        mgr = _ug_out.manager
        t1_second = mgr.pairs[0].state.tangent1

        dot = abs(float(t1_first @ t1_second))
        assert dot > 0.9


class TestActiveSetHysteresis:
    """Active-set ヒステリシスのテスト."""

    def test_activate_on_contact(self):
        """gap <= g_on で活性化."""
        cfg = _ContactConfigInput(g_on=0.0, g_off=1e-6)
        mgr = _ContactManagerInput(config=cfg)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=0,
                elem_b=1,
                nodes_a=conn[0],
                nodes_b=conn[1],
                radius_a=0.5,
                radius_b=0.5,
            )
        )
        mgr = _ap_out.manager

        _ug_out = UpdateGeometryProcess().process(
            UpdateGeometryInput(
                manager=mgr,
                node_coords=coords,
            )
        )
        mgr = _ug_out.manager

        assert mgr.pairs[0].state.status == ContactStatus.ACTIVE

    def test_stays_active_in_hysteresis_band(self):
        """活性状態で gap が g_on < gap < g_off のとき活性のまま."""
        cfg = _ContactConfigInput(g_on=0.0, g_off=0.1)
        mgr = _ContactManagerInput(config=cfg)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.05, 0.0],
                [1.0, 1.05, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=0,
                elem_b=1,
                nodes_a=conn[0],
                nodes_b=conn[1],
                radius_a=0.5,
                radius_b=0.5,
            )
        )
        mgr = _ap_out.manager
        # ペアを ACTIVE に設定
        p = mgr.pairs[0]
        p = _evolve_pair(p, state=_evolve_state(p.state, status=ContactStatus.ACTIVE))
        mgr = _ContactManagerInput(pairs=[p], config=mgr.config)

        _ug_out = UpdateGeometryProcess().process(
            UpdateGeometryInput(
                manager=mgr,
                node_coords=coords,
            )
        )
        mgr = _ug_out.manager

        assert mgr.pairs[0].state.gap == pytest.approx(0.05)
        assert mgr.pairs[0].state.status == ContactStatus.ACTIVE

    def test_deactivate_beyond_g_off(self):
        """gap >= g_off で非活性化."""
        cfg = _ContactConfigInput(g_on=0.0, g_off=0.1)
        mgr = _ContactManagerInput(config=cfg)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=0,
                elem_b=1,
                nodes_a=conn[0],
                nodes_b=conn[1],
                radius_a=0.5,
                radius_b=0.5,
            )
        )
        mgr = _ap_out.manager
        # ペアを ACTIVE に設定
        p = mgr.pairs[0]
        p = _evolve_pair(p, state=_evolve_state(p.state, status=ContactStatus.ACTIVE))
        mgr = _ContactManagerInput(pairs=[p], config=mgr.config)

        _ug_out = UpdateGeometryProcess().process(
            UpdateGeometryInput(
                manager=mgr,
                node_coords=coords,
            )
        )
        mgr = _ug_out.manager

        assert mgr.pairs[0].state.status == ContactStatus.INACTIVE

    def test_inactive_stays_inactive_in_band(self):
        """非活性状態で g_on < gap < g_off → 非活性のまま（ヒステリシス）."""
        cfg = _ContactConfigInput(g_on=0.0, g_off=0.1)
        mgr = _ContactManagerInput(config=cfg)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.05, 0.0],
                [1.0, 1.05, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=0,
                elem_b=1,
                nodes_a=conn[0],
                nodes_b=conn[1],
                radius_a=0.5,
                radius_b=0.5,
            )
        )
        mgr = _ap_out.manager

        _ug_out = UpdateGeometryProcess().process(
            UpdateGeometryInput(
                manager=mgr,
                node_coords=coords,
            )
        )
        mgr = _ug_out.manager

        assert mgr.pairs[0].state.status == ContactStatus.INACTIVE

    def test_sliding_deactivates_beyond_g_off(self):
        """SLIDING 状態でも gap >= g_off で非活性化."""
        cfg = _ContactConfigInput(g_on=0.0, g_off=0.1)
        mgr = _ContactManagerInput(config=cfg)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [1.0, 5.0, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]])
        _ap_out = AddPairProcess().process(
            AddPairInput(
                manager=mgr,
                elem_a=0,
                elem_b=1,
                nodes_a=conn[0],
                nodes_b=conn[1],
                radius_a=0.5,
                radius_b=0.5,
            )
        )
        mgr = _ap_out.manager
        # ペアを SLIDING に設定
        p = mgr.pairs[0]
        p = _evolve_pair(p, state=_evolve_state(p.state, status=ContactStatus.SLIDING))
        mgr = _ContactManagerInput(pairs=[p], config=mgr.config)

        _ug_out = UpdateGeometryProcess().process(
            UpdateGeometryInput(
                manager=mgr,
                node_coords=coords,
            )
        )
        mgr = _ug_out.manager

        assert mgr.pairs[0].state.status == ContactStatus.INACTIVE
