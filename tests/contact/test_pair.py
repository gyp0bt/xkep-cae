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
