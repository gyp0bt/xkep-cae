"""段階的接触アクティベーションのテスト.

ContactManager の層別接触フィルタリングと段階的オンの動作を検証する。
"""

import numpy as np

from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactPair,
    ContactStatus,
)
from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh


class TestElemLayerMap:
    """TwistedWireMesh.build_elem_layer_map() のテスト."""

    def test_3_strand_all_layer_1(self):
        """3本撚りは全素線が layer=1."""
        mesh = make_twisted_wire_mesh(3, 0.002, 0.04, 0.0, 8, n_pitches=1.0)
        lmap = mesh.build_elem_layer_map()
        assert len(lmap) == mesh.n_elems
        # 3本撚りは中心なし、全素線 layer=1
        for v in lmap.values():
            assert v == 1

    def test_7_strand_center_and_layer1(self):
        """7本撚りは中心=layer 0、外=layer 1."""
        mesh = make_twisted_wire_mesh(7, 0.002, 0.04, 0.0, 4, n_pitches=1.0)
        lmap = mesh.build_elem_layer_map()
        assert len(lmap) == mesh.n_elems

        # 中心素線（strand_id=0）の要素は layer 0
        center_elems = mesh.strand_elems(0)
        for e in center_elems:
            assert lmap[int(e)] == 0

        # 外層素線（strand_id=1-6）の要素は layer 1
        for sid in range(1, 7):
            elems = mesh.strand_elems(sid)
            for e in elems:
                assert lmap[int(e)] == 1

    def test_19_strand_three_layers(self):
        """19本撚りは layer 0, 1, 2."""
        mesh = make_twisted_wire_mesh(19, 0.002, 0.04, 0.0, 4, n_pitches=1.0)
        lmap = mesh.build_elem_layer_map()
        layers = set(lmap.values())
        assert 0 in layers
        assert 1 in layers
        assert 2 in layers


class TestComputeActiveLayer:
    """ContactManager.compute_active_layer_for_step() のテスト."""

    def _make_mgr(self, staged_steps: int, elem_layer_map: dict) -> ContactManager:
        return ContactManager(
            config=ContactConfig(
                staged_activation_steps=staged_steps,
                elem_layer_map=elem_layer_map,
            ),
        )

    def test_no_staged_returns_max_layer(self):
        """staged_activation_steps=0 では常に最大層."""
        mgr = self._make_mgr(0, {0: 0, 1: 1, 2: 2})
        assert mgr.compute_active_layer_for_step(1, 20) == 2
        assert mgr.compute_active_layer_for_step(20, 20) == 2

    def test_staged_2_layers(self):
        """2層（0,1）を6ステップで段階的に。"""
        # max_layer=1, staged=6, steps_per_layer = 6/(1+1) = 3
        mgr = self._make_mgr(6, {0: 0, 1: 0, 2: 1, 3: 1})
        # step 1-3: layer 0
        assert mgr.compute_active_layer_for_step(1, 20) == 0
        assert mgr.compute_active_layer_for_step(3, 20) == 0
        # step 4+: layer 1
        assert mgr.compute_active_layer_for_step(4, 20) == 1
        assert mgr.compute_active_layer_for_step(20, 20) == 1

    def test_staged_3_layers(self):
        """3層（0,1,2）を9ステップで段階的に。"""
        # max_layer=2, staged=9, steps_per_layer = 9/3 = 3
        mgr = self._make_mgr(9, {0: 0, 1: 1, 2: 2})
        assert mgr.compute_active_layer_for_step(1, 20) == 0
        assert mgr.compute_active_layer_for_step(3, 20) == 0
        assert mgr.compute_active_layer_for_step(4, 20) == 1
        assert mgr.compute_active_layer_for_step(6, 20) == 1
        assert mgr.compute_active_layer_for_step(7, 20) == 2

    def test_no_layer_map_returns_zero(self):
        """elem_layer_map が空なら常に 0."""
        mgr = self._make_mgr(6, {})
        assert mgr.compute_active_layer_for_step(1, 20) == 0

    def test_none_layer_map_returns_zero(self):
        """elem_layer_map が None なら常に 0."""
        mgr = self._make_mgr(6, None)
        assert mgr.compute_active_layer_for_step(1, 20) == 0


class TestFilterPairsByLayer:
    """ContactManager.filter_pairs_by_layer() のテスト."""

    def _make_pair(self, elem_a: int, elem_b: int, status=ContactStatus.ACTIVE):
        pair = ContactPair(
            elem_a=elem_a,
            elem_b=elem_b,
            nodes_a=np.array([0, 1]),
            nodes_b=np.array([2, 3]),
        )
        pair.state.status = status
        return pair

    def test_filter_layer1_when_max0(self):
        """max_layer=0 なら layer 1 のペアを INACTIVE に."""
        lmap = {0: 0, 1: 0, 2: 1, 3: 1}
        mgr = ContactManager(
            config=ContactConfig(elem_layer_map=lmap),
        )
        # ペア: elem 0(layer0) vs elem 2(layer1) → layer1 があるので除外
        mgr.pairs.append(self._make_pair(0, 2))
        # ペア: elem 0(layer0) vs elem 1(layer0) → 両方layer0 → 保持
        mgr.pairs.append(self._make_pair(0, 1))

        mgr.filter_pairs_by_layer(0)
        assert mgr.pairs[0].state.status == ContactStatus.INACTIVE
        assert mgr.pairs[1].state.status == ContactStatus.ACTIVE

    def test_filter_all_active_at_max_layer(self):
        """max_layer が最大層以上なら全ペア保持."""
        lmap = {0: 0, 1: 1, 2: 1}
        mgr = ContactManager(
            config=ContactConfig(elem_layer_map=lmap),
        )
        mgr.pairs.append(self._make_pair(0, 1))
        mgr.pairs.append(self._make_pair(1, 2))

        mgr.filter_pairs_by_layer(1)
        assert mgr.pairs[0].state.status == ContactStatus.ACTIVE
        assert mgr.pairs[1].state.status == ContactStatus.ACTIVE

    def test_no_layer_map_noop(self):
        """layer_map がなければフィルタなし."""
        mgr = ContactManager(config=ContactConfig())
        mgr.pairs.append(self._make_pair(0, 1))

        mgr.filter_pairs_by_layer(0)
        assert mgr.pairs[0].state.status == ContactStatus.ACTIVE

    def test_inactive_pairs_not_affected(self):
        """既に INACTIVE のペアは影響を受けない."""
        lmap = {0: 0, 1: 1}
        mgr = ContactManager(config=ContactConfig(elem_layer_map=lmap))
        mgr.pairs.append(self._make_pair(0, 1, status=ContactStatus.INACTIVE))

        mgr.filter_pairs_by_layer(0)
        assert mgr.pairs[0].state.status == ContactStatus.INACTIVE

    def test_cross_layer_pair_filtered(self):
        """層をまたぐペア（layer0 vs layer2）は max_layer=1 で除外."""
        lmap = {0: 0, 1: 1, 2: 2}
        mgr = ContactManager(config=ContactConfig(elem_layer_map=lmap))
        mgr.pairs.append(self._make_pair(0, 2))  # layer0 vs layer2
        mgr.pairs.append(self._make_pair(0, 1))  # layer0 vs layer1
        mgr.pairs.append(self._make_pair(1, 2))  # layer1 vs layer2

        mgr.filter_pairs_by_layer(1)
        assert mgr.pairs[0].state.status == ContactStatus.INACTIVE  # layer2 超過
        assert mgr.pairs[1].state.status == ContactStatus.ACTIVE  # 両方 <=1
        assert mgr.pairs[2].state.status == ContactStatus.INACTIVE  # layer2 超過
