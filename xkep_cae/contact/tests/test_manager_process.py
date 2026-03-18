"""ContactManager Process の C3 紐付けテスト.

@binds_to による 1:1 紐付け。
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact._contact_pair import (
    _ContactManagerInput,
    _evolve_pair,
    _evolve_state,
    _n_pairs,
)
from xkep_cae.contact._manager_process import (
    AddPairInput,
    AddPairProcess,
    ResetAllPairsInput,
    ResetAllPairsProcess,
)
from xkep_cae.contact._types import ContactStatus
from xkep_cae.core.testing import binds_to


@binds_to(AddPairProcess)
class TestAddPairProcessAPI:
    """AddPairProcess の C3 紐付けテスト."""

    def test_add_pair_returns_new_manager(self):
        """ペア追加後に新 manager が返る."""
        mgr = _ContactManagerInput()
        out = AddPairProcess().process(
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
        assert _n_pairs(out.manager) == 1
        assert out.pair.elem_a == 0
        assert out.pair.elem_b == 1


@binds_to(ResetAllPairsProcess)
class TestResetAllPairsProcessAPI:
    """ResetAllPairsProcess の C3 紐付けテスト."""

    def test_reset_clears_states(self):
        """全ペアの状態がリセットされる."""
        mgr = _ContactManagerInput()
        out = AddPairProcess().process(
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
        mgr = out.manager
        # ACTIVE 状態に変更
        pair = mgr.pairs[0]
        new_pair = _evolve_pair(pair, state=_evolve_state(pair.state, status=ContactStatus.ACTIVE))
        mgr = _ContactManagerInput(pairs=[new_pair], config=mgr.config)
        reset_out = ResetAllPairsProcess().process(ResetAllPairsInput(manager=mgr))
        assert reset_out.manager.pairs[0].state.status == ContactStatus.INACTIVE
