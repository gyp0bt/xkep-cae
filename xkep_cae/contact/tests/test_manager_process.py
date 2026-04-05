"""ContactManager Process の C3 紐付けテスト.

@binds_to による 1:1 紐付け。
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact._contact_pair import (
    _ContactConfigInput,
    _ContactManagerInput,
    _evolve_pair,
    _evolve_state,
    _n_pairs,
)
from xkep_cae.contact._manager_process import (
    AddPairInput,
    AddPairProcess,
    DetectCandidatesInput,
    DetectCandidatesProcess,
    ResetAllPairsInput,
    ResetAllPairsProcess,
    _build_end_element_set,
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


class TestBuildEndElementSet:
    """端部要素セット構築のテスト（status-296）."""

    def test_single_strand_3_elems_exclude_1(self):
        """3要素の素線で端部1要素を除外."""
        conn = np.array([[0, 1], [1, 2], [2, 3]])
        esm = {0: 0, 1: 0, 2: 0}
        result = _build_end_element_set(conn, esm, n_exclude=1)
        # 端部要素は 0 と 2
        assert result == {0, 2}

    def test_single_strand_5_elems_exclude_2(self):
        """5要素の素線で端部2要素を除外."""
        conn = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
        esm = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        result = _build_end_element_set(conn, esm, n_exclude=2)
        # 左端: 0, 1  右端: 3, 4
        assert result == {0, 1, 3, 4}

    def test_two_strands(self):
        """2素線で各端部1要素を除外."""
        conn = np.array([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7]])
        esm = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
        result = _build_end_element_set(conn, esm, n_exclude=1)
        # strand 0: 端部 = 0, 2  strand 1: 端部 = 3, 5
        assert result == {0, 2, 3, 5}

    def test_exclude_0_returns_empty(self):
        """n_exclude=0で空セット."""
        conn = np.array([[0, 1], [1, 2]])
        esm = {0: 0, 1: 0}
        result = _build_end_element_set(conn, esm, n_exclude=0)
        assert result == set()

    def test_short_strand_all_excluded(self):
        """短い素線（要素数 <= 2*n_exclude）は全要素除外."""
        conn = np.array([[0, 1], [1, 2]])
        esm = {0: 0, 1: 0}
        result = _build_end_element_set(conn, esm, n_exclude=1)
        assert result == {0, 1}


class TestExcludeEndElementsIntegration:
    """端部要素除外の統合テ��ト（status-296）."""

    def test_exclude_end_elements_filters_candidates(self):
        """exclude_end_elements=1 で端部要素の候補が除外される."""
        # 2素線、各3要素、y方向に近接
        r = 0.5
        sep = 2 * r - 0.01  # わずかに貫入
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, sep, 0.0],
                [1.0, sep, 0.0],
                [2.0, sep, 0.0],
                [3.0, sep, 0.0],
            ]
        )
        conn = np.array([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7]])
        esm = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

        config_no_excl = _ContactConfigInput(
            exclude_same_strand=True,
            exclude_end_elements=0,
            elem_strand_map=esm,
        )
        config_excl = _ContactConfigInput(
            exclude_same_strand=True,
            exclude_end_elements=1,
            elem_strand_map=esm,
        )

        mgr_no = _ContactManagerInput(config=config_no_excl, connectivity=conn)
        mgr_ex = _ContactManagerInput(config=config_excl, connectivity=conn)

        out_no = DetectCandidatesProcess().process(
            DetectCandidatesInput(
                manager=mgr_no, node_coords=coords, connectivity=conn, radii=r, margin=0.5
            )
        )
        out_ex = DetectCandidatesProcess().process(
            DetectCandidatesInput(
                manager=mgr_ex, node_coords=coords, connectivity=conn, radii=r, margin=0.5
            )
        )

        # 除外なし: 全異素線ペアが候補（最大9ペア）
        assert len(out_no.candidates) > 0
        # 除外あり: 端部要素(0,2,3,5)を含むペアが除外される
        # 残るのは中央要素同士のペア (1, 4) のみ
        for i, j in out_ex.candidates:
            assert i not in {0, 2, 3, 5}, f"端部要素 {i} が候補に残っている"
            assert j not in {0, 2, 3, 5}, f"端部要素 {j} が候補に残っている"
        assert len(out_ex.candidates) < len(out_no.candidates)
