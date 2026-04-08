"""KD-tree ベース broadphase テスト.

status-308: _broadphase_aabb の KD-tree 実装検証。

[← README](../../../README.md)
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.contact._broadphase import _broadphase_aabb


class TestBroadphaseAPI:
    """broadphase 基本 API テスト."""

    def test_empty_returns_empty(self) -> None:
        """空入力で空リスト."""
        assert _broadphase_aabb([]) == []

    def test_single_segment_returns_empty(self) -> None:
        """1セグメントではペアなし."""
        seg = [(np.array([0, 0, 0.0]), np.array([1, 0, 0.0]))]
        assert _broadphase_aabb(seg) == []

    def test_two_overlapping_segments(self) -> None:
        """2本の近接セグメントが候補に含まれる."""
        seg = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([0, 0.1, 0.0]), np.array([1, 0.1, 0.0])),
        ]
        result = _broadphase_aabb(seg, radii=0.2)
        assert len(result) == 1
        assert result[0] == (0, 1)

    def test_two_distant_segments_no_pair(self) -> None:
        """2本の遠いセグメントは候補に含まれない."""
        seg = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([100, 100, 100.0]), np.array([101, 100, 100.0])),
        ]
        result = _broadphase_aabb(seg, radii=0.1)
        assert len(result) == 0

    def test_pair_order_i_less_j(self) -> None:
        """結果ペアは常に i < j."""
        seg = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([0, 0.1, 0.0]), np.array([1, 0.1, 0.0])),
            (np.array([0, 0.2, 0.0]), np.array([1, 0.2, 0.0])),
        ]
        result = _broadphase_aabb(seg, radii=0.3)
        for i, j in result:
            assert i < j

    def test_scalar_radii(self) -> None:
        """スカラー半径で動作."""
        seg = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([0, 0.5, 0.0]), np.array([1, 0.5, 0.0])),
        ]
        result = _broadphase_aabb(seg, radii=0.3)
        assert len(result) == 1

    def test_array_radii(self) -> None:
        """配列半径で動作."""
        seg = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([0, 0.5, 0.0]), np.array([1, 0.5, 0.0])),
        ]
        result = _broadphase_aabb(seg, radii=np.array([0.3, 0.3]))
        assert len(result) == 1

    def test_margin_expands_search(self) -> None:
        """margin がAABBを拡張して候補を増やす."""
        seg = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([0, 2, 0.0]), np.array([1, 2, 0.0])),
        ]
        # 半径0.1、距離2.0 → margin=0でペアなし
        assert len(_broadphase_aabb(seg, radii=0.1, margin=0.0)) == 0
        # margin=1.0 → AABB拡張でペア検出
        assert len(_broadphase_aabb(seg, radii=0.1, margin=1.0)) == 1


class TestBroadphasePhysics:
    """broadphase 物理的妥当性テスト."""

    def test_parallel_helical_wires(self) -> None:
        """ヘリカル素線の配置で正しくペアを検出."""
        # 7本撚線のような配置: 中心1本 + 周囲6本
        n_seg = 10
        t = np.linspace(0, 2 * np.pi, n_seg + 1)
        segs = []
        r_wire = 0.5
        r_pitch = 2.0
        # 中心素線
        for i in range(n_seg):
            segs.append((np.array([0, 0, t[i]]), np.array([0, 0, t[i + 1]])))
        # 周囲素線1本（x方向オフセット）
        for i in range(n_seg):
            segs.append(
                (
                    np.array([r_pitch, 0, t[i]]),
                    np.array([r_pitch, 0, t[i + 1]]),
                )
            )
        result = _broadphase_aabb(segs, radii=r_wire)
        # 近接セグメント同士がペアとして検出される
        assert len(result) > 0
        # 全ペアが異なるセグメント
        for i, j in result:
            assert i != j

    def test_no_false_negatives_for_touching(self) -> None:
        """接触しているペアが見落とされない（偽陰性なし）."""
        # 2本のセグメントが半径分だけ離れている（ちょうど接触）
        r = 0.5
        seg = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([0, 2 * r, 0.0]), np.array([1, 2 * r, 0.0])),
        ]
        result = _broadphase_aabb(seg, radii=r)
        # ちょうど接触 → AABBは重複するので検出されるべき
        assert len(result) == 1

    def test_cell_size_backward_compat(self) -> None:
        """cell_size パラメータが後方互換で受け入れられる."""
        seg = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([0, 0.1, 0.0]), np.array([1, 0.1, 0.0])),
        ]
        # cell_size は無視されるが、エラーにならない
        result = _broadphase_aabb(seg, radii=0.2, cell_size=0.5)
        assert len(result) == 1

    @pytest.mark.parametrize("n_segs", [50, 100, 200])
    def test_scaling_consistency(self, n_segs: int) -> None:
        """要素数増加で結果の整合性を保つ."""
        rng = np.random.default_rng(42)
        segs = []
        for _ in range(n_segs):
            p0 = rng.uniform(-10, 10, size=3)
            p1 = p0 + rng.uniform(-1, 1, size=3)
            segs.append((p0, p1))
        result = _broadphase_aabb(segs, radii=0.5)
        # ペアは (i, j) で i < j
        for i, j in result:
            assert 0 <= i < j < n_segs
        # 重複なし
        assert len(result) == len(set(result))
