"""Broadphase (AABB格子) のテスト."""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.broadphase import (
    broadphase_aabb,
    compute_segment_aabb,
)


# ---------------------------------------------------------------------------
# compute_segment_aabb
# ---------------------------------------------------------------------------
class TestComputeSegmentAABB:
    """compute_segment_aabb のテスト."""

    def test_basic_aabb(self):
        """基本的な AABB 計算."""
        x0 = np.array([1.0, 2.0, 3.0])
        x1 = np.array([4.0, 1.0, 5.0])
        lo, hi = compute_segment_aabb(x0, x1)

        np.testing.assert_allclose(lo, [1.0, 1.0, 3.0])
        np.testing.assert_allclose(hi, [4.0, 2.0, 5.0])

    def test_aabb_with_radius(self):
        """半径による膨張."""
        x0 = np.array([0.0, 0.0, 0.0])
        x1 = np.array([1.0, 0.0, 0.0])
        lo, hi = compute_segment_aabb(x0, x1, radius=0.5)

        np.testing.assert_allclose(lo, [-0.5, -0.5, -0.5])
        np.testing.assert_allclose(hi, [1.5, 0.5, 0.5])

    def test_aabb_with_margin(self):
        """マージンによる追加膨張."""
        x0 = np.array([0.0, 0.0, 0.0])
        x1 = np.array([1.0, 0.0, 0.0])
        lo, hi = compute_segment_aabb(x0, x1, radius=0.1, margin=0.2)

        expand = 0.3  # radius + margin
        np.testing.assert_allclose(lo, [-expand, -expand, -expand])
        np.testing.assert_allclose(hi, [1.0 + expand, expand, expand])

    def test_aabb_reversed_endpoints(self):
        """端点の順序に依存しない."""
        x0 = np.array([3.0, 1.0, 5.0])
        x1 = np.array([1.0, 4.0, 2.0])
        lo, hi = compute_segment_aabb(x0, x1)

        np.testing.assert_allclose(lo, [1.0, 1.0, 2.0])
        np.testing.assert_allclose(hi, [3.0, 4.0, 5.0])


# ---------------------------------------------------------------------------
# broadphase_aabb
# ---------------------------------------------------------------------------
class TestBroadphaseAABB:
    """broadphase_aabb のテスト."""

    def test_two_crossing_segments(self):
        """交差する2セグメント → 候補として検出."""
        segments = [
            (np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])),
            (np.array([1.0, -1.0, 0.0]), np.array([1.0, 1.0, 0.0])),
        ]
        candidates = broadphase_aabb(segments, radii=0.0, margin=0.1)

        assert (0, 1) in candidates

    def test_two_distant_segments(self):
        """離れた2セグメント → 候補なし."""
        segments = [
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([100.0, 100.0, 100.0]), np.array([101.0, 100.0, 100.0])),
        ]
        candidates = broadphase_aabb(segments, radii=0.0, margin=0.1)

        assert len(candidates) == 0

    def test_radius_brings_segments_closer(self):
        """半径が十分大きいと離れたセグメントも候補に."""
        segments = [
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([0.0, 3.0, 0.0]), np.array([1.0, 3.0, 0.0])),
        ]
        # 半径0 → 候補なし
        c0 = broadphase_aabb(segments, radii=0.0, margin=0.0)
        # 半径2.0 → AABB が重なる
        c2 = broadphase_aabb(segments, radii=2.0, margin=0.0)

        assert len(c0) == 0
        assert (0, 1) in c2

    def test_single_segment(self):
        """1セグメントのみ → 候補なし."""
        segments = [
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        ]
        candidates = broadphase_aabb(segments)

        assert len(candidates) == 0

    def test_empty_segments(self):
        """空のセグメントリスト → 候補なし."""
        candidates = broadphase_aabb([])

        assert len(candidates) == 0

    def test_multiple_segments_grid(self):
        """複数セグメントのグリッド配置: 近接ペアのみ検出."""
        # 10本のx方向セグメント、y方向に等間隔
        segments = []
        for i in range(10):
            y = float(i) * 2.0  # 間隔 2.0
            segments.append((np.array([0.0, y, 0.0]), np.array([1.0, y, 0.0])))

        # 半径0, margin=0.5 → 間隔2.0に対し AABB幅 0.5×2=1.0 < 2.0 → 候補なし
        c_no = broadphase_aabb(segments, radii=0.0, margin=0.5)
        assert len(c_no) == 0

        # 半径1.0, margin=0.1 → 膨張 1.1 → 隣接のみ検出
        c_adj = broadphase_aabb(segments, radii=1.0, margin=0.1)
        # 隣接ペア: (0,1),(1,2),...,(8,9) = 9ペア
        for i in range(9):
            assert (i, i + 1) in c_adj

        # 非隣接（間隔4.0）は含まない
        for i in range(8):
            assert (i, i + 2) not in c_adj

    def test_parallel_overlapping_segments(self):
        """平行で重なるセグメント → 候補検出."""
        segments = [
            (np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0])),
            (np.array([3.0, 0.5, 0.0]), np.array([8.0, 0.5, 0.0])),
        ]
        candidates = broadphase_aabb(segments, radii=0.0, margin=1.0)

        assert (0, 1) in candidates

    def test_3d_skew_segments(self):
        """3D ねじれ配置のセグメント."""
        segments = [
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 1.5])),
        ]
        # margin十分 → 検出
        candidates = broadphase_aabb(segments, radii=0.0, margin=1.0)
        assert (0, 1) in candidates

    def test_per_segment_radii(self):
        """セグメントごとに異なる半径."""
        segments = [
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([0.0, 4.0, 0.0]), np.array([1.0, 4.0, 0.0])),
        ]
        radii = np.array([1.0, 3.5])  # 合計4.5 > 4.0（間隔）
        candidates = broadphase_aabb(segments, radii=radii, margin=0.0)
        assert (0, 1) in candidates

    def test_custom_cell_size(self):
        """カスタムセルサイズの指定."""
        segments = [
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([0.5, 0.5, 0.0]), np.array([1.5, 0.5, 0.0])),
        ]
        # 非常に小さいセルサイズ + margin で重なりを保証
        c1 = broadphase_aabb(segments, margin=1.0, cell_size=0.1)
        assert (0, 1) in c1

        # 非常に大きいセルサイズ（全体が1セル）+ margin
        c2 = broadphase_aabb(segments, margin=1.0, cell_size=100.0)
        assert (0, 1) in c2

    def test_no_self_pairs(self):
        """同一セグメントのペアは含まない."""
        segments = [
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([0.5, 0.0, 0.0]), np.array([1.5, 0.0, 0.0])),
        ]
        candidates = broadphase_aabb(segments, margin=1.0)

        for i, j in candidates:
            assert i != j
            assert i < j  # 正規化済み

    def test_no_duplicates(self):
        """候補ペアに重複がない."""
        segments = [
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([0.5, 0.0, 0.0]), np.array([1.5, 0.0, 0.0])),
            (np.array([0.2, 0.0, 0.0]), np.array([0.8, 0.0, 0.0])),
        ]
        candidates = broadphase_aabb(segments, margin=1.0)

        assert len(candidates) == len(set(candidates))
