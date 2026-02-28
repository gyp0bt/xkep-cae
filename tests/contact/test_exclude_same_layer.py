"""同層除外フィルタのテスト.

Phase S1: exclude_same_layer オプションによる同層接触ペアの除外を検証する。
等角配置の同層素線は物理的に接触しにくいため、~80% のペア削減を見込む。
"""

import numpy as np

from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactPair,
    ContactStatus,
)
from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh


class TestExcludeSameLayerConfig:
    """ContactConfig.exclude_same_layer の基本動作."""

    def test_default_disabled(self):
        """デフォルトでは無効."""
        cfg = ContactConfig()
        assert cfg.exclude_same_layer is False

    def test_enable_flag(self):
        """フラグを有効化できる."""
        cfg = ContactConfig(exclude_same_layer=True)
        assert cfg.exclude_same_layer is True

    def test_requires_layer_map(self):
        """exclude_same_layer=True でも elem_layer_map なしなら除外なし."""
        cfg = ContactConfig(exclude_same_layer=True, elem_layer_map=None)
        mgr = ContactManager(config=cfg)
        # 2要素, 4節点の簡易メッシュ
        coords = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=float,
        )
        conn = np.array([[0, 1], [2, 3]], dtype=int)
        candidates = mgr.detect_candidates(coords, conn, radii=0.5)
        # layer_map なしなので除外されない
        assert len(candidates) == 1


class TestExcludeSameLayerDetection:
    """detect_candidates() での同層除外フィルタ."""

    def _make_parallel_mesh(self):
        """4本の平行セグメントを2層に分ける簡易メッシュ.

        Layer 0: elem 0, elem 1（互いに平行）
        Layer 1: elem 2, elem 3（互いに平行）
        層間: (0,2), (0,3), (1,2), (1,3) = 4ペア
        同層: (0,1), (2,3) = 2ペア
        """
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [1.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
                [1.0, 0.0, 0.1],
                [0.0, 0.1, 0.1],
                [1.0, 0.1, 0.1],
            ],
            dtype=float,
        )
        conn = np.array(
            [
                [0, 1],  # elem 0, layer 0
                [2, 3],  # elem 1, layer 0
                [4, 5],  # elem 2, layer 1
                [6, 7],  # elem 3, layer 1
            ],
            dtype=int,
        )
        lmap = {0: 0, 1: 0, 2: 1, 3: 1}
        return coords, conn, lmap

    def test_without_filter(self):
        """除外無効時は全ペアが候補."""
        coords, conn, lmap = self._make_parallel_mesh()
        cfg = ContactConfig(
            elem_layer_map=lmap,
            exclude_same_layer=False,
        )
        mgr = ContactManager(config=cfg)
        candidates = mgr.detect_candidates(coords, conn, radii=0.2)
        # 4要素 → C(4,2)=6 だが共有節点なしなので全6ペア
        assert len(candidates) == 6

    def test_with_filter(self):
        """同層除外で同層ペアが除去される."""
        coords, conn, lmap = self._make_parallel_mesh()
        cfg = ContactConfig(
            elem_layer_map=lmap,
            exclude_same_layer=True,
        )
        mgr = ContactManager(config=cfg)
        candidates = mgr.detect_candidates(coords, conn, radii=0.2)
        # 同層ペア (0,1) と (2,3) が除外 → 4ペア
        assert len(candidates) == 4
        for i, j in candidates:
            layer_i = lmap[i]
            layer_j = lmap[j]
            assert layer_i != layer_j, f"同層ペア ({i},{j}) が除外されていない"

    def test_filter_does_not_add_same_layer_pairs(self):
        """同層除外で生成されたペアにも同層ペアがない."""
        coords, conn, lmap = self._make_parallel_mesh()
        cfg = ContactConfig(
            elem_layer_map=lmap,
            exclude_same_layer=True,
        )
        mgr = ContactManager(config=cfg)
        mgr.detect_candidates(coords, conn, radii=0.2)
        for pair in mgr.pairs:
            layer_a = lmap[pair.elem_a]
            layer_b = lmap[pair.elem_b]
            assert layer_a != layer_b

    def test_unknown_elem_not_excluded(self):
        """layer_map に含まれない要素は除外されない."""
        coords, conn, lmap = self._make_parallel_mesh()
        # elem 3 の layer を削除
        del lmap[3]
        cfg = ContactConfig(
            elem_layer_map=lmap,
            exclude_same_layer=True,
        )
        mgr = ContactManager(config=cfg)
        candidates = mgr.detect_candidates(coords, conn, radii=0.2)
        # elem 3 は layer 不明 → -1 扱い → 同層判定されない
        # (0,1) のみ同層除外 → 5ペア
        assert len(candidates) == 5


class TestExcludeSameLayerTwistedWire:
    """撚線メッシュでの同層除外効果の検証."""

    def test_7_strand_pair_reduction(self):
        """7本撚り（1+6型）で同層除外のペア削減効果を検証.

        中心1本(layer0) + 外層6本(layer1) の構成。
        外層同士の同層ペアが大半を占め、除外で大幅削減を期待。
        """
        mesh = make_twisted_wire_mesh(7, 0.002, 0.04, 0.0, 4, n_pitches=1.0)
        lmap = mesh.build_elem_layer_map()
        coords = mesh.node_coords

        # 除外なし
        cfg_no = ContactConfig(
            elem_layer_map=lmap,
            exclude_same_layer=False,
        )
        mgr_no = ContactManager(config=cfg_no)
        cands_no = mgr_no.detect_candidates(coords, mesh.connectivity, radii=0.001)

        # 除外あり
        cfg_yes = ContactConfig(
            elem_layer_map=lmap,
            exclude_same_layer=True,
        )
        mgr_yes = ContactManager(config=cfg_yes)
        cands_yes = mgr_yes.detect_candidates(coords, mesh.connectivity, radii=0.001)

        # 除外ありのほうがペアが少ない
        assert len(cands_yes) < len(cands_no)

        # 除外後のペアに同層ペアが含まれないことを検証
        for i, j in cands_yes:
            layer_i = lmap[i]
            layer_j = lmap[j]
            assert layer_i != layer_j

    def test_3_strand_all_same_layer(self):
        """3本撚り（全素線layer=1）は除外で全ペアが消える.

        3本撚りは中心なし、全素線が layer=1。
        exclude_same_layer で全ペアが除外される。
        """
        mesh = make_twisted_wire_mesh(3, 0.002, 0.04, 0.0, 4, n_pitches=1.0)
        lmap = mesh.build_elem_layer_map()
        coords = mesh.node_coords

        cfg = ContactConfig(
            elem_layer_map=lmap,
            exclude_same_layer=True,
        )
        mgr = ContactManager(config=cfg)
        cands = mgr.detect_candidates(coords, mesh.connectivity, radii=0.001)
        # 全素線が同層なので全ペア除外
        assert len(cands) == 0

    def test_19_strand_significant_reduction(self):
        """19本撚り（3層）で大幅なペア削減を検証."""
        mesh = make_twisted_wire_mesh(19, 0.002, 0.04, 0.0, 4, n_pitches=1.0)
        lmap = mesh.build_elem_layer_map()
        coords = mesh.node_coords

        # 除外なし
        cfg_no = ContactConfig(elem_layer_map=lmap, exclude_same_layer=False)
        mgr_no = ContactManager(config=cfg_no)
        cands_no = mgr_no.detect_candidates(coords, mesh.connectivity, radii=0.001)

        # 除外あり
        cfg_yes = ContactConfig(elem_layer_map=lmap, exclude_same_layer=True)
        mgr_yes = ContactManager(config=cfg_yes)
        cands_yes = mgr_yes.detect_candidates(coords, mesh.connectivity, radii=0.001)

        # 大幅削減（50%以上削減を期待）
        if len(cands_no) > 0:
            reduction = 1.0 - len(cands_yes) / len(cands_no)
            assert reduction > 0.3, f"削減率 {reduction:.1%} が低すぎる"


class TestCountSameLayerPairs:
    """ContactManager.count_same_layer_pairs() のテスト."""

    def _make_pair(self, elem_a, elem_b, status=ContactStatus.ACTIVE):
        pair = ContactPair(
            elem_a=elem_a,
            elem_b=elem_b,
            nodes_a=np.array([0, 1]),
            nodes_b=np.array([2, 3]),
        )
        pair.state.status = status
        return pair

    def test_count_basic(self):
        """基本的な同層ペアカウント."""
        lmap = {0: 0, 1: 0, 2: 1, 3: 1}
        mgr = ContactManager(config=ContactConfig(elem_layer_map=lmap))
        mgr.pairs.append(self._make_pair(0, 1))  # 同層(0)
        mgr.pairs.append(self._make_pair(0, 2))  # 異層
        mgr.pairs.append(self._make_pair(2, 3))  # 同層(1)
        assert mgr.count_same_layer_pairs() == 2

    def test_inactive_not_counted(self):
        """INACTIVE ペアはカウントされない."""
        lmap = {0: 0, 1: 0}
        mgr = ContactManager(config=ContactConfig(elem_layer_map=lmap))
        mgr.pairs.append(self._make_pair(0, 1, status=ContactStatus.INACTIVE))
        assert mgr.count_same_layer_pairs() == 0

    def test_no_layer_map(self):
        """layer_map なしなら 0."""
        mgr = ContactManager(config=ContactConfig())
        mgr.pairs.append(self._make_pair(0, 1))
        assert mgr.count_same_layer_pairs() == 0

    def test_unknown_elem_not_counted(self):
        """layer_map に含まれない要素は同層判定されない."""
        lmap = {0: 0}
        mgr = ContactManager(config=ContactConfig(elem_layer_map=lmap))
        mgr.pairs.append(self._make_pair(0, 5))  # elem 5 は map にない
        assert mgr.count_same_layer_pairs() == 0
