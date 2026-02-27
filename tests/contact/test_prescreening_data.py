"""接触プリスクリーニングGNN用データ生成パイプラインのテスト.

テスト項目:
  1. セグメント抽出（初期形状/変形後）
  2. セグメント特徴量計算（形状、次元数）
  3. エッジ特徴量計算（候補ペアの特徴量）
  4. 接触ラベル付与（gap 判定）
  5. サンプル生成（統合テスト: 撚線メッシュ使用）
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.prescreening_data import (
    compute_edge_features,
    compute_segment_features,
    extract_segments,
    generate_prescreening_sample,
    label_contact_pairs,
)


class TestExtractSegments:
    """セグメント抽出のテスト."""

    def test_basic_extraction(self):
        """2要素メッシュからセグメント抽出."""
        nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        conn = np.array([[0, 1], [1, 2]])
        segments = extract_segments(nodes, conn)
        assert len(segments) == 2
        np.testing.assert_array_equal(segments[0][0], [0, 0, 0])
        np.testing.assert_array_equal(segments[0][1], [1, 0, 0])

    def test_with_displacement(self):
        """変位ベクトル付きで変形後座標を取得."""
        nodes = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        conn = np.array([[0, 1]])
        # ndof_per_node=6: [ux, uy, uz, rx, ry, rz]
        u = np.array([0, 0, 0.1, 0, 0, 0, 0.05, 0, 0.2, 0, 0, 0], dtype=float)
        segments = extract_segments(nodes, conn, u, ndof_per_node=6)
        np.testing.assert_allclose(segments[0][0], [0, 0, 0.1])
        np.testing.assert_allclose(segments[0][1], [1.05, 0, 0.2])

    def test_no_displacement(self):
        """変位なしの場合は初期座標."""
        nodes = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        conn = np.array([[0, 1]])
        segments = extract_segments(nodes, conn, None)
        np.testing.assert_array_equal(segments[0][0], [0, 0, 0])
        np.testing.assert_array_equal(segments[0][1], [1, 0, 0])


class TestSegmentFeatures:
    """セグメント特徴量計算のテスト."""

    def test_feature_shape(self):
        """出力が (N, 10) であること."""
        segments = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([0, 1, 0.0]), np.array([1, 1, 0.0])),
        ]
        radii = np.array([0.01, 0.02])
        feat = compute_segment_features(segments, radii)
        assert feat.shape == (2, 10)

    def test_midpoint(self):
        """中点が正しく計算されること."""
        segments = [(np.array([0, 0, 0.0]), np.array([2, 0, 0.0]))]
        radii = np.array([0.01])
        feat = compute_segment_features(segments, radii)
        np.testing.assert_allclose(feat[0, :3], [1, 0, 0])

    def test_direction_normalized(self):
        """方向ベクトルが単位ベクトルであること."""
        segments = [(np.array([0, 0, 0.0]), np.array([3, 4, 0.0]))]
        radii = np.array([0.01])
        feat = compute_segment_features(segments, radii)
        direction = feat[0, 3:6]
        np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-6)

    def test_wire_layer_ids(self):
        """wire_id と layer_id が正しく格納されること."""
        segments = [(np.array([0, 0, 0.0]), np.array([1, 0, 0.0]))]
        radii = np.array([0.01])
        feat = compute_segment_features(
            segments, radii, wire_ids=np.array([3]), layer_ids=np.array([1])
        )
        assert feat[0, 8] == 3.0
        assert feat[0, 9] == 1.0


class TestEdgeFeatures:
    """エッジ特徴量計算のテスト."""

    def test_edge_feature_shape(self):
        """出力が (E, 7) であること."""
        segments = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([0, 1, 0.0]), np.array([1, 1, 0.0])),
        ]
        candidates = [(0, 1)]
        feat = compute_edge_features(segments, candidates)
        assert feat.shape == (1, 7)

    def test_parallel_segments_cos_angle_one(self):
        """平行セグメントの cos_angle が 1 に近いこと."""
        segments = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([0, 0.1, 0.0]), np.array([1, 0.1, 0.0])),
        ]
        feat = compute_edge_features(segments, [(0, 1)])
        assert feat[0, 4] > 0.99  # cos_angle ≈ 1

    def test_perpendicular_segments_cos_angle_zero(self):
        """直交セグメントの cos_angle が 0 に近いこと."""
        segments = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([0.5, -0.5, 0.0]), np.array([0.5, 0.5, 0.0])),
        ]
        feat = compute_edge_features(segments, [(0, 1)])
        assert feat[0, 4] < 0.01  # cos_angle ≈ 0

    def test_same_wire_flag(self):
        """同一素線フラグが正しいこと."""
        segments = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([1, 0, 0.0]), np.array([2, 0, 0.0])),
            (np.array([0, 1, 0.0]), np.array([1, 1, 0.0])),
        ]
        wire_ids = np.array([0, 0, 1])
        feat = compute_edge_features(segments, [(0, 1), (0, 2)], wire_ids=wire_ids)
        assert feat[0, 5] == 1.0  # 同一素線
        assert feat[1, 5] == 0.0  # 異なる素線


class TestLabelContactPairs:
    """接触ラベル付与のテスト."""

    def test_close_segments_labeled_contact(self):
        """近接セグメントが接触ラベル 1 を取得."""
        # 距離 0.1 で半径 0.1 のセグメント同士 → gap < r_i + r_j
        segments = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([0, 0.1, 0.0]), np.array([1, 0.1, 0.0])),
        ]
        radii = np.array([0.1, 0.1])
        labels = label_contact_pairs(segments, [(0, 1)], radii)
        assert labels[0] == 1

    def test_far_segments_labeled_no_contact(self):
        """離れたセグメントが非接触ラベル 0 を取得."""
        segments = [
            (np.array([0, 0, 0.0]), np.array([1, 0, 0.0])),
            (np.array([0, 10, 0.0]), np.array([1, 10, 0.0])),
        ]
        radii = np.array([0.01, 0.01])
        labels = label_contact_pairs(segments, [(0, 1)], radii)
        assert labels[0] == 0


class TestGeneratePrescreeningSample:
    """統合テスト: 撚線メッシュからのサンプル生成."""

    def test_three_strand_sample(self):
        """3本撚りメッシュでサンプル生成が動作すること."""
        from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh

        mesh = make_twisted_wire_mesh(
            3,
            0.002,
            0.04,
            length=0.0,
            n_elems_per_strand=4,
            n_pitches=1.0,
            gap=0.0,
        )

        wire_ids = np.zeros(len(mesh.connectivity), dtype=int)
        for i, (na, _nb) in enumerate(mesh.connectivity):
            for sid, (ns, ne) in enumerate(mesh.strand_node_ranges):
                if ns <= na < ne:
                    wire_ids[i] = sid
                    break

        sample = generate_prescreening_sample(
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            wire_ids=wire_ids,
            broadphase_margin=0.005,
        )

        assert sample["node_features"].shape[1] == 10
        assert sample["edge_features"].shape[1] == 7
        assert sample["edge_index"].shape[0] == 2
        assert sample["n_candidates"] >= 0
        assert sample["n_contact"] >= 0
        assert sample["n_contact"] <= sample["n_candidates"]

    def test_sample_with_displacement(self):
        """変位付きでサンプル生成."""
        from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh

        mesh = make_twisted_wire_mesh(
            3,
            0.002,
            0.04,
            length=0.0,
            n_elems_per_strand=4,
            n_pitches=1.0,
            gap=0.0005,
        )

        n_nodes = len(mesh.node_coords)
        u = np.zeros(n_nodes * 6)
        # 微小引張変位
        for sid in range(mesh.n_strands):
            ns, ne = mesh.strand_node_ranges[sid]
            for nid in range(ns, ne):
                u[nid * 6 + 2] = 0.001 * (nid - ns) / max(1, ne - ns - 1)

        sample = generate_prescreening_sample(
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            u=u,
            broadphase_margin=0.005,
        )
        assert sample["node_features"].ndim == 2

    def test_seven_strand_has_candidates(self):
        """7本撚りメッシュで候補ペアが生成されること."""
        from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh

        mesh = make_twisted_wire_mesh(
            7,
            0.002,
            0.04,
            length=0.0,
            n_elems_per_strand=4,
            n_pitches=1.0,
            gap=0.0,
        )
        sample = generate_prescreening_sample(
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            broadphase_margin=0.005,
        )
        assert sample["n_candidates"] > 0, "7本撚りで候補ペアがゼロ"

    def test_empty_candidates(self):
        """離れたセグメントで候補ペアがゼロの場合のハンドリング."""
        nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [100, 100, 100],
                [101, 100, 100],
            ],
            dtype=float,
        )
        conn = np.array([[0, 1], [2, 3]])
        radii = np.array([0.001, 0.001])

        sample = generate_prescreening_sample(nodes, conn, radii, broadphase_margin=0.0)
        assert sample["n_candidates"] == 0
        assert sample["edge_index"].shape == (2, 0)
        assert sample["labels"].shape == (0,)
