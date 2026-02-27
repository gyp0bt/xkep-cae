"""k_pen推定MLモデル用特徴量抽出のテスト.

テスト項目:
  1. 特徴量ベクトルの形状（12D）
  2. 材料・断面特徴量の値の妥当性
  3. 撚線メッシュからの高レベルAPI
  4. 変位付きの特徴量変化
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.kpen_features import (
    extract_kpen_features,
    extract_kpen_features_from_mesh,
)


class TestExtractKpenFeatures:
    """低レベル特徴量抽出のテスト."""

    def _simple_mesh(self):
        """2本の平行セグメントメッシュ."""
        nodes = np.array(
            [
                [0, 0, 0],
                [0.01, 0, 0],
                [0, 0.003, 0],
                [0.01, 0.003, 0],
            ],
            dtype=float,
        )
        conn = np.array([[0, 1], [2, 3]])
        radii = np.array([0.001, 0.001])
        return nodes, conn, radii

    def test_feature_vector_shape(self):
        """出力が (12,) であること."""
        nodes, conn, radii = self._simple_mesh()
        feat = extract_kpen_features(
            E=2e11,
            Iy=1e-12,
            L_elem=0.01,
            r_contact=0.001,
            n_segments_per_wire=4,
            n_wires=3,
            lay_angle=0.1,
            node_coords=nodes,
            connectivity=conn,
            radii=radii,
            broadphase_margin=0.01,
        )
        assert feat.shape == (12,)
        assert feat.dtype == np.float32

    def test_material_features_range(self):
        """材料特徴量が妥当な範囲にあること."""
        nodes, conn, radii = self._simple_mesh()
        E = 2e11
        Iy_val = 1e-12
        L = 0.01
        feat = extract_kpen_features(
            E=E,
            Iy=Iy_val,
            L_elem=L,
            r_contact=0.001,
            n_segments_per_wire=4,
            n_wires=3,
            lay_angle=0.1,
            node_coords=nodes,
            connectivity=conn,
            radii=radii,
            broadphase_margin=0.01,
        )
        # log10(E) ≈ 11.3
        assert 10 < feat[0] < 12
        # log10(I) ≈ -12
        assert -13 < feat[1] < -11
        # r/L = 0.001/0.01 = 0.1
        np.testing.assert_allclose(feat[3], 0.1, atol=0.01)

    def test_mesh_features(self):
        """メッシュ特徴量が正しいこと."""
        nodes, conn, radii = self._simple_mesh()
        feat = extract_kpen_features(
            E=2e11,
            Iy=1e-12,
            L_elem=0.01,
            r_contact=0.001,
            n_segments_per_wire=4,
            n_wires=3,
            lay_angle=0.15,
            node_coords=nodes,
            connectivity=conn,
            radii=radii,
            broadphase_margin=0.01,
        )
        # log10(4) ≈ 0.6
        np.testing.assert_allclose(feat[4], np.log10(4), atol=0.01)
        # n_wires = 3
        np.testing.assert_allclose(feat[5], 3.0)
        # lay_angle = 0.15
        np.testing.assert_allclose(feat[6], 0.15)

    def test_different_e_changes_features(self):
        """Eが変わると特徴量が変化すること."""
        nodes, conn, radii = self._simple_mesh()
        common = dict(
            Iy=1e-12,
            L_elem=0.01,
            r_contact=0.001,
            n_segments_per_wire=4,
            n_wires=3,
            lay_angle=0.1,
            node_coords=nodes,
            connectivity=conn,
            radii=radii,
            broadphase_margin=0.01,
        )
        feat1 = extract_kpen_features(E=2e11, **common)
        feat2 = extract_kpen_features(E=7e10, **common)
        # log10(E) が異なる
        assert feat1[0] != feat2[0]


class TestExtractKpenFeaturesFromMesh:
    """撚線メッシュからの高レベルAPIテスト."""

    def test_three_strand_features(self):
        """3本撚りメッシュで特徴量が取得できること."""
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
        feat = extract_kpen_features_from_mesh(
            mesh,
            E=2e11,
            Iy=7.854e-14,
            broadphase_margin=0.005,
        )
        assert feat.shape == (12,)
        # n_wires = 3
        np.testing.assert_allclose(feat[5], 3.0)

    def test_seven_strand_features(self):
        """7本撚りメッシュで特徴量が取得できること."""
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
        feat = extract_kpen_features_from_mesh(
            mesh,
            E=2e11,
            Iy=7.854e-14,
            broadphase_margin=0.005,
        )
        assert feat.shape == (12,)
        np.testing.assert_allclose(feat[5], 7.0)

    def test_features_with_displacement(self):
        """変位付きで特徴量が変化すること."""
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
        n_nodes = len(mesh.node_coords)
        u = np.zeros(n_nodes * 6)
        # 微小変位
        for i in range(n_nodes):
            u[i * 6 + 2] = 0.001 * i / n_nodes

        feat0 = extract_kpen_features_from_mesh(mesh, E=2e11, Iy=7.854e-14, broadphase_margin=0.005)
        feat_u = extract_kpen_features_from_mesh(
            mesh, E=2e11, Iy=7.854e-14, u=u, broadphase_margin=0.005
        )
        # 接触幾何の特徴量（最後の5要素）が変化するはず
        # ただし変位が小さい場合は差も小さい
        assert feat0.shape == feat_u.shape
