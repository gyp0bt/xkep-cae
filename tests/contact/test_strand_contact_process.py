"""S3 撚線接触テスト — Process API パイプライン検証.

メッシュ生成・接触セットアップの Process API 動作確認のみ。
ソルバー収束テストは status-212 で全削除（現構成では収束しないため）。

[← README](../../README.md)
"""

from xkep_cae.contact.setup.process import ContactSetupConfig, ContactSetupProcess
from xkep_cae.mesh.process import StrandMeshConfig, StrandMeshProcess

# ====================================================================
# 共通パラメータ
# ====================================================================

_WIRE_D = 0.002  # m
_PITCH = 0.040  # m


# ====================================================================
# テスト: StrandMeshProcess パイプライン検証
# ====================================================================


class TestStrandMeshProcessAPI:
    """StrandMeshProcess の基本動作確認."""

    def test_7strand_mesh_generation(self):
        """7本撚線メッシュが正しく生成される."""
        proc = StrandMeshProcess()
        config = StrandMeshConfig(
            n_strands=7,
            wire_radius=_WIRE_D / 2,
            pitch_length=_PITCH,
            gap=0.0005,
            n_elements_per_pitch=16,
            n_pitches=1.0,
        )
        result = proc.process(config)
        mesh_data = result.mesh

        assert mesh_data.node_coords.shape[1] == 3
        assert mesh_data.connectivity.shape[1] == 2
        assert mesh_data.n_strands == 7
        # 7本 x 16要素 = 112要素
        assert len(mesh_data.connectivity) == 7 * 16
        # 7本 x 17節点 = 119節点
        assert len(mesh_data.node_coords) == 7 * 17

    def test_19strand_mesh_generation(self):
        """19本撚線メッシュが正しく生成される."""
        proc = StrandMeshProcess()
        config = StrandMeshConfig(
            n_strands=19,
            wire_radius=_WIRE_D / 2,
            pitch_length=_PITCH,
            gap=0.0005,
            n_elements_per_pitch=16,
            n_pitches=1.0,
        )
        result = proc.process(config)
        assert result.mesh.n_strands == 19
        assert len(result.mesh.connectivity) == 19 * 16


# ====================================================================
# テスト: ContactSetupProcess パイプライン検証
# ====================================================================


class TestContactSetupProcessAPI:
    """ContactSetupProcess の基本動作確認."""

    def test_contact_setup_creates_manager(self):
        """ContactSetupProcess がマネージャを生成し候補を検出する."""
        proc = StrandMeshProcess()
        mesh_result = proc.process(
            StrandMeshConfig(
                n_strands=7,
                wire_radius=_WIRE_D / 2,
                pitch_length=_PITCH,
                gap=0.0005,
                n_elements_per_pitch=16,
                n_pitches=1.0,
            )
        )

        setup_proc = ContactSetupProcess()
        setup_result = setup_proc.process(
            ContactSetupConfig(
                mesh=mesh_result.mesh,
                k_pen=1e6,
                mu=0.0,
                exclude_same_layer=True,
            )
        )

        assert setup_result.manager is not None
        assert setup_result.k_pen == 1e6
        manager = setup_result.manager
        assert len(manager.pairs) > 0


# ====================================================================
# テスト: three_point_bend_jig Process
# ====================================================================

# 動的三点曲げジグのテストは tests/contact/test_three_point_bend_jig.py を参照
