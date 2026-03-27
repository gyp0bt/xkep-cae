"""RigidEdgeAssemblerProcess のテスト.

[← README](../../../README.md)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae.constraints.rigid_assembler import (
    RigidEdgeAssemblerConfig,
    RigidEdgeAssemblerProcess,
    RigidEdgeAssemblerResult,
)
from xkep_cae.core import binds_to


@binds_to(RigidEdgeAssemblerProcess)
class TestRigidEdgeAssemblerProcessAPI:
    """RigidEdgeAssemblerProcess の API テスト."""

    def test_process_creates_assembler(self) -> None:
        """アセンブラが正しく生成される."""
        jig_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        config = RigidEdgeAssemblerConfig(
            jig_coords=jig_coords,
            n_jig_nodes=2,
            global_node_offset=10,
            total_ndof=72,
        )
        proc = RigidEdgeAssemblerProcess()
        result = proc.process(config)

        assert isinstance(result, RigidEdgeAssemblerResult)
        assert result.n_jig_nodes == 2
        assert result.total_ndof == 72
        assert result.assembler is not None

    def test_assembler_tangent_is_zero(self) -> None:
        """接線剛性行列がゼロ."""
        config = RigidEdgeAssemblerConfig(
            jig_coords=np.zeros((2, 3)),
            n_jig_nodes=2,
            global_node_offset=0,
            total_ndof=12,
        )
        result = RigidEdgeAssemblerProcess().process(config)
        asm = result.assembler

        K = asm.assemble_tangent(np.zeros(12))
        assert isinstance(K, sp.csr_matrix)
        assert K.shape == (12, 12)
        assert K.nnz == 0

    def test_assembler_internal_force_is_zero(self) -> None:
        """内力ベクトルがゼロ."""
        config = RigidEdgeAssemblerConfig(
            jig_coords=np.zeros((2, 3)),
            n_jig_nodes=2,
            global_node_offset=0,
            total_ndof=12,
        )
        result = RigidEdgeAssemblerProcess().process(config)
        asm = result.assembler

        f = asm.assemble_internal_force(np.zeros(12))
        assert f.shape == (12,)
        np.testing.assert_array_equal(f, 0.0)

    def test_assembler_update_reference(self) -> None:
        """update_reference で座標が更新される."""
        jig_coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        config = RigidEdgeAssemblerConfig(
            jig_coords=jig_coords,
            n_jig_nodes=2,
            global_node_offset=0,
            total_ndof=12,
        )
        result = RigidEdgeAssemblerProcess().process(config)
        asm = result.assembler

        u_incr = np.zeros(12)
        u_incr[0] = 0.1  # node0 x
        u_incr[1] = 0.2  # node0 y
        u_incr[6] = 0.3  # node1 x

        asm.update_reference(u_incr)

        np.testing.assert_allclose(asm.coords_ref[0], [1.1, 2.2, 3.0])
        np.testing.assert_allclose(asm.coords_ref[1], [4.3, 5.0, 6.0])

    def test_assembler_checkpoint_rollback(self) -> None:
        """checkpoint/rollback が正しく動作する."""
        config = RigidEdgeAssemblerConfig(
            jig_coords=np.array([[1.0, 0.0, 0.0]]),
            n_jig_nodes=1,
            global_node_offset=0,
            total_ndof=6,
        )
        result = RigidEdgeAssemblerProcess().process(config)
        asm = result.assembler

        asm.checkpoint()

        u_incr = np.zeros(6)
        u_incr[0] = 10.0
        asm.update_reference(u_incr)
        assert asm.coords_ref[0, 0] == 11.0

        asm.rollback()
        assert asm.coords_ref[0, 0] == 1.0

    def test_frozen_config(self) -> None:
        """Config は frozen dataclass."""
        config = RigidEdgeAssemblerConfig(
            jig_coords=np.zeros((1, 3)),
            n_jig_nodes=1,
            global_node_offset=0,
            total_ndof=6,
        )
        try:
            config.n_jig_nodes = 99  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow mutation")
        except AttributeError:
            pass

    def test_frozen_result(self) -> None:
        """Result は frozen dataclass."""
        result = RigidEdgeAssemblerProcess().process(
            RigidEdgeAssemblerConfig(
                jig_coords=np.zeros((1, 3)),
                n_jig_nodes=1,
                global_node_offset=0,
                total_ndof=6,
            )
        )
        try:
            result.n_jig_nodes = 99  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow mutation")
        except AttributeError:
            pass
