"""MPCEliminationProcess のテスト.

剛体結合 MPC 変換行列の正しさを検証する。

[← README](../../../README.md)
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import scipy.sparse as sp

from xkep_cae.constraints.mpc_elimination import (
    MPCEliminationConfig,
    MPCEliminationProcess,
    MPCEliminationResult,
    MPCGroup,
    RebuildMPCTransformInput,
    RebuildMPCTransformProcess,
    _build_mpc_transform,
)
from xkep_cae.core import binds_to


@binds_to(MPCEliminationProcess)
class TestMPCEliminationProcessAPI:
    """MPCEliminationProcess の API テスト."""

    def test_process_returns_result(self) -> None:
        """Process が MPCEliminationResult を返す."""
        grp = MPCGroup(
            master_node=0,
            slave_nodes=np.array([1]),
            slave_coords=np.array([[1.0, 0.0, 0.0]]),
            master_coord=np.array([0.0, 0.0, 0.0]),
        )
        config = MPCEliminationConfig(
            mpc_groups=[grp],
            ndof_total=12,
            ndof_per_node=6,
        )
        proc = MPCEliminationProcess()
        result = proc.process(config)
        assert isinstance(result, MPCEliminationResult)
        assert result.ndof_reduced == 6  # master 6DOF のみ独立
        assert result.T.shape == (12, 6)

    def test_meta_name(self) -> None:
        """meta.name が正しい."""
        assert MPCEliminationProcess.meta.name == "MPCElimination"

    def test_uses_is_empty(self) -> None:
        """uses が空（前処理なので依存なし）."""
        assert MPCEliminationProcess.uses == []


class TestMPCTransformMatrix:
    """変換行列 T の数学的正しさを検証."""

    def _make_single_group(
        self,
        master_node: int = 0,
        slave_nodes: list[int] | None = None,
        slave_coords: np.ndarray | None = None,
        master_coord: np.ndarray | None = None,
        ndof_total: int = 12,
    ) -> MPCEliminationResult:
        if slave_nodes is None:
            slave_nodes = [1]
        if slave_coords is None:
            slave_coords = np.array([[1.0, 0.0, 0.0]])
        if master_coord is None:
            master_coord = np.array([0.0, 0.0, 0.0])
        grp = MPCGroup(
            master_node=master_node,
            slave_nodes=np.array(slave_nodes),
            slave_coords=slave_coords,
            master_coord=master_coord,
        )
        return _build_mpc_transform([grp], ndof_total)

    def test_identity_for_master_dofs(self) -> None:
        """master DOF に対応する T の行は単位行列の部分."""
        result = self._make_single_group()
        T = result.T.toarray()
        # master node=0: DOF 0-5 → reduced 0-5
        for i in range(6):
            assert T[i, i] == 1.0

    def test_translation_coupling(self) -> None:
        """slave 並進 DOF が master 並進 + θ×r で結合."""
        # master at origin, slave at (L, 0, 0)
        L = 2.0
        result = self._make_single_group(
            slave_coords=np.array([[L, 0.0, 0.0]]),
            master_coord=np.array([0.0, 0.0, 0.0]),
        )
        T = result.T.toarray()
        # slave DOF: node 1 → DOF 6,7,8 (trans), 9,10,11 (rot)
        # master DOF: node 0 → DOF 0,1,2 (trans), 3,4,5 (rot)
        # reduced index for master: 0,1,2,3,4,5

        # u_slave_x = u_master_x + 0*θx + 0*θy + 0*θz
        # (r = [L,0,0], [r]× = [[0,0,0],[0,0,-L],[0,L,0]])
        # u_slave_x = u_master_x
        assert T[6, 0] == 1.0  # u_master_x
        # u_slave_y = u_master_y + (-L)*θz → [r]×[1,2] = -L
        # Actually [r]× = [[0, -0, 0], [0, 0, -L], [-0, L, 0]]
        # Wait, r = [L, 0, 0], so:
        # [r]× = [[0, -0, 0], [0, 0, -L], [0, L, 0]]
        # So: u_slave_y = u_master_y + 0*θx + 0*θy + (-L)*θz
        npt.assert_allclose(T[7, 1], 1.0)  # u_master_y
        npt.assert_allclose(T[7, 5], -L)  # θ_master_z × r

        # u_slave_z = u_master_z + 0*θx + L*θy + 0*θz
        npt.assert_allclose(T[8, 2], 1.0)  # u_master_z
        npt.assert_allclose(T[8, 4], L)  # θ_master_y × r

    def test_rotation_coupling(self) -> None:
        """slave 回転 DOF が master 回転に等しい."""
        result = self._make_single_group()
        T = result.T.toarray()
        # slave rotation DOF 9,10,11 = master rotation DOF 3,4,5
        for a in range(3):
            npt.assert_allclose(T[9 + a, 3 + a], 1.0)

    def test_T_transpose_T_invertible(self) -> None:
        """T^T T が正則（独立DOFが冗長でない）."""
        result = self._make_single_group()
        T = result.T.toarray()
        TtT = T.T @ T
        det = np.linalg.det(TtT)
        assert det > 1e-10, f"T^T T is singular: det={det}"

    def test_rigid_body_mode(self) -> None:
        """剛体並進モード: slave も master と同じ並進を示す."""
        L = 3.0
        result = self._make_single_group(
            slave_coords=np.array([[L, 0.0, 0.0]]),
        )
        T = result.T.toarray()
        # 純並進: u_master = [1, 0, 0, 0, 0, 0]
        u_reduced = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u_full = T @ u_reduced
        # slave translation should also be [1, 0, 0]
        npt.assert_allclose(u_full[6:9], [1.0, 0.0, 0.0])

    def test_rigid_rotation_mode(self) -> None:
        """剛体回転モード: slave は θ×r だけ追加並進."""
        L = 5.0
        result = self._make_single_group(
            slave_coords=np.array([[L, 0.0, 0.0]]),
        )
        T = result.T.toarray()
        # 純回転 θ_z = 1: u_master = [0, 0, 0, 0, 0, 1]
        u_reduced = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        u_full = T @ u_reduced
        # r = [L, 0, 0], θ = [0, 0, 1]
        # u_slave = θ × r = [0, 0, 1] × [L, 0, 0] = [0*0-1*0, 1*L-0*0, 0*0-0*L]
        # = [0, L, 0]  ... wait let me compute [r]× @ θ:
        # [r]× = [[0, 0, 0], [0, 0, -L], [0, L, 0]]
        # [r]× @ [0, 0, 1] = [0, -L, 0]
        npt.assert_allclose(u_full[6:9], [0.0, -L, 0.0])
        # slave rotation = master rotation
        npt.assert_allclose(u_full[9:12], [0.0, 0.0, 1.0])

    def test_multiple_slaves(self) -> None:
        """複数slave節点のMPC."""
        grp = MPCGroup(
            master_node=0,
            slave_nodes=np.array([1, 2]),
            slave_coords=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            master_coord=np.array([0.0, 0.0, 0.0]),
        )
        result = _build_mpc_transform([grp], 18)  # 3 nodes × 6
        assert result.ndof_reduced == 6  # only master is independent
        assert len(result.slave_dofs) == 12

    def test_two_groups(self) -> None:
        """2つのMPCグループ（両端結合）."""
        # 3 nodes: node 0 = master_left, node 2 = master_right
        # node 1 = slave of both? No. Let's say:
        # node 0 = master_left, node 1 = slave_left
        # node 2 = master_right, node 3 = slave_right
        grp1 = MPCGroup(
            master_node=0,
            slave_nodes=np.array([1]),
            slave_coords=np.array([[0.5, 0.0, 0.0]]),
            master_coord=np.array([0.0, 0.0, 0.0]),
        )
        grp2 = MPCGroup(
            master_node=3,
            slave_nodes=np.array([2]),
            slave_coords=np.array([[1.5, 0.0, 0.0]]),
            master_coord=np.array([2.0, 0.0, 0.0]),
        )
        result = _build_mpc_transform([grp1, grp2], 24)  # 4 nodes × 6
        assert result.ndof_reduced == 12  # 2 masters × 6 DOF
        assert len(result.slave_dofs) == 12

    def test_stiffness_transform(self) -> None:
        """K_red = T^T K T の静的凝縮テスト."""
        # 2 nodes, node 0 = master, node 1 = slave
        # Simple spring: k * (u1 - u0)^2/2 → K = [[k, -k], [-k, k]]
        # For 6DOF × 2 nodes = 12DOF system
        ndof = 12
        k = 100.0
        K = sp.lil_matrix((ndof, ndof))
        # Only couple translation x: DOF 0 and DOF 6
        K[0, 0] = k
        K[0, 6] = -k
        K[6, 0] = -k
        K[6, 6] = k
        K = K.tocsc()

        result = self._make_single_group(ndof_total=ndof)
        T = result.T

        K_red = T.T @ K @ T
        # After MPC: slave DOFs eliminated, K_red should be 6×6
        assert K_red.shape == (6, 6)
        # With rigid coupling, u_slave_x = u_master_x (r along x, no rotation coupling for x)
        # So K_red should be zero (internal spring with rigid coupling gives no stiffness)
        K_red_dense = K_red.toarray()
        npt.assert_allclose(K_red_dense, 0.0, atol=1e-12)

    def test_force_transform(self) -> None:
        """f_red = T^T f の力変換テスト."""
        ndof = 12
        f = np.zeros(ndof)
        f[6] = 10.0  # slave node x-force

        result = self._make_single_group(ndof_total=ndof)
        T = result.T

        f_red = T.T @ f
        # slave x-force → master x-force (rigid coupling)
        assert f_red.shape == (6,)
        npt.assert_allclose(f_red[0], 10.0)  # transferred to master x

    def test_slave_y_force_creates_moment(self) -> None:
        """slave y方向力がmaster回転モーメントを生成."""
        L = 2.0
        ndof = 12
        f = np.zeros(ndof)
        f[7] = 10.0  # slave node y-force

        result = self._make_single_group(
            slave_coords=np.array([[L, 0.0, 0.0]]),
            ndof_total=ndof,
        )
        T = result.T

        f_red = T.T @ f
        # f_red[1] = 10.0 (y-force on master)
        npt.assert_allclose(f_red[1], 10.0)
        # f_red[5] = [r]×^T @ f → moment about z
        # [r]× = [[0, 0, 0], [0, 0, -L], [0, L, 0]]
        # [r]×^T @ [0, 10, 0] = [[0, 0, 0], [0, 0, L], [0, -L, 0]] @ [0, 10, 0]
        # = [0, 0, -L*10]
        # So f_red[5] (θz) = -L * 10 = -20
        npt.assert_allclose(f_red[5], -L * 10.0)


@binds_to(RebuildMPCTransformProcess)
class TestRebuildMPCTransformProcessAPI:
    """RebuildMPCTransformProcess の API テスト."""

    def test_process_returns_result(self) -> None:
        """Process が MPCEliminationResult を返す."""
        grp = MPCGroup(
            master_node=0,
            slave_nodes=np.array([1]),
            slave_coords=np.array([[1.0, 0.0, 0.0]]),
            master_coord=np.array([0.0, 0.0, 0.0]),
        )
        node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        inp = RebuildMPCTransformInput(
            mpc_groups=[grp],
            node_coords=node_coords,
            ndof_total=12,
            ndof_per_node=6,
        )
        result = RebuildMPCTransformProcess().process(inp)
        assert isinstance(result, MPCEliminationResult)
        assert result.ndof_reduced == 6
        assert result.T.shape == (12, 6)

    def test_meta_name(self) -> None:
        """meta.name が正しい."""
        assert RebuildMPCTransformProcess.meta.name == "RebuildMPCTransform"

    def test_updated_coords_change_transform(self) -> None:
        """変形後座標で再構築するとTが変化する."""
        grp = MPCGroup(
            master_node=0,
            slave_nodes=np.array([1]),
            slave_coords=np.array([[1.0, 0.0, 0.0]]),
            master_coord=np.array([0.0, 0.0, 0.0]),
        )
        # 初期配置
        coords_ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result_ref = RebuildMPCTransformProcess().process(
            RebuildMPCTransformInput(mpc_groups=[grp], node_coords=coords_ref, ndof_total=12)
        )
        # 変形後: slaveがy方向に移動
        coords_def = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
        result_def = RebuildMPCTransformProcess().process(
            RebuildMPCTransformInput(mpc_groups=[grp], node_coords=coords_def, ndof_total=12)
        )
        # T行列が異なる（相対位置ベクトルが変わるため）
        T_ref = result_ref.T.toarray()
        T_def = result_def.T.toarray()
        assert not np.allclose(T_ref, T_def), "変形後のTが変化すべき"
