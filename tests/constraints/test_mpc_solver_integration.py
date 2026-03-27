"""MPC統合テスト — LinearSolveProcess でのMPC DOF消去検証.

MPCEliminationResult を LinearSolveProcess に渡し、
縮退系ソルブ + 復元の正しさを検証する。

[← README](../../README.md)
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import scipy.sparse as sp

from xkep_cae.constraints.mpc_elimination import (
    MPCEliminationConfig,
    MPCEliminationProcess,
    MPCGroup,
)
from xkep_cae.contact.solver._newton_steps import (
    LinearSolveInput,
    LinearSolveProcess,
)


class TestLinearSolveWithMPC:
    """LinearSolveProcess + MPC のソルバー統合テスト."""

    @staticmethod
    def _add_diagonal_stiffness(K: sp.lil_matrix, ndof: int, k_diag: float = 1.0) -> None:
        """全DOFに最小限の対角剛性を付与（特異性回避）."""
        for i in range(ndof):
            K[i, i] += k_diag

    def test_mpc_solve_rigid_coupling(self) -> None:
        """MPC剛体結合で3ノード系の正しいソルブ.

        3ノード: node 0 = master, node 1 = slave, node 2 = free
        スプリング: node 1 - node 2 間 (x方向 k=100)
        MPC: node 1 → node 0
        固定: master全DOF固定 → slave も固定 → du[node2] = F/k
        """
        ndof = 18  # 3 nodes × 6 DOF
        k = 100.0
        K = sp.lil_matrix((ndof, ndof))
        # Spring between node 1 (DOF 6) and node 2 (DOF 12)
        K[6, 6] = k
        K[6, 12] = -k
        K[12, 6] = -k
        K[12, 12] = k
        # 特異性回避: 全DOFに微小対角剛性
        self._add_diagonal_stiffness(K, ndof, k_diag=1.0)
        K = K.tocsc()

        grp = MPCGroup(
            master_node=0,
            slave_nodes=np.array([1]),
            slave_coords=np.array([[1.0, 0.0, 0.0]]),
            master_coord=np.array([0.0, 0.0, 0.0]),
        )
        mpc_proc = MPCEliminationProcess()
        mpc_result = mpc_proc.process(MPCEliminationConfig(mpc_groups=[grp], ndof_total=ndof))

        # 外力: node 2 x方向 10N
        R_u = np.zeros(ndof)
        R_u[12] = -10.0  # -R_u = f_ext

        # 固定: master全DOF固定
        fixed_dofs = np.arange(6, dtype=int)

        proc = LinearSolveProcess()
        result = proc.process(
            LinearSolveInput(
                K_T=K,
                R_u=R_u,
                fixed_dofs=fixed_dofs,
                mpc_transform=mpc_result,
            )
        )
        assert result.success
        # master固定 → slave固定（MPC結合）→ du[node2 x] ≈ F/(k+k_diag)
        npt.assert_allclose(result.du[0], 0.0, atol=1e-10)
        npt.assert_allclose(result.du[6], 0.0, atol=1e-10)  # slave
        npt.assert_allclose(result.du[12], 10.0 / 101.0, atol=1e-8)  # free

    def test_mpc_solve_cantilever(self) -> None:
        """片持ち梁のMPC剛体結合端部テスト.

        3ノード梁: node 0, 1 がslave（固定端）、node 2 が自由端。
        node 0, 1 → master_node = 3（参照点）に結合。
        master参照点を全固定して片持ち梁を模擬。
        """
        ndof = 24  # 4 nodes × 6 DOF
        # 簡易剛性: node 1-2 間にx方向スプリング
        k = 1000.0
        K = sp.lil_matrix((ndof, ndof))
        K[6, 6] = k
        K[6, 12] = -k
        K[12, 6] = -k
        K[12, 12] = k
        self._add_diagonal_stiffness(K, ndof, k_diag=1.0)
        K = K.tocsc()

        # MPC: node 0, 1 → master node 3
        grp = MPCGroup(
            master_node=3,
            slave_nodes=np.array([0, 1]),
            slave_coords=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            master_coord=np.array([0.25, 0.0, 0.0]),
        )
        mpc_proc = MPCEliminationProcess()
        mpc_result = mpc_proc.process(MPCEliminationConfig(mpc_groups=[grp], ndof_total=ndof))

        # 固定: master 全DOF固定
        fixed_dofs = np.arange(18, 24, dtype=int)  # node 3: DOF 18-23

        # 外力: node 2 x方向 50N
        R_u = np.zeros(ndof)
        R_u[12] = -50.0  # -R_u = f_ext

        proc = LinearSolveProcess()
        result = proc.process(
            LinearSolveInput(
                K_T=K,
                R_u=R_u,
                fixed_dofs=fixed_dofs,
                mpc_transform=mpc_result,
            )
        )
        assert result.success
        # node 2 x方向: du ≈ F/(k+k_diag) = 50/1001 ≈ 0.04995
        npt.assert_allclose(result.du[12], 50.0 / 1001.0, atol=1e-8)
        # slave nodes (0, 1) は master (3) に結合 → 全固定
        npt.assert_allclose(result.du[0:6], 0.0, atol=1e-10)
        npt.assert_allclose(result.du[6:12], 0.0, atol=1e-10)

    def test_mpc_solve_without_mpc_unchanged(self) -> None:
        """MPC なしの場合、従来通りの結果."""
        ndof = 12
        k = 100.0
        K = sp.lil_matrix((ndof, ndof))
        K[0, 0] = k
        K[0, 6] = -k
        K[6, 0] = -k
        K[6, 6] = k
        self._add_diagonal_stiffness(K, ndof, k_diag=1.0)
        K = K.tocsc()

        R_u = np.zeros(ndof)
        R_u[6] = -10.0
        fixed_dofs = np.arange(6, dtype=int)  # fix node 0 all DOFs

        proc = LinearSolveProcess()
        result = proc.process(LinearSolveInput(K_T=K, R_u=R_u, fixed_dofs=fixed_dofs))
        assert result.success
        # du[0:6] = 0 (fixed), du[6] = F/(k+k_diag) ≈ 10/101
        npt.assert_allclose(result.du[0], 0.0, atol=1e-10)
        npt.assert_allclose(result.du[6], 10.0 / 101.0, atol=1e-8)

    def test_mpc_rotation_creates_translation(self) -> None:
        """MPC経由で回転が並進変位を生む.

        解析解は T^T K T の縮退系から直接計算し、
        slave DOF の復元（θ×r）の正しさを検証する。
        """
        ndof = 18  # 3 nodes: node 0 master, node 1 slave, node 2
        L = 2.0
        k_rot = 50.0
        K = sp.lil_matrix((ndof, ndof))
        K[5, 5] = k_rot
        self._add_diagonal_stiffness(K, ndof, k_diag=1.0)
        K_csc = K.tocsc()

        grp = MPCGroup(
            master_node=0,
            slave_nodes=np.array([1]),
            slave_coords=np.array([[L, 0.0, 0.0]]),
            master_coord=np.array([0.0, 0.0, 0.0]),
        )
        mpc_proc = MPCEliminationProcess()
        mpc_result = mpc_proc.process(MPCEliminationConfig(mpc_groups=[grp], ndof_total=ndof))

        # トルク: master θ_z に M = 100
        R_u = np.zeros(ndof)
        R_u[5] = -100.0
        fixed_dofs = np.array([0, 1, 2], dtype=int)

        # 参照解: T^T K T を陽に構築してnumpyで解く
        T = mpc_result.T.toarray()
        K_red_dense = T.T @ K_csc.toarray() @ T
        rhs_red = T.T @ (-R_u)
        # BC適用
        indep_to_red = {int(d): j for j, d in enumerate(mpc_result.independent_dofs)}
        fixed_red = [indep_to_red[int(d)] for d in fixed_dofs if int(d) in indep_to_red]
        for d in fixed_red:
            K_red_dense[d, :] = 0.0
            K_red_dense[:, d] = 0.0
            K_red_dense[d, d] = 1.0
            rhs_red[d] = 0.0
        du_red_ref = np.linalg.solve(K_red_dense, rhs_red)
        du_ref = T @ du_red_ref

        proc = LinearSolveProcess()
        result = proc.process(
            LinearSolveInput(
                K_T=K_csc,
                R_u=R_u,
                fixed_dofs=fixed_dofs,
                mpc_transform=mpc_result,
            )
        )
        assert result.success
        # Process結果が参照解と一致
        npt.assert_allclose(result.du, du_ref, atol=1e-10)
        # slave θ_z = master θ_z（MPC結合の基本性質）
        npt.assert_allclose(result.du[11], result.du[5], atol=1e-12)
        # slave y = -L * θ_z（回転→並進結合）
        npt.assert_allclose(result.du[7], -L * result.du[5], atol=1e-10)
