"""MPCEliminationProcess — DOF消去によるMPC（Multi-Point Constraint）実装.

剛体結合をDOF消去（静的凝縮）で実現する PreProcess。
梁要素（6DOF/node）の端部slave節点群をmaster参照点に結合する。

剛体結合の制約式:
  u_slave = u_master + θ_master × r   (並進3DOF)
  θ_slave = θ_master                   (回転3DOF)

ここで r = X_slave - X_master（参照配置での相対位置ベクトル）。

変換行列 T を構築し、全体系を縮退系に変換:
  K_red = T^T K T,  f_red = T^T f,  du_red = solve(K_red, f_red)
  du_full = T @ du_red

status-252 Phase B: 7本撚線端部剛体結合の前提。

[← README](../../README.md)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.core import PreProcess, ProcessMeta

# ====================================================================
# skew-symmetric matrix (hat map)
# ====================================================================


def _skew(r: np.ndarray) -> np.ndarray:
    """3Dベクトル r のスキュー対称行列 [r]× を返す."""
    return np.array(
        [
            [0.0, -r[2], r[1]],
            [r[2], 0.0, -r[0]],
            [-r[1], r[0], 0.0],
        ]
    )


# ====================================================================
# MPC データ構造
# ====================================================================


@dataclass(frozen=True)
class MPCGroup:
    """1つのMPCグループ（1 master + 複数 slave）.

    Attributes:
        master_node: master参照点の全体節点番号
        slave_nodes: slave節点の全体節点番号配列
        slave_coords: slave節点の参照座標 (n_slave, 3)
        master_coord: master参照点の参照座標 (3,)
    """

    master_node: int
    slave_nodes: np.ndarray  # (n_slave,)
    slave_coords: np.ndarray  # (n_slave, 3)
    master_coord: np.ndarray  # (3,)


@dataclass(frozen=True)
class MPCEliminationConfig:
    """MPC消去の入力.

    Attributes:
        mpc_groups: MPCグループのリスト
        ndof_total: 元の全体自由度数
        ndof_per_node: 1節点あたりのDOF数（デフォルト6）
    """

    mpc_groups: list[MPCGroup]
    ndof_total: int
    ndof_per_node: int = 6


@dataclass(frozen=True)
class MPCEliminationResult:
    """MPC消去の出力.

    Attributes:
        T: 変換行列 (ndof_total, ndof_reduced) — CSC疎行列
        slave_dofs: 消去されるslave DOFの配列
        master_dofs: master DOFの配列
        independent_dofs: 独立（非slave）DOFの配列
        ndof_reduced: 縮退後のDOF数
    """

    T: sp.csc_matrix
    slave_dofs: np.ndarray
    master_dofs: np.ndarray
    independent_dofs: np.ndarray
    ndof_reduced: int


# ====================================================================
# 変換行列構築
# ====================================================================


def _build_mpc_transform(
    mpc_groups: list[MPCGroup],
    ndof_total: int,
    ndof_per_node: int = 6,
) -> MPCEliminationResult:
    """MPC制約のDOF消去変換行列 T を構築する.

    全体DOFを独立DOF（master + free）と従属DOF（slave）に分離し、
    u_full = T @ u_reduced を表現する変換行列を構築。

    T の構造:
    - 独立DOF i の行: T[i, j] = δ_{i,j}（単位行列の対応列）
    - slave DOF の行: 対応するmaster DOFの列に剛体結合係数

    Parameters:
        mpc_groups: MPCグループのリスト
        ndof_total: 元の全体自由度数
        ndof_per_node: 1節点あたりのDOF数

    Returns:
        MPCEliminationResult
    """
    if ndof_per_node != 6:
        raise ValueError(f"ndof_per_node={ndof_per_node} は未対応。6DOF/nodeのみ。")

    # 全slave DOFを収集
    all_slave_dofs = []
    all_master_dofs = []
    for grp in mpc_groups:
        m_start = grp.master_node * ndof_per_node
        all_master_dofs.extend(range(m_start, m_start + ndof_per_node))
        for sn in grp.slave_nodes:
            s_start = sn * ndof_per_node
            all_slave_dofs.extend(range(s_start, s_start + ndof_per_node))

    slave_dofs = np.unique(np.array(all_slave_dofs, dtype=int))
    master_dofs = np.unique(np.array(all_master_dofs, dtype=int))

    # 独立DOF = 全DOF - slave DOF
    all_dofs = np.arange(ndof_total, dtype=int)
    slave_set = set(slave_dofs.tolist())
    independent_dofs = np.array([d for d in all_dofs if d not in slave_set], dtype=int)
    ndof_reduced = len(independent_dofs)

    # independent DOF → reduced index のマッピング
    indep_to_reduced = np.full(ndof_total, -1, dtype=int)
    for j, d in enumerate(independent_dofs):
        indep_to_reduced[d] = j

    # T行列をCOO形式で構築
    rows = []
    cols = []
    vals = []

    # 1. 独立DOF: T[d, j] = 1.0（単位写像）
    for j, d in enumerate(independent_dofs):
        rows.append(d)
        cols.append(j)
        vals.append(1.0)

    # 2. slave DOF: 剛体結合
    for grp in mpc_groups:
        m_start = grp.master_node * ndof_per_node
        # master DOFのreduced index
        m_trans_cols = [indep_to_reduced[m_start + k] for k in range(3)]
        m_rot_cols = [indep_to_reduced[m_start + 3 + k] for k in range(3)]

        for i_s, sn in enumerate(grp.slave_nodes):
            s_start = sn * ndof_per_node
            r = grp.slave_coords[i_s] - grp.master_coord  # 相対位置ベクトル
            S = _skew(r)  # [r]×

            # 並進DOF: u_slave = u_master + [r]× θ_master
            for a in range(3):
                # u_master の寄与: δ_{a,a}
                rows.append(s_start + a)
                cols.append(m_trans_cols[a])
                vals.append(1.0)
                # θ_master の寄与: S[a, b]
                for b in range(3):
                    if abs(S[a, b]) > 1e-30:
                        rows.append(s_start + a)
                        cols.append(m_rot_cols[b])
                        vals.append(S[a, b])

            # 回転DOF: θ_slave = θ_master
            for a in range(3):
                rows.append(s_start + 3 + a)
                cols.append(m_rot_cols[a])
                vals.append(1.0)

    T = sp.coo_matrix(
        (np.array(vals), (np.array(rows, dtype=int), np.array(cols, dtype=int))),
        shape=(ndof_total, ndof_reduced),
    ).tocsc()

    return MPCEliminationResult(
        T=T,
        slave_dofs=slave_dofs,
        master_dofs=master_dofs,
        independent_dofs=independent_dofs,
        ndof_reduced=ndof_reduced,
    )


def rebuild_mpc_transform(
    mpc_groups: list[MPCGroup],
    node_coords: np.ndarray,
    ndof_total: int,
    ndof_per_node: int = 6,
) -> MPCEliminationResult:
    """現在のノード座標からMPC変換行列 T を再構築する.

    UL参照配置更新後にTを再構築するためのユーティリティ。
    各MPCGroupの slave_coords/master_coord を node_coords から取得し、
    更新後の相対位置ベクトルでTを構築する。

    status-283: 大回転時のMPC線形化破綻対策。

    Parameters:
        mpc_groups: 元のMPCグループリスト
        node_coords: 現在のノード座標 (n_nodes, 3)
        ndof_total: 全体自由度数
        ndof_per_node: 1節点あたりのDOF数
    """
    updated_groups = []
    for grp in mpc_groups:
        updated_groups.append(
            MPCGroup(
                master_node=grp.master_node,
                slave_nodes=grp.slave_nodes,
                slave_coords=node_coords[grp.slave_nodes],
                master_coord=node_coords[grp.master_node],
            )
        )
    return _build_mpc_transform(updated_groups, ndof_total, ndof_per_node)


# ====================================================================
# Process
# ====================================================================


class MPCEliminationProcess(
    PreProcess[MPCEliminationConfig, MPCEliminationResult],
):
    """MPC DOF消去変換行列を構築する PreProcess.

    梁端部のslave節点群をmaster参照点に剛体結合する変換行列 T を生成。
    後段の LinearSolveProcess で T^T K T の縮退系を求解する。
    """

    meta = ProcessMeta(
        name="MPCElimination",
        module="pre",
        version="1.0.0",
        document_path="../../docs/constraints.md",
    )

    def process(self, input_data: MPCEliminationConfig) -> MPCEliminationResult:
        """MPC変換行列を構築."""
        return _build_mpc_transform(
            mpc_groups=input_data.mpc_groups,
            ndof_total=input_data.ndof_total,
            ndof_per_node=input_data.ndof_per_node,
        )
