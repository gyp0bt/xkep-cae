"""RigidEdgeAssemblerProcess — 剛体エッジアセンブラ生成 Process.

剛体ジグ辺用のダミーアセンブラ（剛性ゼロ、座標のみ保持）を生成する。
HEX8 要素の剛性行列が負定値対角を持つ問題を回避するため、
ジグ節点の座標管理のみ行い、剛性・内力はゼロを返す。
ジグ DOF は境界条件で全固定し、接触検出のみに参加させる。

status-249: _RigidEdgeAssembler の Process 化（脱法修正）。

[← README](../../README.md)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.core import PreProcess, ProcessMeta

# ====================================================================
# アセンブラ本体（内部実装）
# ====================================================================


class _RigidEdgeAssembler:
    """剛体ジグ辺用ダミーアセンブラ（剛性ゼロ、座標のみ保持）.

    assemble_tangent / assemble_internal_force はゼロを返す。
    update_reference で座標と累積変位のみ更新する。
    """

    def __init__(
        self,
        jig_coords: np.ndarray,
        n_jig_nodes: int,
        global_node_offset: int,
        total_ndof: int,
    ) -> None:
        self.coords_ref = jig_coords.copy()
        self._n_jig = n_jig_nodes
        self._offset = global_node_offset
        self._total_ndof = total_ndof
        self._u_total_accum = np.zeros(total_ndof)
        self._ckpt_coords_ref: np.ndarray | None = None
        self._ckpt_u_total_accum: np.ndarray | None = None

    @property
    def ndof(self) -> int:
        return self._total_ndof

    def assemble_tangent(self, u: np.ndarray) -> sp.csr_matrix:
        return sp.csr_matrix((self._total_ndof, self._total_ndof))

    def assemble_internal_force(self, u: np.ndarray) -> np.ndarray:
        return np.zeros(self._total_ndof)

    def update_reference(self, u_incr: np.ndarray) -> None:
        for i in range(self._n_jig):
            gn = self._offset + i
            self.coords_ref[i] += u_incr[6 * gn : 6 * gn + 3]
        self._u_total_accum += u_incr

    def checkpoint(self) -> None:
        self._ckpt_coords_ref = self.coords_ref.copy()
        self._ckpt_u_total_accum = self._u_total_accum.copy()

    def rollback(self) -> None:
        if self._ckpt_coords_ref is not None:
            self.coords_ref = self._ckpt_coords_ref.copy()
            self._u_total_accum = self._ckpt_u_total_accum.copy()

    @property
    def u_total_accum(self) -> np.ndarray:
        return self._u_total_accum


# ====================================================================
# Process I/O データ
# ====================================================================


@dataclass(frozen=True)
class RigidEdgeAssemblerConfig:
    """剛体エッジアセンブラ生成の入力.

    Attributes:
        jig_coords: ジグ節点座標 (n_jig_nodes, 3)
        n_jig_nodes: ジグ節点数
        global_node_offset: 全体節点番号のオフセット（梁節点数）
        total_ndof: 全体自由度数
    """

    jig_coords: np.ndarray
    n_jig_nodes: int
    global_node_offset: int
    total_ndof: int


@dataclass(frozen=True)
class RigidEdgeAssemblerResult:
    """剛体エッジアセンブラ生成の出力.

    Attributes:
        assembler: 生成された剛体エッジアセンブラインスタンス
        n_jig_nodes: ジグ節点数
        total_ndof: 全体自由度数
    """

    assembler: object  # _RigidEdgeAssembler
    n_jig_nodes: int
    total_ndof: int


# ====================================================================
# Process
# ====================================================================


class RigidEdgeAssemblerProcess(
    PreProcess[RigidEdgeAssemblerConfig, RigidEdgeAssemblerResult],
):
    """剛体エッジアセンブラ生成 Process.

    剛体ジグ辺用のダミーアセンブラを生成する。
    剛性・内力はゼロを返し、座標管理のみ行う。
    """

    meta = ProcessMeta(
        name="RigidEdgeAssembler",
        module="pre",
        version="1.0.0",
        document_path="../../docs/constraints.md",
    )

    def process(self, input_data: RigidEdgeAssemblerConfig) -> RigidEdgeAssemblerResult:
        """剛体エッジアセンブラを生成."""
        assembler = _RigidEdgeAssembler(
            jig_coords=input_data.jig_coords,
            n_jig_nodes=input_data.n_jig_nodes,
            global_node_offset=input_data.global_node_offset,
            total_ndof=input_data.total_ndof,
        )
        return RigidEdgeAssemblerResult(
            assembler=assembler,
            n_jig_nodes=input_data.n_jig_nodes,
            total_ndof=input_data.total_ndof,
        )
