"""初期条件（Initial Conditions）.

Abaqus の ``*INITIAL CONDITIONS`` キーワードに相当する初期条件を定義する。
動解析のソルバーに渡す初期変位・初期速度ベクトルを構築する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class InitialConditionType(Enum):
    """初期条件の種別."""

    VELOCITY = "velocity"
    DISPLACEMENT = "displacement"


@dataclass
class InitialConditionEntry:
    """1つの初期条件エントリ.

    Attributes:
        type: 初期条件種別
        node_indices: 対象節点のインデックス配列
        dof: 自由度番号（0始まり、節点ローカル）
        value: 初期値
    """

    type: InitialConditionType
    node_indices: np.ndarray | list[int]
    dof: int
    value: float


@dataclass
class InitialConditions:
    """初期条件の集合.

    複数の初期条件エントリを保持し、ソルバー用の初期ベクトルを構築する。

    Attributes:
        entries: 初期条件エントリのリスト
    """

    entries: list[InitialConditionEntry] = field(default_factory=list)

    def add(
        self,
        type: str,  # noqa: A002
        node_indices: np.ndarray | list[int],
        dof: int,
        value: float,
    ) -> None:
        """初期条件エントリを追加する.

        Args:
            type: 初期条件種別 ("velocity" or "displacement")
            node_indices: 対象節点のインデックス配列
            dof: 自由度番号（0始まり、節点ローカル）
            value: 初期値
        """
        ic_type = InitialConditionType(type)
        self.entries.append(
            InitialConditionEntry(
                type=ic_type,
                node_indices=np.asarray(node_indices, dtype=int),
                dof=dof,
                value=value,
            )
        )

    def build_initial_vectors(
        self,
        ndof_total: int,
        ndof_per_node: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """初期変位・初期速度ベクトルを構築する.

        Args:
            ndof_total: 全自由度数
            ndof_per_node: 1節点あたりの自由度数

        Returns:
            (u0, v0): 初期変位ベクトル (ndof_total,)、初期速度ベクトル (ndof_total,)
        """
        u0 = np.zeros(ndof_total, dtype=float)
        v0 = np.zeros(ndof_total, dtype=float)

        for entry in self.entries:
            nodes = np.asarray(entry.node_indices, dtype=int)
            global_dofs = nodes * ndof_per_node + entry.dof

            # 範囲チェック
            if np.any(global_dofs >= ndof_total) or np.any(global_dofs < 0):
                raise ValueError(
                    f"DOF インデックスが範囲外: nodes={nodes}, dof={entry.dof}, "
                    f"ndof_total={ndof_total}"
                )

            if entry.type == InitialConditionType.VELOCITY:
                v0[global_dofs] = entry.value
            elif entry.type == InitialConditionType.DISPLACEMENT:
                u0[global_dofs] = entry.value

        return u0, v0


__all__ = [
    "InitialConditionType",
    "InitialConditionEntry",
    "InitialConditions",
]
