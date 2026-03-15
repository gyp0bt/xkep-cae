"""接触アセンブリ用ユーティリティ."""

from __future__ import annotations

import numpy as np


def _contact_dofs(pair: object, ndof_per_node: int = 6) -> np.ndarray:
    """接触ペアに関与する全体 DOF インデックスを返す.

    4節点（A0, A1, B0, B1）× ndof_per_node の DOF を返す。

    Args:
        pair: ContactPair（nodes_a, nodes_b 属性を持つ）
        ndof_per_node: 1節点あたりの DOF 数

    Returns:
        dofs: (4 * ndof_per_node,) 全体DOFインデックス
    """
    nodes = np.array(
        [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]],
        dtype=int,
    )
    offsets = np.arange(ndof_per_node, dtype=int)
    return (nodes[:, None] * ndof_per_node + offsets).ravel()
