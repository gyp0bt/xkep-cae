"""要素の抽象インタフェース定義.

すべての要素型はこのProtocolに適合する必要がある。
Protocol を採用する理由:
  - ABCより柔軟（構造的部分型: 明示的な継承不要）
  - NNサロゲート等、異質な実装にも対応しやすい
  - ランタイムチェックは runtime_checkable で補完
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from xkep_cae.core.constitutive import ConstitutiveProtocol


@runtime_checkable
class ElementProtocol(Protocol):
    """有限要素の共通インタフェース.

    Attributes:
        ndof_per_node: 1節点あたりの自由度数（平面問題=2, 梁=3or6, etc.）
        nnodes: 要素の節点数（Q4=4, TRI3=3, TRI6=6, etc.）
        ndof: 要素あたりの総自由度数（= ndof_per_node * nnodes）
    """

    ndof_per_node: int
    nnodes: int
    ndof: int

    def local_stiffness(
        self,
        coords: np.ndarray,
        material: ConstitutiveProtocol,
        thickness: float,
    ) -> np.ndarray:
        """局所剛性行列を計算する.

        Args:
            coords: 要素節点座標 (nnodes, ndim)
            material: 構成則オブジェクト
            thickness: 厚み（平面要素用）

        Returns:
            Ke: (ndof, ndof) 局所剛性行列
        """
        ...

    def dof_indices(self, node_indices: np.ndarray) -> np.ndarray:
        """グローバル節点インデックスから要素DOFインデックスを返す.

        Args:
            node_indices: (nnodes,) グローバル節点インデックス

        Returns:
            edofs: (ndof,) グローバルDOFインデックス
        """
        ...
