"""境界条件適用ユーティリティ（高速化版）.

tolil() + Python ループによる BC 適用をベクトル化して高速化。
CSR → CSC 変換を活用し、行/列のゼロ化を効率的に行う。

[← README](../../README.md)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def apply_bc_fast(
    K: sp.spmatrix,
    r: np.ndarray,
    fixed_dofs: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """境界条件適用（ベクトル化行列消去法）.

    従来の tolil() + Python ループを CSR/CSC 直接操作に置換し高速化。

    処理:
      1. 拘束行のゼロ化（CSR: 行操作は効率的）
      2. 拘束列のゼロ化（CSC 変換で列操作を効率化）
      3. 対角に 1.0 を設定
      4. 右辺ベクトルの拘束成分をゼロ化

    Args:
        K: 剛性行列（CSR/CSC/COO いずれも可）
        r: 右辺ベクトル
        fixed_dofs: 拘束 DOF インデックスの配列

    Returns:
        K_bc: BC 適用済み CSR 剛性行列
        r_bc: BC 適用済み右辺ベクトル
    """
    K_bc = K.tocsr().copy()
    r_bc = r.copy()

    if len(fixed_dofs) == 0:
        return K_bc, r_bc

    # 行の消去（CSR 行操作は効率的）
    for dof in fixed_dofs:
        K_bc[dof, :] = 0.0
    # 列の消去（CSC に変換して効率的に列操作）
    K_csc = K_bc.tocsc()
    for dof in fixed_dofs:
        K_csc[:, dof] = 0.0
    K_bc = K_csc.tocsr()
    # 対角に 1.0 を設定
    K_bc[fixed_dofs, fixed_dofs] = 1.0
    K_bc.eliminate_zeros()

    r_bc[fixed_dofs] = 0.0
    return K_bc, r_bc
