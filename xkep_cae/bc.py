from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse import SparseEfficiencyWarning

from xkep_cae.core.results import DirichletResult

warnings.simplefilter("ignore", SparseEfficiencyWarning)


def apply_dirichlet(
    K: sp.csr_matrix,
    f: np.ndarray,
    fixed_dofs: np.ndarray,
    values: float | np.ndarray = 0.0,
) -> DirichletResult:
    """Dirichlet境界条件（行・列消去＋右辺補正）を適用する.

    アルゴリズム（元のK, fから）:
      1) f <- f - K[:, fixed_dofs] @ values  （元のKで一括補正）
      2) K[:, fixed_dofs] = 0, K[fixed_dofs, :] = 0
      3) K[d,d] = 1, f[d] = val

    Note:
        右辺補正は元のK行列（変更前）を使って一括計算する。
        以前の実装ではCSR行列をループ内で順次変更していたため、
        非ゼロ規定変位でスパース構造の不整合が発生していた。

    Args:
        K: CSR剛性行列
        f: 右辺ベクトル (n,)
        fixed_dofs: 拘束するDOFの配列
        values: 拘束変位値（スカラー or 同長配列）

    Returns:
        DirichletResult: (K, f) の NamedTuple。拘束適用後の剛性行列と右辺ベクトル。
    """
    fbc = f.astype(float, copy=True)

    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    if np.isscalar(values):
        values = np.full_like(fixed_dofs, float(values), dtype=float)
    else:
        values = np.asarray(values, dtype=float)

    n = K.shape[0]
    if fbc.shape[0] != n:
        raise ValueError("K と f のサイズが一致していません。")

    # 1) 元のK行列（変更前）を使って右辺補正を一括実行
    #    f <- f - K[:, fixed_dofs] @ values
    nonzero_mask = values != 0.0
    if np.any(nonzero_mask):
        nz_dofs = fixed_dofs[nonzero_mask]
        nz_vals = values[nonzero_mask]
        K_csc = K.tocsc()
        fbc -= K_csc[:, nz_dofs] @ nz_vals

    # 2) LIL形式で行・列ゼロクリア（行・列両方の変更が効率的）
    Kbc = K.tolil(copy=True)
    for dof in fixed_dofs:
        Kbc[dof, :] = 0.0
        Kbc[:, dof] = 0.0
        Kbc[dof, dof] = 1.0

    # 3) 固定DOFの右辺を規定値に設定
    fbc[fixed_dofs] = values

    return DirichletResult(K=Kbc.tocsr(), f=fbc)


def apply_dirichlet_penalty(
    K: sp.csr_matrix,
    f: np.ndarray,
    fixed_dofs: np.ndarray,
    values: float | np.ndarray = 0.0,
    penalty: float = 1.0e20,
) -> DirichletResult:
    """Penalty法でDirichlet境界条件を課す軽量版.

    目的:
        - CSR行列のスパース構造を壊さない
        - 行列の行や列をゼロクリアしない（高コスト操作を避ける）
        - 対角成分に大きな値を足して、u[dof] ≈ values に強制する

    Args:
        K:
            元の剛性行列 (CSR).
        f:
            元の右辺ベクトル.
        fixed_dofs:
            拘束するDOFインデックス配列 (int配列).
            例: [0, 1, 10, 11, ...]
        values:
            各DOFに対応する拘束値。
            スカラーなら全DOFで同じ値。
            配列なら len(values) == len(fixed_dofs) を想定。
        penalty:
            対角に足すペナルティ係数。大きいほど拘束が強くなる。

    Returns:
        DirichletResult: (K, f) の NamedTuple。Penalty適用後の系。
    """
    if fixed_dofs.size == 0:
        # 拘束なしなら何もしない
        return DirichletResult(K=K, f=f)

    Kbc = K.copy().tocsr()
    fbc = f.copy()

    fixed_dofs = np.asarray(fixed_dofs, dtype=int)

    # values を DOFごとの配列に揃える
    if np.isscalar(values):
        vals = np.full(fixed_dofs.shape, float(values))
    else:
        vals = np.asarray(values, dtype=float)
        if vals.shape[0] != fixed_dofs.shape[0]:
            raise ValueError("values の長さと fixed_dofs の長さが一致していません。")

    # 対角に penalty を足し、右辺にも penalty*val を足す
    # これで u[d] ≈ val になるように強制される
    for dof, val in zip(fixed_dofs, vals, strict=True):
        Kbc[dof, dof] += penalty
        fbc[dof] += penalty * val

    return DirichletResult(K=Kbc, f=fbc)
