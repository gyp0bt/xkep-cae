"""メソッド戻り値の型定義.

各モジュールの公開メソッドが返すデータ構造を NamedTuple で統一的に定義する。
NamedTuple を採用する理由:
  - 名前付きフィールドアクセス（result.u, result.K 等）
  - タプルアンパッキングとの後方互換性（u, info = solve_displacement(...)）
  - 不変（immutable）で安全
  - ReturnMappingResult 等の既存 NamedTuple と一貫性がある
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from xkep_cae.core.state import CosseratFiberPlasticState, CosseratPlasticState


class LinearSolveResult(NamedTuple):
    """線形ソルバーの結果.

    Attributes:
        u: (ndof,) 解ベクトル
        info: ソルバー情報辞書 (method, nit, residual_norm, setup_time, solve_time 等)
    """

    u: np.ndarray
    info: dict[str, Any]


class DirichletResult(NamedTuple):
    """Dirichlet 境界条件適用後の結果.

    Attributes:
        K: 拘束適用後の剛性行列 (CSR)
        f: 拘束適用後の右辺ベクトル (ndof,)
    """

    K: sp.csr_matrix
    f: np.ndarray


class AssemblyResult(NamedTuple):
    """Cosserat rod 弾性アセンブリの結果.

    Attributes:
        K_T: 接線剛性行列 (total_dof, total_dof)。計算しない場合は None。
        f_int: 内力ベクトル (total_dof,)。計算しない場合は None。
    """

    K_T: np.ndarray | None
    f_int: np.ndarray | None


class PlasticAssemblyResult(NamedTuple):
    """弾塑性 Cosserat rod アセンブリの結果.

    Attributes:
        K_T: 接線剛性行列。計算しない場合は None。
        f_int: 内力ベクトル。計算しない場合は None。
        states: 更新後の塑性状態リスト（各積分点に1つ）
    """

    K_T: np.ndarray | None
    f_int: np.ndarray | None
    states: list[CosseratPlasticState]


class FiberAssemblyResult(NamedTuple):
    """ファイバーモデル Cosserat rod アセンブリの結果.

    Attributes:
        K_T: 接線剛性行列。計算しない場合は None。
        f_int: 内力ベクトル。計算しない場合は None。
        states: 更新後のファイバー塑性状態リスト（各積分点に1つ）
    """

    K_T: np.ndarray | None
    f_int: np.ndarray | None
    states: list[CosseratFiberPlasticState]
