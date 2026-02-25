"""要素の抽象インタフェース定義.

Protocol 階層:
  ElementProtocol              — 最小限（local_stiffness + dof_indices）。線形アセンブリで十分。
  NonlinearElementProtocol     — 幾何学的/材料非線形用。internal_force + tangent_stiffness。
  DynamicElementProtocol       — 動解析用。mass_matrix。

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
    """有限要素の共通インタフェース（線形弾性アセンブリ用）.

    Attributes:
        ndof_per_node: 1節点あたりの自由度数（平面問題=2, 梁=3or6, 3D固体=3, etc.）
        nnodes: 要素の節点数（Q4=4, TRI3=3, TRI6=6, HEX8=8, Beam=2, etc.）
        ndof: 要素あたりの総自由度数（= ndof_per_node * nnodes）

    適合クラス例:
      - Quad4PlaneStrain, Tri3PlaneStrain, Tri6PlaneStrain  (2D連続体)
      - Hex8BBar                                             (3D連続体)
      - EulerBernoulliBeam2D, TimoshenkoBeam2D              (2D梁)
      - TimoshenkoBeam3D, CosseratRod                       (3D梁)
    """

    ndof_per_node: int
    nnodes: int
    ndof: int

    def local_stiffness(
        self,
        coords: np.ndarray,
        material: ConstitutiveProtocol,
        thickness: float | None = None,
    ) -> np.ndarray:
        """局所剛性行列を計算する.

        Args:
            coords: 要素節点座標 (nnodes, ndim)
            material: 構成則オブジェクト
            thickness: 厚み（平面要素用）。梁・3D固体要素では None。

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


@runtime_checkable
class NonlinearElementProtocol(ElementProtocol, Protocol):
    """幾何学的/材料非線形解析に対応する要素のインタフェース.

    ElementProtocol を拡張し、Newton-Raphson 反復に必要な
    内力ベクトルと接線剛性行列の計算を追加する。

    適合クラス例:
      - CosseratRod（nonlinear=True 時）
      - Q4 TL/UL 連続体要素（continuum_nl.py 経由、関数ベース）

    Note:
      幾何剛性（geometric_stiffness）は要素実装ごとに分離の要否が異なるため
      本 Protocol には含めない。CosseratRod 等は独自に geometric_stiffness() を
      提供し、tangent_stiffness() = K_mat + K_geo として統合する。
    """

    def internal_force(
        self,
        coords: np.ndarray,
        u_elem: np.ndarray,
        material: ConstitutiveProtocol,
    ) -> np.ndarray:
        """要素の内力ベクトルを計算する.

        f_int = ∫ Bᵀ σ dV （連続体）
        f_int = Tᵀ ∫ Bᵀ σ ds （梁）

        Args:
            coords: (nnodes, ndim) 節点座標
            u_elem: (ndof,) 要素変位ベクトル（全体座標系）
            material: 構成則

        Returns:
            f_int: (ndof,) 内力ベクトル（全体座標系）
        """
        ...

    def tangent_stiffness(
        self,
        coords: np.ndarray,
        u_elem: np.ndarray,
        material: ConstitutiveProtocol,
    ) -> np.ndarray:
        """接線剛性行列を計算する（Newton-Raphson 反復用）.

        K_T = K_material + K_geometric（幾何剛性を含む合計接線剛性）

        Args:
            coords: (nnodes, ndim) 節点座標
            u_elem: (ndof,) 要素変位ベクトル（全体座標系）
            material: 構成則

        Returns:
            K_T: (ndof, ndof) 接線剛性行列（全体座標系）
        """
        ...


@runtime_checkable
class DynamicElementProtocol(ElementProtocol, Protocol):
    """動解析に対応する要素のインタフェース.

    ElementProtocol を拡張し、質量行列の計算を追加する。

    適合クラス例:
      - EulerBernoulliBeam2D（mass_matrix 実装済み）
      - TimoshenkoBeam2D（mass_matrix 実装済み）
      - TimoshenkoBeam3D（mass_matrix 実装済み）
    """

    def mass_matrix(
        self,
        coords: np.ndarray,
        rho: float,
        *,
        lumped: bool = False,
    ) -> np.ndarray:
        """質量行列を計算する.

        Args:
            coords: (nnodes, ndim) 節点座標
            rho: 密度 [kg/m³]
            lumped: True の場合は集中質量行列（HRZ法等）、
                    False の場合は整合質量行列

        Returns:
            Me: (ndof, ndof) 質量行列
        """
        ...
