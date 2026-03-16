"""要素の抽象インタフェース定義.

Protocol 階層:
  ElementProtocol              — 最小限（local_stiffness + dof_indices）。
  NonlinearElementProtocol     — 幾何学的/材料非線形用。
  DynamicElementProtocol       — 動解析用。mass_matrix。

[← README](../../../README.md)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from xkep_cae.core.constitutive import ConstitutiveProtocol


@runtime_checkable
class ElementProtocol(Protocol):
    """有限要素の共通インタフェース（線形弾性アセンブリ用）.

    Attributes:
        ndof_per_node: 1節点あたりの自由度数
        nnodes: 要素の節点数
        ndof: 要素あたりの総自由度数
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
        """局所剛性行列を計算する."""
        ...

    def dof_indices(self, node_indices: np.ndarray) -> np.ndarray:
        """グローバル節点インデックスから要素DOFインデックスを返す."""
        ...


@runtime_checkable
class NonlinearElementProtocol(ElementProtocol, Protocol):
    """幾何学的/材料非線形解析に対応する要素のインタフェース."""

    def internal_force(
        self,
        coords: np.ndarray,
        u_elem: np.ndarray,
        material: ConstitutiveProtocol,
    ) -> np.ndarray:
        """要素の内力ベクトルを計算する."""
        ...

    def tangent_stiffness(
        self,
        coords: np.ndarray,
        u_elem: np.ndarray,
        material: ConstitutiveProtocol,
    ) -> np.ndarray:
        """接線剛性行列を計算する."""
        ...


@runtime_checkable
class DynamicElementProtocol(ElementProtocol, Protocol):
    """動解析に対応する要素のインタフェース."""

    def mass_matrix(
        self,
        coords: np.ndarray,
        rho: float,
        *,
        lumped: bool = False,
    ) -> np.ndarray:
        """質量行列を計算する."""
        ...
