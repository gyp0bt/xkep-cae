"""構成則（材料モデル）の抽象インタフェース定義.

Protocol 定義:
  ConstitutiveProtocol          — 線形弾性用（tangent のみ）。
  PlasticConstitutiveProtocol   — 弾塑性用（return_mapping で応力・接線・状態を一括更新）。
  両者は独立。塑性材料は ConstitutiveProtocol を満たさなくてもよい。

[← README](../../../README.md)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ConstitutiveProtocol(Protocol):
    """構成則の共通インタフェース.

    線形弾性の場合: tangent() は定数テンソル D を返す。応力は sigma = D @ strain。
    梁要素用 1D の場合: tangent() は (1,1) 配列またはスカラーを返す。
    """

    def tangent(self, strain: np.ndarray | None = None) -> np.ndarray:
        """弾性/接線剛性テンソルを返す."""
        ...


@runtime_checkable
class PlasticConstitutiveProtocol(Protocol):
    """弾塑性構成則のインタフェース.

    ConstitutiveProtocol とは独立した Protocol。
    塑性構成則は return_mapping() で応力・consistent tangent・更新後の状態を
    一括で返す。
    """

    def return_mapping(
        self,
        strain: np.ndarray | float,
        state: Any,
    ) -> tuple:
        """Return mapping アルゴリズムを実行する."""
        ...
