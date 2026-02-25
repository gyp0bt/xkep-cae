"""構成則（材料モデル）の抽象インタフェース定義.

Protocol 定義:
  ConstitutiveProtocol          — 線形弾性用（tangent のみ）。
  PlasticConstitutiveProtocol   — 弾塑性用（return_mapping で応力・接線・状態を一括更新）。
  両者は独立。塑性材料は ConstitutiveProtocol を満たさなくてもよい。

将来の拡張:
  - 粘弾性: 内部変数をstate経由で管理
  - NNサロゲート: PyTorchモデルをラップして同じインタフェースで使用
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ConstitutiveProtocol(Protocol):
    """構成則の共通インタフェース.

    線形弾性の場合: tangent() は定数テンソル D を返す。応力は sigma = D @ strain。
    梁要素用 1D の場合: tangent() は (1,1) 配列またはスカラーを返す。

    適合クラス例:
      - PlaneStrainElastic     (3,3)
      - IsotropicElastic3D     (6,6)
      - BeamElastic1D          (1,1)
    """

    def tangent(self, strain: np.ndarray | None = None) -> np.ndarray:
        """弾性/接線剛性テンソルを返す.

        Args:
            strain: ひずみベクトル。線形弾性の場合は不要（None可）。

        Returns:
            D: 弾性/接線剛性テンソル
               平面問題: (3, 3)
               3D: (6, 6)
               1D: (1, 1) or scalar
        """
        ...


@runtime_checkable
class PlasticConstitutiveProtocol(Protocol):
    """弾塑性構成則のインタフェース.

    ConstitutiveProtocol とは独立した Protocol。
    塑性構成則は return_mapping() で応力・consistent tangent・更新後の状態を
    一括で返す。tangent() のような単独の弾性テンソル取得は不要なケースが多い。

    Note:
      ConstitutiveProtocol（線形弾性）とは異なる使われ方をする:
      - 線形弾性: assembly が material.tangent() を呼んで D を取得 → Ke = ∫ BᵀDB dV
      - 弾塑性: assembly が material.return_mapping(strain, state) を呼び、
        応力と consistent tangent を同時に取得

    適合クラス例:
      - Plasticity1D           1D弾塑性（return_mapping → ReturnMappingResult）
      - Plasticity3D           3D von Mises（return_mapping → ReturnMappingResult3D）
    """

    def return_mapping(
        self,
        strain: np.ndarray | float,
        state: Any,
    ) -> tuple:
        """Return mapping アルゴリズムを実行する.

        与えられたひずみと前ステップの塑性状態から、応力・接線剛性・
        更新後の塑性状態を計算する。

        Args:
            strain: 現在のひずみ（1D: float, 3D: ndarray）
            state: 前ステップの塑性状態
                   （PlasticState1D, PlasticState3D 等）

        Returns:
            result: (stress, tangent, state_new) の NamedTuple
                stress: 応力（1D: float, 3D: ndarray）
                tangent: consistent tangent（1D: float, 3D: ndarray）
                state_new: 更新後の塑性状態
        """
        ...
