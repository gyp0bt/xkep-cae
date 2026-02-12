"""構成則（材料モデル）の抽象インタフェース定義.

将来の拡張:
  - 弾塑性: tangent() が接線剛性と応力を返す
  - 粘弾性: 内部変数をstate経由で管理
  - NNサロゲート: PyTorchモデルをラップして同じインタフェースで使用
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ConstitutiveProtocol(Protocol):
    """構成則の共通インタフェース.

    線形弾性の場合: tangent() は定数テンソル D と応力 sigma = D @ strain を返す。
    非線形の場合: tangent() は現在のひずみ状態に依存した接線テンソルと応力を返す。
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
