"""梁要素用の1D弾性構成則."""

from __future__ import annotations

import numpy as np


class BeamElastic1D:
    """梁要素用の1D線形弾性構成則（ConstitutiveProtocol適合）.

    tangent() はヤング率 E をスカラーとして返す。

    Args:
        E: ヤング率
        nu: ポアソン比（せん断弾性率 G の計算用、Timoshenko梁で使用）
    """

    def __init__(self, E: float, nu: float = 0.3) -> None:
        if E <= 0:
            raise ValueError(f"ヤング率 E は正値でなければなりません: {E}")
        self.E = E
        self.nu = nu

    @property
    def G(self) -> float:
        """せん断弾性率."""
        return self.E / (2.0 * (1.0 + self.nu))

    def tangent(self, strain: np.ndarray | None = None) -> np.ndarray:
        """弾性テンソル（1D: スカラー相当）を返す."""
        return np.array([[self.E]])
