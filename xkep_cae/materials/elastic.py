from __future__ import annotations

import numpy as np


def constitutive_plane_strain(E: float, nu: float) -> np.ndarray:
    """平面歪みの弾性マトリクス D を返す。

    Args:
        E: ヤング率
        nu: ポアソン比

    Returns:
        D: (3,3) 弾性マトリクス
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return np.array(
        [[lam + 2.0 * mu, lam, 0.0], [lam, lam + 2.0 * mu, 0.0], [0.0, 0.0, mu]],
        dtype=float,
    )


class PlaneStrainElastic:
    """平面ひずみ線形弾性構成則（ConstitutiveProtocol適合）.

    Args:
        E: ヤング率
        nu: ポアソン比
    """

    def __init__(self, E: float, nu: float) -> None:
        self.E = E
        self.nu = nu
        self._D = constitutive_plane_strain(E, nu)

    def tangent(self, strain: np.ndarray | None = None) -> np.ndarray:
        """弾性テンソル D を返す（線形なのでstrainに依存しない）."""
        return self._D


def constitutive_3d(E: float, nu: float) -> np.ndarray:
    """3D 等方弾性テンソル D (6×6) を返す.

    Voigt 表記: σ = [σxx, σyy, σzz, τyz, τxz, τxy]
                ε = [εxx, εyy, εzz, γyz, γxz, γxy]

    Args:
        E: ヤング率
        nu: ポアソン比

    Returns:
        D: (6, 6) 弾性テンソル
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6), dtype=float)
    # 法線成分
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
    D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
    # せん断成分
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


class IsotropicElastic3D:
    """3D 等方線形弾性構成則（ConstitutiveProtocol 適合）.

    Args:
        E: ヤング率
        nu: ポアソン比
    """

    def __init__(self, E: float, nu: float) -> None:
        self.E = E
        self.nu = nu
        self._D = constitutive_3d(E, nu)

    def tangent(self, strain: np.ndarray | None = None) -> np.ndarray:
        """弾性テンソル D (6×6) を返す."""
        return self._D
