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
