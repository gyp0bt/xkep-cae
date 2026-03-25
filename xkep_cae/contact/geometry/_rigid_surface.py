"""解析的剛体表面の梁接触射影.

梁中心線セグメントから解析的剛体円柱表面への最近接点投影を
ベクトル化バッチ計算で行う。離散化不要で C∞ 連続な接触面を実現。

対応表面:
  - RigidCylinderSurfaceInput: 円柱表面（任意軸方向）

[← README](../../../README.md)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RigidCylinderSurfaceInput:
    """解析的剛体円柱表面.

    Attributes:
        center_ref: (3,) 参照配置での円柱中心座標
        radius: 円柱半径
        axis: (3,) 円柱軸方向（単位ベクトル）
    """

    center_ref: np.ndarray
    radius: float
    axis: np.ndarray


def project_beam_to_cylinder_batch(
    xA0: np.ndarray,
    xA1: np.ndarray,
    center: np.ndarray,
    radius: float,
    axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """梁セグメントから解析的円柱表面への最近接点投影（バッチ）.

    各梁セグメント (A0→A1) について、円柱軸への垂直距離を
    最小化するパラメータ s ∈ [0, 1] を解析的に求め、
    gap と法線を計算する。

    Parameters:
        xA0: (N, 3) 梁セグメント始点
        xA1: (N, 3) 梁セグメント終点
        center: (3,) 円柱中心（現在配置）
        radius: 円柱半径
        axis: (3,) 円柱軸方向（単位ベクトル）

    Returns:
        s: (N,) 梁セグメント上のパラメータ [0, 1]
        distance: (N,) 梁中心線–円柱表面間の距離（> 0: 非接触）
        normal: (N, 3) 法線ベクトル（円柱→梁方向、単位ベクトル）
    """
    dA = xA1 - xA0  # (N, 3)
    w0 = xA0 - center  # (N, 3)

    # 軸方向成分を射影して除去 → 垂直面内の問題に帰着
    w0_dot_a = np.einsum("ij,j->i", w0, axis)  # (N,)
    dA_dot_a = np.einsum("ij,j->i", dA, axis)  # (N,)

    w0_perp = w0 - w0_dot_a[:, None] * axis  # (N, 3)
    dA_perp = dA - dA_dot_a[:, None] * axis  # (N, 3)

    # minimize |w0_perp + s * dA_perp|² over s
    # → s = -(w0_perp · dA_perp) / |dA_perp|²
    a_coeff = np.einsum("ij,ij->i", dA_perp, dA_perp)  # |dA_perp|²
    b_coeff = np.einsum("ij,ij->i", w0_perp, dA_perp)  # w0_perp · dA_perp

    s = np.where(a_coeff > 1e-30, -b_coeff / a_coeff, 0.5)
    s = np.clip(s, 0.0, 1.0)

    # 最適 s での垂直成分
    d_perp = w0_perp + s[:, None] * dA_perp  # (N, 3)
    dist_to_axis = np.sqrt(np.einsum("ij,ij->i", d_perp, d_perp))  # (N,)

    # 法線: 円柱軸→梁方向（梁側から見て外向き）
    normal = np.zeros_like(d_perp)
    mask = dist_to_axis > 1e-15
    normal[mask] = d_perp[mask] / dist_to_axis[mask, None]

    # distance: 梁中心線から円柱表面までの距離
    # gap = distance - beam_radius（呼び出し元で半径を減算）
    distance = dist_to_axis - radius

    return s, distance, normal
