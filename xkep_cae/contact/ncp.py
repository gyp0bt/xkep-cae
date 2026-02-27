"""NCP (Nonlinear Complementarity Problem) 関数 — Phase C6-L3.

接触条件を相補性問題として統一的に扱う NCP 関数群。
Semi-smooth Newton で使用し、Outer loop を廃止する。

相補性条件:
    g_n >= 0  (非貫入)
    lambda_n >= 0  (圧縮のみ)
    g_n * lambda_n = 0  (相補性)

Fischer-Burmeister (FB) NCP 関数:
    C_FB(a, b) = sqrt(a^2 + b^2) - a - b = 0
    <==> a >= 0, b >= 0, a*b = 0

正則化 FB 関数:
    C_FB_reg(a, b) = sqrt(a^2 + b^2 + reg) - a - b

参考文献:
- Alart, Curnier (1991): "A mixed formulation for frictional contact
  problems prone to Newton like solution methods" CMAME
- Fischer (1992): "A special Newton-type optimization method"
- De Luca, Facchinei, Kanzow (1996): "A semismooth equation approach
  to the solution of nonlinear complementarity problems"

設計仕様: docs/contact/contact-algorithm-overhaul-c6.md §5
"""

from __future__ import annotations

import math

import numpy as np


def ncp_fischer_burmeister(
    g: float,
    lam: float,
    *,
    reg: float = 1e-12,
) -> tuple[float, float, float]:
    """Fischer-Burmeister NCP 関数とその一般化微分.

    C_FB(g, lam) = sqrt(g^2 + lam^2 + reg) - g - lam

    一般化ヤコビアン:
        dC/dg   = g / sqrt(g^2 + lam^2 + reg) - 1
        dC/dlam = lam / sqrt(g^2 + lam^2 + reg) - 1

    Args:
        g: ギャップ値（g >= 0 が非貫入条件）
        lam: ラグランジュ乗数（lam >= 0 が圧縮条件）
        reg: 正則化パラメータ（原点での特異性回避）

    Returns:
        fb: NCP 関数値
        dg: ∂C/∂g
        dlam: ∂C/∂lam
    """
    norm = math.sqrt(g * g + lam * lam + reg)
    fb = norm - g - lam
    dg = g / norm - 1.0
    dlam = lam / norm - 1.0
    return fb, dg, dlam


def ncp_min(
    g: float,
    lam: float,
) -> tuple[float, float, float]:
    """min 関数ベースの NCP 関数.

    C_min(g, lam) = min(g, lam)

    Semi-smooth: min の非微分点で一般化微分を使用。

    Args:
        g: ギャップ値
        lam: ラグランジュ乗数

    Returns:
        c: NCP 関数値
        dg: ∂C/∂g（一般化微分）
        dlam: ∂C/∂lam（一般化微分）
    """
    if g < lam:
        return g, 1.0, 0.0
    elif g > lam:
        return lam, 0.0, 1.0
    else:
        # g == lam: 一般化微分（どちらも 0.5 とする）
        return g, 0.5, 0.5


def evaluate_ncp_residual(
    gaps: np.ndarray,
    lambdas: np.ndarray,
    *,
    reg: float = 1e-12,
    ncp_type: str = "fb",
) -> np.ndarray:
    """全接触ペアの NCP 残差ベクトルを計算する.

    Args:
        gaps: (n_contact,) 各ペアのギャップ
        lambdas: (n_contact,) 各ペアのラグランジュ乗数
        reg: FB 関数の正則化パラメータ
        ncp_type: "fb" (Fischer-Burmeister) or "min"

    Returns:
        C: (n_contact,) NCP 残差ベクトル
    """
    n = len(gaps)
    C = np.empty(n)
    ncp_func = ncp_fischer_burmeister if ncp_type == "fb" else ncp_min
    for i in range(n):
        if ncp_type == "fb":
            C[i], _, _ = ncp_func(gaps[i], lambdas[i], reg=reg)
        else:
            C[i], _, _ = ncp_func(gaps[i], lambdas[i])
    return C


def evaluate_ncp_jacobian(
    gaps: np.ndarray,
    lambdas: np.ndarray,
    *,
    reg: float = 1e-12,
    ncp_type: str = "fb",
) -> tuple[np.ndarray, np.ndarray]:
    """全接触ペアの NCP ヤコビアン（対角）を計算する.

    NCP 関数は各ペアで独立なので、ヤコビアンは対角行列。

    Args:
        gaps: (n_contact,) 各ペアのギャップ
        lambdas: (n_contact,) 各ペアのラグランジュ乗数
        reg: FB 関数の正則化パラメータ
        ncp_type: "fb" (Fischer-Burmeister) or "min"

    Returns:
        dC_dg: (n_contact,) ∂C_i/∂g_i
        dC_dlam: (n_contact,) ∂C_i/∂λ_i
    """
    n = len(gaps)
    dC_dg = np.empty(n)
    dC_dlam = np.empty(n)
    ncp_func = ncp_fischer_burmeister if ncp_type == "fb" else ncp_min
    for i in range(n):
        if ncp_type == "fb":
            _, dC_dg[i], dC_dlam[i] = ncp_func(gaps[i], lambdas[i], reg=reg)
        else:
            _, dC_dg[i], dC_dlam[i] = ncp_func(gaps[i], lambdas[i])
    return dC_dg, dC_dlam


def compute_gap_jacobian_wrt_u(
    pair_nodes_a: np.ndarray,
    pair_nodes_b: np.ndarray,
    s: float,
    t: float,
    normal: np.ndarray,
    ndof_per_node: int = 6,
) -> np.ndarray:
    """ギャップの変位に対する微分 ∂g_n/∂u を計算する.

    g_n = ||pA - pB|| - (rA + rB)

    ∂g_n/∂u = n^T · ∂(pA - pB)/∂u
            = n^T · [(1-s)·I, s·I, -(1-t)·I, -t·I]  (各並進DOF)

    これは法線形状ベクトル g_n の符号反転（A→B で定義）に対応。

    Args:
        pair_nodes_a: (2,) セグメントA の節点インデックス
        pair_nodes_b: (2,) セグメントB の節点インデックス
        s: セグメントA上のパラメータ
        t: セグメントB上のパラメータ
        normal: (3,) A→B 法線ベクトル
        ndof_per_node: 1節点あたりの DOF 数

    Returns:
        dg_du: (ndof_total_local,) ギャップの変位微分
            ここで ndof_total_local = 4 * ndof_per_node
    """
    ndof_local = 4 * ndof_per_node
    dg_du = np.zeros(ndof_local)

    # A 側: pA = (1-s)*xA0 + s*xA1 → ∂pA/∂uA0 = (1-s)*I, ∂pA/∂uA1 = s*I
    # B 側: pB = (1-t)*xB0 + t*xB1 → ∂pB/∂uB0 = (1-t)*I, ∂pB/∂uB1 = t*I
    # ∂g/∂u = n^T · ∂(pA-pB)/∂u
    # = n^T · [(1-s)*I, s*I, -(1-t)*I, -t*I]

    # A0
    dg_du[0:3] = (1.0 - s) * normal
    # A1
    dg_du[ndof_per_node : ndof_per_node + 3] = s * normal
    # B0
    dg_du[2 * ndof_per_node : 2 * ndof_per_node + 3] = -(1.0 - t) * normal
    # B1
    dg_du[3 * ndof_per_node : 3 * ndof_per_node + 3] = -t * normal

    return dg_du


def build_augmented_residual(
    R_u: np.ndarray,
    gaps: np.ndarray,
    lambdas: np.ndarray,
    *,
    reg: float = 1e-12,
    ncp_type: str = "fb",
) -> np.ndarray:
    """拡大系の残差ベクトル [R_u; C_ncp] を構築する.

    Args:
        R_u: (ndof,) 力の釣り合い残差
        gaps: (n_contact,) 各ペアのギャップ
        lambdas: (n_contact,) 各ペアのラグランジュ乗数
        reg: FB 関数の正則化パラメータ
        ncp_type: "fb" or "min"

    Returns:
        residual: (ndof + n_contact,) 拡大残差ベクトル
    """
    C_ncp = evaluate_ncp_residual(gaps, lambdas, reg=reg, ncp_type=ncp_type)
    return np.concatenate([R_u, C_ncp])
