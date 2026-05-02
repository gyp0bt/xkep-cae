"""SO(3) Lie 群／Lie 代数ユーティリティ.

幾何学的厳密 (geometrically exact / Simo-Reissner) Cosserat 梁プロトタイプの
基盤となる 3D 回転群 SO(3) の代数演算を集約する。

設計仕様: ``xkep_cae/elements/docs/cosserat_beam.md`` Phase 0.

提供する公開関数:

- ``skew(v)`` / ``vee(S)`` — 3D ベクトルと歪対称行列の相互変換
- ``exp_so3(theta)`` — 指数写像 (Rodrigues 公式)
- ``log_so3(R)`` — 対数写像 (回転行列 → 回転ベクトル)
- ``dexp_so3(theta)`` — 右ヤコビアン $T(\\theta)$
- ``dexp_inv_so3(theta)`` — 右ヤコビアン逆 $T^{-1}(\\theta)$
- ``compose(R_a, R_b)`` — 回転合成 (= ``R_a @ R_b``、API 明確化用)

すべての演算は単点版 ($\\theta\\in\\mathbb{R}^3$) と
バッチ版 (接頭辞 ``batch_``) を提供する。

数値条件:

- $\\phi=\\|\\theta\\|\\to 0$ ではテイラー展開 (4次まで) に切り替える
  ($\\phi < 10^{-8}$ で 倍精度の桁落ちを回避)。
- $\\phi\\to\\pi$ では log を四元数経由で取り、$\\sin\\phi\\to 0$ の発散を回避する。

CR-Timoshenko 梁 (``xkep_cae/elements/_beam_cr.py``) との関係:

- ``_beam_cr.py`` の private 関数 ``_skew`` / ``_rotvec_to_rotmat`` /
  ``_rotmat_to_rotvec`` / ``_tangent_operator`` / ``_tangent_operator_inv``
  と数学的に同等の演算を **公開 API** として再実装する。
- Phase 0 では既存の private 実装を意図的に変更しない (CR 梁は
  status-356 で `rel_err = 2.18e-07` の機械精度まで仕上がっており回帰防止)。
- 将来的に DRY 化する場合は本モジュールへの薄い委譲に切替える (Phase 4 以降)。
"""

from __future__ import annotations

import numpy as np

# テイラー展開切替の閾値. f64 マシン精度との両立を考慮し、
# φ^2 < 10^-16 となる φ < 10^-8 を採用。
_SMALL_ANGLE = 1.0e-8


# =====================================================================
# 単点版
# =====================================================================


def skew(v: np.ndarray) -> np.ndarray:
    """3D ベクトル v から歪対称行列 [v]× を構成する.

    定義: [v]× w = v × w (任意の w に対し).

    Args:
        v: shape (3,) の 1D 配列.

    Returns:
        shape (3, 3) の歪対称行列.
    """
    v = np.asarray(v, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"skew は (3,) ベクトルが必要: shape={v.shape}")
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def vee(S: np.ndarray) -> np.ndarray:
    """歪対称行列 S を 3D ベクトルに射影する (skew の逆).

    入力が厳密な歪対称でない場合でも対称成分を捨てる
    (実装上、対角と上三角の符号反転で組み立て直す)。

    Args:
        S: shape (3, 3) の行列.

    Returns:
        shape (3,) のベクトル.
    """
    S = np.asarray(S, dtype=float)
    if S.shape != (3, 3):
        raise ValueError(f"vee は (3, 3) 行列が必要: shape={S.shape}")
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def exp_so3(theta: np.ndarray) -> np.ndarray:
    """指数写像 (Rodrigues 公式): 回転ベクトル θ → 回転行列 R∈SO(3).

    公式:

    ``R = I + (sin φ / φ) [θ]× + ((1 - cos φ) / φ²) [θ]×²``

    ここで φ = ||θ||。φ → 0 ではテイラー展開:

    ``R ≈ I + [θ]× + 0.5 [θ]×²  (1次)``

    Args:
        theta: shape (3,) の回転ベクトル.

    Returns:
        shape (3, 3) の回転行列 R.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (3,):
        raise ValueError(f"exp_so3 は (3,) ベクトルが必要: shape={theta.shape}")
    phi = float(np.linalg.norm(theta))
    S = skew(theta)
    if phi < _SMALL_ANGLE:
        # テイラー展開: a ≈ 1 - φ²/6, b ≈ 0.5 - φ²/24
        a = 1.0 - phi * phi / 6.0
        b = 0.5 - phi * phi / 24.0
        return np.eye(3) + a * S + b * (S @ S)
    a = np.sin(phi) / phi
    b = (1.0 - np.cos(phi)) / (phi * phi)
    return np.eye(3) + a * S + b * (S @ S)


def log_so3(R: np.ndarray) -> np.ndarray:
    """対数写像: 回転行列 R∈SO(3) → 回転ベクトル θ.

    四元数経由で安定化する (φ → π での 1/sin(φ) 発散を回避)。

    Args:
        R: shape (3, 3) の回転行列.

    Returns:
        shape (3,) の回転ベクトル. ``||θ|| ≤ π`` を保証する.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"log_so3 は (3, 3) 行列が必要: shape={R.shape}")
    return _rotmat_to_rotvec_via_quat(R)


def dexp_so3(theta: np.ndarray) -> np.ndarray:
    """指数写像の右ヤコビアン T(θ).

    定義: $\\frac{d}{dt}\\exp(\\theta(t)) = \\exp(\\theta)\\,\\widehat{T(\\theta)\\dot\\theta}$.

    閉形式:

    ``T = I + ((1 - cos φ) / φ²) [θ]× + ((φ - sin φ) / φ³) [θ]×²``

    φ → 0 でテイラー展開:

    ``T ≈ I + 0.5 [θ]× + (1/6) [θ]×²``

    Args:
        theta: shape (3,) の回転ベクトル.

    Returns:
        shape (3, 3) の T(θ).
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (3,):
        raise ValueError(f"dexp_so3 は (3,) ベクトルが必要: shape={theta.shape}")
    phi = float(np.linalg.norm(theta))
    S = skew(theta)
    S2 = S @ S
    if phi < _SMALL_ANGLE:
        a = 0.5 - phi * phi / 24.0
        b = 1.0 / 6.0 - phi * phi / 120.0
        return np.eye(3) + a * S + b * S2
    a = (1.0 - np.cos(phi)) / (phi * phi)
    b = (phi - np.sin(phi)) / (phi * phi * phi)
    return np.eye(3) + a * S + b * S2


def dexp_inv_so3(theta: np.ndarray) -> np.ndarray:
    """指数写像の右ヤコビアン T(θ) の逆.

    閉形式 (Cardona & Géradin):

    ``T⁻¹ = I - 0.5 [θ]× + (1/φ² - (1 + cos φ) / (2 φ sin φ)) [θ]×²``

    φ → 0 でテイラー展開 (Bernoulli 数より):

    ``T⁻¹ ≈ I - 0.5 [θ]× + (1/12) [θ]×² + O(φ²)``

    Args:
        theta: shape (3,) の回転ベクトル.

    Returns:
        shape (3, 3) の T⁻¹(θ).
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (3,):
        raise ValueError(f"dexp_inv_so3 は (3,) ベクトルが必要: shape={theta.shape}")
    phi = float(np.linalg.norm(theta))
    S = skew(theta)
    S2 = S @ S
    if phi < _SMALL_ANGLE:
        # 1/φ² - (1 + cos φ) / (2 φ sin φ) → 1/12 + φ²/720 + O(φ⁴)
        b = 1.0 / 12.0 + phi * phi / 720.0
        return np.eye(3) - 0.5 * S + b * S2
    s = np.sin(phi)
    c = np.cos(phi)
    b = 1.0 / (phi * phi) - (1.0 + c) / (2.0 * phi * s)
    return np.eye(3) - 0.5 * S + b * S2


def compose(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """回転合成 R_a · R_b. ``R_a @ R_b`` と同等だが意図を明確化する API.

    Args:
        R_a: shape (3, 3) の回転行列.
        R_b: shape (3, 3) の回転行列.

    Returns:
        shape (3, 3) の合成回転.
    """
    return R_a @ R_b


# =====================================================================
# バッチ版 ((N,) サンプル一括処理)
# =====================================================================


def batch_skew(v: np.ndarray) -> np.ndarray:
    """``skew`` のバッチ版.

    Args:
        v: shape (N, 3).

    Returns:
        shape (N, 3, 3).
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"batch_skew は (N, 3) 配列が必要: shape={v.shape}")
    n = v.shape[0]
    S = np.zeros((n, 3, 3), dtype=float)
    S[:, 0, 1] = -v[:, 2]
    S[:, 0, 2] = v[:, 1]
    S[:, 1, 0] = v[:, 2]
    S[:, 1, 2] = -v[:, 0]
    S[:, 2, 0] = -v[:, 1]
    S[:, 2, 1] = v[:, 0]
    return S


def batch_exp_so3(thetas: np.ndarray) -> np.ndarray:
    """``exp_so3`` のバッチ版 (四元数経由で数値安定).

    Args:
        thetas: shape (N, 3).

    Returns:
        shape (N, 3, 3).
    """
    thetas = np.asarray(thetas, dtype=float)
    if thetas.ndim != 2 or thetas.shape[1] != 3:
        raise ValueError(f"batch_exp_so3 は (N, 3) 配列が必要: shape={thetas.shape}")
    n = thetas.shape[0]
    angles = np.linalg.norm(thetas, axis=1)

    half = angles / 2.0
    half_sq = half * half
    small = angles < _SMALL_ANGLE

    sinc = np.empty(n, dtype=float)
    cos_half = np.empty(n, dtype=float)
    sinc[small] = 0.5 - half_sq[small] / 48.0
    cos_half[small] = 1.0 - half_sq[small] / 2.0
    big = ~small
    sinc[big] = np.sin(half[big]) / angles[big]
    cos_half[big] = np.cos(half[big])

    w = cos_half
    xyz = sinc[:, None] * thetas
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    Rs = np.empty((n, 3, 3), dtype=float)
    Rs[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    Rs[:, 0, 1] = 2.0 * (x * y - w * z)
    Rs[:, 0, 2] = 2.0 * (x * z + w * y)
    Rs[:, 1, 0] = 2.0 * (x * y + w * z)
    Rs[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    Rs[:, 1, 2] = 2.0 * (y * z - w * x)
    Rs[:, 2, 0] = 2.0 * (x * z - w * y)
    Rs[:, 2, 1] = 2.0 * (y * z + w * x)
    Rs[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return Rs


def batch_log_so3(Rs: np.ndarray) -> np.ndarray:
    """``log_so3`` のバッチ版 (Shepperd 法ベクトル化).

    Args:
        Rs: shape (N, 3, 3).

    Returns:
        shape (N, 3). ``||θ_i|| ≤ π`` を保証する.
    """
    Rs = np.asarray(Rs, dtype=float)
    if Rs.ndim != 3 or Rs.shape[1:] != (3, 3):
        raise ValueError(f"batch_log_so3 は (N, 3, 3) 配列が必要: shape={Rs.shape}")
    n = Rs.shape[0]
    trace = Rs[:, 0, 0] + Rs[:, 1, 1] + Rs[:, 2, 2]
    diag_vals = np.stack([trace, Rs[:, 0, 0], Rs[:, 1, 1], Rs[:, 2, 2]], axis=1)
    best = np.argmax(diag_vals, axis=1)

    q = np.empty((n, 4), dtype=float)

    m0 = best == 0
    if np.any(m0):
        s = 2.0 * np.sqrt(trace[m0] + 1.0)
        q[m0, 0] = 0.25 * s
        q[m0, 1] = (Rs[m0, 2, 1] - Rs[m0, 1, 2]) / s
        q[m0, 2] = (Rs[m0, 0, 2] - Rs[m0, 2, 0]) / s
        q[m0, 3] = (Rs[m0, 1, 0] - Rs[m0, 0, 1]) / s
    m1 = best == 1
    if np.any(m1):
        s = 2.0 * np.sqrt(1.0 + Rs[m1, 0, 0] - Rs[m1, 1, 1] - Rs[m1, 2, 2])
        q[m1, 0] = (Rs[m1, 2, 1] - Rs[m1, 1, 2]) / s
        q[m1, 1] = 0.25 * s
        q[m1, 2] = (Rs[m1, 0, 1] + Rs[m1, 1, 0]) / s
        q[m1, 3] = (Rs[m1, 0, 2] + Rs[m1, 2, 0]) / s
    m2 = best == 2
    if np.any(m2):
        s = 2.0 * np.sqrt(1.0 + Rs[m2, 1, 1] - Rs[m2, 0, 0] - Rs[m2, 2, 2])
        q[m2, 0] = (Rs[m2, 0, 2] - Rs[m2, 2, 0]) / s
        q[m2, 1] = (Rs[m2, 0, 1] + Rs[m2, 1, 0]) / s
        q[m2, 2] = 0.25 * s
        q[m2, 3] = (Rs[m2, 1, 2] + Rs[m2, 2, 1]) / s
    m3 = best == 3
    if np.any(m3):
        s = 2.0 * np.sqrt(1.0 + Rs[m3, 2, 2] - Rs[m3, 0, 0] - Rs[m3, 1, 1])
        q[m3, 0] = (Rs[m3, 1, 0] - Rs[m3, 0, 1]) / s
        q[m3, 1] = (Rs[m3, 0, 2] + Rs[m3, 2, 0]) / s
        q[m3, 2] = (Rs[m3, 1, 2] + Rs[m3, 2, 1]) / s
        q[m3, 3] = 0.25 * s

    neg_w = q[:, 0] < 0
    q[neg_w] = -q[neg_w]
    norms = np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-15)
    q = q / norms

    vec = q[:, 1:4]
    sin_half = np.linalg.norm(vec, axis=1)
    small = sin_half < _SMALL_ANGLE
    big = ~small

    coeff = np.empty(n, dtype=float)
    coeff[small] = 2.0
    coeff[big] = 2.0 * np.arctan2(sin_half[big], q[big, 0]) / sin_half[big]
    return coeff[:, None] * vec


def batch_dexp_so3(thetas: np.ndarray) -> np.ndarray:
    """``dexp_so3`` のバッチ版.

    Args:
        thetas: shape (N, 3).

    Returns:
        shape (N, 3, 3).
    """
    thetas = np.asarray(thetas, dtype=float)
    if thetas.ndim != 2 or thetas.shape[1] != 3:
        raise ValueError(f"batch_dexp_so3 は (N, 3) 配列が必要: shape={thetas.shape}")
    n = thetas.shape[0]
    phi = np.linalg.norm(thetas, axis=1)
    S = batch_skew(thetas)
    S2 = np.einsum("nij,njk->nik", S, S)

    a = np.empty(n, dtype=float)
    b = np.empty(n, dtype=float)
    small = phi < _SMALL_ANGLE
    big = ~small
    a[small] = 0.5 - phi[small] ** 2 / 24.0
    b[small] = 1.0 / 6.0 - phi[small] ** 2 / 120.0
    a[big] = (1.0 - np.cos(phi[big])) / (phi[big] ** 2)
    b[big] = (phi[big] - np.sin(phi[big])) / (phi[big] ** 3)

    eye = np.broadcast_to(np.eye(3), (n, 3, 3))
    return eye + a[:, None, None] * S + b[:, None, None] * S2


def batch_dexp_inv_so3(thetas: np.ndarray) -> np.ndarray:
    """``dexp_inv_so3`` のバッチ版.

    Args:
        thetas: shape (N, 3).

    Returns:
        shape (N, 3, 3).
    """
    thetas = np.asarray(thetas, dtype=float)
    if thetas.ndim != 2 or thetas.shape[1] != 3:
        raise ValueError(f"batch_dexp_inv_so3 は (N, 3) 配列が必要: shape={thetas.shape}")
    n = thetas.shape[0]
    phi = np.linalg.norm(thetas, axis=1)
    S = batch_skew(thetas)
    S2 = np.einsum("nij,njk->nik", S, S)

    b = np.empty(n, dtype=float)
    small = phi < _SMALL_ANGLE
    big = ~small
    b[small] = 1.0 / 12.0 + phi[small] ** 2 / 720.0
    s_big = np.sin(phi[big])
    c_big = np.cos(phi[big])
    b[big] = 1.0 / (phi[big] ** 2) - (1.0 + c_big) / (2.0 * phi[big] * s_big)

    eye = np.broadcast_to(np.eye(3), (n, 3, 3))
    return eye - 0.5 * S + b[:, None, None] * S2


# =====================================================================
# 内部ユーティリティ — 四元数経由 log
# =====================================================================


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """回転行列 R → 単位四元数 q = (w, x, y, z) (Shepperd 法)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] >= R[1, 1]) and (R[0, 0] >= R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    if q[0] < 0.0:
        q = -q
    norm = float(np.linalg.norm(q))
    if norm < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def _rotmat_to_rotvec_via_quat(R: np.ndarray) -> np.ndarray:
    """回転行列 R → 回転ベクトル θ (||θ||≤π を保証)."""
    q = _rotmat_to_quat(R)
    sin_half = float(np.linalg.norm(q[1:4]))
    if sin_half < _SMALL_ANGLE:
        return 2.0 * q[1:4]
    angle = 2.0 * np.arctan2(sin_half, q[0])
    return (angle / sin_half) * q[1:4]
