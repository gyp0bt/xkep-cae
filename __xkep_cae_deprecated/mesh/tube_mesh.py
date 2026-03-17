"""3D チューブ（円筒管）メッシュジェネレータ.

HEX8 要素で円筒管を離散化する。シースモデルの曲げモード
バリデーション用の FEM 参照解を生成するために使用。

== メッシュ構造 ==

構造格子: (径方向 n_r) × (周方向 n_theta) × (軸方向 n_z) の HEX8 要素

節点順序:
  node_id = k * (n_r+1) * n_theta + i * n_theta + j
  k: 軸方向インデックス (0..n_z)
  i: 径方向インデックス (0..n_r)
  j: 周方向インデックス (0..n_theta-1)
"""

from __future__ import annotations

import numpy as np


def make_tube_mesh(
    r_inner: float,
    r_outer: float,
    length: float,
    n_r: int,
    n_theta: int,
    n_z: int,
) -> tuple[np.ndarray, np.ndarray]:
    """HEX8 要素による円筒管メッシュを生成.

    Parameters
    ----------
    r_inner : float
        内径 [m]
    r_outer : float
        外径 [m]
    length : float
        管長さ [m]（z 方向）
    n_r : int
        径方向要素数
    n_theta : int
        周方向要素数
    n_z : int
        軸方向要素数

    Returns
    -------
    nodes : (n_nodes, 3) ndarray
        節点座標 [x, y, z]
    elements : (n_elems, 8) ndarray
        要素接続（HEX8 節点順序）
    """
    if r_inner <= 0 or r_outer <= r_inner:
        raise ValueError(f"0 < r_inner < r_outer が必要: r_inner={r_inner}, r_outer={r_outer}")
    if length <= 0:
        raise ValueError(f"length > 0 が必要: length={length}")

    r_vals = np.linspace(r_inner, r_outer, n_r + 1)
    theta_vals = np.linspace(0, 2 * np.pi, n_theta + 1)[:-1]  # 周期なので末尾除外
    z_vals = np.linspace(0, length, n_z + 1)

    n_ring = (n_r + 1) * n_theta  # 1断面あたりの節点数
    n_nodes = n_ring * (n_z + 1)

    nodes = np.zeros((n_nodes, 3))
    for k in range(n_z + 1):
        for i in range(n_r + 1):
            for j in range(n_theta):
                nid = k * n_ring + i * n_theta + j
                nodes[nid, 0] = r_vals[i] * np.cos(theta_vals[j])
                nodes[nid, 1] = r_vals[i] * np.sin(theta_vals[j])
                nodes[nid, 2] = z_vals[k]

    # 要素生成
    elements = []
    for k in range(n_z):
        for i in range(n_r):
            for j in range(n_theta):
                j_next = (j + 1) % n_theta
                # 底面 (z=z_k)
                n0 = k * n_ring + i * n_theta + j
                n1 = k * n_ring + (i + 1) * n_theta + j
                n2 = k * n_ring + (i + 1) * n_theta + j_next
                n3 = k * n_ring + i * n_theta + j_next
                # 上面 (z=z_{k+1})
                n4 = (k + 1) * n_ring + i * n_theta + j
                n5 = (k + 1) * n_ring + (i + 1) * n_theta + j
                n6 = (k + 1) * n_ring + (i + 1) * n_theta + j_next
                n7 = (k + 1) * n_ring + i * n_theta + j_next
                elements.append([n0, n1, n2, n3, n4, n5, n6, n7])

    return nodes, np.array(elements, dtype=int)


def tube_face_nodes(
    nodes: np.ndarray,
    face: str,
    *,
    length: float | None = None,
    r_inner: float | None = None,
    r_outer: float | None = None,
    tol: float = 1e-10,
) -> np.ndarray:
    """チューブメッシュの特定面の節点インデックスを返す.

    Parameters
    ----------
    nodes : (n_nodes, 3) ndarray
    face : str
        "z0" (z=0面), "zL" (z=L面), "inner" (内面), "outer" (外面)
    length : float | None
        管長さ（"zL" 面で使用）
    r_inner, r_outer : float | None
        内外径（"inner", "outer" 面で使用）
    tol : float
        判定許容差

    Returns
    -------
    node_ids : (N,) ndarray — 面上の節点インデックス
    """
    r = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)

    if face == "z0":
        return np.where(np.abs(nodes[:, 2]) < tol)[0]
    elif face == "zL":
        if length is None:
            raise ValueError("zL 面には length が必要")
        return np.where(np.abs(nodes[:, 2] - length) < tol)[0]
    elif face == "inner":
        if r_inner is None:
            raise ValueError("inner 面には r_inner が必要")
        return np.where(np.abs(r - r_inner) < tol)[0]
    elif face == "outer":
        if r_outer is None:
            raise ValueError("outer 面には r_outer が必要")
        return np.where(np.abs(r - r_outer) < tol)[0]
    else:
        raise ValueError(f"未知の面: '{face}'。'z0', 'zL', 'inner', 'outer' から選択")
