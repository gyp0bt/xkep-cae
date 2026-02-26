"""2D定常熱伝導FEM — Q4双線形四角形要素.

支配方程式（薄板フィンモデル）:
  k t ∇²T + q t - 2h(T - T_∞) = 0

弱形式:
  (K_cond + K_conv_surf + K_conv_edge) T = Q_gen + Q_conv_surf + Q_conv_edge

ここで:
  K_cond       = ∫∫ ∇Nᵀ k t ∇N dA          （面内熱伝導）
  K_conv_surf  = ∫∫ Nᵀ 2h N dA              （表裏面対流、厚さ t で割ってある）
  K_conv_edge  = ∫ Nᵀ h t N dS              （側面対流、厚さ t 分の側面積）
  Q_gen        = ∫∫ Nᵀ q t dA               （体積発熱）
  Q_conv_surf  = ∫∫ Nᵀ 2h T_∞ dA            （表裏面対流負荷）
  Q_conv_edge  = ∫ Nᵀ h t T_∞ dS            （側面対流負荷）
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spla

# ---------------------------------------------------------------------------
# Q4 形状関数ユーティリティ
# ---------------------------------------------------------------------------


def _gauss_2x2():
    """2×2 Gauss積分点と重み."""
    g = 1.0 / np.sqrt(3.0)
    return np.array([[-g, -g], [g, -g], [g, g], [-g, g]]), np.ones(4)


def _shape_functions(xi: float, eta: float) -> np.ndarray:
    """Q4 形状関数 N (4,)."""
    return 0.25 * np.array(
        [
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ]
    )


def _shape_derivatives(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Q4 形状関数微分 dN/dξ, dN/dη."""
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return dN_dxi, dN_deta


def _jacobian(dN_dxi, dN_deta, node_xy):
    """Jacobian 行列と行列式."""
    J = np.array(
        [
            [dN_dxi @ node_xy[:, 0], dN_deta @ node_xy[:, 0]],
            [dN_dxi @ node_xy[:, 1], dN_deta @ node_xy[:, 1]],
        ]
    )
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ
    return invJ, detJ


# ---------------------------------------------------------------------------
# 要素レベル行列
# ---------------------------------------------------------------------------


def quad4_conductivity(node_xy: np.ndarray, k: float, t: float) -> np.ndarray:
    """Q4要素の熱伝導行列 K_e = ∫ ∇Nᵀ k t ∇N dA.

    Args:
        node_xy: (4, 2) 節点座標
        k: 熱伝導率 [W/(m·K)]
        t: 板厚 [m]

    Returns:
        Ke: (4, 4) 局所伝導行列
    """
    gps, wts = _gauss_2x2()
    Ke = np.zeros((4, 4))
    for (xi, eta), w in zip(gps, wts, strict=True):
        dN_dxi, dN_deta = _shape_derivatives(xi, eta)
        invJ, detJ = _jacobian(dN_dxi, dN_deta, node_xy)
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        B = np.vstack([dN_dx, dN_dy])  # (2, 4)
        Ke += (B.T @ B) * k * t * detJ * w
    return Ke


def quad4_convection_surface(node_xy: np.ndarray, h: float) -> np.ndarray:
    """表裏面対流行列 H_surf = ∫ Nᵀ 2h N dA.

    薄板の表裏面から対流熱伝達。係数 2 は両面分。

    Args:
        node_xy: (4, 2) 節点座標
        h: 対流熱伝達率 [W/(m²·K)]

    Returns:
        He: (4, 4) 局所対流行列
    """
    gps, wts = _gauss_2x2()
    He = np.zeros((4, 4))
    for (xi, eta), w in zip(gps, wts, strict=True):
        N = _shape_functions(xi, eta)
        dN_dxi, dN_deta = _shape_derivatives(xi, eta)
        _, detJ = _jacobian(dN_dxi, dN_deta, node_xy)
        He += np.outer(N, N) * 2.0 * h * detJ * w
    return He


def quad4_heat_load(node_xy: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    """体積発熱荷重 f_q = ∫ Nᵀ q t dA.

    Args:
        node_xy: (4, 2) 節点座標
        q: (4,) 各節点での体積発熱密度 [W/m³]
        t: 板厚 [m]

    Returns:
        fe: (4,) 局所荷重ベクトル
    """
    gps, wts = _gauss_2x2()
    fe = np.zeros(4)
    for (xi, eta), w in zip(gps, wts, strict=True):
        N = _shape_functions(xi, eta)
        dN_dxi, dN_deta = _shape_derivatives(xi, eta)
        _, detJ = _jacobian(dN_dxi, dN_deta, node_xy)
        q_gp = N @ q  # 積分点での発熱密度
        fe += N * q_gp * t * detJ * w
    return fe


def quad4_convection_load_surface(
    node_xy: np.ndarray,
    h: float,
    T_inf: float,
) -> np.ndarray:
    """表裏面対流荷重 f_conv = ∫ Nᵀ 2h T_∞ dA.

    Args:
        node_xy: (4, 2) 節点座標
        h: 対流熱伝達率 [W/(m²·K)]
        T_inf: 周囲温度 [K or °C]

    Returns:
        fe: (4,) 局所荷重ベクトル
    """
    gps, wts = _gauss_2x2()
    fe = np.zeros(4)
    for (xi, eta), w in zip(gps, wts, strict=True):
        N = _shape_functions(xi, eta)
        dN_dxi, dN_deta = _shape_derivatives(xi, eta)
        _, detJ = _jacobian(dN_dxi, dN_deta, node_xy)
        fe += N * 2.0 * h * T_inf * detJ * w
    return fe


# ---------------------------------------------------------------------------
# 辺の対流（側面）
# ---------------------------------------------------------------------------


def _edge_convection(
    n1_xy: np.ndarray,
    n2_xy: np.ndarray,
    h: float,
    t: float,
    T_inf: float,
) -> tuple[np.ndarray, np.ndarray]:
    """2節点辺の対流行列と荷重.

    H_edge = ∫ Nᵀ h t N dS, f_edge = ∫ Nᵀ h t T_∞ dS

    Args:
        n1_xy, n2_xy: 辺の両端の座標 (2,)
        h: 対流熱伝達率
        t: 板厚（側面の高さ）
        T_inf: 周囲温度

    Returns:
        He: (2, 2), fe: (2,)
    """
    L = np.linalg.norm(n2_xy - n1_xy)
    # 1D 2点 Gauss 積分（解析解で十分）
    # ∫₀ᴸ Nᵀ h t N ds = h t L / 6 * [[2, 1], [1, 2]]
    He = h * t * L / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    fe = h * t * T_inf * L / 2.0 * np.ones(2)
    return He, fe


# ---------------------------------------------------------------------------
# メッシュ生成
# ---------------------------------------------------------------------------


def make_rect_mesh(
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """長方形均一メッシュの生成.

    Args:
        Lx, Ly: x方向/y方向の長さ [m]
        nx, ny: x方向/y方向の要素数

    Returns:
        nodes: ((nx+1)*(ny+1), 2) 節点座標
        conn: (nx*ny, 4) 接続配列（反時計回り）
        boundary_edges: 辺別の辺配列 dict
            "bottom", "right", "top", "left" → (n_edges, 2) 節点インデックス
    """
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    conn = np.empty((nx * ny, 4), dtype=int)
    idx = 0
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1) + 1
            n3 = n0 + (nx + 1)
            conn[idx] = [n0, n1, n2, n3]
            idx += 1

    boundary_edges: dict[str, np.ndarray] = {}
    # bottom: y=0
    bottom = np.array([[i, i + 1] for i in range(nx)], dtype=int)
    # top: y=Ly
    top_start = ny * (nx + 1)
    top = np.array([[top_start + i, top_start + i + 1] for i in range(nx)], dtype=int)
    # left: x=0
    left = np.array([[j * (nx + 1), (j + 1) * (nx + 1)] for j in range(ny)], dtype=int)
    # right: x=Lx
    right = np.array(
        [[j * (nx + 1) + nx, (j + 1) * (nx + 1) + nx] for j in range(ny)],
        dtype=int,
    )
    boundary_edges["bottom"] = bottom
    boundary_edges["top"] = top
    boundary_edges["left"] = left
    boundary_edges["right"] = right
    return nodes, conn, boundary_edges


# ---------------------------------------------------------------------------
# 全体アセンブリ
# ---------------------------------------------------------------------------


def assemble_thermal_system(
    nodes: np.ndarray,
    conn: np.ndarray,
    boundary_edges: dict[str, np.ndarray],
    *,
    k: float,
    h_conv: float,
    t: float,
    T_inf: float,
    q_nodal: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """定常熱伝導系のアセンブリ.

    (K_cond + K_conv_surf + K_conv_edge) T = Q_gen + Q_conv

    Args:
        nodes: (N, 2) 節点座標
        conn: (Ne, 4) 接続配列
        boundary_edges: 境界辺 dict（make_rect_mesh 出力）
        k: 熱伝導率 [W/(m·K)]
        h_conv: 対流熱伝達率 [W/(m²·K)]
        t: 板厚 [m]
        T_inf: 周囲温度 [K]
        q_nodal: (N,) 各節点の体積発熱密度 [W/m³]

    Returns:
        K_total: (N, N) CSR行列
        f_total: (N,) 右辺ベクトル
    """
    N = len(nodes)
    rows, cols, data_k = [], [], []
    f_total = np.zeros(N)

    # --- 要素ループ: 伝導 + 表裏面対流 + 発熱 ---
    for elem_nodes in conn:
        xy = nodes[elem_nodes]
        Ke = quad4_conductivity(xy, k, t)
        He_surf = quad4_convection_surface(xy, h_conv)
        Ke_total = Ke + He_surf

        q_e = q_nodal[elem_nodes]
        fe_gen = quad4_heat_load(xy, q_e, t)
        fe_conv = quad4_convection_load_surface(xy, h_conv, T_inf)
        fe = fe_gen + fe_conv

        for i_local in range(4):
            gi = elem_nodes[i_local]
            f_total[gi] += fe[i_local]
            for j_local in range(4):
                gj = elem_nodes[j_local]
                rows.append(gi)
                cols.append(gj)
                data_k.append(Ke_total[i_local, j_local])

    # --- 辺対流（側面） ---
    for _name, edges in boundary_edges.items():
        for e in edges:
            n1, n2 = e
            He, fe = _edge_convection(nodes[n1], nodes[n2], h_conv, t, T_inf)
            for i_local, gi in enumerate(e):
                f_total[gi] += fe[i_local]
                for j_local, gj in enumerate(e):
                    rows.append(gi)
                    cols.append(gj)
                    data_k.append(He[i_local, j_local])

    K_total = sp.csr_matrix(
        (np.array(data_k), (np.array(rows), np.array(cols))),
        shape=(N, N),
    )
    K_total.sum_duplicates()
    return K_total, f_total


def solve_steady_thermal(
    K: sp.csr_matrix,
    f: np.ndarray,
) -> np.ndarray:
    """定常熱伝導の求解 K T = f.

    Args:
        K: (N, N) 全体行列
        f: (N,) 右辺ベクトル

    Returns:
        T: (N,) 温度ベクトル
    """
    return spla.spsolve(K, f)


def compute_heat_flux(
    nodes: np.ndarray,
    conn: np.ndarray,
    T: np.ndarray,
    k: float,
    t: float,
) -> np.ndarray:
    """要素中心の熱流束 q = -k ∇T を計算.

    Args:
        nodes: (N, 2) 節点座標
        conn: (Ne, 4) 接続配列
        T: (N,) 温度分布
        k: 熱伝導率 [W/(m·K)]
        t: 板厚 [m]（未使用だが引数統一）

    Returns:
        flux: (Ne, 2) 要素中心の熱流束 [qx, qy]
    """
    Ne = len(conn)
    flux = np.zeros((Ne, 2))
    # 要素中心 (ξ=0, η=0) で評価
    dN_dxi, dN_deta = _shape_derivatives(0.0, 0.0)
    for e in range(Ne):
        xy = nodes[conn[e]]
        invJ, _ = _jacobian(dN_dxi, dN_deta, xy)
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        T_e = T[conn[e]]
        flux[e, 0] = -k * (dN_dx @ T_e)
        flux[e, 1] = -k * (dN_dy @ T_e)
    return flux
