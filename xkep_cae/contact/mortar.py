"""Mortar 離散化 — Phase C6-L5.

セグメント境界での接触圧の連続性を保証するため、ラグランジュ乗数を
スレーブ側（A 側）節点ベースで定義する。隣接セグメントが同一節点の λ を
共有することで、接触圧の不連続ジャンプを解消する。

重み付きギャップ:
  g̃_k = Σ_{gp} Φ_k(s_gp) · w_gp · gap(s_gp)

Mortar 制約ヤコビアン:
  G_mortar[k, :] = Σ_{gp} Φ_k(s_gp) · w_gp · ∂gap/∂u(s_gp)

Mortar 基底関数（線形梁要素）:
  Φ_{A0}(s) = 1 - s,  Φ_{A1}(s) = s

設計仕様: docs/contact/contact-algorithm-overhaul-c6.md §7
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.assembly import _contact_dofs
from xkep_cae.contact.line_contact import gauss_legendre_01, project_point_to_segment
from xkep_cae.contact.pair import ContactManager, ContactStatus


def identify_mortar_nodes(
    manager: ContactManager,
    active_indices: list[int],
) -> tuple[list[int], dict[int, list[tuple[int, int]]]]:
    """アクティブペアからスレーブ側（A 側）の一意な Mortar 節点を抽出する.

    Args:
        manager: 接触マネージャ
        active_indices: アクティブペアインデックスリスト

    Returns:
        mortar_nodes: ソート済み一意スレーブ節点リスト
        node_to_pairs: {node_id: [(pair_idx, local_pos)]} のマッピング
            local_pos: 0 = A0（始点）, 1 = A1（終点）
    """
    node_to_pairs: dict[int, list[tuple[int, int]]] = {}

    for pair_idx in active_indices:
        pair = manager.pairs[pair_idx]
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        a0 = int(pair.nodes_a[0])
        a1 = int(pair.nodes_a[1])

        if a0 not in node_to_pairs:
            node_to_pairs[a0] = []
        node_to_pairs[a0].append((pair_idx, 0))

        if a1 not in node_to_pairs:
            node_to_pairs[a1] = []
        node_to_pairs[a1].append((pair_idx, 1))

    mortar_nodes = sorted(node_to_pairs.keys())
    return mortar_nodes, node_to_pairs


def build_mortar_system(
    manager: ContactManager,
    active_indices: list[int],
    mortar_nodes: list[int],
    node_coords: np.ndarray,
    ndof_total: int,
    ndof_per_node: int = 6,
    n_gauss: int = 3,
    k_pen: float = 1e4,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Mortar 制約ヤコビアン G_mortar と重み付きギャップ g_mortar を構築する.

    各アクティブペアについて Gauss 積分:
      G_mortar[k, :] += Σ_gp Φ_k(s_gp) · w_gp · ∂gap/∂u
      g_mortar[k]    += Σ_gp Φ_k(s_gp) · w_gp · gap(s_gp)

    Args:
        manager: 接触マネージャ
        active_indices: アクティブペアインデックスリスト
        mortar_nodes: Mortar 節点リスト（identify_mortar_nodes の出力）
        node_coords: (n_nodes, 3) 変形座標
        ndof_total: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数
        n_gauss: Gauss 積分点数
        k_pen: ペナルティ剛性（未使用、将来の拡張用）

    Returns:
        G_mortar: (n_mortar, ndof_total) Mortar 制約ヤコビアン
        g_mortar: (n_mortar,) 重み付きギャップ
    """
    n_mortar = len(mortar_nodes)
    if n_mortar == 0:
        return sp.csr_matrix((0, ndof_total)), np.array([])

    # 節点→Mortar インデックスのマップ
    node_to_idx = {node: idx for idx, node in enumerate(mortar_nodes)}

    g_mortar = np.zeros(n_mortar)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    gp_pts, gp_wts = gauss_legendre_01(n_gauss)

    for pair_idx in active_indices:
        pair = manager.pairs[pair_idx]
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        a0 = int(pair.nodes_a[0])
        a1 = int(pair.nodes_a[1])
        b0 = int(pair.nodes_b[0])
        b1 = int(pair.nodes_b[1])

        # Mortar インデックス
        k0 = node_to_idx.get(a0)
        k1 = node_to_idx.get(a1)
        if k0 is None and k1 is None:
            continue

        # 座標
        xA0 = node_coords[a0]
        xA1 = node_coords[a1]
        xB0 = node_coords[b0]
        xB1 = node_coords[b1]

        dofs = _contact_dofs(pair, ndof_per_node)

        for gp_idx in range(n_gauss):
            s_gp = gp_pts[gp_idx]
            w_gp = gp_wts[gp_idx]

            # A 側の Gauss 点座標
            x_gp = (1.0 - s_gp) * xA0 + s_gp * xA1

            # B 側への射影
            t_gp = project_point_to_segment(x_gp, xB0, xB1)

            # B 側の点
            x_proj = (1.0 - t_gp) * xB0 + t_gp * xB1

            # ギャップベクトル・法線
            gap_vec = x_gp - x_proj
            gap_dist = float(np.linalg.norm(gap_vec))

            # 半径補正
            r_a = pair.state.radius_a if hasattr(pair.state, "radius_a") else 0.0
            r_b = pair.state.radius_b if hasattr(pair.state, "radius_b") else 0.0
            gap_val = gap_dist - (r_a + r_b)

            if gap_dist > 1e-30:
                normal = gap_vec / gap_dist
            else:
                normal = (
                    pair.state.normal
                    if pair.state.normal is not None
                    else np.array([0.0, 0.0, 1.0])
                )

            # Mortar 基底関数
            phi_0 = 1.0 - s_gp  # Φ_{A0}
            phi_1 = s_gp  # Φ_{A1}

            # 重み付きギャップ: g̃_k += Φ_k · w · gap
            if k0 is not None:
                g_mortar[k0] += phi_0 * w_gp * gap_val
            if k1 is not None:
                g_mortar[k1] += phi_1 * w_gp * gap_val

            # ∂gap/∂u の係数:
            # gap = (x_A(s) - x_B(t)) · n
            # x_A(s) = (1-s) xA0 + s xA1
            # x_B(t) = (1-t) xB0 + t xB1
            # ∂gap/∂xA0 = (1-s) n, ∂gap/∂xA1 = s n
            # ∂gap/∂xB0 = -(1-t) n, ∂gap/∂xB1 = -t n
            coeffs = [(1.0 - s_gp), s_gp, -(1.0 - t_gp), -t_gp]

            for mortar_k, phi_k in [(k0, phi_0), (k1, phi_1)]:
                if mortar_k is None:
                    continue
                for node_local in range(4):
                    for d in range(3):
                        global_dof = dofs[node_local * ndof_per_node + d]
                        val = phi_k * w_gp * coeffs[node_local] * normal[d]
                        if abs(val) > 1e-30:
                            rows.append(mortar_k)
                            cols.append(global_dof)
                            vals.append(val)

    if len(vals) == 0:
        return sp.csr_matrix((n_mortar, ndof_total)), g_mortar

    G_mortar = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(n_mortar, ndof_total),
    ).tocsr()
    return G_mortar, g_mortar


def compute_mortar_contact_force(
    manager: ContactManager,
    active_indices: list[int],
    mortar_nodes: list[int],
    lam_mortar: np.ndarray,
    node_coords: np.ndarray,
    ndof_total: int,
    ndof_per_node: int = 6,
    n_gauss: int = 3,
    k_pen: float = 1e4,
) -> np.ndarray:
    """Mortar 乗数から接触力ベクトルを計算する.

    Gauss 点での乗数補間:
      lam_gp = (1-s) · λ_{A0} + s · λ_{A1}
    法線力:
      p_n = max(0, lam_gp + k_pen · (-gap))
    接触力:
      f_c += p_n · g_shape · w_gp

    Args:
        manager: 接触マネージャ
        active_indices: アクティブペアインデックスリスト
        mortar_nodes: Mortar 節点リスト
        lam_mortar: (n_mortar,) Mortar 乗数
        node_coords: (n_nodes, 3) 変形座標
        ndof_total: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数
        n_gauss: Gauss 積分点数
        k_pen: ペナルティ剛性

    Returns:
        f_c: (ndof_total,) 接触力ベクトル
    """
    f_c = np.zeros(ndof_total)

    if len(mortar_nodes) == 0:
        return f_c

    node_to_idx = {node: idx for idx, node in enumerate(mortar_nodes)}
    gp_pts, gp_wts = gauss_legendre_01(n_gauss)

    for pair_idx in active_indices:
        pair = manager.pairs[pair_idx]
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        a0 = int(pair.nodes_a[0])
        a1 = int(pair.nodes_a[1])
        b0 = int(pair.nodes_b[0])
        b1 = int(pair.nodes_b[1])

        k0 = node_to_idx.get(a0)
        k1 = node_to_idx.get(a1)
        if k0 is None and k1 is None:
            continue

        xA0 = node_coords[a0]
        xA1 = node_coords[a1]
        xB0 = node_coords[b0]
        xB1 = node_coords[b1]

        dofs = _contact_dofs(pair, ndof_per_node)

        r_a = pair.state.radius_a if hasattr(pair.state, "radius_a") else 0.0
        r_b = pair.state.radius_b if hasattr(pair.state, "radius_b") else 0.0

        for gp_idx in range(n_gauss):
            s_gp = gp_pts[gp_idx]
            w_gp = gp_wts[gp_idx]

            x_gp = (1.0 - s_gp) * xA0 + s_gp * xA1
            t_gp = project_point_to_segment(x_gp, xB0, xB1)
            x_proj = (1.0 - t_gp) * xB0 + t_gp * xB1

            gap_vec = x_gp - x_proj
            gap_dist = float(np.linalg.norm(gap_vec))
            gap_val = gap_dist - (r_a + r_b)

            if gap_dist > 1e-30:
                normal = gap_vec / gap_dist
            else:
                normal = (
                    pair.state.normal
                    if pair.state.normal is not None
                    else np.array([0.0, 0.0, 1.0])
                )

            # Mortar 乗数の補間
            lam_0 = lam_mortar[k0] if k0 is not None else 0.0
            lam_1 = lam_mortar[k1] if k1 is not None else 0.0
            lam_gp = (1.0 - s_gp) * lam_0 + s_gp * lam_1

            # AL 法線力
            p_n = max(0.0, lam_gp + k_pen * (-gap_val))
            if p_n <= 0.0:
                continue

            # 形状ベクトル: f_c に寄与
            # g_shape = [-(1-s)·n, -s·n, (1-t)·n, t·n]
            coeffs_shape = [-(1.0 - s_gp), -s_gp, (1.0 - t_gp), t_gp]

            for node_local in range(4):
                for d in range(3):
                    global_dof = dofs[node_local * ndof_per_node + d]
                    f_c[global_dof] += p_n * w_gp * coeffs_shape[node_local] * normal[d]

    return f_c


def compute_mortar_p_n(
    mortar_nodes: list[int],
    lam_mortar: np.ndarray,
    g_mortar: np.ndarray,
    k_pen: float,
) -> np.ndarray:
    """各 Mortar 節点の法線力を計算する.

    p_n_k = max(0, λ_k + k_pen · (-g̃_k))

    Args:
        mortar_nodes: Mortar 節点リスト
        lam_mortar: (n_mortar,) Mortar 乗数
        g_mortar: (n_mortar,) 重み付きギャップ
        k_pen: ペナルティ剛性

    Returns:
        p_n: (n_mortar,) 各 Mortar 節点の法線力
    """
    n_mortar = len(mortar_nodes)
    if n_mortar == 0:
        return np.array([])

    p_n = np.maximum(0.0, lam_mortar[:n_mortar] + k_pen * (-g_mortar[:n_mortar]))
    return p_n
