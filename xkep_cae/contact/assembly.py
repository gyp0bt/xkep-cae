"""接触内力・接線剛性のアセンブリ.

Phase C2: 法線接触力のグローバルベクトル/行列への組み込み。
Phase C3: 摩擦力 + 摩擦接線剛性の追加。
Phase C5: 幾何微分込み一貫接線（K_geo）の追加。
Phase C6-L1: Line-to-line Gauss 積分による接触力・剛性評価。

接触力の節点配分:
    セグメント A: p(s) = xA0 + s*(xA1 - xA0)
    → 節点 A0 に (1-s)*f,  節点 A1 に s*f を配分
    セグメント B: 反作用力を同様に (1-t), t で配分

法線接線剛性:
    K_n = k_pen * g_n g_n^T                     (主項)
    K_geo = -p_n/dist * G^T * (I₃ - n⊗n) * G    (幾何剛性, Phase C5)

摩擦接線剛性:
    stick: K_f = k_t * (g_t1 g_t1^T + g_t2 g_t2^T)
    slip:  K_f = ratio * k_t * (I₂ - q̂⊗q̂) に基づく (Phase C5)

設計仕様: docs/contact/beam_beam_contact_spec_v0.1.md §5, §7
         docs/contact/contact-algorithm-overhaul-c6.md §3
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.law_normal import evaluate_normal_force, normal_force_linearization
from xkep_cae.contact.pair import ContactManager, ContactPair, ContactStatus


def _contact_dofs(pair: ContactPair, ndof_per_node: int = 6) -> np.ndarray:
    """接触ペアに関与する全体 DOF インデックスを返す.

    4節点（A0, A1, B0, B1）× ndof_per_node の DOF を返す。
    ただし接触力は並進DOF (最初の3成分) のみに寄与する。

    Args:
        pair: 接触ペア
        ndof_per_node: 1節点あたりの DOF 数

    Returns:
        dofs: (4 * ndof_per_node,) 全体DOFインデックス
    """
    nodes = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]
    dofs = np.empty(4 * ndof_per_node, dtype=int)
    for i, n in enumerate(nodes):
        for d in range(ndof_per_node):
            dofs[i * ndof_per_node + d] = n * ndof_per_node + d
    return dofs


def _contact_shape_vector(pair: ContactPair) -> np.ndarray:
    """接触力の法線方向形状ベクトル N^T n を構築する.

    4節点（A0, A1, B0, B1）の並進 DOF (3成分) に対する
    法線接触力の形状ベクトル。

    N_A n = [(1-s)*n, s*n]  (A 側: 法線方向に押す)
    N_B n = [(1-t)*n, t*n]  (B 側: 反作用)

    返す形状ベクトル g: f_contact = p_n * g  (12成分: 4節点 × 3DOF)

    Args:
        pair: 接触ペア

    Returns:
        g: (12,) 形状ベクトル
    """
    s = pair.state.s
    t = pair.state.t
    n = pair.state.normal  # (3,) A→B 方向

    g = np.zeros(12)
    # A 側に法線方向の力（A を B から押し返す → -n 方向）
    g[0:3] = -(1.0 - s) * n
    g[3:6] = -s * n
    # B 側に反作用（B を A から押す → +n 方向）
    g[6:9] = (1.0 - t) * n
    g[9:12] = t * n
    return g


def _contact_tangent_shape_vector(pair: ContactPair, axis: int) -> np.ndarray:
    """接触力の接線方向形状ベクトルを構築する.

    法線形状ベクトルと同じ配分方式だが、方向が t1 または t2。
    摩擦力は A を滑り方向に引きずり、B を反対方向に引く。

    f_friction = q_t[axis] * g_ti  (i = axis)

    ここで g_ti は法線 n の代わりに ti を使った形状ベクトル:
    A 側: [-(1-s)*ti, -s*ti]  (A を B から引き離す方向)
    B 側: [(1-t)*ti,  t*ti]   (B に反作用)

    Args:
        pair: 接触ペア
        axis: 0 → t1 方向、1 → t2 方向

    Returns:
        g_t: (12,) 接線方向形状ベクトル
    """
    s = pair.state.s
    t = pair.state.t
    ti = pair.state.tangent1 if axis == 0 else pair.state.tangent2

    g_t = np.zeros(12)
    # A 側: 摩擦力は相対滑りに抵抗 → A を B 方向に引く(+ti)
    # B 側: 反作用 → B を A 方向に引く(-ti)
    # 符号規約: q_t > 0 は B が t1 正方向に滑ったとき、
    # B に -ti 方向の摩擦力、A に +ti 方向
    g_t[0:3] = (1.0 - s) * ti
    g_t[3:6] = s * ti
    g_t[6:9] = -(1.0 - t) * ti
    g_t[9:12] = -t * ti
    return g_t


def _contact_geometric_stiffness_local(pair: ContactPair) -> np.ndarray:
    """接触幾何剛性（法線回転効果）の局所行列を計算する.

    K_geo = -p_n / dist * G^T * (I₃ - n⊗n) * G

    ここで:
    - G は gap gradient 行列 (3×12): ∂(pA - pB)/∂u
    - (I₃ - n⊗n) は法線に垂直な面への射影
    - p_n は法線接触反力
    - dist は中心線間距離

    この項は法線方向の回転に伴う剛性補正を表し、
    法線主項（k_pen * g_n g_n^T）と合わせて一貫接線を構成する。

    Args:
        pair: 接触ペア

    Returns:
        K_geo: (12, 12) 幾何剛性行列（局所 DOF 順序: A0, A1, B0, B1 × 3）
    """
    p_n = pair.state.p_n
    if p_n <= 0.0:
        return np.zeros((12, 12))

    s = pair.state.s
    t = pair.state.t
    n = pair.state.normal
    dist = pair.state.gap + pair.radius_a + pair.radius_b

    if dist < 1e-30:
        return np.zeros((12, 12))

    # Gap gradient 行列 G (3×12): ∂Δ/∂u where Δ = pA - pB
    # pA = (1-s)*xA0 + s*xA1, pB = (1-t)*xB0 + t*xB1
    G = np.zeros((3, 12))
    G[:, 0:3] = (1.0 - s) * np.eye(3)
    G[:, 3:6] = s * np.eye(3)
    G[:, 6:9] = -(1.0 - t) * np.eye(3)
    G[:, 9:12] = -t * np.eye(3)

    # 射影行列: I₃ - n⊗n（法線に垂直な面）
    P = np.eye(3) - np.outer(n, n)

    # K_geo = -p_n / dist * G^T * P * G  (12×12)
    PG = P @ G  # (3, 12)
    K_geo = -(p_n / dist) * (G.T @ PG)  # (12, 12)

    return K_geo


def _consistent_st_stiffness_local(
    pair: ContactPair,
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    ds_du: np.ndarray,
    dt_du: np.ndarray,
) -> np.ndarray:
    """接触点移動に伴う一貫接線剛性を計算する（Phase C6-L2）.

    PtP 接触の完全一貫接線：∂(s,t)/∂u による追加の剛性項。

    K_st = outer(g_n, ∂p_n/∂s·ds_du + ∂p_n/∂t·dt_du)
         + p_n * outer(∂g_n/∂s, ds_du)
         + p_n * outer(∂g_n/∂t, dt_du)

    ∂p_n/∂s = -k_pen * (n · dA)
    ∂p_n/∂t = k_pen * (n · dB)
    ∂g_n/∂s, ∂g_n/∂t は形状関数の s,t 微分と法線回転を含む。

    Args:
        pair: 接触ペア
        xA0, xA1, xB0, xB1: 変形後端点座標 (3,)
        ds_du: ∂s/∂u (12,)
        dt_du: ∂t/∂u (12,)

    Returns:
        K_st: (12, 12) 一貫接線剛性行列
    """
    p_n = pair.state.p_n
    if p_n <= 0.0:
        return np.zeros((12, 12))

    s = pair.state.s
    t = pair.state.t
    n = pair.state.normal
    k_pen = pair.state.k_pen

    dA = xA1 - xA0
    dB = xB1 - xB0
    delta = (1.0 - s) * xA0 + s * xA1 - ((1.0 - t) * xB0 + t * xB1)
    dist = float(np.linalg.norm(delta))

    if dist < 1e-30:
        return np.zeros((12, 12))

    # g_n: 形状ベクトル (12,)
    g_n = _contact_shape_vector(pair)

    # --- Term 1: ∂p_n|_{via s,t} ⊗ g_n ---
    dpn_ds = -k_pen * float(n @ dA)
    dpn_dt = k_pen * float(n @ dB)
    dpn_via_st = dpn_ds * ds_du + dpn_dt * dt_du  # (12,)
    K_st = np.outer(g_n, dpn_via_st)  # (12, 12)

    # --- Term 2: p_n * ∂g_n|_{via s,t} ---
    # ∂n/∂s, ∂n/∂t （法線回転）
    P_perp = np.eye(3) - np.outer(n, n)
    dn_ds = P_perp @ dA / dist
    dn_dt = -P_perp @ dB / dist

    # ∂g_n/∂s (12,)
    dg_ds = np.zeros(12)
    dg_ds[0:3] = n - (1.0 - s) * dn_ds
    dg_ds[3:6] = -n - s * dn_ds
    dg_ds[6:9] = (1.0 - t) * dn_ds
    dg_ds[9:12] = t * dn_ds

    # ∂g_n/∂t (12,)
    dg_dt = np.zeros(12)
    dg_dt[0:3] = -(1.0 - s) * dn_dt
    dg_dt[3:6] = -s * dn_dt
    dg_dt[6:9] = -n + (1.0 - t) * dn_dt
    dg_dt[9:12] = n + t * dn_dt

    K_st += p_n * np.outer(dg_ds, ds_du)
    K_st += p_n * np.outer(dg_dt, dt_du)

    return K_st


def _consistent_st_stiffness_at_gp(
    s_gp: float,
    t_closest: float,
    normal: np.ndarray,
    p_n_gp: float,
    k_pen: float,
    dist: float,
    xB0: np.ndarray,
    xB1: np.ndarray,
    dt_du: np.ndarray,
) -> np.ndarray:
    """Line contact Gauss 点での一貫接線剛性（Phase C6-L2）.

    Line contact では s_gp は固定なので ds/du = 0。
    dt/du のみが寄与する。

    K_st_gp = outer(g_n, ∂p_n/∂t·dt_du)
            + p_n * outer(∂g_n/∂t, dt_du)

    Args:
        s_gp: Gauss 点パラメータ（固定）
        t_closest: 最近接パラメータ
        normal: 法線ベクトル (3,)
        p_n_gp: Gauss 点での法線力
        k_pen: ペナルティ剛性
        dist: 中心線間距離
        xB0, xB1: セグメント B の端点 (3,)
        dt_du: ∂t/∂u (12,)

    Returns:
        K_st_gp: (12, 12) Gauss 点での一貫接線剛性
    """
    if p_n_gp <= 0.0 or dist < 1e-30:
        return np.zeros((12, 12))

    n = normal
    dB = xB1 - xB0
    s = s_gp
    t = t_closest

    # g_n at Gauss point
    from xkep_cae.contact.line_contact import _build_shape_vector_at_gp

    g_n = _build_shape_vector_at_gp(s, t, n)

    # ∂p_n/∂t = k_pen * (n · dB)  (s_gp 固定、ds_du = 0)
    dpn_dt = k_pen * float(n @ dB)
    K_st_gp = np.outer(g_n, dpn_dt * dt_du)

    # ∂n/∂t = -(I - n⊗n) · dB / dist
    P_perp = np.eye(3) - np.outer(n, n)
    dn_dt = -P_perp @ dB / dist

    # ∂g_n/∂t
    dg_dt = np.zeros(12)
    dg_dt[0:3] = -(1.0 - s) * dn_dt
    dg_dt[3:6] = -s * dn_dt
    dg_dt[6:9] = -n + (1.0 - t) * dn_dt
    dg_dt[9:12] = n + t * dn_dt

    K_st_gp += p_n_gp * np.outer(dg_dt, dt_du)

    return K_st_gp


def compute_contact_force(
    manager: ContactManager,
    ndof_total: int,
    *,
    ndof_per_node: int = 6,
    friction_forces: dict[int, np.ndarray] | None = None,
    node_coords: np.ndarray | None = None,
) -> np.ndarray:
    """全接触ペアの接触内力ベクトルを計算する.

    各 ACTIVE ペアについて法線 AL 反力を評価し、
    節点力として全体ベクトルに組み込む。
    摩擦力が指定されている場合は接線方向の摩擦力も加算する。

    line_contact=True（ContactConfig）の場合、Gauss 積分で接触力を評価する。
    この場合 node_coords が必要。

    Args:
        manager: 接触マネージャ
        ndof_total: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数
        friction_forces: {pair_index: q_t (2,)} 摩擦力マップ。
            None なら法線力のみ（後方互換）。
        node_coords: (n_nodes, 3) 変形後節点座標（line_contact 用）。
            line_contact=False の場合は不要。

    Returns:
        f_contact: (ndof_total,) 接触内力ベクトル
    """
    use_line_contact = manager.config.line_contact and node_coords is not None

    f_contact = np.zeros(ndof_total)

    for pair_idx, pair in enumerate(manager.pairs):
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        nodes = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]

        if use_line_contact:
            # --- Line-to-line Gauss 積分（Phase C6-L1）---
            from xkep_cae.contact.line_contact import (
                auto_select_n_gauss,
                compute_line_contact_force_local,
            )

            xA0 = node_coords[pair.nodes_a[0]]
            xA1 = node_coords[pair.nodes_a[1]]
            xB0 = node_coords[pair.nodes_b[0]]
            xB1 = node_coords[pair.nodes_b[1]]

            n_gp = manager.config.n_gauss
            if manager.config.n_gauss_auto:
                n_gp = auto_select_n_gauss(xA0, xA1, xB0, xB1, default=n_gp)

            f_local, total_p_n = compute_line_contact_force_local(pair, xA0, xA1, xB0, xB1, n_gp)
            pair.state.p_n = total_p_n

            for i, node in enumerate(nodes):
                for d in range(3):
                    gdof = node * ndof_per_node + d
                    f_contact[gdof] += f_local[i * 3 + d]
        else:
            # --- 従来の PtP 接触力 ---
            p_n = evaluate_normal_force(pair)

            if p_n > 0.0:
                g_n = _contact_shape_vector(pair)
                for i, node in enumerate(nodes):
                    for d in range(3):
                        gdof = node * ndof_per_node + d
                        f_contact[gdof] += p_n * g_n[i * 3 + d]

        # 摩擦力の組み込み（line contact でも PtP の代表点で評価）
        if friction_forces is not None and pair_idx in friction_forces:
            q_t = friction_forces[pair_idx]
            for axis in range(2):
                if abs(q_t[axis]) < 1e-30:
                    continue
                g_t = _contact_tangent_shape_vector(pair, axis)
                for i, node in enumerate(nodes):
                    for d in range(3):
                        gdof = node * ndof_per_node + d
                        f_contact[gdof] += q_t[axis] * g_t[i * 3 + d]

    return f_contact


def _add_local_to_coo(
    K_local: np.ndarray,
    gdofs: np.ndarray,
    rows: list[int],
    cols: list[int],
    data: list[float],
    tol: float = 1e-30,
) -> None:
    """局所行列を COO リストに追加する."""
    for i in range(12):
        for j in range(12):
            val = K_local[i, j]
            if abs(val) > tol:
                rows.append(gdofs[i])
                cols.append(gdofs[j])
                data.append(val)


def _compute_line_contact_st_stiffness_local(
    pair: ContactPair,
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    n_gauss: int,
) -> np.ndarray:
    """Line contact の Gauss 積分で ∂t/∂u 一貫接線を計算する（Phase C6-L2）.

    各 Gauss 点で ∂t/∂u を計算し、K_st 寄与を積分する。
    s_gp は固定（積分点パラメータ）なので ds_du = 0。

    Args:
        pair: 接触ペア
        xA0, xA1, xB0, xB1: 変形後端点座標 (3,)
        n_gauss: Gauss 積分点数

    Returns:
        K_st_local: (12, 12) 一貫接線剛性の Gauss 積分
    """
    from xkep_cae.contact.line_contact import (
        compute_t_jacobian_at_gp,
        gauss_legendre_01,
        project_point_to_segment,
    )

    gp, gw = gauss_legendre_01(n_gauss)
    K_st_local = np.zeros((12, 12))

    lambda_n = pair.state.lambda_n
    k_pen = pair.state.k_pen
    r_sum = pair.radius_a + pair.radius_b

    for s_gp, w in zip(gp, gw, strict=True):
        pA = (1.0 - s_gp) * xA0 + s_gp * xA1
        t_closest = project_point_to_segment(pA, xB0, xB1)
        pB = (1.0 - t_closest) * xB0 + t_closest * xB1

        diff = pA - pB
        dist = float(np.linalg.norm(diff))
        gap = dist - r_sum

        if dist > 1e-30:
            normal = diff / dist
        else:
            normal = pair.state.normal.copy()

        p_n_gp = max(0.0, lambda_n + k_pen * (-gap))
        if p_n_gp <= 0.0:
            continue

        dt_du = compute_t_jacobian_at_gp(s_gp, t_closest, xA0, xA1, xB0, xB1)
        if dt_du is None:
            continue

        K_st_gp = _consistent_st_stiffness_at_gp(
            s_gp,
            t_closest,
            normal,
            p_n_gp,
            k_pen,
            dist,
            xB0,
            xB1,
            dt_du,
        )
        K_st_local += w * K_st_gp

    return K_st_local


def compute_contact_stiffness(
    manager: ContactManager,
    ndof_total: int,
    *,
    ndof_per_node: int = 6,
    friction_tangents: dict[int, np.ndarray] | None = None,
    use_geometric_stiffness: bool = True,
    node_coords: np.ndarray | None = None,
) -> sp.csr_matrix:
    """全接触ペアの接触接線剛性行列を計算する.

    法線主項:
        K_n = k_pen * g_n g_n^T

    幾何剛性（Phase C5, use_geometric_stiffness=True）:
        K_geo = -p_n / dist * G^T * (I₃ - n⊗n) * G

    ∂(s,t)/∂u 一貫接線（Phase C6-L2, consistent_st_tangent=True）:
        K_st = ∂p_n/∂(s,t)·d(s,t)/du ⊗ g_n + p_n · ∂g_n/∂(s,t)·d(s,t)/du

    摩擦接線剛性（friction_tangents が指定されている場合）:
        K_f = Σ_axis Σ_axis2 D_t[a1,a2] * g_t1 g_t2^T

    line_contact=True（ContactConfig）の場合、Gauss 積分で剛性を評価する。
    この場合 node_coords が必要。

    Args:
        manager: 接触マネージャ
        ndof_total: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数
        friction_tangents: {pair_index: D_t (2,2)} 摩擦接線剛性マップ。
            None なら法線剛性のみ（後方互換）。
        use_geometric_stiffness: 幾何微分込み一貫接線の有効化（Phase C5）。
        node_coords: (n_nodes, 3) 変形後節点座標（line_contact / consistent_st_tangent 用）。

    Returns:
        K_contact: (ndof_total, ndof_total) CSR 形式接触剛性行列
    """
    use_line_contact = manager.config.line_contact and node_coords is not None
    use_st_tangent = manager.config.consistent_st_tangent and node_coords is not None

    # COO 形式で組み立て
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for pair_idx, pair in enumerate(manager.pairs):
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        # 全体DOFインデックス（並進DOFのみ抽出）
        nodes = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]
        gdofs = np.empty(12, dtype=int)
        for i, node in enumerate(nodes):
            for d in range(3):
                gdofs[i * 3 + d] = node * ndof_per_node + d

        if use_line_contact:
            # --- Line-to-line Gauss 積分（Phase C6-L1）---
            from xkep_cae.contact.line_contact import (
                auto_select_n_gauss,
                compute_line_contact_stiffness_local,
            )

            xA0 = node_coords[pair.nodes_a[0]]
            xA1 = node_coords[pair.nodes_a[1]]
            xB0 = node_coords[pair.nodes_b[0]]
            xB1 = node_coords[pair.nodes_b[1]]

            n_gp = manager.config.n_gauss
            if manager.config.n_gauss_auto:
                n_gp = auto_select_n_gauss(xA0, xA1, xB0, xB1, default=n_gp)

            K_line_local = compute_line_contact_stiffness_local(
                pair,
                xA0,
                xA1,
                xB0,
                xB1,
                n_gp,
                use_geometric_stiffness=use_geometric_stiffness,
            )

            # --- ∂t/∂u 一貫接線（Phase C6-L2, line contact）---
            if use_st_tangent and pair.state.p_n > 0.0:
                K_st_line = _compute_line_contact_st_stiffness_local(
                    pair,
                    xA0,
                    xA1,
                    xB0,
                    xB1,
                    n_gp,
                )
                K_line_local = K_line_local + K_st_line

            _add_local_to_coo(K_line_local, gdofs, rows, cols, data)
        else:
            # --- 従来の PtP 法線接線剛性（主項）---
            k_eff = normal_force_linearization(pair)
            if k_eff > 0.0:
                g_n = _contact_shape_vector(pair)
                for i in range(12):
                    for j in range(12):
                        val = k_eff * g_n[i] * g_n[j]
                        if abs(val) > 1e-30:
                            rows.append(gdofs[i])
                            cols.append(gdofs[j])
                            data.append(val)

            # --- 幾何剛性（Phase C5）---
            if use_geometric_stiffness and pair.state.p_n > 0.0:
                K_geo_local = _contact_geometric_stiffness_local(pair)
                _add_local_to_coo(K_geo_local, gdofs, rows, cols, data)

            # --- ∂(s,t)/∂u 一貫接線（Phase C6-L2, PtP）---
            if use_st_tangent and pair.state.p_n > 0.0:
                from xkep_cae.contact.geometry import compute_st_jacobian

                xA0 = node_coords[pair.nodes_a[0]]
                xA1 = node_coords[pair.nodes_a[1]]
                xB0 = node_coords[pair.nodes_b[0]]
                xB1 = node_coords[pair.nodes_b[1]]

                result = compute_st_jacobian(
                    pair.state.s,
                    pair.state.t,
                    xA0,
                    xA1,
                    xB0,
                    xB1,
                )
                if result is not None:
                    ds_du, dt_du = result
                    K_st_local = _consistent_st_stiffness_local(
                        pair,
                        xA0,
                        xA1,
                        xB0,
                        xB1,
                        ds_du,
                        dt_du,
                    )
                    _add_local_to_coo(K_st_local, gdofs, rows, cols, data)

        # --- 摩擦接線剛性（PtP 代表点で評価、line contact でも同様）---
        if friction_tangents is not None and pair_idx in friction_tangents:
            D_t = friction_tangents[pair_idx]
            g_t = [
                _contact_tangent_shape_vector(pair, 0),
                _contact_tangent_shape_vector(pair, 1),
            ]
            for a1 in range(2):
                for a2 in range(2):
                    d_val = D_t[a1, a2]
                    if abs(d_val) < 1e-30:
                        continue
                    for i in range(12):
                        for j in range(12):
                            val = d_val * g_t[a1][i] * g_t[a2][j]
                            if abs(val) > 1e-30:
                                rows.append(gdofs[i])
                                cols.append(gdofs[j])
                                data.append(val)

    if not rows:
        return sp.csr_matrix((ndof_total, ndof_total))

    K = sp.coo_matrix(
        (np.array(data), (np.array(rows, dtype=int), np.array(cols, dtype=int))),
        shape=(ndof_total, ndof_total),
    )
    K_csr = K.tocsr()
    K_csr.sum_duplicates()
    return K_csr
