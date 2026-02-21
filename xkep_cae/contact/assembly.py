"""接触内力・接線剛性のアセンブリ.

Phase C2: 法線接触力のグローバルベクトル/行列への組み込み。
Phase C3: 摩擦力 + 摩擦接線剛性の追加。

接触力の節点配分:
    セグメント A: p(s) = xA0 + s*(xA1 - xA0)
    → 節点 A0 に (1-s)*f,  節点 A1 に s*f を配分
    セグメント B: 反作用力を同様に (1-t), t で配分

法線接線剛性（主項のみ, v0.1）:
    K_n = k_pen * g_n g_n^T

摩擦接線剛性（stick 時）:
    K_f = k_t * (g_t1 g_t1^T + g_t2 g_t2^T)

設計仕様: docs/contact/beam_beam_contact_spec_v0.1.md §5, §7
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


def compute_contact_force(
    manager: ContactManager,
    ndof_total: int,
    *,
    ndof_per_node: int = 6,
    friction_forces: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """全接触ペアの接触内力ベクトルを計算する.

    各 ACTIVE ペアについて法線 AL 反力を評価し、
    節点力として全体ベクトルに組み込む。
    摩擦力が指定されている場合は接線方向の摩擦力も加算する。

    Args:
        manager: 接触マネージャ
        ndof_total: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数
        friction_forces: {pair_index: q_t (2,)} 摩擦力マップ。
            None なら法線力のみ（後方互換）。

    Returns:
        f_contact: (ndof_total,) 接触内力ベクトル
    """
    f_contact = np.zeros(ndof_total)

    for pair_idx, pair in enumerate(manager.pairs):
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        # AL 反力を評価
        p_n = evaluate_normal_force(pair)

        nodes = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]

        # 法線力の組み込み
        if p_n > 0.0:
            g_n = _contact_shape_vector(pair)
            for i, node in enumerate(nodes):
                for d in range(3):
                    gdof = node * ndof_per_node + d
                    f_contact[gdof] += p_n * g_n[i * 3 + d]

        # 摩擦力の組み込み
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


def compute_contact_stiffness(
    manager: ContactManager,
    ndof_total: int,
    *,
    ndof_per_node: int = 6,
    friction_tangents: dict[int, np.ndarray] | None = None,
) -> sp.csr_matrix:
    """全接触ペアの接触接線剛性行列を計算する.

    法線主項:
        K_n = k_pen * g_n g_n^T

    摩擦接線剛性（friction_tangents が指定されている場合）:
        K_f = Σ_axis Σ_axis2 D_t[a1,a2] * g_t1 g_t2^T

    幾何微分（法線変化 dn/du）は v0.2 で追加予定。

    Args:
        manager: 接触マネージャ
        ndof_total: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数
        friction_tangents: {pair_index: D_t (2,2)} 摩擦接線剛性マップ。
            None なら法線剛性のみ（後方互換）。

    Returns:
        K_contact: (ndof_total, ndof_total) CSR 形式接触剛性行列
    """
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

        # --- 法線接線剛性 ---
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

        # --- 摩擦接線剛性 ---
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
