"""接触内力・接線剛性のアセンブリ.

Phase C2: 法線接触力のグローバルベクトル/行列への組み込み。

接触力の節点配分:
    セグメント A: p(s) = xA0 + s*(xA1 - xA0)
    → 節点 A0 に (1-s)*f,  節点 A1 に s*f を配分
    セグメント B: 反作用力を同様に (1-t), t で配分

接触接線剛性（主項のみ, v0.1）:
    K_c = k_pen * (N_A - N_B)^T n n^T (N_A - N_B)
    ここで N_A = [(1-s)I, sI], N_B = [(1-t)I, tI]

設計仕様: docs/contact/beam_beam_contact_spec_v0.1.md §7
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


def compute_contact_force(
    manager: ContactManager,
    ndof_total: int,
    *,
    ndof_per_node: int = 6,
) -> np.ndarray:
    """全接触ペアの接触内力ベクトルを計算する.

    各 ACTIVE ペアについて法線 AL 反力を評価し、
    節点力として全体ベクトルに組み込む。

    Args:
        manager: 接触マネージャ
        ndof_total: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数

    Returns:
        f_contact: (ndof_total,) 接触内力ベクトル
    """
    f_contact = np.zeros(ndof_total)

    for pair in manager.pairs:
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        # AL 反力を評価
        p_n = evaluate_normal_force(pair)
        if p_n <= 0.0:
            continue

        # 形状ベクトル（4節点 × 3DOF = 12成分）
        g = _contact_shape_vector(pair)

        # 全体ベクトルへの組み込み（並進DOFのみ）
        nodes = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]
        for i, node in enumerate(nodes):
            for d in range(3):  # 並進DOFのみ
                gdof = node * ndof_per_node + d
                f_contact[gdof] += p_n * g[i * 3 + d]

    return f_contact


def compute_contact_stiffness(
    manager: ContactManager,
    ndof_total: int,
    *,
    ndof_per_node: int = 6,
) -> sp.csr_matrix:
    """全接触ペアの接触接線剛性行列を計算する（主項のみ）.

    各 ACTIVE ペアについて、ペナルティ/AL 主項の接線剛性:
        K_c = k_eff * g g^T
    ここで k_eff = k_pen（接触中）or 0、g は形状ベクトル。

    幾何微分（法線変化 dn/du）は v0.2 で追加予定。

    Args:
        manager: 接触マネージャ
        ndof_total: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数

    Returns:
        K_contact: (ndof_total, ndof_total) CSR 形式接触剛性行列
    """
    # COO 形式で組み立て
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for pair in manager.pairs:
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        # 接線剛性係数
        k_eff = normal_force_linearization(pair)
        if k_eff <= 0.0:
            continue

        # 形状ベクトル
        g = _contact_shape_vector(pair)

        # 全体DOFインデックス（並進DOFのみ抽出）
        nodes = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]
        gdofs = np.empty(12, dtype=int)
        for i, node in enumerate(nodes):
            for d in range(3):
                gdofs[i * 3 + d] = node * ndof_per_node + d

        # K_c = k_eff * g g^T をCOOに追加
        for i in range(12):
            for j in range(12):
                val = k_eff * g[i] * g[j]
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
