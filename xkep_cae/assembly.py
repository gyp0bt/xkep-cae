"""Protocol ベース汎用アセンブリ.

任意の ElementProtocol 適合要素を混在させて全体剛性行列を構築する。
COO 形式で要素ごとの寄与を蓄積し、最終的に CSR 行列を生成する。
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from xkep_cae.core.constitutive import ConstitutiveProtocol
from xkep_cae.core.element import ElementProtocol


def assemble_global_stiffness(
    nodes_xy: np.ndarray,
    element_groups: list[tuple[ElementProtocol, np.ndarray]],
    material: ConstitutiveProtocol,
    *,
    thickness: float | None = None,
    show_progress: bool = True,
) -> sp.csr_matrix:
    """Protocol ベースの汎用全体剛性行列アセンブリ（COO→CSR）.

    要素型と接続配列のペアのリストを受け取り、任意の要素型を混在アセンブルする。

    Args:
        nodes_xy: (N, ndim) 内部インデックス順の節点座標
        element_groups: [(element, connectivity), ...] のリスト
            element: ElementProtocol に適合するオブジェクト
            connectivity: (Ne, nnodes) 内部インデックスの接続配列
        material: ConstitutiveProtocol に適合する材料オブジェクト
        thickness: 厚み（平面要素用）。梁要素など不要な場合は None。
        show_progress: 進捗表示の有無

    Returns:
        K: CSR形式の全体剛性行列 (ndof_total, ndof_total)
    """
    N = int(nodes_xy.shape[0])
    ndof_per_node = element_groups[0][0].ndof_per_node
    ndof_total = ndof_per_node * N

    # nnz見積もり
    nnz_est = sum(elem.ndof * elem.ndof * len(conn) for elem, conn in element_groups)
    nnz_est = max(nnz_est, 1)

    rows = np.empty(nnz_est, dtype=np.int64)
    cols = np.empty(nnz_est, dtype=np.int64)
    data = np.empty(nnz_est, dtype=np.float64)
    k = 0

    n_total = sum(len(conn) for _, conn in element_groups)
    t0 = time.time()
    progress_step = max(1, n_total // 100)
    elem_counter = 0

    for elem, conn in element_groups:
        conn_int = conn.astype(int, copy=False)
        m = elem.ndof
        block_nnz = m * m

        for row in conn_int:
            node_ids = row[-elem.nnodes :]
            coords = nodes_xy[node_ids]
            Ke = elem.local_stiffness(coords, material, thickness)
            edofs = elem.dof_indices(node_ids)

            if k + block_nnz > nnz_est:
                new_nnz = max(nnz_est * 2, k + block_nnz)
                rows.resize(new_nnz, refcheck=False)
                cols.resize(new_nnz, refcheck=False)
                data.resize(new_nnz, refcheck=False)
                nnz_est = new_nnz

            rows[k : k + block_nnz] = np.repeat(edofs, m)
            cols[k : k + block_nnz] = np.tile(edofs, m)
            data[k : k + block_nnz] = Ke.ravel()
            k += block_nnz

            elem_counter += 1
            if show_progress and (elem_counter % progress_step == 0 or elem_counter == n_total):
                ratio = elem_counter / n_total
                bar_len = 40
                filled = int(bar_len * ratio)
                bar = "#" * filled + "-" * (bar_len - filled)
                elapsed = time.time() - t0
                print(
                    f"\rAssemble K [{bar}] {elem_counter}/{n_total} "
                    f"({ratio * 100:5.1f}% in {elapsed:5.2f} sec)",
                    end="",
                    flush=True,
                )
                if elem_counter == n_total:
                    print()

    K = sp.csr_matrix((data[:k], (rows[:k], cols[:k])), shape=(ndof_total, ndof_total))
    K.sum_duplicates()
    return K
