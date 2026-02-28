"""Protocol ベース汎用アセンブリ.

任意の ElementProtocol 適合要素を混在させて全体剛性行列を構築する。
COO 形式で要素ごとの寄与を蓄積し、最終的に CSR 行列を生成する。

Phase S2: n_jobs >= 2 のとき ThreadPoolExecutor で要素行列を並列計算する。
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp

from xkep_cae.core.constitutive import ConstitutiveProtocol
from xkep_cae.core.element import ElementProtocol

# 並列化の最小要素数閾値（これ未満は逐次実行）
_PARALLEL_MIN_ELEMENTS = 64


def _compute_element_batch(
    elem: ElementProtocol,
    nodes_xy: np.ndarray,
    conn_batch: np.ndarray,
    material: ConstitutiveProtocol,
    thickness: float | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """要素バッチの剛性行列と DOF インデックスを計算する."""
    results = []
    for row in conn_batch:
        node_ids = row[-elem.nnodes :]
        coords = nodes_xy[node_ids]
        Ke = elem.local_stiffness(coords, material, thickness)
        edofs = elem.dof_indices(node_ids)
        results.append((Ke, edofs))
    return results


def assemble_global_stiffness(
    nodes_xy: np.ndarray,
    element_groups: list[tuple[ElementProtocol, np.ndarray]],
    material: ConstitutiveProtocol,
    *,
    thickness: float | None = None,
    show_progress: bool = True,
    n_jobs: int = 1,
) -> sp.csr_matrix:
    """Protocol ベースの汎用全体剛性行列アセンブリ（COO→CSR）.

    要素型と接続配列のペアのリストを受け取り、任意の要素型を混在アセンブルする。
    n_jobs >= 2 のとき ThreadPoolExecutor で要素行列計算を並列化する。

    Args:
        nodes_xy: (N, ndim) 内部インデックス順の節点座標
        element_groups: [(element, connectivity), ...] のリスト
            element: ElementProtocol に適合するオブジェクト
            connectivity: (Ne, nnodes) 内部インデックスの接続配列
        material: ConstitutiveProtocol に適合する材料オブジェクト
        thickness: 厚み（平面要素用）。梁要素など不要な場合は None。
        show_progress: 進捗表示の有無
        n_jobs: 並列ワーカー数。1=逐次、-1=全CPUコア使用

    Returns:
        K: CSR形式の全体剛性行列 (ndof_total, ndof_total)
    """
    N = int(nodes_xy.shape[0])
    ndof_per_node = element_groups[0][0].ndof_per_node
    ndof_total = ndof_per_node * N

    n_total = sum(len(conn) for _, conn in element_groups)

    # n_jobs の解決
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs < 1:
        n_jobs = 1

    # 小規模問題は逐次実行
    use_parallel = n_jobs >= 2 and n_total >= _PARALLEL_MIN_ELEMENTS

    if use_parallel:
        return _assemble_parallel(
            nodes_xy,
            element_groups,
            material,
            ndof_total,
            n_total,
            thickness=thickness,
            show_progress=show_progress,
            n_jobs=n_jobs,
        )

    return _assemble_sequential(
        nodes_xy,
        element_groups,
        material,
        ndof_total,
        n_total,
        thickness=thickness,
        show_progress=show_progress,
    )


def _assemble_sequential(
    nodes_xy: np.ndarray,
    element_groups: list[tuple[ElementProtocol, np.ndarray]],
    material: ConstitutiveProtocol,
    ndof_total: int,
    n_total: int,
    *,
    thickness: float | None = None,
    show_progress: bool = True,
) -> sp.csr_matrix:
    """逐次アセンブリ（従来ロジック）."""
    nnz_est = sum(elem.ndof * elem.ndof * len(conn) for elem, conn in element_groups)
    nnz_est = max(nnz_est, 1)

    rows = np.empty(nnz_est, dtype=np.int64)
    cols = np.empty(nnz_est, dtype=np.int64)
    data = np.empty(nnz_est, dtype=np.float64)
    k = 0

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


def _assemble_parallel(
    nodes_xy: np.ndarray,
    element_groups: list[tuple[ElementProtocol, np.ndarray]],
    material: ConstitutiveProtocol,
    ndof_total: int,
    n_total: int,
    *,
    thickness: float | None = None,
    show_progress: bool = True,
    n_jobs: int = 2,
) -> sp.csr_matrix:
    """ThreadPoolExecutor による並列アセンブリ.

    要素を n_jobs 個のバッチに分割し、各バッチで要素剛性行列を並列計算する。
    COO への書き込みは逐次（メモリ安全）。
    """
    nnz_est = sum(elem.ndof * elem.ndof * len(conn) for elem, conn in element_groups)
    nnz_est = max(nnz_est, 1)

    rows = np.empty(nnz_est, dtype=np.int64)
    cols = np.empty(nnz_est, dtype=np.int64)
    data = np.empty(nnz_est, dtype=np.float64)
    k = 0

    t0 = time.time()

    for elem, conn in element_groups:
        conn_int = conn.astype(int, copy=False)
        m = elem.ndof
        block_nnz = m * m
        n_elem = len(conn_int)

        # バッチ分割
        batch_size = max(1, n_elem // n_jobs)
        batches = [conn_int[i : i + batch_size] for i in range(0, n_elem, batch_size)]

        # 並列計算
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = [
                pool.submit(_compute_element_batch, elem, nodes_xy, batch, material, thickness)
                for batch in batches
            ]
            all_results = [f.result() for f in futures]

        # COO 蓄積（逐次）
        for batch_results in all_results:
            for Ke, edofs in batch_results:
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

    if show_progress:
        elapsed = time.time() - t0
        print(f"Assemble K (parallel, {n_jobs} workers): {n_total} elements in {elapsed:.2f} sec")

    K = sp.csr_matrix((data[:k], (rows[:k], cols[:k])), shape=(ndof_total, ndof_total))
    K.sum_duplicates()
    return K
