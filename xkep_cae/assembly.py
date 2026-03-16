"""Protocol ベース汎用アセンブリ.

任意の ElementProtocol 適合要素を混在させて全体剛性行列を構築する。
COO 形式で要素ごとの寄与を蓄積し、最終的に CSR 行列を生成する。

Phase S2+: COO インデックスのベクトル化 + 共有メモリ並列化（status-090）。
- _vectorized_coo_indices: DOF の repeat/tile を要素グループ単位で一括計算
- 共有メモリ: multiprocessing.shared_memory で data 配列を共有し、
  ProcessPoolExecutor の IPC オーバーヘッド（pickle + pipe）を回避。
  10万要素超の大規模撚線モデルに対応。
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from multiprocessing import shared_memory

import numpy as np
import scipy.sparse as sp

from xkep_cae.core.constitutive import ConstitutiveProtocol
from xkep_cae.core.element import ElementProtocol

# 並列化の最小要素数閾値（これ未満は逐次実行）
_PARALLEL_MIN_ELEMENTS = 4096


# ========== COO ベクトル化ヘルパー ==========


def _vectorized_coo_indices(
    conn_int: np.ndarray,
    nnodes: int,
    ndof_per_node: int,
    m: int,
) -> tuple[np.ndarray, np.ndarray]:
    """要素グループの COO row/col インデックスをベクトル化計算.

    per-element の dof_indices + np.repeat + np.tile を排除し、
    全要素分を一括で計算する。

    Args:
        conn_int: (n_elem, ncols) 接続配列（整数）
        nnodes: 要素あたりの節点数
        ndof_per_node: 節点あたりの自由度数
        m: 要素あたりの総自由度数 (= nnodes * ndof_per_node)

    Returns:
        (rows, cols): それぞれ (n_elem * m * m,) の int64 配列
    """
    n_elem = len(conn_int)
    conn_nodes = conn_int[:, -nnodes:]  # (n_elem, nnodes)

    # 全要素の DOF インデックスを一括計算: (n_elem, m)
    dof_offsets = np.arange(ndof_per_node, dtype=np.int64)
    all_edofs = (conn_nodes[:, :, None] * ndof_per_node + dof_offsets[None, None, :]).reshape(
        n_elem, m
    )

    # rows: 各 DOF を m 回繰り返し, cols: m 回タイル
    rows = np.repeat(all_edofs, m, axis=1).ravel()
    cols = np.tile(all_edofs, (1, m)).ravel()

    return rows, cols


# ========== 後方互換: ProcessPoolExecutor 用バッチ関数 ==========


def _compute_element_batch(
    elem: ElementProtocol,
    nodes_xy: np.ndarray,
    conn_batch: np.ndarray,
    material: ConstitutiveProtocol,
    thickness: float | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """要素バッチの剛性行列と DOF インデックスを計算する（後方互換）."""
    results = []
    for row in conn_batch:
        node_ids = row[-elem.nnodes :]
        coords = nodes_xy[node_ids]
        Ke = elem.local_stiffness(coords, material, thickness)
        edofs = elem.dof_indices(node_ids)
        results.append((Ke, edofs))
    return results


# ========== 共有メモリワーカー ==========

# ワーカープロセスのグローバル状態（Pool initializer で設定）
_shm_worker_state: dict = {}


def _shm_worker_init(
    shm_name: str,
    data_size: int,
    elem: ElementProtocol,
    nodes_xy: np.ndarray,
    material: ConstitutiveProtocol,
    thickness: float | None,
) -> None:
    """共有メモリワーカーの初期化（Pool initializer）.

    各ワーカープロセスで1回だけ呼ばれる。重い引数（nodes_xy, elem, material）を
    ワーカーローカル状態に保持し、タスクごとの pickle オーバーヘッドを回避する。
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    _shm_worker_state.update(
        {
            "shm": shm,
            "data": np.ndarray(data_size, dtype=np.float64, buffer=shm.buf),
            "elem": elem,
            "nodes_xy": nodes_xy,
            "material": material,
            "thickness": thickness,
        }
    )


def _shm_worker_compute(args: tuple) -> None:
    """共有メモリワーカー: Ke を計算して data 配列に直接書き込み.

    args: (conn_batch, offset, block_nnz, nnodes)
    - conn_batch: (batch_size, ncols) 接続配列
    - offset: data 配列内の書き込み開始位置
    - block_nnz: 1要素あたりの nnz (= m * m)
    - nnodes: 要素あたりの節点数
    """
    conn_batch, offset, block_nnz, nnodes = args
    s = _shm_worker_state
    data = s["data"]
    elem = s["elem"]
    nodes_xy = s["nodes_xy"]
    material = s["material"]
    thickness = s["thickness"]

    for i, row in enumerate(conn_batch):
        node_ids = row[-nnodes:]
        coords = nodes_xy[node_ids]
        Ke = elem.local_stiffness(coords, material, thickness)
        pos = offset + i * block_nnz
        data[pos : pos + block_nnz] = Ke.ravel()


# ========== 公開 API ==========


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
    n_jobs >= 2 のとき共有メモリ並列化で要素行列計算を並列化する。

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
    """逐次アセンブリ（COO ベクトル化）.

    DOF インデックス → rows/cols を要素グループ単位でベクトル化計算。
    要素剛性行列 Ke の計算のみ要素ループ。
    """
    nnz_total = sum(elem.ndof * elem.ndof * len(conn) for elem, conn in element_groups)
    nnz_total = max(nnz_total, 1)

    rows = np.empty(nnz_total, dtype=np.int64)
    cols = np.empty(nnz_total, dtype=np.int64)
    data = np.empty(nnz_total, dtype=np.float64)

    t0 = time.time()
    progress_step = max(1, n_total // 100)
    elem_counter = 0
    k = 0

    for elem, conn in element_groups:
        conn_int = conn.astype(int, copy=False)
        m = elem.ndof
        block_nnz = m * m
        n_elem = len(conn_int)
        nnodes = elem.nnodes
        ndof_per_node = elem.ndof_per_node

        # COO インデックスをベクトル化で一括計算
        group_nnz = n_elem * block_nnz
        r, c = _vectorized_coo_indices(conn_int, nnodes, ndof_per_node, m)
        rows[k : k + group_nnz] = r
        cols[k : k + group_nnz] = c

        # 要素剛性行列の計算（ここだけ要素ループ）
        for i, row in enumerate(conn_int):
            node_ids = row[-nnodes:]
            coords = nodes_xy[node_ids]
            Ke = elem.local_stiffness(coords, material, thickness)
            pos = k + i * block_nnz
            data[pos : pos + block_nnz] = Ke.ravel()

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

        k += group_nnz

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
    """共有メモリ並列アセンブリ.

    COO インデックスのベクトル化 + 共有メモリで data 配列を直接並列書き込み。
    ProcessPoolExecutor の結果シリアライズ（pickle + pipe）を完全に回避。

    構造:
      1. rows/cols はメインプロセスでベクトル化計算（高速）
      2. data 配列を shared_memory で確保
      3. mp.Pool の各ワーカーが Ke を計算し data に直接書き込み
      4. 各ワーカーは排他的な領域に書き込むためロック不要
    """
    nnz_total = sum(elem.ndof * elem.ndof * len(conn) for elem, conn in element_groups)
    nnz_total = max(nnz_total, 1)

    # COO インデックスをベクトル化で事前計算（メインプロセス）
    rows = np.empty(nnz_total, dtype=np.int64)
    cols = np.empty(nnz_total, dtype=np.int64)

    # data 配列を共有メモリで確保
    shm_size = max(8, nnz_total * 8)  # 最低 8 bytes
    shm = shared_memory.SharedMemory(create=True, size=shm_size)
    data_shm = np.ndarray(nnz_total, dtype=np.float64, buffer=shm.buf)

    try:
        t0 = time.time()
        k = 0

        for elem, conn in element_groups:
            conn_int = conn.astype(int, copy=False)
            m = elem.ndof
            block_nnz = m * m
            n_elem = len(conn_int)
            nnodes = elem.nnodes
            ndof_per_node = elem.ndof_per_node

            # COO インデックスのベクトル化
            group_nnz = n_elem * block_nnz
            r, c = _vectorized_coo_indices(conn_int, nnodes, ndof_per_node, m)
            rows[k : k + group_nnz] = r
            cols[k : k + group_nnz] = c

            # バッチ分割（各ワーカーに均等配分）
            batch_size = max(1, n_elem // n_jobs)
            tasks = []
            for start in range(0, n_elem, batch_size):
                end = min(start + batch_size, n_elem)
                tasks.append(
                    (
                        conn_int[start:end],
                        k + start * block_nnz,
                        block_nnz,
                        nnodes,
                    )
                )

            # 共有メモリ並列計算
            # initializer で重い引数（elem, nodes_xy, material）を1回だけ pickle
            # タスクごとには conn_batch + offset のみ（軽量）
            with mp.Pool(
                n_jobs,
                initializer=_shm_worker_init,
                initargs=(shm.name, nnz_total, elem, nodes_xy, material, thickness),
            ) as pool:
                pool.map(_shm_worker_compute, tasks)

            k += group_nnz

        if show_progress:
            elapsed = time.time() - t0
            print(
                f"Assemble K (shared memory, {n_jobs} workers): "
                f"{n_total} elements in {elapsed:.2f} sec"
            )

        # CSR 行列構築（data をコピーしてから共有メモリを解放）
        K = sp.csr_matrix(
            (data_shm[:k].copy(), (rows[:k], cols[:k])),
            shape=(ndof_total, ndof_total),
        )
    finally:
        shm.close()
        shm.unlink()

    K.sum_duplicates()
    return K
