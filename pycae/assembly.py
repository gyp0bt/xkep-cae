from __future__ import annotations
from typing import Optional
import numpy as np
import scipy.sparse as sp
import time

from .materials.elastic import constitutive_plane_strain
from .elements.quad4 import quad4_ke_plane_strain
from .elements.tri3 import tri3_ke_plane_strain
from .elements.tri6 import tri6_ke_plane_strain  # ★追加


def _edofs_for_quad4(n1: int, n2: int, n3: int, n4: int) -> np.ndarray:
    """Q4のグローバルDOF（u,v連結, 8個）"""
    return np.array(
        [
            2 * n1,
            2 * n1 + 1,
            2 * n2,
            2 * n2 + 1,
            2 * n3,
            2 * n3 + 1,
            2 * n4,
            2 * n4 + 1,
        ],
        dtype=int,
    )


def _eKe_scatter_lil(K_lil: sp.lil_matrix, edofs: np.ndarray, Ke: np.ndarray) -> None:
    """局所剛性 Ke を LIL 行列 K に散布加算（in-place）。"""
    for i, r in enumerate(edofs):
        K_lil[r, edofs] += Ke[i, :]


def estimate_stiffness_memory(
    num_nodes: int,
    num_elem_quads: int,
    num_elem_tris: int,
    num_elem_tri6: int,
    *,
    bytes_per_nnz: float = 16.0,
    safety_factor: float = 2.0,
) -> tuple[int, float]:
    """全体剛性行列 K の必要メモリをざっくり見積もる.

    Q4, TRI3, TRI6 の混在を仮定し，各要素の局所 K から
    nnz の上限を見積る。重複ゼロの分を見ていないので
    「やや多めの上限値」になる。

    Args:
        num_nodes: 節点数
        num_elem_quads: Q4 要素数
        num_elem_tris: TRI3 要素数
        num_elem_tri6: TRI6 要素数
        bytes_per_nnz: 1 非ゼロあたりのバイト数（データ + インデックス）。
            SciPy CSR なら 12〜16 byte 程度を想定。
        safety_factor: LIL→CSR 変換や Python オーバーヘッドを
            見込んだ安全係数。

    Returns:
        mem_bytes, mem_GB: 推定必要メモリ（バイト, ギガバイト）
    """
    # DOF 数（平面問題 2DOF/節点）
    ndof = 2 * num_nodes

    # 各要素型の局所 K サイズから「書き込みイベント」上限を見積もる
    nnz_upper = (
        64 * num_elem_quads  # Q4: 8x8
        + 36 * num_elem_tris  # TRI3: 6x6
        + 144 * num_elem_tri6  # TRI6: 12x12
    )

    # 行列サイズ上限も一応考慮（ほぼ意味はないが形式的に）
    max_nnz = ndof * ndof
    nnz_upper = min(nnz_upper, max_nnz)

    mem_bytes = int(nnz_upper * bytes_per_nnz * safety_factor)
    mem_GB = mem_bytes / (1024.0**3)
    return mem_bytes, mem_GB


def print_progress(
    current: int,
    total: int,
    t0: float,
    *,
    prefix: str = "",
    bar_length: int = 40,
) -> None:
    """簡易テキスト progress bar を表示する.

    Args:
        current: 現在の処理済み件数
        total: 全体件数
        prefix: 行頭に表示する文字列
        bar_length: バー表示の長さ（文字数）
    """
    if total <= 0:
        return

    ratio = current / total
    ratio = max(0.0, min(1.0, ratio))

    filled = int(bar_length * ratio)
    bar = "#" * filled + "-" * (bar_length - filled)
    percent = ratio * 100.0

    text = f"\r{prefix} [{bar}] {current}/{total} ({percent:5.1f}% in {time.time()-t0:5.2f} sec)"
    print(text, end="", flush=True)

    if current >= total:
        print()  # 改行


def _assemble_global_stiffness_mixed_legacy(
    nodes_xy: np.ndarray,
    conn_quads: Optional[np.ndarray],
    conn_tris: Optional[np.ndarray],
    conn_tri6: Optional[np.ndarray],
    E: float,
    nu: float,
    t: float = 1.0,
    *,
    show_progress: bool = True,
) -> sp.csr_matrix:
    """Q4, TRI3, TRI6 が混在する平面歪みメッシュの全体剛性 K（CSR）を構築する."""
    N = nodes_xy.shape[0]
    ndof = 2 * N
    K_lil = sp.lil_matrix((ndof, ndof), dtype=float)
    D = constitutive_plane_strain(E, nu)

    # 要素数カウント
    n_q = 0 if conn_quads is None else len(conn_quads)
    n_t3 = 0 if conn_tris is None else len(conn_tris)
    n_t6 = 0 if conn_tri6 is None else len(conn_tri6)
    n_total = n_q + n_t3 + n_t6

    # メモリ見積もり
    mem_bytes, mem_GB = estimate_stiffness_memory(
        num_nodes=N,
        num_elem_quads=n_q,
        num_elem_tris=n_t3,
        num_elem_tri6=n_t6,
    )
    print(
        f"[assemble] estimated K memory ~ {mem_GB:.2f} GB "
        f"({mem_bytes // (1024**2)} MB)"
    )

    elem_counter = 0
    progress_step = max(1, n_total // 100)  # 1% ごとくらい
    t0 = time.time()

    # --- Q4 ---
    if conn_quads is not None and n_q > 0:
        for row in conn_quads.astype(int):
            _, n1, n2, n3, n4 = row
            Ke = quad4_ke_plane_strain(nodes_xy[[n1, n2, n3, n4], :], D, t)
            edofs = _edofs_for_quad4(n1, n2, n3, n4)
            _eKe_scatter_lil(K_lil, edofs, Ke)

            elem_counter += 1
            if show_progress and (
                elem_counter % progress_step == 0 or elem_counter == n_total
            ):
                print_progress(elem_counter, n_total, t0=t0, prefix="Assemble K")

    # --- TRI3 ---
    if conn_tris is not None and n_t3 > 0:
        for row in conn_tris.astype(int):
            _, n1, n2, n3 = row
            Ke = tri3_ke_plane_strain(nodes_xy[[n1, n2, n3], :], D, t)
            edofs = np.array(
                [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1],
                dtype=int,
            )
            _eKe_scatter_lil(K_lil, edofs, Ke)

            elem_counter += 1
            if show_progress and (
                elem_counter % progress_step == 0 or elem_counter == n_total
            ):
                print_progress(elem_counter, n_total, t0=t0, prefix="Assemble K")

    # --- TRI6 ---
    if conn_tri6 is not None and n_t6 > 0:
        # ★ 要素ラベル付き / なし両対応:
        #   行の *最後の6列* を節点インデックスとして解釈する
        for row in conn_tri6.astype(int):
            _, n1, n2, n3, n4, n5, n6 = row  # 6列 or 7列どちらでもOK

            Ke = tri6_ke_plane_strain(nodes_xy[[n1, n2, n3, n4, n5, n6], :], D, t)
            edofs = np.array(
                [
                    2 * n1,
                    2 * n1 + 1,
                    2 * n2,
                    2 * n2 + 1,
                    2 * n3,
                    2 * n3 + 1,
                    2 * n4,
                    2 * n4 + 1,
                    2 * n5,
                    2 * n5 + 1,
                    2 * n6,
                    2 * n6 + 1,
                ],
                dtype=int,
            )
            _eKe_scatter_lil(K_lil, edofs, Ke)

            elem_counter += 1
            if show_progress and (
                elem_counter % progress_step == 0 or elem_counter == n_total
            ):
                print_progress(elem_counter, n_total, t0=t0, prefix="Assemble K")

    return K_lil.tocsr()


def assemble_global_stiffness_mixed(
    nodes_xy: np.ndarray,
    conn_quads: np.ndarray | None,
    conn_tris: np.ndarray | None,
    conn_tri6: np.ndarray | None,
    E: float,
    nu: float,
    t: float = 1.0,
    *,
    show_progress: bool = True,
) -> sp.csr_matrix:
    """Q4/TRI3/TRI6混在メッシュを COO 形式でアセンブルしてから CSR を作る.

    LIL に1要素ずつ足していく従来版より高速になることを狙う。

    Args:
        nodes_xy:
            (N,2) 内部インデックス順の節点座標。
        conn_quads:
            Q4 要素の接続配列。形状は (Ne4, 4) または (Ne4, 5) を想定。
            行の末尾 4 列を節点インデックスとして扱う。
        conn_tris:
            TRI3 要素の接続配列。形状は (Ne3, 3) または (Ne3, 4)。
            行の末尾 3 列を節点インデックスとして扱う。
        conn_tri6:
            TRI6 要素の接続配列。形状は (Ne6, 6) または (Ne6, 7)。
            行の末尾 6 列を節点インデックスとして扱う。
        E:
            ヤング率.
        nu:
            ポアソン比.
        t:
            厚み.
        show_progress:
            True のとき、要素アセンブリの進捗をTUI表示する。

    Returns:
        K: CSR 形式の全体剛性行列 (2N, 2N).
    """
    N = int(nodes_xy.shape[0])
    ndof = 2 * N
    D = constitutive_plane_strain(E, nu)

    # 要素数
    n_q = 0 if conn_quads is None else int(len(conn_quads))
    n_t3 = 0 if conn_tris is None else int(len(conn_tris))
    n_t6 = 0 if conn_tri6 is None else int(len(conn_tri6))
    n_total = n_q + n_t3 + n_t6

    if n_total == 0:
        raise ValueError(
            "要素が空です。conn_quads/conn_tris/conn_tri6 のいずれかを指定してください。"
        )

    # --- nnz 上限をざっくり見積もって配列確保 ---
    # Q4: 8x8=64, TRI3: 6x6=36, TRI6: 12x12=144
    nnz_est = 64 * n_q + 36 * n_t3 + 144 * n_t6
    # 念のため余裕を持たせる（重複分もあるのでそのままでOK）
    nnz_est = max(nnz_est, 1)

    rows = np.empty(nnz_est, dtype=np.int64)
    cols = np.empty(nnz_est, dtype=np.int64)
    data = np.empty(nnz_est, dtype=np.float64)
    k = 0  # 使用済みポインタ

    # 進捗表示用
    t0 = time.time()
    progress_step = max(1, n_total // 100)

    def _progress(elem_counter: int) -> None:
        """進捗表示（シンプル版）。"""
        if not show_progress:
            return
        if elem_counter % progress_step != 0 and elem_counter != n_total:
            return
        ratio = elem_counter / n_total
        bar_len = 40
        filled = int(bar_len * ratio)
        bar = "#" * filled + "-" * (bar_len - filled)
        elapsed = time.time() - t0
        print(
            f"\rAssemble K [{bar}] {elem_counter}/{n_total} "
            f"({ratio*100:5.1f}% in {elapsed:5.2f} sec)",
            end="",
            flush=True,
        )
        if elem_counter == n_total:
            print()

    elem_counter = 0

    # --- Q4 要素 ---
    if conn_quads is not None and n_q > 0:
        conn_quads_int = conn_quads.astype(int, copy=False)
        for row in conn_quads_int:
            # 行末尾4つを節点インデックスとみなす（[eid, n1,n2,n3,n4] / [n1..n4] 両対応）
            n1, n2, n3, n4 = row[-4:]
            Ke = quad4_ke_plane_strain(nodes_xy[[n1, n2, n3, n4], :], D, t)  # (8,8)
            edofs = np.array(
                [
                    2 * n1,
                    2 * n1 + 1,
                    2 * n2,
                    2 * n2 + 1,
                    2 * n3,
                    2 * n3 + 1,
                    2 * n4,
                    2 * n4 + 1,
                ],
                dtype=np.int64,
            )

            m = edofs.size  # 8
            block_nnz = m * m
            if k + block_nnz > nnz_est:
                # 万一見積もり不足なら2倍に拡張
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
            _progress(elem_counter)

    # --- TRI3 要素 ---
    if conn_tris is not None and n_t3 > 0:
        conn_tris_int = conn_tris.astype(int, copy=False)
        for row in conn_tris_int:
            n1, n2, n3 = row[-3:]
            Ke = tri3_ke_plane_strain(nodes_xy[[n1, n2, n3], :], D, t)  # (6,6)
            edofs = np.array(
                [
                    2 * n1,
                    2 * n1 + 1,
                    2 * n2,
                    2 * n2 + 1,
                    2 * n3,
                    2 * n3 + 1,
                ],
                dtype=np.int64,
            )

            m = edofs.size  # 6
            block_nnz = m * m
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
            _progress(elem_counter)

    # --- TRI6 要素 ---
    if conn_tri6 is not None and n_t6 > 0:
        conn_tri6_int = conn_tri6.astype(int, copy=False)
        for row in conn_tri6_int:
            n1, n2, n3, n4, n5, n6 = row[-6:]
            Ke = tri6_ke_plane_strain(
                nodes_xy[[n1, n2, n3, n4, n5, n6], :], D, t
            )  # (12,12)
            edofs = np.array(
                [
                    2 * n1,
                    2 * n1 + 1,
                    2 * n2,
                    2 * n2 + 1,
                    2 * n3,
                    2 * n3 + 1,
                    2 * n4,
                    2 * n4 + 1,
                    2 * n5,
                    2 * n5 + 1,
                    2 * n6,
                    2 * n6 + 1,
                ],
                dtype=np.int64,
            )

            m = edofs.size  # 12
            block_nnz = m * m
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
            _progress(elem_counter)

    # --- CSR 行列を構築 ---
    rows_used = rows[:k]
    cols_used = cols[:k]
    data_used = data[:k]

    K = sp.csr_matrix((data_used, (rows_used, cols_used)), shape=(ndof, ndof))
    K.sum_duplicates()
    return K


# =====================================================================
# Protocol ベース汎用アセンブリ（Phase 1.3 追加）
# =====================================================================

from pycae.core.element import ElementProtocol
from pycae.core.constitutive import ConstitutiveProtocol


def assemble_global_stiffness(
    nodes_xy: np.ndarray,
    element_groups: list[tuple[ElementProtocol, np.ndarray]],
    material: ConstitutiveProtocol,
    thickness: float = 1.0,
    *,
    show_progress: bool = True,
) -> sp.csr_matrix:
    """Protocol ベースの汎用全体剛性行列アセンブリ（COO→CSR）.

    要素型と接続配列のペアのリストを受け取り、任意の要素型を混在アセンブルする。
    既存の assemble_global_stiffness_mixed() と同等の機能を持つが、
    要素・材料の具体型に依存しない。

    Args:
        nodes_xy: (N, ndim) 内部インデックス順の節点座標
        element_groups: [(element, connectivity), ...] のリスト
            element: ElementProtocol に適合するオブジェクト
            connectivity: (Ne, nnodes) 内部インデックスの接続配列
        material: ConstitutiveProtocol に適合する材料オブジェクト
        thickness: 厚み（平面要素用）
        show_progress: 進捗表示の有無

    Returns:
        K: CSR形式の全体剛性行列 (ndof_total, ndof_total)
    """
    N = int(nodes_xy.shape[0])
    # 全グループの最初の要素からndof_per_nodeを取得
    ndof_per_node = element_groups[0][0].ndof_per_node
    ndof_total = ndof_per_node * N

    # nnz見積もり
    nnz_est = sum(
        elem.ndof * elem.ndof * len(conn)
        for elem, conn in element_groups
    )
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
            node_ids = row[-elem.nnodes:]
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
            if show_progress and (
                elem_counter % progress_step == 0 or elem_counter == n_total
            ):
                ratio = elem_counter / n_total
                bar_len = 40
                filled = int(bar_len * ratio)
                bar = "#" * filled + "-" * (bar_len - filled)
                elapsed = time.time() - t0
                print(
                    f"\rAssemble K [{bar}] {elem_counter}/{n_total} "
                    f"({ratio*100:5.1f}% in {elapsed:5.2f} sec)",
                    end="", flush=True,
                )
                if elem_counter == n_total:
                    print()

    K = sp.csr_matrix(
        (data[:k], (rows[:k], cols[:k])), shape=(ndof_total, ndof_total)
    )
    K.sum_duplicates()
    return K
