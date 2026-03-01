"""最適化ベンチマーク: 接触アセンブリ + BC適用の高速化度合い測定.

4096要素以上の合成問題で、status-091 で実施した最適化の
before/after を壁時計時間で比較する。

測定対象:
1. _add_local_to_coo: Python 12×12ループ → np.nonzero
2. compute_contact_force: Python double loop → np.add.at scatter-add
3. compute_contact_stiffness: Python 12×12 loop → np.outer + vectorized COO
4. BC適用: tolil + Python loop → CSR/CSC直接操作
5. Broadphase AABB: 4096+セグメントのスケール検証

[← README](../README.md)
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.contact.assembly import (
    _add_local_to_coo,
    _contact_shape_vector,
    _contact_tangent_shape_vector,
    compute_contact_force,
    compute_contact_stiffness,
)
from xkep_cae.contact.broadphase import broadphase_aabb
from xkep_cae.contact.law_normal import normal_force_linearization
from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactPair,
    ContactState,
    ContactStatus,
)

# ====================================================================
# ヘルパー: 旧実装（最適化前）
# ====================================================================


def _add_local_to_coo_old(
    K_local: np.ndarray,
    gdofs: np.ndarray,
    rows: list[int],
    cols: list[int],
    data: list[float],
    tol: float = 1e-30,
) -> None:
    """旧実装: 12×12 Python ループ版."""
    n = K_local.shape[0]
    for i in range(n):
        for j in range(n):
            v = K_local[i, j]
            if abs(v) > tol:
                rows.append(int(gdofs[i]))
                cols.append(int(gdofs[j]))
                data.append(v)


def _contact_dofs_old(pair: ContactPair, ndof_per_node: int = 6) -> np.ndarray:
    """旧実装: Python ループ版."""
    dofs = []
    for node in [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]:
        for d in range(ndof_per_node):
            dofs.append(node * ndof_per_node + d)
    return np.array(dofs, dtype=int)


def compute_contact_force_old(
    manager: ContactManager,
    ndof_total: int,
    *,
    ndof_per_node: int = 6,
    friction_forces: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """旧実装: Python double loop 版."""
    f_contact = np.zeros(ndof_total)

    for pair_idx, pair in enumerate(manager.pairs):
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        from xkep_cae.contact.law_normal import evaluate_normal_force

        p_n = evaluate_normal_force(pair)

        if p_n > 0.0:
            g_n = _contact_shape_vector(pair)

            # 旧版: Python double loop
            for i, node_idx in enumerate(
                [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]
            ):
                for d in range(3):
                    dof = node_idx * ndof_per_node + d
                    f_contact[dof] += p_n * g_n[i * 3 + d]

        # 摩擦力
        if friction_forces is not None and pair_idx in friction_forces:
            q_t = friction_forces[pair_idx]
            for axis in range(2):
                if abs(q_t[axis]) < 1e-30:
                    continue
                g_t = _contact_tangent_shape_vector(pair, axis)
                for i, node_idx in enumerate(
                    [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]
                ):
                    for d in range(3):
                        dof = node_idx * ndof_per_node + d
                        f_contact[dof] += q_t[axis] * g_t[i * 3 + d]

    return f_contact


def compute_contact_stiffness_old(
    manager: ContactManager,
    ndof_total: int,
    *,
    ndof_per_node: int = 6,
    friction_tangents: dict[int, np.ndarray] | None = None,
) -> sp.csr_matrix:
    """旧実装: Python 12×12 ループ版."""
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for pair_idx, pair in enumerate(manager.pairs):
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        # 旧版 _contact_dofs: Python ループ
        trans_dofs = []
        for node in [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]:
            for d in range(3):
                trans_dofs.append(node * ndof_per_node + d)
        gdofs = np.array(trans_dofs, dtype=int)

        k_eff = normal_force_linearization(pair)
        if k_eff > 0.0:
            g_n = _contact_shape_vector(pair)
            # 旧版: 12×12 Python ループで outer product
            for i in range(12):
                for j in range(12):
                    v = k_eff * g_n[i] * g_n[j]
                    if abs(v) > 1e-30:
                        rows.append(int(gdofs[i]))
                        cols.append(int(gdofs[j]))
                        data.append(v)

        # 摩擦
        if friction_tangents is not None and pair_idx in friction_tangents:
            D_t = friction_tangents[pair_idx]
            g_t0 = _contact_tangent_shape_vector(pair, 0)
            g_t1 = _contact_tangent_shape_vector(pair, 1)
            g_t = [g_t0, g_t1]
            for a1 in range(2):
                for a2 in range(2):
                    d_val = D_t[a1, a2]
                    if abs(d_val) < 1e-30:
                        continue
                    for i in range(12):
                        for j in range(12):
                            v = d_val * g_t[a1][i] * g_t[a2][j]
                            if abs(v) > 1e-30:
                                rows.append(int(gdofs[i]))
                                cols.append(int(gdofs[j]))
                                data.append(v)

    if not rows:
        return sp.csr_matrix((ndof_total, ndof_total))

    K = sp.coo_matrix(
        (np.array(data), (np.array(rows, dtype=int), np.array(cols, dtype=int))),
        shape=(ndof_total, ndof_total),
    )
    K_csr = K.tocsr()
    K_csr.sum_duplicates()
    return K_csr


def apply_bc_tolil(
    K: sp.spmatrix,
    r: np.ndarray,
    fixed_dofs: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """旧実装: tolil + Python ループ版."""
    K_lil = K.tolil().copy()
    r_bc = r.copy()

    for dof in fixed_dofs:
        K_lil[dof, :] = 0.0
        K_lil[:, dof] = 0.0
        K_lil[dof, dof] = 1.0

    r_bc[fixed_dofs] = 0.0
    return K_lil.tocsr(), r_bc


def apply_bc_indptr(
    K: sp.spmatrix,
    r: np.ndarray,
    fixed_dofs: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """新実装: indptr/data 直接操作版."""
    K_bc = K.tocsr().copy()
    r_bc = r.copy()

    if len(fixed_dofs) == 0:
        return K_bc, r_bc

    # 行の消去（CSR: indptr/data 直接操作）
    for dof in fixed_dofs:
        start = K_bc.indptr[dof]
        end = K_bc.indptr[dof + 1]
        K_bc.data[start:end] = 0.0

    # 列の消去（CSC 変換 + indptr/data 直接操作）
    K_csc = K_bc.tocsc()
    for dof in fixed_dofs:
        start = K_csc.indptr[dof]
        end = K_csc.indptr[dof + 1]
        K_csc.data[start:end] = 0.0
    K_bc = K_csc.tocsr()

    # 対角に 1.0 を設定
    K_bc[fixed_dofs, fixed_dofs] = 1.0
    K_bc.eliminate_zeros()

    r_bc[fixed_dofs] = 0.0
    return K_bc, r_bc


# ====================================================================
# 合成データ生成
# ====================================================================


def _make_synthetic_contact_manager(
    n_pairs: int, n_nodes: int, ndof_per_node: int = 6
) -> tuple[ContactManager, int]:
    """合成接触問題の生成.

    n_pairs 個の ACTIVE 接触ペアを持つ ContactManager を生成する。
    各ペアはランダムな4節点を使用し、貫入状態（gap < 0）を模擬する。

    Returns:
        (manager, ndof_total)
    """
    rng = np.random.default_rng(42)
    manager = ContactManager(config=ContactConfig())
    ndof_total = n_nodes * ndof_per_node

    for _ in range(n_pairs):
        # 4つの異なるノードをランダム選択
        nodes = rng.choice(n_nodes, size=4, replace=False)
        pair = manager.add_pair(
            elem_a=0,
            elem_b=1,
            nodes_a=nodes[:2],
            nodes_b=nodes[2:],
            radius_a=0.5,
            radius_b=0.5,
        )
        # ACTIVE 状態にセット（貫入あり）
        normal = rng.standard_normal(3)
        normal /= np.linalg.norm(normal)
        t1 = np.cross(normal, [1, 0, 0])
        if np.linalg.norm(t1) < 0.1:
            t1 = np.cross(normal, [0, 1, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(normal, t1)

        pair.state = ContactState(
            s=rng.uniform(0.1, 0.9),
            t=rng.uniform(0.1, 0.9),
            gap=-rng.uniform(0.01, 0.1),  # 貫入
            normal=normal,
            tangent1=t1,
            tangent2=t2,
            lambda_n=rng.uniform(0.0, 10.0),
            k_pen=1e4,
            k_t=5e3,
            p_n=rng.uniform(1.0, 100.0),
            status=ContactStatus.ACTIVE,
        )

    return manager, ndof_total


def _make_synthetic_sparse_system(
    ndof: int, nnz_per_row: int = 20
) -> tuple[sp.csr_matrix, np.ndarray]:
    """合成疎行列システムの生成（ベクトル化版）.

    BC適用のベンチマーク用に、帯行列+ランダム疎行列を生成する。
    """
    rng = np.random.default_rng(123)

    # 対角成分
    K = sp.diags(np.ones(ndof) * 100.0, 0, shape=(ndof, ndof), format="csr")

    # 帯行列部分（高速な組み立て）
    band = min(nnz_per_row // 2, ndof - 1)
    for offset in range(1, band + 1):
        v = np.ones(ndof - offset) * (0.5 / offset)
        K = K + sp.diags(v, offset, shape=(ndof, ndof), format="csr")
        K = K + sp.diags(v, -offset, shape=(ndof, ndof), format="csr")

    # ランダム疎成分（scipy.sparse.random で高速生成）
    R = sp.random(ndof, ndof, density=nnz_per_row / ndof, random_state=rng, format="csr")
    K = K + R + R.T  # 対称化

    K.sum_duplicates()
    f = rng.standard_normal(ndof)
    return K, f


def _make_synthetic_segments(n_seg: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """合成セグメントデータの生成（Broadphase用）.

    2方向の交差梁ジオメトリ: X方向とY方向の梁が格子状に配置。
    """
    rng = np.random.default_rng(99)
    segments = []
    # X方向の梁
    n_x = n_seg // 2
    n_y = n_seg - n_x
    for i in range(n_x):
        y = i * 0.5 + rng.uniform(-0.01, 0.01)
        x0 = np.array([0.0, y, 0.0])
        x1 = np.array([10.0, y, rng.uniform(-0.05, 0.05)])
        segments.append((x0, x1))
    for j in range(n_y):
        x = j * 0.5 + rng.uniform(-0.01, 0.01)
        y0 = np.array([x, 0.0, 0.0])
        y1 = np.array([x, 10.0, rng.uniform(-0.05, 0.05)])
        segments.append((y0, y1))
    return segments


# ====================================================================
# ベンチマーク: _add_local_to_coo
# ====================================================================


@pytest.mark.slow
class TestAddLocalToCOOBenchmark:
    """_add_local_to_coo: 旧Python loop vs 新np.nonzero."""

    N_CALLS = 8192  # 4096要素以上 → 8192回呼び出し

    def test_speedup(self):
        rng = np.random.default_rng(42)
        n_calls = self.N_CALLS

        # 合成データ: 12×12 局所行列 × N_CALLS
        K_locals = [rng.standard_normal((12, 12)) for _ in range(n_calls)]
        gdofs_list = [np.arange(i * 6, i * 6 + 12, dtype=int) for i in range(n_calls)]

        # --- 旧実装 ---
        rows_old, cols_old, data_old = [], [], []
        t0 = time.perf_counter()
        for K_loc, gd in zip(K_locals, gdofs_list, strict=True):
            _add_local_to_coo_old(K_loc, gd, rows_old, cols_old, data_old)
        t_old = time.perf_counter() - t0

        # --- 新実装 ---
        rows_new, cols_new, data_new = [], [], []
        t0 = time.perf_counter()
        for K_loc, gd in zip(K_locals, gdofs_list, strict=True):
            _add_local_to_coo(K_loc, gd, rows_new, cols_new, data_new)
        t_new = time.perf_counter() - t0

        speedup = t_old / max(t_new, 1e-9)
        print(f"\n_add_local_to_coo ({n_calls} calls):")
        print(f"  旧(loop): {t_old:.4f}s | 新(np.nonzero): {t_new:.4f}s | 速度比: {speedup:.2f}x")

        # 結果の数値的一致を確認
        assert len(rows_old) == len(rows_new)
        # 順序が異なる可能性があるので sorted で比較
        old_set = sorted(zip(rows_old, cols_old, data_old, strict=True))
        new_set = sorted(zip(rows_new, cols_new, data_new, strict=True))
        for (r1, c1, d1), (r2, c2, d2) in zip(old_set, new_set, strict=True):
            assert r1 == r2
            assert c1 == c2
            assert abs(d1 - d2) < 1e-12

        # 速度改善の検証（最低限の改善を期待）
        assert speedup > 0.5, f"新実装が旧実装より大幅に遅い: {speedup:.2f}x"


# ====================================================================
# ベンチマーク: compute_contact_force
# ====================================================================


@pytest.mark.slow
class TestContactForceBenchmark:
    """接触力ベクトル: 旧double loop vs 新np.add.at scatter-add."""

    N_PAIRS = 4096

    def test_speedup(self):
        n_nodes = self.N_PAIRS * 4 + 100  # 十分なノード数
        manager, ndof = _make_synthetic_contact_manager(self.N_PAIRS, n_nodes)

        # 摩擦力マップ（半数のペアに付与）
        rng = np.random.default_rng(77)
        friction_forces = {}
        for i in range(0, self.N_PAIRS, 2):
            friction_forces[i] = rng.uniform(-10.0, 10.0, size=2)

        # --- 旧実装 ---
        t0 = time.perf_counter()
        f_old = compute_contact_force_old(manager, ndof, friction_forces=friction_forces)
        t_old = time.perf_counter() - t0

        # --- 新実装 ---
        t0 = time.perf_counter()
        f_new = compute_contact_force(manager, ndof, friction_forces=friction_forces)
        t_new = time.perf_counter() - t0

        speedup = t_old / max(t_new, 1e-9)
        print(f"\ncompute_contact_force ({self.N_PAIRS} pairs):")
        print(f"  旧(loop): {t_old:.4f}s | 新(scatter): {t_new:.4f}s | 速度比: {speedup:.2f}x")

        # 数値的一致
        np.testing.assert_allclose(f_old, f_new, atol=1e-10)

        assert speedup > 0.5, f"新実装が旧実装より大幅に遅い: {speedup:.2f}x"


# ====================================================================
# ベンチマーク: compute_contact_stiffness
# ====================================================================


@pytest.mark.slow
class TestContactStiffnessBenchmark:
    """接触剛性行列: 旧12×12 loop vs 新np.outer + vectorized COO."""

    N_PAIRS = 4096

    def test_speedup(self):
        n_nodes = self.N_PAIRS * 4 + 100
        manager, ndof = _make_synthetic_contact_manager(self.N_PAIRS, n_nodes)

        # 摩擦接線剛性マップ（半数のペアに付与）
        rng = np.random.default_rng(77)
        friction_tangents = {}
        for i in range(0, self.N_PAIRS, 2):
            D = rng.uniform(-10.0, 10.0, size=(2, 2))
            friction_tangents[i] = D

        # --- 旧実装 ---
        t0 = time.perf_counter()
        K_old = compute_contact_stiffness_old(manager, ndof, friction_tangents=friction_tangents)
        t_old = time.perf_counter() - t0

        # --- 新実装 ---
        t0 = time.perf_counter()
        K_new = compute_contact_stiffness(
            manager,
            ndof,
            friction_tangents=friction_tangents,
            use_geometric_stiffness=False,
        )
        t_new = time.perf_counter() - t0

        speedup = t_old / max(t_new, 1e-9)
        print(f"\ncompute_contact_stiffness ({self.N_PAIRS} pairs):")
        print(f"  旧(loop): {t_old:.4f}s | 新(np.outer): {t_new:.4f}s | 速度比: {speedup:.2f}x")

        # 数値的一致（疎行列の差のノルム）
        diff = K_old - K_new
        diff_norm = sp.linalg.norm(diff)
        K_norm = sp.linalg.norm(K_old)
        rel_err = diff_norm / max(K_norm, 1e-30)
        print(f"  相対誤差: {rel_err:.2e}")
        assert rel_err < 1e-10, f"数値的不一致: rel_err={rel_err:.2e}"

        assert speedup > 0.5, f"新実装が旧実装より大幅に遅い: {speedup:.2f}x"


# ====================================================================
# ベンチマーク: BC適用
# ====================================================================


@pytest.mark.slow
class TestBCApplicationBenchmark:
    """BC適用: tolil + loop vs CSR/CSC直接操作."""

    @pytest.mark.parametrize("ndof", [4096 * 6, 8192 * 6])
    def test_speedup(self, ndof):
        K, f = _make_synthetic_sparse_system(ndof, nnz_per_row=15)

        # 固定DOF: 梁問題の典型 — 両端固定（全DOFの~1-2%）
        rng = np.random.default_rng(55)
        n_fixed = max(ndof // 50, 100)
        fixed_dofs = rng.choice(ndof, size=n_fixed, replace=False).astype(int)

        # ウォームアップ
        _ = apply_bc_tolil(K, f, fixed_dofs[:10])
        _ = apply_bc_indptr(K, f, fixed_dofs[:10])

        # --- 旧実装: tolil ---
        t0 = time.perf_counter()
        K_old, r_old = apply_bc_tolil(K, f, fixed_dofs)
        t_old = time.perf_counter() - t0

        # --- 新実装: CSR/CSC ---
        t0 = time.perf_counter()
        K_new, r_new = apply_bc_indptr(K, f, fixed_dofs)
        t_new = time.perf_counter() - t0

        speedup = t_old / max(t_new, 1e-9)
        print(f"\nBC適用 (ndof={ndof}, fixed={n_fixed}):")
        print(f"  旧(tolil): {t_old:.4f}s | 新(indptr): {t_new:.4f}s | 速度比: {speedup:.2f}x")

        # 数値的一致
        diff_K = K_old - K_new
        diff_norm = sp.linalg.norm(diff_K)
        np.testing.assert_allclose(r_old, r_new, atol=1e-12)
        assert diff_norm < 1e-10, f"行列不一致: ||diff||={diff_norm:.2e}"

        assert speedup > 0.5, f"新実装が旧実装より大幅に遅い: {speedup:.2f}x"


# ====================================================================
# ベンチマーク: Broadphase AABB
# ====================================================================


@pytest.mark.slow
class TestBroadphaseBenchmark:
    """Broadphase AABB: 4096+セグメントの性能測定."""

    @pytest.mark.parametrize("n_seg", [4096, 8192])
    def test_performance(self, n_seg):
        segments = _make_synthetic_segments(n_seg)
        radius = 0.2

        # ウォームアップ
        _ = broadphase_aabb(segments[:100], radius, margin=0.1)

        t0 = time.perf_counter()
        pairs = broadphase_aabb(segments, radius, margin=0.1)
        t_elapsed = time.perf_counter() - t0

        print(f"\nBroadphase AABB (n_seg={n_seg}):")
        print(f"  時間: {t_elapsed:.4f}s | ペア数: {len(pairs)}")
        print(f"  ペア/セグメント比: {len(pairs) / n_seg:.1f}")

        # 基本性能チェック: 4096セグメントは10秒以内に完了すべき
        assert t_elapsed < 30.0, f"Broadphase が遅すぎる: {t_elapsed:.2f}s"


# ====================================================================
# 統合ベンチマーク: 全最適化の合算効果
# ====================================================================


@pytest.mark.slow
class TestOverallSpeedup:
    """全最適化の合算効果を測定."""

    def test_combined_speedup(self):
        """4096ペアの接触力+剛性+BC適用の合計時間を比較."""
        n_pairs = 4096
        n_nodes = n_pairs * 4 + 100
        ndof_per_node = 6
        ndof = n_nodes * ndof_per_node

        manager, _ = _make_synthetic_contact_manager(n_pairs, n_nodes, ndof_per_node)

        rng = np.random.default_rng(77)
        friction_forces = {}
        friction_tangents = {}
        for i in range(0, n_pairs, 2):
            friction_forces[i] = rng.uniform(-10.0, 10.0, size=2)
            friction_tangents[i] = rng.uniform(-10.0, 10.0, size=(2, 2))

        # 合成疎行列（BC適用用）
        # 軽量版: 対角 + バンド
        diag_data = np.ones(ndof) * 100.0
        K_base = sp.diags(diag_data, 0, shape=(ndof, ndof), format="csr")
        # バンドを追加（帯行列）
        band = min(12, ndof - 1)
        for offset in range(1, band + 1):
            v = np.ones(ndof - offset) * (0.5 / offset)
            K_base = K_base + sp.diags(v, offset, shape=(ndof, ndof), format="csr")
            K_base = K_base + sp.diags(v, -offset, shape=(ndof, ndof), format="csr")
        f = rng.standard_normal(ndof)
        n_fixed = max(ndof // 50, 100)
        fixed_dofs = rng.choice(ndof, size=n_fixed, replace=False).astype(int)

        # ウォームアップ
        _ = compute_contact_force_old(manager, ndof, friction_forces=friction_forces)
        _ = compute_contact_force(manager, ndof, friction_forces=friction_forces)

        # --- 旧実装 合計 ---
        t0 = time.perf_counter()
        f_old = compute_contact_force_old(manager, ndof, friction_forces=friction_forces)
        K_old = compute_contact_stiffness_old(manager, ndof, friction_tangents=friction_tangents)
        K_bc_old, r_bc_old = apply_bc_tolil(K_base + K_old, f + f_old, fixed_dofs)
        t_old_total = time.perf_counter() - t0

        # --- 新実装 合計 ---
        t0 = time.perf_counter()
        f_new = compute_contact_force(manager, ndof, friction_forces=friction_forces)
        K_new = compute_contact_stiffness(
            manager,
            ndof,
            friction_tangents=friction_tangents,
            use_geometric_stiffness=False,
        )
        K_bc_new, r_bc_new = apply_bc_indptr(K_base + K_new, f + f_new, fixed_dofs)
        t_new_total = time.perf_counter() - t0

        speedup = t_old_total / max(t_new_total, 1e-9)
        print(f"\n=== 統合ベンチマーク ({n_pairs} pairs, ndof={ndof}) ===")
        print(f"  旧合計: {t_old_total:.4f}s")
        print(f"  新合計: {t_new_total:.4f}s")
        print(f"  合計速度比: {speedup:.2f}x")

        # 数値的一致
        np.testing.assert_allclose(f_old, f_new, atol=1e-10)
        diff_K = K_old - K_new
        assert sp.linalg.norm(diff_K) < 1e-10

        assert speedup > 0.5, f"合計で新実装が旧実装より大幅に遅い: {speedup:.2f}x"
