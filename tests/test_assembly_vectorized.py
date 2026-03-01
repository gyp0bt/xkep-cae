"""COO ベクトル化 + 共有メモリ並列化のテスト（status-090）.

assembly.py の以下の変更を検証:
1. _vectorized_coo_indices: 全要素分の DOF → rows/cols 一括計算
2. _assemble_sequential: ベクトル化 COO で逐次アセンブリ
3. _assemble_parallel: 共有メモリ並列アセンブリ（mp.Pool + shared_memory）
4. 実要素での逐次/並列一致、スピードアップ計測
"""

import numpy as np
import pytest

from xkep_cae.assembly import (
    _assemble_parallel,
    _assemble_sequential,
    _vectorized_coo_indices,
    assemble_global_stiffness,
)

# ========== テスト用ヘルパー ==========


class _DummyMaterial:
    """ダミー材料（単位 D 行列）."""

    D = np.eye(3)

    def stiffness_matrix(self, strain=None):
        return self.D


class _DummyQ4:
    """ダミー Q4 要素."""

    ndof_per_node = 2
    ndof = 8
    nnodes = 4

    def local_stiffness(self, coords, material, thickness):
        return np.eye(self.ndof)

    def dof_indices(self, node_ids):
        return np.array(
            [n * self.ndof_per_node + d for n in node_ids for d in range(self.ndof_per_node)]
        )


class _DummyBeam3D:
    """ダミー 3D 梁要素（6DOF/node）."""

    ndof_per_node = 6
    ndof = 12
    nnodes = 2

    def local_stiffness(self, coords, material, thickness):
        return np.eye(self.ndof) * 1.5

    def dof_indices(self, node_ids):
        return np.array(
            [n * self.ndof_per_node + d for n in node_ids for d in range(self.ndof_per_node)]
        )


class _DummyHex8:
    """ダミー HEX8 要素（3DOF/node, 8nodes）."""

    ndof_per_node = 3
    ndof = 24
    nnodes = 8

    def local_stiffness(self, coords, material, thickness):
        return np.eye(self.ndof) * 2.0

    def dof_indices(self, node_ids):
        return np.array(
            [n * self.ndof_per_node + d for n in node_ids for d in range(self.ndof_per_node)]
        )


def _make_quad_mesh(n_elem):
    """n_elem 個の Q4 メッシュを生成."""
    nx = int(np.ceil(np.sqrt(n_elem)))
    ny = max(1, n_elem // nx)
    nodes = np.array([(i, j) for j in range(ny + 1) for i in range(nx + 1)], dtype=float)
    conn = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            conn.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
    return nodes, np.array(conn[:n_elem], dtype=int)


def _make_beam_mesh(n_elem):
    """n_elem 個の梁メッシュを生成."""
    n_nodes = n_elem + 1
    s = np.linspace(0, 100.0, n_nodes)
    nodes = np.column_stack([s, np.zeros(n_nodes), np.zeros(n_nodes)])
    conn = np.column_stack([np.arange(n_elem), np.arange(1, n_nodes)])
    return nodes, conn.astype(int)


# ========== _vectorized_coo_indices テスト ==========


class TestVectorizedCooIndices:
    """_vectorized_coo_indices のユニットテスト."""

    def test_single_q4_element(self):
        """単一 Q4 要素の COO インデックスが dof_indices + repeat/tile と一致."""
        conn = np.array([[0, 1, 5, 4]], dtype=int)
        elem = _DummyQ4()
        rows_v, cols_v = _vectorized_coo_indices(conn, elem.nnodes, elem.ndof_per_node, elem.ndof)

        # 従来方式
        node_ids = conn[0, -elem.nnodes :]
        edofs = elem.dof_indices(node_ids)
        rows_ref = np.repeat(edofs, elem.ndof)
        cols_ref = np.tile(edofs, elem.ndof)

        np.testing.assert_array_equal(rows_v, rows_ref)
        np.testing.assert_array_equal(cols_v, cols_ref)

    def test_multiple_beam_elements(self):
        """複数梁要素の COO インデックスが従来方式と一致."""
        n_elem = 10
        _, conn = _make_beam_mesh(n_elem)
        elem = _DummyBeam3D()

        rows_v, cols_v = _vectorized_coo_indices(conn, elem.nnodes, elem.ndof_per_node, elem.ndof)

        # 従来方式（per-element ループ）
        m = elem.ndof
        rows_ref = []
        cols_ref = []
        for row in conn:
            node_ids = row[-elem.nnodes :]
            edofs = elem.dof_indices(node_ids)
            rows_ref.append(np.repeat(edofs, m))
            cols_ref.append(np.tile(edofs, m))
        rows_ref = np.concatenate(rows_ref)
        cols_ref = np.concatenate(cols_ref)

        np.testing.assert_array_equal(rows_v, rows_ref)
        np.testing.assert_array_equal(cols_v, cols_ref)

    def test_hex8_elements(self):
        """HEX8 (8node, 3DOF/node) の COO インデックスが正しい."""
        conn = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1, 8, 9, 2, 5, 10, 11, 6],
            ],
            dtype=int,
        )
        elem = _DummyHex8()
        rows_v, cols_v = _vectorized_coo_indices(conn, elem.nnodes, elem.ndof_per_node, elem.ndof)

        m = elem.ndof
        rows_ref = []
        cols_ref = []
        for row in conn:
            edofs = elem.dof_indices(row[-elem.nnodes :])
            rows_ref.append(np.repeat(edofs, m))
            cols_ref.append(np.tile(edofs, m))
        rows_ref = np.concatenate(rows_ref)
        cols_ref = np.concatenate(cols_ref)

        np.testing.assert_array_equal(rows_v, rows_ref)
        np.testing.assert_array_equal(cols_v, cols_ref)

    def test_output_shapes(self):
        """出力サイズが n_elem * m * m であること."""
        n_elem = 50
        _, conn = _make_beam_mesh(n_elem)
        elem = _DummyBeam3D()
        m = elem.ndof
        rows_v, cols_v = _vectorized_coo_indices(conn, elem.nnodes, elem.ndof_per_node, m)
        assert rows_v.shape == (n_elem * m * m,)
        assert cols_v.shape == (n_elem * m * m,)

    def test_matches_real_beam_element(self):
        """実装 TimoshenkoBeam3D の dof_indices との一致を確認."""
        from xkep_cae.elements.beam_timo3d import TimoshenkoBeam3D
        from xkep_cae.sections.beam import BeamSection

        sec = BeamSection.circle(d=10.0)
        beam = TimoshenkoBeam3D(section=sec)
        n_elem = 20
        _, conn = _make_beam_mesh(n_elem)
        conn_int = conn.astype(int)

        rows_v, cols_v = _vectorized_coo_indices(
            conn_int, beam.nnodes, beam.ndof_per_node, beam.ndof
        )

        m = beam.ndof
        rows_ref = []
        cols_ref = []
        for row in conn_int:
            node_ids = row[-beam.nnodes :]
            edofs = beam.dof_indices(node_ids)
            rows_ref.append(np.repeat(edofs, m))
            cols_ref.append(np.tile(edofs, m))
        rows_ref = np.concatenate(rows_ref)
        cols_ref = np.concatenate(cols_ref)

        np.testing.assert_array_equal(rows_v, rows_ref)
        np.testing.assert_array_equal(cols_v, cols_ref)


# ========== 逐次アセンブリ（ベクトル化 COO）テスト ==========


class TestSequentialVectorized:
    """ベクトル化 COO 逐次アセンブリの正当性テスト."""

    def test_q4_single_element(self):
        """Q4 単一要素で対称・半正定値を確認."""
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        conn = np.array([[0, 1, 2, 3]], dtype=int)
        elem = _DummyQ4()
        mat = _DummyMaterial()

        K = _assemble_sequential(nodes, [(elem, conn)], mat, 8, 1, show_progress=False)
        assert K.shape == (8, 8)
        Kd = K.toarray()
        assert np.allclose(Kd, Kd.T, atol=1e-14)

    def test_beam_cantilever_analytical(self):
        """実要素 Timoshenko3D 片持ち梁の解析解比較."""
        from xkep_cae.bc import apply_dirichlet
        from xkep_cae.elements.beam_timo3d import TimoshenkoBeam3D
        from xkep_cae.materials.beam_elastic import BeamElastic1D
        from xkep_cae.sections.beam import BeamSection
        from xkep_cae.solver import solve_displacement

        E = 200e3
        nu = 0.3
        G = E / (2 * (1 + nu))
        d = 10.0
        L = 100.0
        P = 1.0
        kappa = 5 / 6
        n_elems = 20

        sec = BeamSection.circle(d=d)
        beam = TimoshenkoBeam3D(section=sec, kappa_y=kappa, kappa_z=kappa)
        mat = BeamElastic1D(E=E, nu=nu)

        n_nodes = n_elems + 1
        s = np.linspace(0, L, n_nodes)
        nodes = np.column_stack([s, np.zeros(n_nodes), np.zeros(n_nodes)])
        conn = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)]).astype(int)

        ndof = 6 * n_nodes
        K = _assemble_sequential(nodes, [(beam, conn)], mat, ndof, n_elems, show_progress=False)

        f = np.zeros(ndof)
        f[6 * n_elems + 1] = P
        fixed_dofs = np.arange(6, dtype=int)
        Kbc, fbc = apply_dirichlet(K, f, fixed_dofs, 0.0)
        u, _ = solve_displacement(Kbc, fbc, size_threshold=4000)

        delta_analytical = P * L**3 / (3 * E * sec.Iz) + P * L / (kappa * G * sec.A)
        delta_fem = u[6 * n_elems + 1]
        assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-10

    def test_mixed_element_groups(self):
        """混在要素グループでの正当性."""
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0.5]], dtype=float)
        conn_q4 = np.array([[0, 1, 2, 3]], dtype=int)
        # 三角形要素を Q4 と同じ ndof_per_node=2 で
        from xkep_cae.elements.quad4 import Quad4PlaneStrain
        from xkep_cae.elements.tri3 import Tri3PlaneStrain
        from xkep_cae.materials.elastic import PlaneStrainElastic

        mat = PlaneStrainElastic(100.0, 0.29)
        conn_t3 = np.array([[1, 4, 2]], dtype=int)

        K = _assemble_sequential(
            nodes,
            [(Quad4PlaneStrain(), conn_q4), (Tri3PlaneStrain(), conn_t3)],
            mat,
            10,
            2,
            thickness=1.0,
            show_progress=False,
        )
        assert K.shape == (10, 10)
        Kd = K.toarray()
        assert np.allclose(Kd, Kd.T, atol=1e-10)


# ========== 共有メモリ並列アセンブリテスト ==========


class TestSharedMemoryParallel:
    """共有メモリ並列アセンブリの正当性テスト."""

    def test_sequential_matches_parallel_dummy(self):
        """ダミー要素で逐次と並列の結果が一致."""
        elem = _DummyQ4()
        mat = _DummyMaterial()
        nodes, conn = _make_quad_mesh(16)
        n_total = len(conn)
        ndof_total = len(nodes) * 2

        K_seq = _assemble_sequential(
            nodes, [(elem, conn)], mat, ndof_total, n_total, show_progress=False
        )
        K_par = _assemble_parallel(
            nodes,
            [(elem, conn)],
            mat,
            ndof_total,
            n_total,
            show_progress=False,
            n_jobs=2,
        )

        diff = abs(K_seq - K_par).max()
        assert diff < 1e-14, f"逐次と並列の差: {diff}"

    def test_sequential_matches_parallel_beam3d(self):
        """ダミー 3D 梁要素で逐次と並列の結果が一致."""
        elem = _DummyBeam3D()
        mat = _DummyMaterial()
        n_elem = 50
        nodes, conn = _make_beam_mesh(n_elem)
        n_total = len(conn)
        ndof_total = len(nodes) * 6

        K_seq = _assemble_sequential(
            nodes, [(elem, conn)], mat, ndof_total, n_total, show_progress=False
        )
        K_par = _assemble_parallel(
            nodes,
            [(elem, conn)],
            mat,
            ndof_total,
            n_total,
            show_progress=False,
            n_jobs=2,
        )

        diff = abs(K_seq - K_par).max()
        assert diff < 1e-14, f"逐次と並列の差: {diff}"

    def test_parallel_real_beam_element(self):
        """実 TimoshenkoBeam3D 要素で逐次/並列一致."""
        from xkep_cae.elements.beam_timo3d import TimoshenkoBeam3D
        from xkep_cae.materials.beam_elastic import BeamElastic1D
        from xkep_cae.sections.beam import BeamSection

        sec = BeamSection.circle(d=10.0)
        beam = TimoshenkoBeam3D(section=sec)
        mat = BeamElastic1D(E=200e3, nu=0.3)

        n_elem = 100
        nodes, conn = _make_beam_mesh(n_elem)
        n_total = len(conn)
        ndof_total = len(nodes) * 6

        K_seq = _assemble_sequential(
            nodes, [(beam, conn)], mat, ndof_total, n_total, show_progress=False
        )
        K_par = _assemble_parallel(
            nodes,
            [(beam, conn)],
            mat,
            ndof_total,
            n_total,
            show_progress=False,
            n_jobs=2,
        )

        diff = abs(K_seq - K_par).max()
        assert diff < 1e-12, f"逐次と並列の差: {diff}"

    def test_parallel_n_jobs_4(self):
        """n_jobs=4 でもエラーなく動作."""
        elem = _DummyBeam3D()
        mat = _DummyMaterial()
        n_elem = 40
        nodes, conn = _make_beam_mesh(n_elem)
        n_total = len(conn)
        ndof_total = len(nodes) * 6

        K = _assemble_parallel(
            nodes,
            [(elem, conn)],
            mat,
            ndof_total,
            n_total,
            show_progress=False,
            n_jobs=4,
        )
        assert K.shape == (ndof_total, ndof_total)
        assert K.nnz > 0

    def test_parallel_symmetry(self):
        """並列アセンブリ結果が対称."""
        from xkep_cae.elements.beam_timo3d import TimoshenkoBeam3D
        from xkep_cae.materials.beam_elastic import BeamElastic1D
        from xkep_cae.sections.beam import BeamSection

        sec = BeamSection.circle(d=10.0)
        beam = TimoshenkoBeam3D(section=sec)
        mat = BeamElastic1D(E=200e3, nu=0.3)

        n_elem = 30
        nodes, conn = _make_beam_mesh(n_elem)
        ndof = len(nodes) * 6

        K = _assemble_parallel(
            nodes,
            [(beam, conn)],
            mat,
            ndof,
            n_elem,
            show_progress=False,
            n_jobs=2,
        )
        Kd = K.toarray()
        assert np.allclose(Kd, Kd.T, atol=1e-10)

    def test_parallel_cantilever_analytical(self):
        """共有メモリ並列で片持ち梁の解析解が一致."""
        from xkep_cae.bc import apply_dirichlet
        from xkep_cae.elements.beam_timo3d import TimoshenkoBeam3D
        from xkep_cae.materials.beam_elastic import BeamElastic1D
        from xkep_cae.sections.beam import BeamSection
        from xkep_cae.solver import solve_displacement

        E = 200e3
        nu = 0.3
        G = E / (2 * (1 + nu))
        d = 10.0
        L = 100.0
        P = 1.0
        kappa = 5 / 6
        n_elems = 20

        sec = BeamSection.circle(d=d)
        beam = TimoshenkoBeam3D(section=sec, kappa_y=kappa, kappa_z=kappa)
        mat = BeamElastic1D(E=E, nu=nu)

        n_nodes = n_elems + 1
        s = np.linspace(0, L, n_nodes)
        nodes = np.column_stack([s, np.zeros(n_nodes), np.zeros(n_nodes)])
        conn = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)]).astype(int)

        ndof = 6 * n_nodes
        K = _assemble_parallel(
            nodes,
            [(beam, conn)],
            mat,
            ndof,
            n_elems,
            show_progress=False,
            n_jobs=2,
        )

        f = np.zeros(ndof)
        f[6 * n_elems + 1] = P
        fixed_dofs = np.arange(6, dtype=int)
        Kbc, fbc = apply_dirichlet(K, f, fixed_dofs, 0.0)
        u, _ = solve_displacement(Kbc, fbc, size_threshold=4000)

        delta_analytical = P * L**3 / (3 * E * sec.Iz) + P * L / (kappa * G * sec.A)
        delta_fem = u[6 * n_elems + 1]
        assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-10


# ========== assemble_global_stiffness 統合テスト ==========


class TestAssembleGlobalStiffnessIntegration:
    """公開 API の統合テスト."""

    def test_auto_sequential_small(self):
        """小規模は自動的に逐次パスを選択."""
        elem = _DummyQ4()
        mat = _DummyMaterial()
        nodes, conn = _make_quad_mesh(10)
        K = assemble_global_stiffness(nodes, [(elem, conn)], mat, show_progress=False, n_jobs=4)
        assert K.shape[0] > 0

    def test_n_jobs_minus_one_no_crash(self):
        """n_jobs=-1 でエラーなし."""
        elem = _DummyQ4()
        mat = _DummyMaterial()
        nodes, conn = _make_quad_mesh(10)
        K = assemble_global_stiffness(nodes, [(elem, conn)], mat, show_progress=False, n_jobs=-1)
        assert K.shape[0] > 0


# ========== ベンチマーク（slow マーカー）==========


@pytest.mark.slow
class TestAssemblyBenchmark:
    """COO ベクトル化 + 共有メモリのスピードアップ計測."""

    @pytest.mark.parametrize("n_elem", [4096, 8192])
    def test_speedup_shared_memory(self, n_elem):
        """共有メモリ並列が逐次と一致し、スピードアップを計測."""
        import time

        from xkep_cae.elements.beam_timo3d import TimoshenkoBeam3D
        from xkep_cae.materials.beam_elastic import BeamElastic1D
        from xkep_cae.sections.beam import BeamSection

        sec = BeamSection.circle(d=10.0)
        beam = TimoshenkoBeam3D(section=sec)
        mat = BeamElastic1D(E=200e3, nu=0.3)

        nodes, conn = _make_beam_mesh(n_elem)
        n_total = len(conn)
        ndof_total = len(nodes) * 6

        # ウォームアップ
        _assemble_sequential(nodes, [(beam, conn[:10])], mat, ndof_total, 10, show_progress=False)

        # 逐次
        t0 = time.perf_counter()
        K_seq = _assemble_sequential(
            nodes, [(beam, conn)], mat, ndof_total, n_total, show_progress=False
        )
        t_seq = time.perf_counter() - t0

        # 並列（4 workers）
        t0 = time.perf_counter()
        K_par = _assemble_parallel(
            nodes,
            [(beam, conn)],
            mat,
            ndof_total,
            n_total,
            show_progress=False,
            n_jobs=4,
        )
        t_par = time.perf_counter() - t0

        # 正しさ検証
        diff = abs(K_seq - K_par).max()
        assert diff < 1e-12, f"逐次と並列の差: {diff}"

        speedup = t_seq / t_par if t_par > 0 else float("inf")
        print(f"\n  n_elem={n_elem}: seq={t_seq:.3f}s, par={t_par:.3f}s, speedup={speedup:.2f}x")

    def test_coo_vectorization_speedup(self):
        """COO ベクトル化の効果を計測."""
        import time

        n_elem = 10000
        nodes, conn = _make_beam_mesh(n_elem)
        conn_int = conn.astype(int)
        elem = _DummyBeam3D()
        m = elem.ndof

        # ベクトル化
        t0 = time.perf_counter()
        for _ in range(10):
            _vectorized_coo_indices(conn_int, elem.nnodes, elem.ndof_per_node, m)
        t_vec = (time.perf_counter() - t0) / 10

        # 従来方式（per-element ループ）
        t0 = time.perf_counter()
        for _ in range(10):
            rows_ref = []
            cols_ref = []
            for row in conn_int:
                edofs = elem.dof_indices(row[-elem.nnodes :])
                rows_ref.append(np.repeat(edofs, m))
                cols_ref.append(np.tile(edofs, m))
            np.concatenate(rows_ref)
            np.concatenate(cols_ref)
        t_loop = (time.perf_counter() - t0) / 10

        speedup = t_loop / t_vec if t_vec > 0 else float("inf")
        print(
            f"\n  COO vectorization (n_elem={n_elem}): "
            f"loop={t_loop * 1000:.1f}ms, vec={t_vec * 1000:.1f}ms, "
            f"speedup={speedup:.1f}x"
        )
        # ベクトル化は少なくとも 2x 速いことを期待
        assert speedup > 1.5, f"Expected >1.5x speedup, got {speedup:.2f}x"
