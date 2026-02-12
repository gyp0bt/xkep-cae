from __future__ import annotations
import numpy as np
import scipy.sparse as sp

from pycae.api import assemble_K_from_arrays_mixed
from pycae.bc import apply_dirichlet
from pycae.solver import solve_displacement


def _manufactured_solve_check(
    K: sp.csr_matrix, fixed_dofs: np.ndarray, seed: int = 0
) -> None:
    """製造解 u_true を用いたソルバ検証（固定DOFを0にして f=K u_true を作る）"""
    n = K.shape[0]
    rng = np.random.default_rng(seed)
    u_true = rng.standard_normal(n)
    u_true[fixed_dofs] = 0.0
    f = K @ u_true

    Kbc, fbc = apply_dirichlet(K, f, fixed_dofs, 0.0)
    u, info = solve_displacement(Kbc, fbc, size_threshold=4000)

    free = np.setdiff1d(np.arange(n), fixed_dofs)
    assert np.allclose(u[free], u_true[free], rtol=1e-10, atol=1e-10)

    # SPDっぽさの確認
    Kd = Kbc.toarray()
    assert np.allclose(Kd, Kd.T, atol=1e-12)
    w = np.linalg.eigvalsh(Kd)
    assert np.min(w) > -1e-10


def test_pure_quad4_unit_square():
    """Q4: 単位正方形1要素 [0,1,2,3] でソルバ検証"""
    nodes = np.array(
        [
            [0, 0.0, 0.0],
            [1, 1.0, 0.0],
            [2, 1.0, 1.0],
            [3, 0.0, 1.0],
        ],
        dtype=float,
    )
    quads = np.array([[0, 1, 2, 3]], dtype=int)
    tris = None

    K = assemble_K_from_arrays_mixed(quads, tris, nodes, E=10.0, nu=0.25, thickness=1.0)

    # 節点0(u,v) と 節点3(u,v) を拘束 -> DOF = [0,1, 6,7]
    fixed = np.array([0, 1, 6, 7], dtype=int)

    _manufactured_solve_check(K, fixed, seed=123)


def test_pure_tri3_unit_triangle():
    """TRI3: 単三角形 [0,1,2] でソルバ検証"""
    nodes = np.array(
        [
            [0, 0.0, 0.0],
            [1, 1.0, 0.0],
            [2, 0.0, 1.0],
        ],
        dtype=float,
    )
    quads = None
    tris = np.array([[0, 1, 2]], dtype=int)

    K = assemble_K_from_arrays_mixed(quads, tris, nodes, E=5.0, nu=0.3, thickness=1.0)

    # 節点0(u,v) と 節点2(u,v) を拘束 -> DOF = [0,1, 4,5]
    fixed = np.array([0, 1, 4, 5], dtype=int)

    _manufactured_solve_check(K, fixed, seed=7)


def test_mixed_quad4_tri3():
    """Q4+TRI3: 正方形Q4に右側へTRI3を接続した混在メッシュでソルバ検証"""
    nodes = np.array(
        [
            [0, 0.0, 0.0],  # 左下
            [1, 1.0, 0.0],
            [2, 1.0, 1.0],
            [3, 0.0, 1.0],
            [4, 2.0, 0.5],  # 右側新点
        ],
        dtype=float,
    )
    quads = np.array([[0, 1, 2, 3]], dtype=int)
    tris = np.array([[1, 4, 2]], dtype=int)

    K = assemble_K_from_arrays_mixed(
        quads, tris, nodes, E=100.0, nu=0.29, thickness=1.0
    )

    # 左端2節点(0,3)の全DOF拘束
    fixed = np.array([0, 1, 6, 7], dtype=int)

    _manufactured_solve_check(K, fixed, seed=2025)


if __name__ == "__main__":
    test_pure_quad4_unit_square()
    test_mixed_quad4_tri3()
