"""Protocol ベースの要素・材料・アセンブリのテスト."""

from __future__ import annotations

import numpy as np

from xkep_cae.assembly import assemble_global_stiffness
from xkep_cae.bc import apply_dirichlet
from xkep_cae.core.constitutive import ConstitutiveProtocol
from xkep_cae.core.element import ElementProtocol
from xkep_cae.elements.beam_eb2d import EulerBernoulliBeam2D
from xkep_cae.elements.beam_timo2d import TimoshenkoBeam2D
from xkep_cae.elements.quad4 import Quad4PlaneStrain
from xkep_cae.elements.quad4_bbar import Quad4BBarPlaneStrain
from xkep_cae.elements.tri3 import Tri3PlaneStrain
from xkep_cae.elements.tri6 import Tri6PlaneStrain
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.materials.elastic import PlaneStrainElastic
from xkep_cae.sections.beam import BeamSection2D
from xkep_cae.solver import solve_displacement


def test_protocol_isinstance():
    """全要素・材料クラスがProtocolに適合すること"""
    mat = PlaneStrainElastic(200e3, 0.3)
    assert isinstance(mat, ConstitutiveProtocol)

    for cls in [Quad4PlaneStrain, Tri3PlaneStrain, Tri6PlaneStrain, Quad4BBarPlaneStrain]:
        obj = cls()
        assert isinstance(obj, ElementProtocol), f"{cls.__name__} is not ElementProtocol"


def test_assembly_quad4():
    """Q4単体のアセンブリテスト"""
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    mat = PlaneStrainElastic(10.0, 0.25)
    conn = np.array([[0, 1, 2, 3]])

    K = assemble_global_stiffness(
        nodes,
        [(Quad4PlaneStrain(), conn)],
        mat,
        thickness=1.0,
        show_progress=False,
    )
    assert K.shape == (8, 8)
    Kd = K.toarray()
    assert np.allclose(Kd, Kd.T, atol=1e-12)
    assert np.linalg.eigvalsh(Kd).min() > -1e-10


def test_assembly_tri3():
    """TRI3単体のアセンブリテスト"""
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    mat = PlaneStrainElastic(5.0, 0.3)
    conn = np.array([[0, 1, 2]])

    K = assemble_global_stiffness(
        nodes,
        [(Tri3PlaneStrain(), conn)],
        mat,
        thickness=1.0,
        show_progress=False,
    )
    assert K.shape == (6, 6)
    Kd = K.toarray()
    assert np.allclose(Kd, Kd.T, atol=1e-12)


def test_assembly_tri6():
    """TRI6単体のアセンブリテスト"""
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.0],
            [0.75, 0.5],
            [0.25, 0.5],
        ]
    )
    mat = PlaneStrainElastic(200e3, 0.3)
    conn = np.array([[0, 1, 2, 3, 4, 5]])

    K = assemble_global_stiffness(
        nodes,
        [(Tri6PlaneStrain(), conn)],
        mat,
        thickness=1.0,
        show_progress=False,
    )
    assert K.shape == (12, 12)
    Kd = K.toarray()
    assert np.allclose(Kd, Kd.T, atol=1e-10)


def test_generic_assembly_mixed_solve():
    """汎用アセンブリでQ4+TRI3混在メッシュのソルブが成功すること"""
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [2.0, 0.5],
        ]
    )
    mat = PlaneStrainElastic(100.0, 0.29)

    conn_q4 = np.array([[0, 1, 2, 3]])
    conn_t3 = np.array([[1, 4, 2]])

    K = assemble_global_stiffness(
        nodes,
        [(Quad4PlaneStrain(), conn_q4), (Tri3PlaneStrain(), conn_t3)],
        mat,
        thickness=1.0,
        show_progress=False,
    )

    ndof = K.shape[0]
    fixed_dofs = np.array([0, 1, 6, 7], dtype=int)

    # 製造解テスト
    rng = np.random.default_rng(42)
    u_true = rng.standard_normal(ndof)
    u_true[fixed_dofs] = 0.0
    f = K @ u_true

    Kbc, fbc = apply_dirichlet(K, f, fixed_dofs, 0.0)
    u, info = solve_displacement(Kbc, fbc, size_threshold=4000)

    free = np.setdiff1d(np.arange(ndof), fixed_dofs)
    assert np.allclose(u[free], u_true[free], rtol=1e-10, atol=1e-10)


def test_assembly_beam_eb():
    """EB梁のassemble_global_stiffnessアセンブリテスト（片持ち梁解析解比較）"""
    E = 200e3
    A = 100.0
    I_val = 833.333
    L_total = 1000.0
    n_elems = 10
    P = 1.0

    sec = BeamSection2D(A=A, I=I_val)
    beam = EulerBernoulliBeam2D(section=sec)
    mat = BeamElastic1D(E=E)

    # メッシュ生成
    n_nodes = n_elems + 1
    s = np.linspace(0, L_total, n_nodes)
    nodes = np.column_stack([s, np.zeros(n_nodes)])
    conn = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])

    K = assemble_global_stiffness(
        nodes,
        [(beam, conn)],
        mat,
        show_progress=False,
    )

    ndof = 3 * n_nodes
    assert K.shape == (ndof, ndof)

    Kd = K.toarray()
    assert np.allclose(Kd, Kd.T, atol=1e-10)

    # 先端集中荷重 → ソルブ
    f = np.zeros(ndof)
    f[3 * n_elems + 1] = P

    fixed_dofs = np.array([0, 1, 2], dtype=int)
    Kbc, fbc = apply_dirichlet(K, f, fixed_dofs, 0.0)
    u, info = solve_displacement(Kbc, fbc, size_threshold=4000)

    # 解析解: δ_tip = PL³/(3EI)
    delta_analytical = P * L_total**3 / (3.0 * E * I_val)
    delta_fem = u[3 * n_elems + 1]
    assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-10


def test_assembly_beam_timo():
    """Timoshenko梁のassemble_global_stiffnessアセンブリテスト（片持ち梁解析解比較）"""
    E = 200e3
    nu = 0.3
    A = 100.0
    I_val = 833.333
    L_total = 100.0
    n_elems = 20
    P = 1.0
    kappa = 5.0 / 6.0
    G = E / (2.0 * (1.0 + nu))

    sec = BeamSection2D(A=A, I=I_val)
    beam = TimoshenkoBeam2D(section=sec, kappa=kappa)
    mat = BeamElastic1D(E=E, nu=nu)

    n_nodes = n_elems + 1
    s = np.linspace(0, L_total, n_nodes)
    nodes = np.column_stack([s, np.zeros(n_nodes)])
    conn = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])

    K = assemble_global_stiffness(
        nodes,
        [(beam, conn)],
        mat,
        show_progress=False,
    )

    ndof = 3 * n_nodes
    assert K.shape == (ndof, ndof)

    f = np.zeros(ndof)
    f[3 * n_elems + 1] = P

    fixed_dofs = np.array([0, 1, 2], dtype=int)
    Kbc, fbc = apply_dirichlet(K, f, fixed_dofs, 0.0)
    u, info = solve_displacement(Kbc, fbc, size_threshold=4000)

    # 解析解: δ_tip = PL³/(3EI) + PL/(κGA)
    delta_analytical = P * L_total**3 / (3.0 * E * I_val) + P * L_total / (kappa * G * A)
    delta_fem = u[3 * n_elems + 1]
    assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-10
