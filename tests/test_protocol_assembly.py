"""Protocol ベースの要素・材料・アセンブリのテスト."""

from __future__ import annotations

import numpy as np

from xkep_cae.assembly import assemble_global_stiffness
from xkep_cae.bc import apply_dirichlet
from xkep_cae.core.constitutive import ConstitutiveProtocol, PlasticConstitutiveProtocol
from xkep_cae.core.element import (
    DynamicElementProtocol,
    ElementProtocol,
    NonlinearElementProtocol,
)
from xkep_cae.elements.beam_eb2d import EulerBernoulliBeam2D
from xkep_cae.elements.beam_timo2d import TimoshenkoBeam2D
from xkep_cae.elements.beam_timo3d import TimoshenkoBeam3D
from xkep_cae.elements.quad4 import Quad4PlaneStrain
from xkep_cae.elements.quad4_bbar import Quad4BBarPlaneStrain
from xkep_cae.elements.quad4_eas_bbar import Quad4EASBBarPlaneStrain, Quad4EASPlaneStrain
from xkep_cae.elements.tri3 import Tri3PlaneStrain
from xkep_cae.elements.tri6 import Tri6PlaneStrain
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.materials.elastic import PlaneStrainElastic
from xkep_cae.materials.plasticity_1d import Plasticity1D
from xkep_cae.sections.beam import BeamSection, BeamSection2D
from xkep_cae.solver import solve_displacement


def test_protocol_isinstance():
    """全要素・材料クラスがProtocolに適合すること"""
    mat = PlaneStrainElastic(200e3, 0.3)
    assert isinstance(mat, ConstitutiveProtocol)

    for cls in [
        Quad4PlaneStrain,
        Tri3PlaneStrain,
        Tri6PlaneStrain,
        Quad4BBarPlaneStrain,
        Quad4EASPlaneStrain,
        Quad4EASBBarPlaneStrain,
    ]:
        obj = cls()
        assert isinstance(obj, ElementProtocol), f"{cls.__name__} is not ElementProtocol"


def test_nonlinear_element_protocol():
    """NonlinearElementProtocol 適合テスト — CosseratRod."""
    from xkep_cae.elements.beam_cosserat import CosseratRod
    from xkep_cae.sections.beam import BeamSection

    sec = BeamSection.circle(d=10.0)
    rod = CosseratRod(section=sec)
    assert isinstance(rod, ElementProtocol)
    assert isinstance(rod, NonlinearElementProtocol)


def test_dynamic_element_protocol():
    """DynamicElementProtocol 適合テスト — 梁要素が mass_matrix を持つこと."""
    sec2d = BeamSection2D(A=100.0, I=833.333)
    sec3d = BeamSection.circle(d=10.0)

    eb = EulerBernoulliBeam2D(section=sec2d)
    timo2d = TimoshenkoBeam2D(section=sec2d)
    timo3d = TimoshenkoBeam3D(section=sec3d)

    for elem in [eb, timo2d, timo3d]:
        assert isinstance(elem, ElementProtocol)
        assert isinstance(elem, DynamicElementProtocol), (
            f"{type(elem).__name__} is not DynamicElementProtocol"
        )


def test_plastic_constitutive_protocol():
    """PlasticConstitutiveProtocol 適合テスト — Plasticity1D."""
    from xkep_cae.materials.plasticity_1d import IsotropicHardening

    iso = IsotropicHardening(sigma_y0=250.0, H_iso=1000.0)
    mat = Plasticity1D(E=200e3, iso=iso)
    # Plasticity1D は return_mapping() を持つので PlasticConstitutiveProtocol に適合
    assert isinstance(mat, PlasticConstitutiveProtocol)
    # Plasticity1D は tangent() を持たないので ConstitutiveProtocol には適合しない
    assert not isinstance(mat, ConstitutiveProtocol)


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


# =====================================================================
# 3D Timoshenko 梁のアセンブリテスト（円形断面デフォルト）
# =====================================================================


def test_assembly_beam_timo3d_single():
    """3D Timoshenko梁 単一要素のアセンブリテスト（対称性・半正定値性）."""
    sec = BeamSection.circle(d=10.0)
    beam = TimoshenkoBeam3D(section=sec)
    mat = BeamElastic1D(E=200e3, nu=0.3)

    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )
    conn = np.array([[0, 1]])

    K = assemble_global_stiffness(
        nodes,
        [(beam, conn)],
        mat,
        show_progress=False,
    )

    assert K.shape == (12, 12)
    Kd = K.toarray()
    # 対称性
    assert np.allclose(Kd, Kd.T, atol=1e-10)
    # 半正定値（6つの剛体モード → 6つのゼロ固有値）
    eigenvalues = np.linalg.eigvalsh(Kd)
    assert np.all(eigenvalues > -1e-8)
    assert np.sum(np.abs(eigenvalues) < 1e-6) == 6


def test_assembly_beam_timo3d_cantilever_y():
    """3D Timoshenko梁 片持ち梁 y方向荷重（解析解比較、円形断面）.

    解析解: δ_y = PL³/(3EIz) + PL/(κGA)
    円形断面では Iy = Iz なので xy面曲げ = xz面曲げ
    """
    E = 200e3
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    d = 10.0
    L_total = 100.0
    n_elems = 20
    P = 1.0
    kappa = 5.0 / 6.0

    sec = BeamSection.circle(d=d)
    beam = TimoshenkoBeam3D(section=sec, kappa_y=kappa, kappa_z=kappa)
    mat = BeamElastic1D(E=E, nu=nu)

    # メッシュ生成（x軸方向）
    n_nodes = n_elems + 1
    s = np.linspace(0, L_total, n_nodes)
    nodes = np.column_stack([s, np.zeros(n_nodes), np.zeros(n_nodes)])
    conn = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])

    K = assemble_global_stiffness(
        nodes,
        [(beam, conn)],
        mat,
        show_progress=False,
    )

    ndof = 6 * n_nodes
    assert K.shape == (ndof, ndof)

    # 先端 y方向荷重
    f = np.zeros(ndof)
    f[6 * n_elems + 1] = P  # uy at tip

    # 固定端: 節点0 の全 6 DOF
    fixed_dofs = np.arange(6, dtype=int)
    Kbc, fbc = apply_dirichlet(K, f, fixed_dofs, 0.0)
    u, info = solve_displacement(Kbc, fbc, size_threshold=4000)

    # 解析解: δ_y = PL³/(3EIz) + PL/(κGA)
    delta_analytical = P * L_total**3 / (3.0 * E * sec.Iz) + P * L_total / (kappa * G * sec.A)
    delta_fem = u[6 * n_elems + 1]
    assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-10


def test_assembly_beam_timo3d_cantilever_z():
    """3D Timoshenko梁 片持ち梁 z方向荷重（解析解比較、円形断面）.

    解析解: δ_z = PL³/(3EIy) + PL/(κGA)
    """
    E = 200e3
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    d = 10.0
    L_total = 100.0
    n_elems = 20
    P = 1.0
    kappa = 5.0 / 6.0

    sec = BeamSection.circle(d=d)
    beam = TimoshenkoBeam3D(section=sec, kappa_y=kappa, kappa_z=kappa)
    mat = BeamElastic1D(E=E, nu=nu)

    n_nodes = n_elems + 1
    s = np.linspace(0, L_total, n_nodes)
    nodes = np.column_stack([s, np.zeros(n_nodes), np.zeros(n_nodes)])
    conn = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])

    K = assemble_global_stiffness(
        nodes,
        [(beam, conn)],
        mat,
        show_progress=False,
    )

    ndof = 6 * n_nodes
    f = np.zeros(ndof)
    f[6 * n_elems + 2] = P  # uz at tip

    fixed_dofs = np.arange(6, dtype=int)
    Kbc, fbc = apply_dirichlet(K, f, fixed_dofs, 0.0)
    u, info = solve_displacement(Kbc, fbc, size_threshold=4000)

    # 解析解: δ_z = PL³/(3EIy) + PL/(κGA)
    delta_analytical = P * L_total**3 / (3.0 * E * sec.Iy) + P * L_total / (kappa * G * sec.A)
    delta_fem = u[6 * n_elems + 2]
    assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-10


def test_assembly_beam_timo3d_torsion():
    """3D Timoshenko梁 片持ち梁 ねじり（解析解比較、円形断面）.

    解析解: θ_tip = T·L/(G·J)
    """
    E = 200e3
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    d = 10.0
    L_total = 100.0
    n_elems = 10
    T_torque = 10.0

    sec = BeamSection.circle(d=d)
    beam = TimoshenkoBeam3D(section=sec)
    mat = BeamElastic1D(E=E, nu=nu)

    n_nodes = n_elems + 1
    s = np.linspace(0, L_total, n_nodes)
    nodes = np.column_stack([s, np.zeros(n_nodes), np.zeros(n_nodes)])
    conn = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])

    K = assemble_global_stiffness(
        nodes,
        [(beam, conn)],
        mat,
        show_progress=False,
    )

    ndof = 6 * n_nodes
    f = np.zeros(ndof)
    f[6 * n_elems + 3] = T_torque  # θx at tip

    fixed_dofs = np.arange(6, dtype=int)
    Kbc, fbc = apply_dirichlet(K, f, fixed_dofs, 0.0)
    u, info = solve_displacement(Kbc, fbc, size_threshold=4000)

    # 解析解: θ_tip = T·L/(G·J)
    theta_analytical = T_torque * L_total / (G * sec.J)
    theta_fem = u[6 * n_elems + 3]
    assert abs(theta_fem - theta_analytical) / abs(theta_analytical) < 1e-10


def test_assembly_beam_timo3d_inclined():
    """3D Timoshenko梁 傾斜梁のアセンブリテスト（座標変換検証、円形断面）.

    45度傾斜梁（xz面）で先端荷重を加え、解析解と比較。
    梁長は同じなので、断面が等方（円形）なら結果は軸方向に依存しない。
    """
    E = 200e3
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    d = 10.0
    L_total = 100.0
    n_elems = 20
    P = 1.0
    kappa = 5.0 / 6.0

    sec = BeamSection.circle(d=d)
    beam = TimoshenkoBeam3D(section=sec, kappa_y=kappa, kappa_z=kappa)
    mat = BeamElastic1D(E=E, nu=nu)

    # 45度傾斜梁（xz面内）
    n_nodes = n_elems + 1
    s = np.linspace(0, L_total / np.sqrt(2.0), n_nodes)
    nodes = np.column_stack([s, np.zeros(n_nodes), s])
    conn = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])

    K = assemble_global_stiffness(
        nodes,
        [(beam, conn)],
        mat,
        show_progress=False,
    )

    ndof = 6 * n_nodes
    assert K.shape == (ndof, ndof)

    Kd = K.toarray()
    # 対称性の確認（値が大きいので相対許容差で判定）
    assert np.allclose(Kd, Kd.T, rtol=1e-12, atol=1e-6)

    # 先端に全体y方向荷重（梁の局所座標系では横荷重）
    f = np.zeros(ndof)
    f[6 * n_elems + 1] = P  # uy at tip

    fixed_dofs = np.arange(6, dtype=int)
    Kbc, fbc = apply_dirichlet(K, f, fixed_dofs, 0.0)
    u, info = solve_displacement(Kbc, fbc, size_threshold=4000)

    # 円形断面では梁軸方向に関わらず横方向たわみの解析解は同じ
    # δ = PL³/(3EI) + PL/(κGA) (I = Iy = Iz for circular)
    delta_analytical = P * L_total**3 / (3.0 * E * sec.Iz) + P * L_total / (kappa * G * sec.A)
    delta_fem = u[6 * n_elems + 1]
    assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-8
