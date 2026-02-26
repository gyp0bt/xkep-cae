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
from xkep_cae.elements.hex8 import Hex8BBarMean, Hex8Incompatible, Hex8Reduced, Hex8SRI
from xkep_cae.elements.quad4 import Quad4PlaneStrain
from xkep_cae.elements.quad4_bbar import Quad4BBarPlaneStrain
from xkep_cae.elements.quad4_eas_bbar import Quad4EASBBarPlaneStrain, Quad4EASPlaneStrain
from xkep_cae.elements.tri3 import Tri3PlaneStrain
from xkep_cae.elements.tri6 import Tri6PlaneStrain
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.materials.elastic import IsotropicElastic3D, PlaneStrainElastic
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
        Hex8SRI,
        Hex8Incompatible,
        Hex8BBarMean,
    ]:
        obj = cls()
        assert isinstance(obj, ElementProtocol), f"{cls.__name__} is not ElementProtocol"

    # コンストラクタ引数ありのクラス
    obj = Hex8Reduced(alpha_hg=0.0)
    assert isinstance(obj, ElementProtocol), "Hex8Reduced is not ElementProtocol"
    obj = Hex8SRI(alpha_hg=0.0)
    assert isinstance(obj, ElementProtocol), "Hex8SRI(alpha_hg) is not ElementProtocol"


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


# =====================================================================
# HEX8 要素のアセンブリテスト（3D固体）
# =====================================================================


def _make_unit_cube_nodes():
    """単位立方体の 8 節点座標を返す."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )


def test_assembly_hex8_sri_single():
    """C3D8 (SRI) 単一要素のアセンブリテスト（対称性・半正定値性）.

    SRI は偏差成分1点積分のため rank=12（6 RBM + 6 低減積分ゼロエネルギーモード）。
    """
    nodes = _make_unit_cube_nodes()
    mat = IsotropicElastic3D(200e9, 0.3)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])

    K = assemble_global_stiffness(
        nodes,
        [(Hex8SRI(), conn)],
        mat,
        show_progress=False,
    )

    assert K.shape == (24, 24)
    Kd = K.toarray()
    # 対称性
    assert np.allclose(Kd, Kd.T, atol=1e-6)
    # 半正定値
    eigvals = np.linalg.eigvalsh(Kd)
    assert np.all(eigvals > -1e-4)
    # SRI+B-bar (default): alpha_hg=0.03 によりランク > 12
    rank = np.sum(eigvals > 1e-4 * np.max(eigvals))
    assert rank > 12, f"expected rank > 12 for SRI+B-bar+HG, got {rank}"


def test_assembly_hex8_incompatible_single():
    """C3D8I (非適合モード) 単一要素のアセンブリテスト."""
    nodes = _make_unit_cube_nodes()
    mat = IsotropicElastic3D(200e9, 0.3)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])

    K = assemble_global_stiffness(
        nodes,
        [(Hex8Incompatible(), conn)],
        mat,
        show_progress=False,
    )

    assert K.shape == (24, 24)
    Kd = K.toarray()
    assert np.allclose(Kd, Kd.T, atol=1e-6)
    eigvals = np.linalg.eigvalsh(Kd)
    assert np.all(eigvals > -1e-4)
    n_zero = np.sum(np.abs(eigvals) < 1e-4 * np.max(eigvals))
    assert n_zero == 6, f"expected 6 RBM, got {n_zero}"


def test_assembly_hex8_reduced_single():
    """C3D8R (低減積分) 単一要素のアセンブリテスト."""
    nodes = _make_unit_cube_nodes()
    mat = IsotropicElastic3D(200e9, 0.3)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])

    # アワーグラス制御なし: ランク 6
    K_no_hg = assemble_global_stiffness(
        nodes,
        [(Hex8Reduced(alpha_hg=0.0), conn)],
        mat,
        show_progress=False,
    )
    assert K_no_hg.shape == (24, 24)
    Kd = K_no_hg.toarray()
    assert np.allclose(Kd, Kd.T, atol=1e-6)
    eigvals = np.linalg.eigvalsh(Kd)
    assert np.all(eigvals > -1e-4)
    rank = np.sum(eigvals > 1e-4 * np.max(eigvals))
    assert rank == 6, f"expected rank 6 without HG control, got {rank}"

    # アワーグラス制御あり: ランク上昇
    K_hg = assemble_global_stiffness(
        nodes,
        [(Hex8Reduced(alpha_hg=0.05), conn)],
        mat,
        show_progress=False,
    )
    eigvals_hg = np.linalg.eigvalsh(K_hg.toarray())
    rank_hg = np.sum(eigvals_hg > 1e-4 * np.max(eigvals_hg))
    assert rank_hg > rank, "HG control should increase rank"


def _make_beam_hex8_mesh(nx, ny, nz, Lx, Ly, Lz):
    """nx×ny×nz の HEX8 構造格子メッシュを生成.

    Args:
        nx, ny, nz: x, y, z 方向の要素分割数
        Lx, Ly, Lz: x, y, z 方向の長さ

    Returns:
        nodes: ((nx+1)*(ny+1)*(nz+1), 3)
        conn: (nx*ny*nz, 8)
    """
    xs = np.linspace(0, Lx, nx + 1)
    ys = np.linspace(0, Ly, ny + 1)
    zs = np.linspace(0, Lz, nz + 1)

    node_list = []
    for iz in range(nz + 1):
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                node_list.append([xs[ix], ys[iy], zs[iz]])
    nodes = np.array(node_list)

    def node_id(ix, iy, iz):
        return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix

    conn_list = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                n0 = node_id(ix, iy, iz)
                n1 = node_id(ix + 1, iy, iz)
                n2 = node_id(ix + 1, iy + 1, iz)
                n3 = node_id(ix, iy + 1, iz)
                n4 = node_id(ix, iy, iz + 1)
                n5 = node_id(ix + 1, iy, iz + 1)
                n6 = node_id(ix + 1, iy + 1, iz + 1)
                n7 = node_id(ix, iy + 1, iz + 1)
                conn_list.append([n0, n1, n2, n3, n4, n5, n6, n7])
    conn = np.array(conn_list, dtype=int)
    return nodes, conn


def test_assembly_hex8_multi_element():
    """HEX8 複数要素アセンブリ（2×1×1 メッシュ、C3D8I）.

    C3D8I は完全積分ベースなのでランク不足が最小限。
    """
    nodes, conn = _make_beam_hex8_mesh(2, 1, 1, 2.0, 1.0, 1.0)
    mat = IsotropicElastic3D(200e9, 0.3)

    K = assemble_global_stiffness(
        nodes,
        [(Hex8Incompatible(), conn)],
        mat,
        show_progress=False,
    )

    n_nodes = len(nodes)
    assert K.shape == (3 * n_nodes, 3 * n_nodes)
    Kd = K.toarray()
    # 対称性
    assert np.allclose(Kd, Kd.T, atol=1e-4)
    # 半正定値
    eigvals = np.linalg.eigvalsh(Kd)
    assert np.all(eigvals > -1e-4)
    # 6 RBM（剛体モード）
    n_zero = np.sum(np.abs(eigvals) < 1e-4 * np.max(eigvals))
    assert n_zero == 6, f"expected 6 RBM, got {n_zero}"


def test_assembly_hex8_manufactured_solution():
    """HEX8 製造解テスト（C3D8I, 2×1×1 メッシュ）.

    ランダム変位ベクトルに対して f = K·u を計算し、
    境界条件付きでソルブして一致を確認する。
    """
    nodes, conn = _make_beam_hex8_mesh(2, 1, 1, 2.0, 1.0, 1.0)
    mat = IsotropicElastic3D(200e9, 0.3)

    K = assemble_global_stiffness(
        nodes,
        [(Hex8Incompatible(), conn)],
        mat,
        show_progress=False,
    )

    ndof = K.shape[0]
    # x=0 面の節点を固定（全 DOF）
    fixed_nodes = np.where(np.abs(nodes[:, 0]) < 1e-12)[0]
    fixed_dofs = np.concatenate([3 * fixed_nodes + i for i in range(3)])

    rng = np.random.default_rng(12345)
    u_true = rng.standard_normal(ndof)
    u_true[fixed_dofs] = 0.0
    f = K @ u_true

    Kbc, fbc = apply_dirichlet(K, f, fixed_dofs, 0.0)
    u_sol, info = solve_displacement(Kbc, fbc, size_threshold=4000)

    free = np.setdiff1d(np.arange(ndof), fixed_dofs)
    assert np.allclose(u_sol[free], u_true[free], rtol=1e-10, atol=1e-10)


def test_assembly_hex8_cantilever():
    """HEX8 片持ち梁テスト（C3D8I, Timoshenko理論との比較）.

    L=10, h=w=1, E=200GPa, nu=0.3, P=1000N.
    C3D8I は断面 1×1 でも曲げ精度 < 5%.
    """
    E = 200e9
    nu = 0.3
    L, h, w = 10.0, 1.0, 1.0
    P = 1000.0
    nx, ny, nz = 8, 1, 1  # 8 要素 (長手方向)

    nodes, conn = _make_beam_hex8_mesh(nx, ny, nz, L, h, w)
    mat = IsotropicElastic3D(E, nu)

    K = assemble_global_stiffness(
        nodes,
        [(Hex8Incompatible(), conn)],
        mat,
        show_progress=False,
    )

    ndof = K.shape[0]
    # 固定端: x=0 の全節点
    fixed_nodes = np.where(np.abs(nodes[:, 0]) < 1e-12)[0]
    fixed_dofs = np.concatenate([3 * fixed_nodes + i for i in range(3)])

    # 荷重: x=L の全節点に均等分配 (y 方向)
    tip_nodes = np.where(np.abs(nodes[:, 0] - L) < 1e-12)[0]
    f = np.zeros(ndof)
    p_per_node = P / len(tip_nodes)
    for n in tip_nodes:
        f[3 * n + 1] = p_per_node  # uy

    Kbc, fbc = apply_dirichlet(K, f, fixed_dofs, 0.0)
    u_sol, info = solve_displacement(Kbc, fbc, size_threshold=4000)

    # 先端 y 方向変位の平均
    tip_uy = np.mean([u_sol[3 * n + 1] for n in tip_nodes])

    # Timoshenko 解析解
    Iy = w * h**3 / 12.0
    G = E / (2.0 * (1.0 + nu))
    kappa = 5.0 / 6.0
    A = h * w
    delta_timo = P * L**3 / (3.0 * E * Iy) + P * L / (kappa * G * A)

    rel_err = abs(tip_uy - delta_timo) / abs(delta_timo)
    assert rel_err < 0.05, f"cantilever error {rel_err:.4f} > 5%"
