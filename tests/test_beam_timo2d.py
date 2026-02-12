"""2D Timoshenko 梁要素のテスト.

検証項目:
  - Φ→0 の極限で Euler-Bernoulli 梁と一致
  - せん断変形の影響（太い梁ではEB梁よりたわみが大きい）
  - 解析解比較: Timoshenko片持ち梁の先端たわみ
  - Protocol適合性
  - 剛性行列の対称性
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.bc import apply_dirichlet
from xkep_cae.core.element import ElementProtocol
from xkep_cae.elements.beam_eb2d import eb_beam2d_ke_local
from xkep_cae.elements.beam_timo2d import (
    TimoshenkoBeam2D,
    timo_beam2d_distributed_load,
    timo_beam2d_ke_global,
    timo_beam2d_ke_local,
)
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection2D
from xkep_cae.solver import solve_displacement

# =====================================================================
# テストパラメータ
# =====================================================================
E = 200e3  # MPa
NU = 0.3
G = E / (2.0 * (1.0 + NU))  # ≈ 76923 MPa
KAPPA = 5.0 / 6.0  # 矩形断面のせん断補正係数


def _make_cantilever_mesh(
    n_elems: int, total_length: float, angle_deg: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """片持ち梁のメッシュ生成."""
    angle_rad = np.deg2rad(angle_deg)
    n_nodes = n_elems + 1
    s = np.linspace(0, total_length, n_nodes)
    nodes = np.column_stack([s * np.cos(angle_rad), s * np.sin(angle_rad)])
    connectivity = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])
    return nodes, connectivity


def _assemble_beam_system(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    ke_func,
) -> tuple[np.ndarray, int]:
    """梁構造の全体剛性行列を組み立てる."""
    n_nodes = len(nodes)
    ndof = 3 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)

    for elem_nodes in connectivity:
        n1, n2 = elem_nodes
        coords = nodes[[n1, n2]]
        Ke = ke_func(coords)

        edofs = np.array([3 * n1, 3 * n1 + 1, 3 * n1 + 2, 3 * n2, 3 * n2 + 1, 3 * n2 + 2])
        for ii in range(6):
            for jj in range(6):
                K[edofs[ii], edofs[jj]] += Ke[ii, jj]

    return K, ndof


def _solve_cantilever_point_load(
    n_elems: int,
    total_length: float,
    area: float,
    inertia: float,
    P: float,
    ke_func,
) -> np.ndarray:
    """片持ち梁の先端集中荷重問題を解く."""
    import scipy.sparse as sp

    nodes, conn = _make_cantilever_mesh(n_elems, total_length)
    K, ndof = _assemble_beam_system(nodes, conn, ke_func)

    f = np.zeros(ndof)
    tip_node = n_elems
    f[3 * tip_node + 1] = P

    K_sp = sp.csr_matrix(K)
    fixed_dofs = np.array([0, 1, 2], dtype=int)
    Kbc, fbc = apply_dirichlet(K_sp, f, fixed_dofs)
    u, _ = solve_displacement(Kbc, fbc, show_progress=False)
    return u


# =====================================================================
# テスト
# =====================================================================


class TestLocalStiffnessMatrix:
    """局所剛性行列の基本検証."""

    def test_symmetry(self):
        """Timoshenko局所Keが対称であること."""
        Ke = timo_beam2d_ke_local(E, 100.0, 833.333, 100.0, KAPPA, G)
        assert np.allclose(Ke, Ke.T, atol=1e-10)

    def test_shape(self):
        """Keの形状が(6,6)であること."""
        Ke = timo_beam2d_ke_local(E, 100.0, 833.333, 100.0, KAPPA, G)
        assert Ke.shape == (6, 6)

    def test_rigid_body_modes(self):
        """3つの剛体モード（ゼロ固有値）を持つこと."""
        Ke = timo_beam2d_ke_local(E, 100.0, 833.333, 100.0, KAPPA, G)
        eigenvalues = np.linalg.eigvalsh(Ke)
        assert np.sum(np.abs(eigenvalues) < 1e-6) == 3

    def test_positive_semidefinite(self):
        """半正定値であること."""
        Ke = timo_beam2d_ke_local(E, 100.0, 833.333, 100.0, KAPPA, G)
        eigenvalues = np.linalg.eigvalsh(Ke)
        assert np.all(eigenvalues > -1e-8)


class TestConvergenceToEB:
    """Φ→0 の極限で Euler-Bernoulli 梁と一致."""

    def test_slender_beam_matches_eb(self):
        """細長い梁（Φ≈0）でEB梁の剛性行列と一致すること."""
        # 細長い梁: L=1000, I=0.01 (Φ ≈ 0)
        area = 100.0
        inertia = 0.01  # 非常に小さいI → Φ ≈ 0
        length = 1000.0

        Ke_timo = timo_beam2d_ke_local(E, area, inertia, length, KAPPA, G)
        Ke_eb = eb_beam2d_ke_local(E, area, inertia, length)

        # Φが十分小さいので差が小さいはず
        assert np.allclose(Ke_timo, Ke_eb, rtol=1e-4)

    def test_large_kappa_ga_approaches_eb(self):
        """κGA が非常に大きい場合にEBと一致すること."""
        area = 100.0
        inertia = 833.333
        length = 100.0

        # Gを非常に大きくしてせん断変形をゼロに近づける
        G_huge = 1e20
        Ke_timo = timo_beam2d_ke_local(E, area, inertia, length, KAPPA, G_huge)
        Ke_eb = eb_beam2d_ke_local(E, area, inertia, length)

        assert np.allclose(Ke_timo, Ke_eb, rtol=1e-8)


class TestTimoshenkoAnalyticalSolution:
    """Timoshenko片持ち梁の解析解比較.

    解析解（先端集中荷重P）:
      δ_tip = PL³/(3EI) + PL/(κGA)
      θ_tip = PL²/(2EI)

    第1項はEB梁のたわみ、第2項がせん断変形の追加分。
    """

    @pytest.fixture()
    def thick_beam_params(self):
        """太い梁のパラメータ（せん断変形の影響が顕著）."""
        return {
            "area": 100.0,  # 10x10 矩形
            "inertia": 833.333,
            "length": 100.0,  # L/h = 10（太い梁）
            "n_elems": 20,
            "P": 1.0,
        }

    def test_tip_deflection_thick_beam(self, thick_beam_params):
        """太い梁の先端たわみがTimoshenko解析解と一致すること."""
        p = thick_beam_params
        delta_analytical = p["P"] * p["length"] ** 3 / (3.0 * E * p["inertia"]) + p["P"] * p[
            "length"
        ] / (KAPPA * G * p["area"])

        def ke_func(coords):
            return timo_beam2d_ke_global(coords, E, p["area"], p["inertia"], KAPPA, G)

        u = _solve_cantilever_point_load(
            p["n_elems"], p["length"], p["area"], p["inertia"], p["P"], ke_func
        )

        delta_fem = u[3 * p["n_elems"] + 1]
        assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-10

    def test_tip_rotation_thick_beam(self, thick_beam_params):
        """太い梁の先端回転角が解析解と一致すること.

        Timoshenko梁でも回転角はEBと同じ: θ = PL²/(2EI)
        """
        p = thick_beam_params
        theta_analytical = p["P"] * p["length"] ** 2 / (2.0 * E * p["inertia"])

        def ke_func(coords):
            return timo_beam2d_ke_global(coords, E, p["area"], p["inertia"], KAPPA, G)

        u = _solve_cantilever_point_load(
            p["n_elems"], p["length"], p["area"], p["inertia"], p["P"], ke_func
        )

        theta_fem = u[3 * p["n_elems"] + 2]
        assert abs(theta_fem - theta_analytical) / abs(theta_analytical) < 1e-10

    def test_shear_contribution(self, thick_beam_params):
        """TimoshenkoのたわみがEB梁より大きいこと（せん断変形分）."""
        p = thick_beam_params

        def ke_timo(coords):
            return timo_beam2d_ke_global(coords, E, p["area"], p["inertia"], KAPPA, G)

        from xkep_cae.elements.beam_eb2d import eb_beam2d_ke_global

        def ke_eb(coords):
            return eb_beam2d_ke_global(coords, E, p["area"], p["inertia"])

        u_timo = _solve_cantilever_point_load(
            p["n_elems"], p["length"], p["area"], p["inertia"], p["P"], ke_timo
        )
        u_eb = _solve_cantilever_point_load(
            p["n_elems"], p["length"], p["area"], p["inertia"], p["P"], ke_eb
        )

        delta_timo = u_timo[3 * p["n_elems"] + 1]
        delta_eb = u_eb[3 * p["n_elems"] + 1]

        # Timoshenko のたわみ > EB のたわみ
        assert delta_timo > delta_eb

        # せん断変形の追加分を確認
        shear_extra = p["P"] * p["length"] / (KAPPA * G * p["area"])
        assert abs((delta_timo - delta_eb) - shear_extra) / shear_extra < 1e-8


class TestDistributedLoad:
    """等分布荷重の解析解比較.

    Timoshenko片持ち梁の等分布荷重:
      δ_tip = qL⁴/(8EI) + qL²/(2κGA)
    """

    def test_tip_deflection_distributed(self):
        """等分布荷重時の先端たわみが解析解と一致すること."""
        import scipy.sparse as sp

        area = 100.0
        inertia = 833.333
        total_length = 100.0
        n_elems = 20
        q = 0.01

        delta_analytical = q * total_length**4 / (8.0 * E * inertia) + q * total_length**2 / (
            2.0 * KAPPA * G * area
        )

        nodes, conn = _make_cantilever_mesh(n_elems, total_length)
        K, ndof = _assemble_beam_system(
            nodes,
            conn,
            lambda coords: timo_beam2d_ke_global(coords, E, area, inertia, KAPPA, G),
        )

        f = np.zeros(ndof)
        for elem_nodes in conn:
            n1, n2 = elem_nodes
            coords = nodes[[n1, n2]]
            fe = timo_beam2d_distributed_load(coords, q)
            edofs = np.array([3 * n1, 3 * n1 + 1, 3 * n1 + 2, 3 * n2, 3 * n2 + 1, 3 * n2 + 2])
            f[edofs] += fe

        K_sp = sp.csr_matrix(K)
        Kbc, fbc = apply_dirichlet(K_sp, f, np.array([0, 1, 2]))
        u, _ = solve_displacement(Kbc, fbc, show_progress=False)

        delta_fem = u[3 * n_elems + 1]
        assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-6


class TestShearLocking:
    """せん断ロッキングが発生しないことの確認.

    細長い梁（L/h >> 10）でもTimoshenko解がEB解に近づくこと。
    ロッキングが起きると剛性が過大になり、たわみが過小になる。
    """

    def test_no_locking_slender_beam(self):
        """L/h=100の細長い梁でEB解析解の98%以上のたわみが得られること."""
        # 細長い梁: h=1, L=100 → L/h=100
        b, h = 1.0, 1.0
        area = b * h
        inertia = b * h**3 / 12.0
        total_length = 100.0
        n_elems = 10
        P = 1.0

        delta_eb_analytical = P * total_length**3 / (3.0 * E * inertia)

        def ke_func(coords):
            return timo_beam2d_ke_global(coords, E, area, inertia, KAPPA, G)

        u = _solve_cantilever_point_load(n_elems, total_length, area, inertia, P, ke_func)
        delta_fem = u[3 * n_elems + 1]

        # ロッキングがなければ、たわみ ≥ EB解（せん断分があるのでわずかに大きい）
        assert delta_fem >= delta_eb_analytical * 0.99


class TestProtocolConformance:
    """Protocol適合性の検証."""

    def test_element_protocol(self):
        """TimoshenkoBeam2DがElementProtocolに適合すること."""
        sec = BeamSection2D(A=100.0, I=833.333)
        beam = TimoshenkoBeam2D(section=sec)
        assert isinstance(beam, ElementProtocol)

    def test_element_class_stiffness(self):
        """クラスインタフェース経由の剛性行列が関数版と一致すること."""
        sec = BeamSection2D(A=100.0, I=833.333)
        beam = TimoshenkoBeam2D(section=sec, kappa=KAPPA)
        mat = BeamElastic1D(E=E, nu=NU)

        coords = np.array([[0.0, 0.0], [100.0, 0.0]])
        Ke_class = beam.local_stiffness(coords, mat)
        Ke_func = timo_beam2d_ke_global(coords, E, 100.0, 833.333, KAPPA, G)

        assert np.allclose(Ke_class, Ke_func, atol=1e-10)

    def test_material_shear_modulus(self):
        """BeamElastic1DのGが正しく計算されること."""
        mat = BeamElastic1D(E=E, nu=NU)
        assert abs(mat.G - G) < 1e-6
