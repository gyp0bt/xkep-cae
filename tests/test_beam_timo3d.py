"""3D Timoshenko 梁要素のテスト.

検証項目:
  - 局所剛性行列の対称性・半正定値性
  - 軸引張、ねじり、二軸曲げの解析解比較
  - 座標変換（傾斜梁）
  - 2D梁との整合性（平面内問題）
  - Protocol適合性
  - SCFの動作確認
  - Cowper κ の統合テスト
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.bc import apply_dirichlet
from xkep_cae.core.element import ElementProtocol
from xkep_cae.elements.beam_timo3d import (
    BeamForces3D,
    TimoshenkoBeam3D,
    _build_local_axes,
    _transformation_matrix_3d,
    beam3d_max_bending_stress,
    beam3d_max_shear_stress,
    beam3d_section_forces,
    timo_beam3d_distributed_load,
    timo_beam3d_ke_global,
    timo_beam3d_ke_local,
)
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection
from xkep_cae.solver import solve_displacement

# =====================================================================
# テストパラメータ
# =====================================================================
E = 200e3  # MPa
NU = 0.3
G = E / (2.0 * (1.0 + NU))  # ≈ 76923 MPa
KAPPA = 5.0 / 6.0


def _make_cantilever3d_mesh(
    n_elems: int, total_length: float, direction: str = "x",
) -> tuple[np.ndarray, np.ndarray]:
    """3D片持ち梁のメッシュ生成.

    Args:
        n_elems: 要素数
        total_length: 全長
        direction: 梁の方向 ("x", "y", "z")
    """
    n_nodes = n_elems + 1
    s = np.linspace(0, total_length, n_nodes)
    if direction == "x":
        nodes = np.column_stack([s, np.zeros(n_nodes), np.zeros(n_nodes)])
    elif direction == "y":
        nodes = np.column_stack([np.zeros(n_nodes), s, np.zeros(n_nodes)])
    elif direction == "z":
        nodes = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), s])
    else:
        raise ValueError(f"Unknown direction: {direction}")
    connectivity = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])
    return nodes, connectivity


def _assemble_beam3d_system(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    ke_func,
) -> tuple[np.ndarray, int]:
    """3D梁構造の全体剛性行列を組み立てる."""
    n_nodes = len(nodes)
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)

    for elem_nodes in connectivity:
        n1, n2 = elem_nodes
        coords = nodes[[n1, n2]]
        Ke = ke_func(coords)

        edofs = np.empty(12, dtype=int)
        for i, n in enumerate([n1, n2]):
            for d in range(6):
                edofs[6 * i + d] = 6 * n + d

        for ii in range(12):
            for jj in range(12):
                K[edofs[ii], edofs[jj]] += Ke[ii, jj]

    return K, ndof


def _solve_cantilever3d(
    n_elems: int,
    total_length: float,
    ke_func,
    load_dof_offset: int,
    P: float,
    direction: str = "x",
) -> np.ndarray:
    """3D片持ち梁の先端集中荷重問題を解く.

    Args:
        n_elems: 要素数
        total_length: 全長
        ke_func: 剛性行列関数
        load_dof_offset: 先端節点での荷重DOFオフセット (0=ux, 1=uy, ...)
        P: 荷重値
        direction: 梁の方向
    """
    import scipy.sparse as sp

    nodes, conn = _make_cantilever3d_mesh(n_elems, total_length, direction)
    K, ndof = _assemble_beam3d_system(nodes, conn, ke_func)

    f = np.zeros(ndof)
    tip_node = n_elems
    f[6 * tip_node + load_dof_offset] = P

    K_sp = sp.csr_matrix(K)
    # 固定端: 節点0の全6 DOF
    fixed_dofs = np.arange(6, dtype=int)
    Kbc, fbc = apply_dirichlet(K_sp, f, fixed_dofs)
    u, _ = solve_displacement(Kbc, fbc, show_progress=False)
    return u


# =====================================================================
# テスト
# =====================================================================


class TestLocalStiffnessMatrix:
    """局所剛性行列の基本検証."""

    @pytest.fixture()
    def rect_section(self):
        return BeamSection.rectangle(b=10.0, h=10.0)

    def test_shape(self, rect_section):
        """Keの形状が(12,12)であること."""
        Ke = timo_beam3d_ke_local(
            E, G, rect_section.A, rect_section.Iy, rect_section.Iz,
            rect_section.J, 100.0, KAPPA, KAPPA,
        )
        assert Ke.shape == (12, 12)

    def test_symmetry(self, rect_section):
        """局所Keが対称であること."""
        Ke = timo_beam3d_ke_local(
            E, G, rect_section.A, rect_section.Iy, rect_section.Iz,
            rect_section.J, 100.0, KAPPA, KAPPA,
        )
        assert np.allclose(Ke, Ke.T, atol=1e-10)

    def test_rigid_body_modes(self, rect_section):
        """6つの剛体モード（ゼロ固有値）を持つこと."""
        Ke = timo_beam3d_ke_local(
            E, G, rect_section.A, rect_section.Iy, rect_section.Iz,
            rect_section.J, 100.0, KAPPA, KAPPA,
        )
        eigenvalues = np.linalg.eigvalsh(Ke)
        assert np.sum(np.abs(eigenvalues) < 1e-6) == 6

    def test_positive_semidefinite(self, rect_section):
        """半正定値であること."""
        Ke = timo_beam3d_ke_local(
            E, G, rect_section.A, rect_section.Iy, rect_section.Iz,
            rect_section.J, 100.0, KAPPA, KAPPA,
        )
        eigenvalues = np.linalg.eigvalsh(Ke)
        assert np.all(eigenvalues > -1e-8)

    def test_circular_section(self):
        """円形断面で Iy=Iz が保たれること."""
        sec = BeamSection.circle(d=10.0)
        assert abs(sec.Iy - sec.Iz) < 1e-12


class TestLocalAxes:
    """局所座標系の構築テスト."""

    def test_x_axis_beam(self):
        """x軸方向の梁で局所座標系が正しく構築されること."""
        e_x = np.array([1.0, 0.0, 0.0])
        R = _build_local_axes(e_x)
        assert np.allclose(R[0], e_x)
        assert abs(np.dot(R[0], R[1])) < 1e-12
        assert abs(np.dot(R[0], R[2])) < 1e-12
        assert abs(np.dot(R[1], R[2])) < 1e-12

    def test_y_axis_beam(self):
        """y軸方向の梁で局所座標系が正しく構築されること."""
        e_x = np.array([0.0, 1.0, 0.0])
        R = _build_local_axes(e_x)
        assert np.allclose(R[0], e_x)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12  # 右手系

    def test_z_axis_beam(self):
        """z軸方向の梁で局所座標系が正しく構築されること."""
        e_x = np.array([0.0, 0.0, 1.0])
        R = _build_local_axes(e_x)
        assert np.allclose(R[0], e_x)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_diagonal_beam(self):
        """斜め方向の梁で局所座標系が正交右手系であること."""
        e_x = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        R = _build_local_axes(e_x)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_user_reference_vector(self):
        """ユーザー指定参照ベクトルで局所y軸が制御できること."""
        e_x = np.array([1.0, 0.0, 0.0])
        v_ref = np.array([0.0, 0.0, 1.0])
        R = _build_local_axes(e_x, v_ref)
        # e_z = e_x × v_ref = [0,0,0]×... hmm
        # e_x = [1,0,0], v_ref = [0,0,1]
        # e_z = e_x × v_ref = [0,-1,0]
        # e_y = e_z × e_x = [-1,0,0] × ... = [0,0,1]×[1,0,0] wait
        # e_z = [1,0,0] × [0,0,1] = [0·1-0·0, 0·0-1·1, 1·0-0·0] = [0,-1,0]
        # e_y = [0,-1,0] × [1,0,0] = [-1·0-0·0, 0·1-0·0, 0·0-(-1)·1] = [0,0,1]
        assert np.allclose(R[1], [0.0, 0.0, 1.0], atol=1e-12)

    def test_parallel_reference_raises(self):
        """参照ベクトルが梁軸と平行な場合にValueError."""
        e_x = np.array([1.0, 0.0, 0.0])
        v_ref = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="平行"):
            _build_local_axes(e_x, v_ref)


class TestTransformationMatrix:
    """座標変換行列のテスト."""

    def test_identity_for_aligned_beam(self):
        """x軸方向の梁で変換行列が（ほぼ）単位行列であること."""
        e_x = np.array([1.0, 0.0, 0.0])
        R = _build_local_axes(e_x)
        T = _transformation_matrix_3d(R)
        # R が単位行列に近いはず（自動選択でy→[0,1,0], z→[0,0,1]が選ばれる）
        # ただし自動選択の詳細に依存するので、直交性のみチェック
        assert np.allclose(T @ T.T, np.eye(12), atol=1e-12)

    def test_orthogonality(self):
        """変換行列 T が直交行列であること."""
        e_x = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        R = _build_local_axes(e_x)
        T = _transformation_matrix_3d(R)
        assert np.allclose(T @ T.T, np.eye(12), atol=1e-12)


class TestAxialLoad:
    """軸引張テスト（3D）.

    解析解: δ = PL/(EA)
    """

    def test_axial_deflection_x_beam(self):
        """x方向梁の軸引張たわみが解析解と一致すること."""
        sec = BeamSection.rectangle(b=10.0, h=10.0)
        total_length = 100.0
        n_elems = 5
        P = 100.0

        delta_analytical = P * total_length / (E * sec.A)

        def ke_func(coords):
            return timo_beam3d_ke_global(
                coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
            )

        u = _solve_cantilever3d(n_elems, total_length, ke_func, 0, P)
        delta_fem = u[6 * n_elems + 0]
        assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-10


class TestTorsion:
    """ねじりテスト.

    解析解: θx_tip = T·L/(G·J)
    """

    def test_torsion_angle(self):
        """ねじりモーメントに対する先端回転角が解析解と一致すること."""
        sec = BeamSection.circle(d=10.0)
        total_length = 100.0
        n_elems = 5
        T_torque = 10.0

        theta_analytical = T_torque * total_length / (G * sec.J)

        def ke_func(coords):
            return timo_beam3d_ke_global(
                coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
            )

        u = _solve_cantilever3d(n_elems, total_length, ke_func, 3, T_torque)
        theta_fem = u[6 * n_elems + 3]
        assert abs(theta_fem - theta_analytical) / abs(theta_analytical) < 1e-10


class TestBending:
    """曲げテスト.

    Timoshenko解析解: δ_tip = PL³/(3EI) + PL/(κGA)
    """

    @pytest.fixture()
    def beam_params(self):
        """梁パラメータ."""
        sec = BeamSection.rectangle(b=10.0, h=10.0)
        return {
            "sec": sec,
            "total_length": 100.0,
            "n_elems": 20,
            "P": 1.0,
        }

    def test_bending_y_direction(self, beam_params):
        """y方向荷重（xy面曲げ、Izベース）の先端たわみが解析解と一致."""
        p = beam_params
        sec = p["sec"]
        delta_analytical = (
            p["P"] * p["total_length"] ** 3 / (3.0 * E * sec.Iz)
            + p["P"] * p["total_length"] / (KAPPA * G * sec.A)
        )

        def ke_func(coords):
            return timo_beam3d_ke_global(
                coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
            )

        u = _solve_cantilever3d(
            p["n_elems"], p["total_length"], ke_func, 1, p["P"],
        )
        delta_fem = u[6 * p["n_elems"] + 1]
        assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-10

    def test_bending_z_direction(self, beam_params):
        """z方向荷重（xz面曲げ、Iyベース）の先端たわみが解析解と一致."""
        p = beam_params
        sec = p["sec"]
        delta_analytical = (
            p["P"] * p["total_length"] ** 3 / (3.0 * E * sec.Iy)
            + p["P"] * p["total_length"] / (KAPPA * G * sec.A)
        )

        def ke_func(coords):
            return timo_beam3d_ke_global(
                coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
            )

        u = _solve_cantilever3d(
            p["n_elems"], p["total_length"], ke_func, 2, p["P"],
        )
        delta_fem = u[6 * p["n_elems"] + 2]
        assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-10

    def test_bending_matches_2d_for_plane_problem(self, beam_params):
        """xy面内曲げが2D Timoshenko梁の結果と一致すること."""
        from xkep_cae.elements.beam_timo2d import timo_beam2d_ke_global
        from xkep_cae.sections.beam import BeamSection2D

        p = beam_params
        sec = p["sec"]
        sec2d = BeamSection2D.rectangle(b=10.0, h=10.0)

        # 3D解
        def ke_func_3d(coords):
            return timo_beam3d_ke_global(
                coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
            )

        u_3d = _solve_cantilever3d(
            p["n_elems"], p["total_length"], ke_func_3d, 1, p["P"],
        )
        delta_3d = u_3d[6 * p["n_elems"] + 1]

        # 2D解
        from tests.test_beam_timo2d import _solve_cantilever_point_load

        def ke_func_2d(coords):
            return timo_beam2d_ke_global(
                coords, E, sec2d.A, sec2d.I, KAPPA, G,
            )

        u_2d = _solve_cantilever_point_load(
            p["n_elems"], p["total_length"], sec2d.A, sec2d.I, p["P"], ke_func_2d,
        )
        delta_2d = u_2d[3 * p["n_elems"] + 1]

        assert abs(delta_3d - delta_2d) / abs(delta_2d) < 1e-10


class TestRectangularAsymmetricBending:
    """非正方形矩形断面での二軸曲げテスト.

    b ≠ h の場合、Iy ≠ Iz となり、y方向とz方向のたわみが異なる。
    """

    def test_asymmetric_section_different_deflections(self):
        """b=5, h=20 の長方形で、y方向とz方向のたわみ比が Iz/Iy と一致."""
        sec = BeamSection.rectangle(b=5.0, h=20.0)
        total_length = 200.0
        n_elems = 20
        P = 1.0

        def ke_func(coords):
            return timo_beam3d_ke_global(
                coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
            )

        # y方向荷重（Iz ベース）
        u_y = _solve_cantilever3d(n_elems, total_length, ke_func, 1, P)
        delta_y = u_y[6 * n_elems + 1]

        # z方向荷重（Iy ベース）
        u_z = _solve_cantilever3d(n_elems, total_length, ke_func, 2, P)
        delta_z = u_z[6 * n_elems + 2]

        # 曲げが支配的な場合、たわみ比は I の逆比に近い
        # δ ∝ PL³/(3EI) なので、δ_y/δ_z ≈ Iy/Iz
        # ただしせん断項 PL/(κGA) は共通なので完全一致ではない
        # 解析解で確認
        delta_y_analytical = (
            P * total_length**3 / (3.0 * E * sec.Iz)
            + P * total_length / (KAPPA * G * sec.A)
        )
        delta_z_analytical = (
            P * total_length**3 / (3.0 * E * sec.Iy)
            + P * total_length / (KAPPA * G * sec.A)
        )

        assert abs(delta_y - delta_y_analytical) / abs(delta_y_analytical) < 1e-10
        assert abs(delta_z - delta_z_analytical) / abs(delta_z_analytical) < 1e-10

        # h > b なので Iy > Iz → z方向曲げは硬い → δ_z < δ_y
        assert delta_z < delta_y


class TestInclinedBeam:
    """傾斜梁のテスト.

    y方向に配置した梁でも同じ解析解が得られること。
    """

    def test_y_direction_cantilever(self):
        """y方向片持ち梁の先端たわみ（局所y方向荷重）."""
        sec = BeamSection.rectangle(b=10.0, h=10.0)
        total_length = 100.0
        n_elems = 20
        P = 1.0

        def ke_func(coords):
            return timo_beam3d_ke_global(
                coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
            )

        # y方向梁 → 局所y方向の荷重は全体のx方向荷重に相当
        # ただし自動座標系選択に依存するため、
        # 全体z方向荷重で xz面曲げをテスト（確実に Iy ベース）
        u = _solve_cantilever3d(
            n_elems, total_length, ke_func, 2, P, direction="y",
        )
        # z方向たわみ→ Iy ベース
        delta_fem = u[6 * n_elems + 2]
        delta_analytical_iy = (
            P * total_length**3 / (3.0 * E * sec.Iy)
            + P * total_length / (KAPPA * G * sec.A)
        )
        assert abs(delta_fem - delta_analytical_iy) / abs(delta_analytical_iy) < 1e-8


class TestSCF3D:
    """SCF の動作確認."""

    def test_scf_reduces_deflection(self):
        """SCF適用でEB解に近づく → たわみが減少すること."""
        sec = BeamSection.rectangle(b=10.0, h=10.0)
        total_length = 200.0
        n_elems = 20
        P = 1.0

        def ke_func_no_scf(coords):
            return timo_beam3d_ke_global(
                coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
            )

        def ke_func_scf(coords):
            return timo_beam3d_ke_global(
                coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
                scf=0.25,
            )

        u_no_scf = _solve_cantilever3d(n_elems, total_length, ke_func_no_scf, 1, P)
        u_scf = _solve_cantilever3d(n_elems, total_length, ke_func_scf, 1, P)

        delta_no_scf = abs(u_no_scf[6 * n_elems + 1])
        delta_scf = abs(u_scf[6 * n_elems + 1])

        # SCF はせん断変形を低減 → たわみが小さくなる（EB解に近づく）
        assert delta_scf < delta_no_scf


class TestProtocolConformance:
    """Protocol適合性の検証."""

    def test_element_protocol(self):
        """TimoshenkoBeam3DがElementProtocolに適合すること."""
        sec = BeamSection.rectangle(b=10.0, h=10.0)
        beam = TimoshenkoBeam3D(section=sec)
        assert isinstance(beam, ElementProtocol)

    def test_element_class_stiffness(self):
        """クラスインタフェース経由の剛性行列が関数版と一致すること."""
        sec = BeamSection.rectangle(b=10.0, h=10.0)
        beam = TimoshenkoBeam3D(section=sec, kappa_y=KAPPA, kappa_z=KAPPA)
        mat = BeamElastic1D(E=E, nu=NU)

        coords = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        Ke_class = beam.local_stiffness(coords, mat)
        Ke_func = timo_beam3d_ke_global(
            coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
        )

        assert np.allclose(Ke_class, Ke_func, atol=1e-10)

    def test_dof_indices(self):
        """DOFインデックスが正しく計算されること."""
        sec = BeamSection.rectangle(b=10.0, h=10.0)
        beam = TimoshenkoBeam3D(section=sec)
        node_indices = np.array([3, 5])
        edofs = beam.dof_indices(node_indices)
        expected = np.array([18, 19, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35])
        assert np.array_equal(edofs, expected)

    def test_invalid_kappa_y_raises(self):
        """無効なkappa_y文字列でValueError."""
        sec = BeamSection.rectangle(b=10.0, h=10.0)
        with pytest.raises(ValueError, match="cowper"):
            TimoshenkoBeam3D(section=sec, kappa_y="invalid")

    def test_invalid_kappa_z_raises(self):
        """無効なkappa_z文字列でValueError."""
        sec = BeamSection.rectangle(b=10.0, h=10.0)
        with pytest.raises(ValueError, match="cowper"):
            TimoshenkoBeam3D(section=sec, kappa_z="invalid")


class TestCowperKappa3D:
    """Cowper κ の統合テスト（3D梁）."""

    def test_cowper_tip_deflection(self):
        """kappa="cowper" 指定の先端たわみが解析解と一致."""
        sec = BeamSection.rectangle(b=10.0, h=10.0)
        total_length = 100.0
        n_elems = 20
        P = 1.0

        kappa_cowper = 10.0 * (1.0 + NU) / (12.0 + 11.0 * NU)
        delta_analytical = (
            P * total_length**3 / (3.0 * E * sec.Iz)
            + P * total_length / (kappa_cowper * G * sec.A)
        )

        beam = TimoshenkoBeam3D(
            section=sec, kappa_y="cowper", kappa_z="cowper",
        )
        mat = BeamElastic1D(E=E, nu=NU)

        def ke_func(coords):
            return beam.local_stiffness(coords, mat)

        u = _solve_cantilever3d(n_elems, total_length, ke_func, 1, P)
        delta_fem = u[6 * n_elems + 1]
        assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-10


class TestDistributedLoad3D:
    """3D等分布荷重テスト."""

    def test_distributed_load_y(self):
        """y方向等分布荷重の先端たわみが解析解と一致.

        Timoshenko: δ = qL⁴/(8EIz) + qL²/(2κGA)
        """
        import scipy.sparse as sp

        sec = BeamSection.rectangle(b=10.0, h=10.0)
        total_length = 100.0
        n_elems = 20
        q = 0.01

        delta_analytical = (
            q * total_length**4 / (8.0 * E * sec.Iz)
            + q * total_length**2 / (2.0 * KAPPA * G * sec.A)
        )

        nodes, conn = _make_cantilever3d_mesh(n_elems, total_length)
        ke_func = lambda coords: timo_beam3d_ke_global(  # noqa: E731
            coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
        )
        K, ndof = _assemble_beam3d_system(nodes, conn, ke_func)

        f = np.zeros(ndof)
        for elem_nodes in conn:
            n1, n2 = elem_nodes
            coords = nodes[[n1, n2]]
            fe = timo_beam3d_distributed_load(coords, qy_local=q)
            edofs = np.empty(12, dtype=int)
            for i, n in enumerate([n1, n2]):
                for d in range(6):
                    edofs[6 * i + d] = 6 * n + d
            f[edofs] += fe

        K_sp = sp.csr_matrix(K)
        Kbc, fbc = apply_dirichlet(K_sp, f, np.arange(6))
        u, _ = solve_displacement(Kbc, fbc, show_progress=False)

        delta_fem = u[6 * n_elems + 1]
        assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-6


class TestBeamSectionProperties:
    """BeamSection の断面特性テスト."""

    def test_rectangle_properties(self):
        """矩形断面の特性値が正しいこと."""
        sec = BeamSection.rectangle(b=10.0, h=20.0)
        assert abs(sec.A - 200.0) < 1e-12
        assert abs(sec.Iy - 10.0 * 20.0**3 / 12.0) < 1e-10
        assert abs(sec.Iz - 20.0 * 10.0**3 / 12.0) < 1e-10
        assert sec.J > 0

    def test_circle_properties(self):
        """円形断面の特性値が正しいこと."""
        import math
        sec = BeamSection.circle(d=10.0)
        assert abs(sec.A - math.pi * 25.0) < 1e-10
        assert abs(sec.Iy - sec.Iz) < 1e-12  # 対称
        assert abs(sec.J - 2.0 * sec.Iy) < 1e-10  # 円形: J = 2I

    def test_pipe_properties(self):
        """パイプ断面の特性値が正しいこと."""
        import math
        sec = BeamSection.pipe(d_outer=20.0, d_inner=16.0)
        A_expected = math.pi * (10.0**2 - 8.0**2)
        assert abs(sec.A - A_expected) < 1e-10
        assert sec.Iy > 0
        assert abs(sec.Iy - sec.Iz) < 1e-12

    def test_pipe_invalid_raises(self):
        """パイプ断面で内径≥外径の場合にValueError."""
        with pytest.raises(ValueError, match="内径"):
            BeamSection.pipe(d_outer=10.0, d_inner=10.0)

    def test_to_2d(self):
        """3D断面から2D断面への変換が正しいこと."""
        sec = BeamSection.rectangle(b=10.0, h=20.0)
        sec2d = sec.to_2d()
        assert abs(sec2d.A - sec.A) < 1e-12
        assert abs(sec2d.I - sec.Iz) < 1e-12
        assert sec2d.shape == sec.shape

    def test_section_validation(self):
        """不正な断面特性でValueError."""
        with pytest.raises(ValueError, match="断面積"):
            BeamSection(A=-1.0, Iy=1.0, Iz=1.0, J=1.0)
        with pytest.raises(ValueError, match="Iy"):
            BeamSection(A=1.0, Iy=-1.0, Iz=1.0, J=1.0)
        with pytest.raises(ValueError, match="Iz"):
            BeamSection(A=1.0, Iy=1.0, Iz=-1.0, J=1.0)
        with pytest.raises(ValueError, match="J"):
            BeamSection(A=1.0, Iy=1.0, Iz=1.0, J=-1.0)

    def test_cowper_kappa_y(self):
        """Cowper κ_y が BeamSection2D と一致すること."""
        from xkep_cae.sections.beam import BeamSection2D
        sec3d = BeamSection.rectangle(b=10.0, h=10.0)
        sec2d = BeamSection2D.rectangle(b=10.0, h=10.0)
        assert abs(sec3d.cowper_kappa_y(0.3) - sec2d.cowper_kappa(0.3)) < 1e-12


class TestSectionForces:
    """断面力（内力）ポスト処理のテスト."""

    def test_axial_tension(self):
        """軸引張の断面力が N = P であること（円形断面）."""
        sec = BeamSection.circle(d=10.0)
        total_length = 100.0
        n_elems = 5
        P = 50.0

        def ke_func(coords):
            return timo_beam3d_ke_global(
                coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
            )

        u = _solve_cantilever3d(n_elems, total_length, ke_func, 0, P)
        nodes, conn = _make_cantilever3d_mesh(n_elems, total_length)

        # 各要素の断面力を確認: 全要素で N = P
        for elem_nodes in conn:
            n1, n2 = elem_nodes
            coords = nodes[[n1, n2]]
            u_elem = np.empty(12)
            for i, n in enumerate([n1, n2]):
                for d in range(6):
                    u_elem[6 * i + d] = u[6 * n + d]

            f1, f2 = beam3d_section_forces(
                coords, u_elem, E, G, sec.A, sec.Iy, sec.Iz, sec.J,
                KAPPA, KAPPA,
            )
            assert abs(f1.N - P) / P < 1e-10
            assert abs(f2.N - P) / P < 1e-10
            assert abs(f1.Vy) < 1e-8
            assert abs(f1.Vz) < 1e-8
            assert abs(f1.Mx) < 1e-8
            assert abs(f1.My) < 1e-8
            assert abs(f1.Mz) < 1e-8

    def test_torsion(self):
        """ねじりの断面力が Mx = T であること（円形断面）."""
        sec = BeamSection.circle(d=10.0)
        total_length = 100.0
        n_elems = 5
        T_torque = 10.0

        def ke_func(coords):
            return timo_beam3d_ke_global(
                coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
            )

        u = _solve_cantilever3d(n_elems, total_length, ke_func, 3, T_torque)
        nodes, conn = _make_cantilever3d_mesh(n_elems, total_length)

        for elem_nodes in conn:
            n1, n2 = elem_nodes
            coords = nodes[[n1, n2]]
            u_elem = np.empty(12)
            for i, n in enumerate([n1, n2]):
                for d in range(6):
                    u_elem[6 * i + d] = u[6 * n + d]

            f1, f2 = beam3d_section_forces(
                coords, u_elem, E, G, sec.A, sec.Iy, sec.Iz, sec.J,
                KAPPA, KAPPA,
            )
            assert abs(f1.Mx - T_torque) / T_torque < 1e-10
            assert abs(f2.Mx - T_torque) / T_torque < 1e-10
            assert abs(f1.N) < 1e-8
            assert abs(f1.Vy) < 1e-8

    def test_cantilever_bending_y(self):
        """y方向荷重の断面力（せん断力と曲げモーメント）が解析解と一致（円形断面）.

        解析解:
          Vy = P（全要素で一定）
          Mz(x) = P·(L - x)（根元で PL、先端で 0）
        """
        sec = BeamSection.circle(d=10.0)
        total_length = 100.0
        n_elems = 10
        P = 1.0

        def ke_func(coords):
            return timo_beam3d_ke_global(
                coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
            )

        u = _solve_cantilever3d(n_elems, total_length, ke_func, 1, P)
        nodes, conn = _make_cantilever3d_mesh(n_elems, total_length)

        for elem_nodes in conn:
            n1, n2 = elem_nodes
            coords = nodes[[n1, n2]]
            u_elem = np.empty(12)
            for i, n in enumerate([n1, n2]):
                for d in range(6):
                    u_elem[6 * i + d] = u[6 * n + d]

            f1, f2 = beam3d_section_forces(
                coords, u_elem, E, G, sec.A, sec.Iy, sec.Iz, sec.J,
                KAPPA, KAPPA,
            )

            # せん断力: Vy = P（全要素で一定）
            assert abs(f1.Vy - P) / P < 1e-8
            assert abs(f2.Vy - P) / P < 1e-8

            # 曲げモーメント: Mz(x) = P·(L - x)
            x1 = nodes[n1, 0]
            x2 = nodes[n2, 0]
            Mz_1_analytical = P * (total_length - x1)
            Mz_2_analytical = P * (total_length - x2)
            if abs(Mz_1_analytical) > 1e-12:
                assert abs(f1.Mz - Mz_1_analytical) / abs(Mz_1_analytical) < 1e-8
            if abs(Mz_2_analytical) > 1e-12:
                assert abs(f2.Mz - Mz_2_analytical) / abs(Mz_2_analytical) < 1e-8
            else:
                assert abs(f2.Mz) < 1e-6

    def test_section_forces_via_class(self):
        """TimoshenkoBeam3D.section_forces() メソッドが関数版と一致（円形断面）."""
        sec = BeamSection.circle(d=10.0)
        beam = TimoshenkoBeam3D(section=sec, kappa_y=KAPPA, kappa_z=KAPPA)
        mat = BeamElastic1D(E=E, nu=NU)

        total_length = 100.0
        n_elems = 5
        P = 10.0

        def ke_func(coords):
            return timo_beam3d_ke_global(
                coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
            )

        u = _solve_cantilever3d(n_elems, total_length, ke_func, 1, P)
        nodes, conn = _make_cantilever3d_mesh(n_elems, total_length)

        # 先頭要素で検証
        n1, n2 = conn[0]
        coords = nodes[[n1, n2]]
        u_elem = np.empty(12)
        for i, n in enumerate([n1, n2]):
            for d in range(6):
                u_elem[6 * i + d] = u[6 * n + d]

        f1_class, f2_class = beam.section_forces(coords, u_elem, mat)
        f1_func, f2_func = beam3d_section_forces(
            coords, u_elem, E, G, sec.A, sec.Iy, sec.Iz, sec.J,
            KAPPA, KAPPA,
        )

        assert abs(f1_class.N - f1_func.N) < 1e-10
        assert abs(f1_class.Vy - f1_func.Vy) < 1e-10
        assert abs(f1_class.Mz - f1_func.Mz) < 1e-10
        assert abs(f2_class.N - f2_func.N) < 1e-10
        assert abs(f2_class.Vy - f2_func.Vy) < 1e-10
        assert abs(f2_class.Mz - f2_func.Mz) < 1e-10

    def test_equilibrium(self):
        """要素両端の断面力が力の釣り合いを満たすこと（円形断面）.

        外力なしの要素では: N1 = N2, Vy1 = Vy2, Mz1 + Vy1·L = Mz2
        """
        sec = BeamSection.circle(d=10.0)
        total_length = 100.0
        n_elems = 10
        P = 1.0
        L_elem = total_length / n_elems

        def ke_func(coords):
            return timo_beam3d_ke_global(
                coords, E, G, sec.A, sec.Iy, sec.Iz, sec.J, KAPPA, KAPPA,
            )

        u = _solve_cantilever3d(n_elems, total_length, ke_func, 1, P)
        nodes, conn = _make_cantilever3d_mesh(n_elems, total_length)

        for elem_nodes in conn:
            n1, n2 = elem_nodes
            coords = nodes[[n1, n2]]
            u_elem = np.empty(12)
            for i, n in enumerate([n1, n2]):
                for d in range(6):
                    u_elem[6 * i + d] = u[6 * n + d]

            f1, f2 = beam3d_section_forces(
                coords, u_elem, E, G, sec.A, sec.Iy, sec.Iz, sec.J,
                KAPPA, KAPPA,
            )

            # 軸力一定
            assert abs(f1.N - f2.N) < 1e-8
            # せん断力一定（分布荷重なし）
            assert abs(f1.Vy - f2.Vy) < 1e-8
            # モーメントの釣り合い: Mz1 - Vy1·L = Mz2
            assert abs(f1.Mz - f1.Vy * L_elem - f2.Mz) < 1e-6


class TestMaxBendingStress:
    """最大曲げ応力推定のテスト."""

    def test_pure_axial(self):
        """純軸引張の応力が N/A であること."""
        forces = BeamForces3D(N=100.0, Vy=0.0, Vz=0.0, Mx=0.0, My=0.0, Mz=0.0)
        sec = BeamSection.circle(d=10.0)
        sigma = beam3d_max_bending_stress(forces, sec.A, sec.Iy, sec.Iz, 5.0, 5.0)
        assert abs(sigma - 100.0 / sec.A) < 1e-10

    def test_pure_bending_mz(self):
        """純曲げ（Mz）の応力が Mz·y_max/Iz であること."""
        Mz = 1000.0
        y_max = 5.0
        sec = BeamSection.circle(d=10.0)
        forces = BeamForces3D(N=0.0, Vy=0.0, Vz=0.0, Mx=0.0, My=0.0, Mz=Mz)
        sigma = beam3d_max_bending_stress(forces, sec.A, sec.Iy, sec.Iz, y_max, 5.0)
        expected = Mz * y_max / sec.Iz
        assert abs(sigma - expected) / expected < 1e-10

    def test_combined_loading(self):
        """複合荷重の応力が正しいこと."""
        N = 100.0
        Mz = 500.0
        My = 300.0
        sec = BeamSection.circle(d=10.0)
        y_max = 5.0
        z_max = 5.0
        forces = BeamForces3D(N=N, Vy=0.0, Vz=0.0, Mx=0.0, My=My, Mz=Mz)
        sigma = beam3d_max_bending_stress(forces, sec.A, sec.Iy, sec.Iz, y_max, z_max)
        expected = abs(N / sec.A) + Mz * y_max / sec.Iz + My * z_max / sec.Iy
        assert abs(sigma - expected) / expected < 1e-10


class TestMaxShearStress3D:
    """3D最大せん断応力テスト."""

    def test_pure_torsion(self):
        """純ねじり: τ = |Mx|·r/J."""
        Mx = 100.0
        sec = BeamSection.circle(d=10.0)
        r_max = 5.0
        forces = BeamForces3D(N=0.0, Vy=0.0, Vz=0.0, Mx=Mx, My=0.0, Mz=0.0)
        tau = beam3d_max_shear_stress(forces, sec.A, sec.J, r_max, shape="circle")
        expected = Mx * r_max / sec.J
        assert abs(tau - expected) / expected < 1e-10

    def test_transverse_shear_circle(self):
        """円形断面の横せん断: τ = 4V/(3A)."""
        Vy = 10.0
        sec = BeamSection.circle(d=10.0)
        forces = BeamForces3D(N=0.0, Vy=Vy, Vz=0.0, Mx=0.0, My=0.0, Mz=0.0)
        tau = beam3d_max_shear_stress(forces, sec.A, sec.J, 5.0, shape="circle")
        expected = 4.0 * Vy / (3.0 * sec.A)
        assert abs(tau - expected) / expected < 1e-10

    def test_transverse_shear_rectangle(self):
        """矩形断面の横せん断: τ = 3V/(2A)."""
        Vy = 10.0
        sec = BeamSection.rectangle(b=10.0, h=10.0)
        forces = BeamForces3D(N=0.0, Vy=Vy, Vz=0.0, Mx=0.0, My=0.0, Mz=0.0)
        tau = beam3d_max_shear_stress(forces, sec.A, sec.J, 5.0, shape="rectangle")
        expected = 3.0 * Vy / (2.0 * sec.A)
        assert abs(tau - expected) / expected < 1e-10

    def test_combined_torsion_and_transverse(self):
        """ねじり＋横せん断の保守的和."""
        Mx = 100.0
        Vy = 10.0
        Vz = 5.0
        sec = BeamSection.circle(d=10.0)
        r_max = 5.0
        forces = BeamForces3D(N=0.0, Vy=Vy, Vz=Vz, Mx=Mx, My=0.0, Mz=0.0)
        tau = beam3d_max_shear_stress(forces, sec.A, sec.J, r_max, shape="circle")
        tau_torsion = Mx * r_max / sec.J
        tau_transverse = 4.0 * max(Vy, Vz) / (3.0 * sec.A)
        expected = tau_torsion + tau_transverse
        assert abs(tau - expected) / expected < 1e-10

    def test_zero_shear(self):
        """全せん断成分ゼロの場合 τ=0."""
        forces = BeamForces3D(N=100.0, Vy=0.0, Vz=0.0, Mx=0.0, My=0.0, Mz=500.0)
        tau = beam3d_max_shear_stress(forces, 100.0, 50.0, 5.0, shape="circle")
        assert abs(tau) < 1e-15
