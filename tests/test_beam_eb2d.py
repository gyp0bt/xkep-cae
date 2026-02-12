"""2D Euler-Bernoulli 梁要素のテスト.

解析解との比較:
  - 片持ち梁の先端集中荷重
  - 片持ち梁の等分布荷重
  - 傾斜梁のテスト
  - 剛性行列の対称性・正定値性
  - Protocol適合確認
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.bc import apply_dirichlet
from xkep_cae.core.constitutive import ConstitutiveProtocol
from xkep_cae.core.element import ElementProtocol
from xkep_cae.elements.beam_eb2d import (
    EulerBernoulliBeam2D,
    eb_beam2d_distributed_load,
    eb_beam2d_ke_global,
    eb_beam2d_ke_local,
)
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection2D
from xkep_cae.solver import solve_displacement

# =====================================================================
# テストパラメータ
# =====================================================================
E = 200e3  # MPa
A = 100.0  # mm^2
I_val = 833.333  # mm^4 (10x10 矩形: bh^3/12 = 10*10^3/12 ≈ 833.333)
L_total = 1000.0  # mm
N_ELEMS = 10  # 分割数


def _make_cantilever_mesh(
    n_elems: int, total_length: float, angle_deg: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """片持ち梁のメッシュを生成する.

    Args:
        n_elems: 要素分割数
        total_length: 梁の全長
        angle_deg: x軸からの傾斜角度（度）

    Returns:
        nodes: (n_elems+1, 2) 節点座標
        connectivity: (n_elems, 2) 接続配列
    """
    angle_rad = np.deg2rad(angle_deg)
    n_nodes = n_elems + 1
    s = np.linspace(0, total_length, n_nodes)
    nodes = np.column_stack([s * np.cos(angle_rad), s * np.sin(angle_rad)])
    connectivity = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])
    return nodes, connectivity


def _assemble_beam_system(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    young: float,
    area: float,
    inertia: float,
) -> tuple[np.ndarray, int]:
    """梁構造の全体剛性行列を密行列で組み立てる.

    Returns:
        K: (ndof, ndof) 全体剛性行列
        ndof: 総自由度数
    """
    n_nodes = len(nodes)
    ndof = 3 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)

    for elem_nodes in connectivity:
        n1, n2 = elem_nodes
        coords = nodes[[n1, n2]]
        Ke = eb_beam2d_ke_global(coords, young, area, inertia)

        # DOFマッピング (3 DOF/node)
        edofs = np.array([3 * n1, 3 * n1 + 1, 3 * n1 + 2, 3 * n2, 3 * n2 + 1, 3 * n2 + 2])
        for ii in range(6):
            for jj in range(6):
                K[edofs[ii], edofs[jj]] += Ke[ii, jj]

    return K, ndof


# =====================================================================
# テスト
# =====================================================================


class TestLocalStiffnessMatrix:
    """局所剛性行列の基本検証."""

    def test_symmetry(self):
        """局所Keが対称であること."""
        Ke = eb_beam2d_ke_local(E, A, I_val, 100.0)
        assert np.allclose(Ke, Ke.T, atol=1e-10)

    def test_shape(self):
        """Keの形状が(6,6)であること."""
        Ke = eb_beam2d_ke_local(E, A, I_val, 100.0)
        assert Ke.shape == (6, 6)

    def test_rigid_body_modes(self):
        """剛体モード（3つ）の固有値がゼロであること."""
        Ke = eb_beam2d_ke_local(E, A, I_val, 100.0)
        eigenvalues = np.linalg.eigvalsh(Ke)
        # 3つのゼロ固有値（軸方向並進、横方向並進、回転）
        assert np.sum(np.abs(eigenvalues) < 1e-6) == 3

    def test_positive_semidefinite(self):
        """Keが半正定値であること."""
        Ke = eb_beam2d_ke_local(E, A, I_val, 100.0)
        eigenvalues = np.linalg.eigvalsh(Ke)
        assert np.all(eigenvalues > -1e-8)


class TestGlobalStiffnessMatrix:
    """全体座標系の剛性行列の検証."""

    def test_horizontal_matches_local(self):
        """水平梁の全体Keが局所Keと一致すること."""
        coords = np.array([[0.0, 0.0], [100.0, 0.0]])
        Ke_global = eb_beam2d_ke_global(coords, E, A, I_val)
        Ke_local = eb_beam2d_ke_local(E, A, I_val, 100.0)
        assert np.allclose(Ke_global, Ke_local, atol=1e-8)

    def test_vertical_beam_symmetry(self):
        """垂直梁の全体Keが対称であること."""
        coords = np.array([[0.0, 0.0], [0.0, 100.0]])
        Ke_global = eb_beam2d_ke_global(coords, E, A, I_val)
        assert np.allclose(Ke_global, Ke_global.T, atol=1e-10)

    def test_inclined_beam_symmetry(self):
        """45度傾斜梁の全体Keが対称であること."""
        d = 100.0 / np.sqrt(2.0)
        coords = np.array([[0.0, 0.0], [d, d]])
        Ke_global = eb_beam2d_ke_global(coords, E, A, I_val)
        assert np.allclose(Ke_global, Ke_global.T, atol=1e-10)


class TestCantileverPointLoad:
    """片持ち梁 — 先端集中荷重の解析解比較.

    解析解:
      δ_tip = PL³/(3EI)  (先端たわみ)
      θ_tip = PL²/(2EI)  (先端回転角)
    """

    @pytest.fixture()
    def cantilever_result(self):
        """10要素の片持ち梁を解いて変位を返す."""
        P = 1.0  # 先端荷重
        nodes, conn = _make_cantilever_mesh(N_ELEMS, L_total)
        K, ndof = _assemble_beam_system(nodes, conn, E, A, I_val)

        # 荷重ベクトル: 先端(最終節点)のy方向に P
        f = np.zeros(ndof)
        tip_node = N_ELEMS
        f[3 * tip_node + 1] = P  # uy方向

        # 固定端: 節点0の全DOF (ux, uy, θz)
        import scipy.sparse as sp

        K_sp = sp.csr_matrix(K)
        fixed_dofs = np.array([0, 1, 2], dtype=int)
        Kbc, fbc = apply_dirichlet(K_sp, f, fixed_dofs)
        u, _info = solve_displacement(Kbc, fbc, show_progress=False)

        return u, P

    def test_tip_deflection(self, cantilever_result):
        """先端たわみが解析解と一致すること."""
        u, P = cantilever_result
        delta_analytical = P * L_total**3 / (3.0 * E * I_val)
        delta_fem = u[3 * N_ELEMS + 1]  # 先端のuy
        assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-10

    def test_tip_rotation(self, cantilever_result):
        """先端回転角が解析解と一致すること."""
        u, P = cantilever_result
        theta_analytical = P * L_total**2 / (2.0 * E * I_val)
        theta_fem = u[3 * N_ELEMS + 2]  # 先端のθz
        assert abs(theta_fem - theta_analytical) / abs(theta_analytical) < 1e-10

    def test_no_axial_displacement(self, cantilever_result):
        """横荷重のみなので軸方向変位がほぼゼロであること."""
        u, _P = cantilever_result
        for i in range(N_ELEMS + 1):
            assert abs(u[3 * i]) < 1e-12  # ux ≈ 0


class TestCantileverDistributedLoad:
    """片持ち梁 — 等分布荷重の解析解比較.

    解析解:
      δ_tip = qL⁴/(8EI)  (先端たわみ)
      θ_tip = qL³/(6EI)  (先端回転角)
    """

    @pytest.fixture()
    def cantilever_dist_result(self):
        """等分布荷重の片持ち梁を解く."""
        q = 0.01  # 分布荷重 [force/length]
        nodes, conn = _make_cantilever_mesh(N_ELEMS, L_total)
        K, ndof = _assemble_beam_system(nodes, conn, E, A, I_val)

        # 等価節点力の組み立て
        f = np.zeros(ndof)
        for elem_nodes in conn:
            n1, n2 = elem_nodes
            coords = nodes[[n1, n2]]
            fe = eb_beam2d_distributed_load(coords, q)
            edofs = np.array([3 * n1, 3 * n1 + 1, 3 * n1 + 2, 3 * n2, 3 * n2 + 1, 3 * n2 + 2])
            f[edofs] += fe

        # 固定端: 節点0
        import scipy.sparse as sp

        K_sp = sp.csr_matrix(K)
        fixed_dofs = np.array([0, 1, 2], dtype=int)
        Kbc, fbc = apply_dirichlet(K_sp, f, fixed_dofs)
        u, _info = solve_displacement(Kbc, fbc, show_progress=False)

        return u, q

    def test_tip_deflection(self, cantilever_dist_result):
        """先端たわみが解析解と一致すること."""
        u, q = cantilever_dist_result
        delta_analytical = q * L_total**4 / (8.0 * E * I_val)
        delta_fem = u[3 * N_ELEMS + 1]
        assert abs(delta_fem - delta_analytical) / abs(delta_analytical) < 1e-6

    def test_tip_rotation(self, cantilever_dist_result):
        """先端回転角が解析解と一致すること."""
        u, q = cantilever_dist_result
        theta_analytical = q * L_total**3 / (6.0 * E * I_val)
        theta_fem = u[3 * N_ELEMS + 2]
        assert abs(theta_fem - theta_analytical) / abs(theta_analytical) < 1e-6


class TestInclinedCantilever:
    """傾斜片持ち梁のテスト — 座標変換の検証.

    45度傾斜した片持ち梁の先端たわみが水平梁と同じ値であることを確認。
    """

    def test_inclined_45deg(self):
        """45度傾斜梁の先端たわみが水平梁と同じこと."""
        P = 1.0
        # 水平梁
        nodes_h, conn_h = _make_cantilever_mesh(N_ELEMS, L_total, angle_deg=0.0)
        K_h, ndof_h = _assemble_beam_system(nodes_h, conn_h, E, A, I_val)

        import scipy.sparse as sp

        f_h = np.zeros(ndof_h)
        f_h[3 * N_ELEMS + 1] = P
        Kbc_h, fbc_h = apply_dirichlet(sp.csr_matrix(K_h), f_h, np.array([0, 1, 2]))
        u_h, _ = solve_displacement(Kbc_h, fbc_h, show_progress=False)

        # 45度傾斜梁 — 局所y方向に荷重
        nodes_i, conn_i = _make_cantilever_mesh(N_ELEMS, L_total, angle_deg=45.0)
        K_i, ndof_i = _assemble_beam_system(nodes_i, conn_i, E, A, I_val)

        # 局所y荷重を全体座標に変換: 局所y = (-sin45, cos45) 方向
        angle_rad = np.deg2rad(45.0)
        fx_global = -np.sin(angle_rad) * P
        fy_global = np.cos(angle_rad) * P
        f_i = np.zeros(ndof_i)
        f_i[3 * N_ELEMS] = fx_global
        f_i[3 * N_ELEMS + 1] = fy_global
        Kbc_i, fbc_i = apply_dirichlet(sp.csr_matrix(K_i), f_i, np.array([0, 1, 2]))
        u_i, _ = solve_displacement(Kbc_i, fbc_i, show_progress=False)

        # 先端の局所y方向変位を計算
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        ux_tip = u_i[3 * N_ELEMS]
        uy_tip = u_i[3 * N_ELEMS + 1]
        v_local = -s * ux_tip + c * uy_tip  # 局所y方向変位

        delta_h = u_h[3 * N_ELEMS + 1]  # 水平梁の先端たわみ
        assert abs(v_local - delta_h) / abs(delta_h) < 1e-8


class TestProtocolConformance:
    """Protocol適合性の検証."""

    def test_element_protocol(self):
        """EulerBernoulliBeam2DがElementProtocolに適合すること."""
        sec = BeamSection2D(A=A, I=I_val)
        beam = EulerBernoulliBeam2D(section=sec)
        assert isinstance(beam, ElementProtocol)

    def test_material_protocol(self):
        """BeamElastic1DがConstitutiveProtocolに適合すること."""
        mat = BeamElastic1D(E=E)
        assert isinstance(mat, ConstitutiveProtocol)

    def test_element_class_stiffness(self):
        """クラスインタフェース経由の剛性行列が関数版と一致すること."""
        sec = BeamSection2D(A=A, I=I_val)
        beam = EulerBernoulliBeam2D(section=sec)
        mat = BeamElastic1D(E=E)

        coords = np.array([[0.0, 0.0], [100.0, 0.0]])
        Ke_class = beam.local_stiffness(coords, mat)
        Ke_func = eb_beam2d_ke_global(coords, E, A, I_val)

        assert np.allclose(Ke_class, Ke_func, atol=1e-10)

    def test_dof_indices(self):
        """DOFインデックスが正しく計算されること."""
        sec = BeamSection2D(A=A, I=I_val)
        beam = EulerBernoulliBeam2D(section=sec)
        edofs = beam.dof_indices(np.array([3, 7]))
        expected = np.array([9, 10, 11, 21, 22, 23], dtype=np.int64)
        np.testing.assert_array_equal(edofs, expected)


class TestBeamSection2D:
    """BeamSection2Dの検証."""

    def test_rectangle(self):
        """矩形断面の計算が正しいこと."""
        sec = BeamSection2D.rectangle(b=10.0, h=10.0)
        assert abs(sec.A - 100.0) < 1e-10
        assert abs(sec.I - 833.333333333) < 1e-3

    def test_circle(self):
        """円形断面の計算が正しいこと."""
        import math

        sec = BeamSection2D.circle(d=10.0)
        assert abs(sec.A - math.pi * 25.0) < 1e-10
        assert abs(sec.I - math.pi * 625.0 / 4.0) < 1e-10

    def test_invalid_area(self):
        """A<=0のときValueErrorが発生すること."""
        with pytest.raises(ValueError, match="断面積"):
            BeamSection2D(A=-1.0, I=100.0)

    def test_invalid_inertia(self):
        """I<=0のときValueErrorが発生すること."""
        with pytest.raises(ValueError, match="断面二次モーメント"):
            BeamSection2D(A=100.0, I=0.0)
