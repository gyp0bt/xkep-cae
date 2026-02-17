"""3D弾塑性構成則（von Mises）と平面ひずみ弾塑性アセンブリのテスト.

テスト方針:
  構成則単体:
    1. 降伏未満で弾性（応力、tangent、状態不変）
    2. 単軸引張の降伏・硬化（平面ひずみの解析解比較）
    3. consistent tangent の有限差分検証（最重要）
    4. 除荷で弾性勾配に戻る
    5. 純せん断の降伏
    6. 完全弾塑性 (H_iso=0)
    7. 状態不変性（入力 state が変更されない）
    8. 移動硬化（バウシンガー効果）
    9. 二軸引張の降伏
  要素・構造レベル:
    10. 弾性一致（降伏未満で plastic版 = 弾性版の剛性）
    11. 単軸引張パッチテスト（Q4, Q4_EAS, TRI3）
    12. 全体接線の有限差分検証
    13. NR 二次収束
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.assembly import assemble_global_stiffness
from xkep_cae.assembly_plasticity import assemble_plane_strain_plastic
from xkep_cae.core.state import PlasticState3D
from xkep_cae.elements.quad4 import Quad4PlaneStrain
from xkep_cae.elements.quad4_eas_bbar import Quad4EASPlaneStrain
from xkep_cae.elements.tri3 import Tri3PlaneStrain
from xkep_cae.materials.elastic import PlaneStrainElastic
from xkep_cae.materials.plasticity_3d import (
    IsotropicHardening3D,
    KinematicHardening3D,
    PlaneStrainPlasticity,
)
from xkep_cae.solver import newton_raphson

# ===== テスト用パラメータ =====
E_MAT = 200_000.0  # MPa
NU = 0.3
SIGMA_Y0 = 250.0  # MPa
H_ISO = 1000.0  # 等方硬化係数


def _make_plasticity(
    H_iso: float = H_ISO,
    C_kin: float = 0.0,
    gamma_kin: float = 0.0,
) -> PlaneStrainPlasticity:
    """テスト用 PlaneStrainPlasticity を生成."""
    iso = IsotropicHardening3D(sigma_y0=SIGMA_Y0, H_iso=H_iso)
    kin = KinematicHardening3D(C_kin=C_kin, gamma_kin=gamma_kin)
    return PlaneStrainPlasticity(E=E_MAT, nu=NU, iso=iso, kin=kin)


def _make_plasticity_voce(
    Q_inf: float = 200.0,
    b_voce: float = 10.0,
) -> PlaneStrainPlasticity:
    """Voce硬化のテスト用."""
    iso = IsotropicHardening3D(sigma_y0=SIGMA_Y0, Q_inf=Q_inf, b_voce=b_voce)
    return PlaneStrainPlasticity(E=E_MAT, nu=NU, iso=iso)


# ================================================================
# 構成則単体テスト
# ================================================================


class TestPlasticity3DElastic:
    """降伏未満の弾性応答."""

    def test_elastic_stress_uniaxial(self):
        """降伏未満の単軸引張で弾性応答."""
        plas = _make_plasticity()
        state = PlasticState3D()
        # 平面ひずみ単軸: εxx のみ付与
        eps_y_uniaxial = SIGMA_Y0 / E_MAT
        eps = np.array([eps_y_uniaxial * 0.3, 0.0, 0.0])
        result = plas.return_mapping(eps, state)
        # σxx = (λ+2μ)εxx, σyy = λεxx
        lam = plas.lam
        mu = plas.mu
        expected_xx = (lam + 2 * mu) * eps[0]
        expected_yy = lam * eps[0]
        assert abs(result.stress[0] - expected_xx) < 1e-10 * abs(expected_xx)
        assert abs(result.stress[1] - expected_yy) < 1e-10 * abs(expected_xx)
        assert abs(result.stress[2]) < 1e-10

    def test_elastic_tangent(self):
        """降伏未満で tangent = D_e."""
        plas = _make_plasticity()
        state = PlasticState3D()
        eps = np.array([1e-5, 0.0, 0.0])
        result = plas.return_mapping(eps, state)
        np.testing.assert_allclose(result.tangent, plas.D_e, atol=1e-8)

    def test_elastic_state_unchanged(self):
        """降伏未満で塑性状態が変化しない."""
        plas = _make_plasticity()
        state = PlasticState3D()
        eps = np.array([1e-5, 0.0, 0.0])
        result = plas.return_mapping(eps, state)
        np.testing.assert_allclose(result.state_new.eps_p, np.zeros(3), atol=1e-15)
        assert result.state_new.alpha == 0.0


class TestPlasticity3DUniaxial:
    """単軸引張（平面ひずみ）の降伏後応答."""

    def _uniaxial_yield_strain(self, plas: PlaneStrainPlasticity) -> float:
        """平面ひずみ単軸引張の降伏ひずみを求める.

        平面ひずみ条件 εyy=εzz=0 での単軸引張:
          σxx = (λ+2μ)εxx
          σyy = σzz = λ εxx
        von Mises: σ_vm = |σxx - σyy| = 2μ εxx
        降伏: 2μ εxx = σ_y0
        → εxx_y = σ_y0 / (2μ)
        """
        return SIGMA_Y0 / (2.0 * plas.mu)

    def test_yield_detection(self):
        """降伏点を超えると塑性ひずみが発生."""
        plas = _make_plasticity()
        state = PlasticState3D()
        eps_y = self._uniaxial_yield_strain(plas)
        eps = np.array([1.5 * eps_y, 0.0, 0.0])
        result = plas.return_mapping(eps, state)
        assert result.state_new.alpha > 0.0
        assert np.linalg.norm(result.state_new.eps_p) > 0.0

    def test_consistent_tangent_fd_uniaxial(self):
        """consistent tangent の有限差分検証（単軸）."""
        plas = _make_plasticity()
        state = PlasticState3D()
        eps_y = self._uniaxial_yield_strain(plas)
        eps = np.array([2.0 * eps_y, 0.0, 0.0])

        result = plas.return_mapping(eps, state)
        D_an = result.tangent

        h = 1e-7
        D_fd = np.zeros((3, 3), dtype=float)
        for j in range(3):
            eps_p = eps.copy()
            eps_p[j] += h
            eps_m = eps.copy()
            eps_m[j] -= h
            r_p = plas.return_mapping(eps_p, state)
            r_m = plas.return_mapping(eps_m, state)
            D_fd[:, j] = (r_p.stress - r_m.stress) / (2.0 * h)

        np.testing.assert_allclose(D_an, D_fd, rtol=1e-4, atol=1e-2)

    def test_consistent_tangent_fd_biaxial(self):
        """consistent tangent の有限差分検証（二軸）."""
        plas = _make_plasticity()
        state = PlasticState3D()
        eps_y = self._uniaxial_yield_strain(plas)
        eps = np.array([1.5 * eps_y, 0.8 * eps_y, 0.3 * eps_y])

        result = plas.return_mapping(eps, state)
        D_an = result.tangent

        h = 1e-7
        D_fd = np.zeros((3, 3), dtype=float)
        for j in range(3):
            eps_p = eps.copy()
            eps_p[j] += h
            eps_m = eps.copy()
            eps_m[j] -= h
            r_p = plas.return_mapping(eps_p, state)
            r_m = plas.return_mapping(eps_m, state)
            D_fd[:, j] = (r_p.stress - r_m.stress) / (2.0 * h)

        np.testing.assert_allclose(D_an, D_fd, rtol=1e-4, atol=1e-2)

    def test_consistent_tangent_fd_elastic(self):
        """弾性域での consistent tangent 有限差分検証."""
        plas = _make_plasticity()
        state = PlasticState3D()
        eps = np.array([1e-5, 5e-6, 2e-6])

        result = plas.return_mapping(eps, state)
        D_an = result.tangent

        h = 1e-8
        D_fd = np.zeros((3, 3), dtype=float)
        for j in range(3):
            eps_p = eps.copy()
            eps_p[j] += h
            eps_m = eps.copy()
            eps_m[j] -= h
            r_p = plas.return_mapping(eps_p, state)
            r_m = plas.return_mapping(eps_m, state)
            D_fd[:, j] = (r_p.stress - r_m.stress) / (2.0 * h)

        np.testing.assert_allclose(D_an, D_fd, rtol=1e-5)


class TestPlasticity3DUnloading:
    """除荷テスト."""

    def test_unloading_elastic_slope(self):
        """除荷時に弾性勾配に戻る."""
        plas = _make_plasticity()
        state = PlasticState3D()
        eps_y = SIGMA_Y0 / (2.0 * plas.mu)

        # Step 1: 降伏超え
        eps1 = np.array([2.0 * eps_y, 0.0, 0.0])
        r1 = plas.return_mapping(eps1, state)

        # Step 2: 除荷
        eps2 = np.array([1.5 * eps_y, 0.0, 0.0])
        r2 = plas.return_mapping(eps2, r1.state_new)

        # 除荷中は弾性tangent
        np.testing.assert_allclose(r2.tangent, plas.D_e, atol=1e-6)


class TestPlasticity3DShear:
    """純せん断テスト."""

    def test_pure_shear_yield(self):
        """純せん断の降伏応力 = σ_y0 / √3."""
        plas = _make_plasticity()
        state = PlasticState3D()
        # 純せん断: σxy = τ, 他 = 0
        # von Mises: √3 |τ| = σ_y0  → τ_y = σ_y0/√3
        tau_y = SIGMA_Y0 / np.sqrt(3.0)
        gamma_y = tau_y / plas.mu

        # 降伏未満
        eps_below = np.array([0.0, 0.0, 0.9 * gamma_y])
        r_below = plas.return_mapping(eps_below, state)
        assert r_below.state_new.alpha == 0.0

        # 降伏超え
        eps_above = np.array([0.0, 0.0, 1.5 * gamma_y])
        r_above = plas.return_mapping(eps_above, state)
        assert r_above.state_new.alpha > 0.0

    def test_pure_shear_consistent_tangent_fd(self):
        """純せん断の consistent tangent 有限差分検証."""
        plas = _make_plasticity()
        state = PlasticState3D()
        tau_y = SIGMA_Y0 / np.sqrt(3.0)
        gamma_y = tau_y / plas.mu
        eps = np.array([0.0, 0.0, 2.0 * gamma_y])

        result = plas.return_mapping(eps, state)
        D_an = result.tangent

        h = 1e-7
        D_fd = np.zeros((3, 3), dtype=float)
        for j in range(3):
            eps_p = eps.copy()
            eps_p[j] += h
            eps_m = eps.copy()
            eps_m[j] -= h
            r_p = plas.return_mapping(eps_p, state)
            r_m = plas.return_mapping(eps_m, state)
            D_fd[:, j] = (r_p.stress - r_m.stress) / (2.0 * h)

        np.testing.assert_allclose(D_an, D_fd, rtol=1e-4, atol=1e-2)


class TestPlasticity3DPerfectlyPlastic:
    """完全弾塑性 (H_iso=0)."""

    def test_perfectly_plastic_stress(self):
        """完全弾塑性: von Mises 応力が σ_y0 に飽和."""
        plas = _make_plasticity(H_iso=0.0)
        state = PlasticState3D()
        eps_y = SIGMA_Y0 / (2.0 * plas.mu)

        for factor in [1.5, 2.0, 5.0]:
            eps = np.array([factor * eps_y, 0.0, 0.0])
            result = plas.return_mapping(eps, state)
            assert result.state_new.alpha > 0.0


class TestPlasticity3DStateImmutability:
    """入力状態の不変性テスト."""

    def test_input_state_not_modified(self):
        """return_mapping が入力の state を変更しないこと."""
        plas = _make_plasticity()
        state = PlasticState3D(
            eps_p=np.array([0.001, -0.0005, 0.0002]),
            alpha=0.001,
            beta=np.array([10.0, -5.0, 2.0]),
        )
        eps_p_orig = state.eps_p.copy()
        alpha_orig = state.alpha
        beta_orig = state.beta.copy()

        eps_y = SIGMA_Y0 / (2.0 * plas.mu)
        eps = np.array([5.0 * eps_y, 0.0, 0.0])
        _result = plas.return_mapping(eps, state)

        np.testing.assert_array_equal(state.eps_p, eps_p_orig)
        assert state.alpha == alpha_orig
        np.testing.assert_array_equal(state.beta, beta_orig)


class TestPlasticity3DKinematic:
    """移動硬化テスト."""

    def test_bauschinger_effect(self):
        """バウシンガー効果: 引張→圧縮で降伏応力低下."""
        plas = _make_plasticity(H_iso=0.0, C_kin=5000.0)
        state = PlasticState3D()
        eps_y = SIGMA_Y0 / (2.0 * plas.mu)

        # Step 1: 引張降伏
        eps1 = np.array([3.0 * eps_y, 0.0, 0.0])
        r1 = plas.return_mapping(eps1, state)
        assert r1.state_new.alpha > 0.0

        # 背応力が発生
        assert np.linalg.norm(r1.state_new.beta) > 0.0

        # Step 2: 圧縮で逆降伏
        eps2 = np.array([-3.0 * eps_y, 0.0, 0.0])
        r2 = plas.return_mapping(eps2, r1.state_new)
        assert r2.state_new.alpha > r1.state_new.alpha

    def test_kinematic_consistent_tangent_fd(self):
        """移動硬化の consistent tangent 有限差分検証."""
        plas = _make_plasticity(H_iso=H_ISO, C_kin=5000.0, gamma_kin=50.0)
        state = PlasticState3D()
        eps_y = SIGMA_Y0 / (2.0 * plas.mu)
        eps = np.array([3.0 * eps_y, 0.5 * eps_y, 0.2 * eps_y])

        result = plas.return_mapping(eps, state)
        D_an = result.tangent

        h = 1e-7
        D_fd = np.zeros((3, 3), dtype=float)
        for j in range(3):
            eps_p = eps.copy()
            eps_p[j] += h
            eps_m = eps.copy()
            eps_m[j] -= h
            r_p = plas.return_mapping(eps_p, state)
            r_m = plas.return_mapping(eps_m, state)
            D_fd[:, j] = (r_p.stress - r_m.stress) / (2.0 * h)

        np.testing.assert_allclose(D_an, D_fd, rtol=1e-3, atol=1.0)


class TestPlasticity3DBiaxial:
    """二軸応力テスト."""

    def test_biaxial_tension_yield(self):
        """等二軸引張: σxx=σyy=σ → von Mises = |σzz - σxx| 等."""
        plas = _make_plasticity()
        state = PlasticState3D()
        # 等二軸引張（平面ひずみ）: εxx = εyy = ε
        # σxx = σyy = (λ+2μ)ε + λε = (2λ+2μ)ε
        # σzz = λε + λε = 2λε
        # σ_vm = |σxx - σzz| = |(2λ+2μ)ε - 2λε| = 2μ ε
        # → 降伏ひずみは単軸と同じ: ε_y = σ_y0/(2μ)
        eps_y = SIGMA_Y0 / (2.0 * plas.mu)
        eps = np.array([2.0 * eps_y, 2.0 * eps_y, 0.0])
        result = plas.return_mapping(eps, state)
        assert result.state_new.alpha > 0.0


class TestPlasticity3DValidation:
    """入力検証テスト."""

    def test_negative_E_raises(self):
        with pytest.raises(ValueError):
            PlaneStrainPlasticity(E=-1.0, nu=0.3, iso=IsotropicHardening3D(sigma_y0=SIGMA_Y0))

    def test_invalid_nu_raises(self):
        with pytest.raises(ValueError):
            PlaneStrainPlasticity(E=E_MAT, nu=0.5, iso=IsotropicHardening3D(sigma_y0=SIGMA_Y0))

    def test_negative_sigma_y0_raises(self):
        with pytest.raises(ValueError):
            PlaneStrainPlasticity(E=E_MAT, nu=0.3, iso=IsotropicHardening3D(sigma_y0=-10.0))


# ================================================================
# 要素・構造レベルテスト
# ================================================================


def _make_unit_square_mesh(nx: int, ny: int):
    """単位正方形の Q4 メッシュを生成.

    Returns:
        nodes: (N,2) 節点座標
        conn: (Ne, 4) 接続配列
    """
    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([float(i) / nx, float(j) / ny])
    nodes = np.array(nodes)

    conn = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            conn.append([n0, n1, n2, n3])
    conn = np.array(conn, dtype=int)

    return nodes, conn


def _make_unit_square_tri3_mesh(nx: int, ny: int):
    """単位正方形の TRI3 メッシュを生成（Q4を対角分割）.

    Returns:
        nodes: (N,2) 節点座標
        conn: (Ne, 3) 接続配列
    """
    nodes, conn_q4 = _make_unit_square_mesh(nx, ny)
    conn_tri = []
    for q in conn_q4:
        conn_tri.append([q[0], q[1], q[2]])
        conn_tri.append([q[0], q[2], q[3]])
    return nodes, np.array(conn_tri, dtype=int)


class TestPlasticAssemblyElasticMatch:
    """降伏未満で弾塑性版と弾性版が一致."""

    def test_q4_elastic_match(self):
        """Q4: 降伏未満で弾塑性剛性と弾性剛性が一致."""
        nodes, conn = _make_unit_square_mesh(2, 2)
        ndof = 2 * len(nodes)
        u = np.zeros(ndof)

        plas = _make_plasticity()
        elastic = PlaneStrainElastic(E_MAT, NU)
        states = [PlasticState3D() for _ in range(len(conn) * 4)]

        K_p, _, _ = assemble_plane_strain_plastic(
            nodes,
            conn,
            u,
            plas,
            states,
            element_type="q4",
            stiffness=True,
            internal_force=False,
        )

        K_e = assemble_global_stiffness(
            nodes,
            [(Quad4PlaneStrain(), conn)],
            elastic,
            show_progress=False,
        )

        np.testing.assert_allclose(K_p.toarray(), K_e.toarray(), atol=1e-8)

    def test_q4_eas_elastic_match(self):
        """Q4_EAS: 降伏未満で弾塑性剛性と弾性剛性が一致."""
        nodes, conn = _make_unit_square_mesh(2, 2)
        ndof = 2 * len(nodes)
        u = np.zeros(ndof)

        plas = _make_plasticity()
        elastic = PlaneStrainElastic(E_MAT, NU)
        states = [PlasticState3D() for _ in range(len(conn) * 4)]

        K_p, _, _ = assemble_plane_strain_plastic(
            nodes,
            conn,
            u,
            plas,
            states,
            element_type="q4_eas",
            stiffness=True,
            internal_force=False,
        )

        K_e = assemble_global_stiffness(
            nodes,
            [(Quad4EASPlaneStrain(), conn)],
            elastic,
            show_progress=False,
        )

        np.testing.assert_allclose(K_p.toarray(), K_e.toarray(), atol=1e-8)

    def test_tri3_elastic_match(self):
        """TRI3: 降伏未満で弾塑性剛性と弾性剛性が一致."""
        nodes, conn = _make_unit_square_tri3_mesh(2, 2)
        ndof = 2 * len(nodes)
        u = np.zeros(ndof)

        plas = _make_plasticity()
        elastic = PlaneStrainElastic(E_MAT, NU)
        states = [PlasticState3D() for _ in range(len(conn) * 1)]

        K_p, _, _ = assemble_plane_strain_plastic(
            nodes,
            conn,
            u,
            plas,
            states,
            element_type="tri3",
            stiffness=True,
            internal_force=False,
        )

        K_e = assemble_global_stiffness(
            nodes,
            [(Tri3PlaneStrain(), conn)],
            elastic,
            show_progress=False,
        )

        np.testing.assert_allclose(K_p.toarray(), K_e.toarray(), atol=1e-8)


class TestPlasticPatchTest:
    """パッチテスト: 一様ひずみ場で要素性能を検証."""

    @pytest.mark.parametrize("elem_type", ["q4", "q4_eas", "tri3"])
    def test_uniaxial_tension_patch(self, elem_type):
        """一様引張パッチテスト: 全ガウス点が同一応力."""
        # 4x1 メッシュ（引張方向 x）
        if elem_type == "tri3":
            nodes, conn = _make_unit_square_tri3_mesh(4, 1)
        else:
            nodes, conn = _make_unit_square_mesh(4, 1)

        # スケール
        Lx, Ly = 10.0, 1.0
        nodes[:, 0] *= Lx
        nodes[:, 1] *= Ly
        thickness = 1.0

        ndof = 2 * len(nodes)
        plas = _make_plasticity()

        n_gauss = 4 if elem_type != "tri3" else 1
        n_gp_total = len(conn) * n_gauss
        states = [PlasticState3D() for _ in range(n_gp_total)]

        # 降伏超えの一様 εxx を付与
        eps_y = SIGMA_Y0 / (2.0 * plas.mu)
        eps_target = 2.0 * eps_y

        u = np.zeros(ndof)
        for i in range(len(nodes)):
            u[2 * i] = eps_target * nodes[i, 0]  # ux = εxx * x

        _, f_int, states_new = assemble_plane_strain_plastic(
            nodes,
            conn,
            u,
            plas,
            states,
            element_type=elem_type,
            thickness=thickness,
            stiffness=False,
            internal_force=True,
        )

        # 全ガウス点で同一の塑性ひずみ
        alphas = [s.alpha for s in states_new]
        for a in alphas:
            assert abs(a - alphas[0]) < 1e-10, f"alpha mismatch: {alphas}"


class TestPlasticGlobalTangentFD:
    """全体接線の有限差分検証."""

    @pytest.mark.parametrize("elem_type", ["q4", "q4_eas", "tri3"])
    def test_global_tangent_fd(self, elem_type):
        """K_T の有限差分検証（弾塑性状態）."""
        if elem_type == "tri3":
            nodes, conn = _make_unit_square_tri3_mesh(2, 2)
        else:
            nodes, conn = _make_unit_square_mesh(2, 2)

        ndof = 2 * len(nodes)
        plas = _make_plasticity()
        n_gauss = 4 if elem_type != "tri3" else 1

        # 降伏を超えた変位状態
        eps_y = SIGMA_Y0 / (2.0 * plas.mu)
        u = np.zeros(ndof)
        for i in range(len(nodes)):
            u[2 * i] = 2.0 * eps_y * nodes[i, 0]

        states = [PlasticState3D() for _ in range(len(conn) * n_gauss)]

        # 解析的 tangent
        K_an, _, _ = assemble_plane_strain_plastic(
            nodes,
            conn,
            u,
            plas,
            states,
            element_type=elem_type,
            stiffness=True,
            internal_force=False,
        )
        K_an = K_an.toarray()

        # 有限差分 tangent
        h = 1e-7
        K_fd = np.zeros_like(K_an)
        for j in range(ndof):
            u_p = u.copy()
            u_p[j] += h
            u_m = u.copy()
            u_m[j] -= h

            _, f_p, _ = assemble_plane_strain_plastic(
                nodes,
                conn,
                u_p,
                plas,
                states,
                element_type=elem_type,
                stiffness=False,
                internal_force=True,
            )
            _, f_m, _ = assemble_plane_strain_plastic(
                nodes,
                conn,
                u_m,
                plas,
                states,
                element_type=elem_type,
                stiffness=False,
                internal_force=True,
            )
            K_fd[:, j] = (f_p - f_m) / (2.0 * h)

        # 非ゼロ成分のみ比較
        mask = np.abs(K_an) > 1e-6
        if np.any(mask):
            rel_err = np.abs(K_an[mask] - K_fd[mask]) / np.abs(K_an[mask])
            assert np.max(rel_err) < 1e-3, f"max rel error = {np.max(rel_err):.2e} for {elem_type}"


class TestPlasticNRConvergence:
    """Newton-Raphson の二次収束性."""

    @pytest.mark.parametrize("elem_type", ["q4", "q4_eas"])
    def test_quadratic_convergence(self, elem_type):
        """consistent tangent による NR の二次収束確認."""
        nodes, conn = _make_unit_square_mesh(4, 2)
        Lx, Ly = 10.0, 2.0
        nodes[:, 0] *= Lx
        nodes[:, 1] *= Ly
        thickness = 1.0

        ndof = 2 * len(nodes)
        plas = _make_plasticity()
        n_gauss = 4
        n_gp_total = len(conn) * n_gauss

        # 境界条件: 左端固定、右端に引張荷重
        fixed_dofs = []
        right_nodes = []
        for i in range(len(nodes)):
            if abs(nodes[i, 0]) < 1e-10:
                fixed_dofs.extend([2 * i, 2 * i + 1])
            if abs(nodes[i, 0] - Lx) < 1e-10:
                right_nodes.append(i)
        fixed_dofs = np.array(fixed_dofs, dtype=int)

        # 降伏超え荷重
        P_total = 1.5 * SIGMA_Y0 * Ly * thickness
        f_ext = np.zeros(ndof)
        for n in right_nodes:
            f_ext[2 * n] = P_total / len(right_nodes)

        states = [PlasticState3D() for _ in range(n_gp_total)]
        states_trial = None

        def _fint(u_):
            nonlocal states_trial
            _, f, st = assemble_plane_strain_plastic(
                nodes,
                conn,
                u_,
                plas,
                states,
                element_type=elem_type,
                thickness=thickness,
                stiffness=False,
                internal_force=True,
            )
            states_trial = st
            return f

        def _Kt(u_):
            K, _, _ = assemble_plane_strain_plastic(
                nodes,
                conn,
                u_,
                plas,
                states,
                element_type=elem_type,
                thickness=thickness,
                stiffness=True,
                internal_force=False,
            )
            return K

        result = newton_raphson(
            f_ext,
            fixed_dofs,
            _Kt,
            _fint,
            n_load_steps=3,
            max_iter=30,
            show_progress=False,
        )

        # 収束確認
        assert result.converged, f"NR did not converge for {elem_type}"
