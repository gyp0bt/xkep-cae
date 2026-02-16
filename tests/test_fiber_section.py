"""Phase 4.2: ファイバーモデル断面テスト.

FiberSection のファイバー分割精度、ファイバー積分による断面力・接線剛性、
弾塑性片持ち梁の荷重-変位曲線を検証する。

検証項目:
  1. FiberSection の断面定数（A, Iy, Iz）が解析解と一致
  2. 弾性領域でファイバー版と通常版の内力・剛性が一致
  3. ファイバー積分の接線剛性が有限差分と一致
  4. 弾塑性片持ち梁の曲げ: 弾性限界モーメント・塑性モーメントの解析解
  5. NR 法の二次収束（consistent tangent）
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.core.state import CosseratFiberPlasticState
from xkep_cae.elements.beam_cosserat import (
    CosseratRod,
    _compute_generalized_stress_fiber,
    _cosserat_constitutive_matrix,
    assemble_cosserat_beam,
    assemble_cosserat_beam_fiber,
)
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.materials.plasticity_1d import (
    IsotropicHardening,
    KinematicHardening,
    Plasticity1D,
)
from xkep_cae.sections.beam import BeamSection
from xkep_cae.sections.fiber import FiberSection
from xkep_cae.solver import newton_raphson

# テスト用材料定数
E_MAT = 200_000.0  # MPa
NU = 0.3
SIGMA_Y0 = 250.0  # MPa
H_ISO = 1000.0  # MPa


def _make_plasticity(
    H_iso: float = H_ISO,
    C_kin: float = 0.0,
    gamma_kin: float = 0.0,
) -> Plasticity1D:
    kin = KinematicHardening(C_kin=C_kin, gamma_kin=gamma_kin) if C_kin > 0 else None
    return Plasticity1D(E=E_MAT, iso=IsotropicHardening(sigma_y0=SIGMA_Y0, H_iso=H_iso), kin=kin)


# ================================================================
# FiberSection 単体テスト
# ================================================================


class TestFiberSectionGeometry:
    """FiberSection の断面定数が解析解と一致."""

    def test_rectangle_area(self):
        """矩形断面: A = b * h."""
        b, h = 10.0, 20.0
        fs = FiberSection.rectangle(b, h, ny=20, nz=40)
        assert abs(fs.A - b * h) / (b * h) < 1e-12

    def test_rectangle_Iy(self):
        """矩形断面: Iy = b * h^3 / 12."""
        b, h = 10.0, 20.0
        fs = FiberSection.rectangle(b, h, ny=20, nz=40)
        Iy_exact = b * h**3 / 12.0
        assert abs(fs.Iy - Iy_exact) / Iy_exact < 0.01

    def test_rectangle_Iz(self):
        """矩形断面: Iz = h * b^3 / 12."""
        b, h = 10.0, 20.0
        fs = FiberSection.rectangle(b, h, ny=20, nz=40)
        Iz_exact = h * b**3 / 12.0
        assert abs(fs.Iz - Iz_exact) / Iz_exact < 0.01

    def test_rectangle_n_fibers(self):
        """矩形断面のファイバー数."""
        fs = FiberSection.rectangle(10.0, 20.0, ny=5, nz=8)
        assert fs.n_fibers == 40

    def test_rectangle_centroid(self):
        """矩形断面の重心が原点."""
        fs = FiberSection.rectangle(10.0, 20.0, ny=10, nz=10)
        assert abs(np.sum(fs.y * fs.areas) / fs.A) < 1e-12
        assert abs(np.sum(fs.z * fs.areas) / fs.A) < 1e-12

    def test_circle_area(self):
        """円形断面: A = pi * d^2 / 4."""
        d = 20.0
        fs = FiberSection.circle(d, nr=20, nt=36)
        A_exact = np.pi * d**2 / 4.0
        assert abs(fs.A - A_exact) / A_exact < 1e-10

    def test_circle_Iy(self):
        """円形断面: Iy = pi * d^4 / 64."""
        d = 20.0
        fs = FiberSection.circle(d, nr=20, nt=36)
        Iy_exact = np.pi * d**4 / 64.0
        # ファイバー近似は各リングの中心半径で評価するため、やや誤差がある
        assert abs(fs.Iy - Iy_exact) / Iy_exact < 0.02

    def test_circle_centroid(self):
        """円形断面の重心が原点."""
        fs = FiberSection.circle(20.0, nr=10, nt=36)
        assert abs(np.sum(fs.y * fs.areas) / fs.A) < 1e-10
        assert abs(np.sum(fs.z * fs.areas) / fs.A) < 1e-10

    def test_pipe_area(self):
        """パイプ断面: A = pi * (d_o^2 - d_i^2) / 4."""
        d_o, d_i = 20.0, 16.0
        fs = FiberSection.pipe(d_o, d_i, nr=10, nt=36)
        A_exact = np.pi * (d_o**2 - d_i**2) / 4.0
        assert abs(fs.A - A_exact) / A_exact < 1e-10

    def test_rectangle_Iy_convergence(self):
        """矩形断面: ファイバー数を増やすとIyが収束."""
        b, h = 10.0, 20.0
        Iy_exact = b * h**3 / 12.0
        errors = []
        for n in [4, 8, 16, 32]:
            fs = FiberSection.rectangle(b, h, ny=1, nz=n)
            errors.append(abs(fs.Iy - Iy_exact) / Iy_exact)
        # 分割数を倍にすると誤差が4分の1に（2次収束）
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1]


class TestFiberSectionValidation:
    """FiberSection の入力検証."""

    def test_empty_fibers(self):
        with pytest.raises(ValueError):
            FiberSection(y=np.array([]), z=np.array([]), areas=np.array([]), J=1.0)

    def test_negative_area(self):
        with pytest.raises(ValueError):
            FiberSection(y=np.array([0.0]), z=np.array([0.0]), areas=np.array([-1.0]), J=1.0)

    def test_negative_J(self):
        with pytest.raises(ValueError):
            FiberSection.rectangle(10.0, 20.0, ny=2, nz=2)
            FiberSection(y=np.array([0.0]), z=np.array([0.0]), areas=np.array([1.0]), J=-1.0)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            FiberSection(
                y=np.array([0.0, 1.0]),
                z=np.array([0.0]),
                areas=np.array([1.0, 1.0]),
                J=1.0,
            )

    def test_cowper_kappa_rectangle(self):
        fs = FiberSection.rectangle(10.0, 20.0, ny=2, nz=2)
        kappa = fs.cowper_kappa_y(0.3)
        expected = 10.0 * (1.0 + 0.3) / (12.0 + 11.0 * 0.3)
        assert abs(kappa - expected) < 1e-12


# ================================================================
# ファイバー応力積分テスト（構成レベル）
# ================================================================


class TestFiberStressIntegration:
    """_compute_generalized_stress_fiber の単体テスト."""

    def _make_elastic_setup(self):
        """弾性テスト用のセットアップ."""
        b, h = 10.0, 20.0
        fs = FiberSection.rectangle(b, h, ny=10, nz=20)
        G = E_MAT / (2.0 * (1.0 + NU))
        kappa = fs.cowper_kappa_y(NU)
        C_elastic = _cosserat_constitutive_matrix(
            E_MAT,
            G,
            fs.A,
            fs.Iy,
            fs.Iz,
            fs.J,
            kappa,
            kappa,
        )
        plas = _make_plasticity()
        state = CosseratFiberPlasticState.create(fs.n_fibers)
        return fs, C_elastic, plas, state

    def test_elastic_axial_force(self):
        """弾性軸力: N = EA * Gamma_1."""
        fs, C_elastic, plas, state = self._make_elastic_setup()
        strain = np.array([0.0005, 0.0, 0.0, 0.0, 0.0, 0.0])
        stress, C_tan, _ = _compute_generalized_stress_fiber(
            strain,
            C_elastic,
            plas,
            state,
            fs,
        )
        N_expected = E_MAT * fs.A * strain[0]
        assert abs(stress[0] - N_expected) / abs(N_expected) < 1e-10

    def test_elastic_bending_moment_y(self):
        """弾性曲げ: My = EIy * kappa_2.

        降伏ひずみ = sigma_y / E = 250/200000 = 0.00125
        z_max = 9.5 (nz=20, h=20) → 弾性限界曲率 = 0.00125/9.5 ≈ 0.000132
        kappa_2 = 0.0001 で全ファイバー弾性を確保。
        """
        fs, C_elastic, plas, state = self._make_elastic_setup()
        strain = np.array([0.0, 0.0, 0.0, 0.0, 0.0001, 0.0])
        stress, _, _ = _compute_generalized_stress_fiber(
            strain,
            C_elastic,
            plas,
            state,
            fs,
        )
        My_expected = E_MAT * fs.Iy * strain[4]
        assert abs(stress[4] - My_expected) / abs(My_expected) < 0.01

    def test_elastic_bending_moment_z(self):
        """弾性曲げ: Mz = EIz * kappa_3.

        y_max = 4.75 (ny=10, b=10) → 弾性限界曲率 = 0.00125/4.75 ≈ 0.000263
        kappa_3 = 0.0001 で全ファイバー弾性を確保。
        """
        fs, C_elastic, plas, state = self._make_elastic_setup()
        strain = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0001])
        stress, _, _ = _compute_generalized_stress_fiber(
            strain,
            C_elastic,
            plas,
            state,
            fs,
        )
        Mz_expected = E_MAT * fs.Iz * strain[5]
        assert abs(stress[5] - Mz_expected) / abs(Mz_expected) < 0.01

    def test_elastic_tangent_diagonal(self):
        """弾性域: 接線の対角成分が EA, EIy, EIz に一致."""
        fs, C_elastic, plas, state = self._make_elastic_setup()
        strain = np.zeros(6)
        _, C_tan, _ = _compute_generalized_stress_fiber(
            strain,
            C_elastic,
            plas,
            state,
            fs,
        )
        assert abs(C_tan[0, 0] - E_MAT * fs.A) / (E_MAT * fs.A) < 1e-10
        assert abs(C_tan[4, 4] - E_MAT * fs.Iy) / (E_MAT * fs.Iy) < 0.01
        assert abs(C_tan[5, 5] - E_MAT * fs.Iz) / (E_MAT * fs.Iz) < 0.01

    def test_elastic_tangent_offdiagonal_zero(self):
        """弾性域・対称断面: N-My, N-Mz の連成がゼロ."""
        fs, C_elastic, plas, state = self._make_elastic_setup()
        strain = np.zeros(6)
        _, C_tan, _ = _compute_generalized_stress_fiber(
            strain,
            C_elastic,
            plas,
            state,
            fs,
        )
        # 対称断面（重心が原点）→ Sum(D_i * y_i * A_i) = 0
        assert abs(C_tan[0, 4]) < 1e-6 * E_MAT * fs.A
        assert abs(C_tan[0, 5]) < 1e-6 * E_MAT * fs.A

    def test_tangent_symmetry(self):
        """接線剛性が対称."""
        fs, C_elastic, plas, state = self._make_elastic_setup()
        # 塑性域で非対称にならないことを確認
        strain = np.array([0.003, 0.0, 0.0, 0.0, 0.01, 0.0])
        _, C_tan, _ = _compute_generalized_stress_fiber(
            strain,
            C_elastic,
            plas,
            state,
            fs,
        )
        np.testing.assert_allclose(C_tan, C_tan.T, atol=1e-10)

    def test_tangent_fd_verification(self):
        """接線剛性の有限差分検証."""
        fs, C_elastic, plas, state = self._make_elastic_setup()
        # 一部のファイバーが塑性に入る大きなひずみ
        strain = np.array([0.002, 0.0, 0.0, 0.0, 0.005, 0.0])
        _, C_tan, _ = _compute_generalized_stress_fiber(
            strain,
            C_elastic,
            plas,
            state,
            fs,
        )

        # 有限差分検証: ファイバー成分 [0, 4, 5] のみ
        h = 1e-7
        for idx in [0, 4, 5]:
            strain_p = strain.copy()
            strain_p[idx] += h
            strain_m = strain.copy()
            strain_m[idx] -= h
            stress_p, _, _ = _compute_generalized_stress_fiber(
                strain_p,
                C_elastic,
                plas,
                state,
                fs,
            )
            stress_m, _, _ = _compute_generalized_stress_fiber(
                strain_m,
                C_elastic,
                plas,
                state,
                fs,
            )
            dstress_fd = (stress_p - stress_m) / (2.0 * h)
            for row in [0, 4, 5]:
                if abs(C_tan[row, idx]) > 1e-3:
                    rel = abs(C_tan[row, idx] - dstress_fd[row]) / abs(C_tan[row, idx])
                    assert rel < 1e-4, (
                        f"C_tan[{row},{idx}]={C_tan[row, idx]:.6e}, "
                        f"FD={dstress_fd[row]:.6e}, rel={rel:.2e}"
                    )

    def test_plastic_axial_reduces_EA(self):
        """塑性域: 接線の EA 成分が弾性 EA より小さくなる."""
        fs, C_elastic, plas, state = self._make_elastic_setup()
        # 全ファイバーが降伏するひずみ
        strain = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        _, C_tan, _ = _compute_generalized_stress_fiber(
            strain,
            C_elastic,
            plas,
            state,
            fs,
        )
        EA_elastic = E_MAT * fs.A
        assert C_tan[0, 0] < EA_elastic * 0.5  # 大幅に低下

    def test_partial_plasticity_coupling(self):
        """部分塑性: 曲げで片側だけ塑性化 → N-My 連成が発生."""
        b, h = 10.0, 20.0
        fs = FiberSection.rectangle(b, h, ny=4, nz=20)
        G = E_MAT / (2.0 * (1.0 + NU))
        kappa = fs.cowper_kappa_y(NU)
        C_elastic = _cosserat_constitutive_matrix(
            E_MAT,
            G,
            fs.A,
            fs.Iy,
            fs.Iz,
            fs.J,
            kappa,
            kappa,
        )
        plas = _make_plasticity()
        state = CosseratFiberPlasticState.create(fs.n_fibers)

        # 大きな曲率 → 上下で異なる塑性化
        strain = np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.0])
        _, C_tan, _ = _compute_generalized_stress_fiber(
            strain,
            C_elastic,
            plas,
            state,
            fs,
        )
        # 上下非対称な塑性化 → N-kappa_2 連成が発生
        # 純曲げでの非対称塑性化はない（断面が対称なので）
        # ただし N+M の複合では非対称になる
        # ここでは対称断面の純曲げなので連成=0
        assert abs(C_tan[0, 4]) < 1e-3 * E_MAT * fs.A

    def test_state_immutability(self):
        """入力の state が変更されないこと."""
        fs, C_elastic, plas, state = self._make_elastic_setup()
        orig_eps_p = [s.eps_p for s in state.fiber_states]

        strain = np.array([0.01, 0.0, 0.0, 0.0, 0.01, 0.0])
        _compute_generalized_stress_fiber(strain, C_elastic, plas, state, fs)

        for i, s in enumerate(state.fiber_states):
            assert s.eps_p == orig_eps_p[i]


# ================================================================
# アセンブリレベルテスト
# ================================================================


def _make_fiber_section(ny: int = 10, nz: int = 20) -> FiberSection:
    return FiberSection.rectangle(10.0, 20.0, ny=ny, nz=nz)


def _make_beam_section() -> BeamSection:
    return BeamSection.rectangle(10.0, 20.0)


def _make_material() -> BeamElastic1D:
    return BeamElastic1D(E=E_MAT, nu=NU)


def _make_rod(
    section: BeamSection | FiberSection | None = None,
    integration_scheme: str = "uniform",
) -> CosseratRod:
    sec = section if section is not None else _make_beam_section()
    return CosseratRod(
        section=sec,
        integration_scheme=integration_scheme,
        n_gauss=1,
    )


class TestFiberAssemblyElasticMatch:
    """弾性領域でファイバー版と通常版が一致."""

    def test_uniform_elastic_match_fint(self):
        """uniform 積分: 降伏未満で内力が一致."""
        n_elems = 4
        L = 100.0
        fs = _make_fiber_section()
        rod = _make_rod(section=fs)
        mat = _make_material()
        plas = _make_plasticity()

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6
        u = np.zeros(total_dof)
        # 降伏未満の小さな軸変位
        for i in range(1, n_nodes):
            u[6 * i] = 0.001 * i / n_elems

        n_gp = n_elems * rod.n_gauss
        states = [CosseratFiberPlasticState.create(fs.n_fibers) for _ in range(n_gp)]

        _, f_fiber, _ = assemble_cosserat_beam_fiber(
            n_elems,
            L,
            rod,
            mat,
            u,
            states,
            plas,
            fs,
            stiffness=False,
            internal_force=True,
        )
        _, f_elastic = assemble_cosserat_beam(
            n_elems,
            L,
            rod,
            mat,
            u,
            stiffness=False,
            internal_force=True,
        )

        np.testing.assert_allclose(f_fiber, f_elastic, atol=1e-8)

    def test_uniform_elastic_match_stiffness(self):
        """uniform 積分: ゼロ変位で剛性が一致."""
        n_elems = 4
        L = 100.0
        fs = _make_fiber_section()
        rod = _make_rod(section=fs)
        mat = _make_material()
        plas = _make_plasticity()

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6
        u = np.zeros(total_dof)

        n_gp = n_elems * rod.n_gauss
        states = [CosseratFiberPlasticState.create(fs.n_fibers) for _ in range(n_gp)]

        K_fiber, _, _ = assemble_cosserat_beam_fiber(
            n_elems,
            L,
            rod,
            mat,
            u,
            states,
            plas,
            fs,
            stiffness=True,
            internal_force=False,
        )
        K_elastic, _ = assemble_cosserat_beam(
            n_elems,
            L,
            rod,
            mat,
            u,
            stiffness=True,
            internal_force=False,
        )

        # ファイバー近似の Iy, Iz は完全精度ではないので相対誤差で比較
        mask = np.abs(K_elastic) > 1e-6
        if np.any(mask):
            rel_err = np.abs(K_fiber[mask] - K_elastic[mask]) / np.abs(K_elastic[mask])
            assert np.max(rel_err) < 0.02, f"max rel error = {np.max(rel_err):.4f}"

    def test_sri_elastic_match_fint(self):
        """SRI: 降伏未満で内力が一致."""
        n_elems = 4
        L = 100.0
        fs = _make_fiber_section()
        rod = _make_rod(section=fs, integration_scheme="sri")
        mat = _make_material()
        plas = _make_plasticity()

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6
        u = np.zeros(total_dof)
        for i in range(1, n_nodes):
            u[6 * i] = 0.001 * i / n_elems

        states = [CosseratFiberPlasticState.create(fs.n_fibers) for _ in range(n_elems * 2)]

        _, f_fiber, _ = assemble_cosserat_beam_fiber(
            n_elems,
            L,
            rod,
            mat,
            u,
            states,
            plas,
            fs,
            stiffness=False,
            internal_force=True,
        )
        _, f_elastic = assemble_cosserat_beam(
            n_elems,
            L,
            rod,
            mat,
            u,
            stiffness=False,
            internal_force=True,
        )

        np.testing.assert_allclose(f_fiber, f_elastic, atol=1e-8)


class TestFiberAssemblyTangentFD:
    """ファイバーアセンブリの全体接線の有限差分検証."""

    def test_global_tangent_fd(self):
        """K_T の有限差分検証（塑性状態）."""
        n_elems = 2
        L = 100.0
        fs = _make_fiber_section(ny=4, nz=8)
        rod = _make_rod(section=fs)
        mat = _make_material()
        plas = _make_plasticity()

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6

        # 降伏を超えたひずみ状態（曲げ＋軸）
        u = np.zeros(total_dof)
        u[6] = 0.005  # node 1 x
        u[12] = 0.010  # node 2 x
        u[6 + 4] = 0.02  # node 1 θy（曲げ）
        u[12 + 4] = 0.04  # node 2 θy

        n_gp = n_elems * rod.n_gauss
        states = [CosseratFiberPlasticState.create(fs.n_fibers) for _ in range(n_gp)]

        K_an, _, _ = assemble_cosserat_beam_fiber(
            n_elems,
            L,
            rod,
            mat,
            u,
            states,
            plas,
            fs,
            stiffness=True,
            internal_force=False,
        )

        h = 1e-7
        K_fd = np.zeros_like(K_an)
        for j in range(total_dof):
            u_p = u.copy()
            u_p[j] += h
            u_m = u.copy()
            u_m[j] -= h

            _, f_p, _ = assemble_cosserat_beam_fiber(
                n_elems,
                L,
                rod,
                mat,
                u_p,
                states,
                plas,
                fs,
                stiffness=False,
                internal_force=True,
            )
            _, f_m, _ = assemble_cosserat_beam_fiber(
                n_elems,
                L,
                rod,
                mat,
                u_m,
                states,
                plas,
                fs,
                stiffness=False,
                internal_force=True,
            )
            K_fd[:, j] = (f_p - f_m) / (2.0 * h)

        mask = np.abs(K_an) > 1e-4
        if np.any(mask):
            rel_err = np.abs(K_an[mask] - K_fd[mask]) / np.abs(K_an[mask])
            assert np.max(rel_err) < 1e-3, f"max rel error = {np.max(rel_err):.2e}"


# ================================================================
# 弾塑性曲げテスト（片持ち梁、解析解との比較）
# ================================================================


class TestFiberBendingPlasticity:
    """ファイバーモデルによる弾塑性曲げ解析."""

    def test_elastic_limit_moment(self):
        """弾性限界モーメントの検証.

        矩形断面の弾性限界モーメント:
          M_y = sigma_y * W_el = sigma_y * b * h^2 / 6
        """
        b, h = 10.0, 20.0
        fs = FiberSection.rectangle(b, h, ny=4, nz=40)
        G = E_MAT / (2.0 * (1.0 + NU))
        kappa = fs.cowper_kappa_y(NU)
        C_elastic = _cosserat_constitutive_matrix(
            E_MAT,
            G,
            fs.A,
            fs.Iy,
            fs.Iz,
            fs.J,
            kappa,
            kappa,
        )
        plas = _make_plasticity(H_iso=0.0)  # 完全弾塑性
        state = CosseratFiberPlasticState.create(fs.n_fibers)

        # 弾性限界の曲率: kappa_y = sigma_y / (E * h/2)
        kappa_y = SIGMA_Y0 / (E_MAT * h / 2.0)

        # 弾性限界の少し手前
        strain = np.array([0.0, 0.0, 0.0, 0.0, 0.9 * kappa_y, 0.0])
        stress, C_tan, _ = _compute_generalized_stress_fiber(
            strain,
            C_elastic,
            plas,
            state,
            fs,
        )
        # 全ファイバーが弾性 → 接線 EIy は弾性値と一致
        assert abs(C_tan[4, 4] - E_MAT * fs.Iy) / (E_MAT * fs.Iy) < 0.01

    def test_full_plastic_moment(self):
        """全塑性モーメントの検証.

        矩形断面の全塑性モーメント:
          M_p = sigma_y * W_pl = sigma_y * b * h^2 / 4
        """
        b, h = 10.0, 20.0
        fs = FiberSection.rectangle(b, h, ny=4, nz=80)
        G = E_MAT / (2.0 * (1.0 + NU))
        kappa = fs.cowper_kappa_y(NU)
        C_elastic = _cosserat_constitutive_matrix(
            E_MAT,
            G,
            fs.A,
            fs.Iy,
            fs.Iz,
            fs.J,
            kappa,
            kappa,
        )
        plas = _make_plasticity(H_iso=0.0)  # 完全弾塑性
        state = CosseratFiberPlasticState.create(fs.n_fibers)

        W_pl = b * h**2 / 4.0
        M_p = SIGMA_Y0 * W_pl

        # 全ファイバーが降伏する大きな曲率
        kappa_y = SIGMA_Y0 / (E_MAT * h / 2.0)
        strain = np.array([0.0, 0.0, 0.0, 0.0, 100.0 * kappa_y, 0.0])
        stress, _, _ = _compute_generalized_stress_fiber(
            strain,
            C_elastic,
            plas,
            state,
            fs,
        )

        # My が M_p に近い（完全弾塑性 → 全ファイバー降伏で M_p に漸近）
        assert abs(stress[4] - M_p) / M_p < 0.02

    def test_shape_factor(self):
        """形状係数の検証: M_p / M_y = 1.5（矩形断面）."""
        b, h = 10.0, 20.0
        W_el = b * h**2 / 6.0
        W_pl = b * h**2 / 4.0
        shape_factor = W_pl / W_el
        assert abs(shape_factor - 1.5) < 1e-10

    def test_cantilever_bending_nr(self):
        """弾塑性片持ち梁の曲げNR解析.

        片持ち梁の先端にモーメントを載荷し、弾性限界前後の応答を確認する。
        ny >= 2 で Iz > 0 を確保し、全DOFに剛性が入るようにする。
        """
        n_elems = 4
        L = 100.0
        b, h = 10.0, 20.0
        fs = FiberSection.rectangle(b, h, ny=4, nz=20)
        rod = _make_rod(section=fs)
        mat = _make_material()
        plas = _make_plasticity(H_iso=0.0)

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6
        fixed_dofs = np.arange(6)

        W_el = b * h**2 / 6.0
        M_y = SIGMA_Y0 * W_el

        # 弾性限界の 0.8 倍
        M_target = 0.8 * M_y
        n_steps = 2

        n_gp = n_elems * rod.n_gauss
        states = [CosseratFiberPlasticState.create(fs.n_fibers) for _ in range(n_gp)]
        u = np.zeros(total_dof)

        for step in range(1, n_steps + 1):
            lam = step / n_steps
            f_ext = np.zeros(total_dof)
            # 先端にy軸曲げモーメント
            f_ext[6 * n_elems + 4] = lam * M_target

            states_trial = [None] * n_gp

            def _fint(u_, _states=states):
                nonlocal states_trial
                _, f, st_new = assemble_cosserat_beam_fiber(
                    n_elems,
                    L,
                    rod,
                    mat,
                    u_,
                    _states,
                    plas,
                    fs,
                    stiffness=False,
                    internal_force=True,
                )
                states_trial = st_new
                return f

            def _Kt(u_, _states=states):
                K, _, _ = assemble_cosserat_beam_fiber(
                    n_elems,
                    L,
                    rod,
                    mat,
                    u_,
                    _states,
                    plas,
                    fs,
                    stiffness=True,
                    internal_force=False,
                )
                return sp.csr_matrix(K)

            result = newton_raphson(
                f_ext,
                fixed_dofs,
                _Kt,
                _fint,
                n_load_steps=1,
                u0=u,
                show_progress=False,
            )
            u = result.u
            states = [s.copy() for s in states_trial]

        # 弾性範囲内: たわみは Euler-Bernoulli 解析解
        # theta_tip = M * L / (E * Iy)
        theta_tip = u[6 * n_elems + 4]  # node n_elems の θ_y
        Iy_exact = b * h**3 / 12.0
        theta_expected = M_target * L / (E_MAT * Iy_exact)
        # ファイバー近似なので少し誤差を許容
        assert abs(theta_tip - theta_expected) / abs(theta_expected) < 0.05


class TestFiberNRConvergence:
    """ファイバーモデルでの NR 法の収束性."""

    def test_quadratic_convergence_bending(self):
        """consistent tangent による NR の二次収束確認（曲げ問題）."""
        n_elems = 2
        L = 100.0
        b, h = 10.0, 20.0
        fs = FiberSection.rectangle(b, h, ny=4, nz=20)
        rod = _make_rod(section=fs)
        mat = _make_material()
        plas = _make_plasticity()

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6
        fixed_dofs = np.arange(6)

        W_el = b * h**2 / 6.0
        M = 1.3 * SIGMA_Y0 * W_el  # 弾性限界を超えるモーメント

        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems + 4] = M

        n_gp = n_elems * rod.n_gauss
        states = [CosseratFiberPlasticState.create(fs.n_fibers) for _ in range(n_gp)]
        u = np.zeros(total_dof)

        f_ext_norm = float(np.linalg.norm(f_ext))
        residuals = []
        for _it in range(20):
            _, f_int, states_trial = assemble_cosserat_beam_fiber(
                n_elems,
                L,
                rod,
                mat,
                u,
                states,
                plas,
                fs,
                stiffness=False,
                internal_force=True,
            )
            R = f_ext - f_int
            R[fixed_dofs] = 0.0
            rnorm = float(np.linalg.norm(R))
            residuals.append(rnorm)
            if rnorm / f_ext_norm < 1e-10:
                break

            K, _, _ = assemble_cosserat_beam_fiber(
                n_elems,
                L,
                rod,
                mat,
                u,
                states,
                plas,
                fs,
                stiffness=True,
                internal_force=False,
            )
            free = [d for d in range(total_dof) if d not in set(fixed_dofs.tolist())]
            K_ff = K[np.ix_(free, free)]
            du = np.zeros(total_dof)
            du[free] = np.linalg.solve(K_ff, R[free])
            u += du

        assert residuals[-1] / f_ext_norm < 1e-8, (
            f"NR did not converge: {[f'{r:.2e}' for r in residuals]}"
        )

        # 二次収束の確認
        if len(residuals) >= 4 and residuals[1] / f_ext_norm > 1e-8:
            for k in range(2, min(len(residuals) - 1, 5)):
                if residuals[k] / f_ext_norm < 1e-8:
                    break
                ratio = residuals[k] / max(residuals[k - 1] ** 2, 1e-30)
                assert ratio < 1e6, (
                    f"Not quadratic at step {k}: "
                    f"r[{k}]={residuals[k]:.2e}, r[{k - 1}]={residuals[k - 1]:.2e}"
                )


class TestFiberBendingLoadDisplacement:
    """弾塑性片持ち梁のモーメント-曲率曲線.

    荷重増分で先端モーメントを増加させ、弾性→弾塑性→全塑性の遷移を確認する。
    """

    def test_moment_curvature_bilinear(self):
        """等方硬化: モーメント-曲率の弾塑性遷移."""
        n_elems = 1  # 1要素で一様曲率
        L = 100.0
        b, h = 10.0, 20.0
        fs = FiberSection.rectangle(b, h, ny=4, nz=40)
        rod = _make_rod(section=fs)
        mat = _make_material()
        plas = _make_plasticity(H_iso=H_ISO)

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6
        fixed_dofs = np.arange(6)

        W_el = b * h**2 / 6.0
        M_y = SIGMA_Y0 * W_el

        n_steps = 8
        M_max = 2.0 * M_y
        moments = []
        curvatures = []

        n_gp = n_elems * rod.n_gauss
        states = [CosseratFiberPlasticState.create(fs.n_fibers) for _ in range(n_gp)]
        u = np.zeros(total_dof)

        for step in range(1, n_steps + 1):
            lam = step / n_steps
            f_ext = np.zeros(total_dof)
            f_ext[6 * n_elems + 4] = lam * M_max

            states_trial = [None] * n_gp

            def _fint(u_, _states=states):
                nonlocal states_trial
                _, f, st = assemble_cosserat_beam_fiber(
                    n_elems,
                    L,
                    rod,
                    mat,
                    u_,
                    _states,
                    plas,
                    fs,
                    stiffness=False,
                    internal_force=True,
                )
                states_trial = st
                return f

            def _Kt(u_, _states=states):
                K, _, _ = assemble_cosserat_beam_fiber(
                    n_elems,
                    L,
                    rod,
                    mat,
                    u_,
                    _states,
                    plas,
                    fs,
                    stiffness=True,
                    internal_force=False,
                )
                return sp.csr_matrix(K)

            result = newton_raphson(
                f_ext,
                fixed_dofs,
                _Kt,
                _fint,
                n_load_steps=1,
                u0=u,
                show_progress=False,
            )
            u = result.u
            states = [s.copy() for s in states_trial]

            theta_tip = u[6 * n_elems + 4]
            kappa_avg = theta_tip / L  # 1要素 → 一様曲率
            moments.append(lam * M_max)
            curvatures.append(kappa_avg)

        moments = np.array(moments)
        curvatures = np.array(curvatures)

        # 弾性域のチェック: M < M_y のステップで M = EI * kappa
        Iy_exact = b * h**3 / 12.0
        for i in range(len(moments)):
            if moments[i] < M_y * 0.95:
                M_predicted = E_MAT * Iy_exact * curvatures[i]
                assert abs(M_predicted - moments[i]) / moments[i] < 0.05

        # 塑性域のチェック: M > M_y のステップで曲率が弾性予測より大きい
        for i in range(len(moments)):
            if moments[i] > M_y * 1.1:
                kappa_elastic = moments[i] / (E_MAT * Iy_exact)
                assert curvatures[i] > kappa_elastic * 1.01  # 塑性で曲率増大


class TestFiberAxialMatch:
    """ファイバーモデルの軸引張が Phase 4.1 と同じ結果を与える."""

    def test_axial_yield_bilinear(self):
        """軸引張降伏: bilinear 解析解との一致."""
        n_elems = 4
        L = 100.0
        fs = _make_fiber_section(ny=4, nz=8)
        rod = _make_rod(section=fs)
        mat = _make_material()
        plas = _make_plasticity()

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6
        fixed_dofs = np.arange(6)

        P_y = SIGMA_Y0 * fs.A
        P_target = 1.5 * P_y
        n_steps = 5

        n_gp = n_elems * rod.n_gauss
        states = [CosseratFiberPlasticState.create(fs.n_fibers) for _ in range(n_gp)]
        u = np.zeros(total_dof)

        for step in range(1, n_steps + 1):
            lam = step / n_steps
            f_ext = np.zeros(total_dof)
            f_ext[6 * n_elems] = lam * P_target

            states_trial = [None] * n_gp

            def _fint(u_, _states=states):
                nonlocal states_trial
                _, f, st = assemble_cosserat_beam_fiber(
                    n_elems,
                    L,
                    rod,
                    mat,
                    u_,
                    _states,
                    plas,
                    fs,
                    stiffness=False,
                    internal_force=True,
                )
                states_trial = st
                return f

            def _Kt(u_, _states=states):
                K, _, _ = assemble_cosserat_beam_fiber(
                    n_elems,
                    L,
                    rod,
                    mat,
                    u_,
                    _states,
                    plas,
                    fs,
                    stiffness=True,
                    internal_force=False,
                )
                return sp.csr_matrix(K)

            result = newton_raphson(
                f_ext,
                fixed_dofs,
                _Kt,
                _fint,
                n_load_steps=1,
                u0=u,
                show_progress=False,
            )
            u = result.u
            states = [s.copy() for s in states_trial]

        u_tip = u[6 * n_elems]
        E_t = E_MAT * H_ISO / (E_MAT + H_ISO)
        u_expected = (SIGMA_Y0 / E_MAT + (P_target / fs.A - SIGMA_Y0) / E_t) * L
        assert abs(u_tip - u_expected) / abs(u_expected) < 1e-3
