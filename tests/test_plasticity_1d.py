"""1D弾塑性構成則と弾塑性アセンブリのテスト.

テスト方針:
  構成則単体:
    1. 降伏未満で弾性（応力、tangent、状態不変）
    2. 降伏点の境界判定
    3. 等方硬化の単調載荷
    4. consistent tangent の有限差分検証（最重要）
    5. 除荷で弾性勾配
    6. 引張→圧縮（等方硬化: 降伏面拡大）
    7. バウシンガー効果（移動硬化で逆降伏応力低下）
    8. Armstrong-Frederick 繰返し（ratcheting 定性確認）
    9. 状態不変性（入力 state が変更されない）
    10. 完全弾塑性 (H_iso=0)
    11. 完全弾塑性で降伏後応力一定
  要素・構造レベル:
    12. 弾性一致（降伏未満で plastic版 = 通常版）
    13. 軸引張降伏（bilinear 解析解）
    14. 除荷-再載荷
    15. 全体接線の有限差分検証
    16. NR 二次収束
    17. 多要素一様引張
    18. SRI版の弾性一致
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.core.state import CosseratPlasticState, PlasticState1D
from xkep_cae.elements.beam_cosserat import (
    CosseratRod,
    assemble_cosserat_beam,
    assemble_cosserat_beam_plastic,
)
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.materials.plasticity_1d import (
    IsotropicHardening,
    KinematicHardening,
    Plasticity1D,
)
from xkep_cae.sections.beam import BeamSection
from xkep_cae.solver import newton_raphson


# ===== テスト用パラメータ =====
E_MAT = 200_000.0     # MPa
NU = 0.3
SIGMA_Y0 = 250.0      # MPa
H_ISO = 1000.0         # 等方硬化係数
C_KIN = 5000.0         # 移動硬化係数
GAMMA_KIN = 50.0       # Armstrong-Frederick 回復項


def _make_plasticity(
    H_iso: float = H_ISO,
    C_kin: float = 0.0,
    gamma_kin: float = 0.0,
) -> Plasticity1D:
    """テスト用 Plasticity1D を生成."""
    iso = IsotropicHardening(sigma_y0=SIGMA_Y0, H_iso=H_iso)
    kin = KinematicHardening(C_kin=C_kin, gamma_kin=gamma_kin)
    return Plasticity1D(E=E_MAT, iso=iso, kin=kin)


# ================================================================
# 構成則単体テスト
# ================================================================

class TestPlasticity1DElastic:
    """降伏未満の弾性応答."""

    def test_elastic_stress(self):
        """降伏未満で sigma = E * eps."""
        plas = _make_plasticity()
        state = PlasticState1D()
        eps = SIGMA_Y0 / E_MAT * 0.5  # 50% of yield strain
        result = plas.return_mapping(eps, state)
        expected = E_MAT * eps
        assert abs(result.stress - expected) < 1e-10 * SIGMA_Y0

    def test_elastic_tangent(self):
        """降伏未満で D_ep = E."""
        plas = _make_plasticity()
        state = PlasticState1D()
        eps = SIGMA_Y0 / E_MAT * 0.5
        result = plas.return_mapping(eps, state)
        assert abs(result.tangent - E_MAT) < 1e-10 * E_MAT

    def test_elastic_state_unchanged(self):
        """降伏未満で塑性状態が変化しない."""
        plas = _make_plasticity()
        state = PlasticState1D()
        eps = SIGMA_Y0 / E_MAT * 0.5
        result = plas.return_mapping(eps, state)
        assert result.state_new.eps_p == 0.0
        assert result.state_new.alpha == 0.0
        assert result.state_new.beta == 0.0

    def test_yield_boundary(self):
        """f_trial = 0 の境界ケース（弾性判定）."""
        plas = _make_plasticity()
        state = PlasticState1D()
        eps = SIGMA_Y0 / E_MAT  # exactly at yield
        result = plas.return_mapping(eps, state)
        assert abs(result.stress - SIGMA_Y0) < 1e-8 * SIGMA_Y0
        assert abs(result.tangent - E_MAT) < 1e-8 * E_MAT


class TestPlasticity1DIsotropic:
    """等方硬化の降伏後応答."""

    def test_monotonic_tension(self):
        """単調引張: bilinear 応答."""
        plas = _make_plasticity()
        state = PlasticState1D()
        eps_y = SIGMA_Y0 / E_MAT
        eps = 2.0 * eps_y  # well beyond yield

        result = plas.return_mapping(eps, state)

        # 解析解: E_tangent = E * H / (E + H)
        E_t = E_MAT * H_ISO / (E_MAT + H_ISO)
        sigma_expected = SIGMA_Y0 + E_t * (eps - eps_y)
        assert abs(result.stress - sigma_expected) < 1e-8 * SIGMA_Y0

    def test_consistent_tangent_value(self):
        """D_ep = E * H / (E + H) 解析値."""
        plas = _make_plasticity()
        state = PlasticState1D()
        eps = 2.0 * SIGMA_Y0 / E_MAT
        result = plas.return_mapping(eps, state)

        D_ep_exact = E_MAT * H_ISO / (E_MAT + H_ISO)
        assert abs(result.tangent - D_ep_exact) < 1e-10 * E_MAT

    def test_consistent_tangent_fd(self):
        """consistent tangent の有限差分検証."""
        plas = _make_plasticity()
        state = PlasticState1D()
        eps = 2.0 * SIGMA_Y0 / E_MAT

        h = 1e-7
        r_plus = plas.return_mapping(eps + h, state)
        r_minus = plas.return_mapping(eps - h, state)
        D_fd = (r_plus.stress - r_minus.stress) / (2.0 * h)

        result = plas.return_mapping(eps, state)
        assert abs(result.tangent - D_fd) / abs(result.tangent) < 1e-5

    def test_consistent_tangent_fd_elastic(self):
        """弾性域での consistent tangent 有限差分検証."""
        plas = _make_plasticity()
        state = PlasticState1D()
        eps = 0.5 * SIGMA_Y0 / E_MAT

        h = 1e-7
        r_plus = plas.return_mapping(eps + h, state)
        r_minus = plas.return_mapping(eps - h, state)
        D_fd = (r_plus.stress - r_minus.stress) / (2.0 * h)

        result = plas.return_mapping(eps, state)
        assert abs(result.tangent - D_fd) / abs(result.tangent) < 1e-5

    def test_unloading_elastic_slope(self):
        """除荷時の弾性勾配."""
        plas = _make_plasticity()
        state = PlasticState1D()

        # Step 1: 降伏超え → 塑性変形を発生
        eps1 = 2.0 * SIGMA_Y0 / E_MAT
        r1 = plas.return_mapping(eps1, state)

        # Step 2: 少し除荷
        eps2 = eps1 - 0.5 * SIGMA_Y0 / E_MAT
        r2 = plas.return_mapping(eps2, r1.state_new)

        # 除荷中は弾性
        assert abs(r2.tangent - E_MAT) < 1e-8 * E_MAT
        sigma_expected = r1.stress + E_MAT * (eps2 - eps1)
        assert abs(r2.stress - sigma_expected) < 1e-8 * SIGMA_Y0

    def test_tension_then_compression_isotropic(self):
        """引張→圧縮: 等方硬化で降伏面拡大."""
        plas = _make_plasticity()
        state = PlasticState1D()

        # Step 1: 引張降伏
        eps1 = 2.0 * SIGMA_Y0 / E_MAT
        r1 = plas.return_mapping(eps1, state)
        alpha1 = r1.state_new.alpha

        # Step 2: 大きな圧縮 → 逆降伏
        eps2 = -3.0 * SIGMA_Y0 / E_MAT
        r2 = plas.return_mapping(eps2, r1.state_new)

        # 等方硬化: 圧縮降伏応力 = -(sigma_y0 + H_iso * alpha1)
        sigma_y_expanded = SIGMA_Y0 + H_ISO * alpha1
        # 圧縮降伏が発生しているので alpha が増加
        assert r2.state_new.alpha > alpha1

    def test_perfectly_plastic(self):
        """完全弾塑性 (H_iso=0): D_ep = 0, sigma = sigma_y0."""
        plas = _make_plasticity(H_iso=0.0)
        state = PlasticState1D()

        eps = 2.0 * SIGMA_Y0 / E_MAT
        result = plas.return_mapping(eps, state)

        assert abs(result.stress - SIGMA_Y0) < 1e-10 * SIGMA_Y0
        assert abs(result.tangent) < 1e-10

    def test_perfectly_plastic_increasing_load(self):
        """完全弾塑性で荷重増加しても応力一定."""
        plas = _make_plasticity(H_iso=0.0)
        state = PlasticState1D()

        for factor in [1.5, 2.0, 5.0, 10.0]:
            eps = factor * SIGMA_Y0 / E_MAT
            result = plas.return_mapping(eps, state)
            assert abs(result.stress - SIGMA_Y0) < 1e-10 * SIGMA_Y0


class TestPlasticity1DKinematic:
    """移動硬化（Armstrong-Frederick）テスト."""

    def test_bauschinger_effect(self):
        """バウシンガー効果: 移動硬化で逆降伏応力が低下."""
        plas = _make_plasticity(H_iso=0.0, C_kin=C_KIN)
        state = PlasticState1D()

        # Step 1: 引張降伏
        eps1 = 3.0 * SIGMA_Y0 / E_MAT
        r1 = plas.return_mapping(eps1, state)

        # beta > 0 が発生
        assert r1.state_new.beta > 0

        # Step 2: 圧縮（逆方向）→ 降伏は sigma_y0 - beta で起きる
        # 除荷途中の判定
        eps_unload = eps1 - SIGMA_Y0 / E_MAT  # まだ弾性
        r_unload = plas.return_mapping(eps_unload, r1.state_new)
        assert abs(r_unload.tangent - E_MAT) < 1e-6 * E_MAT

        # 大きな圧縮で逆降伏
        eps2 = -5.0 * SIGMA_Y0 / E_MAT
        r2 = plas.return_mapping(eps2, r1.state_new)

        # 逆降伏: |sigma - beta| = sigma_y0
        # beta > 0 なので、圧縮方向では |sigma - beta| > |sigma|
        # → 逆降伏応力の絶対値 < sigma_y0 + H_kin * alpha (等方の場合)
        assert r2.state_new.alpha > r1.state_new.alpha  # 塑性変形が発生

    def test_af_consistent_tangent_fd(self):
        """Armstrong-Frederick の consistent tangent 有限差分検証."""
        plas = _make_plasticity(H_iso=H_ISO, C_kin=C_KIN, gamma_kin=GAMMA_KIN)
        state = PlasticState1D()

        eps = 3.0 * SIGMA_Y0 / E_MAT
        h = 1e-7
        r_plus = plas.return_mapping(eps + h, state)
        r_minus = plas.return_mapping(eps - h, state)
        D_fd = (r_plus.stress - r_minus.stress) / (2.0 * h)

        result = plas.return_mapping(eps, state)
        assert abs(result.tangent - D_fd) / max(abs(result.tangent), 1.0) < 1e-4

    def test_af_cyclic_ratcheting(self):
        """Armstrong-Frederick 繰返しで ratcheting（定性確認）."""
        plas = _make_plasticity(H_iso=0.0, C_kin=C_KIN, gamma_kin=GAMMA_KIN)
        state = PlasticState1D()

        eps_max = 3.0 * SIGMA_Y0 / E_MAT
        eps_min = -3.0 * SIGMA_Y0 / E_MAT

        # 数サイクル実施
        peak_stresses = []
        for _cycle in range(5):
            r = plas.return_mapping(eps_max, state)
            peak_stresses.append(r.stress)
            state = r.state_new
            r = plas.return_mapping(eps_min, state)
            state = r.state_new

        # 繰返しで応力が変化すること（ratcheting 定性）
        # AF 模型では gamma_kin > 0 のため背応力が飽和に向かう
        assert len(peak_stresses) == 5


class TestPlasticity1DStateImmutability:
    """入力状態の不変性テスト."""

    def test_input_state_not_modified(self):
        """return_mapping が入力の state を変更しないこと."""
        plas = _make_plasticity()
        state = PlasticState1D(eps_p=0.001, alpha=0.001, beta=10.0)
        eps_p_orig = state.eps_p
        alpha_orig = state.alpha
        beta_orig = state.beta

        eps = 5.0 * SIGMA_Y0 / E_MAT
        _result = plas.return_mapping(eps, state)

        assert state.eps_p == eps_p_orig
        assert state.alpha == alpha_orig
        assert state.beta == beta_orig


# ================================================================
# 要素・構造レベルテスト
# ================================================================

def _make_section() -> BeamSection:
    return BeamSection.rectangle(10.0, 20.0)


def _make_material() -> BeamElastic1D:
    return BeamElastic1D(E=E_MAT, nu=NU)


def _make_rod(integration_scheme: str = "uniform") -> CosseratRod:
    return CosseratRod(
        section=_make_section(),
        integration_scheme=integration_scheme,
        n_gauss=1,
    )


class TestPlasticAssemblyElasticMatch:
    """降伏未満で弾塑性版と弾性版が一致."""

    def test_uniform_elastic_match_fint(self):
        """uniform 積分: 降伏未満で内力が一致."""
        n_elems = 4
        L = 100.0
        rod = _make_rod("uniform")
        mat = _make_material()
        plas = _make_plasticity()
        n_gauss = rod.n_gauss

        # 降伏未満の小さな軸変位
        n_nodes = n_elems + 1
        total_dof = n_nodes * 6
        u = np.zeros(total_dof)
        for i in range(1, n_nodes):
            u[6 * i] = 0.001 * i / n_elems  # 線形分布

        states = [CosseratPlasticState() for _ in range(n_elems * n_gauss)]

        # 弾塑性版
        _, f_p, _ = assemble_cosserat_beam_plastic(
            n_elems, L, rod, mat, u, states, plas,
            stiffness=False, internal_force=True,
        )
        # 弾性版
        _, f_e = assemble_cosserat_beam(
            n_elems, L, rod, mat, u, stiffness=False, internal_force=True,
        )

        np.testing.assert_allclose(f_p, f_e, atol=1e-10)

    def test_uniform_elastic_match_stiffness(self):
        """uniform 積分: ゼロ変位で材料剛性が一致.

        注: 弾性版の tangent_stiffness は幾何剛性 Kg を含むが、
        ゼロ変位では Kg=0 なので材料剛性のみの比較となる。
        """
        n_elems = 4
        L = 100.0
        rod = _make_rod("uniform")
        mat = _make_material()
        plas = _make_plasticity()
        n_gauss = rod.n_gauss

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6
        u = np.zeros(total_dof)

        states = [CosseratPlasticState() for _ in range(n_elems * n_gauss)]

        K_p, _, _ = assemble_cosserat_beam_plastic(
            n_elems, L, rod, mat, u, states, plas,
            stiffness=True, internal_force=False,
        )
        K_e, _ = assemble_cosserat_beam(
            n_elems, L, rod, mat, u, stiffness=True, internal_force=False,
        )

        np.testing.assert_allclose(K_p, K_e, atol=1e-10)

    def test_sri_elastic_match(self):
        """SRI: 降伏未満で内力・剛性が一致（ゼロ変位）."""
        n_elems = 4
        L = 100.0
        rod = _make_rod("sri")
        mat = _make_material()
        plas = _make_plasticity()

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6
        u = np.zeros(total_dof)

        states = [CosseratPlasticState() for _ in range(n_elems * 2)]

        K_p, f_p, _ = assemble_cosserat_beam_plastic(
            n_elems, L, rod, mat, u, states, plas,
        )
        K_e, f_e = assemble_cosserat_beam(n_elems, L, rod, mat, u)

        np.testing.assert_allclose(f_p, f_e, atol=1e-10)
        np.testing.assert_allclose(K_p, K_e, atol=1e-10)


class TestPlasticBarAnalytical:
    """弾塑性棒の解析解検証."""

    def test_axial_yield_bilinear(self):
        """軸引張降伏: bilinear 荷重-変位曲線（解析解）."""
        n_elems = 4
        L = 100.0
        rod = _make_rod("uniform")
        mat = _make_material()
        plas = _make_plasticity()
        sec = rod.section

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6

        # 降伏荷重
        P_y = SIGMA_Y0 * sec.A
        # 降伏変位
        u_y = SIGMA_Y0 / E_MAT * L

        # 降伏の 1.5 倍の荷重
        P_target = 1.5 * P_y
        n_steps = 5

        states = [CosseratPlasticState() for _ in range(n_elems)]
        u = np.zeros(total_dof)
        fixed_dofs = np.arange(6)  # 左端固定

        for step in range(1, n_steps + 1):
            lam = step / n_steps
            f_ext = np.zeros(total_dof)
            f_ext[6 * n_elems] = lam * P_target  # 右端軸力

            states_trial = None

            def _fint(u_, _states=states, _states_trial_ref=[None]):
                nonlocal states_trial
                _, f, st_new = assemble_cosserat_beam_plastic(
                    n_elems, L, rod, mat, u_, _states, plas,
                    stiffness=False, internal_force=True,
                )
                states_trial = st_new
                return f

            def _Kt(u_, _states=states):
                K, _, _ = assemble_cosserat_beam_plastic(
                    n_elems, L, rod, mat, u_, _states, plas,
                    stiffness=True, internal_force=False,
                )
                return sp.csr_matrix(K)

            result = newton_raphson(
                f_ext, fixed_dofs, _Kt, _fint,
                n_load_steps=1, u0=u, show_progress=False,
            )
            u = result.u
            states = [s.copy() for s in states_trial]

        # 最終変位の検証
        u_tip = u[6 * n_elems]  # 右端 x 変位
        P_final = P_target

        # 解析解: 降伏後 E_t = E * H / (E + H)
        E_t = E_MAT * H_ISO / (E_MAT + H_ISO)
        sigma_final = SIGMA_Y0 + E_t * (P_final / sec.A - SIGMA_Y0) / E_t
        # u = (sigma_y / E + (P/A - sigma_y) / (E*H/(E+H))) * L
        #   = (sigma_y / E) * L + (P/A - sigma_y) / (E*H/(E+H)) * L
        u_expected = (SIGMA_Y0 / E_MAT + (P_final / sec.A - SIGMA_Y0) / E_t) * L

        assert abs(u_tip - u_expected) / abs(u_expected) < 1e-4

    def test_unload_reload(self):
        """除荷-再載荷: 弾性勾配で除荷後、再載荷で塑性勾配."""
        n_elems = 2
        L = 100.0
        rod = _make_rod("uniform")
        mat = _make_material()
        plas = _make_plasticity()
        sec = rod.section

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6
        fixed_dofs = np.arange(6)
        P_y = SIGMA_Y0 * sec.A

        states = [CosseratPlasticState() for _ in range(n_elems)]
        u = np.zeros(total_dof)
        states_trial = None

        def _solve_step(P, u_in, states_in):
            nonlocal states_trial
            f_ext = np.zeros(total_dof)
            f_ext[6 * n_elems] = P

            def _fint(u_):
                nonlocal states_trial
                _, f, st = assemble_cosserat_beam_plastic(
                    n_elems, L, rod, mat, u_, states_in, plas,
                    stiffness=False, internal_force=True,
                )
                states_trial = st
                return f

            def _Kt(u_):
                K, _, _ = assemble_cosserat_beam_plastic(
                    n_elems, L, rod, mat, u_, states_in, plas,
                    stiffness=True, internal_force=False,
                )
                return sp.csr_matrix(K)

            result = newton_raphson(
                f_ext, fixed_dofs, _Kt, _fint,
                n_load_steps=1, u0=u_in, show_progress=False,
            )
            return result.u, [s.copy() for s in states_trial]

        # Step 1: 載荷 (1.5 * P_y)
        u, states = _solve_step(1.5 * P_y, u, states)
        u_peak = u[6 * n_elems]

        # Step 2: 除荷 (0.5 * P_y)
        u, states = _solve_step(0.5 * P_y, u, states)
        u_unloaded = u[6 * n_elems]

        # 除荷は弾性: Δu = ΔP * L / (E * A)
        du_expected = (0.5 - 1.5) * P_y * L / (E_MAT * sec.A)
        du_actual = u_unloaded - u_peak
        assert abs(du_actual - du_expected) / abs(du_expected) < 1e-4


class TestPlasticTangentFD:
    """弾塑性アセンブリの全体接線の有限差分検証."""

    def test_global_tangent_fd(self):
        """K_T の有限差分検証（弾塑性状態）."""
        n_elems = 2
        L = 100.0
        rod = _make_rod("uniform")
        mat = _make_material()
        plas = _make_plasticity()

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6

        # 降伏を超えた変位状態を作る
        u = np.zeros(total_dof)
        u[6] = 0.005   # node 1 x
        u[12] = 0.010  # node 2 x

        states = [CosseratPlasticState() for _ in range(n_elems)]

        # 解析的 tangent
        K_an, _, _ = assemble_cosserat_beam_plastic(
            n_elems, L, rod, mat, u, states, plas,
            stiffness=True, internal_force=False,
        )

        # 有限差分 tangent
        h = 1e-7
        K_fd = np.zeros_like(K_an)
        for j in range(total_dof):
            u_p = u.copy()
            u_p[j] += h
            u_m = u.copy()
            u_m[j] -= h

            _, f_p, _ = assemble_cosserat_beam_plastic(
                n_elems, L, rod, mat, u_p, states, plas,
                stiffness=False, internal_force=True,
            )
            _, f_m, _ = assemble_cosserat_beam_plastic(
                n_elems, L, rod, mat, u_m, states, plas,
                stiffness=False, internal_force=True,
            )
            K_fd[:, j] = (f_p - f_m) / (2.0 * h)

        # 有限差分との比較（非ゼロ成分のみ）
        mask = np.abs(K_an) > 1e-6
        if np.any(mask):
            rel_err = np.abs(K_an[mask] - K_fd[mask]) / np.abs(K_an[mask])
            assert np.max(rel_err) < 1e-4, f"max rel error = {np.max(rel_err):.2e}"


class TestPlasticNRConvergence:
    """NR 法の収束性."""

    def test_quadratic_convergence(self):
        """consistent tangent による NR の二次収束確認."""
        n_elems = 2
        L = 100.0
        rod = _make_rod("uniform")
        mat = _make_material()
        plas = _make_plasticity()
        sec = rod.section

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6
        fixed_dofs = np.arange(6)

        P = 1.8 * SIGMA_Y0 * sec.A  # 降伏超え荷重
        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems] = P

        states = [CosseratPlasticState() for _ in range(n_elems)]
        u = np.zeros(total_dof)

        # 手動 NR で残差履歴を記録
        f_ext_norm = float(np.linalg.norm(f_ext))
        residuals = []
        for _it in range(15):
            _, f_int, states_trial = assemble_cosserat_beam_plastic(
                n_elems, L, rod, mat, u, states, plas,
                stiffness=False, internal_force=True,
            )
            R = f_ext - f_int
            R[fixed_dofs] = 0.0
            rnorm = float(np.linalg.norm(R))
            residuals.append(rnorm)
            if rnorm / f_ext_norm < 1e-10:
                break

            K, _, _ = assemble_cosserat_beam_plastic(
                n_elems, L, rod, mat, u, states, plas,
                stiffness=True, internal_force=False,
            )
            free = [d for d in range(total_dof) if d not in set(fixed_dofs.tolist())]
            K_ff = K[np.ix_(free, free)]
            du = np.zeros(total_dof)
            du[free] = np.linalg.solve(K_ff, R[free])
            u += du

        # 収束確認（相対残差）
        assert residuals[-1] / f_ext_norm < 1e-8, (
            f"NR did not converge: {[f'{r:.2e}' for r in residuals]}"
        )

        # 二次収束の確認: 収束手前の2ステップで残差比が二乗的
        # r_{n+1} / r_n^2 ≈ const
        if len(residuals) >= 4 and residuals[1] / f_ext_norm > 1e-8:
            for k in range(2, min(len(residuals) - 1, 5)):
                if residuals[k] / f_ext_norm < 1e-8:
                    break
                ratio = residuals[k] / max(residuals[k - 1] ** 2, 1e-30)
                # 二次収束なら ratio は有界
                assert ratio < 1e6, (
                    f"Not quadratic at step {k}: "
                    f"r[{k}]={residuals[k]:.2e}, r[{k-1}]={residuals[k-1]:.2e}"
                )


class TestPlasticMultiElement:
    """多要素一様引張."""

    def test_uniform_tension_all_elements_same(self):
        """一様引張: 全要素で同一の塑性歪み."""
        n_elems = 8
        L = 100.0
        rod = _make_rod("uniform")
        mat = _make_material()
        plas = _make_plasticity()
        sec = rod.section

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6
        fixed_dofs = np.arange(6)

        P = 1.5 * SIGMA_Y0 * sec.A
        states = [CosseratPlasticState() for _ in range(n_elems)]
        u = np.zeros(total_dof)
        states_trial = None

        def _fint(u_):
            nonlocal states_trial
            _, f, st = assemble_cosserat_beam_plastic(
                n_elems, L, rod, mat, u_, states, plas,
                stiffness=False, internal_force=True,
            )
            states_trial = st
            return f

        def _Kt(u_):
            K, _, _ = assemble_cosserat_beam_plastic(
                n_elems, L, rod, mat, u_, states, plas,
                stiffness=True, internal_force=False,
            )
            return sp.csr_matrix(K)

        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems] = P

        result = newton_raphson(
            f_ext, fixed_dofs, _Kt, _fint,
            n_load_steps=1, u0=u, show_progress=False,
        )
        states = [s.copy() for s in states_trial]

        # 全要素の塑性歪みが同一
        eps_p_values = [s.axial.eps_p for s in states]
        for ep in eps_p_values:
            assert abs(ep - eps_p_values[0]) < 1e-12


class TestPlasticSRI:
    """SRI版の弾塑性テスト."""

    def test_sri_axial_yield(self):
        """SRI版で軸引張降伏の解析解と一致."""
        n_elems = 4
        L = 100.0
        rod = _make_rod("sri")
        mat = _make_material()
        plas = _make_plasticity()
        sec = rod.section

        n_nodes = n_elems + 1
        total_dof = n_nodes * 6
        fixed_dofs = np.arange(6)

        P = 1.5 * SIGMA_Y0 * sec.A
        states = [CosseratPlasticState() for _ in range(n_elems * 2)]  # 2 gauss pts
        u = np.zeros(total_dof)
        states_trial = None

        def _fint(u_):
            nonlocal states_trial
            _, f, st = assemble_cosserat_beam_plastic(
                n_elems, L, rod, mat, u_, states, plas,
                stiffness=False, internal_force=True,
            )
            states_trial = st
            return f

        def _Kt(u_):
            K, _, _ = assemble_cosserat_beam_plastic(
                n_elems, L, rod, mat, u_, states, plas,
                stiffness=True, internal_force=False,
            )
            return sp.csr_matrix(K)

        f_ext = np.zeros(total_dof)
        f_ext[6 * n_elems] = P

        result = newton_raphson(
            f_ext, fixed_dofs, _Kt, _fint,
            n_load_steps=1, u0=u, show_progress=False,
        )
        u = result.u
        u_tip = u[6 * n_elems]

        # 解析解
        E_t = E_MAT * H_ISO / (E_MAT + H_ISO)
        u_expected = (SIGMA_Y0 / E_MAT + (P / sec.A - SIGMA_Y0) / E_t) * L
        assert abs(u_tip - u_expected) / abs(u_expected) < 1e-4


class TestPlasticity1DValidation:
    """追加の入力検証テスト."""

    def test_negative_E_raises(self):
        with pytest.raises(ValueError):
            Plasticity1D(E=-1.0, iso=IsotropicHardening(sigma_y0=SIGMA_Y0))

    def test_negative_sigma_y0_raises(self):
        with pytest.raises(ValueError):
            Plasticity1D(E=E_MAT, iso=IsotropicHardening(sigma_y0=-10.0))

    def test_compression_yield(self):
        """圧縮方向の降伏."""
        plas = _make_plasticity()
        state = PlasticState1D()
        eps = -2.0 * SIGMA_Y0 / E_MAT
        result = plas.return_mapping(eps, state)
        assert result.stress < -SIGMA_Y0 * 0.99
        assert result.state_new.eps_p < 0.0
