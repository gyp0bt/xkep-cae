"""Stage S2 テスト: 膜厚分布 Fourier 近似 + 修正コンプライアンス行列.

内面形状プロファイル計算、Fourier 係数抽出、膜厚分布を考慮した
コンプライアンス行列の検証テスト群。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xkep_cae.mesh.ring_compliance import (
    build_ring_compliance_matrix,
    build_variable_thickness_compliance_matrix,
    evaluate_fourier_profile,
    fourier_decompose_profile,
)
from xkep_cae.mesh.twisted_wire import (
    CoatingModel,
    SheathModel,
    compute_inner_surface_profile,
    make_twisted_wire_mesh,
    sheath_compliance_matrix,
)

# ============================================================
# ヘルパー
# ============================================================


def _make_mesh(n_strands: int, *, wire_radius: float = 0.5e-3, n_elems: int = 4):
    """テスト用の撚線メッシュを生成する."""
    return make_twisted_wire_mesh(
        n_strands,
        wire_diameter=wire_radius * 2.0,
        pitch=20.0e-3,
        length=20.0e-3,
        n_elems_per_strand=n_elems,
    )


def _default_sheath():
    """テスト用のシースモデル."""
    return SheathModel(thickness=0.2e-3, E=200e9, nu=0.3)


# ============================================================
# TestInnerSurfaceProfile: シース内面プロファイル
# ============================================================


class TestInnerSurfaceProfile:
    """compute_inner_surface_profile のテスト."""

    def test_seven_wire_profile_length(self):
        """7本撚り: 返り値の長さが n_theta と一致."""
        mesh = _make_mesh(7)
        theta, r = compute_inner_surface_profile(mesh, n_theta=180)
        assert len(theta) == 180
        assert len(r) == 180

    def test_seven_wire_profile_positive(self):
        """7本撚り: 内面半径が全て正."""
        mesh = _make_mesh(7)
        _, r = compute_inner_surface_profile(mesh)
        assert np.all(r > 0)

    def test_seven_wire_6fold_symmetry(self):
        """7本撚り（外層6本）: 内面プロファイルが 6 回対称."""
        mesh = _make_mesh(7)
        # 6の倍数でかつ十分細かいサンプリングで境界問題を回避
        n_theta = 360 * 6
        _, r = compute_inner_surface_profile(mesh, n_theta=n_theta)
        shift = n_theta // 6
        np.testing.assert_allclose(r, np.roll(r, shift), atol=1e-12)

    def test_three_wire_3fold_symmetry(self):
        """3本撚り: 内面プロファイルが 3 回対称."""
        mesh = _make_mesh(3)
        n_theta = 360 * 3
        _, r = compute_inner_surface_profile(mesh, n_theta=n_theta)
        shift = n_theta // 3
        np.testing.assert_allclose(r, np.roll(r, shift), atol=1e-12)

    def test_with_coating_larger(self):
        """被膜付きでは内面半径が素線のみの場合より大きい."""
        mesh = _make_mesh(7)
        coating = CoatingModel(thickness=0.05e-3, E=3e9, nu=0.35)
        _, r_bare = compute_inner_surface_profile(mesh)
        _, r_coated = compute_inner_surface_profile(mesh, coating=coating)
        assert np.all(r_coated > r_bare)

    def test_maximum_equals_envelope(self):
        """内面プロファイルの最大値がエンベロープ半径と一致."""
        from xkep_cae.mesh.twisted_wire import compute_envelope_radius

        mesh = _make_mesh(7)
        _, r = compute_inner_surface_profile(mesh)
        env = compute_envelope_radius(mesh)
        np.testing.assert_allclose(np.max(r), env, rtol=1e-6)

    def test_n_theta_validation(self):
        """n_theta < 3 でエラー."""
        mesh = _make_mesh(7)
        with pytest.raises(ValueError, match="n_theta"):
            compute_inner_surface_profile(mesh, n_theta=2)

    def test_nineteen_wire_12fold_symmetry(self):
        """19本撚り（外層12本）: 内面プロファイルが 12 回対称."""
        mesh = _make_mesh(19)
        n_theta = 360
        _, r = compute_inner_surface_profile(mesh, n_theta=n_theta)
        shift = n_theta // 12
        np.testing.assert_allclose(r, np.roll(r, shift), atol=1e-12)


# ============================================================
# TestFourierDecomposition: Fourier 分解
# ============================================================


class TestFourierDecomposition:
    """fourier_decompose_profile / evaluate_fourier_profile のテスト."""

    def test_constant_profile(self):
        """一定プロファイル: R₀ のみ、高次係数ゼロ."""
        theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        r = np.full_like(theta, 5.0)
        fc = fourier_decompose_profile(theta, r, n_modes=10)
        assert fc["R0"] == pytest.approx(5.0)
        np.testing.assert_allclose(fc["a"][1:], 0.0, atol=1e-12)
        np.testing.assert_allclose(fc["b"][1:], 0.0, atol=1e-12)

    def test_pure_cosine(self):
        """r(θ) = 5 + 0.1 cos(6θ): a₆ ≈ 0.1, 他はゼロ."""
        theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        r = 5.0 + 0.1 * np.cos(6 * theta)
        fc = fourier_decompose_profile(theta, r, n_modes=12)
        assert fc["R0"] == pytest.approx(5.0, abs=1e-10)
        assert fc["a"][6] == pytest.approx(0.1, abs=1e-10)
        # a₆ 以外の cos 係数はゼロ
        for n in range(1, 13):
            if n != 6:
                assert abs(fc["a"][n]) < 1e-10
        # sin 係数は全てゼロ
        np.testing.assert_allclose(fc["b"][1:], 0.0, atol=1e-10)

    def test_pure_sine(self):
        """r(θ) = 3 + 0.2 sin(4θ): b₄ ≈ 0.2."""
        theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        r = 3.0 + 0.2 * np.sin(4 * theta)
        fc = fourier_decompose_profile(theta, r, n_modes=8)
        assert fc["b"][4] == pytest.approx(0.2, abs=1e-10)
        # b₄ 以外の sin 係数はゼロ
        for n in range(1, 9):
            if n != 4:
                assert abs(fc["b"][n]) < 1e-10

    def test_roundtrip(self):
        """Fourier 分解 → 再構築でプロファイルが復元される."""
        theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        r_orig = 5.0 + 0.1 * np.cos(6 * theta) + 0.05 * np.sin(12 * theta)
        fc = fourier_decompose_profile(theta, r_orig, n_modes=24)
        r_recon = evaluate_fourier_profile(fc, theta)
        np.testing.assert_allclose(r_recon, r_orig, atol=1e-10)

    def test_seven_wire_dominant_mode(self):
        """7本撚り（外層6本）: a₆ が支配モード."""
        mesh = _make_mesh(7)
        theta, r = compute_inner_surface_profile(mesh, n_theta=360)
        fc = fourier_decompose_profile(theta, r, n_modes=24)
        # a₆ の絶対値が他の n≥1 のモードの中で最大
        amplitudes = [math.sqrt(fc["a"][n] ** 2 + fc["b"][n] ** 2) for n in range(1, 25)]
        dominant_mode = np.argmax(amplitudes) + 1  # 1-indexed
        assert dominant_mode == 6

    def test_nineteen_wire_dominant_mode(self):
        """19本撚り（外層12本）: a₁₂ が支配モード."""
        mesh = _make_mesh(19)
        theta, r = compute_inner_surface_profile(mesh, n_theta=720)
        fc = fourier_decompose_profile(theta, r, n_modes=36)
        amplitudes = [math.sqrt(fc["a"][n] ** 2 + fc["b"][n] ** 2) for n in range(1, 37)]
        dominant_mode = np.argmax(amplitudes) + 1
        assert dominant_mode == 12

    def test_convergence_with_more_modes(self):
        """Fourier 級数が多いほどプロファイル再構築精度が向上."""
        mesh = _make_mesh(7)
        theta, r_orig = compute_inner_surface_profile(mesh, n_theta=360)
        errors = []
        for nm in [6, 12, 24, 48]:
            fc = fourier_decompose_profile(theta, r_orig, nm)
            r_recon = evaluate_fourier_profile(fc, theta)
            errors.append(np.max(np.abs(r_recon - r_orig)))
        # 単調減少
        for i in range(len(errors) - 1):
            assert errors[i + 1] <= errors[i] + 1e-15

    def test_n_modes_validation(self):
        """n_modes < 1 でエラー."""
        theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        r = np.ones(10)
        with pytest.raises(ValueError, match="n_modes"):
            fourier_decompose_profile(theta, r, n_modes=0)

    def test_length_mismatch_validation(self):
        """theta と r_profile の長さ不一致でエラー."""
        theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        r = np.ones(8)
        with pytest.raises(ValueError, match="長さ"):
            fourier_decompose_profile(theta, r, n_modes=5)


# ============================================================
# TestVariableThicknessCompliance: 膜厚分布コンプライアンス行列
# ============================================================


class TestVariableThicknessCompliance:
    """build_variable_thickness_compliance_matrix のテスト."""

    @pytest.fixture
    def uniform_params(self):
        """均一厚みの基本パラメータ."""
        N = 6
        a = 1.0e-3  # 内径
        b = 1.2e-3  # 外径
        E = 200e9
        nu = 0.3
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        r_inner = np.full(N, a)
        return N, angles, r_inner, b, E, nu

    def test_uniform_matches_s1(self, uniform_params):
        """均一厚みの場合 S1 の build_ring_compliance_matrix と一致."""
        N, angles, r_inner, b, E, nu = uniform_params
        a = r_inner[0]
        C_s1 = build_ring_compliance_matrix(N, a, b, E, nu, n_modes=24)
        C_s2 = build_variable_thickness_compliance_matrix(N, angles, r_inner, b, E, nu, n_modes=24)
        np.testing.assert_allclose(C_s2, C_s1, rtol=1e-10)

    def test_symmetry(self, uniform_params):
        """コンプライアンス行列が対称."""
        N, angles, r_inner, b, E, nu = uniform_params
        # 非均一内径
        r_inner_var = r_inner + np.array([0, 0.01e-3, 0, 0.01e-3, 0, 0.01e-3])
        C = build_variable_thickness_compliance_matrix(N, angles, r_inner_var, b, E, nu)
        np.testing.assert_allclose(C, C.T, atol=1e-20)

    def test_positive_definite(self, uniform_params):
        """コンプライアンス行列が正定値."""
        N, angles, r_inner, b, E, nu = uniform_params
        r_inner_var = r_inner + np.array([0, 0.02e-3, 0, 0.02e-3, 0, 0.02e-3])
        C = build_variable_thickness_compliance_matrix(N, angles, r_inner_var, b, E, nu)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0)

    def test_not_circulant_for_nonuniform(self, uniform_params):
        """非均一厚みの場合、循環行列ではなくなる."""
        N, angles, r_inner, b, E, nu = uniform_params
        # 1点だけ内径を変える
        r_inner_var = r_inner.copy()
        r_inner_var[0] += 0.05e-3
        C = build_variable_thickness_compliance_matrix(N, angles, r_inner_var, b, E, nu)
        # 循環行列なら C[0,:] を回転すると C[1,:] と一致するはず
        row0_shifted = np.roll(C[0, :], -1)
        # 非均一なので一致しない
        assert not np.allclose(row0_shifted, C[1, :], atol=1e-15)

    def test_thinner_is_more_compliant(self, uniform_params):
        """薄い部分のコンプライアンスが大きい."""
        N, angles, r_inner, b, E, nu = uniform_params
        C_uniform = build_variable_thickness_compliance_matrix(N, angles, r_inner, b, E, nu)
        # 接点0の内径を大きく（= 薄く）する
        r_inner_thin = r_inner.copy()
        r_inner_thin[0] += 0.05e-3
        C_thin = build_variable_thickness_compliance_matrix(N, angles, r_inner_thin, b, E, nu)
        # 対角項 C[0,0] が増大（薄い → より柔らかい）
        assert C_thin[0, 0] > C_uniform[0, 0]

    def test_N_validation(self):
        """N < 2 でエラー."""
        with pytest.raises(ValueError, match="N=1"):
            build_variable_thickness_compliance_matrix(
                1, np.array([0.0]), np.array([1.0e-3]), 1.2e-3, 200e9, 0.3
            )

    def test_length_mismatch_angles(self):
        """contact_angles 長さ不一致でエラー."""
        with pytest.raises(ValueError, match="不一致"):
            build_variable_thickness_compliance_matrix(
                3,
                np.array([0, 1]),  # 2個
                np.array([1.0e-3, 1.0e-3, 1.0e-3]),
                1.2e-3,
                200e9,
                0.3,
            )

    def test_length_mismatch_r_inner(self):
        """r_inner_at_contacts 長さ不一致でエラー."""
        with pytest.raises(ValueError, match="不一致"):
            build_variable_thickness_compliance_matrix(
                3,
                np.array([0, 2, 4]),
                np.array([1.0e-3, 1.0e-3]),  # 2個
                1.2e-3,
                200e9,
                0.3,
            )

    def test_invalid_inner_radius(self):
        """内径 >= 外径 でエラー."""
        with pytest.raises(ValueError, match="内径"):
            build_variable_thickness_compliance_matrix(
                2,
                np.array([0, np.pi]),
                np.array([1.3e-3, 1.0e-3]),  # 1.3 > 1.2
                1.2e-3,
                200e9,
                0.3,
            )


# ============================================================
# TestSheathComplianceMatrix: 高レベル API
# ============================================================


class TestSheathComplianceMatrix:
    """sheath_compliance_matrix のテスト."""

    def test_seven_wire_returns_6x6(self):
        """7本撚り → 6×6 行列（外層6本）."""
        mesh = _make_mesh(7)
        sheath = _default_sheath()
        C = sheath_compliance_matrix(mesh, sheath)
        assert C.shape == (6, 6)

    def test_seven_wire_symmetric(self):
        """7本撚り: 対称行列."""
        mesh = _make_mesh(7)
        sheath = _default_sheath()
        C = sheath_compliance_matrix(mesh, sheath)
        np.testing.assert_allclose(C, C.T, atol=1e-20)

    def test_seven_wire_positive_definite(self):
        """7本撚り: 正定値."""
        mesh = _make_mesh(7)
        sheath = _default_sheath()
        C = sheath_compliance_matrix(mesh, sheath)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0)

    def test_nineteen_wire_returns_12x12(self):
        """19本撚り → 12×12 行列（外層12本）."""
        mesh = _make_mesh(19)
        sheath = _default_sheath()
        C = sheath_compliance_matrix(mesh, sheath)
        assert C.shape == (12, 12)

    def test_coating_effect(self):
        """被膜付きでコンプライアンスが変化."""
        mesh = _make_mesh(7)
        sheath = _default_sheath()
        C_bare = sheath_compliance_matrix(mesh, sheath)
        coating = CoatingModel(thickness=0.05e-3, E=3e9, nu=0.35)
        C_coated = sheath_compliance_matrix(mesh, sheath, coating=coating)
        # 被膜でシース内径が変わるため対角項が異なる
        rel_diff = np.abs(np.diag(C_coated) - np.diag(C_bare)) / np.abs(np.diag(C_bare))
        assert np.all(rel_diff > 0.01)  # 1%以上の相対差

    def test_three_wire_returns_3x3(self):
        """3本撚り → 3×3 行列（全3本が最外層）."""
        mesh = _make_mesh(3)
        sheath = _default_sheath()
        C = sheath_compliance_matrix(mesh, sheath)
        assert C.shape == (3, 3)

    def test_thicker_sheath_less_compliant(self):
        """厚いシースの方がコンプライアンスが小さい."""
        mesh = _make_mesh(7)
        sheath_thin = SheathModel(thickness=0.1e-3, E=200e9, nu=0.3)
        sheath_thick = SheathModel(thickness=0.5e-3, E=200e9, nu=0.3)
        C_thin = sheath_compliance_matrix(mesh, sheath_thin)
        C_thick = sheath_compliance_matrix(mesh, sheath_thick)
        # 対角項: 厚いシースの方が小さい
        assert np.all(np.diag(C_thick) < np.diag(C_thin))

    def test_clearance_effect(self):
        """クリアランス変化でコンプライアンスが変化."""
        mesh = _make_mesh(7)
        sheath0 = SheathModel(thickness=0.2e-3, E=200e9, nu=0.3, clearance=0.0)
        sheath1 = SheathModel(thickness=0.2e-3, E=200e9, nu=0.3, clearance=0.1e-3)
        C0 = sheath_compliance_matrix(mesh, sheath0)
        C1 = sheath_compliance_matrix(mesh, sheath1)
        # クリアランス増大 → 内径増大 → 対角項が増大
        rel_diff = np.abs(np.diag(C1) - np.diag(C0)) / np.abs(np.diag(C0))
        assert np.all(rel_diff > 0.01)  # 1%以上の相対差
