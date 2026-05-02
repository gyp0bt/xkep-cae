"""SO(3) Lie 群／代数ユーティリティの単体テスト.

設計仕様: ``xkep_cae/elements/docs/cosserat_beam.md`` Phase 0.

検証範囲:

- skew / vee の対合性と ``a × b = skew(a) @ b`` 整合
- exp_so3(0) = I, log_so3(I) = 0
- exp ↔ log の往復 (2 桁の角度範囲)
- log(exp(θ)) = θ (||θ|| < π)
- exp(θ) は SO(3) (R Rᵀ = I, det R = 1)
- dexp_so3 / dexp_inv_so3 の互逆性
- dexp_so3(θ) → I と dexp_inv_so3(θ) → I (φ → 0)
- 解析公式と FD 整合 (``d/dα exp(α θ) |_{α=1} = exp(θ) skew(T(θ) θ̇)`` 検証は
  Cosserat 梁 Phase 1 で別途。本 Phase 0 は SO(3) 単独の閉形式整合に留める)
- バッチ版と単点版の数値一致
- CR 梁 (``_beam_cr.py``) private 関数との数値同値
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.elements._beam_cr import (
    _rotmat_to_rotvec,
    _rotvec_to_rotmat,
    _skew,
    _tangent_operator,
    _tangent_operator_inv,
)
from xkep_cae.mathematics.so3 import (
    batch_dexp_inv_so3,
    batch_dexp_so3,
    batch_exp_so3,
    batch_log_so3,
    batch_skew,
    compose,
    dexp_inv_so3,
    dexp_so3,
    exp_so3,
    log_so3,
    skew,
    vee,
)

# =====================================================================
# skew / vee
# =====================================================================


class TestSkewVee:
    """歪対称行列とベクトルの相互変換."""

    def test_skew_zero(self) -> None:
        S = skew(np.zeros(3))
        assert np.allclose(S, np.zeros((3, 3)))

    def test_skew_unit_axes(self) -> None:
        S = skew(np.array([1.0, 2.0, 3.0]))
        expected = np.array(
            [
                [0.0, -3.0, 2.0],
                [3.0, 0.0, -1.0],
                [-2.0, 1.0, 0.0],
            ]
        )
        assert np.allclose(S, expected)

    def test_skew_is_antisymmetric(self) -> None:
        rng = np.random.default_rng(42)
        v = rng.standard_normal(3)
        S = skew(v)
        assert np.allclose(S, -S.T)

    def test_skew_implements_cross_product(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        assert np.allclose(skew(a) @ b, np.cross(a, b))

    def test_vee_inverse_of_skew(self) -> None:
        rng = np.random.default_rng(7)
        v = rng.standard_normal(3)
        assert np.allclose(vee(skew(v)), v)

    def test_skew_inverse_of_vee_for_skew_input(self) -> None:
        rng = np.random.default_rng(9)
        v = rng.standard_normal(3)
        S = skew(v)
        assert np.allclose(skew(vee(S)), S)

    def test_skew_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="skew は"):
            skew(np.zeros(4))

    def test_vee_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="vee は"):
            vee(np.zeros((4, 4)))


# =====================================================================
# exp_so3 / log_so3
# =====================================================================


class TestExpLog:
    """SO(3) 指数写像と対数写像."""

    def test_exp_zero_is_identity(self) -> None:
        R = exp_so3(np.zeros(3))
        assert np.allclose(R, np.eye(3))

    def test_log_identity_is_zero(self) -> None:
        theta = log_so3(np.eye(3))
        assert np.allclose(theta, np.zeros(3))

    def test_exp_is_rotation_matrix(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(10):
            theta = rng.standard_normal(3) * 2.0
            R = exp_so3(theta)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
            assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_exp_log_roundtrip_general(self) -> None:
        rng = np.random.default_rng(2)
        for _ in range(20):
            theta = rng.standard_normal(3)
            phi = float(np.linalg.norm(theta))
            if phi < 1e-6 or phi > np.pi - 1e-6:
                continue
            theta2 = log_so3(exp_so3(theta))
            assert np.allclose(theta2, theta, atol=1e-12)

    def test_exp_log_roundtrip_small_angle(self) -> None:
        theta = np.array([1e-10, 2e-10, -1e-10])
        theta2 = log_so3(exp_so3(theta))
        assert np.allclose(theta2, theta, atol=1e-15)

    def test_exp_log_roundtrip_near_pi(self) -> None:
        # 軸 z 周りの π に近い回転で四元数経由 log の安定性を確認
        theta = np.array([0.0, 0.0, np.pi - 1e-6])
        theta2 = log_so3(exp_so3(theta))
        assert np.allclose(theta2, theta, atol=1e-9)

    def test_exp_axis_aligned_known_rotation(self) -> None:
        # x 軸周り 90° で y → z, z → -y
        R = exp_so3(np.array([np.pi / 2.0, 0.0, 0.0]))
        assert np.allclose(R @ np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        assert np.allclose(R @ np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]))

    def test_exp_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="exp_so3 は"):
            exp_so3(np.zeros(4))

    def test_log_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="log_so3 は"):
            log_so3(np.zeros((4, 4)))


# =====================================================================
# dexp / dexp_inv
# =====================================================================


class TestDexpAndInverse:
    """指数写像のヤコビアンとその逆."""

    def test_dexp_zero_is_identity(self) -> None:
        T = dexp_so3(np.zeros(3))
        assert np.allclose(T, np.eye(3))

    def test_dexp_inv_zero_is_identity(self) -> None:
        Ti = dexp_inv_so3(np.zeros(3))
        assert np.allclose(Ti, np.eye(3))

    def test_dexp_inverse_at_general_angle(self) -> None:
        rng = np.random.default_rng(3)
        for _ in range(20):
            theta = rng.standard_normal(3)
            phi = float(np.linalg.norm(theta))
            if phi < 1e-6 or phi > np.pi - 1e-6:
                continue
            T = dexp_so3(theta)
            Ti = dexp_inv_so3(theta)
            assert np.allclose(T @ Ti, np.eye(3), atol=1e-11)
            assert np.allclose(Ti @ T, np.eye(3), atol=1e-11)

    def test_dexp_inverse_small_angle(self) -> None:
        theta = np.array([1e-10, 0.0, 0.0])
        T = dexp_so3(theta)
        Ti = dexp_inv_so3(theta)
        assert np.allclose(T @ Ti, np.eye(3), atol=1e-15)

    def test_dexp_along_axis_acts_as_identity(self) -> None:
        # T(θ) · θ = θ (Bernoulli 数の係数構造より、軸方向は不変).
        rng = np.random.default_rng(4)
        for _ in range(10):
            theta = rng.standard_normal(3)
            phi = float(np.linalg.norm(theta))
            if phi < 1e-6 or phi > np.pi - 1e-6:
                continue
            T = dexp_so3(theta)
            assert np.allclose(T @ theta, theta, atol=1e-12)

    def test_dexp_inv_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="dexp_inv_so3 は"):
            dexp_inv_so3(np.zeros(2))


# =====================================================================
# compose
# =====================================================================


class TestCompose:
    def test_compose_identity_left(self) -> None:
        rng = np.random.default_rng(5)
        R = exp_so3(rng.standard_normal(3))
        assert np.allclose(compose(np.eye(3), R), R)

    def test_compose_identity_right(self) -> None:
        rng = np.random.default_rng(6)
        R = exp_so3(rng.standard_normal(3))
        assert np.allclose(compose(R, np.eye(3)), R)

    def test_compose_inverse(self) -> None:
        rng = np.random.default_rng(11)
        theta = rng.standard_normal(3)
        R = exp_so3(theta)
        Rinv = exp_so3(-theta)
        assert np.allclose(compose(R, Rinv), np.eye(3), atol=1e-13)


# =====================================================================
# バッチ版整合
# =====================================================================


class TestBatchConsistency:
    """バッチ版が単点版とサンプルごとに一致する."""

    def _sample_thetas(self) -> np.ndarray:
        rng = np.random.default_rng(13)
        thetas = rng.standard_normal((8, 3)) * 1.5
        # 小角・大角の両方を強制混合
        thetas[0] = np.array([1e-12, 2e-12, -1e-12])
        thetas[1] = np.array([np.pi - 1e-7, 0.0, 0.0])
        return thetas

    def test_batch_skew_matches_scalar(self) -> None:
        v = self._sample_thetas()
        S_batch = batch_skew(v)
        for i in range(v.shape[0]):
            assert np.allclose(S_batch[i], skew(v[i]))

    def test_batch_exp_matches_scalar(self) -> None:
        thetas = self._sample_thetas()
        Rs = batch_exp_so3(thetas)
        for i in range(thetas.shape[0]):
            assert np.allclose(Rs[i], exp_so3(thetas[i]), atol=1e-12)

    def test_batch_log_matches_scalar(self) -> None:
        thetas = self._sample_thetas()
        Rs = batch_exp_so3(thetas)
        thetas_back = batch_log_so3(Rs)
        for i in range(thetas.shape[0]):
            phi = float(np.linalg.norm(thetas[i]))
            if phi > np.pi - 1e-6:
                continue
            assert np.allclose(thetas_back[i], log_so3(Rs[i]), atol=1e-11)

    def test_batch_dexp_matches_scalar(self) -> None:
        thetas = self._sample_thetas()
        Ts = batch_dexp_so3(thetas)
        for i in range(thetas.shape[0]):
            assert np.allclose(Ts[i], dexp_so3(thetas[i]), atol=1e-12)

    def test_batch_dexp_inv_matches_scalar(self) -> None:
        thetas = self._sample_thetas()
        Tis = batch_dexp_inv_so3(thetas)
        for i in range(thetas.shape[0]):
            assert np.allclose(Tis[i], dexp_inv_so3(thetas[i]), atol=1e-12)

    def test_batch_dexp_inverse_pair(self) -> None:
        thetas = self._sample_thetas()
        Ts = batch_dexp_so3(thetas)
        Tis = batch_dexp_inv_so3(thetas)
        prods = np.einsum("nij,njk->nik", Ts, Tis)
        eye = np.broadcast_to(np.eye(3), prods.shape)
        assert np.allclose(prods, eye, atol=1e-10)

    def test_batch_skew_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="batch_skew は"):
            batch_skew(np.zeros(3))


# =====================================================================
# CR 梁 private 実装との数値同値
# =====================================================================


class TestParityWithCRBeam:
    """``_beam_cr.py`` の private 関数と公開 API の数値同値性.

    Phase 4 以降で _beam_cr が so3 へ委譲する DRY 化を予定しているため、
    本テストはそのリファクタの安全網となる。
    """

    def test_skew_matches_beam_cr(self) -> None:
        rng = np.random.default_rng(21)
        v = rng.standard_normal(3)
        assert np.allclose(skew(v), _skew(v))

    def test_exp_matches_beam_cr(self) -> None:
        rng = np.random.default_rng(22)
        for _ in range(5):
            theta = rng.standard_normal(3)
            phi = float(np.linalg.norm(theta))
            if phi > np.pi - 1e-6:
                continue
            assert np.allclose(exp_so3(theta), _rotvec_to_rotmat(theta), atol=1e-12)

    def test_log_matches_beam_cr(self) -> None:
        rng = np.random.default_rng(23)
        for _ in range(5):
            theta = rng.standard_normal(3)
            phi = float(np.linalg.norm(theta))
            if phi > np.pi - 1e-6:
                continue
            R = exp_so3(theta)
            assert np.allclose(log_so3(R), _rotmat_to_rotvec(R), atol=1e-12)

    def test_dexp_matches_beam_cr(self) -> None:
        rng = np.random.default_rng(24)
        for _ in range(5):
            theta = rng.standard_normal(3)
            phi = float(np.linalg.norm(theta))
            if phi > np.pi - 1e-6:
                continue
            assert np.allclose(dexp_so3(theta), _tangent_operator(theta), atol=1e-12)

    def test_dexp_inv_matches_beam_cr(self) -> None:
        rng = np.random.default_rng(25)
        for _ in range(5):
            theta = rng.standard_normal(3)
            phi = float(np.linalg.norm(theta))
            if phi > np.pi - 1e-6:
                continue
            assert np.allclose(dexp_inv_so3(theta), _tangent_operator_inv(theta), atol=1e-12)
