"""Strategy Protocol の契約テスト.

Protocol の構造的部分型準拠を検証する。
具象Strategy実装の意味論テストは各 test_*.py で行う。

テスト対象: xkep_cae/process/strategies/protocols.py
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from __xkep_cae_deprecated.process.strategies.compatibility import (
    INCOMPATIBLE_COMBINATIONS,
    VERIFIED_COMBINATIONS,
    validate_strategy_combination,
)
from __xkep_cae_deprecated.process.strategies.protocols import (
    ContactForceStrategy,
    ContactGeometryStrategy,
    FrictionStrategy,
    PenaltyStrategy,
    TimeIntegrationStrategy,
)

# ============================================================
# ダミー実装（Protocol準拠テスト用）
# ============================================================


class StubContactForce:
    """ContactForceStrategy の最小実装."""

    def evaluate(
        self, u: np.ndarray, lambdas: np.ndarray, manager: object, k_pen: float
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(u)
        return np.zeros(n), np.zeros(n)

    def tangent(
        self, u: np.ndarray, lambdas: np.ndarray, manager: object, k_pen: float
    ) -> sp.csr_matrix:
        n = len(u)
        return sp.csr_matrix((n, n))


class StubFriction:
    """FrictionStrategy の最小実装."""

    def evaluate(
        self, u: np.ndarray, contact_pairs: list, mu: float
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(u)
        return np.zeros(n), np.zeros(n)

    def tangent(self, u: np.ndarray, contact_pairs: list, mu: float) -> sp.csr_matrix:
        n = len(u)
        return sp.csr_matrix((n, n))


class StubTimeIntegration:
    """TimeIntegrationStrategy の最小実装."""

    def predict(self, u: np.ndarray, dt: float) -> np.ndarray:
        return u.copy()

    def correct(self, u: np.ndarray, du: np.ndarray, dt: float) -> None:
        pass

    def effective_stiffness(self, K: sp.csr_matrix, dt: float) -> sp.csr_matrix:
        return K

    def effective_residual(self, R: np.ndarray, dt: float) -> np.ndarray:
        return R


class StubContactGeometry:
    """ContactGeometryStrategy の最小実装."""

    def detect(
        self, node_coords: np.ndarray, connectivity: np.ndarray, radii: np.ndarray | float
    ) -> list:
        return []

    def compute_gap(self, pair: object, node_coords: np.ndarray) -> float:
        return 0.0

    def update_geometry(
        self, pairs: list, node_coords: np.ndarray, *, config: object | None = None
    ) -> None:
        pass

    def build_constraint_jacobian(
        self, pairs: list, ndof_total: int, ndof_per_node: int = 6
    ) -> tuple[sp.csr_matrix, list[int]]:
        return sp.csr_matrix((0, ndof_total)), []


class StubPenalty:
    """PenaltyStrategy の最小実装."""

    def compute_k_pen(self, step: int, total_steps: int) -> float:
        return 1.0


# ============================================================
# TestProtocolConformance — Protocol 準拠テスト
# ============================================================


class TestProtocolConformance:
    """ダミー実装が Protocol を満たすことを検証."""

    def test_contact_force_conforms(self) -> None:
        assert isinstance(StubContactForce(), ContactForceStrategy)

    def test_friction_conforms(self) -> None:
        assert isinstance(StubFriction(), FrictionStrategy)

    def test_time_integration_conforms(self) -> None:
        assert isinstance(StubTimeIntegration(), TimeIntegrationStrategy)

    def test_contact_geometry_conforms(self) -> None:
        assert isinstance(StubContactGeometry(), ContactGeometryStrategy)

    def test_penalty_conforms(self) -> None:
        assert isinstance(StubPenalty(), PenaltyStrategy)


# ============================================================
# TestProtocolContract — Protocol の契約テスト（意味論）
# ============================================================


class TestProtocolContract:
    """Protocol 実装の不変条件テスト."""

    def test_contact_force_returns_same_size(self) -> None:
        """evaluate() は入力と同サイズの配列を返すこと."""
        cf = StubContactForce()
        u = np.zeros(10)
        lam = np.zeros(10)
        f, r = cf.evaluate(u, lam, None, 1.0)
        assert f.shape == u.shape
        assert r.shape == u.shape

    def test_contact_force_tangent_square(self) -> None:
        """tangent() は正方行列を返すこと."""
        cf = StubContactForce()
        u = np.zeros(10)
        K = cf.tangent(u, np.zeros(10), None, 1.0)
        assert K.shape[0] == K.shape[1] == len(u)

    def test_time_integration_predict_preserves_size(self) -> None:
        """predict() は入力と同サイズの配列を返すこと."""
        ti = StubTimeIntegration()
        u = np.ones(6)
        u_pred = ti.predict(u, 0.01)
        assert u_pred.shape == u.shape

    def test_penalty_positive(self) -> None:
        """compute_k_pen() は正値を返すこと."""
        p = StubPenalty()
        assert p.compute_k_pen(0, 10) > 0


# ============================================================
# TestCompatibilityMatrix — 互換性マトリクステスト
# ============================================================


class TestCompatibilityMatrix:
    """Strategy 互換性マトリクスのテスト."""

    def test_verified_combinations_non_empty(self) -> None:
        """ホワイトリストが空でないこと."""
        assert len(VERIFIED_COMBINATIONS) > 0

    def test_incompatible_has_reason(self) -> None:
        """ブラックリストには reason が含まれること."""
        for incompat in INCOMPATIBLE_COMBINATIONS:
            assert "reason" in incompat

    def test_verified_passes_validation(self) -> None:
        """ホワイトリストの組み合わせは警告なし."""
        for combo in VERIFIED_COMBINATIONS:
            warnings = validate_strategy_combination(combo)
            assert warnings == [], f"ホワイトリスト組み合わせが警告: {warnings}"

    def test_incompatible_generates_warning(self) -> None:
        """ブラックリストの組み合わせは警告を出すこと."""
        # ブラックリストのキーでテスト（reasonを除く）
        combo = {k: v for k, v in INCOMPATIBLE_COMBINATIONS[0].items() if k != "reason"}
        # 不足キーを追加して完全な組み合わせにする
        combo.setdefault("time_integration", "QuasiStaticProcess")
        combo.setdefault("contact_geometry", "PointToPointProcess")
        combo.setdefault("penalty", "AutoBeamEIProcess")
        warnings = validate_strategy_combination(combo)
        assert any("非互換" in w for w in warnings)

    def test_unknown_combination_warns(self) -> None:
        """未知の組み合わせは未検証の警告を出すこと."""
        combo = {
            "contact_force": "UnknownProcess",
            "friction": "UnknownProcess",
            "time_integration": "UnknownProcess",
            "contact_geometry": "UnknownProcess",
            "penalty": "UnknownProcess",
        }
        warnings = validate_strategy_combination(combo)
        assert any("未検証" in w for w in warnings)
