"""Strategy Protocol 定義.

ソルバー内部の直交する振る舞い軸を Protocol で規定する。
具象実装は Phase 2 で各 strategy ファイルに配置する。

設計仕様: xkep_cae/process/process-architecture.md §2.2
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp


@runtime_checkable
class ContactForceStrategy(Protocol):
    """接触力の評価方法を規定する.

    実装:
    - NCPContactForce: Alart-Curnier NCP + 鞍点系
    - SmoothPenaltyContactForce: softplus + Uzawa外部ループ
    """

    def evaluate(
        self,
        u: np.ndarray,
        lambdas: np.ndarray,
        manager: object,
        k_pen: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """接触力と接触残差を評価.

        Returns:
            (contact_force, ncp_residual)
        """
        ...

    def tangent(
        self,
        u: np.ndarray,
        lambdas: np.ndarray,
        manager: object,
        k_pen: float,
    ) -> sp.csr_matrix:
        """接触接線剛性行列."""
        ...


@runtime_checkable
class FrictionStrategy(Protocol):
    """摩擦力の評価方法を規定する.

    実装:
    - NoFriction: 摩擦なし（デフォルト）
    - CoulombReturnMapping: Coulomb摩擦 return mapping
    - SmoothPenaltyFriction: smooth penalty + Uzawa摩擦
    """

    def evaluate(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """摩擦力と摩擦残差を評価.

        Returns:
            (friction_force, friction_residual)
        """
        ...

    def tangent(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
    ) -> sp.csr_matrix:
        """摩擦接線剛性行列."""
        ...


@runtime_checkable
class TimeIntegrationStrategy(Protocol):
    """時間積分方法を規定する.

    実装:
    - QuasiStatic: 準静的（荷重制御）
    - GeneralizedAlpha: Generalized-α 動的解析
    """

    def predict(self, u: np.ndarray, dt: float) -> np.ndarray:
        """予測子."""
        ...

    def correct(self, u: np.ndarray, du: np.ndarray, dt: float) -> None:
        """補正子."""
        ...

    def effective_stiffness(self, K: sp.csr_matrix, dt: float) -> sp.csr_matrix:
        """有効剛性行列 K_eff = K + α/(β*dt²)*M + γ/(β*dt)*C."""
        ...

    def effective_residual(self, R: np.ndarray, dt: float) -> np.ndarray:
        """有効残差."""
        ...


@runtime_checkable
class ContactGeometryStrategy(Protocol):
    """接触幾何の評価方法を規定する.

    実装:
    - PointToPoint: 最近接点ペア（現行デフォルト）
    - LineToLineGauss: line-to-line Gauss積分
    - MortarSegment: Mortar法セグメント
    """

    def detect(
        self,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        radii: np.ndarray | float,
    ) -> list:
        """接触候補ペアの検出."""
        ...

    def compute_gap(self, pair: object, node_coords: np.ndarray) -> float:
        """ギャップの計算."""
        ...


@runtime_checkable
class PenaltyStrategy(Protocol):
    """ペナルティ剛性の決定方法を規定する.

    実装:
    - AutoBeamEI: beam_E * beam_I / L³ ベース（デフォルト）
    - AutoEAL: E * A / L ベース
    - ManualPenalty: 手動指定
    - ContinuationPenalty: 段階的増加
    """

    def compute_k_pen(self, step: int, total_steps: int) -> float:
        """現在ステップのペナルティ剛性."""
        ...
