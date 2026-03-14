"""Strategy Protocol 定義.

ソルバー内部の直交する振る舞い軸を Protocol で規定する。
具象実装は Phase 2 で各 strategy ファイルに配置する。

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

    def update_geometry(
        self,
        pairs: list,
        node_coords: np.ndarray,
        *,
        config: object | None = None,
    ) -> None:
        """全ペアの幾何情報（s, t, gap, frame）を更新."""
        ...

    def build_constraint_jacobian(
        self,
        pairs: list,
        ndof_total: int,
        ndof_per_node: int = 6,
    ) -> tuple[sp.csr_matrix, list[int]]:
        """制約ヤコビアン G = ∂g_n/∂u を構築."""
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


@runtime_checkable
class CoatingStrategy(Protocol):
    """被膜接触モデルの評価方法を規定する.

    実装:
    - KelvinVoigtCoating: Kelvin-Voigt 弾性+粘性被膜モデル（status-137/140）
    - NoCoating: 被膜なし（ゼロ返却）
    """

    def forces(
        self,
        pairs: list,
        node_coords: np.ndarray,
        config: object,
        dt: float,
    ) -> np.ndarray:
        """被膜圧縮による接触力ベクトルを計算.

        Kelvin-Voigt: f = k*δ + c*δ̇

        Returns:
            f_coat: (ndof,) 被膜法線力
        """
        ...

    def stiffness(
        self,
        pairs: list,
        node_coords: np.ndarray,
        config: object,
        ndof_total: int,
        dt: float,
    ) -> sp.csr_matrix:
        """被膜接触の接線剛性行列.

        Returns:
            K_coat: (ndof_total, ndof_total) CSR
        """
        ...

    def friction_forces(
        self,
        pairs: list,
        node_coords: np.ndarray,
        config: object,
        u_cur: np.ndarray,
        u_ref: np.ndarray,
    ) -> np.ndarray:
        """被膜摩擦力を計算.

        Returns:
            f_fric: (ndof,) 被膜摩擦力
        """
        ...

    def friction_stiffness(
        self,
        pairs: list,
        node_coords: np.ndarray,
        config: object,
        ndof_total: int,
    ) -> sp.csr_matrix:
        """被膜摩擦の接線剛性行列.

        Returns:
            K_fric: (ndof_total, ndof_total) CSR
        """
        ...
