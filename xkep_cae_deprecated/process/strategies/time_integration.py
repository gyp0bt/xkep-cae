"""TimeIntegration Strategy 具象実装.

時間積分方法を Strategy として実装する。

Phase 3 統合:
- create_time_integration_strategy() ファクトリで solver_ncp.py の
  動的解析初期化ロジックを Strategy に移譲する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae_deprecated.process.base import ProcessMeta
from xkep_cae_deprecated.process.categories import SolverProcess


@dataclass(frozen=True)
class TimeIntegrationInput:
    """TimeIntegration Strategy の入力."""

    u: np.ndarray
    du: np.ndarray
    dt: float


@dataclass
class TimeIntegrationOutput:
    """TimeIntegration Strategy の出力."""

    u: np.ndarray


class QuasiStaticProcess(SolverProcess[TimeIntegrationInput, TimeIntegrationOutput]):
    """準静的解析（荷重制御）.

    時間積分なし。K_eff = K, R_eff = R をそのまま返す。
    predict/correct は identity 操作。
    """

    meta = ProcessMeta(
        name="QuasiStatic",
        module="solve",
        version="0.1.0",
        document_path="docs/time_integration.md",
    )

    @property
    def is_dynamic(self) -> bool:
        """動的解析かどうか."""
        return False

    @property
    def vel(self) -> np.ndarray:
        """速度: 準静的では None 相当."""
        return np.zeros(0)

    @property
    def acc(self) -> np.ndarray:
        """加速度: 準静的では None 相当."""
        return np.zeros(0)

    def predict(self, u: np.ndarray, dt: float) -> np.ndarray:
        """予測子: 準静的では変位をそのまま返す."""
        return u.copy()

    def correct(self, u: np.ndarray, du: np.ndarray, dt: float) -> None:
        """補正子: 準静的では何もしない."""

    def effective_stiffness(self, K: sp.csr_matrix, dt: float) -> sp.csr_matrix:
        """有効剛性行列: 準静的では K そのまま."""
        return K

    def effective_residual(self, R: np.ndarray, dt: float) -> np.ndarray:
        """有効残差: 準静的では R そのまま."""
        return R

    def checkpoint(self) -> None:
        """チェックポイント保存: 準静的では何もしない."""

    def restore_checkpoint(self) -> None:
        """チェックポイント復元: 準静的では何もしない."""

    def process(self, input_data: TimeIntegrationInput) -> TimeIntegrationOutput:
        u_pred = self.predict(input_data.u, input_data.dt)
        return TimeIntegrationOutput(u=u_pred)


class GeneralizedAlphaProcess(SolverProcess[TimeIntegrationInput, TimeIntegrationOutput]):
    """Generalized-α 動的解析（Chung & Hulbert 1993）.

    Newmark-β の上位互換で、スペクトル半径 rho_inf によって
    高周波の数値減衰を制御できる。

    パラメータ:
        rho_inf=1.0: Newmark 平均加速度法（エネルギー保存）
        rho_inf=0.0: 最大数値減衰（高周波完全減衰）
        推奨: 0.9〜1.0
    """

    meta = ProcessMeta(
        name="GeneralizedAlpha",
        module="solve",
        version="0.1.0",
        document_path="docs/time_integration.md",
    )

    def __init__(
        self,
        mass_matrix: sp.csr_matrix | np.ndarray,
        *,
        damping_matrix: sp.csr_matrix | np.ndarray | None = None,
        rho_inf: float = 0.9,
    ) -> None:
        self.M = sp.csr_matrix(mass_matrix) if not sp.issparse(mass_matrix) else mass_matrix
        self.C = (
            sp.csr_matrix(damping_matrix)
            if damping_matrix is not None and not sp.issparse(damping_matrix)
            else damping_matrix
        )
        self.rho_inf = rho_inf

        # Chung & Hulbert (1993) パラメータ
        self.alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
        self.alpha_f = rho_inf / (rho_inf + 1.0)
        self.gamma = 0.5 - self.alpha_m + self.alpha_f
        self.beta = 0.25 * (1.0 - self.alpha_m + self.alpha_f) ** 2

        # 状態変数
        ndof = self.M.shape[0]
        self.vel = np.zeros(ndof)
        self.acc = np.zeros(ndof)
        self._vel_old = np.zeros(ndof)
        self._acc_old = np.zeros(ndof)
        self._u_pred = np.zeros(ndof)
        self._v_pred = np.zeros(ndof)
        # チェックポイント用
        self._vel_ckpt: np.ndarray | None = None
        self._acc_ckpt: np.ndarray | None = None

    @property
    def is_dynamic(self) -> bool:
        """動的解析かどうか."""
        return True

    def set_initial_state(
        self,
        velocity: np.ndarray | None = None,
        acceleration: np.ndarray | None = None,
    ) -> None:
        """初期速度・加速度を設定."""
        if velocity is not None:
            self.vel = velocity.copy()
        if acceleration is not None:
            self.acc = acceleration.copy()

    def predict(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Newmark 予測子.

        u_pred = u + dt*v + 0.5*dt²*(1-2β)*a
        v_pred = v + dt*(1-γ)*a
        """
        self._acc_old = self.acc.copy()
        self._vel_old = self.vel.copy()
        self._u_pred = u + dt * self.vel + 0.5 * dt**2 * (1.0 - 2.0 * self.beta) * self.acc
        self._v_pred = self.vel + dt * (1.0 - self.gamma) * self.acc
        return self._u_pred.copy()

    def correct(self, u: np.ndarray, du: np.ndarray, dt: float) -> None:
        """Newmark 補正子.

        Newton反復で得られた du を基に加速度・速度を更新する。
        """
        if dt < 1e-30:
            return
        c0 = 1.0 / (self.beta * dt**2)
        # 加速度 = c0 * (u_new - u_pred)
        self.acc = c0 * (u - self._u_pred)
        # 速度 = v_pred + dt * gamma * a_new
        self.vel = self._v_pred + dt * self.gamma * self.acc

    def effective_stiffness(self, K: sp.csr_matrix, dt: float) -> sp.csr_matrix:
        """有効剛性行列.

        K_eff = K + (1-α_m)*c0*M + (1-α_f)*c1*C
        """
        if dt < 1e-30:
            return K
        c0 = 1.0 / (self.beta * dt**2)
        c1 = self.gamma / (self.beta * dt)
        K_eff = K + (1.0 - self.alpha_m) * c0 * self.M
        if self.C is not None:
            K_eff = K_eff + (1.0 - self.alpha_f) * c1 * self.C
        return K_eff

    def effective_residual(self, R: np.ndarray, dt: float) -> np.ndarray:
        """有効残差.

        R_eff = R + f_inertia + f_damping
        f_inertia = M @ a_{n+1-α_m}
        f_damping = C @ v_{n+1-α_f}
        """
        R_eff = R.copy()
        # 慣性力: a_{n+1-α_m} = (1-α_m)*a_{n+1} + α_m*a_n
        a_mid = (1.0 - self.alpha_m) * self.acc + self.alpha_m * self._acc_old
        R_eff += self.M @ a_mid
        if self.C is not None:
            # 減衰力: v_{n+1-α_f} = (1-α_f)*v_{n+1} + α_f*v_n
            v_mid = (1.0 - self.alpha_f) * self.vel + self.alpha_f * self._vel_old
            R_eff += self.C @ v_mid
        return R_eff

    def checkpoint(self) -> None:
        """速度・加速度のチェックポイントを保存."""
        self._vel_ckpt = self.vel.copy()
        self._acc_ckpt = self.acc.copy()

    def restore_checkpoint(self) -> None:
        """速度・加速度をチェックポイントから復元."""
        if self._vel_ckpt is not None:
            self.vel = self._vel_ckpt.copy()
        if self._acc_ckpt is not None:
            self.acc = self._acc_ckpt.copy()

    def process(self, input_data: TimeIntegrationInput) -> TimeIntegrationOutput:
        u_pred = self.predict(input_data.u, input_data.dt)
        return TimeIntegrationOutput(u=u_pred)


def create_time_integration_strategy(
    *,
    mass_matrix: sp.csr_matrix | np.ndarray | None = None,
    damping_matrix: sp.csr_matrix | np.ndarray | None = None,
    dt_physical: float = 0.0,
    rho_inf: float = 0.9,
    velocity: np.ndarray | None = None,
    acceleration: np.ndarray | None = None,
) -> QuasiStaticProcess | GeneralizedAlphaProcess:
    """solver_ncp.py の動的解析初期化ロジックを Strategy に移譲するファクトリ.

    solver_ncp.py (lines 1659-1676) の分岐ロジックを再現する。

    Args:
        mass_matrix: 質量行列（None → 準静的）
        damping_matrix: 減衰行列（オプション）
        dt_physical: 物理時間刻み
        rho_inf: Generalized-α スペクトル半径（0.0-1.0）
        velocity: 初期速度
        acceleration: 初期加速度

    Returns:
        QuasiStaticProcess または GeneralizedAlphaProcess
    """
    is_dynamic = mass_matrix is not None and dt_physical > 0.0
    if not is_dynamic:
        return QuasiStaticProcess()

    strategy = GeneralizedAlphaProcess(
        mass_matrix=mass_matrix,
        damping_matrix=damping_matrix,
        rho_inf=rho_inf,
    )
    strategy.set_initial_state(velocity=velocity, acceleration=acceleration)
    return strategy
