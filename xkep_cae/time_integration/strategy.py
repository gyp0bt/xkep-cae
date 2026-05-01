"""TimeIntegration Strategy 具象実装.

TimeIntegrationStrategy Protocol に従い、時間積分方法を実装する Process 群。

3クラス構成:
- QuasiStaticProcess: 準静的（荷重制御）— 時間積分なし
- GeneralizedAlphaProcess: Generalized-α 動的解析（Chung & Hulbert 1993）
- ExplicitCentralDifferenceProcess: 陽的中央差分（status-377 Phase 1）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.core import ProcessMeta, SolverProcess

# ── Input / Output ─────────────────────────────────────────


@dataclass(frozen=True)
class TimeIntegrationInput:
    """TimeIntegration Strategy の入力."""

    u: np.ndarray
    du: np.ndarray
    dt: float


@dataclass(frozen=True)
class TimeIntegrationOutput:
    """TimeIntegration Strategy の出力."""

    u: np.ndarray


# ── 具象 Process ──────────────────────────────────────────


class QuasiStaticProcess(SolverProcess[TimeIntegrationInput, TimeIntegrationOutput]):
    """準静的解析（荷重制御）.

    時間積分なし。K_eff = K, R_eff = R をそのまま返す。
    predict/correct は identity 操作。
    """

    meta = ProcessMeta(
        name="QuasiStatic",
        module="solve",
        version="1.0.0",
        document_path="docs/time_integration.md",
    )

    @property
    def is_dynamic(self) -> bool:
        """動的解析かどうか."""
        return False

    @property
    def vel(self) -> np.ndarray:
        """速度: 準静的では空ベクトル."""
        return np.zeros(0)

    @property
    def acc(self) -> np.ndarray:
        """加速度: 準静的では空ベクトル."""
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


class GeneralizedAlphaProcess(
    SolverProcess[TimeIntegrationInput, TimeIntegrationOutput],
):
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
        version="1.0.0",
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
        self.acc = c0 * (u - self._u_pred)
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
        a_mid = (1.0 - self.alpha_m) * self.acc + self.alpha_m * self._acc_old
        R_eff += self.M @ a_mid
        if self.C is not None:
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


class ExplicitCentralDifferenceProcess(
    SolverProcess[TimeIntegrationInput, TimeIntegrationOutput],
):
    """陽的中央差分法（status-377 Phase 1）.

    集中質量 M_lump を用いた陽解法時間積分:

        a_n     = M_lump^{-1} · (F_ext_n − F_int(u_n) − C·v_{n−1/2})
        v_{n+1/2} = v_{n−1/2} + dt · a_n
        u_{n+1} = u_n + dt · v_{n+1/2}

    NR 反復を要さず、各時間ステップで F_int / F_ext の 1 回評価で進む。
    Courant 安定条件 dt ≤ dt_c = 2/ω_max（接線剛性 K の最大固有振動数）が必要。

    19 本以上の K_c x/z カップリング不整合（status-344）に対し、陰解法 NR の
    active 集合振動を時間積分自体で抑制することを意図する。

    **Phase 1 (status-377) 制約**:
    - Process 単体実装 + 単体テスト + 設計仕様のみ
    - solver path への配線は Phase 2（次 status）で実施
    - default `solver_mode="implicit"` で既存挙動不変

    集中質量近似:
        M_lump[i,i] = Σ_j M_consistent[i,j]（行和ロンピング）

    質量スケーリング（status-379 Phase 3, 候補 (h1)）:
        M_lump_scaled = β² · M_lump（β >= 1.0）で集中質量を倍化することで
        Δt_c → β·Δt_c に拡大。準静的近似 ε = E_kin/E_strain ≪ 1 を保つ範囲で
        β > 1 を指定。Belytschko §6.4.2。`set_mass_scaling_beta()` で
        実行時に β を上方更新（auto-tune 用）。

    Protocol 適合のため `predict / correct / effective_stiffness / effective_residual`
    は提供するが、陽解法は NR 反復を経ず `step()` で 1 step 完結する。
    """

    meta = ProcessMeta(
        name="ExplicitCentralDifference",
        module="solve",
        version="1.0.0",
        document_path="docs/time_integration_explicit.md",
    )

    def __init__(
        self,
        mass_matrix: sp.csr_matrix | np.ndarray,
        *,
        damping_matrix: sp.csr_matrix | np.ndarray | None = None,
        mass_lumping: str = "row_sum",
        mass_scaling_beta: float = 1.0,
        mass_proportional_damping_alpha: float = 0.0,
    ) -> None:
        """陽解法時間積分 Process を初期化.

        Args:
            mass_matrix: 一貫質量 M または既に集中質量化された対角行列
            damping_matrix: 減衰行列 C（None で減衰なし）
            mass_lumping: "row_sum"（行和）/ "diagonal"（対角抽出）/ "none"
                既に集中質量化済みなら "none" を指定する
            mass_scaling_beta: 質量スケーリング係数 β（status-379 候補 (h1)）。
                M_lump → β² · M_lump で集中質量を倍化することで Δt_c → β·Δt_c。
                準静的近似 ε = E_kin / E_strain ≪ 1 を保つ範囲で β > 1.0 を指定。
                既定 1.0（スケーリング無効）。Belytschko §6.4.2。
            mass_proportional_damping_alpha: 質量比例 Rayleigh damping 係数 α
                （status-382 候補 (p3)）。`C = α · M_lump` を等価適用し、運動方程式は
                $a_n = M_\\mathrm{lump}^{-1}(F_\\mathrm{ext} - F_\\mathrm{int}) - α \\cdot v_{n-1/2}$
                となる。M に独立に α が damping 率として作用するため Courant 安定性
                および β スケーリングと無関係に動的緩和を加速する。既定 0.0（無効）。
        """
        if mass_scaling_beta < 1.0:
            raise ValueError(
                f"mass_scaling_beta must be >= 1.0 (got {mass_scaling_beta}). "
                "β < 1 shrinks Δt_c and provides no benefit."
            )
        if mass_proportional_damping_alpha < 0.0:
            raise ValueError(
                f"mass_proportional_damping_alpha must be >= 0.0 "
                f"(got {mass_proportional_damping_alpha})"
            )
        M_consistent = sp.csr_matrix(mass_matrix) if not sp.issparse(mass_matrix) else mass_matrix
        self.M = M_consistent
        self._mass_lumping = mass_lumping
        self._M_lump_raw = self._lump_mass(M_consistent, mass_lumping)
        self.mass_scaling_beta = float(mass_scaling_beta)
        self.M_lump = (self.mass_scaling_beta**2) * self._M_lump_raw
        self.M_lump_inv = self._invert_diagonal(self.M_lump)
        self.mass_proportional_damping_alpha = float(mass_proportional_damping_alpha)

        self.C = (
            sp.csr_matrix(damping_matrix)
            if damping_matrix is not None and not sp.issparse(damping_matrix)
            else damping_matrix
        )

        ndof = self.M.shape[0]
        self.vel = np.zeros(ndof)
        self.acc = np.zeros(ndof)
        self._vel_old = np.zeros(ndof)
        self._acc_old = np.zeros(ndof)
        self._u_pred = np.zeros(ndof)
        self._vel_ckpt: np.ndarray | None = None
        self._acc_ckpt: np.ndarray | None = None

    def set_mass_scaling_beta(self, beta: float, *, rescale_state: bool = True) -> None:
        """質量スケーリング係数 β を実行時に更新する（status-379 auto-tune 用）.

        ExplicitDynamicProcess の Courant 監視で要求 β を逆算した際に呼ばれる。
        β² · M_lump_raw で集中質量を再計算し、逆数も更新する。

        Args:
            beta: 新しい β（>= 1.0）。既存値以下なら何もしない（β は単調増加のみ）。
            rescale_state: True (default) で v/a を **運動エネルギー保存** で
                リスケールする（v ← v·(β_old/β_new), a ← a·(β_old/β_new)²）。
                これにより β 上方更新時に KE = 0.5·M·v² が β² 倍 spuriously
                injected される **status-381 h-bug-1** を防ぐ。
                False は後方互換用（テスト目的等）。
        """
        if beta < 1.0:
            raise ValueError(f"mass_scaling_beta must be >= 1.0 (got {beta})")
        if beta <= self.mass_scaling_beta:
            return
        beta_old = self.mass_scaling_beta
        self.mass_scaling_beta = float(beta)
        self.M_lump = (self.mass_scaling_beta**2) * self._M_lump_raw
        self.M_lump_inv = self._invert_diagonal(self.M_lump)
        if rescale_state:
            ratio = beta_old / self.mass_scaling_beta
            self.vel = self.vel * ratio
            self.acc = self.acc * (ratio * ratio)

    @staticmethod
    def _lump_mass(M: sp.csr_matrix, lumping: str) -> np.ndarray:
        """質量行列を集中質量に変換し対角ベクトルを返す."""
        if lumping == "row_sum":
            # 行和ロンピング: m_i = Σ_j M[i,j]
            return np.asarray(M.sum(axis=1)).flatten()
        if lumping == "diagonal":
            return np.asarray(M.diagonal()).flatten()
        if lumping == "none":
            return np.asarray(M.diagonal()).flatten()
        raise ValueError(f"unknown mass_lumping: {lumping!r}")

    @staticmethod
    def _invert_diagonal(m_lump: np.ndarray) -> np.ndarray:
        """対角質量の逆数を返す（0 要素は 0 のまま、固定 DOF 想定）."""
        inv = np.zeros_like(m_lump)
        nonzero = m_lump > 0.0
        inv[nonzero] = 1.0 / m_lump[nonzero]
        return inv

    @property
    def is_dynamic(self) -> bool:
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

    def critical_dt(self, k_max_eigenvalue: float) -> float:
        """Courant 臨界時間刻み dt_c = 2/ω_max を返す.

        Args:
            k_max_eigenvalue: 接線剛性 K の最大一般化固有値 λ_max(K, M)

        Returns:
            dt_c [s]。安全側で 0.9·dt_c 程度を実運用 dt として推奨。
        """
        if k_max_eigenvalue <= 0.0:
            return float("inf")
        omega_max = float(np.sqrt(k_max_eigenvalue))
        return 2.0 / omega_max

    def step(
        self,
        u: np.ndarray,
        f_ext: np.ndarray,
        f_int: np.ndarray,
        dt: float,
        *,
        fixed_dofs: np.ndarray | None = None,
    ) -> np.ndarray:
        """陽的中央差分 1 ステップを実行し u_{n+1} を返す.

        集中質量 M_lump と直接 a_n = M_lump^{-1}·(F_ext − F_int − C·v) を計算し、
        v_{n+1/2}・u_{n+1} を更新する。NR 反復は要さない。

        Args:
            u: 現時刻変位 u_n（in-place 更新しない、戻り値が u_{n+1}）
            f_ext: 外力ベクトル F_ext_n
            f_int: 内力ベクトル F_int(u_n) = K_struct·u_n + f_contact 等
            dt: 時間刻み
            fixed_dofs: 拘束 DOF（acc / vel をゼロに固定する）

        Returns:
            u_{n+1}（新しい配列）

        Note:
            Courant 条件 dt ≤ dt_c を満たさない場合、解は数値的に発散する。
            呼び出し側で `critical_dt()` から推定した dt_c に対し dt = 0.9·dt_c
            程度を与えること。
        """
        if dt <= 0.0:
            return u.copy()

        residual = f_ext - f_int
        if self.C is not None:
            residual = residual - self.C @ self.vel

        a_n = self.M_lump_inv * residual
        # 質量比例 Rayleigh damping（status-382 候補 (p3)）:
        # C = α·M を等価適用し a -= α·v_{n−1/2}。M に独立に α が damping 率として
        # 作用するため、β スケーリングおよび Courant 安定性に影響しない。
        if self.mass_proportional_damping_alpha > 0.0:
            a_n = a_n - self.mass_proportional_damping_alpha * self.vel
        if fixed_dofs is not None and len(fixed_dofs) > 0:
            a_n[fixed_dofs] = 0.0

        self._acc_old = self.acc.copy()
        self._vel_old = self.vel.copy()
        self.acc = a_n
        self.vel = self.vel + dt * a_n
        if fixed_dofs is not None and len(fixed_dofs) > 0:
            self.vel[fixed_dofs] = 0.0

        return u + dt * self.vel

    def predict(self, u: np.ndarray, dt: float) -> np.ndarray:
        """陽解法用予測子: u_pred = u + dt·v_n + 0.5·dt²·a_n（Verlet 形）.

        NR 反復を経由しない陽解法では `step()` の方が直接的だが、Protocol 適合と
        将来の対称化（陰陽混合）のため Verlet 予測子を提供する。
        """
        self._acc_old = self.acc.copy()
        self._vel_old = self.vel.copy()
        self._u_pred = u + dt * self.vel + 0.5 * dt**2 * self.acc
        return self._u_pred.copy()

    def correct(self, u: np.ndarray, du: np.ndarray, dt: float) -> None:
        """陽解法補正子: 中央差分から v_{n+1/2} を逆算.

        u_{n+1} = u_n + dt·v_{n+1/2} より v_{n+1/2} = (u_{n+1} − u_n) / dt。
        """
        if dt < 1e-30:
            return
        self.vel = (u - (self._u_pred - 0.5 * dt**2 * self._acc_old - dt * self._vel_old)) / dt

    def effective_stiffness(self, K: sp.csr_matrix, dt: float) -> sp.csr_matrix:
        """有効剛性: 陽解法では K_eff = M_lump/dt² の対角行列のみ.

        K の項を捨て、集中質量で割ることで NR 1 反復が陽的中央差分と等価になる。
        ただし default 運用では NR を経由せず `step()` で直接前進する。
        """
        if dt < 1e-30:
            return K
        ndof = self.M.shape[0]
        return sp.diags(self.M_lump / (dt * dt), format="csr", shape=(ndof, ndof))

    def effective_residual(self, R: np.ndarray, dt: float) -> np.ndarray:
        """有効残差: 陽解法では R_eff = R（外力 − 内力）+ 慣性項."""
        R_eff = R.copy()
        if self.C is not None:
            R_eff -= self.C @ self.vel
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


# ── ファクトリ ─────────────────────────────────────────────


def _create_time_integration_strategy(
    *,
    mass_matrix: sp.csr_matrix | np.ndarray | None = None,
    damping_matrix: sp.csr_matrix | np.ndarray | None = None,
    dt_physical: float = 0.0,
    rho_inf: float = 0.9,
    velocity: np.ndarray | None = None,
    acceleration: np.ndarray | None = None,
    solver_mode: str = "implicit",
    mass_lumping: str = "row_sum",
    mass_scaling_beta: float = 1.0,
    mass_proportional_damping_alpha: float = 0.0,
) -> QuasiStaticProcess | GeneralizedAlphaProcess | ExplicitCentralDifferenceProcess:
    """時間積分 Strategy ファクトリ.

    Args:
        mass_matrix: 質量行列（None → 準静的）
        damping_matrix: 減衰行列（オプション）
        dt_physical: 物理時間刻み
        rho_inf: Generalized-α スペクトル半径（0.0-1.0）
        velocity: 初期速度
        acceleration: 初期加速度
        solver_mode: "implicit"（default、Generalized-α）/ "explicit"（中央差分）
        mass_lumping: 陽解法時の集中質量化方式（"row_sum" / "diagonal" / "none"）
        mass_scaling_beta: 陽解法時の質量スケーリング係数（status-379 候補 (h1)）。
            既定 1.0（スケーリング無効）。
        mass_proportional_damping_alpha: 陽解法時の質量比例 Rayleigh damping
            係数（status-382 候補 (p3)）。既定 0.0（無効）。

    Returns:
        QuasiStaticProcess / GeneralizedAlphaProcess / ExplicitCentralDifferenceProcess
    """
    is_dynamic = mass_matrix is not None and dt_physical > 0.0
    if not is_dynamic:
        return QuasiStaticProcess()

    if solver_mode == "explicit":
        strategy_explicit = ExplicitCentralDifferenceProcess(
            mass_matrix=mass_matrix,
            damping_matrix=damping_matrix,
            mass_lumping=mass_lumping,
            mass_scaling_beta=mass_scaling_beta,
            mass_proportional_damping_alpha=mass_proportional_damping_alpha,
        )
        strategy_explicit.set_initial_state(velocity=velocity, acceleration=acceleration)
        return strategy_explicit

    strategy = GeneralizedAlphaProcess(
        mass_matrix=mass_matrix,
        damping_matrix=damping_matrix,
        rho_inf=rho_inf,
    )
    strategy.set_initial_state(velocity=velocity, acceleration=acceleration)
    return strategy
