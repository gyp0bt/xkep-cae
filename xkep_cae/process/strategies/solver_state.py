"""ソルバー可変状態の集約.

solver_smooth_penalty.py のモノリシック関数から分離した状態管理。
Newton-Raphson + Uzawa + 適応荷重増分の全可変状態を一箇所に集約する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from xkep_cae.contact.diagnostics import ConvergenceDiagnostics, NCPSolveResult
from xkep_cae.contact.graph import ContactGraphHistory


@dataclass
class SolverState:
    """ソルバーの全可変状態.

    solver_smooth_penalty.py 内でローカル変数として散在していた状態を集約。
    チェックポイント保存/復元もこのクラスが管理する。
    """

    # --- 主要変数 ---
    u: np.ndarray
    lam_all: np.ndarray
    u_ref: np.ndarray

    # --- 参照配置（UL時に更新） ---
    node_coords_ref: np.ndarray

    # --- 荷重パラメータ ---
    load_frac_prev: float = 0.0
    ul_frac_base: float = 0.0

    # --- カウンタ ---
    step_display: int = 0
    total_newton: int = 0
    prev_n_active: int = 0

    # --- 履歴 ---
    load_history: list[float] = field(default_factory=list)
    disp_history: list[np.ndarray] = field(default_factory=list)
    contact_force_history: list[float] = field(default_factory=list)
    graph_history: ContactGraphHistory = field(default_factory=ContactGraphHistory)

    # --- 接線予測用 ---
    u_prev_converged: np.ndarray | None = None
    delta_frac_prev: float = 0.0

    # --- チェックポイント ---
    _u_ckpt: np.ndarray | None = None
    _lam_ckpt: np.ndarray | None = None
    _u_ref_ckpt: np.ndarray | None = None
    _ul_frac_base_ckpt: float = 0.0

    def save_checkpoint(self) -> None:
        """現在の状態をチェックポイントに保存."""
        self._u_ckpt = self.u.copy()
        self._lam_ckpt = self.lam_all.copy()
        self._u_ref_ckpt = self.u_ref.copy()
        self._ul_frac_base_ckpt = self.ul_frac_base

    def restore_checkpoint(self) -> None:
        """チェックポイントから状態を復元."""
        if self._u_ckpt is None:
            raise RuntimeError("チェックポイントが保存されていません")
        self.u = self._u_ckpt.copy()
        self.lam_all = self._lam_ckpt.copy()
        self.u_ref = self._u_ref_ckpt.copy()
        self.ul_frac_base = self._ul_frac_base_ckpt
        self.delta_frac_prev = 0.0

    def ensure_lam_size(self, n_pairs: int) -> None:
        """ペア数拡張に対応して lam_all を拡張."""
        if len(self.lam_all) < n_pairs:
            old_n = len(self.lam_all)
            lam_new = np.zeros(n_pairs)
            lam_new[:old_n] = self.lam_all
            self.lam_all = lam_new

    def build_result(
        self,
        *,
        converged: bool,
        ul_assembler: object | None,
        time_strategy: object,
        diagnostics: ConvergenceDiagnostics | None = None,
    ) -> NCPSolveResult:
        """NCPSolveResult を構築."""
        _dynamics = hasattr(time_strategy, "is_dynamic") and time_strategy.is_dynamic
        _ul = ul_assembler is not None
        _u_out = ul_assembler.u_total_accum + self.u if _ul else self.u

        return NCPSolveResult(
            u=_u_out,
            lambdas=self.lam_all,
            converged=converged,
            n_increments=self.step_display,
            total_newton_iterations=self.total_newton,
            n_active_final=0,
            load_history=self.load_history,
            displacement_history=self.disp_history,
            contact_force_history=self.contact_force_history,
            graph_history=self.graph_history,
            diagnostics=diagnostics,
            velocity=time_strategy.vel.copy() if _dynamics else None,
            acceleration=time_strategy.acc.copy() if _dynamics else None,
        )
