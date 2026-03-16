"""ソルバー可変状態の集約（プライベート）.

SolverState を新パッケージに移植。
xkep_cae_deprecated/process/strategies/solver_state.py からの移植。
deprecated 依存を除去し、duck typing で簡素化。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class _GraphSnapshotList:
    """接触グラフスナップショットの簡易リスト.

    ContactGraphHistory の軽量代替。add_snapshot メソッドのみ提供。
    """

    snapshots: list[object] = field(default_factory=list)

    def add_snapshot(self, graph: object) -> None:
        """スナップショットを追加する."""
        self.snapshots.append(graph)


@dataclass(frozen=True)
class SolverState:
    """ソルバーの全可変状態.

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
    graph_history: _GraphSnapshotList = field(default_factory=_GraphSnapshotList)

    # --- 接線予測用 ---
    u_prev_converged: np.ndarray | None = None
    delta_frac_prev: float = 0.0

    # --- チェックポイント ---
    _u_ckpt: np.ndarray | None = None
    _lam_ckpt: np.ndarray | None = None
    _u_ref_ckpt: np.ndarray | None = None
    _ul_frac_base_ckpt: float = 0.0

    def _set(self, name: str, value: object) -> None:
        """frozen バイパス: 内部状態更新用."""
        object.__setattr__(self, name, value)

    def save_checkpoint(self) -> None:
        """現在の状態をチェックポイントに保存."""
        self._set("_u_ckpt", self.u.copy())
        self._set("_lam_ckpt", self.lam_all.copy())
        self._set("_u_ref_ckpt", self.u_ref.copy())
        self._set("_ul_frac_base_ckpt", self.ul_frac_base)

    def restore_checkpoint(self) -> None:
        """チェックポイントから状態を復元."""
        if self._u_ckpt is None:
            raise RuntimeError("チェックポイントが保存されていません")
        self._set("u", self._u_ckpt.copy())
        self._set("lam_all", self._lam_ckpt.copy())
        self._set("u_ref", self._u_ref_ckpt.copy())
        self._set("ul_frac_base", self._ul_frac_base_ckpt)
        self._set("delta_frac_prev", 0.0)

    def ensure_lam_size(self, n_pairs: int) -> None:
        """ペア数拡張に対応して lam_all を拡張."""
        if len(self.lam_all) < n_pairs:
            old_n = len(self.lam_all)
            lam_new = np.zeros(n_pairs)
            lam_new[:old_n] = self.lam_all
            self._set("lam_all", lam_new)

    def build_u_output(self, ul_assembler: object | None) -> np.ndarray:
        """UL込みの最終変位ベクトルを構築."""
        if ul_assembler is not None:
            return ul_assembler.u_total_accum + self.u
        return self.u
