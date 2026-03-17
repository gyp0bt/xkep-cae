"""ソルバー可変状態の集約（プライベート）.

SolverStateOutput を新パッケージに移植。
__xkep_cae_deprecated/process/strategies/solver_state.py からの移植。
deprecated 依存を除去し、duck typing で簡素化。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from xkep_cae.core import ProcessMeta, SolverProcess

_setattr = object.__setattr__


@dataclass(frozen=True)
class SolverStateOutput:
    """ソルバーの全可変状態（純粋データ）."""

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
    graph_snapshots: list[object] = field(default_factory=list)

    # --- 接線予測用 ---
    u_prev_converged: np.ndarray | None = None
    delta_frac_prev: float = 0.0

    # --- チェックポイント ---
    _u_ckpt: np.ndarray | None = None
    _lam_ckpt: np.ndarray | None = None
    _u_ref_ckpt: np.ndarray | None = None
    _ul_frac_base_ckpt: float = 0.0


def _state_set(state: SolverStateOutput, name: str, value: object) -> None:
    """frozen SolverStateOutput のフィールドを更新する."""
    _setattr(state, name, value)


def _save_checkpoint(state: SolverStateOutput) -> None:
    """現在の状態をチェックポイントに保存."""
    _setattr(state, "_u_ckpt", state.u.copy())
    _setattr(state, "_lam_ckpt", state.lam_all.copy())
    _setattr(state, "_u_ref_ckpt", state.u_ref.copy())
    _setattr(state, "_ul_frac_base_ckpt", state.ul_frac_base)


def _restore_checkpoint(state: SolverStateOutput) -> None:
    """チェックポイントから状態を復元."""
    if state._u_ckpt is None:
        raise RuntimeError("チェックポイントが保存されていません")
    _setattr(state, "u", state._u_ckpt.copy())
    _setattr(state, "lam_all", state._lam_ckpt.copy())
    _setattr(state, "u_ref", state._u_ref_ckpt.copy())
    _setattr(state, "ul_frac_base", state._ul_frac_base_ckpt)
    _setattr(state, "delta_frac_prev", 0.0)


def _ensure_lam_size(state: SolverStateOutput, n_pairs: int) -> None:
    """ペア数拡張に対応して lam_all を拡張."""
    if len(state.lam_all) < n_pairs:
        old_n = len(state.lam_all)
        lam_new = np.zeros(n_pairs)
        lam_new[:old_n] = state.lam_all
        _setattr(state, "lam_all", lam_new)


def _build_u_output(state: SolverStateOutput, ul_assembler: object | None) -> np.ndarray:
    """UL込みの最終変位ベクトルを構築."""
    if ul_assembler is not None:
        return ul_assembler.u_total_accum + state.u
    return state.u


@dataclass(frozen=True)
class SolverStateInitInput:
    """SolverStateOutput 初期化の入力."""

    ndof: int
    node_coords: np.ndarray
    n_pairs: int = 0


@dataclass(frozen=True)
class SolverStateInitOutput:
    """SolverStateOutput 初期化の出力."""

    state: SolverStateOutput


class SolverStateInitProcess(
    SolverProcess[SolverStateInitInput, SolverStateInitOutput],
):
    """SolverStateOutput の初期化 Process.

    自由度数・節点座標・ペア数からゼロ初期状態を生成する。
    """

    meta = ProcessMeta(
        name="SolverStateInit",
        module="solve",
        version="1.0.0",
        document_path="docs/contact_friction.md",
    )

    def process(self, input_data: SolverStateInitInput) -> SolverStateInitOutput:
        state = SolverStateOutput(
            u=np.zeros(input_data.ndof),
            lam_all=np.zeros(max(input_data.n_pairs, 1)),
            u_ref=np.zeros(input_data.ndof),
            node_coords_ref=input_data.node_coords.copy(),
        )
        return SolverStateInitOutput(state=state)
