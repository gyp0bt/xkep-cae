"""接触ペア・接触状態のデータ構造 + ユーティリティ関数.

frozen dataclass（純データ）+ モジュールレベル関数を定義。
ビジネスロジック（detect, update_geometry 等）は _manager_process.py の
Process クラスに実装。プライベートモジュール（C16 準拠）。

status-205: dataclass メソッド完全除去（純データ化）。
旧メソッド → 移行先:
  _ContactStateOutput.copy()       → _copy_state()
  _ContactStateOutput._evolve()    → _evolve_state()
  _ContactPairOutput.search_radius → _pair_search_radius()
  _ContactPairOutput.is_active()   → _is_active_pair()
  _ContactPairOutput._evolve()     → _evolve_pair()
  _ContactManagerInput.n_pairs     → _n_pairs()
  _ContactManagerInput.n_active    → _n_active()
  _ContactManagerInput.add_pair()  → AddPairProcess
  _ContactManagerInput.reset_all() → ResetAllPairsProcess
  _ContactManagerInput.get_active_pairs() → _get_active_pairs()
  _ContactManagerInput.detect_candidates() → DetectCandidatesProcess
  _ContactManagerInput.update_geometry()   → UpdateGeometryProcess
  _ContactManagerInput._update_active_set_state() → UpdateGeometryProcess 内部
  _ContactManagerInput.initialize_penalty() → InitializePenaltyProcess
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xkep_cae.contact._types import ContactStatus

# ── frozen dataclass 定義（純データのみ） ──


@dataclass(frozen=True)
class _ContactStateOutput:
    """1接触点の状態変数."""

    s: float = 0.0
    t: float = 0.0
    gap: float = 0.0
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tangent1: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tangent2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    lambda_n: float = 0.0
    k_pen: float = 0.0
    k_t: float = 0.0
    p_n: float = 0.0
    z_t: np.ndarray = field(default_factory=lambda: np.zeros(2))
    q_trial_norm: float = 0.0
    status: ContactStatus = ContactStatus.INACTIVE
    stick: bool = True
    dissipation: float = 0.0
    coating_compression: float = 0.0
    coating_compression_prev: float = 0.0
    coating_z_t: np.ndarray = field(default_factory=lambda: np.zeros(2))
    coating_stick: bool = True
    coating_q_trial_norm: float = 0.0
    coating_dissipation: float = 0.0
    gp_z_t: list[np.ndarray] | None = None
    gp_stick: list[bool] | None = None
    gp_q_trial_norm: list[float] | None = None


@dataclass(frozen=True)
class _ContactPairOutput:
    """接触ペアの定義."""

    elem_a: int
    elem_b: int
    nodes_a: np.ndarray
    nodes_b: np.ndarray
    state: _ContactStateOutput = field(default_factory=_ContactStateOutput)
    radius_a: float = 0.0
    radius_b: float = 0.0
    core_radius_a: float = 0.0
    core_radius_b: float = 0.0


@dataclass(frozen=True)
class _ContactConfigInput:
    """接触解析の設定."""

    k_pen_scale: float = 0.1
    k_pen_mode: str = "beam_ei"
    beam_E: float = 0.0
    beam_I: float = 0.0
    beam_A: float = 0.0
    k_t_ratio: float = 0.5
    mu: float = 0.3
    g_on: float = 0.0
    g_off: float = 1e-6
    n_outer_max: int = 5
    tol_geometry: float = 1e-6
    mu_ramp_steps: int = 0
    use_line_search: bool = False
    line_search_max_steps: int = 5
    merit_alpha: float = 1.0
    merit_beta: float = 1.0
    use_geometric_stiffness: bool = True
    use_pdas: bool = False
    tol_penetration_ratio: float = 0.01
    penalty_growth_factor: float = 2.0
    k_pen_max: float = 1e12
    staged_activation_steps: int = 0
    elem_layer_map: dict[int, int] | None = None
    use_modified_newton: bool = False
    modified_newton_refresh: int = 5
    contact_damping: float = 1.0
    k_pen_scaling: str = "linear"
    contact_tangent_mode: str = "full"
    contact_tangent_scale: float = 1.0
    al_relaxation: float = 1.0
    adaptive_omega: bool = False
    omega_min: float = 0.01
    omega_max: float = 0.3
    omega_growth: float = 2.0
    preserve_inactive_lambda: bool = False
    linear_solver: str = "auto"
    iterative_tol: float = 1e-10
    ilu_drop_tol: float = 1e-4
    gmres_dof_threshold: int = 2000
    no_deactivation_within_step: bool = False
    monolithic_geometry: bool = False
    line_contact: bool = False
    n_gauss: int = 3
    n_gauss_auto: bool = False
    consistent_st_tangent: bool = False
    use_ncp: bool = False
    ncp_type: str = "fb"
    ncp_reg: float = 1e-12
    ncp_block_preconditioner: bool = False
    exclude_same_layer: bool = False
    use_mortar: bool = False
    midpoint_prescreening: bool = True
    prescreening_margin: float = 0.0
    lambda_n_max_factor: float = 0.0
    augmented_threshold: int = 20
    saddle_regularization: float = 0.0
    ncp_active_threshold: float = 0.0
    lambda_relaxation: float = 1.0
    lambda_warmstart_neighbor: bool = False
    chattering_window: int = 0
    adaptive_timestepping: bool = False
    dt_grow_factor: float = 1.5
    dt_shrink_factor: float = 0.5
    dt_grow_attempt_threshold: int = 5
    dt_shrink_attempt_threshold: int = 15
    dt_contact_change_threshold: float = 0.3
    dt_min_fraction: float = 0.0
    dt_max_fraction: float = 0.0
    use_amg_preconditioner: bool = False
    k_pen_continuation: bool = False
    k_pen_continuation_start: float = 0.1
    k_pen_continuation_steps: int = 3
    residual_scaling: bool = False
    contact_force_ramp: bool = False
    contact_force_ramp_iters: int = 5
    adjust_initial_penetration: bool = True
    position_tolerance: float = 0.0
    coating_stiffness: float = 0.0
    coating_damping: float = 0.0
    coating_mu: float = 0.0
    coating_k_t_ratio: float = 0.5
    contact_compliance: float = 0.0
    smoothing_delta: float = 0.0
    exact_tangent: bool = False  # 厳密接線（動的 c0*M 正則化時に有効）


@dataclass(frozen=True)
class _ContactManagerInput:
    """接触ペアの管理（純データ）.

    全操作は _manager_process.py の Process クラスで行う。
    """

    pairs: list[_ContactPairOutput] = field(default_factory=list)
    config: _ContactConfigInput = field(default_factory=_ContactConfigInput)


# ── モジュールレベルユーティリティ関数 ──


def _evolve_state(state: _ContactStateOutput, **kwargs: Any) -> _ContactStateOutput:
    """_ContactStateOutput の指定フィールドを更新した新インスタンスを返す."""
    updates: dict[str, Any] = {}
    for f in dataclasses.fields(state):
        updates[f.name] = kwargs.pop(f.name, getattr(state, f.name))
    if kwargs:
        raise TypeError(f"Unknown fields: {set(kwargs)}")
    return _ContactStateOutput(**updates)


def _evolve_pair(pair: _ContactPairOutput, **kwargs: Any) -> _ContactPairOutput:
    """_ContactPairOutput の指定フィールドを更新した新インスタンスを返す."""
    updates: dict[str, Any] = {}
    for f in dataclasses.fields(pair):
        updates[f.name] = kwargs.pop(f.name, getattr(pair, f.name))
    if kwargs:
        raise TypeError(f"Unknown fields: {set(kwargs)}")
    return _ContactPairOutput(**updates)


def _copy_state(state: _ContactStateOutput) -> _ContactStateOutput:
    """_ContactStateOutput の深いコピーを返す."""
    return _ContactStateOutput(
        s=state.s,
        t=state.t,
        gap=state.gap,
        normal=state.normal.copy(),
        tangent1=state.tangent1.copy(),
        tangent2=state.tangent2.copy(),
        lambda_n=state.lambda_n,
        k_pen=state.k_pen,
        k_t=state.k_t,
        p_n=state.p_n,
        z_t=state.z_t.copy(),
        q_trial_norm=state.q_trial_norm,
        status=state.status,
        stick=state.stick,
        dissipation=state.dissipation,
        coating_compression=state.coating_compression,
        coating_compression_prev=state.coating_compression_prev,
        coating_z_t=state.coating_z_t.copy(),
        coating_stick=state.coating_stick,
        coating_q_trial_norm=state.coating_q_trial_norm,
        coating_dissipation=state.coating_dissipation,
        gp_z_t=[z.copy() for z in state.gp_z_t] if state.gp_z_t is not None else None,
        gp_stick=list(state.gp_stick) if state.gp_stick is not None else None,
        gp_q_trial_norm=list(state.gp_q_trial_norm) if state.gp_q_trial_norm is not None else None,
    )


def _pair_search_radius(pair: _ContactPairOutput) -> float:
    """探索半径: 断面半径の和."""
    return pair.radius_a + pair.radius_b


def _is_active_pair(pair: _ContactPairOutput) -> bool:
    """接触が有効か."""
    return pair.state.status != ContactStatus.INACTIVE


def _n_pairs(manager: _ContactManagerInput) -> int:
    """ペア数."""
    return len(manager.pairs)


def _n_active(manager: _ContactManagerInput) -> int:
    """有効な接触ペア数."""
    return sum(1 for p in manager.pairs if _is_active_pair(p))


def _get_active_pairs(manager: _ContactManagerInput) -> list[_ContactPairOutput]:
    """有効な接触ペアのリストを返す."""
    return [p for p in manager.pairs if _is_active_pair(p)]


def _make_pair(
    elem_a: int,
    elem_b: int,
    nodes_a: np.ndarray,
    nodes_b: np.ndarray,
    radius_a: float = 0.0,
    radius_b: float = 0.0,
    core_radius_a: float = 0.0,
    core_radius_b: float = 0.0,
) -> _ContactPairOutput:
    """新しい接触ペアを作成する."""
    return _ContactPairOutput(
        elem_a=elem_a,
        elem_b=elem_b,
        nodes_a=np.asarray(nodes_a, dtype=int),
        nodes_b=np.asarray(nodes_b, dtype=int),
        radius_a=radius_a,
        radius_b=radius_b,
        core_radius_a=core_radius_a,
        core_radius_b=core_radius_b,
    )
