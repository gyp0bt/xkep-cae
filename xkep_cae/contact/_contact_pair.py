"""接触ペア・接触状態のデータ構造 + ContactManager.

__xkep_cae_deprecated/contact/pair.py から移植。
プライベートモジュール（C16 準拠）。

依存: _broadphase, geometry/_compute のみ（deprecated 参照なし）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from xkep_cae.contact._broadphase import _broadphase_aabb
from xkep_cae.contact._types import ContactStatus
from xkep_cae.contact.geometry._compute import (
    _build_contact_frame_batch,
    _closest_point_segments_batch,
)


@dataclass
class _ContactState:
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

    def copy(self) -> _ContactState:
        """深いコピーを返す."""
        return _ContactState(
            s=self.s,
            t=self.t,
            gap=self.gap,
            normal=self.normal.copy(),
            tangent1=self.tangent1.copy(),
            tangent2=self.tangent2.copy(),
            lambda_n=self.lambda_n,
            k_pen=self.k_pen,
            k_t=self.k_t,
            p_n=self.p_n,
            z_t=self.z_t.copy(),
            q_trial_norm=self.q_trial_norm,
            status=self.status,
            stick=self.stick,
            dissipation=self.dissipation,
            coating_compression=self.coating_compression,
            coating_compression_prev=self.coating_compression_prev,
            coating_z_t=self.coating_z_t.copy(),
            coating_stick=self.coating_stick,
            coating_q_trial_norm=self.coating_q_trial_norm,
            coating_dissipation=self.coating_dissipation,
            gp_z_t=[z.copy() for z in self.gp_z_t] if self.gp_z_t is not None else None,
            gp_stick=list(self.gp_stick) if self.gp_stick is not None else None,
            gp_q_trial_norm=list(self.gp_q_trial_norm)
            if self.gp_q_trial_norm is not None
            else None,
        )


@dataclass
class _ContactPair:
    """接触ペアの定義."""

    elem_a: int
    elem_b: int
    nodes_a: np.ndarray
    nodes_b: np.ndarray
    state: _ContactState = field(default_factory=_ContactState)
    radius_a: float = 0.0
    radius_b: float = 0.0
    core_radius_a: float = 0.0
    core_radius_b: float = 0.0

    @property
    def search_radius(self) -> float:
        """探索半径: 断面半径の和."""
        return self.radius_a + self.radius_b

    def is_active(self) -> bool:
        """接触が有効か."""
        return self.state.status != ContactStatus.INACTIVE


@dataclass
class _ContactConfig:
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
    use_friction: bool = False
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
    dt_grow_iter_threshold: int = 5
    dt_shrink_iter_threshold: int = 15
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
    contact_mode: str = "ncp"
    smoothing_delta: float = 0.0
    n_uzawa_max: int = 5
    tol_uzawa: float = 1e-6


@dataclass
class _ContactManager:
    """接触ペアの管理."""

    pairs: list[_ContactPair] = field(default_factory=list)
    config: _ContactConfig = field(default_factory=_ContactConfig)

    @property
    def n_pairs(self) -> int:
        """ペア数."""
        return len(self.pairs)

    @property
    def n_active(self) -> int:
        """有効な接触ペア数."""
        return sum(1 for p in self.pairs if p.is_active())

    def add_pair(
        self,
        elem_a: int,
        elem_b: int,
        nodes_a: np.ndarray,
        nodes_b: np.ndarray,
        radius_a: float = 0.0,
        radius_b: float = 0.0,
        core_radius_a: float = 0.0,
        core_radius_b: float = 0.0,
    ) -> _ContactPair:
        """接触ペアを追加する."""
        pair = _ContactPair(
            elem_a=elem_a,
            elem_b=elem_b,
            nodes_a=np.asarray(nodes_a, dtype=int),
            nodes_b=np.asarray(nodes_b, dtype=int),
            radius_a=radius_a,
            radius_b=radius_b,
            core_radius_a=core_radius_a,
            core_radius_b=core_radius_b,
        )
        self.pairs.append(pair)
        return pair

    def reset_all(self) -> None:
        """全ペアの状態をリセットする."""
        for pair in self.pairs:
            pair.state = _ContactState()

    def get_active_pairs(self) -> list[_ContactPair]:
        """有効な接触ペアのリストを返す."""
        return [p for p in self.pairs if p.is_active()]

    def detect_candidates(
        self,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        radii: np.ndarray | float = 0.0,
        *,
        margin: float = 0.0,
        cell_size: float | None = None,
        core_radii: np.ndarray | float | None = None,
    ) -> list[tuple[int, int]]:
        """Broadphase で接触候補ペアを検出し pairs を更新する."""
        conn = np.asarray(connectivity, dtype=int)
        coords = np.asarray(node_coords, dtype=float)
        n_elems = len(conn)

        if np.isscalar(radii):
            r_arr = np.full(n_elems, float(radii))
        else:
            r_arr = np.asarray(radii, dtype=float)

        if core_radii is None:
            cr_arr = r_arr.copy()
        elif np.isscalar(core_radii):
            cr_arr = np.full(n_elems, float(core_radii))
        else:
            cr_arr = np.asarray(core_radii, dtype=float)

        segments = []
        for e in range(n_elems):
            ni, nj = conn[e]
            segments.append((coords[ni], coords[nj]))

        raw_candidates = _broadphase_aabb(segments, r_arr, margin=margin, cell_size=cell_size)

        candidates = []
        lm = self.config.elem_layer_map
        exclude_same = self.config.exclude_same_layer and lm is not None
        for i, j in raw_candidates:
            nodes_i = set(int(n) for n in conn[i])
            nodes_j = set(int(n) for n in conn[j])
            if nodes_i & nodes_j:
                continue
            if exclude_same:
                layer_i = lm.get(i, -1)
                layer_j = lm.get(j, -1)
                if layer_i == layer_j and layer_i >= 0:
                    continue
            candidates.append((i, j))

        # 中点距離プリスクリーニング
        if self.config.midpoint_prescreening and candidates:
            cand_arr = np.array(candidates, dtype=int)
            n0 = conn[cand_arr[:, 0], 0]
            n1 = conn[cand_arr[:, 0], 1]
            m0 = conn[cand_arr[:, 1], 0]
            m1 = conn[cand_arr[:, 1], 1]
            mid_a = 0.5 * (coords[n0] + coords[n1])
            mid_b = 0.5 * (coords[m0] + coords[m1])
            half_len_a = 0.5 * np.linalg.norm(coords[n1] - coords[n0], axis=1)
            half_len_b = 0.5 * np.linalg.norm(coords[m1] - coords[m0], axis=1)
            mid_dist = np.linalg.norm(mid_a - mid_b, axis=1)
            r_a = r_arr[cand_arr[:, 0]]
            r_b = r_arr[cand_arr[:, 1]]
            min_possible_dist = np.maximum(0.0, mid_dist - half_len_a - half_len_b)
            extra_margin = self.config.prescreening_margin
            if extra_margin <= 0.0:
                extra_margin = float(np.mean(r_a + r_b)) * 0.5
            cutoff = r_a + r_b + extra_margin
            keep = min_possible_dist <= cutoff
            candidates = [candidates[k] for k in range(len(candidates)) if keep[k]]

        existing: dict[tuple[int, int], int] = {}
        for idx, p in enumerate(self.pairs):
            key = (min(p.elem_a, p.elem_b), max(p.elem_a, p.elem_b))
            existing[key] = idx

        candidate_set = set(candidates)
        for key, idx in existing.items():
            if key not in candidate_set:
                self.pairs[idx].state.status = ContactStatus.INACTIVE

        for i, j in candidates:
            key = (min(i, j), max(i, j))
            if key not in existing:
                self.add_pair(
                    elem_a=i,
                    elem_b=j,
                    nodes_a=conn[i],
                    nodes_b=conn[j],
                    radius_a=float(r_arr[i]),
                    radius_b=float(r_arr[j]),
                    core_radius_a=float(cr_arr[i]),
                    core_radius_b=float(cr_arr[j]),
                )

        return candidates

    def update_geometry(
        self,
        node_coords: np.ndarray,
        *,
        allow_deactivation: bool = True,
        freeze_active_set: bool = False,
    ) -> None:
        """全ペアの幾何情報を更新する（Narrowphase）."""
        coords = np.asarray(node_coords, dtype=float)
        n_pairs = len(self.pairs)

        if n_pairs == 0:
            return

        nodes_a0 = np.array([p.nodes_a[0] for p in self.pairs], dtype=int)
        nodes_a1 = np.array([p.nodes_a[1] for p in self.pairs], dtype=int)
        nodes_b0 = np.array([p.nodes_b[0] for p in self.pairs], dtype=int)
        nodes_b1 = np.array([p.nodes_b[1] for p in self.pairs], dtype=int)

        xA0 = coords[nodes_a0]
        xA1 = coords[nodes_a1]
        xB0 = coords[nodes_b0]
        xB1 = coords[nodes_b1]

        s_all, t_all, _, _, dist_all, normal_all, _ = _closest_point_segments_batch(
            xA0, xA1, xB0, xB1
        )

        _use_coating = self.config.coating_stiffness > 0.0
        if _use_coating:
            core_a = np.array([p.core_radius_a for p in self.pairs])
            core_b = np.array([p.core_radius_b for p in self.pairs])
            radii_a = np.array([p.radius_a for p in self.pairs])
            radii_b = np.array([p.radius_b for p in self.pairs])
            gap_core = dist_all - (core_a + core_b)
            coat_total = (radii_a - core_a) + (radii_b - core_b)
            coat_comp = np.maximum(0.0, coat_total - gap_core)
            gap_all = gap_core
        else:
            radii_a = np.array([p.radius_a for p in self.pairs])
            radii_b = np.array([p.radius_b for p in self.pairs])
            gap_all = dist_all - (radii_a + radii_b)

        prev_t1_all = np.array([p.state.tangent1 for p in self.pairs])
        prev_n_all = np.array([p.state.normal for p in self.pairs])
        has_prev = np.sqrt(np.einsum("ij,ij->i", prev_t1_all, prev_t1_all)) > 1e-10
        has_prev_n = np.sqrt(np.einsum("ij,ij->i", prev_n_all, prev_n_all)) > 1e-10

        n_all, t1_all, t2_all = _build_contact_frame_batch(
            normal_all,
            prev_tangent1s=prev_t1_all,
            prev_normals=prev_n_all,
            has_prev_mask=has_prev,
            has_prev_n_mask=has_prev_n,
        )

        for i, pair in enumerate(self.pairs):
            pair.state.s = float(s_all[i])
            pair.state.t = float(t_all[i])
            pair.state.gap = float(gap_all[i])
            pair.state.normal = n_all[i]
            pair.state.tangent1 = t1_all[i]
            pair.state.tangent2 = t2_all[i]
            if _use_coating:
                pair.state.coating_compression = float(coat_comp[i])

            if not freeze_active_set:
                self._update_active_set(pair, allow_deactivation=allow_deactivation)

    def _update_active_set(
        self,
        pair: _ContactPair,
        *,
        allow_deactivation: bool = True,
    ) -> None:
        """Active-set をヒステリシス付きで更新する."""
        gap = pair.state.gap
        g_on = self.config.g_on
        g_off = self.config.g_off

        _coat_active = self.config.coating_stiffness > 0.0 and pair.state.coating_compression > 0.0

        if pair.state.status == ContactStatus.INACTIVE:
            if gap <= g_on or _coat_active:
                pair.state.status = ContactStatus.ACTIVE
        else:
            if allow_deactivation and gap >= g_off and not _coat_active:
                pair.state.status = ContactStatus.INACTIVE

    def initialize_penalty(self, k_pen: float, k_t_ratio: float | None = None) -> None:
        """全 ACTIVE ペアのペナルティ剛性を初期化する."""
        ratio = k_t_ratio if k_t_ratio is not None else self.config.k_t_ratio
        for pair in self.pairs:
            if pair.state.status != ContactStatus.INACTIVE and pair.state.k_pen <= 0.0:
                pair.state.k_pen = k_pen
                pair.state.k_t = ratio * k_pen
