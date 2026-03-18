"""ContactManager 操作の Process 実装.

_ContactManagerInput の全操作を SolverProcess として実装。
dataclass はメソッドを持たず、本モジュールの Process が唯一の操作手段。

status-205: manager メソッド → Process 直接実装に完全移行。
全 Process が更新済み manager を出力に含める（不変パターン）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from xkep_cae.contact._broadphase import _broadphase_aabb
from xkep_cae.contact._contact_pair import (
    _ContactConfigInput,
    _ContactManagerInput,
    _ContactPairOutput,
    _ContactStateOutput,
    _evolve_pair,
    _evolve_state,
    _make_pair,
    _n_active,
    _n_pairs,
)
from xkep_cae.contact._types import ContactStatus
from xkep_cae.contact.geometry._compute import (
    _build_contact_frame_batch,
    _closest_point_segments_batch,
)
from xkep_cae.core import ProcessMeta, SolverProcess

# ---------------------------------------------------------------------------
# AddPair Process
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AddPairInput:
    """接触ペア追加の入力."""

    manager: _ContactManagerInput
    elem_a: int
    elem_b: int
    nodes_a: np.ndarray
    nodes_b: np.ndarray
    radius_a: float = 0.0
    radius_b: float = 0.0
    core_radius_a: float = 0.0
    core_radius_b: float = 0.0


@dataclass(frozen=True)
class AddPairOutput:
    """接触ペア追加の出力."""

    manager: _ContactManagerInput
    pair: _ContactPairOutput


class AddPairProcess(
    SolverProcess[AddPairInput, AddPairOutput],
):
    """接触ペア追加 Process."""

    meta = ProcessMeta(
        name="AddPair",
        module="solve",
        version="1.0.0",
        document_path="solver/docs/contact_friction.md",
    )

    def process(self, input_data: AddPairInput) -> AddPairOutput:
        pair = _make_pair(
            elem_a=input_data.elem_a,
            elem_b=input_data.elem_b,
            nodes_a=input_data.nodes_a,
            nodes_b=input_data.nodes_b,
            radius_a=input_data.radius_a,
            radius_b=input_data.radius_b,
            core_radius_a=input_data.core_radius_a,
            core_radius_b=input_data.core_radius_b,
        )
        new_pairs = list(input_data.manager.pairs)
        new_pairs.append(pair)
        new_manager = _ContactManagerInput(pairs=new_pairs, config=input_data.manager.config)
        return AddPairOutput(manager=new_manager, pair=pair)


# ---------------------------------------------------------------------------
# ResetAllPairs Process
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResetAllPairsInput:
    """全ペアリセットの入力."""

    manager: _ContactManagerInput


@dataclass(frozen=True)
class ResetAllPairsOutput:
    """全ペアリセットの出力."""

    manager: _ContactManagerInput


class ResetAllPairsProcess(
    SolverProcess[ResetAllPairsInput, ResetAllPairsOutput],
):
    """全ペアの状態をリセットする Process."""

    meta = ProcessMeta(
        name="ResetAllPairs",
        module="solve",
        version="1.0.0",
        document_path="solver/docs/contact_friction.md",
    )

    def process(self, input_data: ResetAllPairsInput) -> ResetAllPairsOutput:
        new_pairs = [_evolve_pair(p, state=_ContactStateOutput()) for p in input_data.manager.pairs]
        new_manager = _ContactManagerInput(pairs=new_pairs, config=input_data.manager.config)
        return ResetAllPairsOutput(manager=new_manager)


# ---------------------------------------------------------------------------
# DetectCandidates Process
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectCandidatesInput:
    """Broadphase 候補検出の入力."""

    manager: _ContactManagerInput
    node_coords: np.ndarray
    connectivity: np.ndarray
    radii: np.ndarray | float = 0.0
    margin: float = 0.0
    cell_size: float | None = None
    core_radii: np.ndarray | float | None = None


@dataclass(frozen=True)
class DetectCandidatesOutput:
    """Broadphase 候補検出の出力."""

    manager: _ContactManagerInput
    candidates: list[tuple[int, int]]
    n_pairs: int


class DetectCandidatesProcess(
    SolverProcess[DetectCandidatesInput, DetectCandidatesOutput],
):
    """Broadphase 候補検出 Process.

    旧 _ContactManagerInput.detect_candidates() のロジックを直接実装。
    """

    meta = ProcessMeta(
        name="DetectCandidates",
        module="solve",
        version="2.0.0",
        document_path="solver/docs/contact_friction.md",
    )

    def process(self, input_data: DetectCandidatesInput) -> DetectCandidatesOutput:
        manager = input_data.manager
        config = manager.config
        conn = np.asarray(input_data.connectivity, dtype=int)
        coords = np.asarray(input_data.node_coords, dtype=float)
        n_elems = len(conn)

        if np.isscalar(input_data.radii):
            r_arr = np.full(n_elems, float(input_data.radii))
        else:
            r_arr = np.asarray(input_data.radii, dtype=float)

        if input_data.core_radii is None:
            cr_arr = r_arr.copy()
        elif np.isscalar(input_data.core_radii):
            cr_arr = np.full(n_elems, float(input_data.core_radii))
        else:
            cr_arr = np.asarray(input_data.core_radii, dtype=float)

        segments = []
        for e in range(n_elems):
            ni, nj = conn[e]
            segments.append((coords[ni], coords[nj]))

        raw_candidates = _broadphase_aabb(
            segments, r_arr, margin=input_data.margin, cell_size=input_data.cell_size
        )

        candidates: list[tuple[int, int]] = []
        lm = config.elem_layer_map
        exclude_same = config.exclude_same_layer and lm is not None
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
        if config.midpoint_prescreening and candidates:
            candidates = self._midpoint_prescreening(candidates, conn, coords, r_arr, config)

        # 既存ペアのインデックスマップ
        existing: dict[tuple[int, int], int] = {}
        pairs = list(manager.pairs)
        for idx, p in enumerate(pairs):
            key = (min(p.elem_a, p.elem_b), max(p.elem_a, p.elem_b))
            existing[key] = idx

        # 候補外の既存ペアを INACTIVE に
        candidate_set = set(candidates)
        for key, idx in existing.items():
            if key not in candidate_set:
                old_pair = pairs[idx]
                pairs[idx] = _evolve_pair(
                    old_pair,
                    state=_evolve_state(old_pair.state, status=ContactStatus.INACTIVE),
                )

        # 新規候補を追加
        for i, j in candidates:
            key = (min(i, j), max(i, j))
            if key not in existing:
                pairs.append(
                    _make_pair(
                        elem_a=i,
                        elem_b=j,
                        nodes_a=conn[i],
                        nodes_b=conn[j],
                        radius_a=float(r_arr[i]),
                        radius_b=float(r_arr[j]),
                        core_radius_a=float(cr_arr[i]),
                        core_radius_b=float(cr_arr[j]),
                    )
                )

        new_manager = _ContactManagerInput(pairs=pairs, config=config)
        return DetectCandidatesOutput(
            manager=new_manager,
            candidates=candidates,
            n_pairs=_n_pairs(new_manager),
        )

    @staticmethod
    def _midpoint_prescreening(
        candidates: list[tuple[int, int]],
        conn: np.ndarray,
        coords: np.ndarray,
        r_arr: np.ndarray,
        config: _ContactConfigInput,
    ) -> list[tuple[int, int]]:
        """中点距離プリスクリーニング."""
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
        extra_margin = config.prescreening_margin
        if extra_margin <= 0.0:
            extra_margin = float(np.mean(r_a + r_b)) * 0.5
        cutoff = r_a + r_b + extra_margin
        keep = min_possible_dist <= cutoff
        return [candidates[k] for k in range(len(candidates)) if keep[k]]


# ---------------------------------------------------------------------------
# UpdateGeometry Process
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UpdateGeometryInput:
    """Narrowphase 幾何情報更新の入力."""

    manager: _ContactManagerInput
    node_coords: np.ndarray
    allow_deactivation: bool = True
    freeze_active_set: bool = False


@dataclass(frozen=True)
class UpdateGeometryOutput:
    """Narrowphase 幾何情報更新の出力."""

    manager: _ContactManagerInput
    n_active: int


class UpdateGeometryProcess(
    SolverProcess[UpdateGeometryInput, UpdateGeometryOutput],
):
    """Narrowphase 幾何情報更新 Process.

    旧 _ContactManagerInput.update_geometry() のロジックを直接実装。
    """

    meta = ProcessMeta(
        name="UpdateGeometry",
        module="solve",
        version="2.0.0",
        document_path="solver/docs/contact_friction.md",
    )

    def process(self, input_data: UpdateGeometryInput) -> UpdateGeometryOutput:
        manager = input_data.manager
        config = manager.config
        coords = np.asarray(input_data.node_coords, dtype=float)
        pairs = list(manager.pairs)
        n_count = len(pairs)

        if n_count == 0:
            return UpdateGeometryOutput(manager=manager, n_active=0)

        nodes_a0 = np.array([p.nodes_a[0] for p in pairs], dtype=int)
        nodes_a1 = np.array([p.nodes_a[1] for p in pairs], dtype=int)
        nodes_b0 = np.array([p.nodes_b[0] for p in pairs], dtype=int)
        nodes_b1 = np.array([p.nodes_b[1] for p in pairs], dtype=int)

        xA0 = coords[nodes_a0]
        xA1 = coords[nodes_a1]
        xB0 = coords[nodes_b0]
        xB1 = coords[nodes_b1]

        s_all, t_all, _, _, dist_all, normal_all, _ = _closest_point_segments_batch(
            xA0, xA1, xB0, xB1
        )

        _use_coating = config.coating_stiffness > 0.0
        if _use_coating:
            core_a = np.array([p.core_radius_a for p in pairs])
            core_b = np.array([p.core_radius_b for p in pairs])
            radii_a = np.array([p.radius_a for p in pairs])
            radii_b = np.array([p.radius_b for p in pairs])
            gap_core = dist_all - (core_a + core_b)
            coat_total = (radii_a - core_a) + (radii_b - core_b)
            coat_comp = np.maximum(0.0, coat_total - gap_core)
            gap_all = gap_core
        else:
            radii_a = np.array([p.radius_a for p in pairs])
            radii_b = np.array([p.radius_b for p in pairs])
            gap_all = dist_all - (radii_a + radii_b)

        prev_t1_all = np.array([p.state.tangent1 for p in pairs])
        prev_n_all = np.array([p.state.normal for p in pairs])
        has_prev = np.sqrt(np.einsum("ij,ij->i", prev_t1_all, prev_t1_all)) > 1e-10
        has_prev_n = np.sqrt(np.einsum("ij,ij->i", prev_n_all, prev_n_all)) > 1e-10

        n_all, t1_all, t2_all = _build_contact_frame_batch(
            normal_all,
            prev_tangent1s=prev_t1_all,
            prev_normals=prev_n_all,
            has_prev_mask=has_prev,
            has_prev_n_mask=has_prev_n,
        )

        new_pairs = []
        for i, pair in enumerate(pairs):
            geom_kw: dict[str, Any] = {
                "s": float(s_all[i]),
                "t": float(t_all[i]),
                "gap": float(gap_all[i]),
                "normal": n_all[i],
                "tangent1": t1_all[i],
                "tangent2": t2_all[i],
            }
            if _use_coating:
                geom_kw["coating_compression"] = float(coat_comp[i])
            new_state = _evolve_state(pair.state, **geom_kw)

            if not input_data.freeze_active_set:
                new_state = _update_active_set_state(
                    config, new_state, allow_deactivation=input_data.allow_deactivation
                )
            new_pairs.append(_evolve_pair(pair, state=new_state))

        new_manager = _ContactManagerInput(pairs=new_pairs, config=config)
        return UpdateGeometryOutput(
            manager=new_manager,
            n_active=_n_active(new_manager),
        )


# ---------------------------------------------------------------------------
# InitializePenalty Process
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InitializePenaltyInput:
    """ペナルティ初期化の入力."""

    manager: _ContactManagerInput
    k_pen: float
    k_t_ratio: float | None = None


@dataclass(frozen=True)
class InitializePenaltyOutput:
    """ペナルティ初期化の出力."""

    manager: _ContactManagerInput
    n_initialized: int


class InitializePenaltyProcess(
    SolverProcess[InitializePenaltyInput, InitializePenaltyOutput],
):
    """ペナルティ剛性初期化 Process.

    旧 _ContactManagerInput.initialize_penalty() のロジックを直接実装。
    """

    meta = ProcessMeta(
        name="InitializePenalty",
        module="solve",
        version="2.0.0",
        document_path="solver/docs/contact_friction.md",
    )

    def process(self, input_data: InitializePenaltyInput) -> InitializePenaltyOutput:
        manager = input_data.manager
        ratio = (
            input_data.k_t_ratio if input_data.k_t_ratio is not None else manager.config.k_t_ratio
        )
        new_pairs = []
        for pair in manager.pairs:
            if pair.state.status != ContactStatus.INACTIVE and pair.state.k_pen <= 0.0:
                new_pairs.append(
                    _evolve_pair(
                        pair,
                        state=_evolve_state(
                            pair.state,
                            k_pen=input_data.k_pen,
                            k_t=ratio * input_data.k_pen,
                        ),
                    )
                )
            else:
                new_pairs.append(pair)
        new_manager = _ContactManagerInput(pairs=new_pairs, config=manager.config)
        return InitializePenaltyOutput(
            manager=new_manager,
            n_initialized=_n_active(new_manager),
        )


# ---------------------------------------------------------------------------
# active set 更新（Process 内部ヘルパー）
# ---------------------------------------------------------------------------


def _update_active_set_state(
    config: _ContactConfigInput,
    state: _ContactStateOutput,
    *,
    allow_deactivation: bool = True,
) -> _ContactStateOutput:
    """Active-set をヒステリシス付きで更新し新 state を返す."""
    gap = state.gap
    g_on = config.g_on
    g_off = config.g_off

    _coat_active = config.coating_stiffness > 0.0 and state.coating_compression > 0.0

    if state.status == ContactStatus.INACTIVE:
        if gap <= g_on or _coat_active:
            return _evolve_state(state, status=ContactStatus.ACTIVE)
    else:
        if allow_deactivation and gap >= g_off and not _coat_active:
            return _evolve_state(state, status=ContactStatus.INACTIVE)
    return state
