"""ContactManager メソッドの Process ラッパー.

_ContactManagerInput の detect_candidates / update_geometry を
SolverProcess 化し、Process API 経由でのアクセスを可能にする。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xkep_cae.core import ProcessMeta, SolverProcess


@dataclass(frozen=True)
class DetectCandidatesInput:
    """Broadphase 候補検出の入力."""

    manager: object
    node_coords: np.ndarray
    connectivity: np.ndarray
    radii: np.ndarray | float = 0.0
    margin: float = 0.0
    cell_size: float | None = None
    core_radii: np.ndarray | float | None = None


@dataclass(frozen=True)
class DetectCandidatesOutput:
    """Broadphase 候補検出の出力."""

    candidates: list[tuple[int, int]]
    n_pairs: int


class DetectCandidatesProcess(
    SolverProcess[DetectCandidatesInput, DetectCandidatesOutput],
):
    """Broadphase 候補検出 Process.

    ContactManager.detect_candidates() を Process API でラップする。
    manager は in-place で更新される（pairs リスト変更）。
    """

    meta = ProcessMeta(
        name="DetectCandidates",
        module="solve",
        version="1.0.0",
        document_path="solver/docs/contact_friction.md",
    )

    def process(self, input_data: DetectCandidatesInput) -> DetectCandidatesOutput:
        candidates = input_data.manager.detect_candidates(
            input_data.node_coords,
            input_data.connectivity,
            input_data.radii,
            margin=input_data.margin,
            cell_size=input_data.cell_size,
            core_radii=input_data.core_radii,
        )
        return DetectCandidatesOutput(
            candidates=candidates,
            n_pairs=input_data.manager.n_pairs,
        )


@dataclass(frozen=True)
class UpdateGeometryInput:
    """Narrowphase 幾何情報更新の入力."""

    manager: object
    node_coords: np.ndarray
    allow_deactivation: bool = True
    freeze_active_set: bool = False


@dataclass(frozen=True)
class UpdateGeometryOutput:
    """Narrowphase 幾何情報更新の出力."""

    n_active: int


class UpdateGeometryProcess(
    SolverProcess[UpdateGeometryInput, UpdateGeometryOutput],
):
    """Narrowphase 幾何情報更新 Process.

    ContactManager.update_geometry() を Process API でラップする。
    manager は in-place で更新される（各ペアの状態変更）。
    """

    meta = ProcessMeta(
        name="UpdateGeometry",
        module="solve",
        version="1.0.0",
        document_path="solver/docs/contact_friction.md",
    )

    def process(self, input_data: UpdateGeometryInput) -> UpdateGeometryOutput:
        input_data.manager.update_geometry(
            input_data.node_coords,
            allow_deactivation=input_data.allow_deactivation,
            freeze_active_set=input_data.freeze_active_set,
        )
        return UpdateGeometryOutput(n_active=input_data.manager.n_active)


@dataclass(frozen=True)
class InitializePenaltyInput:
    """ペナルティ初期化の入力."""

    manager: object
    k_pen: float
    k_t_ratio: float | None = None


@dataclass(frozen=True)
class InitializePenaltyOutput:
    """ペナルティ初期化の出力."""

    n_initialized: int


class InitializePenaltyProcess(
    SolverProcess[InitializePenaltyInput, InitializePenaltyOutput],
):
    """ペナルティ剛性初期化 Process.

    ContactManager.initialize_penalty() を Process API でラップする。
    """

    meta = ProcessMeta(
        name="InitializePenalty",
        module="solve",
        version="1.0.0",
        document_path="solver/docs/contact_friction.md",
    )

    def process(self, input_data: InitializePenaltyInput) -> InitializePenaltyOutput:
        input_data.manager.initialize_penalty(
            input_data.k_pen,
            input_data.k_t_ratio,
        )
        return InitializePenaltyOutput(n_initialized=input_data.manager.n_active)
