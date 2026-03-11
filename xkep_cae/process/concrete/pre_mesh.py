"""StrandMeshProcess — 撚線メッシュ生成の PreProcess.

設計仕様: process-architecture.md §3.1
TwistedWireMesh を AbstractProcess として管理する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import PreProcess


@dataclass(frozen=True)
class StrandMeshConfig:
    """撚線メッシュ生成の入力パラメータ."""

    n_strands: int
    wire_radius: float
    pitch_length: float
    gap: float = 0.0
    n_elements_per_pitch: int = 16
    n_pitches: float = 1.0
    core_radius: float | None = None
    coating_thickness: float = 0.0
    sheath_thickness: float = 0.0


@dataclass
class StrandMeshResult:
    """撚線メッシュ生成結果."""

    node_coords: np.ndarray  # (n_nodes, 3)
    connectivity: np.ndarray  # (n_elems, 2)
    radii: np.ndarray | float
    n_strands: int
    layer_ids: np.ndarray | None = None
    core_radii: np.ndarray | float | None = None


class StrandMeshProcess(PreProcess[StrandMeshConfig, StrandMeshResult]):
    """撚線メッシュ生成プロセス.

    TwistedWireMesh の機能を PreProcess として管理する。
    """

    meta = ProcessMeta(
        name="StrandMesh",
        module="pre",
        version="0.1.0",
        document_path="../docs/process-architecture.md",
    )

    def process(self, input_data: StrandMeshConfig) -> StrandMeshResult:
        """メッシュ生成の実行."""
        from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh

        mesh = make_twisted_wire_mesh(
            n_strands=input_data.n_strands,
            wire_radius=input_data.wire_radius,
            pitch_length=input_data.pitch_length,
            gap=input_data.gap,
            n_elements_per_pitch=input_data.n_elements_per_pitch,
            n_pitches=input_data.n_pitches,
        )

        return StrandMeshResult(
            node_coords=mesh.node_coords,
            connectivity=mesh.connectivity,
            radii=mesh.radii,
            n_strands=input_data.n_strands,
            layer_ids=mesh.layer_ids if hasattr(mesh, "layer_ids") else None,
            core_radii=mesh.core_radii if hasattr(mesh, "core_radii") else None,
        )
