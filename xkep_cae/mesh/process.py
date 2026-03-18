"""StrandMeshProcess — 撚線メッシュ生成の PreProcess.

旧 __xkep_cae_deprecated/process/concrete/pre_mesh.py の完全書き直し。
設計仕様: docs/mesh_process.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xkep_cae.core import MeshData, PreProcess, ProcessMeta
from xkep_cae.mesh._twisted_wire import _make_twisted_wire_mesh, _radii


@dataclass(frozen=True)
class StrandMeshConfig:
    """撚線メッシュ生成の入力パラメータ."""

    n_strands: int = 7
    wire_radius: float = 0.5
    pitch_length: float = 50.0
    gap: float = 0.0
    n_elements_per_pitch: int = 16
    n_pitches: float = 1.0
    core_radius: float | None = None
    coating_thickness: float = 0.0
    sheath_thickness: float = 0.0


@dataclass(frozen=True)
class StrandMeshResult:
    """撚線メッシュ生成結果."""

    mesh: MeshData
    core_radii: np.ndarray | float | None = None


class StrandMeshProcess(PreProcess[StrandMeshConfig, StrandMeshResult]):
    """撚線メッシュ生成プロセス.

    TwistedWireMeshOutput の機能を PreProcess として管理する。
    """

    meta = ProcessMeta(
        name="StrandMesh",
        module="pre",
        version="1.0.0",
        document_path="docs/mesh_process.md",
    )

    def process(self, input_data: StrandMeshConfig) -> StrandMeshResult:
        """メッシュ生成の実行."""
        wire_diameter = input_data.wire_radius * 2.0
        length = input_data.pitch_length * input_data.n_pitches
        n_elems = int(input_data.n_elements_per_pitch * input_data.n_pitches)

        mesh = _make_twisted_wire_mesh(
            n_strands=input_data.n_strands,
            wire_diameter=wire_diameter,
            pitch=input_data.pitch_length,
            length=length,
            n_elems_per_strand=n_elems,
            gap=input_data.gap,
            n_pitches=input_data.n_pitches,
            coating_thickness=input_data.coating_thickness,
        )

        mesh_data = MeshData(
            node_coords=mesh.node_coords,
            connectivity=mesh.connectivity,
            radii=_radii(mesh),
            n_strands=input_data.n_strands,
            layer_ids=getattr(mesh, "layer_ids", None),
        )

        return StrandMeshResult(
            mesh=mesh_data,
            core_radii=getattr(mesh, "core_radii", None),
        )
