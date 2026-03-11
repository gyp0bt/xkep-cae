"""BeamRenderProcess — 梁3Dレンダリングの PostProcess.

設計仕様: process-architecture.md §3.1
ソルバー結果を 3D チューブレンダリングで可視化する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import PostProcess
from xkep_cae.process.data import MeshData, SolverResultData


@dataclass(frozen=True)
class RenderConfig:
    """レンダリング設定."""

    solver_result: SolverResultData
    mesh: MeshData
    output_dir: str = "output"
    prefix: str = "render"
    ndof_per_node: int = 6
    show_contact: bool = True
    tube_segments: int = 8


@dataclass
class RenderResult:
    """レンダリング結果."""

    image_paths: list[str] = field(default_factory=list)


class BeamRenderProcess(PostProcess[RenderConfig, RenderResult]):
    """梁3Dレンダリングプロセス.

    変形後のメッシュを 3D チューブで可視化し、PNG で保存する。
    """

    meta = ProcessMeta(
        name="BeamRender",
        module="post",
        version="0.1.0",
        document_path="xkep_cae/process/docs/process-architecture.md",
    )

    def process(self, input_data: RenderConfig) -> RenderResult:
        """レンダリングの実行."""
        output_dir = Path(input_data.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = input_data.solver_result
        mesh = input_data.mesh

        # 変形後座標の計算
        n_nodes = mesh.node_coords.shape[0]
        u = result.u
        ndof_per_node = input_data.ndof_per_node
        deformed = mesh.node_coords.copy()
        for i in range(n_nodes):
            for d in range(3):
                idx = i * ndof_per_node + d
                if idx < len(u):
                    deformed[i, d] += u[idx]

        # 3Dレンダリング（利用可能な場合）
        image_paths: list[str] = []
        try:
            from xkep_cae.visualization.render_3d import render_beam_tubes

            img_path = str(output_dir / f"{input_data.prefix}_3d.png")
            render_beam_tubes(
                deformed,
                mesh.connectivity,
                mesh.radii,
                output_path=img_path,
                tube_segments=input_data.tube_segments,
            )
            image_paths.append(img_path)
        except ImportError:
            pass

        return RenderResult(image_paths=image_paths)
