"""BeamRenderProcess — 梁3Dレンダリングの PostProcess.

設計仕様: docs/render.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from xkep_cae.core import MeshData, PostProcess, ProcessMeta, SolverResultData


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


@dataclass(frozen=True)
class RenderResult:
    """レンダリング結果."""

    image_paths: list[str] = field(default_factory=list)


class BeamRenderProcess(PostProcess[RenderConfig, RenderResult]):
    """梁3Dレンダリングプロセス.

    変形後のメッシュを可視化し、PNG で保存する。
    """

    meta = ProcessMeta(
        name="BeamRender",
        module="post",
        version="1.0.0",
        document_path="docs/render.md",
    )

    def process(self, input_data: RenderConfig) -> RenderResult:
        """レンダリングの実行."""
        output_dir = Path(input_data.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = input_data.solver_result
        mesh = input_data.mesh

        n_nodes = mesh.node_coords.shape[0]
        u = result.u
        ndof_per_node = input_data.ndof_per_node
        deformed = mesh.node_coords.copy()
        for i in range(n_nodes):
            for d in range(3):
                idx = i * ndof_per_node + d
                if idx < len(u):
                    deformed[i, d] += u[idx]

        image_paths: list[str] = []

        # 2D投影スナップショット（matplotlib）
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            conn = mesh.connectivity
            for ax, proj, xlabel, ylabel in [
                (axes[0], (0, 1), "X", "Y"),
                (axes[1], (0, 2), "X", "Z"),
            ]:
                for e in range(conn.shape[0]):
                    n0, n1 = conn[e]
                    ax.plot(
                        [deformed[n0, proj[0]], deformed[n1, proj[0]]],
                        [deformed[n0, proj[1]], deformed[n1, proj[1]]],
                        "b-",
                        linewidth=0.5,
                    )
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_aspect("equal")

            fig.suptitle("Deformed Beam Configuration")
            fig.tight_layout()
            img_path = str(output_dir / f"{input_data.prefix}_2d.png")
            fig.savefig(img_path, dpi=150)
            plt.close(fig)
            image_paths.append(img_path)
        except ImportError:
            pass

        # 変形座標を CSV 出力（可視化ツール連携用）
        coord_path = output_dir / f"{input_data.prefix}_deformed.csv"
        np.savetxt(coord_path, deformed, delimiter=",", header="x,y,z")
        image_paths.append(str(coord_path))

        return RenderResult(image_paths=image_paths)
