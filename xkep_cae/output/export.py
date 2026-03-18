"""ExportProcess — 結果出力の PostProcess.

設計仕様: docs/export.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from xkep_cae.core import MeshData, PostProcess, ProcessMeta, SolverResultData


@dataclass(frozen=True)
class ExportConfig:
    """エクスポート設定."""

    solver_result: SolverResultData
    mesh: MeshData
    output_dir: str = "output"
    formats: tuple[str, ...] = ("csv",)
    prefix: str = "result"


@dataclass(frozen=True)
class ExportResult:
    """エクスポート結果."""

    exported_files: list[str] = field(default_factory=list)
    n_steps_exported: int = 0


class ExportProcess(PostProcess[ExportConfig, ExportResult]):
    """結果エクスポートプロセス.

    変位履歴・接触力履歴を CSV/JSON でエクスポートする。
    """

    meta = ProcessMeta(
        name="Export",
        module="post",
        version="1.0.0",
        document_path="docs/export.md",
    )

    def process(self, input_data: ExportConfig) -> ExportResult:
        """結果のエクスポート."""
        output_dir = Path(input_data.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported: list[str] = []
        result = input_data.solver_result

        if "csv" in input_data.formats:
            u_path = output_dir / f"{input_data.prefix}_displacement.csv"
            np.savetxt(u_path, result.u.reshape(1, -1), delimiter=",")
            exported.append(str(u_path))

            if result.contact_force_history:
                cf_path = output_dir / f"{input_data.prefix}_contact_force.csv"
                np.savetxt(
                    cf_path,
                    np.array(result.contact_force_history).reshape(-1, 1),
                    delimiter=",",
                    header="contact_force_norm",
                )
                exported.append(str(cf_path))

        if "json" in input_data.formats:
            import json

            summary = {
                "converged": result.converged,
                "n_increments": result.n_increments,
                "total_newton_iterations": result.total_newton_iterations,
                "max_displacement": float(np.max(np.abs(result.u))),
                "elapsed_seconds": result.elapsed_seconds,
            }
            json_path = output_dir / f"{input_data.prefix}_summary.json"
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2)
            exported.append(str(json_path))

        return ExportResult(
            exported_files=exported,
            n_steps_exported=len(result.displacement_history),
        )
