"""StressContour3DProcess の @binds_to 紐付けテスト.

C3 契約: 全 concrete Process に対し @binds_to テストが必須。

[← README](../../../README.md)
"""

from __future__ import annotations

import numpy as np

from xkep_cae.core.testing import binds_to
from xkep_cae.output.stress_contour import (
    ContourFieldInput,
    StressContour3DConfig,
    StressContour3DProcess,
)


@binds_to(StressContour3DProcess)
class TestStressContour3DProcessAPI:
    """StressContour3DProcess の基本動作確認."""

    def test_process_runs(self, tmp_path):
        """複数フィールドでエラーなく完了する."""
        n_nodes = 11
        coords = np.column_stack(
            [np.linspace(0, 10, n_nodes), np.zeros(n_nodes), np.zeros(n_nodes)]
        )
        conn = np.column_stack([np.arange(10), np.arange(1, 11)])

        from xkep_cae.core import MeshData

        mesh = MeshData(
            node_coords=coords,
            connectivity=conn,
            radii=np.full(10, 0.5),
            n_strands=1,
        )

        u = np.zeros(n_nodes * 6)
        strain = np.ones(10) * 0.001
        stress = strain * 100e3
        curvature = strain / 0.5

        cfg = StressContour3DConfig(
            mesh=mesh,
            node_coords_initial=coords,
            displacement_snapshots=[u],
            contour_fields=[
                ContourFieldInput(name="S11", snapshots=[stress]),
                ContourFieldInput(name="LE11", snapshots=[strain]),
                ContourFieldInput(name="SK1", snapshots=[curvature]),
            ],
            time_values=np.array([0.0]),
            wire_radius=0.5,
            output_dir=str(tmp_path),
            prefix="test",
            n_render_frames=1,
        )
        proc = StressContour3DProcess()
        result = proc.process(cfg)
        assert len(result.image_paths) > 0
        assert "S11" in result.field_max_values
        assert "LE11" in result.field_max_values
        assert "SK1" in result.field_max_values
