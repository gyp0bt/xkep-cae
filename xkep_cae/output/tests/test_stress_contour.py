"""StressContour3DProcess の @binds_to 紐付けテスト.

C3 契約: 全 concrete Process に対し @binds_to テストが必須。

[← README](../../../README.md)
"""

from __future__ import annotations

import numpy as np

from xkep_cae.core.testing import binds_to
from xkep_cae.output.stress_contour import (
    StressContour3DConfig,
    StressContour3DProcess,
)


@binds_to(StressContour3DProcess)
class TestStressContour3DProcessAPI:
    """StressContour3DProcess の基本動作確認."""

    def test_process_runs(self, tmp_path):
        """空データでもエラーなく完了する."""
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
        stress = np.ones(10) * 100.0

        cfg = StressContour3DConfig(
            mesh=mesh,
            node_coords_initial=coords,
            displacement_snapshots=[u],
            element_stress_snapshots=[stress],
            time_values=np.array([0.0]),
            wire_radius=0.5,
            output_dir=str(tmp_path),
            prefix="test",
            n_render_frames=1,
        )
        proc = StressContour3DProcess()
        result = proc.process(cfg)
        assert len(result.image_paths) > 0
