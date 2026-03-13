"""StrandMeshProcess の1:1テスト."""

from __future__ import annotations

import pytest

from xkep_cae.process.concrete.pre_mesh import (
    StrandMeshConfig,
    StrandMeshProcess,
    StrandMeshResult,
)
from xkep_cae.process.testing import binds_to


@binds_to(StrandMeshProcess)
class TestStrandMeshProcess:
    """StrandMeshProcess の単体テスト."""

    def test_meta(self):
        assert StrandMeshProcess.meta.name == "StrandMesh"
        assert StrandMeshProcess.meta.module == "pre"
        assert not StrandMeshProcess.meta.deprecated
        assert StrandMeshProcess.meta.stability == "stable"
        assert StrandMeshProcess.meta.support_tier == "ci-required"

    def test_process_returns_result(self):
        """7本撚線の最小構成でメッシュ生成."""
        config = StrandMeshConfig(
            n_strands=7,
            wire_radius=0.1,
            pitch_length=10.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
        )
        proc = StrandMeshProcess()
        result = proc.process(config)
        assert isinstance(result, StrandMeshResult)
        assert result.n_strands == 7
        assert result.node_coords.shape[1] == 3
        assert result.connectivity.shape[1] == 2

    def test_execute_delegates_to_process(self):
        config = StrandMeshConfig(
            n_strands=7,
            wire_radius=0.1,
            pitch_length=10.0,
        )
        proc = StrandMeshProcess()
        result = proc.execute(config)
        assert isinstance(result, StrandMeshResult)

    def test_registry_registered(self):
        from xkep_cae.process.base import AbstractProcess

        assert "StrandMeshProcess" in AbstractProcess._registry
