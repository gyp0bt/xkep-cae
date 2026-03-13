"""StrandBendingBatchProcess の1:1テスト."""

from __future__ import annotations

import pytest

from xkep_cae.process.batch.strand_bending import (
    BatchConfig,
    BatchResult,
    StrandBendingBatchProcess,
)
from xkep_cae.process.testing import binds_to


@binds_to(StrandBendingBatchProcess)
class TestStrandBendingBatchProcess:
    """StrandBendingBatchProcess の単体テスト."""

    def test_meta(self):
        assert StrandBendingBatchProcess.meta.name == "StrandBendingBatch"
        assert StrandBendingBatchProcess.meta.module == "batch"
        assert not StrandBendingBatchProcess.meta.deprecated

    def test_uses_order(self):
        """uses 宣言順が実行順と一致すること."""
        names = [cls.__name__ for cls in StrandBendingBatchProcess.uses]
        assert names == [
            "StrandMeshProcess",
            "ContactSetupProcess",
            "NCPContactSolverProcess",
            "ExportProcess",
            "BeamRenderProcess",
            "ConvergenceVerifyProcess",
        ]

    def test_is_batch_process(self):
        from xkep_cae.process.categories import BatchProcess

        assert issubclass(StrandBendingBatchProcess, BatchProcess)

    def test_registry_registered(self):
        from xkep_cae.process.base import AbstractProcess

        assert "StrandBendingBatchProcess" in AbstractProcess._registry
