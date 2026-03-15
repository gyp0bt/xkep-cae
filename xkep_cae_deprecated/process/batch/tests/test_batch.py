"""StrandBendingBatchProcess の1:1テスト."""

from __future__ import annotations

from xkep_cae_deprecated.process.batch.strand_bending import (
    StrandBendingBatchProcess,
)
from xkep_cae_deprecated.process.testing import binds_to


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
            "ContactFrictionProcess",
            "ExportProcess",
            "BeamRenderProcess",
            "ConvergenceVerifyProcess",
        ]

    def test_is_batch_process(self):
        from xkep_cae_deprecated.process.categories import BatchProcess

        assert issubclass(StrandBendingBatchProcess, BatchProcess)

    def test_registry_registered(self):
        from xkep_cae_deprecated.process.base import AbstractProcess

        assert "StrandBendingBatchProcess" in AbstractProcess._registry
