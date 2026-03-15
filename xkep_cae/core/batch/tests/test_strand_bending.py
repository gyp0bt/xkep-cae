"""StrandBendingBatchProcess のテスト.

@binds_to による 1:1 紐付け + BatchProcess カテゴリ検証。
"""

from __future__ import annotations

from xkep_cae.core import BatchProcess
from xkep_cae.core.batch import (
    StrandBatchConfig,
    StrandBatchResult,
    StrandBendingBatchProcess,
)
from xkep_cae.core.testing import binds_to


@binds_to(StrandBendingBatchProcess)
class TestStrandBendingBatchProcess:
    """StrandBendingBatchProcess の単体テスト."""

    def test_is_batch_process(self):
        proc = StrandBendingBatchProcess()
        assert isinstance(proc, BatchProcess)

    def test_uses_not_empty(self):
        assert len(StrandBendingBatchProcess.uses) > 0

    def test_meta_name(self):
        assert StrandBendingBatchProcess.meta.name == "StrandBendingBatch"

    def test_meta_module(self):
        assert StrandBendingBatchProcess.meta.module == "batch"

    def test_process_returns_result(self):
        proc = StrandBendingBatchProcess()
        config = StrandBatchConfig()
        result = proc.process(config)
        assert isinstance(result, StrandBatchResult)

    def test_process_log_populated(self):
        proc = StrandBendingBatchProcess()
        config = StrandBatchConfig()
        result = proc.process(config)
        assert len(result.process_log) > 0

    def test_default_config(self):
        config = StrandBatchConfig()
        assert config.contact_mode == "smooth_penalty"
        assert config.geometry_mode == "point_to_point"
        assert config.use_friction is True

    def test_custom_config(self):
        config = StrandBatchConfig(
            contact_mode="ncp",
            geometry_mode="line_to_line",
            use_friction=False,
        )
        assert config.contact_mode == "ncp"
        assert config.geometry_mode == "line_to_line"
        assert config.use_friction is False
