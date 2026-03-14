"""ProcessRunner テスト.

設計仕様: phase8-design.md §A
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import PreProcess
from xkep_cae.process.runner import ExecutionContext, ExecutionRecord, ProcessRunner

# --- テスト用プロセス ---


@dataclass(frozen=True)
class _DummyInput:
    value: int


@dataclass
class _DummyOutput:
    result: int


class _AddOneProcess(PreProcess[_DummyInput, _DummyOutput]):
    meta = ProcessMeta(name="AddOne", module="pre", document_path="../docs/process-architecture.md")
    _skip_registry = True

    def process(self, input_data: _DummyInput) -> _DummyOutput:
        return _DummyOutput(result=input_data.value + 1)


@dataclass(frozen=True)
class _NumpyInput:
    arr: np.ndarray


class _MutatingProcess(PreProcess[_NumpyInput, _DummyOutput]):
    """入力の numpy 配列を書き換える違反プロセス."""

    meta = ProcessMeta(
        name="Mutating", module="pre", document_path="../docs/process-architecture.md"
    )
    _skip_registry = True

    def process(self, input_data: _NumpyInput) -> _DummyOutput:
        input_data.arr[0] = 999  # frozen dataclass でも in-place 変更可能
        return _DummyOutput(result=0)


# --- テスト ---


class TestExecutionContext:
    def test_defaults(self):
        ctx = ExecutionContext()
        assert ctx.dry_run is False
        assert ctx.profile is True
        assert ctx.log_file is None

    def test_custom(self, tmp_path):
        log = tmp_path / "test.log"
        ctx = ExecutionContext(dry_run=True, log_file=log)
        assert ctx.dry_run is True
        assert ctx.log_file == log


class TestProcessRunner:
    def test_run_basic(self):
        runner = ProcessRunner()
        proc = _AddOneProcess()
        result = runner.run(proc, _DummyInput(value=10))
        assert result.result == 11

    def test_run_records_execution(self):
        runner = ProcessRunner()
        proc = _AddOneProcess()
        runner.run(proc, _DummyInput(value=5))
        assert len(runner._execution_log) == 1
        rec = runner._execution_log[0]
        assert rec.process_name == "_AddOneProcess"
        assert rec.elapsed_seconds >= 0
        assert rec.dry_run is False

    def test_dry_run_skips_execution(self):
        ctx = ExecutionContext(dry_run=True)
        runner = ProcessRunner(context=ctx)
        proc = _AddOneProcess()
        result = runner.run(proc, _DummyInput(value=10))
        assert result is None
        assert len(runner._execution_log) == 1
        assert runner._execution_log[0].dry_run is True

    def test_checksum_detects_mutation(self):
        ctx = ExecutionContext(checksum_inputs=True)
        runner = ProcessRunner(context=ctx)
        proc = _MutatingProcess()
        inp = _NumpyInput(arr=np.array([1.0, 2.0, 3.0]))
        with pytest.raises(AssertionError, match="入力データを変更"):
            runner.run(proc, inp)

    def test_checksum_ok_for_pure_process(self):
        ctx = ExecutionContext(checksum_inputs=True)
        runner = ProcessRunner(context=ctx)
        proc = _AddOneProcess()
        result = runner.run(proc, _DummyInput(value=42))
        assert result.result == 43
        assert runner._execution_log[0].checksum_ok is True

    def test_run_pipeline(self):
        runner = ProcessRunner()
        proc1 = _AddOneProcess()
        proc2 = _AddOneProcess()
        results = runner.run_pipeline(
            [
                (proc1, _DummyInput(value=1)),
                (proc2, _DummyInput(value=2)),
            ]
        )
        assert len(results) == 2
        assert results[0].result == 2
        assert results[1].result == 3
        assert len(runner._execution_log) == 2

    def test_get_report(self):
        runner = ProcessRunner()
        proc = _AddOneProcess()
        runner.run(proc, _DummyInput(value=0))
        report = runner.get_report()
        assert "ProcessRunner Report" in report
        assert "_AddOneProcess" in report

    def test_log_file_output(self, tmp_path):
        log_file = tmp_path / "runner.log"
        ctx = ExecutionContext(log_file=log_file)
        runner = ProcessRunner(context=ctx)
        proc = _AddOneProcess()
        runner.run(proc, _DummyInput(value=0))
        content = log_file.read_text()
        assert "_AddOneProcess" in content

    def test_validate_deps_warning(self, caplog):
        """レジストリに登録されていない依存を持つプロセスで警告."""
        import logging

        ctx = ExecutionContext(validate_deps=True)
        runner = ProcessRunner(context=ctx)
        proc = _AddOneProcess()
        # _skip_registry=True なので自分自身がレジストリにない
        # uses が空なので依存チェック自体はパスする
        with caplog.at_level(logging.WARNING):
            runner.run(proc, _DummyInput(value=0))
        # 依存がないのでエラーなし
        assert len(runner._execution_log) == 1

    def test_dry_run_pipeline(self):
        ctx = ExecutionContext(dry_run=True)
        runner = ProcessRunner(context=ctx)
        proc = _AddOneProcess()
        results = runner.run_pipeline(
            [
                (proc, _DummyInput(value=1)),
                (proc, _DummyInput(value=2)),
            ]
        )
        assert all(r is None for r in results)
        assert len(runner._execution_log) == 2


class TestExecutionRecord:
    def test_fields(self):
        rec = ExecutionRecord(
            process_name="Test",
            elapsed_seconds=0.123,
            checksum_before="abc",
            checksum_after="abc",
            checksum_ok=True,
            dry_run=False,
        )
        assert rec.process_name == "Test"
        assert rec.elapsed_seconds == 0.123
        assert rec.checksum_ok is True
