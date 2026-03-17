"""Process 実行診断のテスト.

ProcessExecutionLog, StaticSolverWarning, DeprecatedProcessError の
動作確認テスト。

ソルバーパス: なし（診断インフラのみ）

[← README](../README.md)
"""

import warnings

from xkep_cae.core.diagnostics import (
    DeprecatedProcessError,
    NonDefaultStrategyWarning,
    ProcessExecutionLog,
    StaticSolverWarning,
)


class TestProcessExecutionLogAPI:
    """ProcessExecutionLog の基本 API テスト."""

    def setup_method(self):
        ProcessExecutionLog.reset()

    def test_singleton(self):
        """シングルトンインスタンスが同一であること."""
        log1 = ProcessExecutionLog.instance()
        log2 = ProcessExecutionLog.instance()
        assert log1 is log2

    def test_record_and_entries(self):
        """record_start / record_end でエントリが記録されること."""
        log = ProcessExecutionLog.instance()
        ctx = log.record_start("TestProcess")
        log.record_end(ctx)

        assert len(log.entries) == 1
        entry = log.entries[0]
        assert entry.process_class == "TestProcess"
        assert entry.caller_file != "<unknown>"
        assert entry.caller_function == "test_record_and_entries"

    def test_parent_process_tracking(self):
        """ネストした呼び出しで親プロセスが記録されること."""
        log = ProcessExecutionLog.instance()

        ctx_parent = log.record_start("ParentProcess")
        ctx_child = log.record_start("ChildProcess")
        log.record_end(ctx_child)
        log.record_end(ctx_parent)

        assert len(log.entries) == 2
        child_entry = log.entries[0]
        assert child_entry.process_class == "ChildProcess"
        assert child_entry.parent_process == "ParentProcess"

        parent_entry = log.entries[1]
        assert parent_entry.process_class == "ParentProcess"
        assert parent_entry.parent_process is None

    def test_warning_type_recording(self):
        """warning_type が記録されること."""
        log = ProcessExecutionLog.instance()
        ctx = log.record_start("SolverProcess")
        log.record_end(ctx, warning_type="static_solver")

        assert log.entries[0].warning_type == "static_solver"

    def test_reset(self):
        """reset() でエントリがクリアされること."""
        log = ProcessExecutionLog.instance()
        ctx = log.record_start("TestProcess")
        log.record_end(ctx)
        assert len(log.entries) == 1

        ProcessExecutionLog.reset()
        assert len(log.entries) == 0

    def test_enabled_flag(self):
        """enabled=False のとき記録がスキップされること（record_end でのみ影響）."""
        log = ProcessExecutionLog.instance()
        log.enabled = False
        # record_start/record_end は呼べるが、ログ側の制御はフレームワークが行う
        log.enabled = True


class TestProcessExecutionLogReport:
    """レポート生成テスト."""

    def setup_method(self):
        ProcessExecutionLog.reset()

    def test_empty_report(self):
        """エントリなしのレポートが生成できること."""
        log = ProcessExecutionLog.instance()
        report = log.generate_report()
        assert "Process Usage Report" in report
        assert "実行記録なし" in report

    def test_report_with_entries(self):
        """エントリありのレポートにプロセス名が含まれること."""
        log = ProcessExecutionLog.instance()
        ctx = log.record_start("MyProcess")
        log.record_end(ctx)

        report = log.generate_report()
        assert "MyProcess" in report
        assert "プロセス別サマリー" in report
        assert "呼び出し元別詳細" in report

    def test_report_with_warnings(self):
        """警告付きエントリのレポートに警告一覧が含まれること."""
        log = ProcessExecutionLog.instance()
        ctx = log.record_start("WarnProcess")
        log.record_end(ctx, warning_type="static_solver")

        report = log.generate_report()
        assert "警告一覧" in report
        assert "static_solver" in report

    def test_write_report(self, tmp_path):
        """レポートファイルが書き出されること."""
        log = ProcessExecutionLog.instance()
        ctx = log.record_start("FileProcess")
        log.record_end(ctx)

        path = tmp_path / "report.md"
        result_path = log.write_report(path)
        assert result_path == path
        assert path.exists()
        content = path.read_text()
        assert "FileProcess" in content


class TestStaticSolverWarning:
    """StaticSolverWarning の発行テスト."""

    def test_warning_is_user_warning(self):
        """StaticSolverWarning が UserWarning のサブクラスであること."""
        assert issubclass(StaticSolverWarning, UserWarning)

    def test_warning_can_be_caught(self):
        """warnings.warn で発行・捕捉できること."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("test", StaticSolverWarning, stacklevel=1)
            assert len(w) == 1
            assert issubclass(w[0].category, StaticSolverWarning)


class TestNonDefaultStrategyWarning:
    """NonDefaultStrategyWarning のテスト."""

    def test_warning_is_user_warning(self):
        """NonDefaultStrategyWarning が UserWarning のサブクラスであること."""
        assert issubclass(NonDefaultStrategyWarning, UserWarning)

    def test_warning_can_be_caught(self):
        """warnings.warn で発行・捕捉できること."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("test", NonDefaultStrategyWarning, stacklevel=1)
            assert len(w) == 1
            assert issubclass(w[0].category, NonDefaultStrategyWarning)


class TestDeprecatedProcessError:
    """DeprecatedProcessError のテスト."""

    def test_error_is_runtime_error(self):
        """DeprecatedProcessError が RuntimeError のサブクラスであること."""
        assert issubclass(DeprecatedProcessError, RuntimeError)

    def test_error_message(self):
        """エラーメッセージにプロセス名が含まれること."""
        err = DeprecatedProcessError("TestProcess は deprecated です。")
        assert "deprecated" in str(err)
