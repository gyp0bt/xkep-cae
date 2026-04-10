"""ProcessMetaclass プロファイル統計 API のテスト.

`snapshot_profile` / `get_profile_stats` / `get_profile_report` の動作確認。
大規模撚線プロファイリング基盤の最小単位テスト。

[← README](../README.md)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar

from xkep_cae.core.base import ProcessMeta, ProcessMetaclass
from xkep_cae.core.categories import BatchProcess, PreProcess
from xkep_cae.core.testing import binds_to

# --- テスト用ダミープロセス（異なる滞在時間を持つ3種） ---


@dataclass(frozen=True)
class _SleepConfig:
    seconds: float = 0.0


@dataclass(frozen=True)
class _SleepResult:
    slept: float


class _FastProcess(PreProcess["_SleepConfig", "_SleepResult"]):
    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="_FastProcess",
        module="pre",
        version="0.1.0",
        document_path="../xkep_cae/core/docs/benchmark_runner.md",
        stability="experimental",
        support_tier="dev-only",
    )
    _skip_registry = True

    def process(self, input_data: _SleepConfig) -> _SleepResult:
        time.sleep(input_data.seconds)
        return _SleepResult(slept=input_data.seconds)


class _SlowProcess(PreProcess["_SleepConfig", "_SleepResult"]):
    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="_SlowProcess",
        module="pre",
        version="0.1.0",
        document_path="../xkep_cae/core/docs/benchmark_runner.md",
        stability="experimental",
        support_tier="dev-only",
    )
    _skip_registry = True

    def process(self, input_data: _SleepConfig) -> _SleepResult:
        time.sleep(input_data.seconds)
        return _SleepResult(slept=input_data.seconds)


# --- get_profile_stats API テスト ---


@binds_to(_FastProcess)
class TestProfileStatsAPI:
    """`get_profile_stats` の構造化統計 API テスト."""

    def test_snapshot_returns_call_counts(self):
        """snapshot_profile は呼び出し回数の dict を返す."""
        ProcessMetaclass.reset_profile()
        _FastProcess().process(_SleepConfig(seconds=0.001))
        _FastProcess().process(_SleepConfig(seconds=0.001))

        snap = ProcessMetaclass.snapshot_profile()
        assert snap["_FastProcess"] == 2

    def test_stats_contains_expected_keys(self):
        """統計エントリに全フィールドが含まれる."""
        ProcessMetaclass.reset_profile()
        _FastProcess().process(_SleepConfig(seconds=0.001))

        stats = ProcessMetaclass.get_profile_stats()
        assert len(stats) >= 1
        row = next(s for s in stats if s["name"] == "_FastProcess")
        for key in ("name", "n", "total", "avg", "min", "max", "median", "pct"):
            assert key in row

    def test_sorted_by_total_descending(self):
        """sort_by='total' で合計時間降順にソート."""
        ProcessMetaclass.reset_profile()
        # Slow を 1 回、Fast を 1 回（Slow のほうが長い）
        _SlowProcess().process(_SleepConfig(seconds=0.010))
        _FastProcess().process(_SleepConfig(seconds=0.001))

        stats = ProcessMetaclass.get_profile_stats(sort_by="total")
        # 先頭は Slow（合計時間最大）
        slow_idx = next(i for i, s in enumerate(stats) if s["name"] == "_SlowProcess")
        fast_idx = next(i for i, s in enumerate(stats) if s["name"] == "_FastProcess")
        assert slow_idx < fast_idx

    def test_since_filters_historical_calls(self):
        """since スナップショット指定でそれ以降のみ集計."""
        ProcessMetaclass.reset_profile()
        _FastProcess().process(_SleepConfig(seconds=0.001))
        _FastProcess().process(_SleepConfig(seconds=0.001))

        snap = ProcessMetaclass.snapshot_profile()
        # スナップショット以降に追加呼び出し
        _FastProcess().process(_SleepConfig(seconds=0.001))

        stats = ProcessMetaclass.get_profile_stats(since=snap)
        row = next(s for s in stats if s["name"] == "_FastProcess")
        assert int(row["n"]) == 1  # snap 以降の1回のみ

    def test_pct_sums_to_100(self):
        """pct フィールドの合計はほぼ100%."""
        ProcessMetaclass.reset_profile()
        _FastProcess().process(_SleepConfig(seconds=0.001))
        _SlowProcess().process(_SleepConfig(seconds=0.005))

        stats = ProcessMetaclass.get_profile_stats()
        total_pct = sum(float(s["pct"]) for s in stats)
        assert abs(total_pct - 100.0) < 0.01

    def test_sort_by_name(self):
        """sort_by='name' で名前昇順."""
        ProcessMetaclass.reset_profile()
        _SlowProcess().process(_SleepConfig(seconds=0.001))
        _FastProcess().process(_SleepConfig(seconds=0.001))

        stats = ProcessMetaclass.get_profile_stats(sort_by="name")
        names = [s["name"] for s in stats]
        assert names == sorted(names)


# --- get_profile_report API テスト ---


@binds_to(_SlowProcess)
class TestProfileReportAPI:
    """`get_profile_report` テキスト出力 API テスト."""

    def test_report_contains_header(self):
        """レポートにヘッダ行が含まれる."""
        ProcessMetaclass.reset_profile()
        _FastProcess().process(_SleepConfig(seconds=0.001))

        report = ProcessMetaclass.get_profile_report()
        assert "Process Profile Report" in report
        assert "process" in report
        assert "total" in report
        assert "pct" in report

    def test_report_top_n_limits_rows(self):
        """top_n で表示件数を制限."""
        ProcessMetaclass.reset_profile()
        _FastProcess().process(_SleepConfig(seconds=0.001))
        _SlowProcess().process(_SleepConfig(seconds=0.002))

        full = ProcessMetaclass.get_profile_report()
        limited = ProcessMetaclass.get_profile_report(top_n=1)

        # 制限版は _FastProcess または _SlowProcess のいずれか1件のみ
        assert full.count("_FastProcess") + full.count("_SlowProcess") >= 2
        assert limited.count("_FastProcess") + limited.count("_SlowProcess") == 1

    def test_report_with_snapshot(self):
        """since 指定でスナップショット以降のみレポートに反映."""
        ProcessMetaclass.reset_profile()
        _FastProcess().process(_SleepConfig(seconds=0.001))
        snap = ProcessMetaclass.snapshot_profile()
        _SlowProcess().process(_SleepConfig(seconds=0.001))

        report = ProcessMetaclass.get_profile_report(since=snap)
        # スナップショット時点で _FastProcess は既存 → 以降は 0 件 → レポート非登場
        assert "_SlowProcess" in report
        assert "_FastProcess" not in report


# --- wrapper / leaf 分類テスト（status-317） -----------------------------------


class _LeafWorkerProcess(PreProcess["_SleepConfig", "_SleepResult"]):
    """子プロセスを一切呼ばない葉 Process."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="_LeafWorkerProcess",
        module="pre",
        version="0.1.0",
        document_path="../xkep_cae/core/docs/benchmark_runner.md",
        stability="experimental",
        support_tier="dev-only",
    )
    _skip_registry = True

    def process(self, input_data: _SleepConfig) -> _SleepResult:
        time.sleep(input_data.seconds)
        return _SleepResult(slept=input_data.seconds)


class _WrapperBatchProcess(BatchProcess["_SleepConfig", "_SleepResult"]):
    """`_LeafWorkerProcess` を内部で呼び出す wrapper Process."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="_WrapperBatchProcess",
        module="batch",
        version="0.1.0",
        document_path="../xkep_cae/core/docs/benchmark_runner.md",
        stability="experimental",
        support_tier="dev-only",
    )
    _skip_registry = True
    uses: ClassVar[list] = [_LeafWorkerProcess]

    def process(self, input_data: _SleepConfig) -> _SleepResult:
        leaf = _LeafWorkerProcess()
        return leaf.process(input_data)


@binds_to(_LeafWorkerProcess)
class TestProfileWrapperClassification:
    """`is_wrapper` フラグと `_wrapper_classes` 追跡のテスト."""

    def test_leaf_only_call_has_no_wrapper(self) -> None:
        """葉 Process を単独呼び出しすると wrapper 判定は False."""
        ProcessMetaclass.reset_profile()
        _LeafWorkerProcess().process(_SleepConfig(seconds=0.001))

        stats = ProcessMetaclass.get_profile_stats()
        row = next(s for s in stats if s["name"] == "_LeafWorkerProcess")
        assert "is_wrapper" in row
        assert row["is_wrapper"] is False

    def test_wrapper_calling_leaf_marks_wrapper_true(self) -> None:
        """wrapper → leaf の呼び出しで wrapper は is_wrapper=True、leaf は False."""
        ProcessMetaclass.reset_profile()
        _WrapperBatchProcess().process(_SleepConfig(seconds=0.001))

        stats = ProcessMetaclass.get_profile_stats()
        wrapper_row = next(s for s in stats if s["name"] == "_WrapperBatchProcess")
        leaf_row = next(s for s in stats if s["name"] == "_LeafWorkerProcess")

        assert wrapper_row["is_wrapper"] is True
        assert leaf_row["is_wrapper"] is False

    def test_reset_profile_clears_wrapper_classes(self) -> None:
        """reset_profile で wrapper 集合もクリアされる."""
        ProcessMetaclass.reset_profile()
        _WrapperBatchProcess().process(_SleepConfig(seconds=0.001))
        assert "_WrapperBatchProcess" in ProcessMetaclass._wrapper_classes

        ProcessMetaclass.reset_profile()
        assert "_WrapperBatchProcess" not in ProcessMetaclass._wrapper_classes

        # reset 後に葉だけ呼び出せば wrapper 集合は空のまま
        _LeafWorkerProcess().process(_SleepConfig(seconds=0.001))
        assert "_LeafWorkerProcess" not in ProcessMetaclass._wrapper_classes

    def test_wrapper_flag_sticks_across_snapshots(self) -> None:
        """`is_wrapper` は一度立てば `since` snapshot 以降でも True のまま."""
        ProcessMetaclass.reset_profile()
        # 最初の wrapper 呼び出しで _WrapperBatchProcess を wrapper 認定
        _WrapperBatchProcess().process(_SleepConfig(seconds=0.001))
        snap = ProcessMetaclass.snapshot_profile()
        # snap 以降で再度 wrapper を呼ぶ
        _WrapperBatchProcess().process(_SleepConfig(seconds=0.001))

        stats = ProcessMetaclass.get_profile_stats(since=snap)
        wrapper_row = next(s for s in stats if s["name"] == "_WrapperBatchProcess")
        assert wrapper_row["is_wrapper"] is True
