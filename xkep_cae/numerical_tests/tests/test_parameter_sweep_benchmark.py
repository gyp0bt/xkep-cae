"""ParameterSweepBenchmarkProcess のテスト.

C3 契約: @binds_to 紐付け + API テスト。
実際の solver は走らせず、軽量ダミープロセスで掃引ロジックだけを検証する。

[← README](../../../README.md)
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from xkep_cae.core import binds_to
from xkep_cae.core.base import ProcessMeta
from xkep_cae.core.benchmark import BenchmarkRunResult
from xkep_cae.core.categories import PreProcess
from xkep_cae.numerical_tests.parameter_sweep_benchmark import (
    ParameterSweepBenchmarkInput,
    ParameterSweepBenchmarkProcess,
    ParameterSweepBenchmarkResult,
    _collect_uses_graph,
    _first_leaf_breakdown_entry,
    _is_leaf_process,
)

# --- テスト用ダミープロセス（軽量スケーラブル） -------------------------------


@dataclass(frozen=True)
class _SweepConfig:
    n: int = 1
    label: str = "default"


@dataclass(frozen=True)
class _SweepResult:
    value: int
    label: str


class _SweepTargetProcess(PreProcess[_SweepConfig, _SweepResult]):
    """`n` に比例して軽負荷をこなすダミープロセス."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="_SweepTargetProcess",
        module="pre",
        version="0.1.0",
        document_path="../docs/parameter_sweep_benchmark.md",
        stability="experimental",
        support_tier="dev-only",
    )
    _skip_registry = True

    def process(self, input_data: _SweepConfig) -> _SweepResult:
        # n に比例した微小スリープで profile 差分を確実に生む
        time.sleep(0.001 * max(1, input_data.n))
        return _SweepResult(value=input_data.n * 2, label=input_data.label)


class _SweepWrapperProcess(PreProcess[_SweepConfig, _SweepResult]):
    """`_SweepTargetProcess` をネストして呼ぶ wrapper（status-317 葉判定テスト用）."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="_SweepWrapperProcess",
        module="pre",
        version="0.1.0",
        document_path="../docs/parameter_sweep_benchmark.md",
        stability="experimental",
        support_tier="dev-only",
    )
    uses: ClassVar[list[type]] = [_SweepTargetProcess]
    _skip_registry = True

    def __init__(self) -> None:
        self._inner = _SweepTargetProcess()

    def process(self, input_data: _SweepConfig) -> _SweepResult:
        # wrapper 自身も若干時間を使うが、子プロセスの方がほぼ全時間を占める
        time.sleep(0.0005)
        return self._inner.process(input_data)


# --- API テスト --------------------------------------------------------------


@binds_to(ParameterSweepBenchmarkProcess)
class TestParameterSweepBenchmarkProcessAPI:
    """ParameterSweepBenchmarkProcess の API テスト."""

    def test_meta_name(self) -> None:
        """meta.name が正しい."""
        assert ParameterSweepBenchmarkProcess.meta.name == "ParameterSweepBenchmark"

    def test_uses_declared(self) -> None:
        """uses に BenchmarkRunnerProcess が宣言されている."""
        names = [u.__name__ for u in ParameterSweepBenchmarkProcess.uses]
        assert "BenchmarkRunnerProcess" in names

    def test_sweep_runs_each_value(self) -> None:
        """param_values の各要素について target_process が 1 回ずつ実行される."""
        cfg = _SweepConfig(n=1, label="base")
        proc = _SweepTargetProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_input = ParameterSweepBenchmarkInput(
                target_process=proc,
                base_config=cfg,
                param_name="n",
                param_values=(2, 5, 8),
                output_dir=tmpdir,
            )
            result = ParameterSweepBenchmarkProcess().process(sweep_input)

            assert isinstance(result, ParameterSweepBenchmarkResult)
            assert result.param_name == "n"
            assert result.param_values == (2, 5, 8)
            assert len(result.cases) == 3
            # 各ケースは BenchmarkRunResult
            assert all(isinstance(c, BenchmarkRunResult) for c in result.cases)

            # 各ケースの config は掃引値で差し替わっている
            case_ns = [c.manifest.config_params["n"] for c in result.cases]
            assert case_ns == [2, 5, 8]

            # 各ケースの元プロセス結果も掃引値を反映
            case_results = [c.result.value for c in result.cases]
            assert case_results == [4, 10, 16]

    def test_summary_rows_contain_dominant_process(self) -> None:
        """summary_rows に dominant_process と dominant_pct が記録される."""
        cfg = _SweepConfig()
        proc = _SweepTargetProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_input = ParameterSweepBenchmarkInput(
                target_process=proc,
                base_config=cfg,
                param_name="n",
                param_values=(1, 4),
                output_dir=tmpdir,
            )
            result = ParameterSweepBenchmarkProcess().process(sweep_input)

            assert len(result.summary_rows) == 2
            for row, expected_value in zip(result.summary_rows, (1, 4), strict=True):
                assert row["param_name"] == "n"
                assert row["value"] == expected_value
                assert row["elapsed_seconds"] >= 0.0
                # profile_breakdown 有効なので dominant_process は空文字でないはず
                assert row["dominant_process"]
                assert row["dominant_pct"] >= 0.0
                assert row["manifest_path"] is not None

    def test_summary_yaml_written(self) -> None:
        """集約サマリ YAML が output_dir に保存される."""
        cfg = _SweepConfig()
        proc = _SweepTargetProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_input = ParameterSweepBenchmarkInput(
                target_process=proc,
                base_config=cfg,
                param_name="n",
                param_values=(1, 2),
                output_dir=tmpdir,
                status_file="docs/status/status-315.md",
            )
            result = ParameterSweepBenchmarkProcess().process(sweep_input)

            assert result.summary_yaml_path is not None
            path = Path(result.summary_yaml_path)
            assert path.exists()
            content = path.read_text()
            assert "ParameterSweepBenchmarkProcess" in content
            assert "_SweepTargetProcess" in content
            assert "param_name: n" in content
            assert "cases:" in content
            assert "status-315.md" in content

    def test_each_case_gets_own_manifest(self) -> None:
        """ケースごとに BenchmarkRunnerProcess の manifest YAML が生成される."""
        cfg = _SweepConfig()
        proc = _SweepTargetProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_input = ParameterSweepBenchmarkInput(
                target_process=proc,
                base_config=cfg,
                param_name="n",
                param_values=(1, 2, 3),
                output_dir=tmpdir,
            )
            result = ParameterSweepBenchmarkProcess().process(sweep_input)

            manifest_paths = [c.manifest_path for c in result.cases]
            assert len(manifest_paths) == 3
            assert all(p is not None for p in manifest_paths)
            # status-315: 同一秒内スイープでも BenchmarkRunnerProcess._save_manifest
            # の連番フォールバックで衝突回避されるはず。
            assert len({p for p in manifest_paths}) == 3

    def test_result_extractors_passed_through(self) -> None:
        """result_extractors が各ケースの manifest.results_summary に反映される."""
        cfg = _SweepConfig()
        proc = _SweepTargetProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_input = ParameterSweepBenchmarkInput(
                target_process=proc,
                base_config=cfg,
                param_name="n",
                param_values=(3, 7),
                result_extractors={
                    "value": lambda r: r.value,
                    "label": lambda r: r.label,
                },
                output_dir=tmpdir,
            )
            result = ParameterSweepBenchmarkProcess().process(sweep_input)

            values = [c.manifest.results_summary["value"] for c in result.cases]
            labels = [c.manifest.results_summary["label"] for c in result.cases]
            assert values == [6, 14]
            assert labels == ["default", "default"]

    def test_empty_param_values_raises(self) -> None:
        """param_values が空なら ValueError."""
        cfg = _SweepConfig()
        proc = _SweepTargetProcess()

        import pytest

        sweep_input = ParameterSweepBenchmarkInput(
            target_process=proc,
            base_config=cfg,
            param_name="n",
            param_values=(),
        )
        with pytest.raises(ValueError, match="param_values"):
            ParameterSweepBenchmarkProcess().process(sweep_input)

    def test_invalid_param_name_raises(self) -> None:
        """base_config に存在しないフィールド名なら ValueError."""
        cfg = _SweepConfig()
        proc = _SweepTargetProcess()

        import pytest

        sweep_input = ParameterSweepBenchmarkInput(
            target_process=proc,
            base_config=cfg,
            param_name="not_a_field",
            param_values=(1, 2),
        )
        with pytest.raises(ValueError, match="not_a_field"):
            ParameterSweepBenchmarkProcess().process(sweep_input)

    def test_non_dataclass_base_raises(self) -> None:
        """base_config が dataclass でないなら TypeError."""
        proc = _SweepTargetProcess()

        import pytest

        sweep_input = ParameterSweepBenchmarkInput(
            target_process=proc,
            base_config={"n": 1},  # ただの dict → NG
            param_name="n",
            param_values=(1,),
        )
        with pytest.raises(TypeError, match="frozen dataclass"):
            ParameterSweepBenchmarkProcess().process(sweep_input)

    # --- status-317: dominant_leaf_process 追加検証 --------------------------

    def test_summary_rows_include_dominant_leaf_fields(self) -> None:
        """summary_rows に dominant_leaf_process/_pct/_total が載る."""
        cfg = _SweepConfig()
        proc = _SweepTargetProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_input = ParameterSweepBenchmarkInput(
                target_process=proc,
                base_config=cfg,
                param_name="n",
                param_values=(1, 3),
                output_dir=tmpdir,
            )
            result = ParameterSweepBenchmarkProcess().process(sweep_input)

            for row in result.summary_rows:
                assert "dominant_leaf_process" in row
                assert "dominant_leaf_pct" in row
                assert "dominant_leaf_total" in row
                # 葉 target だけ走る構成なので leaf = dominant となるはず
                assert row["dominant_leaf_process"] == row["dominant_process"]
                assert row["dominant_leaf_pct"] >= 0.0
                assert row["dominant_leaf_total"] >= 0.0

    def test_dominant_leaf_skips_wrapper(self) -> None:
        """wrapper を target にしても dominant_leaf は葉を指す."""
        cfg = _SweepConfig()
        proc = _SweepWrapperProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_input = ParameterSweepBenchmarkInput(
                target_process=proc,
                base_config=cfg,
                param_name="n",
                param_values=(2, 4),
                output_dir=tmpdir,
            )
            result = ParameterSweepBenchmarkProcess().process(sweep_input)

            for row in result.summary_rows:
                # dominant_process は wrapper 自身（inclusive 時間で先頭）
                assert row["dominant_process"] == "_SweepWrapperProcess"
                # dominant_leaf_process は uses=[] の葉
                assert row["dominant_leaf_process"] == "_SweepTargetProcess"
                # 葉の total は wrapper 以下（inclusive より短い）
                assert row["dominant_leaf_total"] > 0.0

    def test_summary_yaml_contains_leaf_fields(self) -> None:
        """集約 YAML に dominant_leaf_process/_pct/_total が書き出される."""
        cfg = _SweepConfig()
        proc = _SweepWrapperProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_input = ParameterSweepBenchmarkInput(
                target_process=proc,
                base_config=cfg,
                param_name="n",
                param_values=(1,),
                output_dir=tmpdir,
            )
            result = ParameterSweepBenchmarkProcess().process(sweep_input)

            assert result.summary_yaml_path is not None
            content = Path(result.summary_yaml_path).read_text()
            assert "dominant_leaf_process" in content
            assert "_SweepTargetProcess" in content
            assert "dominant_leaf_pct" in content
            assert "dominant_leaf_total" in content


# --- status-317: 葉判定ヘルパ単体テスト --------------------------------------


class TestDominantLeafHelpers:
    """`_collect_uses_graph` / `_is_leaf_process` / `_first_leaf_breakdown_entry`."""

    def test_collect_uses_graph_includes_root(self) -> None:
        graph = _collect_uses_graph(_SweepTargetProcess)
        assert "_SweepTargetProcess" in graph
        # uses=[] の葉なので自分自身のみ
        assert graph["_SweepTargetProcess"] is _SweepTargetProcess

    def test_collect_uses_graph_traverses_uses(self) -> None:
        graph = _collect_uses_graph(_SweepWrapperProcess)
        assert "_SweepWrapperProcess" in graph
        assert "_SweepTargetProcess" in graph

    def test_is_leaf_process_uses_empty_is_leaf(self) -> None:
        graph = _collect_uses_graph(_SweepWrapperProcess)
        assert _is_leaf_process("_SweepTargetProcess", graph) is True

    def test_is_leaf_process_uses_nonempty_is_wrapper(self) -> None:
        graph = _collect_uses_graph(_SweepWrapperProcess)
        assert _is_leaf_process("_SweepWrapperProcess", graph) is False

    def test_is_leaf_process_unknown_is_treated_as_leaf(self) -> None:
        """registry 外のプロセスは保守的に葉扱い."""
        graph: dict[str, type] = {}
        assert _is_leaf_process("SomeUnknownProcess", graph) is True

    def test_first_leaf_breakdown_entry_skips_wrapper(self) -> None:
        graph = _collect_uses_graph(_SweepWrapperProcess)
        breakdown = (
            {"name": "_SweepWrapperProcess", "pct": 55.0, "total": 0.01},
            {"name": "_SweepTargetProcess", "pct": 45.0, "total": 0.008},
        )
        name, pct, total = _first_leaf_breakdown_entry(breakdown, graph)
        assert name == "_SweepTargetProcess"
        assert pct == 45.0
        assert total == 0.008

    def test_first_leaf_breakdown_entry_empty(self) -> None:
        name, pct, total = _first_leaf_breakdown_entry((), {})
        assert name == ""
        assert pct == 0.0
        assert total == 0.0

    def test_first_leaf_breakdown_entry_all_wrappers_returns_empty(self) -> None:
        """全部 wrapper なら空返り（葉未検出）."""
        # _SweepWrapperProcess だけ含むグラフなので、これは wrapper 扱い
        graph = {"_SweepWrapperProcess": _SweepWrapperProcess}
        breakdown = ({"name": "_SweepWrapperProcess", "pct": 100.0, "total": 1.0},)
        name, pct, total = _first_leaf_breakdown_entry(breakdown, graph)
        assert name == ""
        assert pct == 0.0
        assert total == 0.0
