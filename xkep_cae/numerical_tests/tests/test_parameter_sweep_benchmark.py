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
