"""ParameterSweepBenchmarkProcess — frozen dataclass の 1 フィールド掃引 + profile 集約.

status-315: BenchmarkRunnerProcess（status-314 で profile_breakdown 自動記録
を獲得）を n 本撚線スケール走査に適用するための汎用薄ラッパー Process。

1 回の `process()` 呼び出しで `param_values` ぶんの BenchmarkRunnerProcess 走査
を直列実行し、ケースごとの manifest + dominant process を集約サマリ YAML に
まとめる。

[← README](../../README.md)
"""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Callable
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, ClassVar

from xkep_cae.core.base import ProcessMeta
from xkep_cae.core.benchmark import (
    BenchmarkRunInput,
    BenchmarkRunnerProcess,
    BenchmarkRunResult,
    _dict_to_yaml,
)
from xkep_cae.core.categories import BatchProcess

# ====================================================================
# 入出力データ
# ====================================================================


@dataclass(frozen=True)
class ParameterSweepBenchmarkInput:
    """ParameterSweepBenchmarkProcess の入力.

    Attributes:
        target_process: 掃引対象のプロセス（AbstractProcess 互換）。
        base_config: 掃引元となる frozen dataclass。各ケースで
            `dataclasses.replace(base_config, **{param_name: value})`
            により値を差し替える。
        param_name: 掃引対象フィールド名（`base_config` のフィールド名）。
        param_values: 差し替え値の列。空 tuple は禁止。
        result_extractors: `BenchmarkRunInput.result_extractors` にそのまま
            渡す。各ケースで同じ抽出器が適用される。
        output_dir: 個別/集約マニフェストの保存先。None なら
            `docs/benchmarks/` が使用される（BenchmarkRunnerProcess と揃え）。
        status_file: マニフェストに記録する status ドキュメントリンク。
        profile_sort_by: `BenchmarkRunInput.profile_sort_by` に転送。
        profile_top_n: `BenchmarkRunInput.profile_top_n` に転送。
    """

    target_process: Any
    base_config: Any
    param_name: str
    param_values: tuple[Any, ...]
    result_extractors: dict[str, Callable] = field(default_factory=dict)
    output_dir: str | None = None
    status_file: str | None = None
    profile_sort_by: str = "total"
    profile_top_n: int | None = None


@dataclass(frozen=True)
class ParameterSweepBenchmarkResult:
    """ParameterSweepBenchmarkProcess の出力.

    Attributes:
        param_name: 掃引したフィールド名（入力のコピー）。
        param_values: 掃引した値列（入力のコピー）。
        cases: 各ケースの BenchmarkRunResult。`param_values` と同じ順序。
        summary_rows: ケースごとの集約行。YAML にもそのまま保存される。
            各行のキー: `param_name`, `value`, `elapsed_seconds`,
            `dominant_process`, `dominant_pct`, `manifest_path`。
        summary_yaml_path: 集約 YAML の保存先。失敗時 None。
    """

    param_name: str
    param_values: tuple[Any, ...]
    cases: tuple[BenchmarkRunResult, ...]
    summary_rows: tuple[dict[str, Any], ...]
    summary_yaml_path: str | None = None


# ====================================================================
# Process
# ====================================================================


class ParameterSweepBenchmarkProcess(
    BatchProcess[ParameterSweepBenchmarkInput, ParameterSweepBenchmarkResult]
):
    """frozen dataclass フィールドを掃引し、各ケースの profile_breakdown を集約.

    status-314 の BenchmarkRunnerProcess をそのまま再利用し、掃引ループと
    サマリ YAML 保存だけを上乗せする薄いラッパー。個別ケースの manifest YAML
    は BenchmarkRunnerProcess の既存機構がそのまま書き出す。
    """

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="ParameterSweepBenchmark",
        module="batch",
        version="0.1.0",
        document_path="docs/parameter_sweep_benchmark.md",
        stability="experimental",
        support_tier="ci-required",
    )

    uses: ClassVar[list[type]] = [BenchmarkRunnerProcess]

    def process(self, input_data: ParameterSweepBenchmarkInput) -> ParameterSweepBenchmarkResult:
        """掃引走査を実行し集約サマリを生成する."""
        if not input_data.param_values:
            raise ValueError("param_values は空にできません")
        if not is_dataclass(input_data.base_config):
            raise TypeError("base_config は frozen dataclass である必要があります")

        base_fields = {f.name for f in dataclasses.fields(input_data.base_config)}
        if input_data.param_name not in base_fields:
            raise ValueError(
                f"param_name={input_data.param_name!r} は base_config "
                f"({type(input_data.base_config).__name__}) のフィールドではありません"
            )

        runner = BenchmarkRunnerProcess()
        cases: list[BenchmarkRunResult] = []
        summary_rows: list[dict[str, Any]] = []

        for value in input_data.param_values:
            case_config = dataclasses.replace(
                input_data.base_config, **{input_data.param_name: value}
            )
            run_input = BenchmarkRunInput(
                process=input_data.target_process,
                config=case_config,
                result_extractors=dict(input_data.result_extractors),
                status_file=input_data.status_file,
                output_dir=input_data.output_dir,
                capture_profile=True,
                profile_sort_by=input_data.profile_sort_by,
                profile_top_n=input_data.profile_top_n,
            )
            case_result = runner.process(run_input)
            cases.append(case_result)

            breakdown = case_result.manifest.profile_breakdown
            if breakdown:
                dominant = breakdown[0]
                dominant_name = str(dominant.get("name", ""))
                dominant_pct = float(dominant.get("pct", 0.0))
            else:
                dominant_name = ""
                dominant_pct = 0.0

            summary_rows.append(
                {
                    "param_name": input_data.param_name,
                    "value": _scalarize(value),
                    "elapsed_seconds": case_result.manifest.elapsed_seconds,
                    "dominant_process": dominant_name,
                    "dominant_pct": round(dominant_pct, 3),
                    "manifest_path": case_result.manifest_path,
                }
            )

        summary_yaml_path = self._save_summary(
            input_data=input_data,
            summary_rows=tuple(summary_rows),
        )

        return ParameterSweepBenchmarkResult(
            param_name=input_data.param_name,
            param_values=tuple(input_data.param_values),
            cases=tuple(cases),
            summary_rows=tuple(summary_rows),
            summary_yaml_path=summary_yaml_path,
        )

    # ----------------------------------------------------------------
    # 集約サマリ保存
    # ----------------------------------------------------------------

    def _save_summary(
        self,
        *,
        input_data: ParameterSweepBenchmarkInput,
        summary_rows: tuple[dict[str, Any], ...],
    ) -> str | None:
        """集約サマリ YAML を output_dir に保存."""
        try:
            base = Path(input_data.output_dir) if input_data.output_dir else Path("docs/benchmarks")
            base.mkdir(parents=True, exist_ok=True)

            ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
            filename = f"ParameterSweepBenchmark_{ts}.yaml"
            path = base / filename

            target_cls_name = type(input_data.target_process).__name__
            base_cls_name = type(input_data.base_config).__name__
            doc: dict[str, Any] = {
                "process": type(self).__name__,
                "target_process": target_cls_name,
                "base_config_type": base_cls_name,
                "param_name": input_data.param_name,
                "param_values": [_scalarize(v) for v in input_data.param_values],
                "profile_sort_by": input_data.profile_sort_by,
                "profile_top_n": (
                    input_data.profile_top_n if input_data.profile_top_n is not None else "all"
                ),
                "status_file": input_data.status_file,
                "cases": [dict(row) for row in summary_rows],
            }
            path.write_text(_dict_to_yaml(doc), encoding="utf-8")
            return str(path)
        except Exception:
            return None


def _scalarize(value: Any) -> Any:
    """YAML シリアライズ可能な単純型に落とし込む."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)
