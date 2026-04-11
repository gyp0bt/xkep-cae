"""ParameterSweepBenchmarkProcess — frozen dataclass の 1 フィールド掃引 + profile 集約.

status-315: BenchmarkRunnerProcess（status-314 で profile_breakdown 自動記録
を獲得）を n 本撚線スケール走査に適用するための汎用薄ラッパー Process。

1 回の `process()` 呼び出しで `param_values` ぶんの BenchmarkRunnerProcess 走査
を直列実行し、ケースごとの manifest + dominant process を集約サマリ YAML に
まとめる。

status-317: wrapper/nested process に隠された真のボトルネックを特定するため
`summary_rows` に `dominant_leaf_process` / `dominant_leaf_pct` /
`dominant_leaf_total` を追加。`uses` グラフを `target_process` から再帰走査
し、`uses` が空のクラスを葉とみなす（registry 非依存）。

## 使い方

`BenchmarkRunResult` の各ケースから欲しい情報を取り出すサンプル。
**`BenchmarkRunResult` の属性名は `result` / `manifest` / `manifest_path`** で、
結果サマリは `case.manifest.results_summary` に入る（`case.extracted` ではない）。

```python
from xkep_cae.numerical_tests.parameter_sweep_benchmark import (
    ParameterSweepBenchmarkInput,
    ParameterSweepBenchmarkProcess,
)

sweep = ParameterSweepBenchmarkProcess().process(
    ParameterSweepBenchmarkInput(
        target_process=my_process,
        base_config=my_config,
        param_name="n_strands",
        param_values=(7, 19, 37),
        result_extractors={
            "n_incr": lambda r: r.n_increments,
            "converged": lambda r: r.converged,
        },
    )
)

# 集約 summary 表示（dominant_leaf_process が真のボトルネックを示す）
for row in sweep.summary_rows:
    print(
        f"{row['param_name']}={row['value']}: "
        f"elapsed={row['elapsed_seconds']:.2f}s  "
        f"dominant={row['dominant_process']}({row['dominant_pct']:.1f}%)  "
        f"leaf={row['dominant_leaf_process']}"
        f"({row['dominant_leaf_pct']:.1f}%, "
        f"{row['dominant_leaf_total']:.2f}s)"
    )

# ケース個別の詳細（extractors 値は case.manifest.results_summary）
for case in sweep.cases:
    print(case.manifest_path, case.manifest.results_summary)
```

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
            `dominant_process`, `dominant_pct`, `dominant_leaf_process`,
            `dominant_leaf_pct`, `dominant_leaf_total`, `manifest_path`。

            `dominant_process` は `profile_breakdown` 先頭（wrapper 含む
            inclusive 時間最大）、`dominant_leaf_process` は `uses` が空の
            葉プロセスのうち先頭（真のボトルネック）。status-317 で追加。
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

        # status-317: target_process の uses グラフを再帰走査して
        # 「葉プロセス」判定用のクラス辞書を作る（registry 非依存）。
        known_classes = _collect_uses_graph(type(input_data.target_process))

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

            # status-317: dominant 葉プロセス抽出（wrapper を読み飛ばす）
            leaf_name, leaf_pct, leaf_total = _first_leaf_breakdown_entry(breakdown, known_classes)

            summary_rows.append(
                {
                    "param_name": input_data.param_name,
                    "value": _scalarize(value),
                    "elapsed_seconds": case_result.manifest.elapsed_seconds,
                    "dominant_process": dominant_name,
                    "dominant_pct": round(dominant_pct, 3),
                    "dominant_leaf_process": leaf_name,
                    "dominant_leaf_pct": round(leaf_pct, 3),
                    "dominant_leaf_total": round(leaf_total, 6),
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


def _collect_uses_graph(root_cls: type) -> dict[str, type]:
    """root_cls の `uses` グラフを再帰走査し class_name -> class の辞書を返す.

    status-317: `dominant_leaf_process` 判定のために、target_process の
    クラスから宣言上到達可能な全 Process クラスを name 引きできるよう集約する。
    ProcessRegistry には依存しないので、テスト用 `_skip_registry=True` の
    ダミープロセス（registry に載らない）でも正しく葉判定できる。

    status-320: `StrategySlot.default_types` も静的依存として扱い、Strategy 経由
    で注入される Process（例: `ContactFrictionProcess.contact_force_slot` →
    `HuberContactForceProcess` → `ContactForceStStiffnessProcess`）まで到達可能に
    拡張した。インスタンス化しなくても宣言上の uses グラフから n² スケーリング
    プロセスを dominant_leaf_process 側が抽出できる。
    """
    try:
        from xkep_cae.core.slots import StrategySlot
    except Exception:  # pragma: no cover - safety net
        StrategySlot = None  # type: ignore[assignment]

    visited: dict[str, type] = {}
    stack: list[type] = [root_cls]
    while stack:
        cls = stack.pop()
        key = cls.__name__
        if key in visited:
            continue
        visited[key] = cls
        for dep in getattr(cls, "uses", []):
            if dep.__name__ not in visited:
                stack.append(dep)
        # status-320: StrategySlot の default_types をクラスレベルで展開する。
        # `collect_strategy_slots(cls)` は MRO を遡って全 StrategySlot を集める
        # ので、サブクラスで上書きされた宣言も正しく拾える。
        if StrategySlot is not None:
            for klass in reversed(cls.__mro__):
                for attr in vars(klass).values():
                    if not isinstance(attr, StrategySlot):
                        continue
                    for dep_type in attr.default_types:
                        if dep_type.__name__ not in visited:
                            stack.append(dep_type)
    return visited


def _is_leaf_process(name: str, known_classes: dict[str, type]) -> bool:
    """profile_breakdown の name エントリを葉プロセス判定.

    - `known_classes` に載っていて `uses` が空かつ StrategySlot の
      `default_types` も空 → 葉
    - `known_classes` に載っていない → 保守的に葉扱い（未知の非 Process 経路を
      誤って wrapper 扱いしないため）
    - `known_classes` に載っていて `uses` が非空 or StrategySlot 経由依存あり
      → wrapper（非葉）

    status-320: StrategySlot.default_types が非空のクラスは、静的 `uses` が空でも
    Strategy 経由で他 Process を呼び出すため wrapper 扱いする。これにより
    `ContactFrictionProcess` を target にしても dominant_leaf_process が本当の葉
    （例: `ContactForceStStiffnessProcess` の uses 先 `ComputeStJacobianProcess`）
    まで降りられる。
    """
    cls = known_classes.get(name)
    if cls is None:
        return True
    if getattr(cls, "uses", []):
        return False
    try:
        from xkep_cae.core.slots import StrategySlot
    except Exception:  # pragma: no cover - safety net
        return True
    for klass in cls.__mro__:
        for attr in vars(klass).values():
            if isinstance(attr, StrategySlot) and attr.default_types:
                return False
    return True


def _first_leaf_breakdown_entry(
    breakdown: tuple[dict[str, Any], ...],
    known_classes: dict[str, type],
) -> tuple[str, float, float]:
    """profile_breakdown の先頭葉プロセスを抽出.

    Returns:
        `(name, pct, total)` のタプル。breakdown が空 or 葉未検出の場合は
        `("", 0.0, 0.0)`。breakdown は事前に `total` 降順でソート済み想定。
    """
    for entry in breakdown:
        ename = str(entry.get("name", ""))
        if not ename:
            continue
        if _is_leaf_process(ename, known_classes):
            return (
                ename,
                float(entry.get("pct", 0.0)),
                float(entry.get("total", 0.0)),
            )
    return ("", 0.0, 0.0)
