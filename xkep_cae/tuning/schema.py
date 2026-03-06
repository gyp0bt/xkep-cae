"""チューニングタスクスキーマ定義.

CAE パラメータチューニングの宣言的定義・実行結果・判定基準を
構造化データとして表現する。

設計原則:
  - 不変（Immutable）: Task/Param/Criterion は定義後に変更しない
  - 直列化可能: JSON/YAML への変換を想定した単純なデータ構造
  - 検証可能: AcceptanceCriterion による自動判定
  - 拡張可能: カスタムメトリクスを辞書で自由に追加
"""

from __future__ import annotations

import json
import operator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TuningParam:
    """チューニング対象パラメータの定義.

    Attributes:
        name: パラメータ名（ソルバー引数名と一致させる）
        low: 探索範囲下限
        high: 探索範囲上限
        default: デフォルト値（現在の推奨値）
        log_scale: 対数スケールで探索するか
        description: パラメータの説明
    """

    name: str
    low: float
    high: float
    default: float | None = None
    log_scale: bool = False
    description: str = ""

    def contains(self, value: float) -> bool:
        """値が探索範囲内かを判定."""
        return self.low <= value <= self.high


_OPS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}


@dataclass(frozen=True)
class AcceptanceCriterion:
    """合格判定基準.

    メトリクス名・比較演算子・目標値の三つ組で判定ルールを表現する。

    Attributes:
        metric: メトリクス名（TuningRun.metrics のキーと一致）
        op: 比較演算子 ("eq", "ne", "lt", "le", "gt", "ge")
        target: 目標値
        description: 基準の説明
    """

    metric: str
    op: str
    target: Any
    description: str = ""

    def evaluate(self, value: Any) -> bool:
        """メトリクス値が基準を満たすか判定."""
        if self.op not in _OPS:
            raise ValueError(f"未知の演算子: {self.op}")
        return _OPS[self.op](value, self.target)


@dataclass
class TuningRun:
    """1回のチューニング実行結果.

    Attributes:
        params: 使用したパラメータ値の辞書
        metrics: 計測されたメトリクス値の辞書
        time_series: 時系列データ（荷重係数・接触力等）
        metadata: 補足情報（素線数、メッシュ情報等）
    """

    params: dict[str, Any]
    metrics: dict[str, Any]
    time_series: dict[str, list[float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def evaluate_criteria(self, criteria: list[AcceptanceCriterion]) -> dict[str, bool]:
        """全基準を評価し、結果を返す."""
        results = {}
        for c in criteria:
            if c.metric in self.metrics:
                results[c.metric] = c.evaluate(self.metrics[c.metric])
            else:
                results[c.metric] = False
        return results

    @property
    def passed(self) -> bool:
        """全メトリクスが存在するか（基準評価は別途）."""
        return len(self.metrics) > 0


@dataclass
class TuningTask:
    """チューニングタスクの宣言的定義.

    「何を変えて・何を測って・何で判定するか」を構造化する。

    Attributes:
        name: タスク識別名
        description: タスクの説明
        params: チューニング対象パラメータのリスト
        criteria: 合格判定基準のリスト
        fixed_params: 固定パラメータ（変動させない）
        tags: 分類タグ（"s3", "convergence", "scaling" 等）
    """

    name: str
    description: str
    params: list[TuningParam] = field(default_factory=list)
    criteria: list[AcceptanceCriterion] = field(default_factory=list)
    fixed_params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @property
    def param_names(self) -> list[str]:
        """パラメータ名のリスト."""
        return [p.name for p in self.params]

    def default_params(self) -> dict[str, Any]:
        """デフォルト値の辞書を返す."""
        d = dict(self.fixed_params)
        for p in self.params:
            if p.default is not None:
                d[p.name] = p.default
        return d

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換（YAML/JSON直列化用）."""
        return {
            "name": self.name,
            "description": self.description,
            "params": [asdict(p) for p in self.params],
            "criteria": [asdict(c) for c in self.criteria],
            "fixed_params": self.fixed_params,
            "tags": self.tags,
        }

    def save_yaml(self, path: Path | str) -> None:
        """タスク定義をYAMLファイルに保存."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("YAML保存には PyYAML が必要です: pip install pyyaml") from e

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(),
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )

    @classmethod
    def load_yaml(cls, path: Path | str) -> TuningTask:
        """YAMLファイルからタスク定義を復元."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("YAML読込には PyYAML が必要です: pip install pyyaml") from e

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            name=data["name"],
            description=data["description"],
            params=[TuningParam(**p) for p in data.get("params", [])],
            criteria=[AcceptanceCriterion(**c) for c in data.get("criteria", [])],
            fixed_params=data.get("fixed_params", {}),
            tags=data.get("tags", []),
        )


@dataclass
class TuningResult:
    """チューニングタスク全体の結果.

    複数の TuningRun を集約し、最良結果やパラメータ感度分析の
    入力データを提供する。

    Attributes:
        task: 元のタスク定義
        runs: 実行結果のリスト
    """

    task: TuningTask
    runs: list[TuningRun] = field(default_factory=list)

    def add_run(self, run: TuningRun) -> None:
        """実行結果を追加."""
        self.runs.append(run)

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def best_run(self, metric: str, minimize: bool = True) -> TuningRun | None:
        """指定メトリクスの最良実行を返す."""
        valid = [r for r in self.runs if metric in r.metrics]
        if not valid:
            return None
        return min(valid, key=lambda r: r.metrics[metric])  # type: ignore[return-value]

    def passed_runs(self) -> list[TuningRun]:
        """全基準を満たした実行のリスト."""
        passed = []
        for run in self.runs:
            verdicts = run.evaluate_criteria(self.task.criteria)
            if verdicts and all(verdicts.values()):
                passed.append(run)
        return passed

    def param_values(self, param_name: str) -> list[Any]:
        """指定パラメータの全実行における値リスト."""
        return [r.params.get(param_name) for r in self.runs if param_name in r.params]

    def metric_values(self, metric_name: str) -> list[Any]:
        """指定メトリクスの全実行における値リスト."""
        return [r.metrics.get(metric_name) for r in self.runs if metric_name in r.metrics]

    def to_dict(self) -> dict[str, Any]:
        """JSON直列化可能な辞書に変換."""
        return {
            "task": {
                "name": self.task.name,
                "description": self.task.description,
                "params": [asdict(p) for p in self.task.params],
                "criteria": [asdict(c) for c in self.task.criteria],
                "fixed_params": self.task.fixed_params,
                "tags": self.task.tags,
            },
            "runs": [
                {
                    "params": r.params,
                    "metrics": r.metrics,
                    "time_series": r.time_series,
                    "metadata": r.metadata,
                }
                for r in self.runs
            ],
        }

    def save_json(self, path: Path | str) -> None:
        """結果をJSONファイルに保存."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load_json(cls, path: Path | str) -> TuningResult:
        """JSONファイルから結果を復元."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> TuningResult:
        """辞書からTuningResultを復元（JSON/YAML共通）."""
        task_data = data["task"]
        task = TuningTask(
            name=task_data["name"],
            description=task_data["description"],
            params=[TuningParam(**p) for p in task_data.get("params", [])],
            criteria=[AcceptanceCriterion(**c) for c in task_data.get("criteria", [])],
            fixed_params=task_data.get("fixed_params", {}),
            tags=task_data.get("tags", []),
        )

        runs = []
        for r in data.get("runs", []):
            runs.append(
                TuningRun(
                    params=r["params"],
                    metrics=r["metrics"],
                    time_series=r.get("time_series", {}),
                    metadata=r.get("metadata", {}),
                )
            )

        return cls(task=task, runs=runs)

    def save_yaml(self, path: Path | str) -> None:
        """結果をYAMLファイルに保存.

        PyYAML が利用可能な場合のみ動作する。
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError("YAML保存には PyYAML が必要です: pip install pyyaml") from e

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(),
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )

    @classmethod
    def load_yaml(cls, path: Path | str) -> TuningResult:
        """YAMLファイルから結果を復元."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("YAML読込には PyYAML が必要です: pip install pyyaml") from e

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
