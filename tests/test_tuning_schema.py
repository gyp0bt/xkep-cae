"""TuningTask スキーマのプログラムテスト.

TuningTask / TuningParam / AcceptanceCriterion / TuningRun / TuningResult の
API正しさ・直列化・判定ロジックを検証する。
"""

import json
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skip(reason="xkep_cae_deprecated 参照のため無効化 (status-193)")

from xkep_cae_deprecated.tuning import (  # noqa: E402
    AcceptanceCriterion,
    TuningParam,
    TuningResult,
    TuningRun,
    TuningTask,
)


class TestTuningParamAPI:
    """TuningParam のAPI正しさテスト."""

    def test_create_param(self):
        p = TuningParam("omega_max", 0.1, 1.0, default=0.3)
        assert p.name == "omega_max"
        assert p.low == 0.1
        assert p.high == 1.0
        assert p.default == 0.3

    def test_contains_in_range(self):
        p = TuningParam("x", 0.0, 1.0)
        assert p.contains(0.5)
        assert p.contains(0.0)
        assert p.contains(1.0)

    def test_contains_out_of_range(self):
        p = TuningParam("x", 0.0, 1.0)
        assert not p.contains(-0.1)
        assert not p.contains(1.1)

    def test_frozen(self):
        p = TuningParam("x", 0.0, 1.0)
        with pytest.raises(AttributeError):
            p.name = "y"  # type: ignore[misc]


class TestAcceptanceCriterionAPI:
    """AcceptanceCriterion の判定ロジックテスト."""

    def test_eq(self):
        c = AcceptanceCriterion("converged", "eq", True)
        assert c.evaluate(True)
        assert not c.evaluate(False)

    def test_lt(self):
        c = AcceptanceCriterion("penetration", "lt", 0.05)
        assert c.evaluate(0.01)
        assert not c.evaluate(0.05)
        assert not c.evaluate(0.10)

    def test_ge(self):
        c = AcceptanceCriterion("n_active", "ge", 1)
        assert c.evaluate(1)
        assert c.evaluate(5)
        assert not c.evaluate(0)

    def test_invalid_op(self):
        c = AcceptanceCriterion("x", "invalid", 0)
        with pytest.raises(ValueError, match="未知の演算子"):
            c.evaluate(0)

    def test_frozen(self):
        c = AcceptanceCriterion("x", "eq", 0)
        with pytest.raises(AttributeError):
            c.op = "lt"  # type: ignore[misc]


class TestTuningTaskAPI:
    """TuningTask のAPI正しさテスト."""

    def _make_task(self):
        return TuningTask(
            name="test_task",
            description="テスト用タスク",
            params=[
                TuningParam("p1", 0.0, 1.0, default=0.5),
                TuningParam("p2", 1.0, 10.0, default=5.0),
            ],
            criteria=[
                AcceptanceCriterion("converged", "eq", True),
                AcceptanceCriterion("error", "lt", 0.01),
            ],
            fixed_params={"fixed_key": "fixed_val"},
            tags=["test"],
        )

    def test_param_names(self):
        task = self._make_task()
        assert task.param_names == ["p1", "p2"]

    def test_default_params(self):
        task = self._make_task()
        defaults = task.default_params()
        assert defaults["p1"] == 0.5
        assert defaults["p2"] == 5.0
        assert defaults["fixed_key"] == "fixed_val"


class TestTuningRunAPI:
    """TuningRun のメトリクス評価テスト."""

    def test_evaluate_criteria_pass(self):
        run = TuningRun(
            params={"p1": 0.5},
            metrics={"converged": True, "error": 0.001},
        )
        criteria = [
            AcceptanceCriterion("converged", "eq", True),
            AcceptanceCriterion("error", "lt", 0.01),
        ]
        results = run.evaluate_criteria(criteria)
        assert results["converged"]
        assert results["error"]

    def test_evaluate_criteria_fail(self):
        run = TuningRun(
            params={"p1": 0.5},
            metrics={"converged": False, "error": 0.05},
        )
        criteria = [
            AcceptanceCriterion("converged", "eq", True),
            AcceptanceCriterion("error", "lt", 0.01),
        ]
        results = run.evaluate_criteria(criteria)
        assert not results["converged"]
        assert not results["error"]

    def test_missing_metric(self):
        run = TuningRun(params={}, metrics={})
        criteria = [AcceptanceCriterion("missing", "eq", True)]
        results = run.evaluate_criteria(criteria)
        assert not results["missing"]


class TestTuningResultAPI:
    """TuningResult の集約・直列化テスト."""

    def _make_result(self):
        task = TuningTask(
            name="test",
            description="test",
            params=[TuningParam("p1", 0.0, 1.0, default=0.5)],
            criteria=[
                AcceptanceCriterion("converged", "eq", True),
                AcceptanceCriterion("error", "lt", 0.01),
            ],
        )
        result = TuningResult(task=task)
        result.add_run(
            TuningRun(
                params={"p1": 0.3},
                metrics={"converged": True, "error": 0.005, "time": 1.0},
            )
        )
        result.add_run(
            TuningRun(
                params={"p1": 0.7},
                metrics={"converged": True, "error": 0.002, "time": 2.0},
            )
        )
        result.add_run(
            TuningRun(
                params={"p1": 0.9},
                metrics={"converged": False, "error": 0.05, "time": 0.5},
            )
        )
        return result

    def test_n_runs(self):
        r = self._make_result()
        assert r.n_runs == 3

    def test_best_run(self):
        r = self._make_result()
        best = r.best_run("error", minimize=True)
        assert best is not None
        assert best.metrics["error"] == 0.002

    def test_passed_runs(self):
        r = self._make_result()
        passed = r.passed_runs()
        assert len(passed) == 2
        for run in passed:
            assert run.metrics["converged"]
            assert run.metrics["error"] < 0.01

    def test_param_values(self):
        r = self._make_result()
        vals = r.param_values("p1")
        assert vals == [0.3, 0.7, 0.9]

    def test_metric_values(self):
        r = self._make_result()
        vals = r.metric_values("time")
        assert vals == [1.0, 2.0, 0.5]

    def test_json_roundtrip(self):
        r = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            r.save_json(path)

            # ファイルが存在し、有効なJSONであることを確認
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["task"]["name"] == "test"
            assert len(data["runs"]) == 3

            # 復元
            loaded = TuningResult.load_json(path)
            assert loaded.task.name == "test"
            assert loaded.n_runs == 3
            assert loaded.runs[0].params["p1"] == 0.3
            assert loaded.runs[1].metrics["error"] == 0.002


class TestTuningYAMLAPI:
    """TuningTask / TuningResult の YAML 直列化テスト."""

    def test_task_yaml_roundtrip(self):
        """TuningTask の YAML 保存・復元が一致する."""
        task = TuningTask(
            name="yaml_test",
            description="YAML往復テスト",
            params=[
                TuningParam("p1", 0.0, 1.0, default=0.5),
                TuningParam("p2", 1.0, 10.0, default=5.0, log_scale=True),
            ],
            criteria=[
                AcceptanceCriterion("converged", "eq", True, "収束確認"),
            ],
            fixed_params={"fixed_key": "fixed_val"},
            tags=["test", "yaml"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "task.yaml"
            task.save_yaml(path)
            assert path.exists()

            loaded = TuningTask.load_yaml(path)
            assert loaded.name == "yaml_test"
            assert loaded.param_names == ["p1", "p2"]
            assert loaded.params[1].log_scale is True
            assert loaded.criteria[0].op == "eq"
            assert loaded.fixed_params["fixed_key"] == "fixed_val"
            assert loaded.tags == ["test", "yaml"]

    def test_result_yaml_roundtrip(self):
        """TuningResult の YAML 保存・復元が一致する."""
        task = TuningTask(
            name="yaml_result_test",
            description="test",
            params=[TuningParam("p1", 0.0, 1.0, default=0.5)],
            criteria=[AcceptanceCriterion("converged", "eq", True)],
        )
        result = TuningResult(task=task)
        result.add_run(
            TuningRun(
                params={"p1": 0.3},
                metrics={"converged": True, "error": 0.005},
                time_series={"load_factor": [0.1, 0.5, 1.0]},
            )
        )
        result.add_run(
            TuningRun(
                params={"p1": 0.7},
                metrics={"converged": False, "error": 0.1},
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.yaml"
            result.save_yaml(path)
            assert path.exists()

            loaded = TuningResult.load_yaml(path)
            assert loaded.task.name == "yaml_result_test"
            assert loaded.n_runs == 2
            assert loaded.runs[0].params["p1"] == 0.3
            assert loaded.runs[0].metrics["converged"] is True
            assert loaded.runs[1].metrics["converged"] is False

    def test_task_to_dict(self):
        """TuningTask.to_dict の基本検証."""
        task = TuningTask(
            name="dict_test",
            description="dict変換テスト",
            params=[TuningParam("x", 0.0, 1.0)],
            tags=["unit"],
        )
        d = task.to_dict()
        assert d["name"] == "dict_test"
        assert len(d["params"]) == 1
        assert d["params"][0]["name"] == "x"


class TestTuningPresetsAPI:
    """プリセットタスク定義のAPIテスト."""

    def test_s3_convergence_task(self):
        from xkep_cae.tuning.presets import s3_convergence_task

        task = s3_convergence_task(19)
        assert "19" in task.name
        assert len(task.params) > 0
        assert len(task.criteria) > 0
        assert task.fixed_params["n_strands"] == 19

    def test_s3_scaling_task(self):
        from xkep_cae.tuning.presets import s3_scaling_task

        task = s3_scaling_task()
        assert "scaling" in task.name
        assert len(task.criteria) > 0

    def test_s3_timing_breakdown_task(self):
        from xkep_cae.tuning.presets import s3_timing_breakdown_task

        task = s3_timing_breakdown_task(7)
        assert "7" in task.name
        assert task.fixed_params["n_strands"] == 7


class TestTuningExecutorAPI:
    """Executor の import/基本APIテスト（実行はslow）."""

    def test_import_executor(self):
        from xkep_cae.tuning.executor import (
            execute_s3_benchmark,
            run_convergence_tuning,
            run_scaling_analysis,
            run_sensitivity_analysis,
        )

        assert callable(execute_s3_benchmark)
        assert callable(run_scaling_analysis)
        assert callable(run_convergence_tuning)
        assert callable(run_sensitivity_analysis)

    def test_execute_s3_benchmark_error_handling(self, monkeypatch):
        """ソルバー例外時でも TuningRun が返ること."""
        import xkep_cae.tuning.executor as executor_mod

        def _mock_run_bending(*args, **kwargs):
            raise RuntimeError("mock solver failure")

        monkeypatch.setattr(
            "xkep_cae.numerical_tests.wire_bending_benchmark.run_bending_oscillation",
            _mock_run_bending,
        )
        run = executor_mod.execute_s3_benchmark(7)
        assert run.metrics["converged"] is False
        assert "error" in run.metrics
        assert run.metadata["n_strands"] == 7

    def test_run_convergence_tuning_no_grid(self, monkeypatch):
        """param_grid=None でデフォルト1回実行."""
        import xkep_cae.tuning.executor as executor_mod
        from xkep_cae.tuning.schema import TuningRun

        mock_run = TuningRun(
            params={"n_strands": 7},
            metrics={"converged": True, "total_newton_iterations": 10},
        )
        monkeypatch.setattr(
            executor_mod,
            "execute_s3_benchmark",
            lambda n, **kw: mock_run,
        )
        result = executor_mod.run_convergence_tuning(n_strands=7, param_grid=None)
        assert len(result.runs) == 1

    def test_run_convergence_tuning_with_grid(self, monkeypatch):
        """param_grid 指定時にグリッドサーチが実行されること."""
        import xkep_cae.tuning.executor as executor_mod
        from xkep_cae.tuning.schema import TuningRun

        calls = []

        def _mock_bench(n, **kw):
            calls.append(kw)
            return TuningRun(
                params={"n_strands": n, **kw},
                metrics={"converged": True},
            )

        monkeypatch.setattr(executor_mod, "execute_s3_benchmark", _mock_bench)
        result = executor_mod.run_convergence_tuning(
            n_strands=7,
            param_grid={"alpha": [0.1, 0.5], "beta": [1.0, 2.0]},
        )
        assert len(result.runs) == 4  # 2x2 grid
        assert len(calls) == 4


class TestOptunaTunerAPI:
    """Optuna 連携の import/基本APIテスト."""

    def test_import_optuna_tuner(self):
        from xkep_cae.tuning.optuna_tuner import (
            create_objective,
            run_optuna_study,
        )

        assert callable(create_objective)
        assert callable(run_optuna_study)

    def test_create_objective_returns_callable(self):
        """create_objective が呼び出し可能な関数を返す."""
        from xkep_cae.tuning.optuna_tuner import create_objective
        from xkep_cae.tuning.presets import s3_convergence_task

        task = s3_convergence_task(7)
        result = TuningResult(task=task)
        obj = create_objective(task, result, n_strands=7)
        assert callable(obj)
