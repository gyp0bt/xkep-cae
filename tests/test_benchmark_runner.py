"""BenchmarkRunner のテスト.

RunManifest, serialize_config, BenchmarkRunnerProcess の
動作確認テスト。

ソルバーパス: なし（インフラのみ）

[← README](../README.md)
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from xkep_cae.core.base import ProcessMeta
from xkep_cae.core.benchmark import (
    BenchmarkRunInput,
    BenchmarkRunnerProcess,
    RunManifest,
    capture_environment,
    serialize_config,
)
from xkep_cae.core.categories import PreProcess
from xkep_cae.core.testing import binds_to

# --- テスト用ダミープロセス ---


@dataclass(frozen=True)
class _DummyConfig:
    x: float = 1.0
    n: int = 10
    name: str = "test"
    arr: np.ndarray | None = None


@dataclass(frozen=True)
class _DummyResult:
    value: float
    data: np.ndarray


class _DummyProcess(PreProcess["_DummyConfig", "_DummyResult"]):
    """テスト用ダミープロセス."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="_DummyProcess",
        module="pre",
        version="0.1.0",
        document_path="../xkep_cae/core/docs/benchmark_runner.md",
        stability="experimental",
        support_tier="dev-only",
    )
    _skip_registry = True

    def process(self, input_data: _DummyConfig) -> _DummyResult:
        return _DummyResult(
            value=input_data.x * input_data.n,
            data=np.ones(5),
        )


# --- serialize_config テスト ---


class TestSerializeConfigAPI:
    """serialize_config の基本 API テスト."""

    def test_primitive_passthrough(self):
        """プリミティブ型はそのまま返す."""
        assert serialize_config(1) == 1
        assert serialize_config(1.5) == 1.5
        assert serialize_config("hello") == "hello"
        assert serialize_config(True) is True
        assert serialize_config(None) is None

    def test_numpy_scalar(self):
        """numpy スカラーは Python 型に変換."""
        assert serialize_config(np.float64(3.14)) == 3.14
        assert isinstance(serialize_config(np.float64(3.14)), float)
        assert serialize_config(np.int64(42)) == 42
        assert isinstance(serialize_config(np.int64(42)), int)

    def test_ndarray_summary(self):
        """np.ndarray は shape + dtype + md5 のサマリーに変換."""
        arr = np.array([1.0, 2.0, 3.0])
        result = serialize_config(arr)
        assert result["__ndarray__"] is True
        assert result["shape"] == [3]
        assert result["dtype"] == "float64"
        assert len(result["md5"]) == 12

    def test_frozen_dataclass(self):
        """frozen dataclass を再帰的に dict 化."""
        cfg = _DummyConfig(x=2.5, n=20, name="bench")
        result = serialize_config(cfg)
        assert isinstance(result, dict)
        assert result["x"] == 2.5
        assert result["n"] == 20
        assert result["name"] == "bench"
        assert result["arr"] is None

    def test_nested_dataclass_with_array(self):
        """ndarray フィールドを含む dataclass."""
        cfg = _DummyConfig(x=1.0, n=5, arr=np.zeros(3))
        result = serialize_config(cfg)
        assert result["arr"]["__ndarray__"] is True
        assert result["arr"]["shape"] == [3]

    def test_callable_serialization(self):
        """Callable は qualified name 文字列に変換."""

        def my_func():
            pass

        result = serialize_config(my_func)
        assert "my_func" in result

    def test_list_serialization(self):
        """リストは再帰的に変換."""
        result = serialize_config([1, 2.0, "a"])
        assert result == [1, 2.0, "a"]

    def test_long_sequence_summary(self):
        """長い sequence は length + type のサマリー."""
        long_list = list(range(200))
        result = serialize_config(long_list)
        assert result["__sequence__"] is True
        assert result["length"] == 200

    def test_path_serialization(self):
        """Path は文字列に変換."""
        result = serialize_config(Path("/tmp/test.txt"))
        assert result == "/tmp/test.txt"

    def test_type_serialization(self):
        """クラス型は module.qualname に変換."""
        result = serialize_config(int)
        assert "int" in result


# --- capture_environment テスト ---


class TestCaptureEnvironmentAPI:
    """capture_environment の基本 API テスト."""

    def test_returns_environment_info(self):
        """EnvironmentInfo が全フィールド埋まって返る."""
        env = capture_environment()
        assert env.python_version
        assert env.numpy_version
        assert env.timestamp
        # git フィールドは CI 環境では空の場合があるため存在チェックのみ
        assert isinstance(env.git_commit, str)
        assert isinstance(env.git_branch, str)
        assert isinstance(env.git_dirty, bool)


# --- RunManifest テスト ---


class TestRunManifestAPI:
    """RunManifest の基本 API テスト."""

    def test_to_dict(self):
        """to_dict でシリアライズ可能な dict を生成."""
        env = capture_environment()
        manifest = RunManifest(
            process_name="TestProcess",
            process_version="0.1.0",
            environment=env,
            config_params={"x": 1.0, "n": 10},
            results_summary={"value": 42.0},
            elapsed_seconds=1.234,
            status_file="docs/status/status-265.md",
        )
        d = manifest.to_dict()
        assert d["process"]["name"] == "TestProcess"
        assert d["config"]["x"] == 1.0
        assert d["results"]["value"] == 42.0
        assert d["elapsed_seconds"] == 1.234
        assert d["status_file"] == "docs/status/status-265.md"

    def test_to_yaml(self):
        """to_yaml で YAML 文字列を生成."""
        env = capture_environment()
        manifest = RunManifest(
            process_name="TestProcess",
            process_version="0.1.0",
            environment=env,
            config_params={"x": 1.0},
            results_summary={"value": 42.0},
            elapsed_seconds=1.0,
        )
        yaml_str = manifest.to_yaml()
        assert "process:" in yaml_str
        assert "TestProcess" in yaml_str
        assert "config:" in yaml_str
        assert "results:" in yaml_str


# --- BenchmarkRunnerProcess テスト ---


@binds_to(BenchmarkRunnerProcess)
class TestBenchmarkRunnerProcessAPI:
    """BenchmarkRunnerProcess の API テスト."""

    def test_basic_run(self):
        """ダミープロセスを実行しマニフェスト付き結果を返す."""
        cfg = _DummyConfig(x=3.0, n=5)
        proc = _DummyProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_input = BenchmarkRunInput(
                process=proc,
                config=cfg,
                result_extractors={
                    "value": lambda r: r.value,
                },
                output_dir=tmpdir,
            )
            run_result = BenchmarkRunnerProcess().process(run_input)

        # 元プロセスの結果
        assert run_result.result.value == 15.0
        assert np.allclose(run_result.result.data, 1.0)

        # マニフェスト
        m = run_result.manifest
        assert m.process_name == "_DummyProcess"
        assert m.config_params["x"] == 3.0
        assert m.config_params["n"] == 5
        assert m.results_summary["value"] == 15.0
        assert m.elapsed_seconds > 0

    def test_manifest_yaml_saved(self):
        """YAML ファイルが指定ディレクトリに保存される."""
        cfg = _DummyConfig(x=1.0, n=1)
        proc = _DummyProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_input = BenchmarkRunInput(
                process=proc,
                config=cfg,
                output_dir=tmpdir,
            )
            run_result = BenchmarkRunnerProcess().process(run_input)

            assert run_result.manifest_path is not None
            path = Path(run_result.manifest_path)
            assert path.exists()
            content = path.read_text()
            assert "_DummyProcess" in content

    def test_status_file_recorded(self):
        """status_file がマニフェストに記録される."""
        cfg = _DummyConfig()
        proc = _DummyProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_input = BenchmarkRunInput(
                process=proc,
                config=cfg,
                status_file="docs/status/status-265.md",
                output_dir=tmpdir,
            )
            run_result = BenchmarkRunnerProcess().process(run_input)
            assert run_result.manifest.status_file == "docs/status/status-265.md"

    def test_extractor_error_handling(self):
        """結果抽出エラーは ERROR メッセージとして記録される."""
        cfg = _DummyConfig()
        proc = _DummyProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_input = BenchmarkRunInput(
                process=proc,
                config=cfg,
                result_extractors={
                    "bad_key": lambda r: r.nonexistent_field,
                },
                output_dir=tmpdir,
            )
            run_result = BenchmarkRunnerProcess().process(run_input)
            assert "ERROR" in str(run_result.manifest.results_summary["bad_key"])

    def test_environment_captured(self):
        """環境情報がマニフェストに含まれる."""
        cfg = _DummyConfig()
        proc = _DummyProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_input = BenchmarkRunInput(
                process=proc,
                config=cfg,
                output_dir=tmpdir,
            )
            run_result = BenchmarkRunnerProcess().process(run_input)
            env = run_result.manifest.environment
            assert env.python_version
            assert env.timestamp

    def test_profile_breakdown_captured(self):
        """capture_profile=True で profile_breakdown が記録される."""
        cfg = _DummyConfig(x=1.0, n=3)
        proc = _DummyProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_input = BenchmarkRunInput(
                process=proc,
                config=cfg,
                output_dir=tmpdir,
                capture_profile=True,
            )
            run_result = BenchmarkRunnerProcess().process(run_input)

        breakdown = run_result.manifest.profile_breakdown
        assert len(breakdown) >= 1
        row = next(r for r in breakdown if r["name"] == "_DummyProcess")
        # このベンチマーク走査で記録された _DummyProcess の呼び出しは 1 回
        assert row["n"] == 1
        assert row["total"] >= 0.0
        assert row["pct"] >= 0.0
        # 一括記録フィールドが全て揃っていること
        for key in ("name", "n", "total", "avg", "min", "max", "median", "pct"):
            assert key in row

    def test_profile_breakdown_disabled(self):
        """capture_profile=False で profile_breakdown が空."""
        cfg = _DummyConfig()
        proc = _DummyProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_input = BenchmarkRunInput(
                process=proc,
                config=cfg,
                output_dir=tmpdir,
                capture_profile=False,
            )
            run_result = BenchmarkRunnerProcess().process(run_input)

        assert run_result.manifest.profile_breakdown == ()

    def test_profile_top_n(self):
        """profile_top_n で記録件数を制限."""
        cfg = _DummyConfig()
        proc = _DummyProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_input = BenchmarkRunInput(
                process=proc,
                config=cfg,
                output_dir=tmpdir,
                capture_profile=True,
                profile_top_n=1,
            )
            run_result = BenchmarkRunnerProcess().process(run_input)

        assert len(run_result.manifest.profile_breakdown) == 1

    def test_profile_breakdown_in_yaml(self):
        """profile_breakdown がマニフェスト YAML に出力される."""
        cfg = _DummyConfig()
        proc = _DummyProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_input = BenchmarkRunInput(
                process=proc,
                config=cfg,
                output_dir=tmpdir,
                capture_profile=True,
            )
            run_result = BenchmarkRunnerProcess().process(run_input)
            path = Path(run_result.manifest_path)
            content = path.read_text()

        assert "profile_breakdown:" in content
        assert "_DummyProcess" in content

    def test_manifest_filename_collision_avoided(self):
        """同一秒内に複数回走っても manifest ファイル名が衝突せず連番で退避される.

        status-315: パラメータスイープで 1 秒以内に N ケース走ると、
        タイムスタンプ [:15] が同じになり上書きされていた。
        `_save_manifest` の連番フォールバックでユニーク化される。
        """
        cfg = _DummyConfig()
        proc = _DummyProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            paths: list[str] = []
            for _ in range(3):
                run_input = BenchmarkRunInput(
                    process=proc,
                    config=cfg,
                    output_dir=tmpdir,
                )
                run_result = BenchmarkRunnerProcess().process(run_input)
                assert run_result.manifest_path is not None
                paths.append(run_result.manifest_path)

            # 全 path が存在し、かつ一意
            assert all(Path(p).exists() for p in paths)
            assert len(set(paths)) == 3
