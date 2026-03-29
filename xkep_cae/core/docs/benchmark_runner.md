# BenchmarkRunner — 実行マニフェスト自動記録

[← README](../../../README.md)

## 概要

STA2防止（担当者間再現性ルール）のため、プロセス実行時の全パラメータ・
環境情報・結果サマリーを自動記録する仕組み。

## 目的

1. **パラメータ自動記録**: frozen dataclass の全フィールドをYAMLシリアライズ
2. **環境記録**: git commit/branch/dirty、Python/NumPy バージョン
3. **結果紐付け**: ソルバー結果サマリー + statusファイルリンク
4. **再現手順生成**: 同一結果を得るためのコマンド列を自動出力

## アーキテクチャ

```
RunManifest (frozen dataclass)
├── environment: EnvironmentInfo  ← git/Python自動取得
├── config_params: dict           ← dataclass → dict 再帰シリアライズ
├── results_summary: dict         ← スカラー結果抽出
├── process_name: str
├── elapsed_seconds: float
└── timestamp: str

BenchmarkRunnerProcess (BatchProcess)
├── input: BenchmarkRunInput[TIn]
│   ├── process: AbstractProcess[TIn, TOut]
│   ├── config: TIn (frozen dataclass)
│   ├── status_file: str | None
│   └── result_extractors: dict[str, Callable]
├── output: BenchmarkRunResult[TOut]
│   ├── result: TOut  ← 元プロセスの出力
│   └── manifest: RunManifest
└── YAML出力: docs/benchmarks/{process}_{timestamp}.yaml
```

## Config シリアライズ規則

- `frozen dataclass` → 再帰的に `dict` 化
- `np.ndarray` → shape + dtype + hash（データ本体は除外）
- `Callable` → `module.qualname`
- `object`（不明型） → `repr()` 要約
- `None` → null

## 使用例

```python
from xkep_cae.core.benchmark import BenchmarkRunnerProcess, BenchmarkRunInput

cfg = DynamicThreePointBendContactJigConfig(E=25.0, n_periods=30.0)
proc = DynamicThreePointBendContactJigProcess()

result = BenchmarkRunnerProcess().process(BenchmarkRunInput(
    process=proc,
    config=cfg,
    result_extractors={
        "frac": lambda r: r.solver_result.load_history[-1],
        "n_increments": lambda r: r.solver_result.n_increments,
        "n_cutbacks": lambda r: r.solver_result.n_cutbacks,
    },
    status_file="docs/status/status-265.md",
))

# result.result → 元プロセスの結果
# result.manifest → RunManifest (自動保存済み)
```
