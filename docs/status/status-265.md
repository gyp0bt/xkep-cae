# status-265: BenchmarkRunnerProcess — STA2自動記録基盤

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-29
- **ブランチ**: `claude/check-status-todos-7X1cP`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3+9+4+18（新規18） → **合計592 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. BenchmarkRunnerProcess 実装

STA2防止ルール（担当者間再現性ルール）を自動化する `BenchmarkRunnerProcess` を実装。
任意の `AbstractProcess` を実行し、以下を自動でYAMLマニフェストに記録する。

#### 新規ファイル

| ファイル | 内容 |
|----------|------|
| `xkep_cae/core/benchmark.py` | RunManifest, BenchmarkRunnerProcess, serialize_config, capture_environment |
| `xkep_cae/core/docs/benchmark_runner.md` | 設計文書 |
| `tests/test_benchmark_runner.py` | 単体テスト 18件 |
| (設計文書内に使用例を記載) | 三点曲げジグとの統合例は `benchmark_runner.md` 参照 |

#### 主要コンポーネント

1. **EnvironmentInfo**: git commit/branch/dirty、Python/NumPy バージョン、タイムスタンプを自動取得
2. **serialize_config**: frozen dataclass を再帰的に dict 化（ndarray → shape+dtype+md5、Callable → qualname）
3. **RunManifest**: 1回の実行の全記録（process名、環境、パラメータ、結果サマリー、elapsed、status_file リンク）
4. **BenchmarkRunnerProcess**: BatchProcess として Process 実行をラップし、マニフェストをYAML保存

#### 自動記録される情報

```yaml
process:
  name: DynamicThreePointBendContactJigProcess
  version: 0.1.0
environment:
  git_commit: abc1234...
  git_branch: claude/check-status-todos-7X1cP
  git_dirty: false
  python_version: 3.11.14
  numpy_version: 2.4.3
  timestamp: 2026-03-29T12:00:00+00:00
config:
  wire_length: 100.0
  wire_diameter: 17.0
  E: 200000.0
  ...（全パラメータ自動展開）
results:
  frac: 1.0
  n_increments: 150
  n_cutbacks: 20
elapsed_seconds: 45.2
status_file: docs/status/status-265.md
```

### 2. STA2防止ルールとの対応

| STA2ルール | BenchmarkRunner での対応 |
|------------|------------------------|
| ベンチマーク条件の記録 | `config_params` に全パラメータ自動シリアライズ |
| 変更前ベースラインの先行取得 | `environment.git_commit` + `git_dirty` で追跡可能 |
| 再現手順の status 記載 | `status_file` リンク + YAML マニフェスト自体が再現手順 |
| Process profiling の活用 | `ProcessMetaclass._profile_data` と併用可能 |

### 3. core/__init__.py エクスポート追加

`BenchmarkRunnerProcess`, `BenchmarkRunInput`, `BenchmarkRunResult`, `RunManifest`, `capture_environment`, `serialize_config` を `xkep_cae.core` からインポート可能に。

---

## テスト結果

- 新規テスト: 18件（`tests/test_benchmark_runner.py`）
  - `TestSerializeConfigAPI`: 10件（プリミティブ、ndarray、dataclass、Callable、Path 等）
  - `TestCaptureEnvironmentAPI`: 1件（環境情報取得）
  - `TestRunManifestAPI`: 2件（to_dict, to_yaml��
  - `TestBenchmarkRunnerProcessAPI`: 5件（基本実行、YAML保存、status記録、エラーハンドリング、環境情報）
- 既存テスト: 574 passed（回帰なし）
- 全体: **592 passed**, 20 skipped, 1 xfailed
- 契約違反: 0件
- lint: 全合格

---

## 再現手順

```bash
git checkout claude/check-status-todos-7X1cP
pip install -e .

# 単体テスト
python -m pytest tests/test_benchmark_runner.py -v --timeout=30

# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"

# 契約検証
python contracts/validate_process_contracts.py

# BenchmarkRunner 統合例（重い: 三点曲げジグ実行）
# python contracts/bench_with_manifest.py 2>&1 | tee /tmp/log-bench-manifest.log
```

---

## 次セッションへの引き継ぎ

### 残課題（優先度順）

1. **frozen_hermite_tangent=False でのNR安定化**（status-264 から継続）
2. **E=25 frac=1.0 到達**（status-264 から継続）
3. **Hermite 非局所 ∂g/∂u 対応**（status-262 から継続）
4. **NR 力収束改善**（status-262 から継続）

### BenchmarkRunner 拡張TODO

1. **既存ベンチマークスクリプトの BenchmarkRunner 移行**: `contracts/bench_*.py` を順次 BenchmarkRunner 経由に移行
2. **マニフェスト比較ツール**: 2つのマニフェストを比較して差分を表示するユーティリティ
3. **status ファイルテンプレート自動生成**: マニフェストから status ファイルの「再現手順」セクションを自動生成

---

## 懸念・設計メモ

1. **PyYAML非依存**: 軽量YAML出力を自前実装（`_dict_to_yaml`）。複雑なネスト構造では整形が不完全な可能性あり。必要に応じてPyYAML依存を追加検討
2. **ndarray データ本体の記録**: 現在はshape+dtype+md5のみ。大規模な中間結果の保存が必要な場合は別途NPZ出力を検討
3. **output_dir デフォルト**: `docs/benchmarks/` に保存。.gitignore に追加するかは運用次第
