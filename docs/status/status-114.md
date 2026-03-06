# status-114: チューニングタスクスキーマ + 検証プロット5種

[← README](../../README.md) | [← status-index](status-index.md) | [← status-113](status-113.md)

日付: 2026-03-06

## 概要

1. **`xkep_cae/tuning/` モジュール新設**: チューニングタスクの宣言的スキーマ（TuningTask / TuningParam / AcceptanceCriterion / TuningRun / TuningResult）
2. **S3プリセットタスク定義**: 収束チューニング・スケーリング分析・タイミング内訳の3種
3. **実行エンジン**: TuningTask に基づくソルバー実行とメトリクス収集
4. **検証プロット5種**: `generate_verification_plots.py` にS3チューニング可視化を追加
5. **プログラムテスト24件**: スキーマAPI・判定ロジック・JSON直列化・プリセット

## 背景・位置づけ

status-113 の TODO「視覚的妥当性検証スクリプトの実装」の実現。

チューニングタスクスキーマは**開発の根幹インフラ**として位置づけ:
- **現在**: パラメータチューニングの再現性と自動判定
- **発展**: CAE後処理のAIアシスト容易化
- **将来**: 実務最適化タスクの標準化・自動チューニングループ

## 実装詳細

### 1. TuningTask スキーマ (`xkep_cae/tuning/schema.py`)

| クラス | 役割 |
|--------|------|
| `TuningParam` | パラメータ定義（名前・範囲・デフォルト・対数スケール） |
| `AcceptanceCriterion` | 合格判定基準（メトリクス名・比較演算子・目標値） |
| `TuningRun` | 1回の実行結果（パラメータ・メトリクス・時系列・メタデータ） |
| `TuningTask` | タスク定義（パラメータリスト・判定基準・固定パラメータ・タグ） |
| `TuningResult` | 結果集約（複数Run・最良検索・合格フィルタ・JSON直列化） |

設計原則:
- **不変（Immutable）**: TuningParam/AcceptanceCriterion は `frozen=True`
- **直列化可能**: `save_json()` / `load_json()` によるラウンドトリップ
- **宣言的判定**: 比較演算子ベース（eq/ne/lt/le/gt/ge）

### 2. S3プリセット (`xkep_cae/tuning/presets.py`)

| プリセット | 用途 |
|-----------|------|
| `s3_convergence_task(n)` | n本撚りNCP収束パラメータ探索 |
| `s3_scaling_task()` | 素線数スケーリング分析 |
| `s3_timing_breakdown_task(n)` | 工程別処理時間分析 |

### 3. 実行エンジン (`xkep_cae/tuning/executor.py`)

- `execute_s3_benchmark()`: 単一ベンチマーク実行 → TuningRun
- `run_scaling_analysis()`: 複数素線数でのスケーリング → TuningResult
- `run_convergence_tuning()`: パラメータグリッドサーチ → TuningResult

test_s3_benchmark_timing.py の `_run_benchmark()` と同等の機能を
TuningRun スキーマに沿って構造化。

### 4. 検証プロット5種 (`tests/generate_verification_plots.py`)

| プロット | ファイル名 | 内容 |
|---------|-----------|------|
| スケーリング分析 | `tuning_scaling_analysis.png` | DOF・計算時間・Newton反復数 vs 素線数 |
| 接触トポロジー | `tuning_contact_topology.png` | 活性ペア・接触力・荷重係数の時間推移 |
| タイミング内訳 | `tuning_timing_breakdown.png` | 工程別処理時間のスタックバー |
| 断面接触マップ | `tuning_wire_cross_section.png` | ワイヤ断面2D投影 + 接触ペア線 |
| 合格判定サマリー | `tuning_acceptance_summary.png` | AcceptanceCriterion ヒートマップ |

加えて `tuning_result.json` を出力（TuningResult の完全な直列化データ）。

### 5. テスト (`tests/test_tuning_schema.py`)

| クラス | テスト数 | 内容 |
|--------|---------|------|
| `TestTuningParamAPI` | 4 | 生成・範囲判定・frozen |
| `TestAcceptanceCriterionAPI` | 5 | 各演算子・無効演算子・frozen |
| `TestTuningTaskAPI` | 2 | param_names・default_params |
| `TestTuningRunAPI` | 3 | 基準評価（Pass/Fail/Missing） |
| `TestTuningResultAPI` | 6 | 集約・最良検索・合格フィルタ・JSON往復 |
| `TestTuningPresetsAPI` | 3 | 3プリセットのAPI |
| `TestTuningExecutorAPI` | 1 | import確認 |

## 影響ファイル

### 新規
- `xkep_cae/tuning/__init__.py`
- `xkep_cae/tuning/schema.py`
- `xkep_cae/tuning/presets.py`
- `xkep_cae/tuning/executor.py`
- `tests/test_tuning_schema.py`

### 変更
- `tests/generate_verification_plots.py` — S3チューニング検証プロット5種追加

### 生成物（docs/verification/）
- `tuning_scaling_analysis.png`
- `tuning_contact_topology.png`
- `tuning_timing_breakdown.png`
- `tuning_wire_cross_section.png`
- `tuning_acceptance_summary.png`
- `tuning_result.json`

## テスト結果

- 新規テスト: **24 passed** (test_tuning_schema.py)
- 既存テストへの影響: なし
- lint: ruff check + format 通過

## 確認事項

- [ ] 検証プロットの日本語フォント — CI環境ではCJKフォント未インストールのため文字化けの可能性あり（英語ラベルへの差し替え or フォント同梱を検討）
- [ ] executor のブロックソルバー対応 — `newton_raphson_block_contact` パスは実装済みだが timing 収集はブロックソルバー側で未対応

## TODO

- パラメータ感度分析プロット（omega_max × al_relaxation のヒートマップ）
- 応力コンター・曲率コンターの自動判定テスト（隣接要素間変化率チェック）
- TuningTask の YAML 定義ファイル対応（宣言的タスク管理）
- 自動チューニングループ（Optuna 連携）
