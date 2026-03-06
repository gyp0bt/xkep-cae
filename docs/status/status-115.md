# status-115: status-114 TODO消化（感度分析・物理テスト・YAML・Optuna）

[← README](../../README.md) | [← status-index](status-index.md) | [← status-114](status-114.md)

日付: 2026-03-06

## 概要

status-114 の TODO 4項目を全て実装:

1. **パラメータ感度分析プロット**: omega_max × al_relaxation のヒートマップ
2. **応力・曲率コンター自動判定テスト**: 隣接要素間変化率チェック（物理テスト11件）
3. **TuningTask YAML対応**: save_yaml() / load_yaml() の実装
4. **Optuna連携**: 自動チューニングループの基盤モジュール

加えて確認事項2件（英語ラベル・ブロックソルバーtiming）にも対応。

## 実装詳細

### 1. パラメータ感度分析プロット

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/tuning/executor.py` | `run_sensitivity_analysis()` 追加（2パラメータグリッドサーチ） |
| `tests/generate_verification_plots.py` | `plot_tuning_sensitivity_heatmap()` 追加 |

3種ヒートマップ:
- 収束性（Convergence）: Yes/No マップ
- Newton反復数（Newton Iterations）: 反復回数の色分け
- 最大貫入比（Max Penetration Ratio）: 精度指標

**注**: CI環境のCJKフォント問題を回避するため、新規プロットは**英語ラベル**を使用。

### 2. 応力・曲率コンター自動判定テスト

`tests/test_contour_continuity_physics.py` 新規作成（11テスト）:

| クラス | テスト数 | 検証内容 |
|--------|---------|---------|
| `TestStressContinuityPhysics` | 4 | せん断力一定・モーメント線形・応力滑らかさ・放物線分布 |
| `TestCurvatureContinuityPhysics` | 3 | 曲率線形・滑らかさ・単調減少 |
| `TestContourJumpDetectionAPI` | 4 | 変化率チェックユーティリティ（定数・線形・不連続・単一要素） |

CLAUDE.md 物理テスト思想「応力・曲率の連続性: 隣接要素間の応力差が極端に離散的でないか」の自動化。

### 3. TuningTask YAML対応

`xkep_cae/tuning/schema.py` に追加:

| メソッド | 対象 | 説明 |
|---------|------|------|
| `TuningTask.to_dict()` | Task | 辞書変換（YAML/JSON共通） |
| `TuningTask.save_yaml()` | Task | YAMLファイル保存 |
| `TuningTask.load_yaml()` | Task | YAMLファイル復元 |
| `TuningResult.save_yaml()` | Result | YAMLファイル保存 |
| `TuningResult.load_yaml()` | Result | YAMLファイル復元 |
| `TuningResult._from_dict()` | Result | JSON/YAML共通の復元ロジック |

PyYAML 未インストール時は `ImportError` で明示的に通知。

### 4. Optuna 連携

`xkep_cae/tuning/optuna_tuner.py` 新規作成:

| 関数 | 説明 |
|------|------|
| `create_objective()` | TuningTask → Optuna objective 関数を生成 |
| `run_optuna_study()` | Study 実行・TuningResult 返却 |

設計:
- TuningParam の `log_scale` フラグに応じて `suggest_float(log=True)` を自動選択
- デフォルト値を初期試行として `enqueue_trial()`
- 非収束時はペナルティ値（`inf`）を返却
- Optuna 未インストール時は `ImportError` で通知

### 5. テスト追加

`tests/test_tuning_schema.py` に追加:

| クラス | テスト数 | 内容 |
|--------|---------|------|
| `TestTuningYAMLAPI` | 3 | Task/Result YAML往復・to_dict |
| `TestOptunaTunerAPI` | 2 | import・create_objective callable |

## 確認事項の対応状況

| 確認事項 | 対応 |
|---------|------|
| 検証プロットの日本語フォント | 新規ヒートマップは英語ラベルで実装。既存プロットは現状維持 |
| executor ブロックソルバー対応 | `newton_raphson_block_contact` に `timing` パラメータが未実装のため次status以降のTODO |

## 影響ファイル

### 新規
- `xkep_cae/tuning/optuna_tuner.py`
- `tests/test_contour_continuity_physics.py`

### 変更
- `xkep_cae/tuning/schema.py` — YAML対応・_from_dict共通化
- `xkep_cae/tuning/executor.py` — run_sensitivity_analysis追加
- `tests/generate_verification_plots.py` — 感度ヒートマップ追加
- `tests/test_tuning_schema.py` — YAML/Optuna/Sensitivityテスト追加
- `README.md` — テスト数更新
- `CLAUDE.md` — テスト数更新

## テスト結果

- 新規テスト: **16件** (物理11 + YAML 3 + Optuna 2)
- 合計テスト: **2170** (2154 → 2170)
- 既存テストへの影響: なし
- lint: ruff check + format 通過

## TODO

- ブロックソルバー (`newton_raphson_block_contact`) の `timing` パラメータ対応
- Optuna 連携の実行テスト（slow marker付き）
- 感度分析プロットの3パラメータ以上への拡張（多次元可視化）
- 既存検証プロットの英語ラベル統一

---
