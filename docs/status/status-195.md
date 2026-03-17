# status-195: numerical_tests モジュール新 xkep_cae 移植

[← README](../../README.md) | [← status-index](status-index.md)

## 日付

2026-03-17

## 作業者

Claude Code

## ブランチ

claude/check-status-todos-OaCjB

## 概要

status-194 の TODO「numerical_tests モジュールの新 xkep_cae への移植」を実行。
BackendRegistry パターン（依存性注入）で C14 準拠を維持しながら、8ファイル・約1400行を移植。

## アーキテクチャ: BackendRegistry パターン

```
xkep_cae/numerical_tests/          # C14 準拠（deprecated import なし）
├── _backend.py                     # Protocol 定義 + BackendRegistry シングルトン
├── core.py                         # データクラス・解析解・メッシュ生成（純粋関数）
├── runner.py                       # 静的試験ランナー（backend 経由）
├── frequency.py                    # 周波数応答試験（backend 経由）
├── dynamic_runner.py               # 動的試験ランナー（backend 経由）
├── csv_export.py                   # CSV 出力
├── inp_input.py                    # Abaqus ライク入力パーサー
├── wire_bending_benchmark.py       # 撚線曲げ揺動ベンチマーク（backend 経由）
└── __init__.py                     # 公開 API re-export

tests/conftest.py                   # ← ここで deprecated 実装を注入（C14 対象外）
```

**設計思想**: numerical_tests はテストフレームワーク（定義 + 解析解）であり、
FEM 計算実装（要素剛性、BC 適用、線形ソルバー）は別モジュールの責務。
BackendRegistry で Protocol ベースのインターフェースを定義し、
テスト conftest.py で deprecated 実装を注入する。

## 変更内容

### 1. 新規ファイル（xkep_cae/numerical_tests/）

| ファイル | 行数 | 内容 |
|---------|------|------|
| `_backend.py` | 288 | Protocol 定義 + BackendRegistry（static/frequency/dynamic 3段階 configure） |
| `core.py` | ~580 | データクラス・解析解・メッシュ生成（deprecated 依存ゼロ） |
| `runner.py` | ~320 | 静的試験ランナー（backend.ke_func_factory/solve 経由） |
| `frequency.py` | ~304 | 周波数応答（backend.beam*_mass_* 経由） |
| `dynamic_runner.py` | ~290 | 動的試験（backend.transient_solver 経由） |
| `csv_export.py` | ~240 | CSV 出力（core のみ依存） |
| `inp_input.py` | ~216 | 入力パーサー（core のみ依存） |
| `wire_bending_benchmark.py` | ~257 | 撚線ベンチマーク（backend._bending_oscillation_runner 経由） |
| `__init__.py` | ~40 | 公開 API re-export |

### 2. バックエンド注入（tests/conftest.py）

`_configure_numerical_tests_backend()` 関数を追加:

- **静的試験**: `apply_dirichlet` / `solve_displacement` / `_ke_func_factory` / `_section_force_computer`
- **周波数応答**: `eb_beam2d_*` / `timo_beam3d_*` 質量行列関数 5 種
- **動的試験**: `NonlinearTransientConfig` / `solve_nonlinear_transient` / CR・Cosserat assembler factory

### 3. テスト有効化

- `tests/test_numerical_tests.py`: `pytest.importorskip()` ガードを除去

## BackendRegistry Protocol 一覧

| カテゴリ | Protocol/型 | 用途 |
|---------|-------------|------|
| 静的 | `DirichletApplier` | 境界条件適用 |
| 静的 | `LinearSolver` | 線形連立方程式ソルバー |
| 静的 | `KeFuncFactory` | 要素剛性行列ファクトリ |
| 静的 | `SectionForceComputer` | 断面力計算 |
| 周波数 | `MassAssembler2D/3D` | 質量行列アセンブリ |
| 周波数 | 要素レベル質量行列関数 5 種 | lumped/consistent mass |
| 動的 | `TransientSolver` | 過渡応答ソルバー |
| 動的 | `NLAssemblerFactory` | 非線形 assembler ファクトリ |

## テスト結果

- `tests/test_numerical_tests.py`: **70 passed** in 0.58s
- C14 違反: **0 件**（新規追加）
- ruff check: エラー 0 件
- ruff format: 全ファイルフォーマット済み

## TODO

- [ ] S3 xfail テストの Process API 対応版作成

---
