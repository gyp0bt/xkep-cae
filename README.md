# xkep-cae

[![CI](https://github.com/gyp0bt/xkep-cae/actions/workflows/ci.yml/badge.svg)](https://github.com/gyp0bt/xkep-cae/actions/workflows/ci.yml)

ニッチドメイン問題を解くための自作有限要素ソルバー基盤。
構成則・要素・ソルバー・接触をモジュール化し、問題特化ソルバーを構成する。

## ターゲットマイルストーン

> **1000本撚線（10万節点）の曲げ揺動シミュレーションを6時間以内に完了する。**

| 項目 | 現状 | 目標 |
|------|------|------|
| 素線数 | **37本**（径方向圧縮Layer1で収束達成）| 1000本（~30,000 DOF, 長手分割で~100,000節点） |
| 計算時間 | 91本で~25分/曲げ揺動 | 1000本で6時間以内 |
| ソルバー | NCP: **37本収束達成**、S3改良12項目実装済み | NCP: 91本収束 |
| 接触ペア | 91本で~66,000候補 | 1000本で~730万候補→ML削減 |

## 現在の状態

**459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8 テスト** — 2026-04-17時点 | **MCDD Phase B-2 — 数理台帳 6 章完備 + `equation_index.py` + C15 拡張**（status-349）— status-348 の Huber 章（03）に続き、残り 5 章（**01 梁運動学 CR/TL/UL+Hermite / 02 接触幾何 StJacobian / 04 smooth penalty 摩擦 / 05 バリア関数被膜 / 06 Generalized-α 時間積分**）を整備し、**計 6 章 / 55 アンカー** の離散化方程式台帳を完成。`xkep_cae/mathematics/equation_index.py` を新設し、`<a id="(eq|inv|sym|sec)-...">` 抽出 + `MathematicalContract.equation_ref` の `<file>#<anchor>` 参照解決 API を提供（21 テスト、重複・未解決・bad_format 検出）。`contracts/validate_process_contracts.py` に **C15 拡張** を配線、`ProcessContractRegistry` と `AbstractProcess.contracts` ClassVar の二経路から `equation_ref` を収集して台帳と突合、docstring 例題は ClassVar 未宣言のため自然除外（8 テスト）。Phase A〜E / status-346〜356 の **4/11 完了** | 契約違反 **0件** | 条例違反 **0件** | [ロードマップ](docs/roadmap.md) | [ステータス一覧](docs/status/status-index.md)

| 分野 | 概要 |
|------|------|
| FEM基盤 | 梁（EB/Timo/CR/Cosserat）+ 非線形 + 動的解析 |
| 接触 | NCP + Line contact + smooth penalty Coulomb摩擦 |
| 撚線 | 7本摩擦曲げ+揺動収束、被膜+シース |
| アーキテクチャ | Process Architecture + Strategy Protocol + BenchmarkRunner |

**推奨ソルバー構成**: `contact_mode="smooth_penalty"` + NCP + 同層除外（[詳細](docs/roadmap.md#推奨ソルバー構成)）

## パッケージ構成

```
xkep_cae/
├── core/              # プロセスアーキテクチャ基盤（base, registry, runner 等）
│   ├── strategies/    # Strategy Protocol 定義
│   └── batch/         # BatchProcess（ワークフローオーケストレーション）
├── mathematics/       # MCDD 基盤（status-346〜）: MathematicalContract 型 + ProcessContractRegistry + @verified_by
├── contact/           # 接触モジュール
│   ├── penalty/       # PenaltyStrategy + 法線力 Process
│   ├── friction/      # FrictionStrategy + return mapping
│   ├── coating/       # CoatingStrategy + Kelvin-Voigt
│   ├── contact_force/ # ContactForceStrategy
│   ├── geometry/      # ContactGeometryStrategy
│   ├── setup/         # ContactSetupProcess
│   └── solver/        # ContactFrictionProcess + NUzawa
├── time_integration/  # TimeIntegrationStrategy（準静的/動的）
├── elements/          # 要素（CR梁/UL梁アセンブラ）
├── mesh/              # メッシュ生成（撚線メッシュ）
├── numerical_tests/   # 数値試験フレームワーク
├── output/            # 出力（CSV/JSON/VTK/GIF）
├── verify/            # 検証 Process（収束/エネルギー/接触）
└── tuning/            # チューニング
```

## ドキュメント

| ドキュメント | 内容 |
|------------|------|
| [ロードマップ](docs/roadmap.md) | 全体計画・マイルストーン・TODO |
| [ステータス一覧](docs/status/status-index.md) | 全statusファイル + テスト数推移 |
| [設計文書一覧](docs/design/README.md) | 設計仕様書リンク集 |
| [MCDD 設計仕様](xkep_cae/mathematics/docs/mathematics.md) | 数理契約駆動開発の型システム設計（status-346〜）|
| [数理台帳（docs/math/）](docs/math/README.md) | 離散化方程式の単一のソース・オブ・トゥルース（6 章 / 55 アンカー、status-348〜349）|
| [撚線ファイバー梁 設計仕様](xkep_cae/elements/docs/fiber_beam_strand.md) | 内部摩擦ヒステリシスを1本の梁で等価化する設計 |
| [beam_hysteresis 概念検証](work/beam_hysteresis/README.md) | 上記設計の裏付け数値実験 |

## インストール

```bash
pip install -e ".[dev]"
```

## テスト実行

```bash
# 高速テストのみ（~3分）
pytest tests/ -v -m "not slow and not external"

# 全テスト（~30分, slow含む）
pytest tests/ -v -m "not external"

# 新パッケージテストのみ
pytest xkep_cae/ -v
```

## Lint / Format

```bash
ruff check xkep_cae/ tests/
ruff format xkep_cae/ tests/
```

## ライセンス

[MIT License](LICENSE)

## 運用

本プロジェクトはCodexとClaude Codeの2交代制で運用。
引き継ぎ情報は [docs/status/](docs/status/status-index.md) を参照。
