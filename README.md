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

**766 passed 5 skipped** — 2026-05-16 | 契約違反 **0件** | 条例違反 **0件**

**最新 status**: [status-400](docs/status/status-400.md) — `VtkExportProcess` 実装。ParaView 用 VTK XML 出力 PostProcess、依存追加なし（生 XML）、汎用 1D 梁モデル対応。`.pvd` 時系列 + `.vtu` を出力し ParaView で曲げ揺動を視覚確認可能。`MeshData` + `SolverResultData` から `displacement` / `rotation` (PointData) + `axial_strain` (CellData) を書き出す。

**前 status**: [status-399](docs/status/status-399.md) — `explicit_n_sub_cycles_per_increment` 実装、ε-1 で N=2000 で rel_err 0.01% asymptote 到達、MCDD 凍結解除条件 (5) を単 strand 規模で PASS。

**Phase ε 進行中**: ε-1 = 1 strand 接触なし ✅ → ε-2 = 3 strand 接触あり（status-401 予定） → ε-3 = 7 strand → ε-4 = 19 strand（本命、MCDD 凍結解除条件 (2)(3)(5) 同時達成試行）。

**完了履歴**: Phase A/B/C/E + NR escape hatch 全候補 (a)〜(g) 検証 + explicit 時間積分 Phase 1〜3 + Phase α/β/γ foundation 再検証 + ε-1 sub-cycling 実装 + ParaView 連携。詳細は [status-index](docs/status/status-index.md) および [verification_matrix](docs/status/verification_matrix.md) 参照。

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
