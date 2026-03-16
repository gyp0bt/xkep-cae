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

**~2260テスト + 204 新パッケージテスト** — 2026-03-16時点 | [ロードマップ](docs/roadmap.md) | [ステータス一覧](docs/status/status-index.md)

| 分野 | 概要 | 状態 |
|------|------|------|
| FEM基盤 | 梁（EB/Timo/CR/Cosserat）+ 平面 + HEX8 + 非線形 + 動的解析 | 完了（deprecated） |
| 接触 | NCP + Line contact + Mortar + smooth penalty Coulomb摩擦 | 完了（deprecated） |
| 撚線 | 7本摩擦曲げ+揺動収束、被膜+シース、ヒステリシス | 完了（deprecated） |
| 高速化 | NCP 6x + 要素12.6x バッチ化、ソルバー一本化 | 完了（deprecated） |
| **脱出ポット計画** | **新 xkep_cae を Process Architecture でゼロ構築** | **Phase 2 完了 + C16スコープ拡大 + process/削除** |

**推奨ソルバー構成**: `contact_mode="smooth_penalty"` + NCP + 同層除外（[詳細](docs/roadmap.md#推奨ソルバー構成)）

## パッケージ構成

### xkep_cae/（新パッケージ — Process Architecture ベース）

脱出ポット計画により Process Architecture でゼロから構築中。
C14（deprecated インポート禁止）/ C15（ドキュメント存在検証）/ C16（滅菌チェック）の契約ルール適用。

```
xkep_cae/
├── core/              # プロセスアーキテクチャ基盤（base, registry, runner 等）
│   └── strategies/    # Strategy Protocol 定義
├── contact/           # 接触 Strategy 実装
│   ├── penalty/       # ✅ PenaltyStrategy + 法線力 Process（34テスト）
│   ├── friction/      # ✅ FrictionStrategy + return mapping（52テスト）
│   ├── coating/       # ✅ CoatingStrategy + Kelvin-Voigt（12テスト）
│   ├── contact_force/ # ✅ ContactForceStrategy（テスト済み）
│   └── geometry/      # ✅ ContactGeometryStrategy（テスト済み）
├── time_integration/  # ✅ TimeIntegrationStrategy（準静的/動的）
├── elements/          # 要素（移行予定）
├── materials/         # 構成則（移行予定）
├── sections/          # 断面モデル（移行予定）
├── math/              # 数学ユーティリティ（移行予定）
├── mesh/              # メッシュ生成（移行予定）
├── io/                # Abaqus .inp パーサー（移行予定）
├── output/            # 出力（移行予定）
├── thermal/           # 熱伝導（移行予定）
└── tuning/            # チューニング（移行予定）
```

### xkep_cae_deprecated/（旧パッケージ — 段階的移行元）

全機能が実装済み。脱出ポット計画で新 xkep_cae に順次移行される。

```
xkep_cae_deprecated/
├── core/           # Protocol 定義（Element, Constitutive, State）
├── elements/       # 要素（Q4, TRI3/6, Beam, Cosserat, HEX8）
├── materials/      # 構成則（弾性, 1D/3D弾塑性）
├── sections/       # 断面モデル（BeamSection, FiberSection）
├── math/           # 数学ユーティリティ（四元数, SO(3)）
├── process/        # プロセスアーキテクチャ基盤（AbstractProcess + Strategy）
├── contact/        # 梁–梁接触（NCP, Mortar, 摩擦, グラフ）
├── mesh/           # メッシュ生成（撚線, シース, チューブ）
├── thermal/        # 熱伝導FEM + GNN/PINNサロゲート
├── numerical_tests/ # 数値試験フレームワーク
├── output/         # 過渡応答出力（CSV/JSON/VTK/GIF）
├── io/             # Abaqus .inp パーサー
├── solver.py       # 線形/非線形ソルバー
├── assembly.py     # アセンブリ
├── dynamics.py     # 動的解析
├── bc.py           # 境界条件
└── api.py          # 高レベル API
```

## ドキュメント

| ドキュメント | 内容 |
|------------|------|
| [ロードマップ](docs/roadmap.md) | 全体計画・マイルストーン・TODO |
| [ステータス一覧](docs/status/status-index.md) | 全statusファイル + テスト数推移 |
| [設計文書一覧](docs/design/README.md) | 新旧設計仕様書リンク集 |
| [検証画像ギャラリー](docs/verification/gallery.md) | 全検証画像の一覧（新しい順） |
| [検証文書](docs/verification/validation.md) | 解析解・厳密解との比較（検証図20枚） |
| [接触テストカタログ](docs/verification/contact_test_catalog.md) | 全接触テスト（~240テスト） |
| [使用例](docs/reference/examples.md) | API・梁要素・非線形・弾塑性のコード例（旧API） |
| [Abaqus差異](docs/reference/abaqus-differences.md) | xkep-caeとAbaqusの要素定式化の差異 |

### 旧モジュール別ドキュメント

各旧モジュールの設計仕様は `xkep_cae_deprecated/` 配下に配置。

| モジュール | README | 主な設計仕様 |
|-----------|--------|------------|
| [contact/](xkep_cae_deprecated/contact/docs/README.md) | 接触アルゴリズム総覧 | NCP/AL/Mortar/摩擦/Line contact |
| [elements/](xkep_cae_deprecated/elements/docs/README.md) | 要素ライブラリ | Q4/TRI/Beam/Cosserat/HEX8 |
| [mesh/](xkep_cae_deprecated/mesh/docs/README.md) | メッシュ生成 | 撚線/シース/被膜 |

## インストール

```bash
pip install -e ".[dev]"
# ML機能（GNN/PINN）を使う場合
pip install -e ".[dev,ml]"
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
