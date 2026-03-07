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
| ソルバー | NCP: **37本収束達成**、S3改良12項目（適応Δt/AMG/k_pen自動推定/残差スケーリング/接触力ランプ/初期貫入補正等）実装済み | NCP: 91本収束 |
| 接触ペア | 91本で~66,000候補 | 1000本で~730万候補→ML削減 |

## 現在の状態

**2271テスト（fast: 1691 + 1 xfailed / slow: 362 - 10 xfailed / deprecated: 218）** — 2026-03-07時点

| 分野 | 内容 | 状態 |
|------|------|------|
| FEM基盤 | 梁要素（EB/Timo/Cosserat/CR）、平面要素（Q4/TRI）、固体要素（HEX8）| 完了 |
| 非線形 | 幾何学的非線形（NR/弧長/CR/TL/UL）、弾塑性、ファイバーモデル | 完了 |
| 動的解析 | Newmark-β/HHT-α/Generalized-α/陽解法/モーダル減衰 | 完了 |
| 接触 | 梁–梁接触（NCP+Mortar+Line contact+摩擦）| 完了 |
| 撚線モデル | 7本撚り収束、被膜/シース、ヒステリシス観測 | 完了 |
| 高速化 | COO/CSRベクトル化、共有メモリ並列、ブロック前処理 | 完了 |
| GNN/PINN | 2D熱伝導サロゲートPoC（R²=0.995） | 完了 |
| **大規模収束** | **37本NCP収束達成（径方向圧縮Layer1）、段階的活性化NCP移植、曲げ揺動テスト追加** | **91本曲げ揺動で収束** |
| **チューニング基盤** | TuningTaskスキーマ + 検証プロット14種 + YAML/JSON直列化 + Optuna連携 | AI自動チューニング |
| **可視化** | 四元数2D投影・接触診断パネル・変形比較・層別レンダリング（mplot3d→2D投影置換済み） | 数値困難の診断 |

**推奨ソルバー構成**: NCP Semi-smooth Newton + Line contact + Mortar + 同層除外（[詳細](docs/roadmap.md#推奨ソルバー構成リファレンス構成)）

## ドキュメント

| ドキュメント | 内容 |
|------------|------|
| [ロードマップ](docs/roadmap.md) | 全体計画・マイルストーン・TODO |
| [ステータス一覧](docs/status/status-index.md) | 全127件のstatus + テスト数推移 |
| [検証文書](docs/verification/validation.md) | 解析解・厳密解との比較（検証図20枚） |
| [接触テストカタログ](docs/verification/contact_test_catalog.md) | 全接触テスト（~240テスト） |
| [使用例](docs/examples.md) | API・梁要素・非線形・弾塑性のコード例 |

### モジュール別ドキュメント

各モジュールの設計仕様は、対応するソースディレクトリの README.md に配置。

| モジュール | README | 主な設計仕様 |
|-----------|--------|------------|
| [contact/](xkep_cae/contact/README.md) | 接触アルゴリズム総覧 | NCP/AL/Mortar/摩擦/Line contact |
| [elements/](xkep_cae/elements/README.md) | 要素ライブラリ | Q4/TRI/Beam/Cosserat/HEX8 |
| [mesh/](xkep_cae/mesh/README.md) | メッシュ生成 | 撚線/シース/被膜 |
| [tuning/](xkep_cae/tuning/) | チューニングタスク | TuningTask/スケーリング/検証プロット |
| [solver](xkep_cae/solver.py) | 非線形ソルバー | NR/弧長法 |

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
```

## Lint / Format

```bash
ruff check xkep_cae/ tests/
ruff format xkep_cae/ tests/
```

## プロジェクト構成

```
xkep_cae/
├── core/           # Protocol 定義（Element, Constitutive, State）
├── elements/       # 要素（Q4, TRI3/6, Beam, Cosserat, HEX8）
├── materials/      # 構成則（弾性, 1D/3D弾塑性）
├── sections/       # 断面モデル（BeamSection, FiberSection）
├── math/           # 数学ユーティリティ（四元数, SO(3)）
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

## ライセンス

[MIT License](LICENSE)

## 運用

本プロジェクトはCodexとClaude Codeの2交代制で運用。
引き継ぎ情報は [docs/status/](docs/status/status-index.md) を参照。
