# contact/ — 梁–梁接触モジュール

[← README](../../../README.md) | [← roadmap](../../../docs/roadmap.md)

## 概要

梁–梁接触の検出・力評価・接線剛性・ソルバーを実装する。
撚線モデルの接触NR収束が本モジュールの主要課題。

## アーキテクチャ

```
pair.py          ContactPair / ContactConfig / ContactManager（接触ペア管理）
geometry.py      最近接点探索（Newton法 + 安全探索）
broadphase.py    AABB格子ベースの候補検出 + 中点距離プリスクリーニング
law_normal.py    法線力（AL乗数更新 + λ_nキャッピング）
law_friction.py  Coulomb摩擦 return mapping + μランプ
line_contact.py  Line-to-line Gauss積分（セグメント間力評価）
mortar.py        Mortar離散化（セグメント境界の力連続化）
ncp.py           NCP関数（Fischer-Burmeister / min）
assembly.py      接触力・接触剛性のアセンブリ（numpyベクトル化済み）
line_search.py   merit line search（接触付きNRのステップ長制御）
graph.py         ContactGraph / ContactGraphHistory（接触状態の統計）
solver_ncp.py    NCP Semi-smooth Newton ソルバー ← 推奨
solver_hooks.py  AL法ソルバー（Outer/Inner分離）← レガシー
bc_utils.py      接触付き境界条件ユーティリティ
sheath_contact.py  シース-素線接触
kpen_features.py   k_pen推定用特徴量
prescreening_data.py  ML用プリスクリーニングデータ
```

## ソルバー構成

### 推奨: NCP Semi-smooth Newton（`solver_ncp.py`）

- Outer loop 不要（変位 u とラグランジュ乗数 λ を同時更新）
- NCP関数（Fischer-Burmeister）で活性セット判定を自動化
- Line contact + Mortar + 摩擦統合済み
- k_pen は正則化パラメータのみ（AL乗数暴走なし）

### レガシー: AL法（`solver_hooks.py`）

- Outer/Inner分離（Inner NR → Outer AL乗数更新）
- adaptive omega / λ_n キャッピング / best-state fallback 等のワークアラウンドが必要
- 7本撚りでは安定だが、大規模での収束に課題

## 設計仕様書

| 仕様書 | 内容 |
|--------|------|
| [beam_beam_contact_spec_v0.1](beam_beam_contact_spec_v0.1.md) | 接触基盤設計（C0-C5） |
| [contact-algorithm-overhaul-c6](contact-algorithm-overhaul-c6.md) | C6接触アルゴリズム全面改修 |
| [contact-prescreening-gnn-design](contact-prescreening-gnn-design.md) | 接触プリスクリーニングGNN |
| [kpen-estimation-ml-design](kpen-estimation-ml-design.md) | k_pen推定MLモデル |
| [twisted_wire_contact_improvement](twisted_wire_contact_improvement.md) | 撚線接触改善 |
| [design-index](design-index.md) | 接触設計文書の一覧・実装状況 |

## 現在の課題

- **19本以上のNCP収束**: ブロック前処理の精度不足。ILU drop_tol / Schur近似の改良が必要。
- **broadphase O(n²)**: 1000本で733万候補ペア。ML削減（S5）が前提。
- **geometry_update**: 大規模で処理時間が急増（O(n²)的スケーリング）。

## テスト

- `tests/contact/` — 518 passed, 26 skipped（fast）
- 接触テストカタログ: [docs/verification/contact_test_catalog.md](../../../docs/verification/contact_test_catalog.md)
