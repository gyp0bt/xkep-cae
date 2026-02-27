# xkep-cae コーディング規約

## 言語・文書化

- 全ての回答・設計仕様は**日本語**で記述する
- すべての markdown 文書には原則 project 直下の `README.md` へのバックリンクを貼る

## 2交代制運用（Codex / Claude Code）

本プロジェクトは **Codex と Claude Code の2交代制**で運用する。常に互いへの引き継ぎを想定すること。

### ステータス管理

- 実装状況は `docs/status/status-{index}.md` に記録する
- **現在の状況**は index が最大の status ファイルに書かれている
- **`docs/status/status-index.md`** にステータス一覧を管理する（新規status作成時に必ず行を追加すること）
- status に書いた内容は **git の commit メッセージと整合**を取ること
- 実装状況は細かく書き出す（別の AI アシスタントが参照して簡便に状況を把握する目的）

### 作業完了時の必須手順

1. **README.md** を更新（現在の状態、ステータスリンク）
2. **status ファイル**を新規作成 or 更新（TODO は status に記入）
3. **status-index.md** を更新（新規statusの行を追加）
4. **roadmap.md** を更新（チェックボックス、テスト数、「現在地」）
5. 実装とドキュメントの不整合を発見したら**その場で修正**するか、TODO に追加
6. **feature ごとにコミットを切って**、最後に push

### 確認事項・懸念

- ユーザーへの確認事項や設計上の懸念は **status ファイルに書き出す**こと

## コード規約

- テスト駆動: 各要素・構成則は解析解またはリファレンスソルバーとの比較テスト必須
- 後方互換性を保ちながら拡張（既存テストを破壊しない）
- lint/format: `ruff check xkep_cae/ tests/` && `ruff format xkep_cae/ tests/`

### 検証結果のドキュメント化

- 文献値や解析解との比較結果は `docs/verification/` に**図付き**で残す
- 図は `tests/generate_verification_plots.py` で生成（matplotlib → PNG）
- `pytest` 実行時にはプロットを生成しない（図生成は別スクリプト）
- 図には解析解（実線）と数値解（マーカー）を重ね描きし、一致を視覚的に確認できるようにする

## プロジェクト構成

```
xkep_cae/
├── core/           # Protocol 定義（Element, Constitutive, State）
├── elements/       # 要素（Q4, TRI3/6, Beam, Cosserat, HEX8）
├── materials/      # 構成則（弾性, 1D/3D弾塑性）
├── sections/       # 断面モデル（BeamSection, FiberSection）
├── math/           # 数学ユーティリティ（四元数, SO(3)）
├── contact/        # 梁–梁接触（Broadphase, AL, 摩擦, グラフ）
├── mesh/           # メッシュ生成（撚線, シース, チューブ）
├── thermal/        # 熱伝導FEM + GNN/PINNサロゲート
├── numerical_tests/ # 数値試験フレームワーク
├── output/         # 過渡応答出力（CSV/JSON/VTK/GIF）
├── io/             # Abaqus .inp パーサー
├── solver.py       # 線形/非線形ソルバー（NR, 弧長法）
├── assembly.py     # アセンブリ
├── dynamics.py     # 動的解析（Newmark-β, HHT-α, 陽解法）
├── bc.py           # 境界条件
└── api.py          # 高レベル API
docs/
├── roadmap.md      # 全体ロードマップ + TODO
├── archive/        # 完了済みPhase詳細設計
├── status/         # ステータスファイル群（76個）
├── contact/        # 接触モジュール仕様群
└── verification/   # バリデーション文書・検証図
```

## 現在の状態

Phase 1〜3 + Phase 4.1〜4.2 + Phase 5.1〜5.4 + Phase C0〜C5 + Phase C6-L1〜L2 + Phase 4.7 Level 0 + Phase 4.7 Level 0.5 S1-S4 + ブロック前処理ソルバー + adaptive omega + HEX8要素ファミリ + 過渡応答出力 + Phase 6.0 GNN/PINNサロゲートPoC + GitHub Actions CI 完了（1775テスト: fast 1381 / slow 336 / skip 56）。Phase 4.3（von Mises 3D）は凍結。
詳細は `docs/roadmap.md` および最新の status ファイル（`docs/status/status-index.md` で一覧確認）を参照。
