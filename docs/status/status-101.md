# Status 101: ドキュメント大整理 — status-100記念リファクタリング

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-02
**ブランチ**: `claude/resolve-master-conflicts-biPJr`
**テスト数**: 1916（fast: 1542 / slow: 374）※変更なし

## 概要

status-100到達を機に、ドキュメント全体を整理・圧縮・構造化した。
プロジェクトの現在地と次のマイルストーンを明確にする。

## 実施内容

### 1. statusファイルのアーカイブ化

- status-001〜096（96ファイル）を `docs/status/archive/` に移動
- `status-index.md` を刷新:
  - アクティブstatus（097〜）をトップに表示
  - アーカイブは主要マイルストーンのみのサマリーテーブル
  - テスト数推移も主要マイルストーンのみ

### 2. README.md刷新

- フェーズ表を46行→8行に圧縮（分野×状態のコンパクト形式）
- **ターゲットマイルストーン**を明記:
  > 1000本撚線（10万節点）の曲げ揺動シミュレーションを6時間以内に完了する。
- モジュール別READMEへのリンクテーブル追加
- クイックスタート・依存ライブラリ等の冗長セクション削除

### 3. roadmap.md刷新

- 514行→190行に圧縮
- 完了フェーズをコンパクトなテーブルに集約
- **「現在の壁」**を明示（19本以上NCP収束失敗、処理時間）
- 計測済みスケーリングデータを掲載
- Phase S の依存関係を明確化
- 完了Phase詳細はarchive/とモジュールREADMEに委譲

### 4. モジュールREADME作成

以下の3つのモジュールREADMEを新規作成:

| ファイル | 内容 |
|---------|------|
| `xkep_cae/contact/README.md` | 接触アルゴリズム総覧、ソルバー構成、設計仕様書リンク、課題 |
| `xkep_cae/elements/README.md` | 要素ライブラリ一覧（DOF/特徴）、設計仕様書リンク |
| `xkep_cae/mesh/README.md` | 撚線メッシュ構成、スケーリングデータ |

設計仕様書（docs/contact/等）はそのまま残し、モジュールREADMEからリンクする方式。

### 5. CLAUDE.md簡潔化

- 「現在の状態」を1行サマリーに圧縮
- マイルストーン（10万節点6時間）を明記
- プロジェクト構成にarchiveパスを追加

## 変更ファイル

- `docs/status/archive/status-{001..096}.md` — 96ファイル移動（git mv）
- `docs/status/status-index.md` — 全面書き換え
- `README.md` — 全面書き換え
- `docs/roadmap.md` — 全面書き換え（514行→190行）
- `CLAUDE.md` — 簡潔化
- `xkep_cae/contact/README.md` — 新規
- `xkep_cae/elements/README.md` — 新規
- `xkep_cae/mesh/README.md` — 新規

## TODO

- [ ] S3: NCPソルバーで19本収束達成（前処理改良）
- [ ] S4: フルモデル（素線+被膜+シース）剛性BM
- [ ] S4: 大変形+文献値比較
- [ ] S5: 接触プリスクリーニングGNN
- [ ] S6: 1000本10万節点の曲げ揺動6時間以内

## 検証方針

- 6時間の1000本フル検証はリードタイムが大きいため、スケールを落とした同等の収束困難な問題で検証する
- 少ない本数（7/19/37本）ではフル検証を実施
- 計算時間が30分を超えるテストはstatus/roadmapに明記し、ユーザーがローカルで実行

## 確認事項

- docs/contact/ 配下の設計仕様書はそのまま残し、モジュールREADMEからリンクする方式とした。ファイル移動は既存の内部リンク・外部参照を壊すリスクがあるため見送り。
- 旧statusファイルのarchive移動により、roadmap.mdやstatus内の相対リンク（`status/status-XXX.md`）は `status/archive/status-XXX.md` に変わる。roadmap内のリンクは `archive/` プレフィックスに更新済み。

---
