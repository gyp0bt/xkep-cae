# status-161: フォーカスガード逆転 — Process Architecture Phase 7 開始

[← README](../../README.md) | [← status-index](status-index.md) | [← status-160](status-160.md)

**日付**: 2026-03-11
**テスト数**: 2477（変更なし — 回帰テスト全通過）

## 概要

フォーカスガードを逆転し、Process Architecture Phase 7 完遂を最優先に切替。
S3 スケーリング（NCP収束改善）は凍結。

## 背景

- Process Architecture Phase 1-6 完了（status-150〜160）
- Phase 7 残: BatchProcess, VerifyProcess, 1:1テスト, C3-C12契約違反検知
- コンテキスト肥大化によりAI/担当者双方が迷子になる問題 → Process Architecture 完遂で整理
- 前セッション（status-160）はS3重視・Phase 7凍結だったが、今回は完全逆転

## 実施内容

### A: CLAUDE.md フォーカスガード逆転

- **やるべきこと**: Process Phase 7 完遂 + C3-C12 契約違反検知
- **やってはいけないこと**: S3 スケーリング（NCP収束改善は凍結）
- 「次の課題」セクションも R1 Phase 7 に更新
- セッション開始手順に `validate_process_contracts.py` 実行を追加

### B: roadmap.md 更新

- R1 Phase 7 を「現在地」に変更
- S3 を「凍結中（R1 Phase 7 完了まで）」に変更

### C: 契約違反テストスクリプト拡張

`scripts/validate_process_contracts.py` に C6, C9, C11, C12 チェックを追加。
後続AIセッションがエラー出力を見て自然にPhase 7実装を進められるようにする。

### D: pytest 契約テスト新規作成

`xkep_cae/process/tests/test_contracts.py` を新規作成。
process-architecture.md §13 に基づくテスト群。**意図的にFAILさせるテストを含む**。
リファクタリングの完了条件 = 全テストパス。

## Phase 7 作業計画

| サブフェーズ | 内容 | ゴール |
|---|---|---|
| 7-A | concrete/ 1:1テスト追加（5ファイル） | C3エラーをゼロに |
| 7-B | VerifyProcess 具象実装（3クラス+テスト） | verify/ 空ディレクトリ解消 |
| 7-C | BatchProcess 具象実装 | batch/ 空ディレクトリ解消 + C12 対応 |
| 7-D | C6,C9,C11 契約違反検知実装 | validate スクリプトエラーゼロ |
| 7-E | Mortar/L2L統合, update_geometry委譲 | status-159残TODO消化 |

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `CLAUDE.md` | フォーカスガード逆転 |
| `docs/roadmap.md` | Phase 7 現在地 + S3 凍結 |
| `docs/status/status-161.md` | 本ファイル |
| `docs/status/status-index.md` | インデックス追加 |
| `scripts/validate_process_contracts.py` | C6,C9,C11,C12 チェック追加 |
| `xkep_cae/process/tests/test_contracts.py` | 契約テスト新規作成 |

## TODO（後続セッション向け）

- [ ] Phase 7-A: concrete/ 1:1テスト（5ファイル）→ C3エラーゼロ
- [ ] Phase 7-B: VerifyProcess 3クラス + テスト
- [ ] Phase 7-C: StrandBendingBatchProcess + テスト → C12対応
- [ ] Phase 7-D: C6意味論テスト, C9チェックサム検証, C11推移的依存チェック
- [ ] Phase 7-E: Mortar/L2L ContactForceStrategy 統合
- [ ] Phase 7-E: manager.update_geometry() 完全委譲
- [ ] process-architecture.md §10 を Phase 7 完了で更新
- [ ] roadmap.md 最終更新

## 凍結事項

- S3 スケーリング（19本→37本→61本→91本 NCP 収束）
- ソルバー性能改善
- scripts/ での新規収束検証スクリプト

## 運用メモ

- フォーカスガードの逆転は意図的。2交代制運用でタスク優先度を切り替える仕組み。
- 契約違反テストの FAIL は意図的。後続AIセッションのエラー修正駆動でリファクタリングを進める設計。
- CI は一時的に赤になるが、Phase 7 完遂で全テストパスに復帰する想定。
