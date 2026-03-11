# xkep-cae コーディング規約

## 基本

- 全ての回答・設計仕様は**日本語**で記述
- markdown 文書には `README.md` へのバックリンクを貼る
- lint/format: `ruff check xkep_cae/ tests/` && `ruff format xkep_cae/ tests/`

## 2交代制運用（Codex / Claude Code）

常に互いへの引き継ぎを想定。statusファイルに状況を詳細記録。

### ステータス管理

- `docs/status/status-{index}.md` に記録（index最大が現在の状況）
- `docs/status/status-index.md` に一覧管理
- status に書いた内容は **commit メッセージと整合**を取る

### 作業完了時の必須手順

1. README.md 更新 → 2. status 新規作成/更新 → 3. status-index.md 更新 → 4. roadmap.md 更新
5. 不整合はその場で修正 or TODO追加 → 6. feature ごとにコミット → push

### ログ出力ルール

- 計算実行は**必ず tee でファイル出力**: `python script.py 2>&1 | tee /tmp/log-$(date +%s).log`
- `| tail -N` のみは禁止（途中経過が残らない）
- 収束ログには以下を含める: 時間増分カットバック、接触チャタリング、エネルギー収支、条件数

## 新機能の収束検証フロー（厳格化）

**原則: 新機能の収束テストは `scripts/` に書き、pytestを使わずに実行する。**

1. **scripts/で検証**: `scripts/verify_*.py` に収束確認スクリプトを作成
   - tee でログファイル出力必須
   - 収束後は3D梁形状の2D投影スナップショットで物理的妥当性を目視確認
   - 判断材料: カットバック回数、接触状態変化、エネルギー収支、条件数
2. **scriptsで確認が取れた機能をpytestに移行**: `tests/` に正式テストを追加
3. **視覚検証**: 変形メッシュの2D投影図をdocs/verification/に保存

## テストの分類

### プログラムテスト（API・収束）
- ソルバー収束、例外発生、API仕様準拠
- **16要素/ピッチ以上**厳守
- クラス名: `Test〇〇API`, `Test〇〇Convergence`

### 物理テスト（物理的妥当性）
- 貫入量、応力連続性、荷重オーダー、変形対称性、エネルギー保存
- クラス名: `Test〇〇Physics`

## 互換ヒストリー

機能置き換え時はstatusに記録。旧機能はdeprecatedマーカー付与。

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `newton_raphson_with_contact` | `newton_raphson_contact_ncp` | status-107→108 |

## 推奨ソルバー構成

- `newton_raphson_contact_ncp`（solver_ncp.py）
- UL+NCP統合: `ul_assembler` + `adaptive_timestepping=True`
- 解析的接線剛性: `analytical_tangent=True`（デフォルト）
- Line-to-line Gauss積分 + 同層除外 + Fischer-Burmeister NCP
- **摩擦あり**: `contact_mode="smooth_penalty"`（必須。NCP鞍点系は摩擦接線剛性符号問題で発散: status-147）

## 現在の状態

**2477テスト** — 2026-03-11

### ターゲット

> **1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了する。**

### 次の課題

**S3: 19本NCP収束達成** → 37本以上 → S4 → S5: ML → S6: 1000本6時間 → S7: GPU

詳細は `docs/roadmap.md` および `docs/status/status-index.md` を参照。

## フォーカスガード（AI セッション向け）

**以下を厳守すること。違反は作業のやり直しになる。**

### やるべきこと
- **S3 スケーリング**（19本→37本→61本→91本 NCP 収束）
- ソルバー性能改善（スパース最適化、並列化、メモリ削減）
- scripts/ での収束検証 → tests/ への移行

### やってはいけないこと
- **Process フレームワークのリファクタリング**（Phase 7 は凍結中）
- process-architecture.md §13 の C3-C12 対応（Phase 7 解凍まで着手禁止）
- 既存テストの大規模リファクタリング
- フォルダ構造の変更

### セッション開始時の確認手順
1. `docs/status/status-index.md` → 最新 status 番号を確認
2. 最新 `docs/status/status-{N}.md` を読む
3. 上の「やるべきこと」に合致する作業のみ実施
