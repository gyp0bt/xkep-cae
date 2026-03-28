# xkep-cae コーディング規約

## 基本

- 全ての回答・設計仕様は**日本語**で記述
- markdown 文書には `README.md` へのバックリンクを貼る
- lint/format: `ruff check xkep_cae/ tests/` && `ruff format xkep_cae/ tests/`
- 機能は可能な限りprocessクラスとして実装すること。

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

**原則: 新機能の収束テストは pytest で実行する。必要に応じて `contracts/` に検証スクリプトを配置。**

1. **テストで検証**: `tests/` に正式テストを追加
   - tee でログファイル出力必須
   - 収束後は3D梁形状の2D投影スナップショットで物理的妥当性を目視確認
   - 判断材料: カットバック回数、接触状態変化、エネルギー収支、条件数
2. **視覚検証**: 変形メッシュの2D投影図をdocs/verification/に保存

## テストの分類

### プログラムテスト（API・収束）
- ソルバー収束、例外発生、API仕様準拠
- **16要素/ピッチ以上**厳守
- クラス名: `Test〇〇API`, `Test〇〇Convergence`

### 物理テスト（物理的妥当性）
- 貫入量、応力連続性、荷重オーダー、変形対称性、エネルギー保存
- クラス名: `Test〇〇Physics`

## 互換ヒストリー

移行完了。`__xkep_cae_deprecated/` は status-207 で完全削除。
詳細な移行履歴は status-107〜206 を参照。

## 推奨ソルバー構成

- Fischer-Burmeister NCP（Huber）が主力接触力評価
- UL+NCP統合: `ul_assembler` + `adaptive_timestepping=True`
- 解析的接線剛性: `analytical_tangent=True`（デフォルト）
- Line-to-line Gauss積分 + 同素線除外（`exclude_same_strand=True`）
- **摩擦あり**: `contact_mode="smooth_penalty"`（必須。NCP鞍点系は摩擦接線剛性符号問題で発散: status-147）
- **Uzawa凍結**: `n_uzawa_max=1`（純粋ペナルティ。拡大ラグランジアンは status-221 で凍結）

## 現在の状態

**200+10s+16+3+23+1+6 テスト** — 2026-03-28 | 契約違反 **0件** | 条例違反 **0件**

### ターゲット

> **1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了する。**

### 次の課題

**MPC+接触の接線剛性不整合** — status-255:
- MPC u伝搬修正 + NR内再射影 + 拡張系ラッパー（status-254で完了）
- MPC縮退系残差判定 + u_pred MPC射影 + ストール検知拡張（status-255で完了）
- frac=0.35到達（NR残差停滞: 接線剛性不整合の可能性）
- **次**: MPC+接触の接線剛性FD診断 → 不整合箇所特定
**並行**: 摩擦アセンブリProcess化（B1-B4）

詳細は `docs/roadmap.md` および `docs/status/status-index.md` を参照。

## フォーカスガード（AI セッション向け）

**以下を厳守すること。違反は作業のやり直しになる。**

## やるべきこと
- **MPC+接触のNR収束改善**（frac=0.35→1.0）
  - ~~DOF消去MPC実装（端部剛体結合）~~ ← status-253で完了
  - ~~StrandBendingOscillationProcess 実装~~ ← status-253で完了
  - ~~MPC u伝搬修正 + 収束実行テスト~~ ← status-254で完了（frac=0.35到達）
  - ~~MPC + 動的ソルバーの力残差整合性確認~~ ← status-254で確認（slave残差問題特定）
  - ~~MPC縮退系残差判定 + u_pred射影 + ストール検知拡張~~ ← status-255で完了
  - MPC+接触の接線剛性FD診断（du方向が残差を減少させない原因特定）
- プロセス脱法修正（Phase D〜E、status-249 参照）
  - ~~A1-A3: アセンブラProcess化~~ ← status-250で完了
  - ~~C2-C3: 幾何計算Process化~~ ← status-255で完了
- NR 残差収束速度の改善（中盤後〜終盤で 25 反復が力収束に不足、disp 収束で抜ける状態）
- Hermite 非局所 ∂g/∂u 対応（4ノードペア外の DOF 結合）

## やってはいけないこと
- 管理上processクラスとすべきロジックをあえてプライベート関数や迂回ロジックに替えること
- 収束トライ時に目標を緩和して本質的対策を先送りにすること

## STA2 防止ルール（STAP細胞の二の舞防止）
- **increment の定義**: increment は成功した dt ステップの数。カットバック（時間増分の縮小リトライ）は increment に含めない。`_incr_count` は成功パスでのみインクリメントし、`max_increments` はカットバック回数に侵食されない。
- **結果の再現性**: 全ての収束結果は tee でログ保存し、YAML 出力と照合可能にすること。ベースライン（変更前）を先に確認してから改善テストを実施。
- **数値の捏造禁止**: 収束しない場合は「収束しなかった」と報告する。目標を事後的に緩和して達成を装わない。

### 担当者間再現性ルール（status-246 追加）
- **ベンチマーク条件の記録**: テスト名、ブランチ名、コミットハッシュ、実行コマンドを tee ログおよび status ファイルに記録すること。
- **変更前ベースラインの先行取得**: 性能改善テスト前に必ず `git stash` で変更前コードのベースラインを計測し、ログに残す。
- **再現手順の status 記載**: status ファイルに「再現手順」セクションを設け、別の担当者が同じ結果を得られるコマンド列を明記する。
- **Process profiling の活用**: `ProcessMetaclass._profile_data` による自動計測結果を活用し、手動計測に頼らない仕組みを推進する。

### セッション開始時の確認手順
1. `docs/status/status-index.md` → 最新 status 番号を確認
2. 最新 `docs/status/status-{N}.md` を読む
3. `python contracts/validate_process_contracts.py` を実行し、エラー一覧を確認
4. 上の「やるべきこと」に合致する作業のみ実施
