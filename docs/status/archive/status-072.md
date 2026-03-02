# status-072: ドキュメント整理・TODO棚卸し

[← README](../../README.md) | [← status-index](status-index.md)

- **日付**: 2026-02-27
- **テスト数**: 1629（変更なし、ドキュメント整理のみ）
- **ブランチ**: claude/organize-todos-and-docs-AxE3g

## 概要

status-060〜071 に蓄積された未実行TODOを棚卸しし、README・roadmap・docs を現在のコードベースに合わせて整理・圧縮した。

## 実施内容

### 1. roadmap.md 大幅圧縮（863行 → ~230行）

- **完了済みPhase（1-3, C, 5）の詳細設計情報**（Protocol定義のコード例、数値試験のASCII図、試験配置図、参考文献など）を `docs/archive/completed-phases.md` に分割
- **「現在地」セクション**を1行の超長文列挙からフェーズ一覧表に変更
- **「実装済み」テーブル**を4カテゴリ（要素、材料・断面、ソルバー・解析、撚線・メッシュ・I/O）に分割整理
- **完了済みの「次の優先」リスト**（取り消し線13項目）を削除
- **TODO一覧セクション**を新設（高・中・低優先の3段階）

### 2. README.md 簡潔化

- **「現在の状態」**を超長文列挙からフェーズ一覧表（13行）に変更
- **「ドキュメント」**を計画・設計/設計仕様/バリデーション/利用ガイドの4カテゴリに整理
- **プロジェクト構成**を現在のディレクトリ構造に合わせて更新（contact/, mesh/, thermal/, output/, io/, dynamics.py を追加）
- **インストール**に `[ml]` オプション追加
- **テスト実行**に fast/全テストの2パターンを記載

### 3. CLAUDE.md 更新

- プロジェクト構成を現在の実態に合わせて更新
- 「現在の状態」を簡潔化

### 4. docs/contact/README.md 新設

- 接触モジュール設計文書群（5ファイル）の目次・対応Phaseを一覧表で整理

### 5. docs/archive/ 新設

- `completed-phases.md` — roadmapから分割した完了済みPhase詳細設計情報

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `docs/roadmap.md` | 863行→~230行に圧縮、完了済み詳細をアーカイブ移動 |
| `README.md` | 現在の状態を表形式化、ドキュメント構成整理 |
| `CLAUDE.md` | プロジェクト構成・現在の状態を更新 |
| `docs/contact/README.md` | **新規** — 接触設計文書の目次 |
| `docs/archive/completed-phases.md` | **新規** — 完了済みPhase詳細設計アーカイブ |
| `docs/status/status-072.md` | **新規** — 本ステータス |
| `docs/status/status-index.md` | status-072 行を追加 |

---

## TODO 棚卸し（status-060〜071 から収集・整理）

### 高優先（次の実装セッションで対応）

| # | TODO | 出典 | 備考 |
|---|------|------|------|
| 1 | CI実行結果確認（初回push後） | status-070/071 | force push後のGHA動作確認 |
| 2 | CIキャッシュ導入（actions/cache） | status-070 | pip install 高速化 |
| 3 | CIバッジをREADMEに追加 | status-070 | |
| 4 | Stage S3: シース-素線/被膜 有限滑り | status-064/065 | θ再配置 + friction_return_mapping統合 |
| 5 | Stage S4: シース-シース接触 | status-064/065 | 円-円ペナルティ、ContactPair流用 |
| 6 | 7本撚りブロック分解ソルバー | status-061-063 | Schur補完法 or Uzawa法、36+ペア同時NR収束の根本解決 |

### 中優先（フェーズ進行に応じて対応）

| # | TODO | 出典 | 備考 |
|---|------|------|------|
| 7 | pen_ratio改善（adaptive omega） | status-065 | AL乗数の段階的蓄積で改善 |
| 8 | 7本撚りサイクリック荷重 | status-065 | ヒステリシス観測の7本版 |
| 9 | 接触プリスクリーニングGNN Step 1 | status-068/069 | データ生成パイプライン、撚線フェーズ次まで**ペンディング** |
| 10 | k_pen推定MLモデル Step 1 | status-068/069 | グリッドサーチデータ生成、撚線フェーズ次まで**ペンディング** |
| 11 | PINN学習スパース行列対応 | status-069 | 大規模メッシュ（441ノード以上）高速化 |
| 12 | ハイブリッドGNN + PINN 組み合わせ検証 | status-069 | |

### 低優先（必要に応じて対応）

| # | TODO | 出典 | 備考 |
|---|------|------|------|
| 13 | Q4 TL/UL → NonlinearElementProtocol 適合化 | status-060 | 現在は関数ベース |
| 14 | CosseratRod → DynamicElementProtocol 適合化 | status-060 | mass_matrix() 追加 |
| 15 | ContactProtocol 策定 | status-060 | 接触力・接触接線剛性の標準I/F |
| 16 | SectionProtocol 策定 | status-060 | 梁断面の共通I/F（A, Iy, Iz, J） |
| 17 | Phase 4.3: von Mises 3D テスト解凍 | status-029 | 45テスト計画済み（status-025） |
| 18 | Mortar離散化 | status-065 | 接触界面の適合離散化、高k_penでの安定性 |
| 19 | Phase 6.1-6.3: NN構成則・PI制約・ハイブリッド | roadmap | |
| 20 | Phase 7: モデルレジストリ・パラメータフィッティング | roadmap | |

### 完了済み（前回TODOから消化されたもの）

- ~~HEX8 アセンブリ統合~~ → status-062 で実装
- ~~撚撚線統合解析テスト~~ → status-063 で実装
- ~~Stage S2 Fourier近似~~ → status-064 で実装
- ~~Physics-Informed ロス（PINN）~~ → status-068 で実装
- ~~不規則メッシュGNN再検証~~ → status-069 で実装
- ~~7本撚り接触NR収束~~ → status-065 で達成
- ~~GitHub Actions CI~~ → status-070 で構成
- ~~slowテストマーカー~~ → status-070 で導入

---

## 懸念事項・知見

1. **roadmapの肥大化防止**: 完了済みPhaseの詳細は `docs/archive/` に退避する運用を今後も継続
2. **status TODOの形骸化**: 古いstatusのTODOが放置されがち。定期的な棚卸し（今回のような）が有効
3. **ドキュメントの重複**: README/roadmap/CLAUDE.md で「現在の状態」が3箇所に分散。roadmapを正とし、README/CLAUDEはフェーズ一覧表で簡潔に参照するのが良い

## 運用改善提案

- roadmap の「TODO一覧」セクションを正の TODO 管理場所とし、個別statusのTODOはそこへの差分として機能させる
- 新しいstatusでTODOを追加した場合、roadmapのTODO一覧にも反映する
