# status-156: Process document_path バリデーション追加

[← status-index](status-index.md) | [← README](../../README.md)

**日付**: 2026-03-11
**テスト数**: 2474（+3）

## 概要

status-155（Phase 4 完了）のTODOに基づき、全具象Processクラスに `document_path` クラス変数をハードコードし、
対応するドキュメントファイルが存在しない場合にクラス定義時点で `FileNotFoundError` を発生させるバリデーションを追加した。

## 成果物

### AbstractProcess 基底クラス拡張

| 変更 | 内容 |
|------|------|
| `document_path: ClassVar[str]` 追加 | ソースファイルからの相対パスでドキュメントを指定 |
| `__init_subclass__` 検証追加 | (1) `document_path` 未定義 → `TypeError` (2) ファイル不在 → `FileNotFoundError` |
| `document_markdown()` 拡張 | 設計文書パスを出力に含める |

### Strategy 設計文書作成（5ファイル）

| ファイル | 対象クラス |
|---------|-----------|
| `xkep_cae/process/strategies/docs/penalty.md` | AutoBeamEI, AutoEAL, ManualPenalty, ContinuationPenalty |
| `xkep_cae/process/strategies/docs/contact_force.md` | NCPContactForce, SmoothPenaltyContactForce |
| `xkep_cae/process/strategies/docs/friction.md` | NoFriction, CoulombReturnMapping, SmoothPenaltyFriction |
| `xkep_cae/process/strategies/docs/time_integration.md` | QuasiStatic, GeneralizedAlpha |
| `xkep_cae/process/strategies/docs/contact_geometry.md` | PointToPoint, LineToLineGauss, MortarSegment |

### テスト用ドキュメント

| ファイル | 用途 |
|---------|------|
| `xkep_cae/process/tests/docs/dummy.md` | テスト用Stubクラスのdocument_path検証用 |
| `scripts/docs/benchmark.md` | ベンチマークスクリプト内Dummyクラス用 |

### document_path 追加対象（全14具象クラス + テスト用Stub）

**Strategy具象クラス（14クラス）**:
- penalty.py: 4クラス → `docs/penalty.md`
- contact_force.py: 2クラス → `docs/contact_force.md`
- friction.py: 3クラス → `docs/friction.md`
- time_integration.py: 2クラス → `docs/time_integration.md`
- contact_geometry.py: 3クラス → `docs/contact_geometry.md`

**テスト用Stub（11クラス）**: 各テストファイル内のDummyProcess/Tree*Process等

### テスト追加（+3テスト）

| テストファイル | 追加テスト数 | 内容 |
|--------------|------------|------|
| `test_base.py` | +3 | document_path未定義検出、ファイル不在検出、Markdown出力確認 |

## 設計判断

| 判断 | 選択 | 理由 |
|------|------|------|
| 検証タイミング | `__init_subclass__` （クラス定義時） | ランタイムで失敗するよりモジュールimport時に即座に検出 |
| パス解決 | `inspect.getfile(cls)` + 相対パス | コロケーション方式と整合、テストファイル内クラスも正しく解決 |
| ドキュメント粒度 | Strategy カテゴリ単位（1ファイル/カテゴリ） | 同一モジュール内の具象クラスが同じ設計文書を参照 |

## CI 状況

### ローカルテスト: 198件全通過（process 28件 + strategy 170件）

## TODO（次セッション）

- [ ] Phase 5: solver_ncp.py の newton_raphson_contact_ncp() を Strategy 注入に書き換え（status-155継承）
- [ ] Mortar の完全制約ヤコビアン（status-155継承）
- [ ] Friction Strategy のCoulomb/SmoothPenalty 共通ロジック抽出（status-155継承）
- [ ] slow-test 3件の収束問題対応（status-155継承）
- [ ] 全テストのmm-ton-MPa移行（status-149継承）
- [ ] 被膜摩擦μ=0.25の収束達成（status-149継承）
- [ ] 19本→37本のスケールアップ（status-149継承）
- [ ] GitHub Actions CI結果確認（ghコマンド未接続のためスキップ）

## 運用メモ

- ghコマンドがローカルプロキシ環境で使用不可。CI結果はpush後にWebで確認。
