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
| `_runtime_uses` | `collect_strategy_types()` + `effective_uses()` | status-165 |
| AL solver (`solver_hooks.py`, `line_search.py`) | 完全削除（NCP一本化） | status-167 |
| `NCPContactSolverProcess` | `NCPDynamic/QuasiStaticContactFrictionProcess` → 完全削除 | status-168 |
| `ContactManager.compute_coating_*()` | `CoatingStrategy.*()` | status-169 |
| `ContactManager.{max_layer,compute_active_layer_for_step,...}` | `staged_activation.*()` / `initial_penetration.*()` | status-170 |
| テスト名 `*_ncp.*` | NCP サフィックス除去 | status-170 |
| `ContactManager` deprecated 10メソッド | 完全削除（呼び出し元直接移行済み） | status-171 |
| `_solve_linear_system()` if分岐 | `LinearSolverStrategy` Protocol + 委譲 | status-171 |
| `NCPQuasiStatic/DynamicContactFrictionProcess` | `ContactFrictionProcess`（統一型） | status-172 |
| `executor.py` 4関数 NotImplementedError | NCP版実装 | status-172 |
| `NCPQuasiStatic/DynamicContactFrictionProcess` | 完全削除（→ ContactFrictionProcess） | status-173 |
| `QuasiStaticFrictionInputData` / `DynamicFrictionInputData` | 完全削除（→ ContactFrictionInputData） | status-173 |
| `contact/solver_smooth_penalty.py` | `process/strategies/{solver_state,newton_uzawa,adaptive_stepping}.py` + Process実体化 | status-174 |
| `xkep_cae/process/` 基盤ファイル | `xkep_cae/core/` | status-178 |
| `process/strategies/penalty/` | `contact/penalty/` | status-178 |
| `xkep_cae_deprecated/contact/law_friction.py` | `contact/friction/law_friction.py` | status-178 |
| `xkep_cae_deprecated/process/strategies/friction.py` | `contact/friction/strategy.py` | status-178 |
| `xkep_cae/core/time_integration/` | `xkep_cae/time_integration/` | status-182 |
| `xkep_cae/process/`（re-export shim） | 完全削除（core 直接参照） | status-182 |
| C16: `core/strategies/` + `contact/` のみ | C16: core/ 以外の全モジュール | status-182 |
| deprecated `ContactFrictionProcess` | `xkep_cae/contact/solver/process.py` | status-184 |
| `StrandBendingBatchProcess` v2.0.0 | v3.0.0（Solver 統合） | status-184 |
| `StrandBendingBatchProcess` v3.0.0 | v4.0.0（Export/Render/Verify 連携） | status-185 |
| `output/__init__.py` 全量 re-export | 明示的エクスポート + `__getattr__` 遅延ロード | status-185 |
| `contact/solver/process.py` importlib 9箇所 | プライベートモジュール群 + `_create_working_strategies` | status-186 |
| C14: direct import のみ検出 | importlib.import_module() も検出 | status-186 |
| `xkep_cae_deprecated.mesh.twisted_wire` importlib | `xkep_cae.mesh._twisted_wire` 直接 import | status-187 |
| `output/__init__.py __getattr__` deprecated lazy-load | 完全削除（使用箇所ゼロ） | status-187 |

## 推奨ソルバー構成

- `newton_raphson_contact_ncp`（solver_ncp.py）
- UL+NCP統合: `ul_assembler` + `adaptive_timestepping=True`
- 解析的接線剛性: `analytical_tangent=True`（デフォルト）
- Line-to-line Gauss積分 + 同層除外 + Fischer-Burmeister NCP
- **摩擦あり**: `contact_mode="smooth_penalty"`（必須。NCP鞍点系は摩擦接線剛性符号問題で発散: status-147）

## 現在の状態

**~2260テスト + 284 新パッケージテスト** — 2026-03-16 | C14 違反 **2件**（status-187）

### ターゲット

> **1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了する。**

### 次の課題

**脱出ポット計画 Phase 7 後半〜8** — ContactManager 新パッケージ移植 + friction/geometry stub 解消。

契約違反 **2件**（status-187: mesh/output C14 除去済み。残りは contact/setup + contact/solver — ContactManager 移植が前提）。詳細は `docs/roadmap.md` および `docs/status/status-index.md` を参照。

## フォーカスガード（AI セッション向け）

**以下を厳守すること。違反は作業のやり直しになる。**

### やるべきこと
- **Process Architecture 完全移行**: テスト名正規化（_ncp除去）、NCP直接呼出→Process API移行
- **S3 凍結解除**: 変位制御7本撚線曲げ揺動のPhase2 xfail解消
- `scripts/validate_process_contracts.py` のエラーをゼロに**維持**する
- コンテキスト整理（ドキュメント構造の明確化）

### やってはいけないこと
- ソルバー性能改善（スパース最適化、並列化、メモリ削減）
- NCP ソルバー（solver_ncp.py）の収束ロジック変更

### セッション開始時の確認手順
1. `docs/status/status-index.md` → 最新 status 番号を確認
2. 最新 `docs/status/status-{N}.md` を読む
3. `python scripts/validate_process_contracts.py` を実行し、エラー一覧を確認
4. 上の「やるべきこと」に合致する作業のみ実施
