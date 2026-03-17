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
| `__xkep_cae_deprecated/contact/law_friction.py` | `contact/friction/law_friction.py` | status-178 |
| `__xkep_cae_deprecated/process/strategies/friction.py` | `contact/friction/strategy.py` | status-178 |
| `xkep_cae/core/time_integration/` | `xkep_cae/time_integration/` | status-182 |
| `xkep_cae/process/`（re-export shim） | 完全削除（core 直接参照） | status-182 |
| C16: `core/strategies/` + `contact/` のみ | C16: core/ 以外の全モジュール | status-182 |
| deprecated `ContactFrictionProcess` | `xkep_cae/contact/solver/process.py` | status-184 |
| `StrandBendingBatchProcess` v2.0.0 | v3.0.0（Solver 統合） | status-184 |
| `StrandBendingBatchProcess` v3.0.0 | v4.0.0（Export/Render/Verify 連携） | status-185 |
| `output/__init__.py` 全量 re-export | 明示的エクスポート + `__getattr__` 遅延ロード | status-185 |
| `contact/solver/process.py` importlib 9箇所 | プライベートモジュール群 + `_create_working_strategies` | status-186 |
| C14: direct import のみ検出 | importlib.import_module() も検出 | status-186 |
| `__xkep_cae_deprecated.mesh.twisted_wire` importlib | `xkep_cae.mesh._twisted_wire` 直接 import | status-187 |
| `output/__init__.py __getattr__` deprecated lazy-load | 完全削除（使用箇所ゼロ） | status-187 |
| `__xkep_cae_deprecated.contact.pair.ContactManager` | `xkep_cae.contact._contact_pair._ContactManager` | status-188 |
| `__xkep_cae_deprecated.contact.broadphase.broadphase_aabb` | `xkep_cae.contact._broadphase._broadphase_aabb` | status-188 |
| `solver/process.py _create_working_strategies()` | `core.data.default_strategies()` 直接使用 | status-188 |
| C14: `importlib` のみ検出 | `importlib` + エイリアス検出 | status-189 |
| 8モジュール暫定 re-export (`_il.import_module`) | 空化（deprecated 直接参照に移行） | status-189 |
| friction evaluate()/tangent() ゼロ返却 stub | `_assembly.py` 経由の完全実装 | status-189 |
| geometry detect() 空リスト stub | `_detect_candidates()` 経由 broadphase 実装 | status-189 |
| solver 純関数5モジュール | Process 化（SolverProcess 継承） | status-190 |
| `NewtonUzawaProcess`（統合型） | `NewtonUzawaStaticProcess` + `NewtonUzawaDynamicProcess` | status-190 |
| `process.py` プライベート関数5種直接呼び出し | Process API 経由 | status-191 |
| `manager.detect_candidates()` 直接呼び出し | `DetectCandidatesProcess` | status-191 |
| `manager.update_geometry()` 直接呼び出し | `UpdateGeometryProcess` | status-191 |
| `NewtonUzawaProcess = StaticProcess` | `NewtonUzawaProcess = DynamicProcess` | status-191 |
| `_deformed_coords()` / `_ncp_line_search()` Process内直接呼び出し | `DeformedCoordsProcess` / `NCPLineSearchProcess` API 経由 | status-192 |
| Strategy `_k_pen`/`_k_t_ratio`/`_ndof` 直接代入 | `set_k_pen()`/`set_k_t_ratio()`/`set_ndof()` 公開 API | status-192 |
| `xkep_cae_deprecated/` ディレクトリ名 | `__xkep_cae_deprecated/`（C14 実効性強化） | status-194 |
| `__xkep_cae_deprecated.numerical_tests` 直接参照 | `xkep_cae.numerical_tests` + BackendRegistry DI | status-195 |
| `__xkep_cae_deprecated.elements.beam_timo3d` ULCRBeamAssembler 等 | `xkep_cae.elements._beam_*` モジュール群 | status-197 |
| `__xkep_cae_deprecated.sections.beam` BeamSection/2D | `xkep_cae.elements._beam_section` | status-197 |
| `__xkep_cae_deprecated.math.quaternion` (rotvec↔rotmat用) | `xkep_cae.elements._beam_cr` にインライン化 | status-197 |
| `numerical_tests/__init__.py` 全量 re-export | 型のみ re-export（関数除去） | status-200 |
| C16: `__init__.py` 関数 re-export のみ検査 | クラス re-export も型検査対象 | status-200 |
| 条例 O1 のみ | O2（BackendRegistry 検出）+ O3（backend.configure 検出）追加 | status-200 |
| C16 のみ（プライベートモジュールスキップ） | C17 追加（プライベートモジュール dataclass 衛生） | status-202 |
| `_ContactConfig` non-frozen | `_ContactConfigInput` frozen=True | status-202 |
| `BeamForces3D`/`BeamSection`/etc. 命名不統一 | `*Input`/`*Output` 命名規約準拠（82ファイル） | status-202 |

## 推奨ソルバー構成

- `newton_raphson_contact_ncp`（solver_ncp.py）
- UL+NCP統合: `ul_assembler` + `adaptive_timestepping=True`
- 解析的接線剛性: `analytical_tangent=True`（デフォルト）
- Line-to-line Gauss積分 + 同層除外 + Fischer-Burmeister NCP
- **摩擦あり**: `contact_mode="smooth_penalty"`（必須。NCP鞍点系は摩擦接線剛性符号問題で発散: status-147）

## 現在の状態

**~2260テスト + 374 新パッケージテスト** — 2026-03-17 | C16 違反 **0件** | C17 違反 **0件**（既知例外3件） | O2/O3 条例違反 5件（警告, status-202）

### ターゲット

> **1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了する。**

### 次の課題

**脱出ポット計画 Phase 16** — BackendRegistry 完全廃止（O2/O3 条例違反5件解消）+ 被膜モデル物理検証テスト + C17 non-frozen例外3件解消（ContactManager Process分割）（status-202）。

詳細は `docs/roadmap.md` および `docs/status/status-index.md` を参照。

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
