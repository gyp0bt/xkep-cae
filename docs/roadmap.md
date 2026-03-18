# xkep-cae ロードマップ

[← README](../README.md)

## プロジェクトビジョン

汎用FEMソフトでは解けないニッチドメイン問題を解くための自作有限要素ソルバー基盤。
構成則・要素・ソルバー・積分スキーマ・接触・非線形をモジュール化し、
問題特化ソルバーを構成するフレームワーク。

> **ターゲット: 1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了する。**

---

## 現在地（2026-03-18）

**~2260テスト + 374 新パッケージテスト** | Phase 16 完了 — C16 違反ゼロ / C17 違反ゼロ（3件→0件 frozen 化完了） | [最新status](status/status-index.md)

| 到達点 | 概要 |
|--------|------|
| FEM基盤 | 梁（EB/Timo/CR/Cosserat）+ 平面 + HEX8、非線形、動的解析 — 完了 |
| 接触 | NCP + Line contact + Mortar + smooth penalty Coulomb摩擦 — 完了 |
| 撚線 | 7本摩擦曲げ+揺動収束、被膜+シース、ヒステリシス — 完了 |
| 高速化 | NCP 6x + 要素12.6x バッチ化、ソルバー一本化 — 完了 |
| **壁** | **61本以上の曲げ揺動収束** / 1000本6時間 / broadphase最適化 |

---

## フェーズ依存関係

```
Phase 1-5, C0-C6, S1-S2 ← 完了（status-001〜096）
  ↓
S3 (大規模収束) ← 現在地（status-097〜）
  ↓ + R1 (プロセスアーキテクチャ) ← S3並行
S4 (剛性比較) ← S3並行可
  ↓
S5 (ML導入) → 候補ペア削減でS6の前提条件
  ↓
S6 (1000本6時間) ← ターゲットマイルストーン
  ↓
S7 (GPU)
```

---

## S3: 大規模収束改善 ← 凍結中（R1 Phase 7 完了まで）

**目標**: NCPソルバーで91本撚りの曲げ揺動が収束する。

**完了済み: 53項目** → [詳細一覧](status/archive/s3-completed.md)

### アクティブTODO

- [ ] 全テストのmm-ton-MPa移行（~100ファイルの定数変換）
- [ ] 被膜摩擦μ=0.25の収束達成（接触チャタリング対策が必要）
- [ ] 19本→37本のスケールアップ
- [ ] CR梁の摩擦接触不収束の原因調査
- [ ] 37本Layer1+2圧縮の段階的活性化による収束改善確認
- [x] ~~NCPソルバー版S3ベンチマーク（AL法との計算時間比較）~~ — AL完全削除済み（status-167）
- [ ] Cosserat Rodの解析的接線剛性実装

### 既知の問題

- **NCP摩擦接線剛性符号問題**: `d(f_fric)/du = -k_t*g_t⊗g_t`（負定値）で鞍点系が不安定化。smooth penalty+Uzawaで回避中。Alart-Curnier拡大鞍点系で根本解決予定。（status-147）
- **slow テスト不収束**: NCP 7本90°曲げ Phase1 が不安定（環境依存）。xfail で安定化済み。

### 計測済みスケーリングデータ

| 素線数 | DOF | 計算時間 | 対7本比 | 候補ペア |
|---:|---:|---:|---:|---:|
| 7 | 210 | 92s | 1.0x | — |
| 19 | 570 | 239s | 2.6x | — |
| 37 | 1,110 | 501s | 5.4x | — |
| 61 | 1,830 | 903s | 9.8x | — |
| 91 | 2,730 | 1,476s | 16.0x | 66,066 |
| 1000 | 30,000 | — | — | 7,335,879 |

---

## 推奨ソルバー構成

| 項目 | 設定 | 根拠 |
|------|------|------|
| **ソルバー** | `newton_raphson_contact_ncp`（`solver_ncp.py`） | Outer loop 不要 |
| **摩擦** | `contact_mode="smooth_penalty"` | NCP鞍点系は符号問題あり（status-147） |
| **接触離散化** | Line-to-line Gauss 積分 | セグメント間力の連続性 |
| **同層除外** | `exclude_same_layer=True` | ~80% ペア削減 |
| **k_pen** | 自動推定（beam EI ベース） | 手動設定不要 |
| **線形ソルバー** | DOF閾値自動切替（直接法 / GMRES+ILU） | スケーラビリティ |

> AL法（`solver_hooks.py`）は status-167 で完全削除済み。NCP一本化。

---

## 後続フェーズ

### R1: プロセスアーキテクチャリファクタリング（S3並行）

AbstractProcess + Strategy分解によるソルバー契約化。10セッション計画。
- [設計仕様書](../__xkep_cae_deprecated/process/docs/process-architecture.md)（status-150）
- ✅ Phase 1: 基盤 + Strategy Protocol（status-151、39テスト）
- ✅ Phase 2: Strategy具象実装 13クラス（status-153、100テスト）
- ✅ Phase 3: Strategy 実ロジック移植 + ファクトリ関数（status-154、+35テスト）
  - U1判断: Process維持（オーバーヘッド 0.8μs/call）
  - 4軸のファクトリ関数: create_{penalty,time_integration,friction,contact_force}_strategy()
- ✅ Phase 4: ContactGeometry 移植 + ファクトリ関数（status-155、+26テスト）
  - Protocol 拡張: update_geometry + build_constraint_jacobian
  - 5軸のファクトリ関数完備: create_contact_geometry_strategy() 追加
- ✅ Phase 5: solver_ncp.py Strategy 注入 + 統合（status-159: 5軸注入完了）
  - ContactForce Strategy: PtP ケース委譲、Mortar/L2L は今後
  - ContactGeometry Strategy: build_constraint_jacobian 委譲
  - default_strategies() で全5軸生成
- ✅ Phase 6: concrete/ 具象プロセス（status-159）
  - PreProcess: StrandMeshProcess, ContactSetupProcess
  - PostProcess: ExportProcess, BeamRenderProcess
- ✅ **Phase 7（完了: status-162）**: BatchProcess + VerifyProcess + 1:1テスト + C3-C12契約違反検知 → 0件
  - ConvergenceVerify/EnergyBalanceVerify/ContactVerify の3 VerifyProcess
  - StrandBendingBatchProcess（ワークフローオーケストレーション）
  - ProcessMeta に stability/support_tier 追加（断片H）
  - execute() チェックサム検証（C9）
- ✅ **Phase 8（完了: status-164）**: ProcessRunner / StrategySlot / CompatibilityProcess / Preset
  - [設計文書](../__xkep_cae_deprecated/process/docs/phase8-design.md)
  - 8-A: ProcessRunner + ExecutionContext（実行管理一元化）— 13テスト
  - 8-B: StrategySlot ディスクリプタ（型安全な Strategy 宣言）— 12テスト
  - 8-C: CompatibilityProcess カテゴリ（deprecated 隔離）— 1テスト
  - 8-D: SolverPreset ファクトリ（検証済み組み合わせ）— 10テスト
  - 8-E: NCPContactSolverProcess StrategySlot 統合 — 3テスト（status-168 で完全削除・新Process移行）
  - 8-F: validate_process_contracts.py C8/C13 更新
- ✅ **Phase 8 完遂（status-165）**: ManualPenaltyProcess CompatibilityProcess 移行 + C13 実効化
- ✅ **Phase 9-A/B（完了: status-165）**: 契約自動化 + StrategySlot 完全移行
  - 9-A: `_import_all_modules()` ファイルシステム走査化（ハードコード廃止）
  - 9-B: `_runtime_uses` 廃止 → `StrategySlot` + `collect_strategy_types()` 完全移行
- ✅ **NCP ソルバーリファクタリング（status-168）**: solver_ncp.py 分離 + Process 移行
  - solver_smooth_penalty.py: Strategy 経由 smooth penalty ソルバー
  - NCPDynamicContactFrictionProcess / NCPQuasiStaticContactFrictionProcess 新設（→ status-173 で完全削除）
  - NCPContactSolverProcess 完全削除
- ✅ **CoatingStrategy 抽出（status-169）**: ContactManager 被膜メソッド Strategy 移行
- ✅ **テスト名正規化 + 純関数化（status-170）**: _ncp 除去 + StagedActivation/InitialPenetration 純関数抽出
- ✅ **deprecated除去+LinearSolverStrategy（status-171）**: 薄ラッパー10メソッド除去 + LinearSolverStrategy Protocol抽出
- ✅ **ContactFrictionProcess統合+executor NCP版（status-172）**:
  - LinearSolverStrategy の _solve_saddle_point_contact 統合
  - NCPQuasiStatic/DynamicContactFrictionProcess → ContactFrictionProcess 統合
  - tuning/executor.py 4関数を NCP 版で再実装
- ✅ **deprecated プロセス完全削除（status-173）**:
  - NCPQuasiStatic/NCPDynamic の全呼び出し元移行 + ファイル完全削除
  - QuasiStaticFrictionInputData/DynamicFrictionInputData 完全削除
  - executor.py 単体テスト3件追加
- **Phase 9-C/D（未実施）**: S3 凍結解除判断 + BatchProcess パイプライン改善
- ✅ **脱出ポット計画 Phase 1（status-175）**: xkep_cae → __xkep_cae_deprecated リネーム + PenaltyStrategy 完全書き直し
  - 新 xkep_cae/ を Process Architecture でゼロ構築
  - C14（deprecated インポート禁止）/ C15（ドキュメント存在検証）契約ルール追加
  - 34テスト（penalty strategy + 法線力物理検証）
- ✅ **脱出ポット計画 Phase 2 前半（status-178）**: process→core移行 + FrictionStrategy 完全書き直し
  - process/ 基盤10ファイルを core/ に移動、penalty/ を contact/ に移動
  - FrictionStrategy 52テスト（return mapping物理検証 + Protocol適合）
  - 契約違反 5→3件に改善
- ✅ **脱出ポット計画 Phase 2 後半（status-179）**: Strategy 全移行 + 契約違反ゼロ
  - ContactForceStrategy / ContactGeometryStrategy / TimeIntegrationStrategy 移行完了
  - 186テスト
- ✅ **C16 契約ギャップ修正（status-180）**: `__init__.py` re-export チェック強化
- ✅ **Penalty/Coating ファクトリ完備（status-181）**: `_create_penalty_strategy` + Coating パッケージ移行
  - ConstantPenalty 新設 + _create_penalty_strategy ファクトリ
  - Coating を deprecated から新パッケージに移行 + _create_coating_strategy ファクトリ
  - default_strategies() で 7軸全 Strategy 生成完備
  - 204テスト
- ✅ **脱出ポット計画 Phase 3（status-183）**: concrete プロセス移行（Mesh/Setup/Export/Render/Verify）
- ✅ **脱出ポット計画 Phase 4（status-184）**: ContactFrictionProcess 移行 + StrandBendingBatchProcess v3.0.0 完全ワークフロー
  - StrandMeshProcess / ContactSetupProcess（PreProcess）
  - ExportProcess / BeamRenderProcess（PostProcess）
  - ConvergenceVerify / EnergyBalanceVerify / ContactVerify（VerifyProcess）
  - StrandBendingBatchProcess v2.0.0 concrete 統合
  - 62テスト追加（合計266テスト）
- ✅ **脱出ポット計画 Phase 5（status-185）**: ソルバー結果連携 + output re-export クリーンアップ
  - StrandBendingBatchProcess v4.0.0: Export/Render/Verify ワイヤリング完成
  - output/__init__.py: 全量 re-export → 明示的エクスポート + __getattr__ 遅延ロード
  - 4テスト追加（合計275テスト）
- ✅ **脱出ポット計画 Phase 6（status-186）**: C14 強化 + ソルバー deprecated 依存除去
  - C14: importlib 経由の deprecated インポートも契約違反に追加
  - C16: プライベートモジュール（_*.py）をスキップルール追加
  - contact/solver プライベートモジュール群新設（7ファイル）
  - C14 違反 13件 → 4件に削減
- ✅ **脱出ポット計画 Phase 7（status-187〜188、完了）**: deprecated 依存除去
  - ✅ mesh/process.py: twisted_wire 移植（C14 除去）
  - ✅ output/__init__.py: __getattr__ deprecated lazy-load 削除（C14 除去）
  - ✅ ContactManager/ContactConfig/ContactState/ContactPair 新パッケージ移植（status-188）
  - ✅ broadphase_aabb 新パッケージ移植（status-188）
  - ✅ contact/setup/process.py + contact/solver/process.py の C14 除去（status-188）
  - ✅ C16 チェッカー: プライベートモジュール除外ルール追加（status-188）
  - ✅ Friction Strategy シグネチャ互換性修正（**kwargs 追加）
  - **C14/C16 違反 0件達成**
- ✅ **脱出ポット計画 Phase 8（status-189、完了）**: C14 抜け道修正 + friction/geometry 実装完成
  - C14 チェッカー: importlib エイリアス検出追加（`import importlib as _il` パターン）
  - 8モジュールの暫定 re-export 除去（deprecated 直接参照に移行）
  - friction evaluate()/tangent(): _assembly.py 経由で完全実装
  - geometry detect(): _detect_candidates() 経由で broadphase 接続
- ✅ **脱出ポット計画 Phase 9（status-190〜192、完了）**: solver Process 化 + NUzawa 分離 + プライベート関数移行
  - solver 純関数5モジュール → Process 化（SolverProcess 継承）
  - NewtonUzawaStaticProcess + NewtonUzawaDynamicProcess 分離
  - Process 内部プライベート関数 → Process API 経由に移行
  - Strategy 公開 setter/property 追加、O1 条例違反検知
- ✅ **脱出ポット計画 Phase 10（status-193、完了）**: deprecated 参照テスト無効化
  - tests/conftest.py 新設（pytest_ignore_collect で未移行テスト自動スキップ）
  - 状態操作ユーティリティ（_state_set 等）は Process 内部として維持判断
- ✅ **脱出ポット計画 Phase 11（status-194、完了）**: `xkep_cae_deprecated` → `__xkep_cae_deprecated` リネーム（C14 実効性強化）
- ✅ **脱出ポット計画 Phase 12（status-195、完了）**: numerical_tests モジュール新 xkep_cae 移植
  - BackendRegistry パターン（依存性注入）で C14 準拠維持
  - 8ファイル・約1400行移植: core/runner/frequency/dynamic/csv/inp/benchmark
  - tests/conftest.py に deprecated 実装注入（static/frequency/dynamic 3段階）
  - 70テスト全PASS
- ✅ **脱出ポット計画 Phase 13（status-197、完了）**: ビームアセンブラ（CR/UL）の新 xkep_cae 移植
  - BeamSection + CR梁関数群 + assemble_cr_beam3d + ULCRBeamAssembler
  - 四元数関数インライン化（C14準拠）、32テストPASS
  - 移植先: `xkep_cae/elements/` プライベートモジュール群
- ✅ **脱出ポット計画 Phase 14（status-198、完了）**: S3 xfail テスト Process API 版作成
  - 11テスト（8 passed + 2 xfailed + 1 xpassed）
  - ULCRBeamAssembler + ContactFrictionProcess 統合テスト
  - geometry/__init__.py 循環参照修正
- ✅ **Process 実行診断インフラ（status-199、完了）**: 警告・エラー・使用レポート
  - ProcessExecutionLog: inspect.stack() で呼び出し元自動検知、atexit レポート出力
  - StaticSolverWarning: 準静的ソルバー使用時の警告
  - NonDefaultStrategyWarning: デフォルト以外の Strategy 構成使用時の警告
  - DeprecatedProcessError: deprecated プロセス実行時のエラー
  - 16テスト追加（合計374テスト）
- **脱出ポット計画 Phase 15 完了**: C16 違反 40→0 件（純粋関数 private 化 + frozen DC 化 + elements 移行 + 検出ルール精緻化）
- **C17 例外リスト廃止 + replace() 検知（status-203）**: 違反は正規報告。replace() をC17違反として検知追加
- ✅ **脱出ポット計画 Phase 16（status-204、完了）**: C17 違反3件→0件（frozen 化 + _evolve パターン）
  - `_ContactStateOutput`/`_ContactPairOutput`/`_ContactManagerInput` 全て frozen=True
  - `_evolve(**kwargs)` メソッドで不変的インスタンス更新パターン導入
  - 41変異箇所を7ファイルにわたり書き換え
  - 315テスト全PASS
- ✅ **ContactManager Process 分割（status-205、完了）**: dataclass メソッド完全除去→Process 直接実装
  - `_ContactManagerInput` の全メソッド除去（純データ化）
  - `_evolve()` / `copy()` / `is_active()` / `search_radius` をモジュールレベル関数に移動
  - DetectCandidates/UpdateGeometry/InitializePenalty: ロジック直接実装（v2.0.0）
  - AddPairProcess / ResetAllPairsProcess 新設
  - 全 Process 出力に `manager` フィールド追加（不変パターン）
  - 呼び出し元の整合は次PR対応
- **Phase 17（次）**: 呼び出し元整合 + BackendRegistry 完全廃止（O2 条例違反2件解消）+ 被膜モデル物理検証テスト

### S4: 撚線構造剛性比較

被膜/シース付き撚線の等価剛性を計測し、文献値（Costello, Foti）と比較。
- ✅ 素線+被膜/シース等価剛性 20テスト（status-098）
- ❌ フルモデル + 文献値比較

### S5: ML導入

接触候補削減と k_pen 推定の自動化。GNN/PINN サロゲート PoC 完了（R²=0.995）。

### S6: 1000本撚線

1000本撚線の曲げ揺動計算を6時間以内。メッシュ生成・broadphaseは実装済み。

### S7: GPU対応

S6のボトルネックに応じたGPU化（CuPy/JAX）。

---

## 完了済みフェーズ

| Phase | 内容 | テスト数 | status |
|-------|------|---------|--------|
| 1 | アーキテクチャ（Protocol/ABC） | 16 | 001-003 |
| 2 | 空間梁要素（EB/Timo/Cosserat） | ~360 | 004-015 |
| 3 | 幾何学的非線形（NR/弧長/CR/TL/UL） | ~100 | 015-042 |
| 4.1-4.2 | 弾塑性 + ファイバーモデル | ~70 | 021-023 |
| 5 | 動的解析 | ~60 | 026-030 |
| C0-C6 | 梁–梁接触（AL→NCP+Mortar+摩擦） | ~320 | 033-086 |
| 4.7 | 撚線基礎 + シース | ~420 | 052-064 |
| HEX8/I/O | 3D固体 + Abaqusパーサー | ~210 | 031-063, 105-106 |
| 6.0 | GNN/PINNサロゲート | ~100 | 066-069 |
| S1-S2 | 同層除外 + CPU並列化基盤 | ~90 | 083-096 |

> 詳細: [status-index](status/status-index.md) | 各モジュール README: [contact](../__xkep_cae_deprecated/contact/docs/README.md) / [elements](../__xkep_cae_deprecated/elements/docs/README.md) / [mesh](../__xkep_cae_deprecated/mesh/docs/README.md)

---

## 凍結・将来計画

| Phase | 内容 | 状態 |
|-------|------|------|
| 4.3 | von Mises 3D 塑性 | 凍結（45件テスト済） |
| 4.4-4.6 | ヒステリシス減衰、粘弾性、異方性 | 未実装 |
| 6.1-6.3 | NN構成則、PI制約、ハイブリッド | 未実装 |
| 7-8 | モデルレジストリ、FE² | 未実装 |

---

## 設計原則

1. **モジュール合成可能性**: 要素・構成則・ソルバー・積分スキーマを自由に組み合わせ
2. **Protocol/ABCベース**: インタフェース依存
3. **テスト駆動**: 解析解・リファレンスソルバーとの比較必須
4. **段階的拡張**: 後方互換性保持

---

## 参考文献

- Crisfield, M.A. "Non-linear Finite Element Analysis of Solids and Structures" Vol. 1 & 2
- Bathe, K.J. "Finite Element Procedures"
- de Souza Neto et al. "Computational Methods for Plasticity"
- Simo, J.C. & Hughes, T.J.R. "Computational Inelasticity"
- Costello, G.A. "Theory of Wire Rope"
- Foti, F. & Martinelli, L. (2016) "Hysteretic bending of spiral strands"
