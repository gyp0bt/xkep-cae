# xkep-cae ロードマップ

[← README](../README.md)

## プロジェクトビジョン

汎用FEMソフトでは解けないニッチドメイン問題を解くための自作有限要素ソルバー基盤。
構成則・要素・ソルバー・積分スキーマ・接触・非線形をモジュール化し、
問題特化ソルバーを構成するフレームワーク。

> **ターゲット: 1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了する。**

---

## 現在地（2026-04-03）

**606 テスト** | 契約違反**0件** | [最新status](status/status-index.md)

| 到達点 | 概要 |
|--------|------|
| FEM基盤 | 梁（EB/Timo/CR/Cosserat）+ 平面 + HEX8、非線形、動的解析 — 完了 |
| 接触 | NCP + Line contact + Mortar + smooth penalty Coulomb摩擦 — 完了 |
| 撚線 | 7本摩擦曲げ+揺動収束、被膜+シース、ヒステリシス — 完了 |
| 高速化 | NCP 6x + 要素12.6x バッチ化 + 接触アセンブリ12-16x バッチ化（大規模向け） — status-246 |
| NR安定化 | **NR接触チャタリング対策**: 接触力リラクゼーション + 接線スケーリング → n_periods=30 frac=1.0 完走 — status-247 |
| MPC | **DOF消去MPC剛体結合**: T^T K T 変換 + LinearSolveProcess統合 + 端部参照点 — status-253 |
| MPC収束 | **MPC u伝搬修正 + NR内再射影 + 縮退系残差判定**: frac=0.35到達（接線不整合で壁） — status-255 |
| FD診断 | **TangentFDDiagnosticProcess**: MPC+接触の接線方向有効性をFDで検証 — status-256 |
| K_c特定 | **FD診断でK_c不整合を決定的に特定**: 全体系94-100%誤差、MPC変換は原因でない — status-257 |
| K_c再解析 | **K_c不整合は活性集合変化が原因**: consistent_st_tangent=TrueでK_c自体は完全一致(4.4e-10) — status-258 |
| smoothing_delta | **Huber smoothing_deltaパイプライン貫通**: ContactSetupConfig→HuberProcess全経路で設定可能、自動推定(5000/r)有効化 — status-259 |
| δチューニング | **smoothing_delta最適化**: 1000/rに変更でfrac 0.35→0.59（69%改善）、δ=1000手動指定でfrac=1.0完走達成。FD診断に活性DOFフィルタ追加 — status-260 |
| delta_h API | **huber_delta_h直接指定API**: k_penスケール非依存の遷移幅指定。active_contact_dofs NRソルバー結合。δ=1000完走テスト追加 — status-261 |
| delta_h探索 | **delta_h=0.025が最速完走（132s）**: 梁-梁で有効範囲[0.020,0.025]∪{0.040}、非単調性あり。three_point_bend貫通。3Dパイプ貫入なし — status-262 |
| delta_h検討 | **delta_hデフォルト値は0.0維持**: 剛体-梁では直接指定改善なし、問題依存性高くグローバルデフォルト時期尚早 — status-263 |
| E=25回帰修正 | **frozen_hermite_tangent + _cur_ratio統一 + n_elems=8**: E=25 frac=0.0003→0.67回復 — status-264 |
| STA2自動記録 | **BenchmarkRunnerProcess**: 実行マニフェスト自動記録（git+Config+結果）で担当者間再現性を保証 — status-265 |
| frozen_hm安定化 | **frozen_hermite_tangent=False安定化**: tangent()はdm凍結、evaluate()のみdm補正（修正NR法）。E=25 frac=0.0003→0.47 — status-266 |
| chattering分析 | **チャタリング詳細分析 + divergedフラグ修正**: リラクゼーション91/91全失敗の原因特定。abort時diverged=False化でE=25 frac 0.4837→0.4950 — status-267 |
| delta_hブースト | **チャタリング時delta_hブースト + NR反復動的拡張**: ボトルネック確定=frozen_hermite_tangent線形収束率(0.97/iter)。delta_hは深い貫入に無効。frac 0.4950→0.4978 — status-268 |
| NRリストア | **NR残差最小値リストア**: 発散検知時に最小残差状態にロールバック。過修正防止でfrozen=True 0.4978→0.5341、frozen=False 0.4732→0.5408 — status-269 |
| frac1回帰修正 | **n_elems_wire=20復元**: パラメータbisectで主因特定（n_elems 20→8が唯一の原因、use_rigid_surface無影響）。frac進行率9x改善 — status-270 |
| frozen=False検証 | **frozen_hermite_tangent=False + n_elems=20**: frac=1.0, incr=607, cutback=389。frozen=True比**35%高速、43%カットバック減** — status-271 |
| Hermite非局所Step1 | **StJacobian隣接ノード微分**: ds_du_adj/dt_du_adj計算実装。FD検証atol=1e-5合格（2テスト追加） — status-271 |
| Hermite非局所Step2 | **K_st隣接ノードDOF拡張**: adj_node_map計算、ds_du_adj/dt_du_adjをK_stアセンブリに結合。FD検証atol=1e-4合格（2テスト追加） — status-272 |
| Hermite非局所Step3 | **K_c隣接ノードDOF拡張**: K_mat+K_geoにalpha_adjベースの隣接ノード寄与追加。FD検証atol=1e-2合格（2テスト追加） — status-273 |
| 摩擦K_st非局所Step4 | **摩擦K_st隣接ノードDOF拡張**: _assemble_friction_st_stiffnessにHermite非局所寄与追加。ソルバーパイプライン貫通。3テスト追加 — status-274 |
| テスト品質改善 | **非平行座標化 + atol厳格化**: TestKstNonlocalFD/TestKcAdjFDのtrivially passing問題修正。asymmetric atol 1e-2→1e-5 — status-275 |
| NR壁根本原因 | **evaluate/tangent dm不整合を特定**: 複合回帰（NR+接触コード相互作用）、NR min restore OFF + diverged=True + tangent scaling復元 — status-277 |
| チェックポイント再開 | **チェックポイント途中再開パイプライン実装**: load_frac_start + stepping/state初期化。N-サイクル検知/リスタートは逆効果で無効化。ul_frac_base処方変位バグ修正。ベースライン frac=0.5543 維持 — status-279 |
| free_end_mode | **MPC不使用端部直接処方**: 各素線θ_z直接処方。MPC frac=0.55→free_end frac=1.0完走 — status-280 |
| UL参照配置更新 | **接触なし90度曲げ完走**: ContactFrictionProcess UL増分変位+update_reference()。理論値0.02%一致 — status-281 |
| 接触ありベースライン | **接触あり90度曲げfrac=0.40**: active=8-9で2サイクルチャタリング停滞。接触なし比で60%低下 — status-282 |
| MPC T動的再構築 | **MPC接触なし90度曲げ完走**: T行列をUL更新時に変形座標で再構築。frac 0.14→1.0 — status-283 |
| **壁** | 接触チャタリング対策 / 61本以上 / 1000本6時間 |

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

**完了済み: 53項目**

### アクティブTODO

- [ ] 全テストのmm-ton-MPa移行（~100ファイルの定数変換）
- [ ] 被膜摩擦μ=0.25の収束達成（接触チャタリング対策が必要）
- [ ] 19本→37本のスケールアップ
- [ ] CR梁の摩擦接触不収束の原因調査
- [ ] 37本Layer1+2圧縮の段階的活性化による収束改善確認
- [x] ~~NCPソルバー版S3ベンチマーク（AL法との計算時間比較）~~ — AL完全削除済み（status-167）
- [ ] Cosserat Rodの解析的接線剛性実装
- [x] ~~**UL+GeneralizedAlpha結合修正**: state.u増分/累積管理の明確化~~ — status-215 で修正完了（動的時UL更新スキップ + モーダル質量補正）
- [x] ~~**動的三点曲げ解析解一致**: FFT振動周期5%以内+振幅10%以内~~ — status-217 で達成
- [x] ~~**UnifiedTimeStepProcess統合**: ContactFrictionProcess内のdt_sub二重管理解消~~ — status-217 で統合完了
- [x] ~~**数値粘性の定量評価**: rho_inf 依存性の検証~~ — status-217 でパラメータ感度81.5%確認
- [x] ~~**動的三点曲げ接触収束**: increment カウント修正で frac=1.0 到達（Hermite OFF 202N, Hermite ON 176N）~~ — status-231 で修正完了（旧 frac=0.86/0.98 はカットバックがmax_increments予算を食い潰すバグが原因）
- [x] ~~**n_periods=30 での数百 N 確認**: n_periods=30 frac=1.0 到達、208.6N（status-234）~~
- [x] ~~**摩擦アセンブリの Hermite 完全対応**: use_hermite=True デフォルト化（status-245）~~
- [ ] **freeze_geometry_in_nr の見直し**: NR内 s,t 凍結は K_st と相互排他（status-239）。freeze=True は修正Newton相当で力2次収束不可。freeze=False + K_st + LM が正しい組合せ
- [ ] **Node tangent 計算の局所化**: 現在 `_compute_node_tangents()` が全体メッシュ依存で、大変形時に接線急変→Hermite曲線形状ジャンプ→gap不連続→active接触点激減。隣接要素のみで局所計算に変更
- [ ] **曲面連続関数化の代替手法調査**: Hermite 補間は接線感度が高すぎて大規模モデルで破綻。代替候補: B-spline/NURBS 曲線表現、Subdivision surface、Moving least squares (MLS) 近似、Isogeometric 接触（IGA-C）など。接線ベクトルへの過敏性を回避しつつ C1 連続性を確保する手法を比較検討
- [x] **解析的剛体円柱表面**: ジグ離散セグメント→C∞連続な解析的円柱（status-237）
- [x] **梁メッシュ粗化**: L_elem > wire_diameter で梁サーフェス面連続化（status-237: n_elems 20→4）
- [x] **SDI 排除**: 全候補ペア Huber 評価 + 力ベース dt 制御 + g_off ワイド化（status-233）
- [x] ~~**n_periods=30 で frac=1.0 到達**: SDI 排除後の dt 改善検証（status-234: 1592 incr, 4403s, fc=208.6N）~~
- [x] ~~**adaptive stepping 高速化**: dt_max緩和+growth damping撤廃+接触力閾値緩和~~ — status-236 で **完全リバート**（n_periods=30 で逆効果: frac=0.24 で壁、98%カットバック）。パラメータ調整だけでは NR 収束性問題を解決できない
- [ ] **NR力収束改善**: λ自動推定のc=20はアルミで悪化、dof_scale_rot<1.0は全値で悪化（status-242）。Hermite K_st の 33% FD不整合は凍結接線近似(frozen-m)が原因。次: frozen-m解消（非局所DOF結合が必要）
- [x] ~~**n_periods=30 剛体表面効果検証**: incr 1592→707（55%削減）、cutback 2477→400（84%削減）、frac=1.0 fc=216.96N（status-238）~~

### 既知の問題

- **NCP摩擦接線剛性符号問題**: `d(f_fric)/du = -k_t*g_t⊗g_t`（負定値）で鞍点系が不安定化。smooth penalty（Uzawa凍結、n_uzawa_max=1）で回避中。（status-147, status-221）
- ~~**slow テスト不収束**: NCP 7本90°曲げ Phase1 が不安定（環境依存）。xfail で安定化済み。~~ → status-212 で接触収束テスト全削除。

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
| **同素線除外** | `exclude_same_strand=True` | ~80% ペア削減 |
| **k_pen** | 自動推定（beam EI ベース） | 手動設定不要 |
| **線形ソルバー** | DOF閾値自動切替（直接法 / GMRES+ILU） | スケーラビリティ |

> AL法（`solver_hooks.py`）は status-167 で完全削除済み。NCP一本化。

---

## 後続フェーズ

### R1: プロセスアーキテクチャリファクタリング（完了）

AbstractProcess + Strategy分解によるソルバー契約化。status-150〜174 で完了。

### 脱出ポット計画（Phase 1〜16 完了）

新 xkep_cae を Process Architecture でゼロ構築。旧パッケージからの完全移行。

| Phase | status | 概要 |
|-------|--------|------|
| 1 | 175 | xkep_cae → __xkep_cae_deprecated リネーム + PenaltyStrategy |
| 2 | 178-181 | Strategy 全移行（Friction/ContactForce/Geometry/TimeIntegration/Coating）|
| 3-4 | 183-184 | concrete プロセス + ContactFrictionProcess + BatchProcess |
| 5-6 | 185-186 | ソルバー結果連携 + C14 強化 |
| 7-8 | 187-189 | deprecated 依存完全除去 + friction/geometry 実装完成 |
| 9 | 190-192 | solver Process 化 + NUzawa 分離 |
| 10-11 | 193-194 | deprecated テスト無効化 + __deprecated リネーム |
| 12-13 | 195-197 | numerical_tests + ビームアセンブラ移植 |
| 14 | 198 | S3 xfail テスト Process API 版 |
| 15-16 | 200-206 | C16/C17 違反ゼロ + frozen dataclass + 旧テスト一掃 |

| 17 | 208 | BackendRegistry 完全廃止 + 被膜モデル物理検証テスト（O2 条例違反0件） |

- **次**: S3 凍結解除（変位制御7本撚線曲げ揺動 Phase2 xfail 解消）

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

> 詳細: [status-index](status/status-index.md)

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
