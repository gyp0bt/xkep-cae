# xkep-cae ロードマップ

[← README](../README.md)

## プロジェクトビジョン

汎用FEMソフトでは解けないニッチドメイン問題を解くための自作有限要素ソルバー基盤。
構成則・要素・ソルバー・積分スキーマ・接触・非線形をモジュール化し、
問題特化ソルバーを構成するフレームワーク。

> **ターゲット: 1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了する。**

---

## 現在地（2026-03-11）

**2477テスト** | S3フェーズ + R1 Phase 5進行中 | [最新status](status/status-index.md)

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

## S3: 大規模収束改善 ← 現在地

**目標**: NCPソルバーで91本撚りの曲げ揺動が収束する。

**完了済み: 53項目** → [詳細一覧](status/s3-completed.md)

### アクティブTODO

- [ ] 全テストのmm-ton-MPa移行（~100ファイルの定数変換）
- [ ] 被膜摩擦μ=0.25の収束達成（接触チャタリング対策が必要）
- [ ] 19本→37本のスケールアップ
- [ ] CR梁の摩擦接触不収束の原因調査
- [ ] 37本Layer1+2圧縮の段階的活性化による収束改善確認
- [ ] NCPソルバー版S3ベンチマーク（AL法との計算時間比較）
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

> AL法（`solver_hooks.py`）はレガシー比較用。詳細は [status-098](status/status-098.md)。

---

## 後続フェーズ

### R1: プロセスアーキテクチャリファクタリング（S3並行）

AbstractProcess + Strategy分解によるソルバー契約化。10セッション計画。
- [設計仕様書](../xkep_cae/process/process-architecture.md)（status-150）
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
- 🔄 Phase 7（残り）: バッチプロセス + 1:1テスト + 検証プロセス

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

> 詳細: [status-index](status/status-index.md) | 各モジュール README: [contact](../xkep_cae/contact/README.md) / [elements](../xkep_cae/elements/README.md) / [mesh](../xkep_cae/mesh/README.md)

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
