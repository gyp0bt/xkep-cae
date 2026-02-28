# xkep-cae ロードマップ

[← README](../README.md)

## プロジェクトビジョン

汎用FEMソフトでは解けないニッチドメイン問題を解くための自作有限要素ソルバー基盤。
支配方程式・構成則・要素・更新則・積分スキーマ・接触・非線形をモジュール化し、
組み合わせて問題特化ソルバーを構成できるフレームワークを目指す。

### 第一ターゲット：空間梁モデル

撚線電線・単線・ロッド等の「線」の曲げ・引張・ねじり・揺動試験を再現する。
試験荷重および運動経路を再現するモデルレジストリと、パラメータフィッティングアルゴリズムを備える。

---

## 現在地

**Phase 1〜3 + Phase 4.1〜4.2 + Phase 5.1〜5.4 + Phase C0〜C5 + Phase C6-L1〜L4 + Phase 4.7 Level 0 + L0.5 S1-S4 + ブロック前処理ソルバー + adaptive omega + Phase 6.0 PoC + ML基盤 完了。1821テスト（fast: 1448 / slow: 283）。**

### 完了済みフェーズ一覧

| Phase | 内容 | テスト数 |
|-------|------|---------|
| **Phase 1** | アーキテクチャ再構成（Protocol/ABC設計） | 16 |
| **Phase 2** | 空間梁要素（EB/Timo 2D・3D, Cosserat rod, 数値試験F/W） | ~360 |
| **Phase 3** | 幾何学的非線形（NR, 弧長法, CR定式化, TL/UL） | ~100 |
| **Phase 4.1-4.2** | 1D弾塑性 + ファイバーモデル | ~70 |
| **Phase 5** | 動的解析（Newmark-β, HHT-α, 陽解法, モーダル減衰） | ~60 |
| **Phase C0-C5** | 梁–梁接触（AL法, 摩擦, merit LS, 一貫接線, PDAS） | ~190 |
| **Phase 4.7 L0** | 撚線モデル基礎（メッシュF/W, 接触グラフ, 7本撚り収束, ヒステリシス, 被膜, シース） | ~350 |
| **Phase 4.7 L0.5 S1-S2** | シース挙動（リングコンプライアンス, Fourier近似） | ~70 |
| **HEX8** | 3D固体要素ファミリ（C3D8/C3D8B/C3D8R/C3D8I） | ~80 |
| **過渡応答出力** | Step/Increment/Frame + CSV/JSON/VTK + FIELD ANIMATION + GIF | ~80 |
| **I/O** | Abaqus .inp パーサー, テーブル硬化則, KINEMATIC→AF変換 | ~50 |
| **Phase 6.0 PoC** | 2D熱伝導FEM + GNN/PINN サロゲート | ~100 |
| **CI/CD** | GitHub Actions 3ジョブ + slow テストマーカー | — |

> 完了済みPhaseの詳細設計情報は [archive/completed-phases.md](archive/completed-phases.md) を参照。

### 実装済み機能サマリー

#### 要素

| カテゴリ | 実装 |
|---------|------|
| **平面要素** | Q4, TRI3, TRI6, Q4_BBAR, Q4_EAS（デフォルト） |
| **梁要素** | EB2D, Timo2D, Timo3D（12DOF）, CR定式化（幾何学的非線形）, CR梁ファイバー弾塑性 |
| **Cosserat rod** | 四元数回転, B行列, SRI, 初期曲率, 幾何剛性 |
| **3D固体** | C3D8（SRI+B-bar）, C3D8B（B-bar）, C3D8R（低減積分+HG制御）, C3D8I（非適合モード） |

#### 材料・断面

| カテゴリ | 実装 |
|---------|------|
| **弾性** | 2D平面ひずみ, 3D等方（Voigt 6×6）, 1D梁弾性 |
| **塑性** | 1D弾塑性（return mapping, consistent tangent, 等方/移動硬化, AF） |
| **断面** | 矩形, 円形, パイプ（2D/3D）, ファイバーモデル（FiberSection + FiberIntegrator） |

#### ソルバー・解析

| カテゴリ | 実装 |
|---------|------|
| **非線形** | Newton-Raphson, 弧長法（Crisfield）, 荷重増分法 |
| **動的** | Newmark-β, HHT-α, 陽解法（Central Difference）, モーダル減衰 |
| **接触** | Broadphase（AABB格子）, 法線AL, 摩擦（Coulomb return mapping + μランプ）, merit line search, 一貫接線（K_geo）, PDAS, 適応的ペナルティ増大 |
| **直接法** | spsolve, GMRES+ILU（反復）, AMG（pyamg） |

#### 撚線・メッシュ・I/O

| カテゴリ | 実装 |
|---------|------|
| **撚線** | メッシュF/W（3/7/19/37/61/91本）, 接触グラフ, k_pen自動推定, 段階的アクティベーション, ヒステリシス観測・統計 |
| **被膜・シース** | CoatingModel, SheathModel, リングコンプライアンス（S1-S2） |
| **I/O** | Abaqus .inp パーサー, CSV/JSON/VTK出力, FIELD ANIMATION, GIF |
| **熱伝導+ML** | Q4フィン要素, GNNサロゲート（R²=0.995）, PINN, ハイブリッドGNN |

### 未実装（現状の制約）

- ラインサーチ・Lee's frame 等の追加ベンチマーク（Phase 3 オプション）
- 3D von Mises 塑性（実装コード完了、テスト凍結中）
- ~~シース挙動 Stage S3-S4（有限滑り・シース-シース接触）~~ → ✓ status-074
- Phase 6.1-6.3（NN構成則・PI制約・ハイブリッド）
- Phase 7-8（モデルレジストリ・応用展開）

---

## Phase 1: アーキテクチャ再構成 ✓

- [x] `pyproject.toml` 作成、pytest統一、テスト整理
- [x] Protocol/ABC 導入（ElementProtocol, ConstitutiveProtocol）
- [x] 既存コード（Q4, TRI3, TRI6, Q4_BBAR）の Protocol 適合
- [x] 既存テスト通過確認（16テスト）

---

## Phase 2: 空間梁要素 ✓

- [x] 2.1: Euler-Bernoulli梁（2D, 6DOF）— 21テスト
- [x] 2.2: Timoshenko梁（2D, Cowper κ(ν)）— 25テスト
- [x] 2.3: Timoshenko梁（3D, 12DOF, ねじり・二軸曲げ）— 43テスト
- [x] 2.4: 断面モデル（矩形, 円形, パイプ）; [ ] 一般断面（メッシュベース, 未実装）
- [x] 2.5: Cosserat rod（四元数, B行列, SRI, Euler elastica検証）— 56テスト; [ ] ヘリカルばねテスト
- [x] 2.6: 数値試験F/W（3点曲げ・4点曲げ・引張・ねん回, 周波数応答, CSV出力）

---

## Phase 3: 幾何学的非線形 ✓

- [x] 3.1: Newton-Raphson, 弧長法（Crisfield）; [ ] ラインサーチ
- [x] 3.2: Cosserat非線形（四元数, Euler elastica検証）; [ ] Lee's frame
- [x] 3.3: 共回転（CR）定式化（Timo3D, dynamic_runner統合）— 24テスト
- [x] 3.4: Total/Updated Lagrangian（Q4要素）— 37テスト

---

## Phase 4: 材料非線形

- [x] 4.1: 1D弾塑性（return mapping, consistent tangent, 等方/移動硬化）✓
- [x] 4.2: ファイバーモデル（FiberSection, 曲げの塑性化）✓ — 36テスト
- [ ] 4.3: 3D von Mises — **凍結**（実装コード完了, テスト45件は凍結, [status-025](status/status-025.md)）
- [x] 4.4: Rayleigh減衰（Phase 2.6 で実装済み）; [ ] ヒステリシス減衰; [ ] 粘性項
- [ ] 4.5: 粘弾性（一般化Maxwell, Prony級数）
- [ ] 4.6: 異方性（断面方向依存剛性, 曲げ-ねじり連成）

---

## Phase 5: 動的解析 ✓

- [x] 5.1: Newmark-β, HHT-α, 陽解法（Central Difference）
- [x] 5.2: 整合質量行列, 集中質量行列（HRZ法）
- [x] 5.3: Rayleigh減衰, モーダル減衰
- [x] 5.4: 非線形動解析（NR + Newmark-β/HHT-α）— 6テスト

---

## Phase C: 梁–梁接触モジュール ✓

設計仕様: [beam_beam_contact_spec_v0.1](contact/beam_beam_contact_spec_v0.1.md)

- [x] C0: ContactPair/ContactState/geometry — 30テスト
- [x] C1: Broadphase（AABB格子）+ Active-setヒステリシス — 31テスト
- [x] C2: 法線AL + 接触接線剛性 + 接触付きNR（Outer/Inner分離）— 43テスト
- [x] C3: Coulomb摩擦 return mapping + μランプ — 27テスト
- [x] C4: merit line search + step length制御 — 26テスト
- [x] C5: 一貫接線（K_geo）+ slip consistent tangent + PDAS + 平行輸送 — 35テスト

---

## Phase 4.7: 撚線モデル

**目的**: 撚線の曲げ・ねじり連成挙動（ヒステリシス・疲労）を再現する。
**前提**: Phase 2.5（Cosserat rod）✓ + Phase 3（非線形）✓ + Phase C（接触基盤）✓

### Level 0: 基礎同定用 ✓

- [x] 撚線メッシュファクトリ（3/7/19/37/61/91本, 32テスト）
- [x] 多点接触撚線テスト（3本撚り5荷重 + 摩擦, 26テスト）
- [x] 接触グラフ表現（ContactGraph/ContactGraphHistory, 24テスト）
- [x] k_pen自動推定 + 段階的アクティベーション + ヘリカル摩擦安定化（31テスト）
- [x] 7本撚り収束達成（AL乗数緩和 + GMRES+ILU + Active Set Freeze + Pure Penalty, 9テスト）
- [x] 撚線ヒステリシス観測 + 統計分析 + 可視化 + ダッシュボード（42テスト）
- [x] 被膜モデル（CoatingModel, 19テスト）
- [x] シースモデル（SheathModel, 41テスト）
- [x] 撚撚線統合解析テスト（被膜付き3本撚線成功, 7本xfail, 20テスト）

### Level 0.5: シース挙動モデル（Stage S1〜S4）

解析的リング理論で剛性を表現（DOF追加ゼロ、Fourier近似内面、有限滑り対応）。

| Stage | 内容 | 状態 |
|-------|------|------|
| **S1** | 均一厚リングコンプライアンス行列（Fourier閉形式, 35テスト） | ✓ |
| **S2** | 膜厚分布 t(θ) Fourier近似 + 修正コンプライアンス（34テスト） | ✓ |
| **S3** | シース-素線/被膜 有限滑り（θ再配置 + 摩擦, 37テスト） | ✓ |
| **S4** | シース-シース接触（broadphase + インターケーブルフィルター, 9テスト） | ✓ |

### Level 1: 撚り解き（未実装）

- [ ] `θ_i(s)` を未知量化（ヘリックス拘束を解く）
- [ ] 被膜を「周方向せん断ばね＋圧縮ばね」で平均化

### Level 2: 素線曲げ・局所座屈（未実装）

- [ ] 素線を Cosserat rod 化（曲げ・ねじり含む）
- [ ] 接触ペア爆増の対策（近接のみ、代表接触、連続平均化）

---

## Phase 6: NNサロゲートモデル対応

### 6.0 2D定常熱伝導GNNサロゲート（PoC）✓

**問題設定**: 100mm×100mm アルミ板, 10×10 Q4メッシュ, ランダム発熱体 → 全ノード温度場予測。

**達成精度**:
- メッシュGNN: R²=0.973
- 全結合GNN: R²=0.995
- ハイブリッドGNN: R²=0.979（O(N²)→O(N)）

**PINN**:
- 不規則メッシュでPINN効果顕著（ΔR²+0.021 vs 正則+0.006）
- 大規模メッシュ（20×20, 441ノード）検証済み

**設計仕様（実装ペンディング）**:
- [接触プリスクリーニングGNN設計](contact/contact-prescreening-gnn-design.md)
- [k_pen推定MLモデル設計 v2](contact/kpen-estimation-ml-design.md)
- [**接触アルゴリズム根本整理 Phase C6**](contact/contact-algorithm-overhaul-c6.md)

### 6.1-6.3: NN構成則・PI制約・ハイブリッド（未実装）

- [ ] ConstitutiveProtocolのNN実装（PyTorch, 自動微分接線剛性）
- [ ] Physics-Informed制約（散逸不等式, 対称性, 正値性）
- [ ] ハイブリッドモデル（既知物理+残差補正NN）

---

## Phase 7: モデルレジストリとパラメータフィッティング（未実装）

- [ ] モデルレジストリ（JSON/YAMLシリアライゼーション, カタログ）
- [ ] パラメータフィッティング（scipy.optimize, Optuna, 感度解析）
- [ ] 実験データインタフェース（フォーマット定義, 前処理）

---

## Phase 8: 応用展開（将来計画）

- [ ] ミクロ-マクロ連成（FE²）
- [ ] 連続体要素拡張（平面応力, シェル）

---

## 優先度と依存関係

```
Phase 1 (アーキテクチャ) ✓
    ↓
Phase 2 (梁要素) ✓
    ↓
Phase 3 (幾何学的非線形) ✓
    ├── Phase 4 (材料非線形: 4.1-4.2 ✓, 4.3 凍結)
    ├── Phase 5 (動的解析) ✓
    └── Phase C (梁–梁接触) ✓
            ↓
        Phase 4.7 (撚線モデル: L0 ✓, L0.5 S1-S4 ✓, ブロック前処理ソルバー ✓)
            ↓
Phase 6 (NNサロゲート: 6.0 PoC ✓)
    ↓
Phase 7 (レジストリ/フィッティング)
    ↓
Phase 8 (応用展開)
```

**クリティカルパス**: Phase 4.7 Level 1-2 → Phase 7

---

## TODO 一覧（優先度順）

> 詳細は最新の [status ファイル](status/status-index.md) を参照。

### 高優先

- [x] CI実行結果確認 + CIキャッシュ導入（actions/cache）+ CIバッジ（status-073）
- [x] Stage S3: シース-素線/被膜 有限滑り（status-074, 37テスト）
- [x] Stage S4: シース-シース接触（status-074, 9テスト）
- [x] 7本撚りブロック前処理ソルバー（素線ブロック + GMRES, status-074, 9テスト）

### 中優先

- [x] pen_ratio改善（adaptive omega で AL乗数段階的蓄積, status-075, 5テスト）
- [x] 7本撚りサイクリック荷重テスト（status-075, 3テスト）
- [x] ブロックソルバー大規模メッシュ検証（16要素/素線, status-075, 3テスト）
- [x] adaptive omega 効果定量評価（n_outer_max=3〜5 での収束性比較, status-076, 4テスト）
- [x] 7本撚りサイクリック荷重でのヒステリシスループ面積計測（status-076, 1テスト）
- [x] 接触プリスクリーニングGNN Step 1（データ生成パイプライン, status-076, 17テスト）
- [x] k_pen推定MLモデル Step 1（特徴量抽出ユーティリティ, status-076, 7テスト）
- [x] PINN学習スパース行列対応（status-076, 8テスト）
- [ ] **Phase C6: 接触アルゴリズム根本整理**（ML に先立つ理論基盤整備、[設計仕様](contact/contact-algorithm-overhaul-c6.md)）
  - [x] C6-L1: Segment-to-segment Gauss 積分（Line-to-line 接触, 28テスト, status-077）
  - [x] C6-L2: 一貫接線の完全化（∂s/∂u, ∂t/∂u Jacobian, 15テスト, status-078）
  - [x] C6-L3: Semi-smooth Newton + NCP 関数（Outer loop 廃止, 35テスト, status-079）
  - [x] C6-L4: ブロック前処理強化（接触 Schur 補集合, 11テスト, status-080）
  - [ ] C6-L5: Mortar 離散化（必要に応じて）
- [ ] 接触プリスクリーニングGNN Step 2-5（グラフ構築 → モデル実装 → 推論統合 → 性能評価）
- [ ] k_pen推定ML v2 Step 2-7（グラフ構築 → 残差ベースデータ生成 → 共有GNN実装 → 学習 → ContactConfig統合 → ベンチマーク）
- [ ] ハイブリッドGNN+PINN組み合わせ検証

### 低優先

- [ ] Protocol拡張（NonlinearElement/DynamicElement適合化, ContactProtocol, SectionProtocol）
- [ ] Phase 4.3: von Mises 3D テスト解凍
- [ ] Phase 4.4-4.6: ヒステリシス減衰, 粘弾性, 異方性
- [ ] Phase 6.1-6.3: NN構成則, PI制約, ハイブリッド

---

## 設計原則

1. **モジュール合成可能性**: 要素・構成則・ソルバー・積分スキーマを自由に組み合わせ可能
2. **Protocol/ABCベース**: 具象クラスへの依存を避け、インタフェースに依存
3. **状態変数の明示管理**: 履歴変数は `StateVariable` で明示的に管理
4. **テスト駆動**: 各要素・構成則は解析解またはリファレンスソルバーとの比較テスト必須
5. **NN互換設計**: 構成則インタフェースはNN代替を考慮した入出力設計
6. **段階的拡張**: 既存テストを破壊しないよう後方互換性を保持

---

## 参考文献

- Crisfield, M.A. "Non-linear Finite Element Analysis of Solids and Structures" Vol. 1 & 2
- Bathe, K.J. "Finite Element Procedures"
- de Souza Neto et al. "Computational Methods for Plasticity"
- Simo, J.C. & Hughes, T.J.R. "Computational Inelasticity"
- Felippa, C.A. "A unified formulation of small-strain corotational finite elements"
- Simo, J.C. (1985) "A finite strain beam formulation"（Cosserat rod）
- Antman, S.S. "Nonlinear Problems of Elasticity"（Cosserat rod 理論）
- Costello, G.A. "Theory of Wire Rope"（撚線力学の基礎）
- Cardou, A. & Jolicoeur, C. (1997) "Mechanical Models of Helical Strands"
- Foti, F. & Martinelli, L. (2016) "Hysteretic bending of spiral strands"
