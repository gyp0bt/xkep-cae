# xkep-cae ロードマップ

[← README](../README.md)

## プロジェクトビジョン

汎用FEMソフトでは解けないニッチドメイン問題を解くための自作有限要素ソルバー基盤。
支配方程式・構成則・要素・更新則・積分スキーマ・接触・非線形をモジュール化し、
組み合わせて問題特化ソルバーを構成できるフレームワークを目指す。

### 第一ターゲット：1000本撚線の曲げ揺動シミュレーション

> **マイルストーン: 1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了する。**

---

## 現在地（2026-03-08）

**2271テスト。Phase 1〜C6 + S1-S2 + 高速化基盤 完了。S3改良12項目実装 + 段階的活性化NCP移植。37本NCP収束達成（Layer1径方向圧縮）。**NCP 6x高速化（解析的接線+バッチ接触幾何）+ 要素ループ12.6xバッチ化達成。7本撚線90°曲げ+揺動1周期が130秒で完全収束。**ソルバー一本化完了（n_load_steps廃止、NCPSolverInput導入、adaptive_timestepping=Trueデフォルト化）。Updated Lagrangian CR梁アセンブラで大回転収束障壁を解消。接触診断2D投影可視化。被膜接触モデル再構築（gap_offset廃止→被膜厚考慮メッシュ+被膜スプリング）。mm-ton-MPa単位系移行開始。被膜Kelvin-Voigt粘性減衰。k_pen材料ベース自動推定強制。**

### 到達点

| 領域 | 到達点 | 実証 |
|------|--------|------|
| FEM要素 | 梁（EB/Timo/CR/Cosserat）+ 平面（Q4/TRI）+ 固体（HEX8） | 解析解一致 |
| 非線形 | 幾何学的非線形（NR/弧長/CR/TL/UL）+ 弾塑性 + ファイバーモデル | Abaqus比較 |
| 動的解析 | Newmark-β/HHT-α/Generalized-α/陽解法/モーダル減衰 | 解析解一致 |
| 接触 | NCP Semi-smooth Newton + Line contact + Mortar + Coulomb摩擦 | 7本撚り収束 |
| 撚線 | 7本撚り曲げ揺動収束、被膜/シース、ヒステリシス観測 | 計測データあり |
| 高速化 | COO/CSRベクトル化、共有メモリ並列、ブロック前処理、修正NR法 | 91本まで計測 |
| ML | GNN/PINNサロゲートPoC（R²=0.995） | 熱伝導2D |

### 未到達（現在の壁）

| 問題 | 現状 | 必要なこと |
|------|------|-----------|
| **大規模収束** | **37本NCP収束達成**（径方向圧縮Layer1） | 61本以上、曲げ揺動での収束 |
| **処理時間** | 91本で~25分（曲げ揺動） | 1000本6時間 → 要100倍以上の高速化 |
| **broadphase** | 1000本で~18秒、733万候補ペア | ML候補削減、空間分割改善 |
| **メモリ** | 未計測 | 1000本で推定~2GB（K行列 nnz ~1.5M） |

---

## 完了済みフェーズ（詳細は [archive/](archive/) を参照）

| Phase | 内容 | テスト数 | status |
|-------|------|---------|--------|
| 1 | アーキテクチャ（Protocol/ABC） | 16 | 001-003 |
| 2 | 空間梁要素（EB/Timo/Cosserat/数値試験） | ~360 | 004-015 |
| 3 | 幾何学的非線形（NR/弧長/CR/TL/UL） | ~100 | 015-042 |
| 4.1-4.2 | 弾塑性 + ファイバーモデル | ~70 | 021-023 |
| 5 | 動的解析（Newmark-β/HHT-α/陽解法/モーダル減衰） | ~60 | 026-030 |
| C0-C5 | 梁–梁接触（AL/摩擦/merit LS/PDAS/一貫接線） | ~190 | 033-046 |
| 4.7 L0 | 撚線基礎（7本/被膜/シース/ヒステリシス） | ~350 | 052-056 |
| 4.7 L0.5 | シース挙動（リングコンプライアンス/Fourier近似） | ~70 | 058-064 |
| HEX8 | 3D固体要素（C3D8/C3D8B/C3D8R/C3D8I） | ~80 | 059-063 |
| I/O | Abaqusパーサー/過渡応答出力/GIF | ~130 | 031-041, 105-106 |
| 6.0 | GNN/PINNサロゲートPoC | ~100 | 066-069 |
| C6 | Line contact + NCP + Mortar + 摩擦 | ~130 | 077-086 |
| S1 | 同層除外 + NCP摩擦 + Alart-Curnier | ~50 | 083-086 |
| S2 | CPU並列化基盤 + GMRES + COO/CSR + 共有メモリ | ~40 | 087-096 |

> 詳細設計は各モジュールの README.md を参照:
> - 接触: [xkep_cae/contact/README.md](../xkep_cae/contact/README.md)
> - 要素: [xkep_cae/elements/README.md](../xkep_cae/elements/README.md)
> - メッシュ: [xkep_cae/mesh/README.md](../xkep_cae/mesh/README.md)

---

## 推奨ソルバー構成（リファレンス構成）

**今後の全ベンチマークは以下の構成を基準として実施する。**

| 項目 | 設定 | 根拠 |
|------|------|------|
| **ソルバー** | `newton_raphson_contact_ncp`（`solver_ncp.py`） | Outer loop 不要、λ暴走なし |
| **接触離散化** | Line-to-line Gauss 積分 | セグメント間力の連続性 |
| **NCP関数** | Fischer-Burmeister | 微分可能、正則化容易 |
| **摩擦** | Coulomb return mapping + NCP ハイブリッド | 法線NCP + 摩擦return mapping |
| **Mortar** | 適応ペナルティ付き Mortar | 力の連続化 |
| **同層除外** | `exclude_same_layer=True` | ~80% ペア削減 |
| **k_pen** | 自動推定（beam EI ベース） | 手動設定不要 |
| **線形ソルバー** | DOF閾値自動切替（直接法 / GMRES+ILU） | スケーラビリティ |
| **前処理** | ブロック前処理 Schur 補集合 | 接触+構造結合系の効率的解法 |

> AL法ソルバー（`solver_hooks.py`）はレガシー比較用。詳細は [status-098](status/status-098.md)。

---

## Phase S: 1000本撚線への道（アクティブ）

### 計測済みスケーリングデータ

| 素線数 | DOF | 計算時間 | 対7本比 | broadphase | 候補ペア |
|---:|---:|---:|---:|---:|---:|
| 7 | 210 | 92s | 1.0x | — | — |
| 19 | 570 | 239s | 2.6x | — | — |
| 37 | 1,110 | 501s | 5.4x | — | — |
| 61 | 1,830 | 903s | 9.8x | — | — |
| 91 | 2,730 | 1,476s | 16.0x | 0.12s | 66,066 |
| 1000 | 30,000 | — | — | 18s | 7,335,879 |

> 91本でスケーリング効率 0.81。structural_tangent が支配的（65%）、geometry_update が O(n²) で急増（6.5%）。

### S3: 大規模収束改善 ← **現在地**

**目標**: NCPソルバーで91本撚りの曲げ揺動が収束する。

| 項目 | 状態 |
|------|------|
| 7本 NCP収束 | ✅ チューニング済み（adaptive omega + λ_nキャッピング） |
| 19本 NCP収束 | ✅ 径方向圧縮テスト収束達成（status-112）、24ペアアクティブ |
| 37本 NCP収束（Layer1） | ✅ 径方向圧縮Layer1収束達成（status-121）|
| NCP版摩擦バリデーション | ✅ 旧ソルバーテストNCP移行16件（status-121）|
| NCP版ヒステリシス | ✅ 旧ソルバーテストNCP移行9件（status-121）|
| S3パラメータチューニング基盤 | ✅ 6テスト（status-097） |
| NCP収束安定化機能 | ✅ line search / MNR / 接線予測子 / エネルギー収束 / 変位制御 / チェックポイント二分法（status-103） |
| 91本タイミング計測 | ✅ ~25分（status-099） |
| スクリプトNCP対応 | ✅ `scripts/run_bending_oscillation.py` に `--ncp` オプション追加（status-103） |
| メッシュ非貫入制約 | ✅ `strand_diameter` 指定時の非貫入配置自動計算（status-104） |
| ILU drop_tol 適応制御 | ✅ 失敗時に10倍緩和リトライ（status-107） |
| Schur正則化改善 | ✅ 対角最大値ベースの適応正則化（status-107） |
| GMRES restart適応 | ✅ restart = min(max(30, n/10), 200)（status-107） |
| λウォームスタート | ✅ 近傍ペア中央値で初期推定（status-107） |
| Active setチャタリング抑制 | ✅ 過半数投票（時間方向畳み込み）（status-107） |
| レガシーテストdeprecated化 | ✅ 旧ソルバー5ファイルにdeprecatedマーカー（status-107） |
| 適応時間増分制御 | ✅ 収束反復数+接触変化率ベースの動的Δt（status-109） |
| AMG前処理 | ✅ PyAMG SA前処理（ILU代替）（status-109） |
| k_pen continuation | ✅ 段階的ペナルティ増大（status-109） |
| k_pen自動推定（NCP） | ✅ beam_ei/ea_l 自動推定をNCPソルバーに移植（status-109） |
| omega回復メカニズム | ✅ 最小値張り付き脱出（status-109） |
| スライドテスト修正 | ✅ 適応Δtで摩擦スライド不収束を解消（status-109） |
| ステップ二分法deprecated化 | ✅ max_step_cuts→adaptive_timestepping統合（status-110） |
| 残差スケーリング | ✅ 対角スケーリング前処理（status-110） |
| 接触力ランプ | ✅ Newton初期の接触力段階的増大（status-110） |
| チューニングタスクスキーマ | ✅ TuningTask/Param/Criterion/Run/Result + JSON/YAML直列化（status-114,115） |
| 検証プロット6種 | ✅ スケーリング・接触トポロジー・タイミング内訳・断面マップ・合格判定・感度ヒートマップ（status-114,115） |
| チューニング実行エンジン | ✅ execute_s3_benchmark + グリッドサーチ + 感度分析（status-114,115） |
| Optuna連携 | ✅ 自動チューニングループ基盤（create_objective/run_optuna_study）（status-115） |
| 応力・曲率連続性テスト | ✅ 隣接要素間変化率チェック物理テスト11件（status-115） |
| テスト失敗修正+deprecated追加 | ✅ block preconditioner修正、旧ソルバーテスト5ファイル+1クラス deprecated化（status-116） |
| 動的解析物理テスト | ✅ エネルギー保存・大変形・対称性・周波数・安定性の13テスト（status-117） |
| Generalized-α法 | ✅ Chung-Hulbert 1993ベースの時間離散化、14テスト（status-117） |
| CR vs Cosserat比較 | ✅ 物理・収束性・計算コスト定量比較、段階的移行方針（status-117） |

| 接触診断2D投影可視化 | ✅ 四元数回転→2D投影、接触力ベクトル場・ギャップ分布・断面ビュー（status-123） |
| 変形前後3Dレンダリング | ✅ NCP解適用変形メッシュの多視点2D投影比較（status-123） |
| 19/37本初期形状レンダリング | ✅ 層別色分け2D投影（status-123） |
| 61/91本段階的テスト | ✅ Layer1径方向圧縮テスト追加（status-123） |
| NCPソルバー段階的活性化 | ✅ solver_ncp.pyにstaged_activation移植（status-126） |
| NCP曲げ揺動テスト | ✅ 7本/19本CR梁曲げ揺動テスト8件追加（status-126） |
| 3Dプロット2D投影置換 | ✅ mplot3d→四元数2D投影に完全移行（status-126） |
| NCP摩擦接触の行列特異化修正 | ✅ J_t_t正則化でTimo3D摩擦収束達成（status-128） |
| CR梁収束問題の根本原因診断 | ✅ 寄生軸力が原因（EA/(EI/L²)≈9000）（status-129） |
| Updated Lagrangian CR梁アセンブラ | ✅ 7本90°曲げ収束、~13°障壁解消（status-130） |
| UL+NCP統合（adaptive_timestepping連動） | ✅ 手動ループ廃止、角度増分自動制御（status-131） |
| NCP 6x高速化（解析的接線+バッチ接触） | ✅ 49.6秒→8.3秒、7本90°曲げ+揺動130秒（status-132） |
| Phase2揺動収束達成 | ✅ prescribed_dofs+adaptive方式、7本90°+揺動1周期収束（status-132） |
| 初期時間増分の物理ベース推定 | ✅ 曲げ3°相当の等価増分でPhase2初期Δt自動決定（status-132） |
| n_load_steps=1対応 + 安定化成長戦略 | ✅ dt_initial_fraction物理ベース初期Δt、TCP類似安定化成長、Phase1 1.37x高速化（status-133） |
| 要素ループバッチ化（46%ボトルネック解消） | ✅ 全要素NumPyベクトル化、12.6x高速化（status-134） |
| ソルバー一本化（n_load_steps廃止） | ✅ adaptive_timestepping=Trueデフォルト、NCPSolverInput追加、用語統一（status-134） |

**TODO**:
- [x] 7本NCP曲げ揺動のCI確認（slowテスト）→ xfailで安定化（status-127）
- [x] NCP摩擦接触の行列特異化修正（status-128）
- [x] CR梁ヘリカル要素の収束問題の根本原因診断（status-129）
- [x] Updated Lagrangian実装で7本45°/90°曲げ収束達成（status-130）
- [x] UL+NCP統合: adaptive_timesteppingとUL参照更新の一体化（status-131）
- [x] UL Phase 2（揺動）の特異行列問題修正（status-132）
- [x] NCP 6x高速化: 解析的接線剛性+バッチ接触幾何（status-132）
- [x] 非線形接触動解析ソルバーモジュール完全一本化（status-134）
- [x] 要素ループのベクトル化（12.6x高速化、status-134）
- [x] 19本撚線の曲げ揺動収束確認（status-135: 45°+90°+揺動 ALL PASS）
- [x] **gap_offset手法の廃止と被膜接触モデルの再構築**（status-137: メッシュ側で被膜厚gap確保 + 被膜弾性スプリングモデル）
- [x] **被膜接線剛性実装 + 6DOFバグ修正 + 収束検証**（status-139: compute_coating_stiffness追加、k=1e6で完全収束、3D投影可視化）
- [x] **mm-ton-MPa移行 + Kelvin-Voigt粘性減衰 + k_pen材料ベース強制**（status-140）
- [ ] 全テストのmm-ton-MPa移行（~100ファイルの定数変換）
- [x] **被膜Coulomb摩擦モデル実装 + 摩擦core関数抽出**（status-141: return_mapping_core/tangent_2x2_core純粋関数化、被膜摩擦のsolver_ncp統合）
- [x] **Mortarギャップ計算バグ修正**（status-142: pair.state.radius_a→pair.radius_a、全Mortar NCPテストで接触力ゼロだった致命的バグ）
- [x] **撚線メッシュ初期貫入の解消**（status-143: mesh_gap=0.15mmで弦近似誤差による全貫入ペアをゼロ化）
- [x] **7本NCP曲げ揺動の接触あり収束達成**（status-143: Point contact + mesh_gap方式、45°/90°曲げ収束）
- [x] **摩擦あり（μ=0.1）曲げ揺動収束検証**（status-143: 45°曲げ+摩擦μ=0.1収束確認）
- [ ] 被膜摩擦μ=0.25の収束達成（接触チャタリング対策が必要）
- [ ] 19本→37本のスケールアップ
- [ ] CR梁の摩擦接触不収束の原因調査
- [ ] 37本Layer1+2圧縮の段階的活性化による収束改善確認
- [ ] NCPソルバー版S3ベンチマーク（AL法との計算時間比較）
- [ ] Cosserat Rodの解析的接線剛性実装

### S4: 撚線構造剛性比較

**目標**: 被膜/シース付き撚線の等価剛性を系統的に計測し、文献値と比較する。

| 項目 | 状態 |
|------|------|
| 単体要素比較（HEX8 vs 梁） | ✅ 梁が適切と結論（status-097） |
| 素線+被膜 等価剛性（引張/曲げ/ねじり/圧縮） | ✅ 10テスト（status-098） |
| 素線+シース 等価剛性 | ✅ 10テスト（status-098） |
| フルモデル（素線+被膜+シース） | ❌ 未実装 |
| 大変形 + 文献値比較（Costello, Foti） | ❌ 未実装 |

### S5: ML導入

**目標**: 接触候補削減と k_pen 推定の自動化。

- [ ] 接触プリスクリーニングGNN（Step 2-5）
- [ ] k_pen推定ML v2（Step 2-7）
- [ ] graphベースMLによる時間増分スキーマ最適化（接触グラフ+収束履歴→最適Δt予測）
- [ ] ハイブリッドGNN+PINN組み合わせ

### S6: 1000本撚線

**目標**: 1000本撚線の曲げ揺動計算を6時間以内に完了する。

| 項目 | 状態 |
|------|------|
| メッシュ生成 | ✅ 1000本レイアウト対応 |
| broadphaseスケーリング | ✅ ~18秒、733万候補（status-097） |
| メモリプロファイリング | ❌ 未計測 |
| 接触NR収束テスト | ❌ S3改善後 |
| **6時間目標達成** | ❌ |

### S7: GPU対応

S6の結果を踏まえ、ボトルネックに応じたGPU化。

- [ ] プロファイリングに基づくホットスポット特定
- [ ] CuPy/JAX疎行列演算GPU化
- [ ] GPU GMRES

### 依存関係

```
S3 (大規模収束) ← 現在地
  ↓
S4 (剛性比較)
  ↓
S5 (ML導入) → 候補ペア削減で S6 の前提条件
  ↓
S6 (1000本 6時間) ← ターゲットマイルストーン
  ↓
S7 (GPU)
```

---

## 凍結・将来計画

| Phase | 内容 | 状態 |
|-------|------|------|
| 4.3 | von Mises 3D 塑性 | 凍結（コード完了、テスト45件凍結） |
| 4.4-4.6 | ヒステリシス減衰、粘弾性、異方性 | 未実装 |
| 6.1-6.3 | NN構成則、PI制約、ハイブリッド | 未実装 |
| 7 | モデルレジストリ、パラメータフィッティング | 未実装 |
| 8 | FE²、連続体要素拡張 | 未実装 |

---

## 設計原則

1. **モジュール合成可能性**: 要素・構成則・ソルバー・積分スキーマを自由に組み合わせ可能
2. **Protocol/ABCベース**: 具象クラスへの依存を避け、インタフェースに依存
3. **テスト駆動**: 各要素・構成則は解析解またはリファレンスソルバーとの比較テスト必須
4. **段階的拡張**: 既存テストを破壊しないよう後方互換性を保持

---

## 参考文献

- Crisfield, M.A. "Non-linear Finite Element Analysis of Solids and Structures" Vol. 1 & 2
- Bathe, K.J. "Finite Element Procedures"
- de Souza Neto et al. "Computational Methods for Plasticity"
- Simo, J.C. & Hughes, T.J.R. "Computational Inelasticity"
- Costello, G.A. "Theory of Wire Rope"
- Foti, F. & Martinelli, L. (2016) "Hysteretic bending of spiral strands"
