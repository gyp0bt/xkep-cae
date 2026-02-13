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

## 現在地（Phase 2.2 完了、Cosserat rod & 撚線モデルをロードマップに追加）

### 実装済み

| カテゴリ | 内容 |
|---------|------|
| **平面要素** | Q4（双線形四角形）, TRI3（一次三角形）, TRI6（二次三角形）, Q4_BBAR（B̄法）, **Q4_EAS（EAS-4, デフォルト）** |
| **梁要素** | Euler-Bernoulli梁（2D）, Timoshenko梁（2D, **Cowper κ(ν) 対応**） |
| **材料** | 線形弾性（平面ひずみ）, 1D梁弾性 |
| **断面** | 矩形断面, 円形断面（断面形状情報付き） |
| **ソルバー** | 直接法（spsolve）, AMG反復法（pyamg） |
| **境界条件** | Dirichlet（行列消去法 / Penalty法） |
| **API** | Protocol API（一本化）, ラベルベース高レベルAPI |
| **I/O** | Abaqus .inp パーサー（自前実装） |
| **検証** | 製造解テスト, Abaqusベンチマーク, 解析解比較, ロッキングテスト, **CPE4I精度検証**（115テスト） |
| **ドキュメント** | [Abaqus差異ドキュメント](abaqus-differences.md) |

### 未実装（現状の制約）

- 3次元要素なし（平面問題限定、3D梁未実装）
- 非線形解析なし（幾何学/材料）
- 応力・ひずみのポスト処理なし
- 動的解析なし

---

## Phase 1: アーキテクチャ再構成

**目的**: 線形平面ソルバーの実装を保持しつつ、多様な要素・構成則・ソルバーを組み合わせ可能なモジュール構成へ移行する。

### 1.1 プロジェクト基盤整備

- [x] `pyproject.toml` 作成（依存関係、ビルド設定）
- [x] テストフレームワーク（pytest）統一
- [x] 既存テストの `tests/` 配下への整理・命名規約統一
- [x] CI設定（GitHub Actions）

### 1.2 コア抽象レイヤー設計

以下のプロトコル（Protocol / ABC）を導入する。

```
xkep_cae/
├── core/
│   ├── element.py          # ElementProtocol: 要素の抽象インタフェース
│   ├── constitutive.py     # ConstitutiveProtocol: 構成則の抽象インタフェース
│   ├── integrator.py       # IntegratorProtocol: 時間積分スキーマ
│   ├── section.py          # SectionProtocol: 断面特性
│   ├── dof.py              # DOFの管理・番号付け
│   └── state.py            # StateVariable: 履歴変数・内部変数の管理
├── elements/               # 具象要素（既存 + 新規）
├── materials/              # 具象構成則
├── sections/               # 断面モデル
├── solvers/                # 線形・非線形ソルバー
├── assembly.py             # 要素アセンブリ
├── bc.py                   # 境界条件
└── api.py                  # ユーザーAPI
```

#### ElementProtocol の設計方針

```python
class ElementProtocol(Protocol):
    """要素の共通インタフェース"""
    ndof: int                           # 要素あたりのDOF数
    nnodes: int                         # 節点数

    def local_stiffness(self, coords, material, section, state) -> ndarray:
        """局所剛性行列"""
        ...

    def internal_force(self, coords, u_elem, material, section, state) -> ndarray:
        """内力ベクトル（非線形解析用）"""
        ...

    def update_state(self, coords, u_elem, material, section, state) -> StateVariable:
        """状態変数の更新"""
        ...

    def mass_matrix(self, coords, material, section) -> ndarray:
        """質量行列（動的解析用）"""
        ...
```

#### ConstitutiveProtocol の設計方針

```python
class ConstitutiveProtocol(Protocol):
    """構成則の共通インタフェース"""

    def tangent(self, strain, state) -> Tuple[ndarray, ndarray]:
        """接線剛性テンソルと応力を返す
        Returns: (D_tangent, stress)
        """
        ...

    def update(self, strain, state) -> StateVariable:
        """履歴変数の更新"""
        ...
```

### 1.3 既存コードの移行

- [x] 現行の `quad4.py`, `tri3.py`, `tri6.py`, `quad4_bbar.py` を `ElementProtocol` に適合
- [x] `elastic.py` を `ConstitutiveProtocol` に適合
- [x] `assembly.py` のアセンブリロジックを Protocol ベースに汎化
- [x] 既存テストが引き続きパスすることを確認（16 passed, 2 skipped）

---

## Phase 2: 空間梁要素

**目的**: Timoshenko梁理論に基づく空間梁要素を実装する。Phase 1のアーキテクチャ上に構築する。

### 2.1 Euler-Bernoulli梁（2D）

基礎として2D Euler-Bernoulli梁を実装。6 DOF/要素（各節点: ux, uy, θz）。

- [x] 形状関数（Hermite補間）
- [x] 局所剛性行列 Ke
- [x] 座標変換行列 T
- [x] 分布荷重の等価節点力
- [x] 単体テスト（片持ち梁解析解比較）— 21テスト

### 2.2 Timoshenko梁（2D）

せん断変形を考慮。同じDOF構成。

- [x] せん断補正係数 κ の導入
- [x] Cowper (1966) のν依存 κ(ν) 実装（`kappa="cowper"`、Abaqus準拠）
- [x] せん断ロッキング対策（整合定式化: Φ = 12EI/(κGAL²)）
- [x] 解析解との比較（太い梁問題）— 25テスト（Cowperκ含む）

### 2.3 Timoshenko梁（3D空間）

12 DOF/要素（各節点: ux, uy, uz, θx, θy, θz）。

- [ ] 3D座標変換行列（局所→全体）
- [ ] ねじり剛性 GJ
- [ ] 二軸曲げ（EIy, EIz）
- [ ] せん断変形（κy, κz）
- [ ] ワーピング（オプション、薄肉断面用）
- [ ] テスト：3D片持ち梁、ねじり問題

### 2.4 断面モデル

```python
class BeamSection:
    """梁の断面特性"""
    A: float          # 断面積
    Iy: float         # y軸まわり断面二次モーメント
    Iz: float         # z軸まわり断面二次モーメント
    J: float          # ねじり定数（St.Venant）
    ky: float         # y方向せん断補正係数
    kz: float         # z方向せん断補正係数
```

- [x] 基本断面（矩形、円形）— `BeamSection2D`
- [ ] 一般断面（任意形状、メッシュベース数値積分）
- [ ] ファイバーモデル断面（材料非線形用、Phase 4で拡張）

### 2.5 Cosserat rod（幾何学的厳密梁）

Timoshenko梁を超える一般的な梁定式化。中心線＋断面回転で幾何を記述し、
せん断・伸び・曲率・ねじりを統一的に扱う。撚線モデル（Phase 4.6）の土台となる。

#### 幾何の記述

```
中心線:     r(s)          ∈ R³
断面回転:   R(s)          ∈ SO(3)
一般化歪み: ν = Rᵀr'      （せん断＋軸伸び）
            κ = axial(RᵀR') （曲率＋ねじり）
```

#### 実装項目

- [ ] SO(3) 回転パラメトライゼーション（四元数 or 回転ベクトル）
- [ ] 一般化歪み (ν, κ) の計算
- [ ] Cosserat rod の弾性構成則（N = EA·ε, M = EI·κ 等の一般化）
- [ ] 弱形式・変分定式化（仮想仕事式）
- [ ] 有限要素離散化（Hermite or Lagrange interpolation on SO(3)）
- [ ] 接線剛性行列（材料剛性 + 幾何剛性）
- [ ] テスト：Euler elastica、ヘリカルばね、大回転片持ち梁
- [ ] Phase 3（幾何学的非線形）との統合

#### 参考文献

- Simo, J.C. (1985) "A finite strain beam formulation — Part I"
- Crisfield, M.A. & Jelenić, G. (1999) "Objectivity of strain measures in the geometrically exact beam"
- Antman, S.S. "Nonlinear Problems of Elasticity"

#### 備考

Cosserat rod は本質的に幾何学的非線形定式化であり、Phase 3 と密接に連携する。
Phase 3.2（共回転定式化）の代替として位置づけることも可能。
Phase 4.6（撚線モデル）では Cosserat rod が「外側の梁」および「個別素線」の
両方の幾何記述として使用される。

---

## Phase 3: 幾何学的非線形

**目的**: 大変形・大回転に対応する。梁の大たわみ・座屈・スナップスルーを解ける。

### 3.1 非線形ソルバーフレームワーク

- [ ] Newton-Raphson法
  - 残差ベクトル R = f_ext - f_int
  - 接線剛性行列 K_T
  - 収束判定（力・変位・エネルギーノルム）
- [ ] 荷重増分法（Load stepping）
- [ ] 弧長法（Arc-length / Riks法）：スナップスルー・座屈追跡用
- [ ] ラインサーチ（収束加速）

### 3.2 共回転（Corotational）定式化

空間梁の大回転を扱う主要手法。

- [ ] 要素ごとのローカルフレーム追従
- [ ] 剛体回転の分離と変形成分の抽出
- [ ] 接線剛性の幾何学的剛性項 Kg
- [ ] 回転のパラメタ化（四元数 or 回転ベクトル）
- [ ] テスト：大たわみ片持ち梁（Lee's frame等の標準ベンチマーク）

### 3.3 Updated Lagrangian（オプション）

- [ ] 参照配置の更新
- [ ] Green-Lagrangeひずみ
- [ ] 第二Piola-Kirchhoffストレス
- [ ] 共回転定式化との比較検証

---

## Phase 4: 材料非線形

**目的**: 弾塑性、粘弾性、異方性を梁要素に組み込む。

### 4.1 1次元弾塑性

梁の軸方向に対する1D弾塑性構成則から着手。

- [ ] 1D等方硬化モデル（線形硬化）
- [ ] 1D移動硬化モデル（Armstrong-Frederick）
- [ ] Return mapping アルゴリズム
- [ ] 接線剛性の一貫線形化（consistent tangent）
- [ ] 単軸引張・圧縮テスト

### 4.2 ファイバーモデル

断面をファイバー（微小断面要素）に分割し、各ファイバーに1D構成則を適用。

- [ ] ファイバー断面の離散化
- [ ] 断面力 ← ファイバーの応力積分
- [ ] 断面接線剛性 ← ファイバーの接線剛性積分
- [ ] テスト：弾塑性片持ち梁の荷重変位曲線

### 4.3 構造減衰

- [ ] Rayleigh減衰（C = αM + βK）
- [ ] ヒステリシス減衰（履歴変数ベース）
- [ ] 構成則レベルの粘性項

### 4.4 粘弾性

- [ ] 一般化Maxwell（Prony級数）モデル
- [ ] 再帰的更新アルゴリズム（計算効率）
- [ ] 緩和試験・クリープ試験シミュレーション

### 4.5 異方性

- [ ] 異方性弾性（断面方向依存の剛性）
- [ ] 梁の曲げ-ねじり連成（偏心、非対称断面）

### 4.6 撚線モデル（拡張ファイバー理論）

**目的**: ファイバー理論を拡張し、撚線電線の素線幾何剛性＋接触剛性＋摩擦散逸＋撚り解き＋
塑性履歴＋被膜の寄与を1Dモデルとして理想化する。
「離散素線モデルの縮約（homogenization/ROM）」。

**前提**: Phase 2.5（Cosserat rod）の完成が必須。Phase 4.1–4.2（弾塑性＋ファイバーモデル）が基盤。

#### 設計判断（確定事項）

| 項目 | 判断 | 備考 |
|------|------|------|
| **第一ターゲット用途** | 曲げ（＋ねじり連成曲げ・疲労） | θ_i の縮約が最重要。ねじり-曲げ連成時の δ_ij 追跡も必要 |
| **被膜モデルの目的** | 剛性寄与＋摩擦制御 | 理想化弾性体として扱う。温度・損傷は当面スコープ外 |
| **接触ペア戦略** | 隣接ペアのみ＋Nイテレーションごとにペア更新 | 実験的アプローチ。全ペア列挙は行わない |

##### 状態変数の選定方針（曲げ＋ねじり連成主目的）

- **θ_i(s)**: 曲げ時に素線がどう滑るかを決める最重要変数。Level 1 で未知量化。
- **δ_ij(s)**: ねじり-曲げ連成時の素線間相対すべり。摩擦ヒステリシスと疲労評価の基盤。
  疲労寿命推定には δ_ij の累積値（サイクルカウント）の追跡が必要。
- **被膜**: 弾性ばね（せん断剛性 G_c, 圧縮剛性 K_c）でモデル化。
  剛性への寄与は素線間の摩擦係数 μ を実効的に変化させる役割を持つ。

##### 接触ペア管理の詳細

```
初期ペアリスト: 幾何的に隣接する素線ペアのみ（撚り構造から自動生成）
更新頻度:       N_update イテレーションごとにペアリストを再評価
更新基準:       g_ij < g_threshold の素線ペアを活性ペアとして保持
               大変形で隣接関係が変化した場合にペアを追加/除去
```

この戦略は実験的であり、N_update, g_threshold のチューニングが必要。
大変形時にペア更新が遅れると力の不連続が生じるリスクがある。

#### 概念：通常のファイバー理論との差異

通常のファイバー梁では、断面上の繊維が軸方向に並行に走り、断面積分で合力を計算する：
```
ε_xx(y,z) = ε₀ + z·κ  →  断面積分で N, M を計算
```

撚線の拡張ファイバーでは、**ヘリカルに走る素線（1D）**を集めて合力・合モーメントを作る：
```
素線 i の位置:  x_i(s) = r(s) + R(s) · ρ_i(θ_i(s))
一般化内力:     n(s) = Σᵢ R·nᵢ^local + 接触反力の合力
一般化モーメント: m(s) = Σᵢ [R·mᵢ^local + (xᵢ-r)×(素線力)] + 接触偶力
```

ここで `θ_i(s)` はヘリックス位相であり、撚り解きが起きると**拘束ではなく未知量**になる。
これが通常のファイバー梁との決定的な差。

#### 内部変数

| 変数 | 物理的意味 | 役割 |
|------|-----------|------|
| `θ_i(s)` | 素線ヘリックス位相 | 撚り戻り・撚り解きの自由度 |
| `δ_ij(s)` | 素線 i–j の接線方向相対すべり | 摩擦散逸の主役 |
| `g_ij(s)` | ギャップ（接触開閉） | 接触状態管理 |
| stick/slip状態 | 離散フラグ or 連続正則化変数 | 摩擦モード |
| `α_i(s)` | 素線材料の塑性履歴 | 等方硬化等 |
| `γ_c(s)` | 被膜せん断変形 | 被膜の寄与（最小限） |

#### 構成則の構造

**素線の幾何剛性**:
各素線を Cosserat rod とみなし、外側の (r, R) と内部 θ_i から素線の歪みを写像で作る：
```
(νᵢ, κᵢ) = Gᵢ(ν, κ, θᵢ, θᵢ', ...)
U_wire = Σᵢ ∫ ψᵢ(νᵢ, κᵢ, αᵢ) ds
```

**接触剛性（法線方向）**:
```
U_contact = Σ_(i,j) ∫ ½ kₙ ⟨-g_ij⟩² ds    （penalty法）
```

**摩擦散逸（接線方向）**:
増分変分（incremental potential）で定式化：
```
min_Δ [ΔU(·) - ΔW_ext + D(Δδ, Δα)]
D_ij ~ ∫ μ·N_ij·|Δδ_ij| ds    （Coulomb摩擦の散逸ポテンシャル）
```

正則化（tanh等）で滑らかにしてNewtonで回すか、非滑らか最適化で扱う。

#### 段階的実装計画

**Level 0: 基礎同定用**（最初に作るべき）
- [ ] 素線は軸方向のみ（曲げ剛性無視）
- [ ] 接触は法線方向 penalty
- [ ] 摩擦は正則化 Coulomb
- [ ] 内部変数: `δ_ij` と素線塑性 `α_i` のみ
- [ ] テスト: 引張・ねじり・曲げの基本ヒステリシスが出るか確認

**Level 1: 撚り解き**（コア機能）
- [ ] `θ_i(s)` を未知量化（ヘリックス拘束を解く）
- [ ] 被膜を「周方向せん断ばね＋圧縮ばね」で平均化して追加
- [ ] テスト: 撚り戻り＋摩擦散逸の主要現象が乗ることを検証

**Level 2: 素線曲げ・局所座屈**
- [ ] 素線を Cosserat rod 化（曲げ・ねじり含む）
- [ ] 接触ペア爆増の対策（近接のみ、代表接触、連続平均化）
- [ ] テスト: 局所座屈モードの再現

#### 設計上の警告

1. **接触ペアの管理**: 隣接ペア＋定期更新で進めるが、ペア更新タイミングの不連続性に注意。
   大変形で隣接関係が急変するケースの検出ロジックが要る。
2. **被膜の弾性モデルの限界**: 剛性＋摩擦制御目的で弾性ばねとするが、
   温度依存性・粘弾性・損傷が必要になった場合はインタフェースを拡張できるよう設計しておく。
3. **摩擦の数値安定性**: 摩擦は数値的に悪条件。増分ポテンシャル（非滑らか最適化 or 正則化）を最初から前提にする。
4. **疲労評価のための δ_ij 追跡**: 曲げ＋ねじり連成での疲労を見るため、
   δ_ij のサイクルカウント（雨流計数法等）を追跡する仕組みが必要。Level 0 で基盤を作る。

#### 参考文献

- Costello, G.A. "Theory of Wire Rope"
- Cardou, A. & Jolicoeur, C. (1997) "Mechanical Models of Helical Strands"
- Jiang, W.G. et al. (2006) "Statically indeterminate contacts in axially loaded wire strand"
- Foti, F. & Martinelli, L. (2016) "An analytical approach to model the hysteretic bending behavior of spiral strands"

---

## Phase 5: 動的解析

**目的**: 揺動試験等の過渡応答解析を可能にする。

### 5.1 時間積分スキーマ

- [ ] Newmark-β法（暗黙的）
- [ ] α法（HHT法、数値減衰付き）
- [ ] 陽解法（Central Difference、オプション）

### 5.2 質量行列

- [ ] 集中質量行列（lumped mass）
- [ ] 整合質量行列（consistent mass）
- [ ] 梁要素の質量行列実装

### 5.3 減衰行列

- [ ] Rayleigh減衰（Phase 4.3と連携）
- [ ] モーダル減衰

---

## Phase 6: NNサロゲートモデル対応

**目的**: 構成則や物理スキーマの一部をニューラルネットワークで代替可能にする。

### 6.1 ConstitutiveProtocolのNN実装

- [ ] PyTorchバックエンドの `NNConstitutive` クラス
- [ ] 入力: ひずみ（＋履歴変数）→ 出力: 応力（＋接線剛性）
- [ ] 自動微分による接線剛性の算出
- [ ] 訓練データ生成インタフェース（従来構成則からの合成データ）

### 6.2 Physics-Informed制約

- [ ] 熱力学的整合性の制約（散逸不等式）
- [ ] 対称性拘束（接線剛性の主対称性）
- [ ] 物理量の正値性保証

### 6.3 ハイブリッドモデル

- [ ] 既知の物理モデル＋残差補正NN
- [ ] スケール分離（マクロモデル＋ミクロ補正）

---

## Phase 7: モデルレジストリとパラメータフィッティング

**目的**: 実験データから最適なモデルとパラメータを同定する。

### 7.1 モデルレジストリ

- [ ] モデル定義のシリアライゼーション（JSON/YAML）
- [ ] モデルカタログ（梁モデル種類 × 構成則 × 断面）
- [ ] バージョン管理と再現性保証

### 7.2 パラメータフィッティング

- [ ] 目的関数定義（荷重-変位曲線の残差）
- [ ] 最適化バックエンド（scipy.optimize, Optuna等）
- [ ] 感度解析（パラメータの影響度）
- [ ] ベイズ推定（パラメータの不確実性定量化、オプション）

### 7.3 実験データインタフェース

- [ ] 試験データフォーマット定義
- [ ] 前処理（フィルタリング、リサンプリング）
- [ ] 複数試験条件の同時フィッティング

---

## Phase 8: 応用展開（将来計画）

ロードマップの主軸完了後に検討する領域。

### 8.1 接触・多体連成

- 電線束内の芯線間接触
- 加工ロールと被加工材の接触境界条件

### 8.2 ミクロ-マクロ連成

- 微小クラックの変形追跡
- 計算ホモジナイゼーション（FE²）

### 8.3 連続体要素の拡張

- 既存2D要素の平面応力対応
- 3D固体要素（Hex8, Tet4, Tet10）
- シェル要素

---

## 優先度と依存関係

```
Phase 1 (アーキテクチャ)
    ↓
Phase 2 (梁要素)
    ├── 2.1-2.4: EB/Timoshenko/断面
    └── 2.5: Cosserat rod ─────────────┐
        ↓                              │
Phase 3 (幾何学非線形) ────────────────┤
    ↓                                  │
Phase 4 (材料非線形)                   │
    ├── 4.1-4.5: 弾塑性/粘弾性        │
    └── 4.6: 撚線モデル ←─────────────┘
        (Cosserat rod + ファイバー拡張 + 接触/摩擦)
    ↓
Phase 5 (動的解析) ← Phase 3完了後に並行可
    ↓
Phase 6 (NNサロゲート) ← Phase 4の構成則設計に依存
    ↓
Phase 7 (レジストリ/フィッティング) ← Phase 2-6の全要素が必要
    ↓
Phase 8 (応用展開) ← 必要に応じて
```

**クリティカルパス**: Phase 1 → 2 → 3 → 4

**撚線モデルのクリティカルパス**: Phase 2.5 (Cosserat rod) → Phase 3 → Phase 4.1-4.2 → Phase 4.6 (撚線モデル)

**並行開発可能**: Phase 5は Phase 3完了後に並行可、Phase 6は Phase 4の構成則設計と並行可

---

## 設計原則

1. **モジュール合成可能性**: 要素・構成則・ソルバー・積分スキーマを自由に組み合わせ可能
2. **Protocol/ABCベース**: 具象クラスへの依存を避け、インタフェースに依存
3. **状態変数の明示管理**: 履歴変数は `StateVariable` で明示的に管理、暗黙のグローバル状態を排除
4. **テスト駆動**: 各要素・構成則は解析解またはリファレンスソルバーとの比較テスト必須
5. **NN互換設計**: 構成則インタフェースは最初からNN代替を考慮した入出力設計
6. **段階的拡張**: 既存のテストを破壊しないよう、後方互換性を保ちながら拡張

---

## 参考文献（予定）

- Crisfield, M.A. "Non-linear Finite Element Analysis of Solids and Structures" Vol. 1 & 2
- Bathe, K.J. "Finite Element Procedures"
- de Souza Neto et al. "Computational Methods for Plasticity"
- Simo, J.C. & Hughes, T.J.R. "Computational Inelasticity"
- Felippa, C.A. "A unified formulation of small-strain corotational finite elements"
- Simo, J.C. (1985) "A finite strain beam formulation"（Cosserat rod）
- Antman, S.S. "Nonlinear Problems of Elasticity"（Cosserat rod 理論）
- Costello, G.A. "Theory of Wire Rope"（撚線力学の基礎）
- Cardou, A. & Jolicoeur, C. (1997) "Mechanical Models of Helical Strands"（撚線モデル）
- Foti, F. & Martinelli, L. (2016) "Hysteretic bending of spiral strands"（撚線曲げヒステリシス）
