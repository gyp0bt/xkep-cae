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

## 現在地（Phase 5.1〜5.2 実装完了 + 戻り値型リファクタリング完了）

Phase 1〜3 + Phase 4.1〜4.2 完了。Phase 5.1（Newmark-β/HHT-α時間積分）および Phase 5.2（集中質量行列）実装完了（498テスト）。
メソッド戻り値の NamedTuple 型クラス化（LinearSolveResult, DirichletResult, AssemblyResult 等）完了。
Phase 4.3（von Mises 3D弾塑性）の実装完了、テスト未実装（[テスト計画](status/status-025.md)策定済み、45テスト予定）。
非線形 Cosserat rod（回転ベクトル定式化）+ 弧長法が動作し、Euler elastica ベンチマーク検証済み。
1D弾塑性構成則（return mapping, consistent tangent, 等方/移動硬化, Armstrong-Frederick）実装完了。
ファイバーモデル断面（曲げの塑性化）実装完了。FiberSection + ファイバー積分アセンブリ。
全 Phase のバリデーションテストを[検証文書](verification/validation.md)に図付きで文書化済み。
ラインサーチと Lee's frame 等の追加ベンチマークはオプションとして残存。
動的解析: 整合質量行列・Rayleigh減衰・FRFは Phase 2.6 で先行実装、Newmark-β/HHT-α時間積分・集中質量行列（HRZ法）を追加。

### 実装済み

| カテゴリ | 内容 |
|---------|------|
| **平面要素** | Q4（双線形四角形）, TRI3（一次三角形）, TRI6（二次三角形）, Q4_BBAR（B̄法）, **Q4_EAS（EAS-4, デフォルト）** |
| **梁要素** | Euler-Bernoulli梁（2D）, Timoshenko梁（2D, Cowper κ(ν), SCF対応）, **Timoshenko梁（3D空間, 12DOF）** |
| **Cosserat rod** | 四元数回転, B行列定式化, 線形化版要素, 内力ベクトル, 幾何剛性行列, 初期曲率, SRI |
| **材料** | 線形弾性（平面ひずみ）, 1D梁弾性 |
| **断面** | 矩形, 円形, パイプ（2D/3D, Iy/Iz/J 対応） |
| **非線形ソルバー** | Newton-Raphson（荷重増分 + K_T = K_m + K_g）, 弧長法（Crisfield）, 非線形 Cosserat rod, Euler elastica 検証済み |
| **動的解析** | Newmark-β/HHT-α時間積分, 整合質量行列（2D/3D梁）, 集中質量行列（HRZ法, 2D/3D梁）, Rayleigh減衰, 周波数応答関数（FRF） |
| **ポスト処理** | 2D/3D断面力, 最大曲げ応力, 最大せん断応力（ねじり+横せん断） |
| **数値試験** | 3点曲げ・4点曲げ・引張・ねん回・周波数応答（`numerical_tests`パッケージ） |
| **ソルバー** | 直接法（spsolve）, AMG反復法（pyamg） |
| **境界条件** | Dirichlet（行列消去法 / Penalty法） |
| **API** | Protocol API（一本化）, ラベルベース高レベルAPI |
| **I/O** | Abaqus .inp パーサー, CSV出力, Abaqusライクテキスト入力 |
| **材料（非線形）** | 1D弾塑性（return mapping, consistent tangent, 等方/移動硬化, Armstrong-Frederick）, ファイバーモデル断面（曲げの塑性化） |
| **断面（非線形）** | ファイバーモデル断面（FiberSection: 矩形/円形/パイプ, ファイバー積分による断面力・接線剛性） |
| **検証** | 製造解テスト, Abaqusベンチマーク, 解析解比較, ロッキングテスト, 周波数応答解析解比較, Euler elastica, 弧長法, 弾塑性棒, ファイバーモデル曲げ, 過渡応答（SDOF/梁/集中質量）（**498テスト**）, [バリデーション文書](verification/validation.md) |
| **ドキュメント** | [Abaqus差異](abaqus-differences.md), [Cosserat設計](cosserat-design.md), [接触仕様](contact/beam_beam_contact_spec_v0.1.md) |

### 未実装（現状の制約）

- ラインサーチ・Lee's frame 等の追加ベンチマーク（Phase 3 オプション）
- 3次元連続体要素なし（平面問題限定）
- 材料非線形は1D弾塑性+ファイバーモデルのみ実装済み（3D von Mises塑性・粘弾性等は未実装、Phase 4.3 で計画）
- 陽解法（Central Difference）未実装
- モーダル減衰未実装
- ElementProtocol への `mass_matrix()` 統合未実装
- 梁–梁接触モジュールは設計完了・実装未着手

---

## Phase 1: アーキテクチャ再構成 ✓

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

## Phase 2: 空間梁要素 ✓

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

- [x] 3D座標変換行列（局所→全体）— 自動選択 + v_ref 指定
- [x] ねじり剛性 GJ
- [x] 二軸曲げ（EIy, EIz）
- [x] せん断変形（κy, κz）— Cowper κ 対応
- [x] ~~ワーピング~~ — スキップ（薄肉開断面用、円形/パイプ断面では不要。[status-010](status/status-010.md) 参照）
- [x] 断面力ポスト処理（`BeamForces3D`, `beam3d_section_forces()`, `beam3d_max_bending_stress()`）
- [x] 3Dアセンブリレベルテスト（`test_protocol_assembly.py` に5件追加）
- [x] テスト：3D片持ち梁、ねじり問題 — 43テスト（断面力テスト含む）

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
- [x] 3D断面（矩形, 円形, パイプ）— `BeamSection`（A, Iy, Iz, J, Cowper κy/κz）
- [ ] 一般断面（任意形状、メッシュベース数値積分）
- [x] ファイバーモデル断面（FiberSection: 矩形/円形/パイプ、Phase 4.2 で実装）

### 2.5 Cosserat rod（幾何学的厳密梁） ✓

Timoshenko梁を超える一般的な梁定式化。中心線＋断面回転で幾何を記述し、
せん断・伸び・曲率・ねじりを統一的に扱う。撚線モデル（Phase 4.7）の土台となる。

#### 幾何の記述

```
中心線:     r(s)          ∈ R³
断面回転:   R(s)          ∈ SO(3)
一般化歪み: ν = Rᵀr'      （せん断＋軸伸び）
            κ = axial(RᵀR') （曲率＋ねじり）
```

#### 実装項目

- [x] SO(3) 回転パラメトライゼーション（**四元数**で実装）— `xkep_cae/math/quaternion.py`
- [x] 一般化歪み (Γ, κ) の計算 — B行列ベース
- [x] Cosserat rod の弾性構成則 C = diag(EA, κy·GA, κz·GA, GJ, EIy, EIz)
- [x] 有限要素離散化（2節点線形要素、1点ガウス求積）— `beam_cosserat.py`
- [x] 線形化版テスト: 軸引張(厳密)、ねじり(厳密)、曲げ(メッシュ収束)、3点曲げ — 36テスト
- [x] 設計仕様書 — `docs/cosserat-design.md`
- [x] 内力ベクトル `internal_force()`（Phase 3 非線形用）— 線形等価性検証済み
- [x] 幾何剛性行列 `geometric_stiffness()`（軸力N + ねじりMx）— 対称性・正定値性検証済み
- [x] 初期曲率 `kappa_0` サポート（ヘリカル構造基盤）— ストレスフリー配位検証済み
- [x] 数値試験フレームワークへの統合（`beam_type="cosserat"`）— 56テスト
- [x] SRI（選択的低減積分）: せん断のみ1点低減、他2点完全積分
- [x] 大変形梁テスト（Newton-Raphson + 接線剛性）
- [x] テスト：Euler elastica（解析解比較）— 端モーメント(5ケース) + 先端荷重(4ケース)
- [ ] テスト：ヘリカルばね
- [x] Phase 3（幾何学的非線形）との統合（Newton-Raphson基盤完了）

#### 参考文献

- Simo, J.C. (1985) "A finite strain beam formulation — Part I"
- Crisfield, M.A. & Jelenić, G. (1999) "Objectivity of strain measures in the geometrically exact beam"
- Antman, S.S. "Nonlinear Problems of Elasticity"

#### 備考

Cosserat rod は本質的に幾何学的非線形定式化であり、Phase 3 と密接に連携する。
Phase 3.2（共回転定式化）の代替として位置づけることも可能。
Phase 4.7（撚線モデル）では Cosserat rod が「外側の梁」および「個別素線」の
両方の幾何記述として使用される。

### 2.6 数値試験フレームワーク（一括・部分実行） ✓

**目的**: 材料試験（3点曲げ・4点曲げ・引張・ねん回）の数値シミュレーションを
統一インタフェースで定義・実行・比較できるフレームワークを構築する。
一括実行（全試験）と部分実行（指定試験のみ）の両方をサポートする。

#### 対象試験

| 試験種別 | 略称 | 荷重条件 | 主要な断面力 | 解析解の有無 |
|---------|------|---------|-------------|------------|
| **3点曲げ試験** | `bend3p` | 中央集中荷重、両端単純支持 | V, M | あり: δ = PL³/(48EI) + PL/(4κGA) |
| **4点曲げ試験** | `bend4p` | 2点荷重、両端単純支持 | V, M（純曲げ区間あり） | あり: δ = Pa(3L²-4a²)/(48EI) + Pa/(κGA) |
| **引張試験** | `tensile` | 軸方向引張荷重 | N | あり: δ = PL/(EA) |
| **ねん回試験** | `torsion` | 軸方向ねじりモーメント | Mx | あり: θ = TL/(GJ) |

#### 設計方針

```python
@dataclass
class NumericalTest:
    """数値試験の定義."""
    name: str                   # 試験名 ("bend3p", "bend4p", "tensile", "torsion")
    beam_type: str              # "eb2d", "timo2d", "timo3d"
    section: BeamSection | BeamSection2D
    material: BeamElastic1D
    length: float               # 試料長さ
    n_elems: int                # 要素分割数
    load_value: float           # 荷重値 (P or T)
    # 4点曲げ用
    load_span: float | None     # 荷重スパン a（4点曲げのみ）

@dataclass
class TestResult:
    """試験結果."""
    name: str
    displacement_max: float     # 最大変位
    displacement_analytical: float | None  # 解析解
    forces: list[BeamForces2D | BeamForces3D]  # 各要素の断面力
    max_bending_stress: float
    max_shear_stress: float
    relative_error: float | None  # 解析解との相対誤差
```

#### 実装項目

- [x] `NumericalTestConfig` / `StaticTestResult` データクラス定義
- [x] 試験別のメッシュ生成・境界条件・荷重条件の自動設定
  - [x] 3点曲げ: 単純支持 + 中央集中荷重
  - [x] 4点曲げ: 単純支持 + 2点対称荷重
  - [x] 引張: 一端固定 + 他端軸方向荷重
  - [x] ねん回: 一端固定 + 他端ねじりモーメント
- [x] 一括実行API: `run_all_tests()`
- [x] 部分実行API: `run_tests()`
- [x] 解析解との自動比較・誤差レポート生成
- [x] pytest マーカーによる試験種別選択実行（`-m bend3p`, `-m tensile`, `-m cosserat` 等）
- [x] 結果のCSV出力（`export_static_csv()`, `export_frequency_response_csv()`）
- [x] **周波数応答試験**（`run_frequency_response()`）— 整合質量行列 + Rayleigh減衰 + FRF
- [x] **Abaqusライクテキスト入力**（`parse_test_input()`）
- [x] **摩擦滑り影響の実用的判定**（`assess_friction_effect()`）

#### 試験の詳細

<details>
<summary>3点曲げ試験</summary>

```
   P
   ↓
   ┌───────────────────┐
   ▲         L         ▲
 支点A    (中央荷重)   支点B

支持条件: 両端ピン支持（ux固定, uy固定, θ自由）
荷重: 中央節点に P（y方向負）

解析解（Timoshenko）:
  δ_mid = PL³/(48EI) + PL/(4κGA)
  V = P/2（中央で符号反転）
  M_max = PL/4（中央）
```

</details>

<details>
<summary>4点曲げ試験</summary>

```
      P           P
      ↓           ↓
   ┌──────────────────┐
   ▲  a    L-2a    a  ▲
 支点A               支点B

支持条件: 両端ピン支持
荷重: 左右対称の2点に P（荷重スパン a）

解析解（Timoshenko）:
  δ_mid = Pa(3L²-4a²)/(48EI) + Pa/(κGA)
  V = P（荷重点間はゼロ）
  M = Pa（荷重点間で一定 = 純曲げ区間）
```

</details>

<details>
<summary>引張試験</summary>

```
  ■────────────────→ P
  固定端             荷重端

支持条件: 一端全拘束（ux, uy, θ / ux, uy, uz, θx, θy, θz）
荷重: 他端に軸方向力 P

解析解:
  δ = PL/(EA)
  N = P（全要素で一定）
```

</details>

<details>
<summary>ねん回試験（3Dのみ）</summary>

```
  ■────────────────⟳ T
  固定端             トルク端

支持条件: 一端全拘束（6DOF）
荷重: 他端にねじりモーメント T

解析解:
  θ = TL/(GJ)
  Mx = T（全要素で一定）
  τ_max = T·r_max/J
```

</details>

#### 依存関係

- Phase 2.1–2.4 の梁要素・断面モデル・ポスト処理が前提
- Phase 3（非線形）完了後、荷重増分対応に拡張予定

---

## Phase 3: 幾何学的非線形 ✓

**目的**: 大変形・大回転に対応する。梁の大たわみ・座屈・スナップスルーを解ける。

### 3.1 非線形ソルバーフレームワーク

- [x] Newton-Raphson法
  - 残差ベクトル R = f_ext - f_int
  - 接線剛性行列 K_T = K_material + K_geometric
  - 収束判定（力・変位・エネルギーノルム）
- [x] 荷重増分法（Load stepping）
- [x] 弧長法（Crisfield cylindrical arc-length）：スナップスルー・座屈追跡用
  - 弧長拘束 ||Δu||² = Δl²、自動弧長推定、ステップサイズ適応（カットバック）
  - リミットポイント通過の検証済み（1-DOF スナップスルー + 線形問題）
- [ ] ラインサーチ（収束加速）

### 3.2 共回転（Corotational）定式化 / Cosserat非線形

空間梁の大回転を扱う。Cosserat rod ベースの幾何学的厳密定式化を主軸とする。

- [x] 接線剛性の幾何学的剛性項 Kg（tangent_stiffness = K_m + K_g）
- [x] 回転のパラメタ化（四元数）
- [x] テスト：大たわみ片持ち梁（幾何剛性効果の定性検証）
- [x] SO(3) 右ヤコビアン J_r(θ) / J_r⁻¹(θ) — `quaternion.py`
- [x] 非線形歪み計算: Γ = R(θ)ᵀ R₀ᵀ r' - e₁, κ = J_r(θ)·θ'
- [x] 非線形内力 f_int = L₀·B_nlᵀ·C·[Γ; κ-κ₀]（1点ガウス求積）
- [x] 非線形接線剛性（内力の中心差分ヤコビアン、対称化）
- [x] `CosseratRod(nonlinear=True)` で非線形モードへディスパッチ
- [x] 有限差分接線検証テスト（13テスト: `test_nonlinear_cosserat.py`）
- [x] テスト：Euler elastica（解析解との定量比較）
  - 端モーメント: π/4, π/2, π, 3π/2, 2π（完全円）— 全パス（< 3%L）
  - 先端荷重: PL²/EI = 1, 2, 5, 10 — elastica 厳密解と一致（< 5%）
- [ ] テスト：Lee's frame等の標準ベンチマーク

### 3.3 共回転定式化（オプション）

Cosserat非線形と別ルートの定式化。必要に応じて実装。

- [ ] 要素ごとのローカルフレーム追従
- [ ] 剛体回転の分離と変形成分の抽出

### 3.4 Updated Lagrangian（**次の優先**）

- [ ] 参照配置の更新
- [ ] Green-Lagrangeひずみ
- [ ] 第二Piola-Kirchhoffストレス

---

## Phase 4: 材料非線形

**目的**: 弾塑性、粘弾性、異方性を梁要素に組み込む。

### 4.1 1次元弾塑性 ✓

梁の軸方向に対する1D弾塑性構成則から着手。

- [x] 1D等方硬化モデル（線形硬化）
- [x] 1D移動硬化モデル（Armstrong-Frederick）
- [x] Return mapping アルゴリズム
- [x] 接線剛性の一貫線形化（consistent tangent）
- [x] 単軸引張・圧縮テスト
- [x] 弾塑性アセンブリ関数（uniform / SRI 対応）
- [x] NR 統合（外部荷重ステップ + 状態管理）
- [x] 検証図（応力-歪み、ヒステリシス、バウシンガー、荷重-変位）

### 4.2 ファイバーモデル ✓

断面をファイバー（微小断面要素）に分割し、各ファイバーに1D構成則を適用。

- [x] ファイバー断面の離散化（FiberSection: 矩形/円形/パイプ）
- [x] 断面力 ← ファイバーの応力積分（N, My, Mz）
- [x] 断面接線剛性 ← ファイバーの接線剛性積分（3×3 サブ行列 [N,My,Mz]×[Γ₁,κ₂,κ₃]）
- [x] アセンブリ関数 `assemble_cosserat_beam_fiber()`（uniform + SRI 対応）
- [x] テスト：弾塑性片持ち梁の荷重変位曲線（36テスト）
- [x] 検証図：モーメント-曲率曲線、片持ち梁荷重-変位曲線

### 4.3 3D弾塑性（von Mises）

3次元応力空間での降伏判定・return mapping を実装し、連続体要素（Q4, TRI3等）に適用。
1D弾塑性（Phase 4.1）を多軸応力状態に拡張する。

- [x] von Mises 降伏関数 f = √(3/2) ||dev(σ)|| − (σ_y + R)
- [x] 3D return mapping アルゴリズム（radial return）
- [x] 3D consistent tangent（弾塑性接線剛性テンソル、Simo & Taylor (1985)）
- [x] 等方硬化（線形・Voce）/ 移動硬化（Armstrong-Frederick）
- [x] PlasticState3D 状態変数（塑性ひずみテンソル ε^p, 等価塑性ひずみ α, 背応力テンソル β）
- [x] 平面ひずみ要素（Q4, TRI3, Q4_EAS）との統合
- [ ] 単軸・二軸・純せん断テスト（解析解比較）— テスト計画策定済み（[status-025](status/status-025.md)）、45テスト + 3検証図
- [ ] パッチテスト（塑性域での要素性能検証）
- [ ] 検証図（降伏曲面、応力パス、荷重-変位曲線）

#### 参考文献

- Simo, J.C. & Taylor, R.L. (1985) "Consistent tangent operators for rate-independent elastoplasticity"
- de Souza Neto et al. "Computational Methods for Plasticity" Ch.7–8
- Simo, J.C. & Hughes, T.J.R. "Computational Inelasticity" Ch.3–4

### 4.4 構造減衰

- [x] Rayleigh減衰（C = αM + βK）— 周波数応答用として Phase 2.6 で実装済み
- [ ] ヒステリシス減衰（履歴変数ベース）
- [ ] 構成則レベルの粘性項

### 4.5 粘弾性

- [ ] 一般化Maxwell（Prony級数）モデル
- [ ] 再帰的更新アルゴリズム（計算効率）
- [ ] 緩和試験・クリープ試験シミュレーション

### 4.6 異方性

- [ ] 異方性弾性（断面方向依存の剛性）
- [ ] 梁の曲げ-ねじり連成（偏心、非対称断面）

---

## Phase 5: 動的解析

**目的**: 揺動試験等の過渡応答解析を可能にする。

**注**: 周波数応答解析の基盤（整合質量行列・Rayleigh減衰・FRF）は Phase 2.6 で実装済み。
ここでは時間領域の過渡応答解析を追加する。

### 5.1 時間積分スキーマ

- [x] Newmark-β法（暗黙的）— `dynamics.py`, 平均加速度法デフォルト, LU事前分解
- [x] α法（HHT法、数値減衰付き）— α ∈ [-1/3, 0], β/γ自動連動
- [ ] 陽解法（Central Difference、オプション）

### 5.2 質量行列

- [x] 整合質量行列（consistent mass）— 2D/3D梁、Phase 2.6 で実装済み
- [x] 集中質量行列（lumped mass）— HRZ法, 2D/3D梁, 回転DOF非特異
- [ ] ElementProtocol への `mass_matrix()` 統合

### 5.3 減衰行列

- [x] Rayleigh減衰（C = αM + βK）— Phase 2.6 で実装済み
- [ ] モーダル減衰

### 5.4 非線形動解析（**次の優先**）

Newton-Raphson + Newmark-β による非線形過渡応答解析。
幾何学非線形（UL, Phase 3.4）と組み合わせて大変形動的問題を解く。

- [ ] 非線形 Newmark-β（Newton-Raphson 反復付き暗黙的時間積分）
- [ ] 数値三点曲げ試験の非線形動解析対応（`numerical_tests` フレームワーク拡張）
- [ ] エネルギー保存性検証テスト

---

## Phase C: 梁–梁接触モジュール

**目的**: 撚線モデル（Phase 4.7）の前提となる梁–梁接触を実装する。
**着手条件**: Phase 3 完了後。Phase 4/5 と並行して進めることも可。

設計仕様: [梁–梁接触モジュール 設計仕様書 v0.1](contact/beam_beam_contact_spec_v0.1.md)

- [x] 設計仕様書 v0.1（AL + Active-set + return mapping + Outer/Inner分離）
- [ ] C0: ContactPair/ContactState と solver_hooks の骨格実装
- [ ] C1: segment-to-segment 最近接 + broadphase（AABB格子）
- [ ] C2: 法線AL + Active-setヒステリシス + 主項接線
- [ ] C3: 摩擦return mapping + μランプ
- [ ] C4: merit line search + 探索/求解分離の運用強化

---

## Phase 4.7: 撚線モデル

**目的**: 撚線の曲げ・ねじり連成挙動（ヒステリシス・疲労）を再現する。
離散素線モデルでの接触観測を経て、縮約（homogenization/ROM）可否を判断する。

**前提**: Phase 2.5（Cosserat rod）✓ + Phase 3（非線形）+ Phase C（接触基盤）

### アプローチ

最小構成の接触実験モデル（30〜100素線、1ピッチ）で観測可能量を取り、
縮約対象を同定してから理論を固定する。詳細設計は別文書に分離。

### 設計判断（確定事項）

| 項目 | 判断 | 備考 |
|------|------|------|
| **第一ターゲット用途** | 曲げ＋ねじり連成曲げ・疲労 | 接触観測後に θ_i 縮約可否を判断 |
| **被膜モデルの目的** | 剛性寄与＋摩擦制御 | 理想化弾性体。温度・損傷はスコープ外 |
| **接触ペア戦略** | 隣接ペアのみ＋定期更新 | 実験的アプローチ |

### 段階的実装計画

**Level 0: 基礎同定用**
- [ ] 素線は軸方向のみ（曲げ剛性無視）
- [ ] 接触は法線方向 penalty、摩擦は正則化 Coulomb
- [ ] テスト: 引張・ねじり・曲げの基本ヒステリシス

**Level 1: 撚り解き**
- [ ] `θ_i(s)` を未知量化（ヘリックス拘束を解く）
- [ ] 被膜を「周方向せん断ばね＋圧縮ばね」で平均化
- [ ] テスト: 撚り戻り＋摩擦散逸

**Level 2: 素線曲げ・局所座屈**
- [ ] 素線を Cosserat rod 化（曲げ・ねじり含む）
- [ ] 接触ペア爆増の対策（近接のみ、代表接触、連続平均化）
- [ ] テスト: 局所座屈モードの再現

### 参考文献

- Costello, G.A. "Theory of Wire Rope"
- Cardou, A. & Jolicoeur, C. (1997) "Mechanical Models of Helical Strands"
- Jiang, W.G. et al. (2006) "Statically indeterminate contacts in axially loaded wire strand"
- Foti, F. & Martinelli, L. (2016) "An analytical approach to model the hysteretic bending behavior of spiral strands"

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

### 8.1 ミクロ-マクロ連成

- 微小クラックの変形追跡
- 計算ホモジナイゼーション（FE²）

### 8.2 連続体要素の拡張

- 既存2D要素の平面応力対応
- 3D固体要素（Hex8, Tet4, Tet10）
- シェル要素

---

## 優先度と依存関係

```
Phase 1 (アーキテクチャ) ✓
    ↓
Phase 2 (梁要素) ✓
    ├── 2.1-2.4: EB/Timoshenko/断面 ✓
    ├── 2.5: Cosserat rod ✓
    └── 2.6: 数値試験 ✓ (整合質量/Rayleigh減衰/FRF含む)
        ↓
Phase 3 (幾何学的非線形) ✓
    ↓
    ├── Phase 4 (材料非線形: 4.1〜4.6)
    ├── Phase 5 (動的解析: 時間積分追加)
    └── Phase C (梁–梁接触) ← Phase 3完了後に着手可
            ↓
        Phase 4.7 (撚線モデル) ← Phase C + Cosserat rod + 非線形基盤
            ↓
Phase 6 (NNサロゲート) ← Phase 4の構成則設計に依存
    ↓
Phase 7 (レジストリ/フィッティング) ← Phase 2-6の全要素が必要
    ↓
Phase 8 (応用展開) ← 必要に応じて
```

**クリティカルパス**: Phase 3 → Phase C → Phase 4.7

**次の優先（上から順）**:
1. **幾何学非線形 Updated Lagrangian (UL)** — Phase 3.4 の UL 定式化。参照配置の更新、Green-Lagrange ひずみ、第二 Piola-Kirchhoff 応力
2. **非線形動解析** — Phase 5 拡張。Newton-Raphson + Newmark-β による非線形過渡応答
3. **数値三点曲げ試験の非線形動解析対応** — 数値試験フレームワークの非線形動解析への拡張

**並行開発可能**:
- Phase 4（4.1〜4.6）と Phase 5 は Phase 3 完了後に並行可
- Phase C は Phase 3 完了後、Phase 4/5 と並行して着手可
- Phase 6 は Phase 4 の構成則設計と並行可

---

## 設計原則

1. **モジュール合成可能性**: 要素・構成則・ソルバー・積分スキーマを自由に組み合わせ可能
2. **Protocol/ABCベース**: 具象クラスへの依存を避け、インタフェースに依存
3. **状態変数の明示管理**: 履歴変数は `StateVariable` で明示的に管理、暗黙のグローバル状態を排除
4. **テスト駆動**: 各要素・構成則は解析解またはリファレンスソルバーとの比較テスト必須
5. **NN互換設計**: 構成則インタフェースは最初からNN代替を考慮した入出力設計
6. **段階的拡張**: 既存のテストを破壊しないよう、後方互換性を保ちながら拡張

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
- Cardou, A. & Jolicoeur, C. (1997) "Mechanical Models of Helical Strands"（撚線モデル）
- Foti, F. & Martinelli, L. (2016) "Hysteretic bending of spiral strands"（撚線曲げヒステリシス）
