# 完了済み Phase 詳細設計アーカイブ

[← README](../../README.md) | [← roadmap](../roadmap.md)

> 本文書は roadmap.md から分割した、完了済みPhaseの詳細設計情報です。
> 各Phaseの概要・チェックボックスは roadmap.md を参照してください。

---

## Phase 1: アーキテクチャ再構成 ✓

### 1.2 コア抽象レイヤー設計

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

---

## Phase 2: 空間梁要素 ✓

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

### 2.5 Cosserat rod（幾何学的厳密梁）✓

#### 幾何の記述

```
中心線:     r(s)          ∈ R³
断面回転:   R(s)          ∈ SO(3)
一般化歪み: ν = Rᵀr'      （せん断＋軸伸び）
            κ = axial(RᵀR') （曲率＋ねじり）
```

#### 参考文献

- Simo, J.C. (1985) "A finite strain beam formulation — Part I"
- Crisfield, M.A. & Jelenić, G. (1999) "Objectivity of strain measures in the geometrically exact beam"
- Antman, S.S. "Nonlinear Problems of Elasticity"

### 2.6 数値試験フレームワーク ✓

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

---

## Phase 3: 幾何学的非線形 ✓

### 3.2 Cosserat 非線形の詳細

- SO(3) 右ヤコビアン J_r(θ) / J_r⁻¹(θ)
- 非線形歪み計算: Γ = R(θ)ᵀ R₀ᵀ r' - e₁, κ = J_r(θ)·θ'
- 非線形内力 f_int = L₀·B_nlᵀ·C·[Γ; κ-κ₀]（1点ガウス求積）
- 非線形接線剛性（内力の中心差分ヤコビアン、対称化）
- `CosseratRod(nonlinear=True)` で非線形モードへディスパッチ

### 3.3 共回転（Corotational）定式化 ✓

- 要素ごとのローカルフレーム追従（corotated フレーム構築）
- 剛体回転の分離と変形成分の抽出（`R_def = R_cr @ R_node @ R_0^T`）
- CR内力ベクトル `timo_beam3d_cr_internal_force()`
- 数値微分接線剛性 `timo_beam3d_cr_tangent()`（中心差分, eps=1e-7）
- グローバルアセンブリ `assemble_cr_beam3d()`
- dynamic_runner 統合（`nlgeom=True` + `beam_type="timo3d"`）

### 3.4 Total/Updated Lagrangian ✓

- Green-Lagrangeひずみ（Q4要素, 2×2ガウス求積）
- 第二Piola-Kirchhoffストレス（Saint-Venant Kirchhoff材料）
- 変形勾配 F = I + ∂u/∂X、線形化B行列 B_L = B_0 + B_NL
- 幾何剛性行列 K_geo、材料剛性行列 K_mat
- Updated Lagrangian（ULAssemblerQ4）— ガウス点Cauchy応力追跡, S→σプッシュフォワード

---

## Phase 4: 材料非線形（完了済み部分）

### 4.3 3D弾塑性（von Mises）— **凍結**

> **凍結理由**: 実装コードは完了済みだが、テスト（45テスト計画）と検証図の作成は優先度を下げ凍結とする。テスト計画は [status-025](../status/status-025.md) を参照。

- von Mises 降伏関数 f = √(3/2) ||dev(σ)|| − (σ_y + R)
- 3D return mapping アルゴリズム（radial return）
- 3D consistent tangent（Simo & Taylor (1985)）
- 等方硬化（線形・Voce）/ 移動硬化（Armstrong-Frederick）
- PlasticState3D 状態変数（塑性ひずみテンソル ε^p, 等価塑性ひずみ α, 背応力テンソル β）
- 平面ひずみ要素（Q4, TRI3, Q4_EAS）との統合

---

## Phase C: 梁–梁接触モジュール ✓

設計仕様: [梁–梁接触モジュール 設計仕様書 v0.1](../contact/beam_beam_contact_spec_v0.1.md)

### 詳細実装内容

- **C0**: ContactPair/ContactState/ContactManager + geometry（closest_point_segments, compute_gap, build_contact_frame）— 30テスト
- **C1**: broadphase（AABB格子）+ ContactManager幾何更新（detect_candidates/update_geometry）+ Active-setヒステリシス — 31テスト
- **C2**: 法線AL + 接触接線剛性（主項）+ 接触付きNRソルバー（Outer/Inner分離）— 43テスト
- **C3**: 摩擦return mapping + μランプ（27テスト）
- **C4**: merit line search + 探索/求解分離の運用強化（26テスト）
- **C5**: 幾何微分込み一貫接線 + slip consistent tangent + PDAS + 平行輸送フレーム（35テスト）

---

## Phase 5: 動的解析 ✓

### 詳細実装内容

- **Newmark-β法**: 暗黙的、平均加速度法デフォルト, LU事前分解
- **HHT-α法**: 数値減衰付き、α ∈ [-1/3, 0], β/γ自動連動
- **陽解法（Central Difference）**: 対角質量行列高速パス, 安定性監視
- **集中質量行列（HRZ法）**: 2D/3D梁, 回転DOF非特異
- **モーダル減衰**: `build_modal_damping_matrix()`, 一般化固有値問題ベース
- **非線形動解析**: Newton-Raphson + Newmark-β/HHT-α、エネルギー保存性検証済み

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
