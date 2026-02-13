# Cosserat Rod 設計仕様書

[← README](../README.md) | [ロードマップ](roadmap.md)

## 概要

Cosserat rod（幾何学的厳密梁）は、中心線の変形 r(s) と断面回転 R(s) を独立に記述する
梁定式化であり、Timoshenko 梁の一般化に相当する。
撚線モデル（Phase 4.6）において素線の大回転・ねじり・接触フレーム管理の基盤となる。

### なぜ Cosserat rod が必要か

| 比較項目 | Timoshenko + Updated Lagrangian | Cosserat rod（群上更新） |
|---------|-------------------------------|------------------------|
| 回転更新 | 要素依存・実装でブレやすい | 四元数で統一的に管理 |
| ねじり＋大回転の混合 | 接触と結合した瞬間に仮想仕事に誤差 | body/spatial の整理がしやすい |
| 摩擦散逸の増分管理 | 「要素の流儀」依存 | 増分ポテンシャル（min問題）と相性良 |
| 接触の接線方向 | push/pull-back が実装依存 | 最初から群上で定義 → 事故りにくい |

**判断基準**: 摩擦ヒステリシスや stick/slip 境界の予測精度が要件なら Cosserat を推奨。
平均剛性を見るだけなら Timoshenko + UL で十分。

---

## 数学的定式化

### 配位空間

```
中心線:     r(s) ∈ R³        — 梁軸の空間曲線
断面回転:   q(s) ∈ S³ ⊂ R⁴   — 単位四元数で表現
回転行列:   R(s) = R(q(s))   — q から構成される SO(3) の元
```

### 四元数の規約

```
q = [w, x, y, z] = w + x·i + y·j + z·k
||q|| = 1 （単位四元数制約）

回転軸 n、回転角 θ のとき:
  q = [cos(θ/2), sin(θ/2)·n]

ベクトル回転:
  v' = q ⊗ (0,v) ⊗ q* = R(q)·v
```

### 一般化歪み

```
力歪み:     Γ = R(q)ᵀ r' - e₁  ∈ R³
モーメント歪み: κ = 2·Im(q* ⊗ q')  ∈ R³
```

ここで e₁ = [1, 0, 0] は参照接線方向（直線梁の参照配位）。

| 成分 | 物理的意味 |
|------|-----------|
| Γ₁ | 軸伸び歪み (extension) |
| Γ₂ | y方向せん断歪み |
| Γ₃ | z方向せん断歪み |
| κ₁ | ねじり (twist) |
| κ₂ | y軸まわり曲率 (xz面曲げ) |
| κ₃ | z軸まわり曲率 (xy面曲げ) |

### 線形化

小変形近似（R ≈ I + skew(θ)）のもとで:

```
Γ₁ ≈ u₁'              — 軸伸び
Γ₂ ≈ u₂' - θ₃         — y方向せん断（Timoshenko）
Γ₃ ≈ u₃' + θ₂         — z方向せん断（Timoshenko）
κ₁ ≈ θ₁'              — ねじり
κ₂ ≈ θ₂'              — y軸曲率
κ₃ ≈ θ₃'              — z軸曲率
```

→ 線形化時は Timoshenko 3D と同じ物理を記述する。

### 弾性構成則

```
n = C_Γ · Γ = diag(EA, κy·GA, κz·GA) · Γ
m = C_κ · κ = diag(GJ, EIy, EIz) · κ
```

統一構成行列 C (6×6):
```
C = diag(EA, κy·GA, κz·GA, GJ, EIy, EIz)
```

---

## 有限要素離散化

### 要素タイプ

- 2節点線形要素
- 各節点 6 DOF: (ux, uy, uz, θx, θy, θz) — 線形化版
- 内部状態: 各節点の参照四元数 q₀（Phase 3 で更新される）

### 形状関数

```
N₁(ξ) = 1 - ξ,  N₂(ξ) = ξ    (ξ ∈ [0,1])
dN₁/ds = -1/L,   dN₂/ds = 1/L
```

変位・回転とも線形補間（等次補間）。

### 歪み-変位行列 B

```
e = [Γ₁, Γ₂, Γ₃, κ₁, κ₂, κ₃]ᵀ = B · u

u = [u₁₁,u₂₁,u₃₁,θ₁₁,θ₂₁,θ₃₁, u₁₂,u₂₂,u₃₂,θ₁₂,θ₂₂,θ₃₂]ᵀ
```

B 行列 (6×12) at ξ:
```
     u₁₁   u₂₁   u₃₁   θ₁₁   θ₂₁   θ₃₁   u₁₂   u₂₂   u₃₂   θ₁₂   θ₂₂   θ₃₂
Γ₁  [dN₁    0      0      0      0      0    dN₂    0      0      0      0      0  ]
Γ₂  [ 0    dN₁     0      0      0    -N₁    0    dN₂     0      0      0    -N₂  ]
Γ₃  [ 0      0    dN₁     0     N₁     0     0      0    dN₂     0     N₂     0   ]
κ₁  [ 0      0      0    dN₁     0      0     0      0      0    dN₂     0      0  ]
κ₂  [ 0      0      0      0    dN₁     0     0      0      0      0    dN₂     0  ]
κ₃  [ 0      0      0      0      0    dN₁    0      0      0      0      0    dN₂ ]
```

### ガウス求積

```
Ke = ∫₀ᴸ Bᵀ · C · B ds ≈ Σ wᵢ · L · B(ξᵢ)ᵀ · C · B(ξᵢ)
```

| 積分点数 | 点（[0,1]区間） | 重み | 特性 |
|---------|----------------|------|------|
| **1点（標準）** | ξ = 0.5 | w = 1.0 | せん断ロッキング回避。軸・ねじり・曲げ項は厳密 |
| 2点 | ξ = 0.5 ± 1/(2√3) | w = 0.5 | せん断項も厳密だがロッキングの恐れ |

**推奨**: 1点ガウス求積（文献標準、幾何学的厳密梁の定番）。

### Timoshenko 3D との違い

| 項目 | Timoshenko 3D (beam_timo3d) | Cosserat rod (beam_cosserat) |
|------|---------------------------|------------------------------|
| 定式化 | 解析的剛性行列（Φ補正） | B行列 + ガウス求積 |
| 回転表現 | 回転行列（座標変換用） | 四元数（内部状態として保持） |
| 1要素曲げ精度 | 高い（解析ベース） | 低い（線形補間、要メッシュ細分割） |
| 非線形拡張 | 大幅な再設計が必要 | 歪み計算の非線形化のみで拡張可能 |
| 用途 | 線形解析メイン | 大回転・接触・撚線の基盤 |

---

## コード構成

### ファイル一覧

| ファイル | 概要 |
|---------|------|
| `xkep_cae/math/__init__.py` | 数学パッケージ |
| `xkep_cae/math/quaternion.py` | 四元数演算（15関数） |
| `xkep_cae/elements/beam_cosserat.py` | Cosserat rod 要素 |
| `tests/test_quaternion.py` | 四元数テスト（37テスト） |
| `tests/test_beam_cosserat.py` | Cosserat rod テスト（36テスト） |

### 四元数演算モジュール (`quaternion.py`)

| 関数 | 説明 |
|------|------|
| `quat_identity()` | 恒等四元数 [1,0,0,0] |
| `quat_multiply(p, q)` | Hamilton積 p ⊗ q |
| `quat_conjugate(q)` | 共役 q* = [w,-x,-y,-z] |
| `quat_norm(q)` | ノルム |
| `quat_normalize(q)` | 単位四元数に正規化 |
| `quat_rotate_vector(q, v)` | ベクトル回転 v' = R(q)·v |
| `quat_to_rotation_matrix(q)` | q → R (3×3) |
| `rotation_matrix_to_quat(R)` | R → q（Shepperd法） |
| `quat_from_axis_angle(axis, angle)` | 軸-角 → q |
| `quat_from_rotvec(rotvec)` | 回転ベクトル → q（指数写像） |
| `quat_to_rotvec(q)` | q → 回転ベクトル（対数写像） |
| `quat_slerp(q0, q1, t)` | 球面線形補間 |
| `quat_angular_velocity(q, q_dot)` | 角速度（body frame） |
| `quat_material_curvature(q, q_prime)` | 物質曲率 κ |
| `skew(v)` / `axial(S)` | hat/vee 写像 |

### Cosserat rod 要素 (`beam_cosserat.py`)

| クラス/関数 | 説明 |
|------------|------|
| `CosseratStrains` | 一般化歪み (Γ, κ) データクラス |
| `CosseratForces` | 一般化断面力 (N,Vy,Vz,Mx,My,Mz) データクラス |
| `CosseratRod` | ElementProtocol 適合クラス |
| `cosserat_ke_local()` | 局所剛性行列 (12×12) |
| `cosserat_ke_global()` | 全体剛性行列 |
| `cosserat_section_forces()` | 断面力計算 |
| `cosserat_generalized_strains()` | 一般化歪み計算 |

---

## 検証結果

### テスト結果サマリ

**全314テストパス**（四元数37 + Cosserat 36 + 既存241）

### 軸力・ねじり（1要素厳密）

| テスト | 解析解 | 1要素結果 | 相対誤差 |
|--------|--------|----------|---------|
| 軸引張 δ=PL/(EA) | 厳密 | 厳密 | < 1e-12 |
| ねじり θ=TL/(GJ) | 厳密 | 厳密 | < 1e-12 |

### 曲げ（メッシュ収束）

片持ち梁 y方向曲げ: δ = PL³/(3EI) + PL/(κGA)

| 要素数 | 相対誤差 |
|--------|---------|
| 2 | ~25% |
| 4 | ~7% |
| 8 | ~1.7% |
| 16 | ~0.4% |
| 32 | < 0.1% |

→ 2次収束。32要素以上で工学的に十分な精度。

### 3点曲げ（単純支持）

δ_mid = PL³/(48EI) + PL/(4κGA)

| 要素数 | 相対誤差 |
|--------|---------|
| 4 | ~6% |
| 8 | ~1.6% |
| 16 | ~0.4% |
| 32 | ~0.1% |
| 64 | < 0.03% |

---

## Phase 3 への拡張方針

### 非線形化の手順

1. **四元数状態の更新**: DOFは増分回転ベクトル Δθ のまま
   ```
   q_{n+1} = quat_from_rotvec(Δθ) ⊗ q_n
   ```

2. **歪みの非線形化**: B行列の再計算
   ```
   Γ = R(q)ᵀ · r' - e₁   （線形化なし）
   κ = 2·Im(q* ⊗ q')      （線形化なし）
   ```

3. **接線剛性**: 材料剛性 + 幾何剛性
   ```
   K_T = K_material + K_geometric
   K_geometric: 回転-力のカップリング項
   ```

4. **Newton-Raphson**: 残差 R = f_ext - f_int の反復

### 必要な追加実装

- `internal_force()`: 内力ベクトル f_int（非線形）
- `geometric_stiffness()`: 幾何剛性行列 K_geo
- 四元数の増分更新ロジック
- 収束判定（力・変位・エネルギーノルム）

---

## Phase 4.6 撚線モデルとの接続

Cosserat rod は撚線モデルの基盤として以下の役割を果たす:

1. **外側の梁**: 撚線全体を1本の Cosserat rod として記述 → (r, q)
2. **個別素線**: 各素線も Cosserat rod → 曲げ剛性を持つ（Level 2）
3. **接触のフレーム**: 接触の法線・接線方向が body frame で自然に定義される
4. **摩擦散逸**: body/spatial の push/pull-back が四元数で整理される

### 接続点

```
素線位置:  x_i(s) = r(s) + R(q(s)) · ρ_i(θ_i(s))
素線歪み:  (ν_i, κ_i) = G_i(Γ, κ, θ_i, θ_i', ...)
合力:      n(s) = Σ_i R · n_i^local + 接触反力の合力
合モーメント: m(s) = Σ_i [R · m_i^local + (x_i - r) × (素線力)] + 接触偶力
```

---

## 参考文献

1. Simo, J.C. (1985) "A finite strain beam formulation — Part I", CMAME
2. Crisfield, M.A. & Jelenić, G. (1999) "Objectivity of strain measures in the geometrically exact beam", IJNME
3. Antman, S.S. "Nonlinear Problems of Elasticity", Springer
4. Romero, I. (2004) "The interpolation of rotations and its application to finite element models of geometrically exact rods", Comp. Mech.
