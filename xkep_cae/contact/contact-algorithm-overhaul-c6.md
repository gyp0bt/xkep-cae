# 接触アルゴリズム根本整理 — Phase C6 設計仕様

[← README](../../README.md) | [← roadmap](../roadmap.md) | [← status-076](../status/status-076.md)

**日付**: 2026-02-27
**作成者**: Claude Code
**位置づけ**: ML（GNN/k_pen推定）に先立つ理論基盤の整備

---

## 1. 現状の問題構造

### 1.1 純ペナルティ退行の経緯

status-065 の14段階診断の結果、7本撚り（36+ペア同時活性化）の収束は以下の設定でのみ達成:

```
n_outer_max = 1        # Outer loop を廃止
al_relaxation = 0.01   # λ_n 更新を 1% のみ（≈ゼロ）
penalty_growth_factor = 1.0  # k_pen 成長なし
```

**これは AL 法の利点を完全に放棄した純ペナルティ法への退行である。**

結果として:
- 貫入率 15–16% が恒久化（接触半径比）
- k_pen を上げれば条件数悪化 → NR 発散
- k_pen を下げれば貫入増大 → 力学的精度低下
- adaptive omega は改善策だが根本解決ではない

### 1.2 根本原因の分解

| 層 | 問題 | 現在の対処 | 限界 |
|----|------|-----------|------|
| **L1: 接触離散化** | Point-to-point (s,t 固定) → Outer/Inner 分離が必須 | Outer loop | 多ペアで Outer 発散 |
| **L2: AL 乗数更新** | λ_n 蓄積 → Inner Jacobian との不整合 | omega 緩和 | omega → 0 で純ペナルティ |
| **L3: 接線剛性** | ∂(s,t)/∂u 未実装 → monolithic 幾何更新が使えない | s,t 固定 | 大変形で Outer 多反復 |
| **L4: 線形ソルバー** | k_pen 大 → 条件数悪化 | GMRES+ILU fallback | ILU の fill-in 制御が粗い |
| **L5: Active set** | チャタリング（48↔42↔36ペア変動） | freeze + no_deactivation | 過制約のリスク |

**ML で L1–L5 を回避する（k_pen を賢く推定する）のは対症療法に過ぎない。**

### 1.3 この設計の方針

L1–L5 を段階的に理論ベースで改善し、ML は「仕上げの最適化」に留める。

---

## 2. 改善ロードマップ（Phase C6 レベル分割）

```
C6-L1: Segment-to-segment Gauss 積分（Line-to-line 接触）
C6-L2: 一貫接線の完全化（∂s/∂u, ∂t/∂u Jacobian）
C6-L3: Semi-smooth Newton + NCP 関数（Outer loop 廃止）
C6-L4: ブロック前処理強化（接触 Schur 補集合）
C6-L5: Mortar 離散化（セグメント対セグメント拘束）
```

優先順は **C6-L1 → L2 → L3** → L4 → L5。L1–L3 が最もインパクトが大きく、L4–L5 は大規模化時に必要。

---

## 3. C6-L1: Segment-to-segment Gauss 積分

### 3.1 現状

**Point-to-point (PtP) 接触**: 各セグメントペアに対して1点の最近接点 (s*, t*) のみで接触力を評価。

```python
# 現在の assembly.py
g_n = _contact_shape_vector(pair)    # s,t 固定の 1 点
K_n = k_pen * g_n @ g_n.T           # rank-1 行列
```

**問題点**:
- 平行に近い梁（撚線の同層梁）では1点評価が不正確
- 接触力の空間分布を捉えられない → 局所的な過大貫入
- セグメント長に対する接触力の積分精度が低い

### 3.2 改善: Line-to-line 接触の Gauss 積分

**Meier et al. (2016) "A unified approach for beam-to-beam contact"** に基づく。

```
セグメント A: p(s) = xA0 + s·(xA1 - xA0),  s ∈ [0,1]
セグメント B: q(t) = xB0 + t·(xB1 - xB0),  t ∈ [0,1]

接触力の線積分:
  f_c = ∫₀¹ p_n(s)·n(s)·N_A(s) ds  （A 側セグメントに沿って積分）

Gauss 積分:
  f_c ≈ Σᵢ w_i · p_n(sᵢ)·n(sᵢ)·N_A(sᵢ) · L_A
```

ここで:
- sᵢ: Gauss 点（i = 1, ..., n_gp）
- w_i: Gauss 重み
- p_n(sᵢ): 各 Gauss 点での法線接触圧
- n(sᵢ): 各 Gauss 点での法線方向
- N_A(sᵢ): 形状関数値
- L_A: セグメント長

### 3.3 実装設計

```python
# 新規: contact/line_contact.py

def evaluate_line_contact_force(
    pair: ContactPair,
    n_gauss: int = 3,
) -> tuple[np.ndarray, float]:
    """Line-to-line 接触力を Gauss 積分で評価.

    Args:
        pair: 接触ペア（4節点）
        n_gauss: Gauss 積分点数（2–5）

    Returns:
        f_local: (12,) 局所接触力ベクトル
        total_p_n: 合計法線力（診断用）
    """
    gp, gw = gauss_legendre_01(n_gauss)  # [0,1] 区間
    f_local = np.zeros(12)
    total_p_n = 0.0

    for s_gp, w in zip(gp, gw):
        # A 側の Gauss 点位置
        pA = (1 - s_gp) * xA0 + s_gp * xA1

        # pA から B セグメントへの最近接点
        t_closest = project_point_to_segment(pA, xB0, xB1)
        pB = (1 - t_closest) * xB0 + t_closest * xB1

        # ギャップと法線
        d = pA - pB
        dist = np.linalg.norm(d)
        gap = dist - (r_A + r_B)
        normal = d / max(dist, 1e-30)

        # 法線力（AL）
        p_n = max(0.0, lambda_n_at_gp + k_pen * (-gap))

        # 形状ベクトル（s_gp, t_closest で評価）
        g_n = build_shape_vector(s_gp, t_closest, normal)

        # 重み付き加算
        f_local += w * p_n * g_n * L_A
        total_p_n += w * p_n * L_A

    return f_local, total_p_n
```

### 3.4 接線剛性の Gauss 積分

```python
def compute_line_contact_stiffness(
    pair: ContactPair,
    n_gauss: int = 3,
) -> np.ndarray:
    """Line-to-line 接触剛性を Gauss 積分で評価.

    K_c = ∫₀¹ k_eff(s)·g_n(s)·g_n(s)ᵀ ds · L_A
        + ∫₀¹ K_geo(s) ds · L_A
    """
    gp, gw = gauss_legendre_01(n_gauss)
    K_local = np.zeros((12, 12))

    for s_gp, w in zip(gp, gw):
        # 各 Gauss 点で k_eff, g_n, K_geo を評価
        k_eff = compute_k_eff_at_gp(pair, s_gp)
        g_n = build_shape_vector_at_gp(pair, s_gp)
        K_geo = compute_geometric_stiffness_at_gp(pair, s_gp)

        K_local += w * (k_eff * np.outer(g_n, g_n) + K_geo) * L_A

    return K_local
```

### 3.5 Gauss 点数の選択

| 接触角度 θ | 推奨 n_gp | 理由 |
|-----------|----------|------|
| θ > 30° (大角度) | 2 | PtP と同等精度、追加コスト最小 |
| 10° < θ < 30° | 3 | 中間領域、標準 |
| θ < 10° (準平行) | 4–5 | 長い接触帯、高精度必須 |

**自動選択**: cos_angle > 0.985 (θ < 10°) で n_gp を増やす。

### 3.6 互換性と段階的移行

- `ContactConfig.line_contact: bool = False` で ON/OFF
- `line_contact = False` (デフォルト) では既存の PtP 動作を完全維持
- テスト: 既存の PtP テスト全パス + 新規 line contact テスト追加

---

## 4. C6-L2: 一貫接線の完全化

### 4.1 現状

∂(s,t)/∂u の Jacobian が未実装のため、Inner NR 中に (s,t) を更新できない。
現在は Outer loop で (s,t) を更新し、Inner では固定。

### 4.2 ∂(s,t)/∂u の導出

最近接点条件（非拘束の場合）:

```
F₁(s, t, u) = (pA(s) - pB(t)) · dA/ds = 0
F₂(s, t, u) = (pA(s) - pB(t)) · dB/dt = 0

ここで:
  dA/ds = xA1 - xA0 + ∂uA/∂s  (変形後の接線)
  dB/dt = xB1 - xB0 + ∂uB/∂t  (同)
```

暗関数の定理:

```
∂F/∂(s,t) · d(s,t)/du = -∂F/∂u

[∂F₁/∂s  ∂F₁/∂t] [ds/du]   [∂F₁/∂u]
[∂F₂/∂s  ∂F₂/∂t] [dt/du] = -[∂F₂/∂u]
```

J = ∂F/∂(s,t) は 2×2 行列（最近接点の Hessian に相当）。
∂F/∂u は 2×12 行列（4節点 × 3DOF）。

```python
def compute_st_jacobian(pair: ContactPair) -> tuple[np.ndarray, np.ndarray]:
    """∂(s,t)/∂u を陰関数の定理で計算.

    Returns:
        ds_du: (12,) ∂s/∂u
        dt_du: (12,) ∂t/∂u
    """
    s, t = pair.state.s, pair.state.t
    xA0, xA1 = coords[pair.nodes_a[0]], coords[pair.nodes_a[1]]
    xB0, xB1 = coords[pair.nodes_b[0]], coords[pair.nodes_b[1]]

    dA = xA1 - xA0  # セグメント A の方向
    dB = xB1 - xB0  # セグメント B の方向
    delta = (1-s)*xA0 + s*xA1 - ((1-t)*xB0 + t*xB1)  # pA - pB

    # J = ∂F/∂(s,t)
    J = np.array([
        [dA @ dA,  -dA @ dB],
        [-dA @ dB,  dB @ dB],
    ])
    # 曲率項（直線セグメントでは ∂²p/∂s² = 0 のため不要）
    # 高次要素の場合はここに追加

    # ∂F/∂u = [∂F₁/∂uA0, ∂F₁/∂uA1, ∂F₁/∂uB0, ∂F₁/∂uB1]  (2×12)
    dF_du = np.zeros((2, 12))
    # ∂F₁/∂uA0 = (1-s)·dA,  ∂F₁/∂uA1 = s·dA, ...
    dF_du[0, 0:3] = (1-s) * dA
    dF_du[0, 3:6] = s * dA
    dF_du[0, 6:9] = -(1-t) * dA
    dF_du[0, 9:12] = -t * dA
    dF_du[1, 0:3] = -(1-s) * dB
    dF_du[1, 3:6] = -s * dB
    dF_du[1, 6:9] = (1-t) * dB
    dF_du[1, 9:12] = t * dB

    # 暗関数の定理: d(s,t)/du = -J⁻¹ · dF_du
    if abs(np.linalg.det(J)) < 1e-20:
        # 平行ケース: ds/du = dt/du = 0（s,t 固定にフォールバック）
        return np.zeros(12), np.zeros(12)

    J_inv = np.linalg.inv(J)
    dst_du = -J_inv @ dF_du  # (2, 12)
    return dst_du[0], dst_du[1]
```

### 4.3 接線剛性への反映

```python
# 完全な接触接線剛性（∂f_c/∂u の全項）
K_c = k_eff · g_n · g_nᵀ           # 主項（既存）
    + p_n · ∂g_n/∂u                 # 法線回転 + 幾何（既存: K_geo）
    + ∂p_n/∂(s,t) · (ds/du, dt/du) · g_nᵀ  # 接触点移動に伴う法線力変化（新規）
    + p_n · ∂g_n/∂(s,t) · (ds/du, dt/du)    # 接触点移動に伴う形状変化（新規）
```

最後の2項が欠落していたため、monolithic 幾何更新で NR が発散していた。

### 4.4 期待効果

- **Outer loop の廃止可能性**: (s,t) を Inner 内で更新できれば、Outer/Inner 分離が不要に
- **二次収束の回復**: 完全一貫接線により Newton の二次収束特性を維持
- **C6-L3 (Semi-smooth Newton) の前提条件**

---

## 5. C6-L3: Semi-smooth Newton + NCP 関数

### 5.1 現状の Outer/Inner 分離の限界

現在の AL 法:
```
Outer: λ_n 更新, (s,t) 更新, k_pen 成長
Inner: R(u) = 0 を NR で解く（λ_n, s, t は固定）
```

**問題**: Outer 反復ごとに λ_n が急変 → Inner の Jacobian が陳腐化 → 収束劣化

### 5.2 NCP (Nonlinear Complementarity Problem) 定式化

接触条件を相補性問題として統一的に扱う:

```
力の釣り合い:    R(u, λ) = f_int(u) + C^T λ - f_ext = 0
相補性条件:      C(g, λ) = 0

NCP 関数（Fischer-Burmeister）:
  C_FB(a, b) = √(a² + b²) - a - b = 0
  ⟺ a ≥ 0, b ≥ 0, a·b = 0

法線接触に適用:
  C_n(g_n, λ_n) = FB(g_n, λ_n) = 0
  ⟺ g_n ≥ 0 (非貫入), λ_n ≥ 0 (圧縮), g_n·λ_n = 0 (相補性)
```

### 5.3 Semi-smooth Newton 法

NCP を含む非線形系を一括で Newton 解法する:

```
[K_T    C^T  ] [Δu] = -[R   ]
[∂C/∂u  ∂C/∂λ] [Δλ]    [C_FB]
```

∂C_FB/∂(g, λ) は semi-smooth（微分可能でない点で一般化微分を使う）:

```python
def ncp_fischer_burmeister(g, lam, *, reg=1e-12):
    """Fischer-Burmeister NCP 関数とその一般化微分."""
    norm = math.sqrt(g**2 + lam**2 + reg)
    fb = norm - g - lam

    # 一般化ヤコビアン
    dg = g / norm - 1.0
    dlam = lam / norm - 1.0
    return fb, dg, dlam
```

### 5.4 実装設計

```python
def newton_raphson_contact_ncp(
    model, f_ext, u0, manager,
    *, max_iter=50, tol=1e-8,
):
    """Semi-smooth Newton 法による接触解析.

    Outer loop 不要。u と λ を同時に更新。
    """
    u = u0.copy()
    lam = np.array([p.state.lambda_n for p in manager.pairs])

    for it in range(max_iter):
        # 1. 幾何更新（(s,t) を毎反復更新）
        update_all_geometry(manager, u)

        # 2. 残差
        R_u = assemble_residual(u) + contact_force(manager, lam) - f_ext
        C_fb = np.array([
            ncp_fischer_burmeister(p.state.gap, lam[i])[0]
            for i, p in enumerate(manager.pairs)
        ])
        residual = np.concatenate([R_u, C_fb])

        if np.linalg.norm(residual) < tol:
            break

        # 3. ヤコビアン（拡大系）
        K_T = assemble_tangent(u)
        C_T = contact_constraint_jacobian(manager)  # ∂C/∂u, ∂C/∂λ
        J = build_augmented_jacobian(K_T, C_T)

        # 4. Newton ステップ
        delta = solve(J, -residual)
        du, dlam = split(delta)

        u += du
        lam += dlam
        lam = np.maximum(lam, 0.0)  # λ ≥ 0 の射影

    return u, lam
```

### 5.5 段階的導入

| フェーズ | 内容 | 前提 |
|---------|------|------|
| α | NCP 関数の実装 + 単体テスト | なし |
| β | 2梁交差接触で NCP Newton 検証 | C6-L2 (∂s/∂u) |
| γ | 3本撚りへの適用 | α + β |
| δ | 7本撚り + 多ペアへの適用 | γ + C6-L4 |

### 5.6 期待効果

- **Outer loop の完全廃止**: λ_n と u を同時更新するため、Outer/Inner 分離が不要
- **二次収束**: Semi-smooth Newton は局所的に超線形収束（FB 関数の semi-smoothness による）
- **Active set の自動判定**: NCP が g > 0 ↔ λ = 0 を自然に切り替え（ヒステリシス不要）
- **k_pen 依存性の低減**: λ_n が直接解かれるため、k_pen はせいぜい正則化パラメータ

---

## 6. C6-L4: ブロック前処理強化

### 6.1 現状

GMRES + ILU(drop_tol=1e-4) フォールバック。ILU は接触ブロックの構造を考慮しない汎用前処理。

### 6.2 接触 Schur 補集合前処理

拡大系の構造を利用:

```
[K_T    C^T] [Δu]   [r_u]
[C      D  ] [Δλ] = [r_λ]

Schur 補集合: S = D - C · K_T⁻¹ · C^T

ブロック対角前処理:
P = [K_T    0  ]
    [0      S  ]
```

K_T⁻¹ の近似に ILU、S⁻¹ の近似に接触ペアの対角ブロック逆行列を使用。

### 6.3 実装の位置

`xkep_cae/contact/solver_hooks.py` の `_solve_linear_system()` を拡張。

---

## 7. C6-L5: Mortar 離散化

### 7.1 目的

セグメント境界での接触力の不連続を解消し、パッチテスト合格を保証する。

### 7.2 梁 Mortar の定式化

**Gay Neto & Wriggers (2020), Bosten et al. (2022) "A mortar formulation for frictionless line-to-line beam contact"** に基づく。

```
Mortar 拘束:
  ∫_Γc λ_h · (g_n · N_A) ds = 0  ∀ δλ_h

λ_h の離散化:
  λ_h(s) = Σ_k Φ_k(s) · Λ_k   （Mortar 基底関数）

Mortar 積分:
  D[k,i] = ∫_Γc Φ_k(s) · N_i(s) ds   （Mortar 行列）
  M[k,j] = ∫_Γc Φ_k(s) · N_j(t(s)) ds
```

### 7.3 実装コスト

Mortar 離散化は以下が必要:
- セグメント間のオーバーラップ検出（接触面の分割）
- Mortar 積分セルの構築
- 非適合メッシュ間の Mortar 行列計算

**C6-L1–L3 が成功すれば、Mortar は「仕上げ」に位置づけ可能。**
L1 の Gauss 積分 + L3 の NCP で多くの問題が解決する見込み。

---

## 8. 実装優先順位と依存関係

```
                   C6-L1 (Gauss 積分)
                     ↓
                   C6-L2 (∂s/∂u Jacobian)
                     ↓
                   C6-L3 (Semi-smooth Newton)  ← ここが最大目標
                     ↓
              ┌──────┴──────┐
           C6-L4           C6-L5
       (Schur 前処理)    (Mortar 離散化)
              └──────┬──────┘
                     ↓
               ML 最適化層
          (GNN k_pen / プリスクリーニング)
```

### 推奨実装順序

| 順序 | レベル | 概要 | テスト数見込 | 依存 |
|------|--------|------|------------|------|
| **1** | C6-L1 | Gauss 積分による line contact | +15–20 | なし |
| **2** | C6-L2 | ∂(s,t)/∂u Jacobian + 完全一貫接線 | +10–15 | L1 |
| **3** | C6-L3 | NCP + Semi-smooth Newton | +15–20 | L2 |
| **4** | C6-L4 | 接触 Schur 前処理 | +5–10 | L3 |
| **5** | C6-L5 | Mortar 離散化（必要に応じて） | +10–15 | L1 |

### 各レベルのファイル変更見込

| レベル | 新規ファイル | 変更ファイル |
|--------|------------|------------|
| C6-L1 | `contact/line_contact.py` | `contact/assembly.py`, `contact/pair.py` (ContactConfig) |
| C6-L2 | — | `contact/geometry.py`, `contact/assembly.py` |
| C6-L3 | `contact/ncp.py`, `contact/solver_ncp.py` | `contact/solver_hooks.py` |
| C6-L4 | — | `contact/solver_hooks.py` |
| C6-L5 | `contact/mortar.py` | `contact/assembly.py` |

---

## 9. 検証基準

各レベルで以下を検証:

### C6-L1 検証
- [ ] 2梁交差接触（PtP と line contact の力・変位一致、θ > 30°）
- [ ] 2梁平行接触（line contact が PtP より高精度、θ < 10°）
- [ ] 3本撚り引張（既存テストと同等精度 + 貫入率改善）
- [ ] Gauss 点数収束テスト（n_gp = 2,3,4,5 で力が収束）
- [ ] 既存テスト全パス（line_contact=False で後方互換）

### C6-L2 検証
- [ ] ∂(s,t)/∂u の数値微分検証（有限差分との一致）
- [ ] 完全一貫接線で Newton 二次収束（収束率プロット）
- [ ] monolithic 幾何更新で NR 収束（status-065 で失敗したケース）

### C6-L3 検証
- [ ] NCP Fischer-Burmeister 関数の単体テスト
- [ ] 2梁接触で Outer loop なし収束
- [ ] 3本撚りで Outer loop なし収束
- [ ] **7本撚りで貫入率 < 2% を達成**（最重要目標）
- [ ] Active set チャタリングの解消確認

---

## 10. ML との関係の再整理

### C6 完了後の ML の役割

| ML モジュール | C6 前の役割 | C6 後の役割 |
|-------------|-----------|-----------|
| **k_pen 推定 GNN** | k_pen の最適初期値推定（収束の生命線） | 正則化パラメータの初期値（重要度低下） |
| **プリスクリーニング GNN** | Narrowphase 候補削減（コスト削減） | 同じ（変わらず有用） |
| **GNN/PINN サロゲート** | 独立した熱解析サロゲート | 同じ |

**結論**: C6 の理論整備により、ML は「必須」から「最適化」にダウングレードされる。
これは健全な設計であり、ML の失敗がシステム全体の破綻につながらない。

---

## 11. リスクと緩和策

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| ∂(s,t)/∂u の平行特異性 | J が特異 → ds/du 計算不能 | 検出して PtP フォールバック |
| NCP の初期値感度 | 初期 λ が遠いと非収束 | continuation（λ=0 から開始） |
| 拡大系の行列サイズ増大 | (ndof + n_contact) × (ndof + n_contact) | Schur 補集合で n_contact を消去 |
| 既存テストへの影響 | 後方互換性破壊 | デフォルト OFF で段階的移行 |
| Line contact + NCP の組合せ爆発 | デバッグ困難 | L1→L2→L3 を順番に検証 |

---

## 12. 参考文献

- Meier, Popp, Wall (2016): "A unified approach for beam-to-beam contact" CMAME
- Meier, Popp, Wall (2016): "A finite element approach for the line-to-line contact interaction of thin beams" CMAME
- Bosten, Denoël, Cosimo, Cardona, Brüls (2022): "A mortar formulation for frictionless line-to-line beam contact" Multibody System Dynamics
- Durville (2012): "Contact-friction modeling within elastic beam assemblies" Computers & Structures
- Wriggers, Zavarise (1997): "On contact between three-dimensional beams undergoing large deflections" CMAME
- Alart, Curnier (1991): "A mixed formulation for frictional contact problems prone to Newton like solution methods" CMAME（NCP/Semi-smooth Newton の原典）

---
