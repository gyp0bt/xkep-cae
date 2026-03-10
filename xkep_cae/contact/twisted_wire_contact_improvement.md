# 撚線接触収束改善 設計仕様書

[← README](../../README.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-26
**目的**: 7本撚り（1+6構成）の接触NR収束を達成し、xfailテストを解消する

---

## 1. 問題の診断

### 1.1 根本原因

7本撚り（36+接触ペア同時活性化）で**ペナルティ法の本質的ジレンマ**が発生:

| k_pen | 貫入(pen_ratio) | Inner NR | Outer loop | 結果 |
|-------|-----------------|----------|------------|------|
| 低 | 15.5% (過大) | ✓ 3-5反復収束 | ✗ merit倍増発散 | NG |
| 高 | < 2% | ✗ 条件数悪化→spsolve破綻 | — | NG |

### 1.2 商用ソルバーの知見

| ソルバー | 撚線接触の推奨手法 |
|---------|-------------------|
| **Abaqus** | Augmented Lagrange + contact stabilization + Explicit→Standard import |
| **LS-DYNA** | Mortar接触(implicit推奨) + BFGS準Newton + Explicit→Implicit自動切替 |
| **ANSYS** | CONTA177 + Augmented Lagrange (FKN=0.01〜0.1) |

**共通知見**:
1. 陰解法ではAugmented Lagrangianが推奨（乗数蓄積で低k_penでも正確な接触力）
2. 陽解法で接触確立→陰解法に切替が常套手段
3. 反復ソルバー（BFGS等）が直接法より ill-conditioned に強い

---

## 2. 改善戦略（3段構え）

### 優先度1: AL乗数緩和 + 保守的ペナルティ成長（陰解法改善）

**問題**: 現在の `lambda_n <- p_n` は最もシンプルなUzawa更新。k_penが低いと
乗数が十分に蓄積しないうち貫入が大きくなり、k_penを増大させると条件数が悪化する。

**改善**: AL乗数の緩和（under-relaxation）と保守的なk_pen成長で、
低いk_penのまま乗数を徐々に蓄積させ、正確な接触力に近づける。

```python
# 現在: lambda_n <- p_n (= max(0, lambda_n + k_pen * (-gap)))
# 改善: lambda_n <- lambda_n + omega * (p_n - lambda_n)
#       omega ∈ (0, 1] でアンダーリラクゼーション
```

**パラメータ追加（ContactConfig）**:

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `al_relaxation` | 1.0 | AL乗数更新の緩和係数 ω ∈ (0,1]。1.0で従来動作 |
| (既存) `penalty_growth_factor` | 2.0 → **1.5** | 保守的成長に変更 |
| (既存) `n_outer_max` | 5 → **20** | 多くのOuter反復を許容 |

### 優先度2: 反復線形ソルバー（GMRES + ILU前処理）

**問題**: `spsolve`（直接法）は高k_pen時の条件数悪化で精度劣化。

**改善**: `scipy.sparse.linalg.gmres` + `ILU(0)前処理`で ill-conditioned な
K_T + K_c を安定的に解く。直接法より条件数に強い。

```python
# 前処理付き反復法
M = sp.linalg.spilu(K_bc.tocsc(), drop_tol=1e-4)
M_op = sp.linalg.LinearOperator(K_bc.shape, M.solve)
du, info = sp.linalg.gmres(K_bc, r_bc, M=M_op, atol=1e-10, maxiter=500)
```

**パラメータ追加（ContactConfig）**:

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `linear_solver` | `"direct"` | `"direct"` / `"iterative"` / `"auto"` |
| `iterative_tol` | `1e-10` | GMRES収束判定 |
| `ilu_drop_tol` | `1e-4` | ILU前処理のdrop tolerance |

`"auto"` モード: まず direct で試行し、MatrixRankWarning検出時に iterative にフォールバック。

### 優先度3: 陽解法準静的接触ソルバー

**問題**: 陰解法の収束が根本的に困難な場合のフォールバック。

**改善**: 既存のCentral Difference陽解法にペナルティ接触を組込み、
mass scalingで準静的解析を実現する。

```
solve_explicit_contact():
  1. mass scaling: M_scaled = M * (target_dt / dt_critical)^2
  2. Central Difference + penalty contact force
  3. 運動エネルギーチェック: ALLKE < 0.05 * ALLIE
  4. optional: explicit解をimplicit NRの初期値として使用
```

---

## 3. 実装計画

### Step 1: AL乗数緩和 + 保守的k_pen成長

**変更ファイル**:
- `xkep_cae/contact/pair.py`: ContactConfigに`al_relaxation`追加
- `xkep_cae/contact/law_normal.py`: `update_al_multiplier()`に緩和ロジック追加
- `xkep_cae/contact/solver_hooks.py`: Outer loopの`n_outer_max`デフォルト調整不要（呼び出し側で設定）

**テスト**:
- AL緩和の単体テスト（乗数が正しく蓄積するか）
- 3本撚りの後方互換テスト（omega=1.0で従来と同一）
- 7本撚りテスト（omega=0.5, n_outer=20で改善確認）

### Step 2: 反復線形ソルバー

**変更ファイル**:
- `xkep_cae/contact/pair.py`: ContactConfigに`linear_solver`等追加
- `xkep_cae/contact/solver_hooks.py`: spsolveの代わりにgmres+ILU
- `xkep_cae/contact/iterative_solver.py`: 新規。反復ソルバーのラッパー

**テスト**:
- 交差梁テスト（iterativeで従来と同精度）
- 3本撚り（iterativeでの収束確認）
- 7本撚り（高k_penでのiterativeの安定性確認）

### Step 3: 陽解法準静的接触ソルバー

**変更ファイル**:
- `xkep_cae/contact/explicit_contact.py`: 新規。陽解法+接触ソルバー
- `xkep_cae/dynamics.py`: mass scaling ユーティリティ追加

**テスト**:
- 2梁交差テスト（explicit contactの基本動作）
- 3本撚り（explicit quasi-staticで安定収束確認）
- 7本撚り（explicitで接触確立→implicit初期値）

---

## 4. 参考文献

- Simo, J.C. & Laursen, T.A. (1992) "An augmented Lagrangian treatment of contact problems involving friction"
- Wriggers, P. (2006) "Computational Contact Mechanics" Ch.6
- Meier, C., Popp, A. & Wall, W.A. (2017) "A unified approach for beam-to-beam contact" (CMAME)
- LS-DYNA Theory Manual — Mortar Contact, Implicit Contact Treatment
- Abaqus Analysis User's Manual — Edge-to-edge Contact, Augmented Lagrange
- ANSYS Contact Technology Guide — CONTA177, 3D Beam-to-Beam Contact

---
