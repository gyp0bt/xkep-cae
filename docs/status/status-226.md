# status-226: 整合接線剛性 ∂(s,t)/∂u の実装

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-22
**ブランチ**: `claude/fix-focus-guard-bending-RlaJe`

---

## 概要

接触最近接点パラメータ (s, t) の変位微分 ∂(s,t)/∂u を実装し、
接触接線剛性に K_st（接触点滑り剛性）を追加。
法線接触力・摩擦力の両方に対応。有限差分検証テスト11件追加。

**結論**: K_st は数学的に正しい（FD テスト通過）が、動的三点曲げでは
収束を改善しない。frac=0.60 の壁は K_st 以外に原因がある。

---

## 実装内容

### 1. ComputeStJacobianProcess（新規）

**ファイル**: `xkep_cae/contact/geometry/_st_jacobian.py`

最近接点条件の陰関数微分:
```
F₁ = δ · dA = 0, F₂ = -δ · dB = 0
J = [[a, -b], [-b, c]]  (a=dA·dA, b=dA·dB, c=dB·dB)
[ds/du, dt/du] = -J⁻¹ · [∂F₁/∂u, ∂F₂/∂u]
```

∂F/∂u の完全版（δ·∂dA/∂u, δ·∂dB/∂u 項含む）を実装。
境界処理: s/t クランプ → 1×1 縮退系、平行 → valid=False。

### 2. 接触力 K_st

**ファイル**: `xkep_cae/contact/contact_force/strategy.py`

`HuberContactForceProcess.tangent()` に K_st 項を追加:
```
K_st = -(outer(∂f_raw/∂s, ds/du) + outer(∂f_raw/∂t, dt/du))

∂f_raw/∂s = p_n * (∂c_k/∂s · n + c_k · ∂n/∂s)
∂n/∂s = (1/dist)(I - n⊗n) · dA
```

形状関数係数変化 + 法線回転の連鎖微分を含む完全な実装。

### 3. 摩擦力 K_st_fric

**ファイル**: `xkep_cae/contact/friction/_assembly.py`

`_assemble_friction_st_stiffness()` を追加:
```
K_st_fric = outer(∂f_fric/∂s, ds/du) + outer(∂f_fric/∂t, dt/du)

∂f_fric/∂s = Σ_α q_α · dc_k/ds · tα_i
```

### 4. TangentAssemblyProcess 統合

**ファイル**: `xkep_cae/contact/solver/_newton_steps.py`

`consistent_st_tangent` フラグに応じて K_st を自動統合。
`node_coords` を tangent メソッドにオプショナル引数で渡す。

---

## テスト結果

### 有限差分検証（11件全通過）

| テストクラス | テスト数 | 内容 |
|-------------|---------|------|
| TestComputeStJacobian | 7 | ∂(s,t)/∂u の数値微分検証 |
| TestConsistentStTangent | 4 | 接触力の完全接線 FD 検証 |

### 三点曲げ実行結果

| 設定 | K_st | frac到達 | 時間 | 備考 |
|------|------|---------|------|------|
| E=25, 幾何剛性あり | OFF | 0.6011 | 107s | status-225 baseline |
| E=25, 幾何剛性あり, K_st | ON | 0.5338 | >300s | 線形収束で遅延 |

### 分析

- K_st ON で収束が**遅化**（二次収束ではなく線形収束）
- 残差 ratio: att=0 で 1.0 → att=25 で 0.56（率 ≈ 0.98/iter）
- 変位収束（||du||/||u|| < tol）で前進するが、力収束は達成できない
- K_st は非対称行列を導入 → 条件数悪化の可能性
- **frac=0.60 の壁の根本原因は K_st 以外にある**

---

## 次のステップ（TODO）

1. **frac=0.60 壁の条件数・スペクトル分析**: K_T の固有値分布を frac=0.59 付近で調査
2. **NR 内幾何凍結の検討**: NR 反復内で s, t を凍結し、ステップ間のみ更新する方式の検証
3. **ラインサーチ強化**: 現在の二分法（diverge_factor=3.0）→ Armijo backtracking の導入
4. **変位収束のみモード**: 力収束を無視し変位収束のみで進行するオプション
5. **K_st の選択的適用**: 大変形域（push/L > 0.2）でのみ K_st を有効化
6. **E=100 での検証**: k_pen=48EI/L³ 明示指定で K_st の効果を確認

---

## 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
|---------|---------|------|
| `xkep_cae/contact/geometry/_st_jacobian.py` | 新規 | ComputeStJacobianProcess |
| `xkep_cae/contact/contact_force/strategy.py` | 変更 | tangent() に K_st 追加 |
| `xkep_cae/contact/friction/_assembly.py` | 変更 | _assemble_friction_st_stiffness 追加 |
| `xkep_cae/contact/friction/strategy.py` | 変更 | tangent() に K_st_fric 統合 |
| `xkep_cae/contact/solver/_newton_steps.py` | 変更 | TangentAssembly に node_coords 渡し |
| `tests/contact/test_st_jacobian.py` | 新規 | 7件の FD 検証テスト |
| `tests/contact/test_consistent_st_tangent.py` | 新規 | 4件の完全接線 FD 検証 |

---

## 設計メモ

- status-078 の旧実装を Process Architecture で再実装
- `ContactConfig.consistent_st_tangent` フラグ（既存、デフォルト False）を活用
- K_st は非対称行列（scipy.sparse.linalg.spsolve で対応可能）
- 三点曲げ config ではデフォルト OFF のまま（収束改善が確認されるまで）

---

## テスト

**175+11 passed** — 契約違反 0件、条例違反 0件
