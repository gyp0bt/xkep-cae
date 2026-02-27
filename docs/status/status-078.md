# status-078: Phase C6-L2 — 一貫接線の完全化（∂(s,t)/∂u Jacobian）

[← README](../../README.md) | [← status-index](status-index.md)

- **日付**: 2026-02-27
- **テスト数**: 1775（fast: +15）
- **ブランチ**: claude/execute-status-todos-nj0KN

## 概要

Phase C6: 接触アルゴリズム根本整理の第2段階として、∂(s,t)/∂u Jacobian を実装。従来の PtP 接触および Line-to-line 接触の両方で、接触点移動に伴う一貫接線剛性 K_st を追加した。

設計仕様: [contact-algorithm-overhaul-c6.md](../contact/contact-algorithm-overhaul-c6.md) §4

## 背景・動機

現行の接触接線剛性は (s,t) を固定した状態で ∂f_c/∂u を計算している（K_n 主項 + K_geo）。しかし、最近接パラメータ (s,t) 自体が変位 u に依存するため、∂(s,t)/∂u の寄与（K_st）が欠落している。この欠落により:

1. monolithic 幾何更新（Inner NR 内で (s,t) を毎反復更新）で Newton-Raphson が発散
2. 完全な二次収束が得られない
3. C6-L3（Semi-smooth Newton）の前提条件が満たされない

## 実施内容

### 1. compute_st_jacobian（geometry.py）

PtP 最近接点条件の陰関数微分。

**最近接点条件**:
```
F₁ = δ · dA = 0
F₂ = -δ · dB = 0
ただし δ = pA(s) - pB(t)
```

**陰関数の定理**:
```
J · d(s,t)/du = -∂F/∂u

J = [[dA·dA, -(dA·dB)],
     [-(dA·dB), dB·dB]]
```

**∂F/∂u の完全版**: δ · ∂dA/∂u、δ · ∂dB/∂u 項を含む。

**境界処理**:
- s クランプ → ds/du = 0, dt のみ 1×1 系で計算
- t クランプ → dt/du = 0, ds のみ 1×1 系で計算
- 両方クランプ → ds/du = dt/du = 0
- 平行特異 → None（フォールバック）

### 2. compute_t_jacobian_at_gp（line_contact.py）

Line contact Gauss 点での ∂t/∂u。s_gp は固定（積分点パラメータ）なので ds/du = 0。

**射影条件**: G(t, u) = (pA(s_gp) - pB(t)) · dB = 0

```
dt/du = -(1/∂G/∂t) · ∂G/∂u|_{t fixed}
∂G/∂t = -|dB|²
```

### 3. 一貫接線剛性 K_st（assembly.py）

接触力 f_c = p_n · g_n の完全微分:
```
K_c = K_n + K_geo + K_st

K_st = outer(g_n, ∂p_n/∂s·ds_du + ∂p_n/∂t·dt_du)      # 接触点移動→力変化
     + p_n · outer(∂g_n/∂s, ds_du)                       # 接触点移動→形状変化(s)
     + p_n · outer(∂g_n/∂t, dt_du)                       # 接触点移動→形状変化(t)
```

∂g_n/∂s, ∂g_n/∂t は形状関数の明示的微分と法線回転（∂n/∂s, ∂n/∂t）を含む。

### 4. ContactConfig 拡張

`consistent_st_tangent: bool = False` を追加。デフォルト OFF で後方互換を維持。

### 5. テスト（+15テスト、全 fast）

| テストクラス | テスト数 | 内容 |
|-------------|---------|------|
| TestComputeStJacobian | 5 | 数値微分検証（直交/斜交/平行/クランプ） |
| TestComputeTJacobianAtGP | 3 | 数値微分検証（内部/クランプ/複数GP） |
| TestConsistentStStiffness | 3 | 完全剛性の数値微分検証・非活性ケース |
| TestAssemblyConsistentTangent | 4 | 後方互換・PtP/line contact 統合 |

## ファイル変更

### 新規
- `tests/contact/test_consistent_st_tangent.py` — 15テスト
- `docs/status/status-078.md` — 本ステータス

### 変更
- `xkep_cae/contact/geometry.py` — `compute_st_jacobian` 追加
- `xkep_cae/contact/line_contact.py` — `compute_t_jacobian_at_gp` 追加
- `xkep_cae/contact/assembly.py` — `_consistent_st_stiffness_local`, `_consistent_st_stiffness_at_gp`, `_compute_line_contact_st_stiffness_local`, `_add_local_to_coo` 追加。`compute_contact_stiffness` に K_st 統合
- `xkep_cae/contact/pair.py` — ContactConfig に `consistent_st_tangent` 追加
- `xkep_cae/contact/__init__.py` — `compute_st_jacobian`, `compute_t_jacobian_at_gp` エクスポート追加
- `README.md` — テスト数更新
- `docs/roadmap.md` — C6-L2 チェック + テスト数更新
- `docs/status/status-index.md` — status-078 追加
- `CLAUDE.md` — 現在の状態更新

## 設計上の懸念・TODO

### 消化済み（status-077 → 078）
- [x] C6-L2: 一貫接線の完全化（∂s/∂u, ∂t/∂u Jacobian）

### 未解決（引き継ぎ）
- [ ] **C6-L3: Semi-smooth Newton + NCP 関数** — Outer loop 廃止の鍵。C6-L2 が前提。
- [ ] **C6-L4: ブロック前処理強化（接触 Schur 補集合）**
- [ ] **C6-L5: Mortar 離散化**
- [ ] 摩擦力の line contact 拡張 — 現在は PtP 代表点で評価
- [ ] 接触プリスクリーニング GNN Step 2-5
- [ ] k_pen推定ML v2 Step 2-7

### 設計メモ
- ∂F/∂u の完全版（δ·∂dA/∂u, δ·∂dB/∂u 項含む）を実装。近似版（δ項省略）との差は O(gap/L) スケールだが、精度のため完全版を採用。
- 平行セグメント（Gram 行列の行列式 ≈ 0）は None を返して PtP フォールバック。
- 境界ケース（s or t = 0,1）は縮退した 1×1 系で処理。
- 全15テストで数値微分との一致を確認（atol ≤ 1e-4）。

---
