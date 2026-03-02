# status-079: Phase C6-L3 — Semi-smooth Newton + NCP 関数

[← README](../../README.md) | [← status-index](status-index.md)

- **日付**: 2026-02-27
- **テスト数**: 1810（fast: +35）
- **ブランチ**: claude/execute-status-todos-CasnK

## 概要

Phase C6: 接触アルゴリズム根本整理の第3段階として、NCP（非線形相補性問題）関数と Semi-smooth Newton ソルバーを実装。従来の Augmented Lagrangian Outer loop を廃止し、変位 u とラグランジュ乗数 λ を鞍点系で同時に更新する。

設計仕様: [contact-algorithm-overhaul-c6.md](../contact/contact-algorithm-overhaul-c6.md) §5

## 背景・動機

従来の接触ソルバー（`newton_raphson_with_contact`）は Inner NR + Outer AL 乗数更新の二重ループ構造。この構造には:

1. Outer loop の収束が遅い（特に k_pen が不適切な場合）
2. Inner/Outer の分離が条件数に依存し、ロバスト性が低い
3. Active set の変化を明示的にハンドリングできない

NCP 定式化は接触の相補性条件（g ≥ 0, λ ≥ 0, g·λ = 0）を直接扱い、Semi-smooth Newton で一括解法する。

## 実施内容

### Phase C6-L3α: NCP 関数（ncp.py）

Fischer-Burmeister NCP 関数と min 関数を実装。

**Fischer-Burmeister**: `FB(a,b) = sqrt(a² + b² + reg) - a - b`
- `FB(a,b) = 0` ⟺ `a ≥ 0, b ≥ 0, a·b = 0`
- 一般化微分（∂FB/∂a, ∂FB/∂b）付き
- 正則化パラメータ `reg` で原点での特異性を回避

**min 関数**: `C(a,b) = min(a,b)`
- 区分線形、semi-smooth

**ユーティリティ関数**:
- `evaluate_ncp_residual()`: ベクトル版 NCP 残差
- `evaluate_ncp_jacobian()`: ベクトル版 NCP ヤコビアン
- `compute_gap_jacobian_wrt_u()`: ∂g_n/∂u の解析計算
- `build_augmented_residual()`: 拡大残差 [R_u; C_ncp]

### Phase C6-L3β: Semi-smooth Newton ソルバー（solver_ncp.py）

**AL-NCP ハイブリッド + Alart-Curnier 鞍点定式化**:

接触力は AL 形式で計算:
```
p_n_i = max(0, λ_i + k_pen * (-g_i))
```

NCP 条件は Alart-Curnier 型:
```
Active  (p_n > 0): C_i = k_pen * g_i  → g = 0 を強制
Inactive:          C_i = λ_i          → λ = 0 を強制
```

**鞍点系**:
```
[K_eff   -G_A^T] [Δu  ] = [-R_u    ]
[G_A      0    ] [Δλ_A]   [-g_active]

K_eff = K_T + k_pen * G_A^T * G_A
```

制約空間 Schur complement (n_active × n_active) で効率的に解く。k_pen がペナルティ正則化として K_eff を確実に正定値に保つ。

**設計上の経緯**: 初期は FB-based Schur complement を試みたが、接触解（g ≈ 0, p_n > 0）で dC_FB/dp_n → 0 となり D 行列が特異化する問題が発覚。Alart-Curnier 鞍点アプローチに切り替え、制約空間 Schur complement で安定かつロバストな解法を実現した。

**ソルバー機能**:
- Active set の自動判定（λ + k_pen·(-g) > 0）
- 荷重増分（n_load_steps）
- Broadphase は各ステップ先頭でのみ実行（反復中は freeze）
- FB/min 両方の NCP モニタリング対応
- 既存 AL ソルバーとの結果比較検証済み
- ContactGraphHistory 対応

### ContactConfig 拡張

`pair.py` に3フィールド追加:
- `use_ncp: bool = False` — NCP ソルバーの有効化
- `ncp_type: str = "fb"` — NCP 関数の種類（"fb" | "min"）
- `ncp_reg: float = 1e-12` — FB 正則化パラメータ

### テスト（+35テスト、全 fast）

| テストファイル | テストクラス | テスト数 | 内容 |
|--------------|-------------|---------|------|
| test_ncp.py | TestFischerBurmeister | 8 | FB 関数: 相補性, 活性/非活性, 貫入, 微分, 対称性 |
| test_ncp.py | TestNCPMin | 4 | min NCP: 大小比較, 相補性 |
| test_ncp.py | TestNCPResidualJacobian | 5 | ベクトル版残差・ヤコビアン形状・整合性 |
| test_ncp.py | TestGapJacobianWrtU | 3 | ∂g_n/∂u 形状・数値微分・回転DOF |
| test_ncp.py | TestBuildAugmentedResidual | 4 | 拡大残差構造・NCP 零条件 |
| test_solver_ncp.py | TestNCPSolverBasic | 3 | 離間・接触・結果構造 |
| test_solver_ncp.py | TestNCPSolverConvergence | 3 | Outer loop free 収束, FB/min 比較, λ ≥ 0 |
| test_solver_ncp.py | TestNCPSolverComparison | 1 | 既存 AL ソルバーとの変位比較 |
| test_solver_ncp.py | TestConstraintJacobianAndForce | 4 | G 行列形状, inactive skip, f_c 計算 |

## ファイル変更

### 新規
- `xkep_cae/contact/ncp.py` — NCP 関数モジュール
- `xkep_cae/contact/solver_ncp.py` — Semi-smooth Newton ソルバー
- `tests/contact/test_ncp.py` — 24テスト
- `tests/contact/test_solver_ncp.py` — 11テスト
- `docs/status/status-079.md` — 本ステータス

### 変更
- `xkep_cae/contact/pair.py` — ContactConfig に NCP 関連フィールド追加
- `xkep_cae/contact/__init__.py` — ncp, solver_ncp モジュールのエクスポート追加
- `README.md` — テスト数更新
- `docs/roadmap.md` — C6-L3 チェック + テスト数更新
- `docs/status/status-index.md` — status-079 追加
- `CLAUDE.md` — 現在の状態更新

## 設計上の懸念・TODO

### 消化済み（status-078 → 079）
- [x] C6-L3: Semi-smooth Newton + NCP 関数

### 未解決（引き継ぎ）
- [ ] **C6-L4: ブロック前処理強化（接触 Schur 補集合）**
- [ ] **C6-L5: Mortar 離散化**
- [ ] 摩擦力の line contact 拡張 — 現在は PtP 代表点で評価
- [ ] 接触プリスクリーニング GNN Step 2-5
- [ ] k_pen推定ML v2 Step 2-7
- [ ] NCP ソルバーの摩擦拡張（Coulomb 摩擦の NCP 定式化）
- [ ] NCP ソルバーの line contact 統合

### 設計メモ
- **FB vs Alart-Curnier**: FB 関数は残差モニタリングに使用するが、線形化には Alart-Curnier 鞍点アプローチを採用。FB の微分特異性（dC/dp_n → 0 at solution boundary）を回避。
- **鞍点系 Schur complement**: 制約空間は n_active × n_active（通常非常に小さい）。K_eff を n_active + 1 回解くが、K_eff は k_pen 正則化で SPD が保証される。
- **Active set**: λ + k_pen·(-g) > 0 で判定。Inactive ペアは λ = 0 に強制。PDAS（primal-dual active set）と数学的に等価。
- **既存 AL ソルバーとの互換性**: ContactConfig.use_ncp フラグで切り替え可能。デフォルトは従来 AL ソルバー。

---
