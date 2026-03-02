# status-080: Phase C6-L4 — ブロック前処理強化（接触 Schur 補集合）

[← README](../../README.md) | [← status-index](status-index.md)

- **日付**: 2026-02-28
- **テスト数**: 1821（fast: +11）
- **ブランチ**: claude/execute-status-todos-SE1Uh

## 概要

Phase C6: 接触アルゴリズム根本整理の第4段階として、NCP 鞍点系のブロック対角前処理付き GMRES ソルバーを実装。大規模問題（n_active が大きい場合）での効率改善を目的とする。

設計仕様: [contact-algorithm-overhaul-c6.md](../contact/contact-algorithm-overhaul-c6.md) §6

## 背景・動機

従来の NCP 鞍点系解法（`_solve_saddle_point_contact()`）は制約空間 Schur complement で解いており、K_eff^{-1} を (1 + n_active) 回計算する必要があった。n_active が大きい場合（多数の接触ペアがアクティブ）、計算コストがスケールしない。

ブロック対角前処理付き GMRES により、これを回避する。

## 実施内容

### _solve_saddle_point_gmres()

拡大系（鞍点系）全体を GMRES で解く。前処理にブロック対角構造を利用。

**鞍点系**:
```
[K_eff   -G_A^T] [Δu  ] = [-R_u    ]
[G_A      0    ] [Δλ_A]   [-g_active]
```

**ブロック対角前処理**:
```
P = [K_eff^{-1}   0      ]    K_eff^{-1} ≈ ILU
    [0            S_d^{-1}]    S_d ≈ diag(G_A * K_eff^{-1} * G_A^T)
```

- **K_eff^{-1} の近似**: ILU(drop_tol) を前処理に使用
- **S_d^{-1} の近似**: Schur 補集合の対角近似。各 active ペア i に対して S_ii = G_A[i,:] * K_eff^{-1} * G_A[i,:]^T を ILU 経由で近似計算
- **拡大系行列**: scipy.sparse.linalg.LinearOperator で行列-ベクトル積を定義（メモリ効率）
- **GMRES 不収束時**: 直接 Schur complement にフォールバック

### _solve_saddle_point_direct()

従来の Schur complement 解法を `_solve_saddle_point_direct()` としてリファクタリング。

### _solve_saddle_point_contact() のディスパッチ化

`use_block_preconditioner` フラグにより GMRES と直接法を切り替えるディスパッチ関数に変更。

### ContactConfig 拡張

`pair.py` に1フィールド追加:
- `ncp_block_preconditioner: bool = False` — NCP 鞍点系のブロック前処理 GMRES の有効化

### テスト（+11テスト、全 fast）

| テストファイル | テストクラス | テスト数 | 内容 |
|--------------|-------------|---------|------|
| test_block_preconditioner.py | TestSaddlePointGMRES | 5 | GMRES vs 直接法の結果比較、BC、ディスパッチ |
| test_block_preconditioner.py | TestBlockPreconditionerConvergence | 4 | NCP ソルバー収束、直接法との変位比較、接触なし、λ ≥ 0 |
| test_block_preconditioner.py | TestPreconditionerConfig | 2 | デフォルト設定、有効化設定 |

## ファイル変更

### 新規
- `tests/contact/test_block_preconditioner.py` — 11テスト
- `docs/status/status-080.md` — 本ステータス

### 変更
- `xkep_cae/contact/solver_ncp.py` — `_solve_saddle_point_gmres()` 追加, `_solve_saddle_point_direct()` リファクタリング, `_solve_saddle_point_contact()` ディスパッチ化
- `xkep_cae/contact/pair.py` — ContactConfig に `ncp_block_preconditioner` 追加
- `xkep_cae/contact/__init__.py` — 新関数のエクスポート追加
- `README.md` — テスト数更新
- `docs/roadmap.md` — C6-L4 チェック + テスト数更新
- `docs/status/status-index.md` — status-080 追加
- `CLAUDE.md` — 現在の状態更新

## 設計上の懸念・TODO

### 消化済み（status-079 → 080）
- [x] C6-L4: ブロック前処理強化（接触 Schur 補集合）

### 未解決（引き継ぎ）
- [ ] **C6-L5: Mortar 離散化**（必要に応じて）
- [ ] 摩擦力の line contact 拡張 — 現在は PtP 代表点で評価
- [ ] 接触プリスクリーニング GNN Step 2-5
- [ ] k_pen推定ML v2 Step 2-7
- [ ] NCP ソルバーの摩擦拡張（Coulomb 摩擦の NCP 定式化）
- [ ] NCP ソルバーの line contact 統合

### 設計メモ
- **直接法 vs GMRES**: 小規模問題（n_active < 10）では直接 Schur complement が高速。大規模問題（n_active > 50）では GMRES のほうがスケーラブル。デフォルトは直接法（`ncp_block_preconditioner=False`）。
- **前処理の品質**: Schur 補集合の対角近似は粗いが、接触ペア間の結合が弱い場合（典型的な梁-梁接触）では十分。ILU 失敗時は粗い k_pen^{-1} 近似にフォールバック。
- **フォールバック**: GMRES 不収束時は直接 Schur complement に自動フォールバック。

---
