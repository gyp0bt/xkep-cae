# status-145: δ正則化鞍点系 + Phase2接触収束改善（WIP）

[← README](../../README.md) | [← status-index](status-index.md) | [前: status-144](status-144.md)

**日付**: 2026-03-08
**テスト数**: 2271（変更なし）

## 概要

7本撚線 90° 曲げ揺動のPhase2（揺動フェーズ）収束を目指し、
δ正則化（Uzawa regularization）を NCP 鞍点系に導入。
Phase2不収束の根本原因を段階的に特定し、複数の改善を実装した。

## 実装内容

### 1. δ正則化鞍点系（Uzawa regularization）

**ファイル**: `xkep_cae/contact/solver_ncp.py`

鞍点系の(2,2)ブロックに -δI を追加:

```
[K_eff   -G^T] [Δu  ] = [-R_u        ]
[G       -δI ] [Δλ_A]   [-g_A + δ*λ_A]
```

- δ > 0 のとき Schur complement S = G*V + δI は正定値保証 → 直接solve
- δ = 0 のとき従来の SVD 截断を使用
- δ の自動計算: `δ = 1/(100*k_pen)` — ペナルティ剛性の1/100
- `contact_compliance` パラメータとして `newton_raphson_contact_ncp` に追加
- `wire_bending_benchmark.py` Phase2 で自動適用（`contact_compliance=None` → auto）

### 2. 単調活性セット戦略（monotone active-set）

**ファイル**: `xkep_cae/contact/solver_ncp.py`

δ正則化時、Newton反復中の活性セット振動を抑制:
- 一度活性化したペアは同一増分内で非活性化しない（単調成長）
- 増分間では活性セットを自由にリセット（`_frozen_ncp_active_mask = None`）

### 3. δ正則化の一貫性確保

すべての接触力・ギャップ計算でδ*λ項を統一:

- **NCP残差**: `C_ac = k_pen * (g + δ*λ)` for active pairs
- **活性セット判定**: `p_n = max(0, λ + k_pen*(-(g + δ*λ)))`
- **代表点接触力**: `_compute_contact_force_from_lambdas` にδ項追加
- **pair.state.p_n**: 正則化ギャップで計算
- **line_contact力**: `compute_line_contact_force_local` にδ項追加
- **line_contact剛性**: `compute_line_contact_stiffness_local` にδ項追加
- **ContactConfig**: `contact_compliance` フィールド追加

### 4. Phase2で代表点接触力を使用

δ正則化時、line-contact Gauss積分ではなく代表点方式で接触力を計算:
- 鞍点系のgaps（代表点ギャップ）と整合的
- line-contact Gauss積分のギャップとのミスマッチを解消

### 5. その他の修正

- `tests/contact/test_block_preconditioner.py`: z_sep を 0.035 → 0.045 に変更（初期貫入回避）
- `xkep_cae/mesh/twisted_wire.py`: 弦近似誤差に基づく最小安全ギャップの自動推定

## Phase2不収束の根本原因分析

### 特定された問題

1. **活性セット振動**: Newton反復中に接触ペアが活性/非活性を繰り返し発散
   → 単調活性セット戦略で解消

2. **Gauss積分vs代表点の力不整合**: 線接触Gauss積分の接触力と鞍点系の代表点ギャップが不一致
   → 代表点方式への切り替えで解消

3. **AL-NCP力と鞍点線形化の不整合（未解決）**:
   - AL接触力: `f_c = G^T * p_n` where `p_n = max(0, λ - k_pen*(g+δλ))`
   - 鞍点線形化: `f_c ≈ G^T * λ`（Lagrange乗数のみ）
   - 差分: `f_c - G^T*λ = G^T*(-k_pen*(g+δλ))` → 力残差に系統的オフセット
   - 結果: `||du||` は1e-14（機械精度）まで収束するが、`||R_u||/||f||` は ~2e-3 で停滞

### 現在の状態

- **Phase1（90° 曲げ）**: 収束OK（93増分, 1308 NR反復, 95秒）
- **Phase2（揺動）**: 不収束
  - `||du||` → 1e-14（変位は機械精度で収束）
  - `||R_u||/||f||` → 1.96e-3 で停滞（tol_force=1e-4 を満たさない）
  - `||C_n||` → 3.6e-2 で停滞
  - 活性セット安定（6/16）、発散なし

### 次の方針（検討中）

1. **AL penalty法への切替**: 鞍点系を廃止し `K_total * Δu = -R_u` のみ
2. **変位収束判定**: `||du|| < tol_du` で収束判定（力残差に依存しない）
3. **Phase1と同一の力評価+許容誤差緩和**: `tol_force` を 5e-3 に緩和

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver_ncp.py` | δ正則化鞍点系、単調活性セット、代表点力、一貫性修正 |
| `xkep_cae/contact/pair.py` | `ContactConfig.contact_compliance` 追加 |
| `xkep_cae/contact/assembly.py` | `contact_compliance` パラメータ透過 |
| `xkep_cae/contact/line_contact.py` | δ正則化対応の力・剛性計算 |
| `xkep_cae/numerical_tests/wire_bending_benchmark.py` | Phase2 `contact_compliance` 自動適用 |
| `xkep_cae/mesh/twisted_wire.py` | 弦近似誤差ベース最小安全ギャップ |
| `tests/contact/test_block_preconditioner.py` | z_sep 修正 |
| `scripts/verify_90deg_*.py` | Phase1/Phase2検証スクリプト群 |

---
