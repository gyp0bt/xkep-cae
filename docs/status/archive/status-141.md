# status-141: 被膜Coulomb摩擦モデル実装 + 摩擦core関数抽出

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-08
**テスト数**: 2271（変更なし）

---

## 概要

被膜接触面にCoulomb摩擦モデルを追加し、梁-梁摩擦のreturn mappingアルゴリズムを
被膜接触にも共有できる構造にリファクタリングした。

1. **摩擦core関数抽出**: `return_mapping_core()` / `tangent_2x2_core()` を純粋関数化
2. **被膜摩擦フィールド追加**: ContactState/Configに被膜摩擦パラメータ追加
3. **solver_ncp統合**: Newton loop内で被膜摩擦力・接線剛性を組み込み

---

## 1. 摩擦core関数抽出

**影響ファイル**: `law_friction.py`

### 設計方針

`pair.state` に依存しない純粋関数を抽出し、梁-梁/被膜-被膜/被膜-梁の
全接触タイプで同一アルゴリズムを共有する。

### 新規関数

| 関数 | 入力 | 出力 |
|------|------|------|
| `return_mapping_core(z_t_old, delta_ut, k_t, p_n, mu)` | スカラー/ベクトル | (q, is_stick, q_trial_norm, dissipation) |
| `tangent_2x2_core(k_t, p_n, mu, z_t, q_trial_norm, is_stick)` | スカラー/ベクトル | D_t (2×2) |

### リファクタリング

- `friction_return_mapping()` → `return_mapping_core()` で委譲
- `friction_tangent_2x2()` → `tangent_2x2_core()` で委譲

---

## 2. 被膜摩擦フィールド

**影響ファイル**: `pair.py`

### ContactState追加フィールド

| フィールド | 型 | 用途 |
|-----------|---|------|
| `coating_z_t` | `np.ndarray(2,)` | 被膜接線履歴ベクトル |
| `coating_stick` | `bool` | 被膜stick/slip状態 |
| `coating_q_trial_norm` | `float` | 被膜trial forceノルム |
| `coating_dissipation` | `float` | 被膜摩擦散逸増分 |

### ContactConfig追加パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `coating_mu` | 0.0 | 被膜摩擦係数（0=無効） |
| `coating_k_t_ratio` | 0.5 | 被膜接線/法線ペナルティ比 |

---

## 3. solver_ncp統合

**影響ファイル**: `solver_ncp.py`

### Newton loop内の追加箇所

- **4e**: 被膜Coulomb摩擦力 `f_coat_fric`（被膜法線力の直後）
- **8b3**: 被膜摩擦接線剛性 `K_coat_fric`（被膜法線剛性の直後）

### 条件

`coating_mu > 0.0` かつ `_use_coating` が真の場合のみ有効。

---

## テスト結果

- **1761 fast テスト**: 全PASS
- 既存テストに回帰なし

---

## TODO

- [ ] 被膜摩擦の検証スクリプト作成（scripts/verify_coating_friction.py）
- [ ] 被膜塑性散逸モデル（バイリニア/弾完全塑性）
- [ ] 全テストのmm-ton-MPa移行（~100ファイルの定数変換）
