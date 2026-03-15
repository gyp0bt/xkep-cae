# status-140: mm-ton-MPa単位系移行 + 被膜Kelvin-Voigt粘性減衰 + k_pen材料ベース強制

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-08
**テスト数**: 2271（変更なし）

---

## 概要

被膜接触モデルの物理的妥当性を改善する3つの構造的変更を実施。

1. **mm-ton-MPa単位系移行**: ベンチマーク基盤のパラメータをSI（m, Pa）からmm-ton-MPa（mm, MPa）に変換
2. **Kelvin-Voigt粘性減衰**: 被膜スプリングに粘性ダッシュポット項を追加（f = kδ + cδ̇）
3. **k_pen材料ベース自動推定の構造的強制**: 手動k_pen設定パスを非推奨化、beam_Eからの材料ベース導出を標準化

---

## 1. mm-ton-MPa単位系移行

**影響ファイル**: `wire_bending_benchmark.py`, `verify_coating_contact_convergence.py`, `test_inp_metadata_validation.py`

### 変換一覧

| パラメータ | 旧値 (SI) | 新値 (mm-ton-MPa) |
|---|---|---|
| `_DEFAULT_E` | 200e9 Pa | 200e3 MPa |
| `_WIRE_D` | 0.002 m | 2.0 mm |
| `pitch` | 0.040 m | 40.0 mm |
| `g_off` | 1e-5 m | 0.01 mm |
| `broadphase_margin` | 0.01 m | 10.0 mm |
| 被膜E（デフォルト） | 1e8 Pa | 100 MPa |

### 表示コード変更

`* 1000`（m→mm変換）を全て除去。mm-ton-MPa系では座標が既にmm単位のため変換不要。

---

## 2. Kelvin-Voigt粘性減衰

**影響ファイル**: `pair.py`, `solver_ncp.py`

### 物理モデル

被膜を弾性スプリング + 粘性ダッシュポットの並列結合（Kelvin-Voigt体）としてモデル化:

```
f_coat = k * δ + c * δ̇
```

- `k = E_coat / t_coat` — Winkler基盤剛性 [MPa/mm]
- `c` — 粘性減衰係数 [MPa·s/mm]
- `δ̇ ≈ (δ - δ_prev) / dt` — 後退差分近似
- `dt = load_frac - load_frac_prev` — 荷重増分幅

### 接線剛性

後退Euler離散化により、接線剛性は弾性 + 粘性の合算:
```
K_eff = (k + c/dt) * N^T (n ⊗ n) N
```

### 新規フィールド

- `ContactConfig.coating_damping`: 粘性減衰係数 [MPa·s/mm]
- `ContactState.coating_compression_prev`: 前ステップ被膜圧縮量

---

## 3. k_pen材料ベース自動推定の構造的強制

**影響ファイル**: `pair.py`, `solver_ncp.py`, `solver_hooks.py`, `wire_bending_benchmark.py`

### 設計変更

- `ContactConfig.k_pen_mode` デフォルトを `"manual"` → `"beam_ei"` に変更
- `ContactConfig.k_pen_scale` デフォルトを `1.0` → `0.1` に変更（無次元スケール係数）
- ソルバー内の判定ロジックを `k_pen_mode` ベースから `beam_E > 0` ベースに統一
- `beam_E` 未設定時は `k_pen_scale` をフォールバック使用（DeprecationWarning）
- ベンチマークでは常に `beam_E`, `beam_I`, `beam_A` を設定（`auto_kpen` パラメータ無効化）

### k_coat自動導出

`coating_stiffness` が未指定（≤0）の場合、被膜ヤング率と厚さから自動計算:
```python
k_coat = E_coat / t_coat  # Winkler基盤モデル
```

---

## テスト結果

- **1761 fast テスト**: 全PASS
- DeprecationWarning: `k_pen_scale >= 1.0` 使用時に警告出力（後方互換維持）

---

## TODO

- [ ] 全テストのmm-ton-MPa移行（~100ファイルの定数変換）
- [ ] 被膜接触の摩擦モデル統合（梁-梁、被膜-被膜、被膜-梁で摩擦係数を分離管理）
- [ ] k_pen手動設定の完全廃止（DeprecationWarning → エラー化）
- [ ] 被膜塑性散逸モデル（バイリニア/弾完全塑性）
- [ ] 検証スクリプトの物理的に妥当な被膜剛性（E/t ≈ 1000 MPa/mm）での収束確認
