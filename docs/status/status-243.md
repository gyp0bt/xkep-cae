# status-243: frozen-m 解消 + λ自動推定定数設定可能化

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-26
**テスト**: 190+10s+8+9+7+10 | 契約違反 1件（既存C3） | 条例違反 0件

---

## 概要

status-242 の TODO 3項目のうち2項目を実施:
1. **凍結接線 (frozen-m) の解消**: StJacobian と strategy.py に ∂m/∂u のローカル寄与を追加
2. **λ自動推定の改善**: `lm_auto_c` パラメータ追加（定数 c のハードコード除去）+ スケール不変性の文書化
3. ~~摩擦 Hermite 完全対応~~: frozen-m 解消が前提 → 次ステップ

---

## 1. frozen-m 解消

### 問題

Hermite 接触では接線ベクトル m = Catmull-Rom(隣接要素方向平均) が節点座標に依存する。
しかし StJacobian の `_compute_rhs_hermite()` は **m を定数として微分**していた（frozen-m 近似）。

これにより:
- ds/du, dt/du が過大評価（直線セグメントで Hermite dh=-1.5 vs 正解 dh=-1.0）
- K_st が 1.5² = 2.25 倍に膨張
- FD テストで **33% の不整合**（status-238/242 で確認）

### 解法

∂m/∂x のローカル寄与（4ノードペア内で閉じる分）を計算し、h/dh を補正:

```
h_eff = H_Ak(s) + H10(s)·∂mA0/∂x_Ak + H11(s)·∂mA1/∂x_Ak
dh_eff = H_Ak'(s) + H10'(s)·∂mA0/∂x_Ak + H11'(s)·∂mA1/∂x_Ak
```

∂m/∂x 係数は節点の接続数（count）から決定:
- **端点 (count=1)**: dm_self = -1（左端）or +1（右端）、dm_cross = ±1
- **内部 (count=2)**: dm_self = 0（±I が相殺）、dm_cross = ±1/2
- **非局所項** (∂m/∂x_prev, ∂m/∂x_next): 4ノードペア外 → 省略

### 端点での完全修正の証明

端点（count=1）の場合:
```
coeff_A0 = H00(s) + H10(s)·(-1) + H11(s)·(-1)
         = H00 - H10 - H11 = 1-s  ← 線形と同一 ✓

dh_A0 = H00'(s) + H10'(s)·(-1) + H11'(s)·(-1)
      = H00' - H10' - H11' = -1  ← 線形と同一 ✓
```

### FD テスト結果

| テスト | 変更前 rel_err | 変更後 rel_err |
|--------|---------------|---------------|
| K_st なし | 100% | 100% (K_st 必須) |
| 線形 + K_st | 0.00% | 0.00% (変化なし) |
| **Hermite + K_st** | **33%** | **0.00%** ✓ |

端点のみの配置で FD 整合を完全達成。

### 内部ノードでの部分改善

内部ノード（count=2）では非局所項の省略により完全修正ではないが、
dh が -1.5 → -1.375 に改善（正解 -1.0、frozen 比で 25% の誤差削減）。

### 補正箇所

| 層 | 補正内容 |
|---|---------|
| 力評価 (`evaluate`) | g_shape の形状関数係数を dm 補正 |
| K_mat / K_geo (`tangent`) | coeffs を dm 補正 |
| K_st (`_add_kst_contact`) | coeffs, dc_ds, dc_dt を dm 補正 |
| StJacobian | h, dh を dm 補正（2×2 系、1×1 フォールバック両方） |

---

## 2. λ自動推定の改善

### スケール不変性の発見

status-242 の結果を再分析:

| 材料 | E | k_pen (auto) | λ=c/E | λ·k_pen | λ·E |
|------|---|-------------|-------|---------|-----|
| 鉄鋼 | 200e3 | 3.68 | 1.00e-4 | 0.0004 | 20 |
| 銅 | 120e3 | 2.21 | 1.67e-4 | 0.0004 | 20 |
| アルミ | 70e3 | 1.29 | 2.86e-4 | 0.0004 | 20 |

**発見**: auto k_pen ∝ E のため、λ·k_pen と λ·E は材料間で定数。
LM 正則化は構造・接触 DOF 両方に同一の相対補正を適用する。
c=20/E 公式は**スケール不変**。

アルミの悪化（Δfrac=-0.026）はスケーリング問題ではなく、
c=20 の最適値が幾何依存（要素数、アスペクト比等）であることを示唆。

### 実装

- `lm_auto_c` パラメータを追加（`SolverSetupInput`, `DynamicThreePointBendContactJigConfig`）
- ハードコード `20.0` → `lm_auto_c`（デフォルト 20.0）
- 幾何別のチューニングが可能に

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/geometry/_compute.py` | `_compute_node_counts()`, `_compute_dm_coeffs()` 追加 |
| `xkep_cae/contact/geometry/_st_jacobian.py` | `StJacobianInput` に `dm_A/dm_B` 追加、`_compute_rhs_hermite` dm 補正、1×1 フォールバック dm 補正 |
| `xkep_cae/contact/contact_force/strategy.py` | `_hermite_corrected_coeffs()` 追加、`evaluate`/`tangent`/`_add_kst_contact` で dm 補正使用 |
| `xkep_cae/core/data.py` | `lm_auto_c` フィールド追加 |
| `xkep_cae/contact/solver/process.py` | `lm_auto_c` パラメータ使用 |
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | `lm_auto_c` パラメータ追加・伝播 |

---

## TODO

- [ ] **摩擦アセンブリの Hermite 完全対応**: use_hermite=False デフォルトの解消（frozen-m 解消完了により前提条件クリア）
- [ ] **内部ノードの非局所 dm 項**: 隣接要素ノードへの ∂m/∂x（現在は 4ノードペア外として省略）
- [ ] **Hermite + frozen-m 解消の NR 収束テスト**: n_periods=30 で freeze=F, K_st=ON, dm 補正有りの収束検証
- [ ] **lm_auto_c の幾何別最適値調査**: 要素数・アスペクト比依存の c 値テーブル作成

---

## 設計上の懸念

1. **内部ノードの残余誤差**: dm 補正後も内部ノードでは dh=-1.375（正解 -1.0）。非局所項（4ノードペア外のDOF結合）を追加すれば完全修正可能だが、アーキテクチャ変更が必要
2. **力ベクトルの dm 補正**: evaluate() の g_shape にも dm 補正を適用。接触力の節点配分が変わるため、既存の収束挙動に影響する可能性

---

## 開発運用メモ

- STA2 防止: FD テスト結果は tee でログ保存済み。Hermite rel_err=0.00% は再現可能
- frozen-m 解消はローカル補正のみ。アーキテクチャ変更なし
- test_beam_oscillation の1件失敗は既存（変更前も同一失敗を確認）

---
