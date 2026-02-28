# status-085: Phase C6-L5 Mortar 離散化の実装

[← README](../../README.md) | [← status-index](status-index.md)

- **日付**: 2026-02-28
- **テスト数**: 1797（fast 1507 / slow 290）（+9: Mortar テスト）
- **ブランチ**: claude/execute-status-todos-uhlKc

## 概要

NCP Semi-smooth Newton ソルバーに **Mortar 離散化** を実装。従来の point-to-point（PtP）接触に加え、slave 側節点ベースの Lagrange 乗数を用いた連続的な接触圧分布を実現。

## 背景・動機

PtP 接触では：
1. **セグメント境界の不連続**: 接触力がペア単位で独立に計算され、セグメント境界で力が不連続
2. **ロッキング**: ペアベースの Lagrange 乗数が要素サイズに依存し、細かいメッシュでロッキングの恐れ
3. **精度向上の上限**: line contact + NCP + Alart-Curnier でも、離散化精度はペア数に依存

Mortar 離散化により、slave 側の形状関数による重み付き平均ギャップを使い、セグメント境界の連続性を保証。

## 実装詳細

### Mortar 離散化の定式化

slave（A）側の節点 k に対して：
- **重み付きギャップ**: g̃_k = Σ_gp Φ_k(s_gp) · w_gp · gap(s_gp)
- **拘束 Jacobian**: G_mortar[k, :] = Σ_gp Φ_k(s_gp) · w_gp · ∂gap/∂u
- **Mortar 基底関数**: Φ_0(s) = 1-s, Φ_1(s) = s（梁の線形形状関数）

NCP active set は Mortar 節点ベースで判定：
- p_n_k = max(0, λ_k + k_pen · (-g̃_k))
- active: p_n_k > 0

### 新規モジュール: `xkep_cae/contact/mortar.py`

| 関数 | 説明 |
|------|------|
| `identify_mortar_nodes(manager, active_indices)` | 活性ペアから slave 側節点を抽出 |
| `build_mortar_system(manager, active_indices, mortar_nodes, ...)` | G_mortar, g_mortar 構築 |
| `compute_mortar_contact_force(manager, active_indices, mortar_nodes, lam_mortar, ...)` | Mortar 接触力ベクトル |
| `compute_mortar_p_n(mortar_nodes, lam_mortar, g_mortar, k_pen)` | 節点ベース p_n（active set 判定用） |

### solver_ncp.py の変更

- `newton_raphson_contact_ncp()` に `use_mortar: bool = False` パラメータ追加
- `_use_mortar` フラグ: `(use_mortar or config.use_mortar) and _line_contact`（Mortar は line contact 必須）
- NR ループ内分岐:
  - Step 2m: `build_mortar_system()` で G_mortar, g_mortar 構築
  - Step 3: Mortar 節点ベースの NCP active set 判定
  - Step 4a: `compute_mortar_contact_force()` で接触力計算
  - Step 6: Mortar NCP 残差 C_mortar 計算
  - Step 7: 収束判定に C_mortar ノルムを使用
  - Steps 9-10: Mortar 鞍点系ソルブ（active mortar rows を使って `_solve_saddle_point_contact()` を呼出）
  - Step 11m: Mortar 乗数更新 + 非活性ゼロ化 + λ ≥ 0 射影
- Mortar 使用時は per-pair λ ゼロ化をスキップ

### pair.py の変更

- `ContactConfig` に `use_mortar: bool = False` フィールド追加

### λ リマッピング

反復中に active pair set が変化すると mortar 節点セットも変わる。新旧節点セットの対応を取り、既存の λ 値を引き継ぐリマッピングロジックを実装。

## テスト（9件）

### TestMortarBasic（3件）
- `test_mortar_converges`: Mortar 有効で NCP ソルバーが収束
- `test_mortar_lambda_nonneg`: Mortar 乗数が非負
- `test_config_mortar_propagated`: `ContactConfig.use_mortar` フラグの伝搬確認

### TestMortarWeightedGap（3件）
- `test_identify_mortar_nodes`: slave 節点の正しい抽出
- `test_mortar_gap_uniform`: 均一ギャップでの重み付きギャップ値検証
- `test_mortar_p_n_active`: active set 判定の正確性

### TestMortarMultiSegment（1件）
- `test_parallel_beams_converge`: 複数セグメントでの Mortar 収束

### TestMortarVsPtP（2件）
- `test_mortar_vs_line_contact_same_direction`: Mortar と PtP で同方向荷重時の解比較
- `test_mortar_requires_line_contact`: line contact 無効時に Mortar が自動 fallback

### テスト設計方針
- 全テスト 1 秒以内（合計 1.06 秒）
- n_gauss=2, n_load_steps=3 の軽量設定
- 簡易バネ系モデルを使用

### リグレッション確認
- NCP 全テスト（47件）: PASSED（3.24s）
- Alart-Curnier 摩擦テスト: PASSED

## 確認事項・今後の課題

- [ ] Line contact + Mortar + Alart-Curnier 摩擦の3重統合テスト
- [ ] 多ペア環境（7本撚り）での Mortar 収束性能評価
- [ ] Phase S2: CPU 並列化への進行
