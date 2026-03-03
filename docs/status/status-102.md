# Status 102: S3 NCP収束改善 — 初期貫入オフセット + 安全弁line search + energy convergence

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-03
**ブランチ**: `claude/s3-implementation-fc3ZD`
**テスト数**: 1916（fast: 1542 / slow: 374）※変更なし（既存テスト全パス）

## 概要

S3（19本以上のNCP収束）に向けた基盤改良。3つの主要機能を実装:

1. **初期貫入オフセット**（LS-DYNA IGNORE=1 相当）
2. **安全弁 line search**（CR梁の大回転NR特性に適合）
3. **エネルギー収束判定**（|du·R|/|du₀·R₀| < 1e-10）

## 実装詳細

### 1. 初期貫入オフセット（pair.py）

ヘリックス離散化の chord-sag 効果により、外層ほど初期貫入が深刻になる問題を解決。

- `ContactPair.gap_offset`: 初期貫入量を保存
- `ContactManager.store_initial_offsets(node_coords)`: t=0 での全ペアのギャップを計測し、負（貫入）のものを gap_offset として保存
- `update_geometry()`: `g_effective = g_raw - gap_offset` を適用

**原理**: 7本撚線では chord-sag ≈ 0.04mm/side（16要素/ピッチ）、91本撚線では外層 lay_radius=10mm で chord-sag ≈ 0.19mm/side。wire_diameter×2 にしても解消しない。

### 2. 安全弁 line search（solver_ncp.py）

CR梁の大回転問題では NR の残差が一時的に増加するのが正常な挙動。完全な残差減少を要求する line search は収束を阻害する。

- `_ncp_line_search()`: NaN/Inf または 1000倍以上の残差増大のみを防止
- `newton_raphson_contact_ncp()` に `use_line_search` / `line_search_max_steps` パラメータ追加
- 3つの更新ブランチ（Mortar, 摩擦, 法線のみ）全てに適用

### 3. エネルギー収束判定（solver_ncp.py）

NCP ソルバーに basic NR ソルバーと同じエネルギー基準を追加。

- `|du · R_u| / |du₀ · R₀| < 1e-10` かつ NCP 収束時に step 完了
- CR梁16+要素で力残差が収束しにくい場合でもエネルギー基準で収束判定

### 4. ベンチマーク NCP 統合（wire_bending_benchmark.py）

- `use_ncp=True` で NCP ソルバーを使用可能
- `n_elems_per_pitch=16`（ユーザー要件: 1ピッチあたり最低16要素）
- `n_elems_per_strand` による後方互換維持
- 初期貫入オフセットをソルバー実行前に自動計算
- S3 ContactConfig パラメータ群（augmented_threshold, saddle_regularization 等）

### 5. S3 ContactConfig パラメータ（pair.py）

- `augmented_threshold`: 鞍点系拡大直接法の閾値
- `saddle_regularization`: 鞍点系(2,2)ブロックの正則化δ
- `ncp_active_threshold`: NCP活性セットのヒステリシス閾値
- `lambda_relaxation`: λ更新の under-relaxation 係数

## テスト結果

- NCP テスト: 49 パス
- 接触テスト全体: 560 パス（0 失敗）
- ベンチマークテスト: 1 パス（lightweight, 後方互換確認）

## 前セッションの知見（再実装に反映）

- **ALソルバーは曲げで構造的に収束しない**: Inner/outer loop の悪循環。NCP + Mortar が正解
- **CR梁 16+要素の NR 収束特性**: 残差一時増加は正常。line search は「安全弁」方式が最適
- **4要素/1ピッチでは 22ステップ目で収束失敗**: 16要素/ピッチ以上が必要（幾何表現）
- **接触は0ペアで活性化しなかった**（前回テスト）: 初期貫入オフセットの適用前

## TODO

- [ ] NCP + Mortar + 16要素/ピッチ で7本撚線の曲げ収束確認
- [ ] 19本撚線でのNCP収束確認（S3主目標）
- [ ] 37/61/91本の段階的収束確認
