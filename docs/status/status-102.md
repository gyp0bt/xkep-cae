# Status 102: S3 NCP収束改善 — 初期貫入オフセット + Modified NR + ステップ二分法

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-03
**ブランチ**: `claude/s3-implementation-fc3ZD`
**テスト数**: 1916（fast: 1542 / slow: 374）※変更なし（既存テスト全パス）

## 概要

S3（19本以上のNCP収束）に向けた基盤改良。6つの主要機能を実装:

1. **初期貫入オフセット**（LS-DYNA IGNORE=1 相当）
2. **エネルギー収束判定**（|du·R|/|du₀·R₀| < 1e-10）
3. **ステップ二分法**（NR不収束時に荷重増分を自動二分）
4. **Modified NR**（N反復後に接線凍結、線形収束だが安定）
5. **適応 line search**（Full NR: 安全弁 / Modified NR: 厳密バックトラッキング）
6. **ベンチマーク NCP 統合**（16要素/ピッチ + 全パラメータ）

## 実装詳細

### 1. 初期貫入オフセット（pair.py）

ヘリックス離散化の chord-sag 効果により、外層ほど初期貫入が深刻になる問題を解決。

- `ContactPair.gap_offset`: 初期貫入量を保存
- `ContactManager.store_initial_offsets(node_coords)`: t=0 での全ペアのギャップを計測し、負（貫入）のものを gap_offset として保存
- `update_geometry()`: `g_effective = g_raw - gap_offset` を適用

**原理**: 7本撚線では chord-sag ≈ 0.04mm/side（16要素/ピッチ）、91本撚線では外層 lay_radius=10mm で chord-sag ≈ 0.19mm/side。wire_diameter×2 にしても解消しない。

### 2. エネルギー収束判定（solver_ncp.py）

NCP ソルバーに basic NR ソルバーと同じエネルギー基準を追加。

- `|du · R_u| / |du₀ · R₀| < 1e-10` かつ NCP 収束時に step 完了
- CR梁16+要素で力残差が収束しにくい場合でもエネルギー基準で収束判定

### 3. ステップ二分法（solver_ncp.py）

- `max_step_cuts` パラメータ追加（デフォルト0=後方互換）
- NR不収束時に荷重増分を自動二分して再試行
- チェックポイント保存でロールバック可能
- 最大 2^max_step_cuts 分割（3→8分割）

### 4. Modified NR + 適応 line search（solver_ncp.py）

CR梁16要素/ピッチの **NR振動発散** に対する根本対策。

- `modified_nr_threshold` パラメータ: N反復後に接線剛性を凍結
- Full NR モード（iter < N）: `diverge_factor=1000`（安全弁、一時的残差増加を許容）
- Modified NR モード（iter >= N）: `diverge_factor=1.5`（厳密バックトラッキング）
- 線形収束（二次ではない）だが、振動発散を構造的に防止

### 5. ベンチマーク NCP 統合（wire_bending_benchmark.py）

- `use_ncp=True` で NCP ソルバーを使用可能
- `n_elems_per_pitch=16`（ユーザー要件: 1ピッチあたり最低16要素）
- `n_elems_per_strand` による後方互換維持
- 初期貫入オフセットをソルバー実行前に自動計算
- `max_step_cuts=3`, `modified_nr_threshold=5` デフォルト

### 6. S3 ContactConfig パラメータ（pair.py）

- `augmented_threshold`: 鞍点系拡大直接法の閾値
- `saddle_regularization`: 鞍点系(2,2)ブロックの正則化δ
- `ncp_active_threshold`: NCP活性セットのヒステリシス閾値
- `lambda_relaxation`: λ更新の under-relaxation 係数

## テスト結果

- NCP テスト: 50 パス
- ベンチマークテスト: 1 パス（lightweight, 後方互換確認）

## 7本撚線 16要素/ピッチ ベンチマーク結果

| 設定 | 結果 |
|------|------|
| 7本撚線, 16要素/ピッチ, 45° 曲げ, 45ステップ | Phase 1: Step 1-10 収束 (frac=0.175), Step 11 (frac=0.178) 不収束 |
| max_iter=200, max_step_cuts=3, modified_nr_threshold=5 | 二分法3回発動、Modified NR で線形収束 |
| **接触活性ペア**: 0/276（全ステップ） | 純粋な構造NR問題（接触なし） |

### 収束パターン

| Step | Frac | Iters | 方式 |
|------|------|-------|------|
| 1-4 | 0.02-0.09 | 5-9 | Full NR + energy |
| 5-7 | 0.11-0.16 | 12-32 | Full NR + energy |
| 8 | 0.167 | 69 | Full NR → Modified NR（二分法で生成） |
| 9 | 0.172 | 152 | Full NR → Modified NR（二分法で生成） |
| 10 | 0.175 | 192 | Full NR → Modified NR（二分法で生成） |
| 11 | 0.178 | >200 | 不収束（限界点の可能性） |

### 知見

- **frac≈0.178（~8°曲げ）に数値的困難点**がある: サブステップが近づくほど反復数が指数増加
- **接触は1回も活性化しない**: 初期貫入オフセット後、曲げ荷重では圧縮側の追加貫入が不足
- **Modified NR は効果的**: Step 8 で Full NR は振動発散するが、MNR（iter ≥ 5）で単調減少→収束
- **適応 line search が鍵**: Full NR の安全弁（1000x）→ MNR の厳密LS（1.5x）の組み合わせ

## 前セッションの知見（再実装に反映）

- **ALソルバーは曲げで構造的に収束しない**: Inner/outer loop の悪循環。NCP + Mortar が正解
- **CR梁 16+要素の NR 収束特性**: 残差一時増加は正常。Modified NR + 適応 LS が最適
- **4要素/1ピッチでは 22ステップ目で収束失敗**: 16要素/ピッチ以上が必要（幾何表現）

## TODO

- [ ] frac≈0.178 の困難点の原因調査（限界点/分岐解析）
- [ ] ステップ数増加（90+）または弧長法で困難点突破
- [ ] 接触活性化の条件調査（曲率駆動の接触が必要か？）
- [ ] 19本撚線でのNCP収束確認（S3主目標）
- [ ] 37/61/91本の段階的収束確認
