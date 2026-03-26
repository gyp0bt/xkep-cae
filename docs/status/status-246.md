# status-246: 接触力・摩擦アセンブリのバッチベクトル化

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-26
**テスト**: 200+10s | 契約違反 1件（pre-existing C3） | 条例違反 0件
**ブランチ**: `claude/improve-calculation-speed-kL22D`

## 概要

接触力・摩擦力のアセンブリをPythonループからNumPyバッチ演算に移行。
beam oscillation テスト全体で **22.3% 高速化**（215s → 167s）。

## 変更内容

### 1. HuberContactForceProcess バッチ化 (`contact_force/strategy.py`)

- **evaluate()**: ペアデータのバッチ抽出 → ベクトル化 Huber → バッチ g_shape → `np.add.at` scatter
- **tangent()**: K_mat + K_geo の 4×4×3×3 ループを (N,12,12) バッチ行列で一括構築
- **ヘルパー追加**: `_huber_batch`, `_huber_deriv_batch`, `_extract_pair_arrays`, `_batch_hermite_coeffs`, `_batch_hermite_corrected_coeffs`, `_batch_dm_coeffs`, `_batch_shape_coeffs`
- K_st は StJacobian プロセス呼び出しのためペアごとループを維持

### 2. 摩擦アセンブリバッチ化 (`friction/_assembly.py`)

- **_assemble_friction_force**: ペアデータ一括抽出 → バッチ f_local 計算 → `np.add.at` scatter
- **_assemble_friction_tangent_stiffness**: D_t×g_t⊗g_t を (N,12,12) で一括構築 → COO 一括生成
- **_assemble_friction_geometric_stiffness**: M 行列バッチ計算（n⊗t1, t2⊗n, skew(t1) をバッチ演算）

### 3. LinearSolveProcess BC 適用改善 (`solver/_newton_steps.py`)

- CSC 行列への行単位ループを lil_matrix 変換 + CSC 変換に変更
- `_rhs[fixed] = 0.0` でバッチ RHS ゼロ化

## ベンチマーク（STA2: tee ログ保存済み）

| 項目 | 変更前 | 変更後 | 改善率 |
|------|--------|--------|--------|
| beam oscillation 13テスト | 215.15s | 167.16s | **22.3%** |
| SparseEfficiency warnings | 17,760 | 8,880 | **50%減** |
| マイクロベンチ evaluate (1000pairs) | 6.73ms | 0.54ms | **12.4x** |
| マイクロベンチ tangent (1000pairs) | 73.28ms | 4.67ms | **15.7x** |

**ログファイル**:
- `/tmp/log-bench-beam-before-*.log` — ベースライン
- `/tmp/log-bench-beam-after-*.log` — 改善後
- `/tmp/log-baseline-bench-*.log` — マイクロベンチマーク

## 再現手順

```bash
# ベースライン（main ブランチ）
python -m pytest tests/test_beam_oscillation.py -x -q -k "not test_render_produces" 2>&1 | tee /tmp/log-baseline.log

# 改善後（本ブランチ）
git checkout claude/improve-calculation-speed-kL22D
python -m pytest tests/test_beam_oscillation.py -x -q -k "not test_render_produces" 2>&1 | tee /tmp/log-improved.log
```

## 設計判断

1. **frozen dataclass のペア更新はループ維持**: NumPy 配列ではなく frozen dataclass を使うアーキテクチャ上、`_evolve_pair` はループ必須。数値計算のみをバッチ化。
2. **K_st はバッチ化せず**: `ComputeStJacobianProcess` がペアごとに呼ばれるため、プロセスアーキテクチャとの整合性を優先。
3. **COO 一括構築**: `(N,12,12)` の局所行列を `broadcast_to` + `ravel` で高速 COO 変換。

## 今後のロードマップ（計算高速化）

### Phase 2: カットバック削減（中期）
- [ ] 接触安定化ダンピング（gap≈0 近傍の velocity-dependent regularization）
- [ ] 接触 active set 履歴安定化（2↔3 振動パターン凍結）
- [ ] dt_grow damping 緩和（consecutive_good > 2 の抑制が強すぎる）

### Phase 3: 線形ソルバー改善（中～長期）
- [ ] スパースパターンキャッシュ（接触トポロジ不変時に値のみ更新）
- [ ] 反復ソルバー導入（DOF > 10000 で GMRES + ILU(0) 自動切替）

### Phase 4: GPU 対応（S7）
- [ ] CuPy sparse 演算の GPU オフロード（Phase 1 のバッチ化が前提）

## STA2 再現性ルール（強化提案）

> 担当者違いで再現性が取れない成果を防ぐため、以下を追加提案:
>
> 1. **ベンチマーク条件の記録**: テスト名、ブランチ、コミットハッシュ、実行環境を tee ログに記録
> 2. **変更前ベースラインの先行取得**: 改善テスト前に必ず `git stash` でベースライン計測
> 3. **Process アーキテクチャでの計測埋め込み**: ProcessMetaclass の profiling をベンチマーク自動記録に拡張検討

## TODO

- [ ] Phase 2 実装（接触安定化ダンピング）
- [ ] n_periods=30 Hermite ON 完走（現在 frac=0.9838）
- [ ] NR 力収束速度改善
- [ ] Hermite 非局所 ∂g/∂u 対応（4ノードペア外の DOF 結合）
- [ ] 1000本スケールアップ向け反復ソルバー導入
