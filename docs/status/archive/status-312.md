# status-312: BC適用ベクトル化 + 責務分離違反修正 + MPC forループ排除

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-09
- **ブランチ**: `claude/check-status-todos-nmt8k`
- **テスト数**: 459 passed + 10 skipped + 1 xfailed（stress_contour既存バグ1件のみFAIL）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-311のTODO 4項目のうち3項目を実施:
1. BC適用のforループをNumPyベクトル化（`_zero_sparse_rows`ヘルパー）
2. MPC `_indep_to_reduced` 構築のforループをNumPy配列演算に置換
3. `strand_bending_oscillation.py` の責務分離違反修正（tolil+spsolve→LinearSolveProcess）

spluキャッシュとMPC triple productキャッシュは再評価の結果見送り（理由下記）。

---

## 実施内容

### 1. `_zero_sparse_rows` ベクトル化ヘルパー

CSR行列の指定行ゼロ化をforループからNumPy配列演算に置換。

| 項目 | 旧実装 | 新実装 |
|------|--------|--------|
| 処理 | `for d in fixed: data[indptr[d]:indptr[d+1]] = 0` | `data[offsets + within] = 0` |
| 計算量 | O(n_fixed) Pythonループ | O(nnz_in_fixed_rows) NumPy一括 |
| ボトルネック | 固定DOF数が数千以上の大規模問題 | なし（NumPy C拡張） |

**アルゴリズム**:
```python
starts = indptr[fixed]
ends = indptr[fixed + 1]
lengths = ends - starts
offsets = np.repeat(starts, lengths)
within = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
data[offsets + within] = 0.0
```

`_solve_standard` と `_solve_with_mpc` の両パスで使用。

### 2. MPC `_indep_to_reduced` ベクトル化

| 項目 | 旧実装 | 新実装 |
|------|--------|--------|
| mapping構築 | `for j, d in enumerate(indep): arr[d] = j` | `arr[indep] = np.arange(len(indep))` |
| fixed_reduced | `for d in fixed: if arr[d]>=0: append` | `rd = arr[fixed]; rd[rd>=0]` |

### 3. 責務分離違反修正

`strand_bending_oscillation.py` の静的NRソルバー内で、`tolil()` + forループ + `spsolve` を直接呼び出していた箇所を `LinearSolveProcess` に置換。

| 項目 | 旧 | 新 |
|------|-----|-----|
| BC適用 | `K.tolil(); for d: K[d,:]=0; K[:,d]=0; K[d,d]=1` | `LinearSolveProcess.process()` |
| ソルブ | `spla.spsolve(K, -R)` | `LinearSolveProcess.process()` |
| import | `scipy.sparse.linalg as spla` | 削除 |

### 4. 見送り項目の再評価

**splu symbolic factorizationキャッシュ**:
- pypardiso使用時は内部で自動キャッシュ済み
- scipy fallback時は接触ペア活性変化でスパースパターンが変わり、キャッシュ無効化が頻発
- 結論: 実装コスト > 期待効果。pypardiso利用を前提としTODOから除外

**MPC triple product `T.T @ K @ T` キャッシュ**:
- T不変でもKが毎NR反復変わるため積の再計算は不可避
- T自体が変わるのはUL参照配置更新時のみ（NR反復間では不変）
- 結論: スキップ不可。TODOから除外

---

## 変更ファイル

- `xkep_cae/contact/solver/_newton_steps.py`: `_zero_sparse_rows` 追加、`_solve_standard`/`_solve_with_mpc` ベクトル化
- `xkep_cae/contact/solver/tests/test_process.py`: TestZeroSparseRowsAPI(4件) + test_solve_with_fixed_dofs(1件) 追加
- `xkep_cae/numerical_tests/strand_bending_oscillation.py`: tolil+spsolve→LinearSolveProcess

---

## 再現手順

```bash
# ブランチ
git checkout claude/check-status-todos-nmt8k

# _zero_sparse_rows テスト
python -m pytest xkep_cae/contact/solver/tests/test_process.py::TestZeroSparseRowsAPI -v

# LinearSolve BCテスト
python -m pytest xkep_cae/contact/solver/tests/test_process.py::TestLinearSolveProcessAPI -v

# strand_bending_oscillation テスト
python -m pytest xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py -v

# 全体テスト
python -m pytest xkep_cae/ -v -k "not slow"

# lint
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/

# 契約チェック
python contracts/validate_process_contracts.py
```

---

## TODO

- [ ] 被膜幾何接線剛性（∂n/∂u, ∂s/∂u）の実装
- [ ] シース-素線接触統合（旧SheathModel/HEX8のProcess化）
- [ ] リスタート解析方式への移行
- [ ] 1000本撚線ベンチマーク: 現在のソルバーで大規模問題のプロファイリング

---

## 次の担当者向け

### 重要ポイント

1. **BC適用forループ完全排除済み**: `_zero_sparse_rows` はCSR/CSCの indptr を使って一括ゼロ化。標準/MPCの両パスで使用
2. **MPC mapping構築もベクトル化済み**: `_indep_to_reduced` と `fixed_reduced` のforループ排除
3. **責務分離違反修正**: strand_bending_oscillation.py の独自ソルバーロジックを LinearSolveProcess に統一。pypardiso自動検出の恩恵を受けるようになった
4. **splu/MPC cacheは見送り**: pypardiso利用が前提。Kが毎反復変わるためcache不可
5. **高速化フェーズ3弾の残課題**: 大規模問題での実測プロファイリングが次のステップ。ボトルネックが接触アセンブリからスパース求解にシフトしたかの確認が必要

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: テスト結果をそのまま記録
- [x] **再現手順記載**: コマンド列を明記
- [x] **回帰なし**: 459テスト合格（stress_contourは既存バグ）、契約違反0件
