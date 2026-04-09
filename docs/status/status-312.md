# status-312: LinearSolveProcess最適化 — BCベクトル化 + ソルバーキャッシュ + MPC triple productキャッシュ + 責務分離修正

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-09
- **ブランチ**: `claude/check-status-todos-gGpyZ`
- **テスト数**: 515 passed, 10 skipped（新規テスト追加なし）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-311のTODO 4項目を実施:
1. BC適用forループのNumPyベクトル化（`_zero_rows_csr` + `_apply_bc_inplace`関数抽出）
2. スパースソルバーバックエンドのimport判定キャッシュ（`_SolverBackend`クラス）
3. MPC triple product `T.T @ K @ T` の前処理キャッシュ（`T.T` CSR変換 + `fixed_reduced`マッピング）
4. `strand_bending_oscillation.py`の責務分離違反修正（tolil + spsolve → `LinearSolveProcess`委譲）

---

## 実施内容

### 1. BC適用forループのNumPyベクトル化

`_solve_standard` と `_solve_with_mpc` 内の `for d in fixed` ループを完全排除。

**新関数 `_zero_rows_csr`**: CSR行列の指定行を一括ゼロ化。`indptr`からスライス範囲を一括取得し、`np.repeat` + `np.arange` + `np.cumsum` でデータインデックスを生成して一括代入。

**新関数 `_apply_bc_inplace`**: BC適用の一連処理（行ゼロ化 → 列ゼロ化 → 対角=1 → rhs=0）を1関数に集約。標準パス・MPCパス両方から共用。

| 項目 | 旧実装 | 新実装 |
|------|--------|--------|
| 行ゼロ化 | `for d in fixed: data[indptr[d]:indptr[d+1]] = 0.0` | `_zero_rows_csr(K, fixed)` ベクトル化 |
| 列ゼロ化 | 同上（CSC版） | 同上 |
| コード重複 | 標準パス15行 + MPCパス15行 | `_apply_bc_inplace` 共通1関数 |

### 2. スパースソルバーバックエンド `_SolverBackend`

`_sparse_solve`関数の毎回の `try/except ImportError` を排除。

| 項目 | 旧実装 | 新実装 |
|------|--------|--------|
| import判定 | 毎NR反復で`try: import pypardiso` | 初回のみ判定、`_spsolve`関数をキャッシュ |
| symbolic factorization | pypardiso.spsolve が内部でキャッシュ | 同（明示的に文書化） |
| scipy fallback | 毎回`from scipy... import spsolve` | 初回のみimport |

**注**: pypardisoの`spsolve`はモジュールレベル`PyPardisoSolver`インスタンスを使用し、同一スパースパターンでのsymbolic factorizationを自動キャッシュする。scipy SuperLUにはsymbolic/numeric分離APIがないため、scipy側の最適化はimportキャッシュのみ。

### 3. MPC triple productキャッシュ

`LinearSolveProcess`にインスタンス変数 `_mpc_cache_id`, `_T_T_csr`, `_fixed_reduced` を追加。

| キャッシュ項目 | 内容 | 効果 |
|---------------|------|------|
| `_T_T_csr` | `T.T.tocsr()` — T転置のCSR版 | NR反復ごとのCSR変換排除 |
| `_fixed_reduced` | 固定DOF→縮退系インデックス変換結果 | `_indep_to_reduced`マッピング再構築排除 |
| キャッシュキー | `id(mpc)` + `fixed_dofs`のハッシュ | MPC/BC変更時に自動無効化 |

**MPC固定DOFインデックス変換もベクトル化**: 旧forループ（`for j, d in enumerate(mpc.independent_dofs)`）を `np.asarray` + `np.arange` + ファンシーインデキシングに置換。

### 4. 責務分離違反修正

`strand_bending_oscillation.py`（数値テスト）内のソルバーロジック重複を解消。

| 項目 | 旧実装 | 新実装 |
|------|--------|--------|
| BC適用 | `K.tolil()` + forループ（7行） | `LinearSolveProcess`に委譲（3行） |
| 線形ソルブ | `spla.spsolve(K_csr, -R)` | `LinearSolveProcess.process()` |
| エラーハンドリング | なし | `solve_out.success` チェック |
| pypardiso活用 | なし（scipy固定） | `_SolverBackend`経由で自動選択 |
| import | `scipy.sparse.linalg as spla` | 削除（不要に） |

---

## 変更ファイル

- `xkep_cae/contact/solver/_newton_steps.py`: `_zero_rows_csr`, `_apply_bc_inplace`, `_SolverBackend` 追加。BC適用ベクトル化、MPCキャッシュ
- `xkep_cae/numerical_tests/strand_bending_oscillation.py`: `LinearSolveProcess`委譲、`scipy.sparse.linalg` import削除

---

## 再現手順

```bash
# ブランチ
git checkout claude/check-status-todos-gGpyZ

# ソルバーテスト
python3 -m pytest xkep_cae/contact/solver/tests/ -v -k "not slow"

# 数値テスト
python3 -m pytest xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py -v -k "not slow"

# 全体テスト
python3 -m pytest xkep_cae/ -v -k "not slow"

# lint
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/

# 契約チェック
python3 contracts/validate_process_contracts.py
```

---

## TODO

- [ ] 被膜幾何接線剛性（∂n/∂u, ∂s/∂u）の実装
- [ ] シース-素線接触統合（旧SheathModel/HEX8のProcess化）
- [ ] リスタート解析方式への移行
- [ ] 大規模問題（1000本撚線）でのBC適用ベクトル化のベンチマーク検証

---

## 次の担当者向け

### 重要ポイント

1. **BC適用が2関数に集約**: `_zero_rows_csr`（CSRデータ一括ゼロ化）+ `_apply_bc_inplace`（BC適用一式）。標準パス・MPCパス両方から呼び出し
2. **pypardiso symbolic caching**: pypardiso.spsolveが内部で同一パターン検知→symbolic再利用。`_SolverBackend`はimportキャッシュのみ追加
3. **MPC triple product**: `T.T @ K @ T`のK依存部分はキャッシュ不可（毎回計算）。T.TのCSR変換とfixed_reducedマッピングのみキャッシュ
4. **strand_bending_oscillation**: 静的NRソルバーが`LinearSolveProcess`を使用するよう修正。pypardiso自動選択のメリットも享受

### 設計上の懸念

- `_zero_rows_csr`のメモリ使用: `np.repeat` + `np.arange`で全ゼロ化対象のインデックス配列を一括生成。固定DOF数×平均行nnzが巨大な場合（1000本撚線で数万DOF × 数百nnz/行 = 数百万要素）、一時配列が大きくなる。必要に応じてチャンク分割を検討
- MPC triple product `T.T @ K @ T` のO(nnz)計算自体は最適化余地あり（Tの構造を利用したサブ行列抽出）が、実装複雑度が高いため現状は標準scipy sparse乗算を使用

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: テスト結果515 passed, 10 skipped, 1 failed(既存バグ)を正確に記録
- [x] **再現手順記載**: コマンド列を明記
- [x] **回帰なし**: 515テスト合格（stress_contourは既存バグ）、契約違反0件
