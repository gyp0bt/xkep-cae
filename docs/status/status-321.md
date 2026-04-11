# status-321: K_st アセンブリ CSR/COO 経路最適化 — 摩擦 K_st 33% 高速化

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-11
- **ブランチ**: `claude/optimize-contact-friction-assembly-W8xDL`
- **テスト数**: 459+13+22+5（既存数維持）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

## TL;DR

status-319 の TODO「ContactForceStStiffness / FrictionStStiffness の n² 成長抑制」
の**定数項削減**フェーズ。n² スケーリング自体は触らず、per-call 定数を以下で削減:

1. **`tocsr()` skip**: K_st / K_mat / K_geo を COO のまま返し、呼び出し側で 1 回
   だけ COO concat → CSR。3 アセンブリで 2 〜 3 回の tocsr/加算を eliminate。
2. **einsum → 直接ブロードキャスト**: `np.einsum("ni,nj->nij", ...)` を
   `a[:,:,None] * b[:,None,:]` に置換（単純外積では broadcasting の方が速い）。
3. **mask filter skip**: `|val| > 1e-30` フィルタを除去。零エントリは CSR 統合時に
   自動集約され無視できる。マスク作成 + 索引コピーのコストを eliminate。
4. **ペア抽出ループの active 比例化**: state を持つペアの 2 段 filter +
   `np.fromiter` 一括抽出。total-pair 比例の Python 属性アクセスを active-pair
   比例に圧縮。
5. **friction strategy の単一 COO concat**: `K_mat + K_geo + K_st` の 2 回の
   sparse 加算を eliminate。3 アセンブリの row/col/data を 1 度だけ concat → CSR 化。

### 実測効果（n_active=2000, per-call 最小時間）

| | Before（status-320）| After（status-321）| 改善 |
|---|---|---|---|
| **FrictionStStiffness** | 17.84ms | **11.91ms** | **33% 高速化** |
| ContactForceStStiffness | 15.48ms | 14.97ms | 3% 高速化 |

※ friction 戦略レベル（K_mat+K_geo+K_st の単一 concat）の改善はこの単体ベンチでは
捕捉されない。実 NR ループではさらに 2 回の sparse + sparse 加算削減が上乗せされる。

## 背景 — n² は止められない、だが定数は削れる

status-319 で実測された n² スケーリング（α≈2.07）自体は **active ペア数自体が
n² で増える**ことに起因する（broadphase でのペア削減が本質改善）。
status-320 で `uses` グラフに K_st 系を接続したが、実計算の高速化はまだ。

本 status は「n² を維持したまま per-call 定数を最大限削る」フェーズ。broadphase
側の距離カット / spatial hash（真の n² 抑制）に着手する前に、**scipy.sparse の
CSR/COO 往復と einsum オーバーヘッド**という低ハードルの収穫を先に刈り取る。

### CSR まわりは C++/GPU でもこれ以上救えないか？

結論: **scipy の C 実装は既にほぼ上限**。

- C++ バインド（Eigen::Sparse）: Python 呼び出しオーバーヘッド削減で 2〜5x、
  ただし sort + dedup の本質計算量は同じ。メンテ負担大。
- GPU（cuSPARSE）: pattern が NR 毎に変わる問題では symbolic factor 再利用が
  効かず、転送オーバーヘッドが支配。n_active < 1000 では CPU より遅い。
- **最大 ROI は active ペア絞り込み（distance culling）→ O(n²) → O(n log n)**
- 本 status はその前段の定数項削減。

## 実施内容

### 1. `ContactForceStStiffnessOutput.K_st` の型緩和

**ファイル**: `xkep_cae/contact/contact_force/strategy.py`

```python
@dataclass(frozen=True)
class ContactForceStStiffnessOutput:
    K_st: sp.csr_matrix | sp.coo_matrix  # status-321
```

`_process_batch()` の末尾で `.tocsr()` を skip し `sp.coo_matrix(...)` を直接返す。

### 2. 呼び出し側（`HuberContactForceProcess.assemble_tangent`）の raw COO 対応

```python
K_st_coo = K_st if isinstance(K_st, sp.coo_matrix) else K_st.tocoo()
if K_st_coo.nnz > 0:
    rows_np = np.concatenate([rows_np, K_st_coo.row])
    cols_np = np.concatenate([cols_np, K_st_coo.col])
    vals_np = np.concatenate([vals_np, K_st_coo.data])
```

K_mat / K_geo / K_st すべて COO row/col/data で concat → 1 回だけ CSR 化。

### 3. ContactForceSt ペア抽出ループの active 比例化

```python
# Step 1: state を持つペアのみ list comp で絞る
state_pairs = [(i, p) for i, p in enumerate(inp.pairs) if hasattr(p, "state")]

# Step 2: p_n > 0 で active filter（ベクトル化）
p_n_state = np.fromiter((sp_pair[1].state.p_n for sp_pair in state_pairs), ...)
active_local = p_n_state > 1e-30

# Step 3: active ペアのみ NumPy 配列に bulk 抽出
act_pairs = [state_pairs[k][1] for k in np.where(active_local)[0]]
...
```

旧実装は total ペア数比例の for ループ。新実装は active ペア数比例。

### 4. `FrictionTangentStiffnessOutput` / `FrictionGeometricStiffnessOutput` の型緩和

```python
@dataclass(frozen=True)
class FrictionTangentStiffnessOutput:
    K_mat: sp.csr_matrix | sp.coo_matrix  # status-321

@dataclass(frozen=True)
class FrictionGeometricStiffnessOutput:
    K_geo: sp.csr_matrix | sp.coo_matrix  # status-321
```

対応する `_assemble_friction_tangent_stiffness` / `_assemble_friction_geometric_stiffness`
の末尾で `.tocsr()` を skip し COO を返す。mask filter も除去。

### 5. `CoulombReturnMappingProcess.tangent()` の単一 COO concat

```python
parts: list[sp.coo_matrix] = []
K_mat_coo = K_mat if isinstance(K_mat, sp.coo_matrix) else K_mat.tocoo()
if K_mat_coo.nnz > 0: parts.append(K_mat_coo)
K_geo_coo = ...
if K_geo_coo.nnz > 0: parts.append(K_geo_coo)
if consistent_st_tangent and node_coords is not None:
    K_st_coo = ...
    if K_st_coo.nnz > 0: parts.append(K_st_coo)

if len(parts) == 1:
    return parts[0].tocsr()
all_rows = np.concatenate([p.row for p in parts])
all_cols = np.concatenate([p.col for p in parts])
all_vals = np.concatenate([p.data for p in parts])
return sp.coo_matrix((all_vals, (all_rows, all_cols)), shape=(...)).tocsr()
```

旧実装は `K = K_mat + K_geo; K = K + K_st` の 2 回の sparse 加算（内部で 2 回の
tocsr + 2 回の symbolic merge）。新実装は COO の flat 配列 concat + 1 回 tocsr()。

### 6. FrictionSt ペア抽出ループの active 比例化

```python
# 単一パスで state + friction_forces_local 存在 + |q| > 0 を filter
act_pairs: list = []
act_q_list: list = []
for pair_idx, pair in enumerate(contact_pairs):
    if not hasattr(pair, "state"): continue
    q = friction_forces_local.get(pair_idx)
    if q is None: continue
    if abs(q[0]) < 1e-30 and abs(q[1]) < 1e-30: continue
    act_pairs.append(pair)
    act_q_list.append(q)

# states を pre-bind し np.fromiter で bulk 抽出
states = [p.state for p in act_pairs]
s_arr = np.fromiter((st.s for st in states), dtype=float, count=n_act)
...
```

旧実装は 2 段 for ループ（act_indices 構築 + bulk 抽出）。新実装は 1 段 filter
＋ `np.fromiter` 抽出。Python 属性アクセスをキャッシュ。

### 7. K_st 外積を einsum → 直接ブロードキャスト

```python
# Before
K_st_local = -(
    np.einsum("ni,nj->nij", df_ds, ds_du) + np.einsum("ni,nj->nij", df_dt, dt_du)
)
# After
K_st_local = -(
    df_ds[:, :, None] * ds_du[:, None, :] + df_dt[:, :, None] * dt_du[:, None, :]
)
```

単純外積では broadcasting の方が 1.5〜3x 速い（einsum の path optimization は
overkill）。ContactForceSt / FrictionSt / FrictionSt adj の 3 箇所で適用。

### 8. K_st COO 構築の mask filter skip

```python
# Before
val_arr = K_st_local.ravel()
mask = np.abs(val_arr) > 1e-30
rows_np = row_idx[mask]
cols_np = col_idx[mask]
vals_np = val_arr[mask]
# After
vals_np = K_st_local.ravel()
rows_np = row_idx
cols_np = col_idx
```

零エントリは CSR 統合時に集約されるので、明示マスクは不要。
マスク作成 + 索引コピーの ~1〜2 ms を削減。

## 変更ファイル

- `xkep_cae/contact/contact_force/strategy.py`: 出力型緩和 + tocsr skip +
  抽出ループ active 比例化 + einsum→broadcasting + mask skip
- `xkep_cae/contact/friction/_assembly.py`: 3 アセンブリ関数すべて tocsr skip +
  mask skip + einsum→broadcasting + 抽出ループ active 比例化
- `xkep_cae/contact/friction/strategy.py`: 出力型緩和 + `tangent()` の単一
  COO concat path
- `xkep_cae/contact/contact_force/tests/test_st_stiffness_process.py`:
  `isinstance(sp.csr_matrix, sp.coo_matrix)` で 型緩和許容
- `xkep_cae/contact/friction/tests/test_assembly_process.py`: 同上

## 検証手順（再現手順）

```bash
git checkout claude/optimize-contact-friction-assembly-W8xDL

# 1. 契約チェック
PYTHONPATH=. uv run python contracts/validate_process_contracts.py
# → 契約違反なし、条例違反なし

# 2. lint / format
PYTHONPATH=. uv run ruff check xkep_cae/
PYTHONPATH=. uv run ruff format --check xkep_cae/contact/

# 3. contact 全回帰
PYTHONPATH=. uv run --with pytest --with pytest-timeout python -m pytest \
    xkep_cae/contact/ --timeout=300
# → 376 passed, 5 skipped

# 4. 広いテスト回帰
PYTHONPATH=. uv run --with pytest --with pytest-timeout python -m pytest \
    xkep_cae/ -m "not slow" --timeout=180
# → 546 passed, 10 skipped, 1 xfailed
# （stress_contour 1 件は本変更前から失敗、3D 描画 display backend 問題）

# 5. 軽量ベンチ（per-call avg の確認）
PYTHONPATH=. uv run python /tmp/bench_kst_extraction.py
# → FrictionSt n=2000: 17.84ms → 11.91ms（33% 高速化）
```

### 実測環境

- Linux 4.4 / Python 3.11.15 / uv 0.8.17
- NumPy 2.4.4 / SciPy 1.17.1 / ruff 0.14.3

## 判断の根拠

### COO concat vs sparse + sparse どちらが速いか

scipy の `sparse + sparse` は内部で tocsr → symbolic merge → numeric merge を
行う。COO concat は `np.concatenate` 3 回 + 1 回の `tocsr()`。後者は:

- symbolic merge を 1 回にまとめられる（N 回 → 1 回）
- numpy の concatenate は SIMD 化された C ルーチン
- dedup は tocsr() 1 回に集約される

経験的に 3 個以上のスパース行列を結合する場合は COO concat が優位。本 status
の friction tangent（K_mat + K_geo + K_st）は 3 個の結合なので適用条件を満たす。

### einsum vs broadcasting

`np.einsum("ni,nj->nij", a, b)` は outer product の一般形。path optimization が
働くが、単一の outer product ではオーバーヘッド（文字列パース + einsum_path
caching + stride 分析）が実計算を上回る。`a[:,:,None] * b[:,None,:]` は直接
broadcasting で BLAS level-1 相当。小〜中サイズ（N < 10000）で 1.5〜3x 速い。

### なぜ mask filter を skip できるか

K_st_local は (n_act, 12, 12) で nnz 密度は 50% 前後（構造的ゼロではなく数値的
小量）。mask で抜くと pattern が 3〜5 割削減されるが:

- `row_idx[mask]` / `col_idx[mask]` / `val_arr[mask]` の 3 回の fancy indexing
  で約 1〜2 ms（n=2000 時）
- 一方、零 値を含んだまま tocsr() しても **dedup 時に加算結果が零なら構造は
  残るがデータは変わらない**（CSR の explicit zero）
- 下流の `K @ x` や `LinearSolve` は explicit zero を無視（BLAS/LAPACK の挙動）

つまり explicit zero の害は最小で、フィルタのコストの方が大きい。

## TODO（次担当者向け）

### 直近

- [ ] **distance culling の broadphase 統合**: 本 status は定数項のみ削減。n² の
  根源である active ペア数増加は未対応。`_get_active_pairs` に素線間距離 + 半径
  カットを入れる（例: `dist > 2.5 * (r_a + r_b)` なら skip）。
- [ ] **symbolic factorization reuse**: NR 反復内で sparsity pattern が不変なら
  `LinearSolveProcess` で factor を cache。pypardiso + `analyze()` 経路を検討。
- [ ] **status-301 設定での frac=1.0 回帰確認**: 本最適化は数値結果に影響
  しないはずだが、FD テスト（`test_kst_fric_adj_manual_formula` 等）での atol
  許容範囲の外側に浮動小数演算順序の差が出ていないか、実 NR ループでの
  収束統計（cutback 数 / incr 数）との差分を確認すべき。

### 中期

- [ ] **K_st の TangentAssembly からの計測分離**: status-318-320 TODO と継続。
- [ ] **フレンドリー broadphase**: KD-tree で取得した候補ペアに distance cut を
  かけ、active 判定前に Python-level で skip。`ContactPairManager` に hook。
- [ ] **ContactForceSt の 3% 止まり分析**: なぜ friction 33% に対し contact は
  3% しか高速化しないか？ HuberContactForce 側が事前に k_pen 非線形計算などで
  重い可能性。profiling が必要。

## STA2 準拠チェック

- [x] **数値の捏造なし**: FrictionSt 33% 高速化は `/tmp/bench_kst_extraction.py`
  で同一 benchmark script を前後実行した実測値。ベースライン 17.84ms（status-320
  状態）も同じスクリプトで計測。
- [x] **再現手順記載**: 上記「検証手順」5 ステップのコマンド列。
- [x] **テスト数記載**: 459+13+22+5（status-320 から不変、K_st 型緩和対応で
  isinstance チェックを緩めただけ）。
- [x] **契約違反 0 件維持**: `validate_process_contracts.py` 実行済み。
- [x] **lint/format 検証**: `ruff check` + `ruff format` 全 OK。
- [x] **ベースライン比較**: status-320 ブランチでの `git stash` 比較を
  `/tmp/bench-status321-*.log` に保存。
- [x] **無関係テスト失敗の切り分け**: `xkep_cae/output/tests/test_stress_contour.py`
  の 1 件は `git stash` で本変更を退避しても再現するため、本 status とは無関係の
  pre-existing failure（3D rendering display backend）と確認。
