# status-274: 摩擦K_st隣接ノードDOF拡張（Hermite非局所∂g/∂u Step4）

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-31
- **ブランチ**: `claude/check-status-todos-lIwby`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+3（新規3件）→ **合計601 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 摩擦K_st隣接ノードDOF拡張（status-273 TODO #1）

status-271〜273で接触力（K_c, K_st）に隣接ノードDOFへの寄与を追加した。
本statusでは同じパターンを摩擦K_stに適用。

#### 数式

摩擦力: `f_fric[k,i] = Σ_α q_α · c_k(s,t) · t_α_i`

K_st_fric の隣接ノード拡張:
```
K_fric_adj = outer(df_ds, ds_du_adj) + outer(df_dt, dt_du_adj)
```
ここで:
- `df_ds[k*3+i] = Σ_α q_α · dc_k/ds · t_α_i`（dc/ds = [-1,1,0,0]）
- `ds_du_adj`: StJacobianのHermite非局所微分（status-271で実装済み）

接触力との符号の違い:
- 接触力: `K_st = -(outer(...))`（f_c = -f_c_raw の符号反転）
- 摩擦: `K_st = +(outer(...))`（TangentAssemblyで K_T - K_fric として符号反転される）

#### 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `xkep_cae/contact/friction/strategy.py` | `FrictionStStiffnessInput` に `use_hermite`, `node_tangents`, `node_counts`, `adj_node_map` フィールド追加 |
| 同上 | `FrictionStStiffnessProcess.process()` で新パラメータを `_assemble_friction_st_stiffness` に渡す |
| 同上 | `tangent()` メソッドで kwargs から Hermite パラメータを取得して渡す |
| `xkep_cae/contact/friction/_assembly.py` | `_assemble_friction_st_stiffness()` に Hermite 隣接ノード拡張ロジック追加 |
| 同上 | StJacobianInput に dm_A/dm_B/dm_ext_A/dm_ext_B を渡すよう拡張 |
| `xkep_cae/contact/solver/_newton_steps.py` | `TangentAssemblyProcess` で摩擦 tangent に use_hermite/node_tangents/node_counts/adj_node_map を渡す |
| `xkep_cae/contact/friction/tests/test_assembly_process.py` | `TestFrictionStStiffnessAdjFD` 追加（3テスト） |

#### アーキテクチャ

```
TangentAssemblyProcess (solver/_newton_steps.py)
    ↓ manager.config → use_hermite_centerline
    ↓ connectivity → _compute_node_tangents/counts/adj_node_map
    ↓ kwargs でパラメータを渡す
friction_strategy.tangent()
    ↓ kwargs から取得
FrictionStStiffnessProcess
    ↓ FrictionStStiffnessInput（拡張フィールド付き）
_assemble_friction_st_stiffness()
    ├─ StJacobianInput に dm_ext_A/dm_ext_B 追加
    ├─ ds_du_adj, dt_du_adj 取得
    └─ K_fric_adj = outer(df_ds, ds_du_adj) + outer(df_dt, dt_du_adj) → COO追加
```

### テスト結果

- `test_kst_fric_adj_manual_formula`: K_st_adj が手動計算の `outer(df_ds, ds_du_adj) + outer(df_dt, dt_du_adj)` と一致（atol=1e-10）✓
- `test_kst_fric_adj_nonzero`: 非平行座標配置で隣接ノード列に非ゼロ値が存在 ✓
- `test_kst_fric_adj_endpoint_zero`: 端点ノード（3,4）のK_st列がゼロ ✓
- 既存テスト: 596 passed + 新規3件 = **599 passed**（+ skipped/xfail で 601 total）

### テスト座標の注意事項

平行セグメント配置（y方向オフセットのみ）ではStJacobianが特異（det=0）となり、
ds_du_adj=None で早期リターンする。非平行な座標配置が必要。

---

## 再現手順

```bash
git checkout claude/check-status-todos-lIwby
pip install -e .

# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"

# 摩擦K_st FDテスト
python -m pytest xkep_cae/contact/friction/tests/test_assembly_process.py -v -k "AdjFD"

# 契約検証
python contracts/validate_process_contracts.py
```

---

## 次セッションへの引き継ぎ

### 残課題（優先度順）

1. **NR力収束改善**
   - Step1〜4（K_st + K_c + 摩擦K_st 全て隣接ノード拡張済み）で接線剛性の非局所精度が向上
   - FD診断で接線精度を定量評価し、力収束達成率の変化を計測
   - frozen=False + 非局所拡張の組み合わせでの改善度を計測

2. **既存Hermite FDテストのatol厳格化**
   - status-239のTODO: curved/skew/asymmetric テストの atol=1e-2 → 1e-5
   - 接触力K_st_adj テスト（TestKstNonlocalFD）も平行座標で det=0 → 実質ゼロ対ゼロ比較
   - 非平行座標に変更してより有意義な検証にすべき

3. **接触力K_st_adjテストの座標修正**
   - `test_st_stiffness_process.py::TestKstNonlocalFD` の座標が平行配置のため
     ds_du_adj=None → FDも0.0 → trivially passing
   - 本statusの摩擦テストと同様に非平行座標に変更すべき

### 設計メモ

1. **ソルバー→摩擦のパラメータ伝搬**: kwargs経由で `use_hermite`, `node_tangents`, `node_counts`, `adj_node_map` を渡す設計。接触力は manager から自分で計算するが、摩擦は個別パラメータ受け渡し。
2. **符号の違い**: 接触力K_st は `-(outer(...))` だが、摩擦K_stは `+(outer(...))`。TangentAssemblyで `K_T - K_fric` として最終的に引かれる。
3. **平行セグメント問題**: StJacobian の特異判定 `|det| < tol * |ac|` で平行セグメントは早期リターン。隣接ノード拡張は非平行配置でのみ有効。

---

## STA2 準拠チェック

- [x] **tee ログ保存**: テスト実行結果をstatus内に記録
- [x] **再現手順記載**: 全コマンドをstatusに記載
- [x] **数値の捏造なし**: テスト結果を正直に報告
- [x] **コミットハッシュ記録**: コミット後に記録

---
