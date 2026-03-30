# status-272: Hermite非局所∂g/∂u Step2 — K_st隣接ノードDOF拡張

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-30
- **ブランチ**: `claude/check-status-todos-yQU9P`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2（新規2件）→ **合計596 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### Hermite非局所∂g/∂u Step2: K_st拡張

status-271のStep1で計算した `ds_du_adj`/`dt_du_adj`（隣接ノードに対する∂(s,t)/∂u）を
K_stアセンブリに結合し、隣接ノードDOFへの接線剛性エントリを追加。

#### 数式

K_st = -(df_ds ⊗ ds_du + df_dt ⊗ dt_du) の ds_du を拡張:

```
K_st_adj[i, j_adj] = -(df_ds[i] * ds_du_adj[j_adj] + df_dt[i] * dt_du_adj[j_adj])
```

- 行方向（力DOF）: 4ノード × 3次元 = 12 のまま
- 列方向: 隣接4ノード × 3次元 = 12 が追加（端点は-1でスキップ）
- ds_du_adj レイアウト: [A-1_xyz(3), A+2_xyz(3), B-1_xyz(3), B+2_xyz(3)]

#### 実装内容

| ファイル | 変更内容 |
|----------|----------|
| `xkep_cae/contact/geometry/_compute.py` | `_compute_adj_node_map()` 追加 — connectivity から各要素の隣接ノードインデックスを計算 |
| `xkep_cae/contact/contact_force/strategy.py` | `_add_kst_contact_to_coo()`: dm_ext_A/B → StJacobian → K_st_adj アセンブリ追加 |
| 同上 | `ContactForceStStiffnessInput`: `adj_node_map` フィールド追加 |
| 同上 | `ContactForceStStiffnessProcess.process()`: adj_node_map パスthrough |
| 同上 | `HuberContactForceProcess.assemble_tangent()`: `_compute_adj_node_map` 計算・注入 |
| `xkep_cae/contact/contact_force/tests/test_st_stiffness_process.py` | `TestKstNonlocalFD` 追加（FD検証2テスト） |

#### アーキテクチャ

```
connectivity → _compute_adj_node_map() → adj_node_map: dict[elem_idx, (adj_left, adj_right)]
                                              ↓
HuberContactForceProcess.assemble_tangent()
    → ContactForceStStiffnessInput(adj_node_map=...)
        → _add_kst_contact_to_coo(adj_node_map=...)
            → StJacobianInput(dm_ext_A=..., dm_ext_B=...)
                → ds_du_adj, dt_du_adj
            → K_st_adj = -(df_ds ⊗ ds_du_adj + df_dt ⊗ dt_du_adj)
            → グローバルCOOにアセンブリ
```

### テスト結果

- `test_kst_adj_nodes_fd`: FD一致（atol=1e-4）✓
- `test_kst_adj_endpoint_zero`: 端点ノードのK_st列がゼロ ✓
- 既存テスト: 594 passed（回帰なし）+ 新規2件 = **596 passed**

---

## 再現手順

```bash
git checkout claude/check-status-todos-yQU9P
pip install -e .

# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"

# StJacobianテスト
python -m pytest tests/contact/test_st_jacobian.py -v

# K_st FDテスト
python -m pytest xkep_cae/contact/contact_force/tests/test_st_stiffness_process.py -v -k "Nonlocal"

# 契約検証
python contracts/validate_process_contracts.py
```

---

## 次セッションへの引き継ぎ

### 残課題（優先度順）

1. **Hermite非局所∂g/∂u Step3: K_c拡張**
   - f_c[k] = p_n · c_k · n の c_k がHermite形状関数に依存し、m経由で隣接ノードに依存
   - s,t固定でも f_c が隣接ノード位置に依存: ∂f_c/∂x_adj |_{s,t=const}
   - Step2と独立に追加可能（K_st_adjと同様にCOOアセンブリ）

2. **NR力収束改善**
   - Step2-3完了後にFD検証で接線精度を定量評価
   - 力収束達成率の変化を計測

3. **既存Hermite FDテストのatol厳格化**
   - status-239のTODO: curved/skew/asymmetric テストの atol=1e-2 → 1e-5
   - Step3完了後に実施

### 設計メモ

1. **adj_node_mapの設計判断**: ContactPairへのフィールド追加ではなく、adj_node_mapを外部から渡す方式を選択。理由: (a) frozen dataclass の ContactPair を変更しない、(b) adj情報はメッシュ全体のプロパティでペア固有ではない、(c) パイプライン貫通のみで完結。
2. **K_st_adj の非対称性**: K_st_adj は 12×12（行:4ノードDOF、列:4隣接ノードDOF）で非対称。グローバル剛性行列には非対称エントリとして追加される。対称化は不要（接触剛性は元々非対称）。
3. **端点処理**: adj_node=-1 の場合はスキップ。dm_ext_coeffs でも端点は0を返すので、ds_du_adj も自動的にゼロ。二重の安全策。

---

## STA2 準拠チェック

- [x] **tee ログ保存**: テスト実行結果をstatus内に記録
- [x] **再現手順記載**: 全コマンドをstatusに記載
- [x] **数値の捏造なし**: FD一致を正直に報告
- [x] **コミットハッシュ記録**: コミット後に記録

---
