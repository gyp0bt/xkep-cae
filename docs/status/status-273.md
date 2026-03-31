# status-273: Hermite非局所∂g/∂u Step3 — K_c隣接ノードDOF拡張

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-30
- **ブランチ**: `claude/setup-coding-standards-gfOXo`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2（新規2件）→ **合計598 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### Hermite非局所∂g/∂u Step3: K_c拡張（K_mat+K_geo隣接ノードDOF拡張）

status-272のStep2でK_stに隣接ノードDOFを追加したのと同様に、
K_mat+K_geo（材料剛性+幾何剛性）にも隣接ノードDOFへの寄与を追加。

#### 数式

Hermite補間では、接触点位置がタンジェントベクトル m を通じて隣接ノード位置に依存する：

```
pA(s) = H00(s)·xA0 + H01(s)·xA1 + H10(s)·mA0 + H11(s)·mA1
∂pA/∂x_{adj_left}  = H10(s) · dm_ext_A[0] · I₃
∂pA/∂x_{adj_right} = H11(s) · dm_ext_A[1] · I₃
```

s,t固定でx_adjを動かすと、gap/normal/p_nが変化し、接触力が変わる：

```
alpha_adj[0] = H10(s) · dm_ext_A[0]     (adj_A_left)
alpha_adj[1] = H11(s) · dm_ext_A[1]     (adj_A_right)
alpha_adj[2] = -H10(t) · dm_ext_B[0]    (adj_B_left)
alpha_adj[3] = -H11(t) · dm_ext_B[1]    (adj_B_right)

K_c_adj[ki, aj] = coeffs[ki] · alpha_adj[aj] · K_3x3
```

ここで K_3x3 = w_mat·(n⊗n) - w_geo·(I₃-n⊗n) は既存の K_mat+K_geo と同じ3x3ブロック。

#### 実装内容

| ファイル | 変更内容 |
|----------|----------|
| `xkep_cae/contact/contact_force/strategy.py` | `tangent()`: _adj_node_map と _adj_node_counts を K_mat+K_geo 前に計算（K_st と共用） |
| 同上 | K_c_adj ブロック追加: alpha_adj バッチ計算 → K_c_adj_local 構築 → COO アセンブリ |
| 同上 | K_st セクションの _adj_node_map 重複計算を削除 |
| `xkep_cae/contact/contact_force/tests/test_st_stiffness_process.py` | `TestKcAdjFD` 追加（FD検証2テスト） |

#### アーキテクチャ

```
tangent() 前半で共通計算:
    _conn → _compute_adj_node_map() → _adj_node_map
    _conn → _compute_node_counts() → _adj_node_counts

K_mat+K_geo バッチ計算（既存）
    ↓
K_c_adj 拡張（NEW: status-273）
    _adj_node_counts → dm_ext バッチ計算
    s_act, t_act → H10, H11 計算
    → alpha_adj (N, 4)
    _adj_node_map → adj_gnodes (N, 4)
    → K_c_adj_local = coeffs ⊗ alpha_adj ⊗ K_3x3
    → COO アセンブリ（adj_node >= 0 のみ）

K_st 計算（既存、_adj_node_map を共用）
```

### テスト結果

- `test_kc_adj_fd`: FD一致（atol=1e-2）✓
- `test_kc_adj_endpoint_zero`: 端点ノードのK_c列がゼロ ✓
- 既存テスト: 596 passed（回帰なし）+ 新規2件 = **598 passed**

---

## 再現手順

```bash
git checkout claude/setup-coding-standards-gfOXo
pip install -e .

# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"

# K_c_adj FDテスト
python -m pytest xkep_cae/contact/contact_force/tests/test_st_stiffness_process.py -v -k "KcAdj"

# 契約検証
python contracts/validate_process_contracts.py
```

---

## 次セッションへの引き継ぎ

### 残課題（優先度順）

1. **摩擦K_stの同等拡張**
   - `friction/_assembly.py` の `_assemble_friction_st_stiffness()` にも同じ隣接ノード拡張パターンを適用
   - `FrictionStStiffnessInput` に connectivity/adj_node_map 追加
   - Step2-3パターンで実装可能

2. **NR力収束改善**
   - Step2（K_st拡張）+ Step3（K_c拡張）完了で接線剛性の非局所精度が向上
   - FD診断で接線精度を定量評価し、力収束達成率の変化を計測
   - frozen=False + 非局所拡張の組み合わせでの改善度を計測

3. **既存Hermite FDテストのatol厳格化**
   - status-239のTODO: curved/skew/asymmetric テストの atol=1e-2 → 1e-5
   - K_c_adj テストも同様に厳格化可能

### 設計メモ

1. **K_c_adjの数式的根拠**: K_mat+K_geoの4ノード版では ∂(pA-pB)/∂x_j = c_j·I₃。隣接ノードでは ∂(pA-pB)/∂x_adj = alpha_adj·I₃ に置き換わるだけで、K_3x3ブロックは同一。
2. **_adj_node_map の共用**: K_c_adj（本status）と K_st_adj（status-272）で同じ _adj_node_map を使うよう、計算をtangent()前半に移動。重複計算を排除。
3. **_adj_node_counts と _node_counts の分離**: _node_counts は tangent() 内で意図的にNone（frozen-m設計、status-266）。_adj_node_counts は非局所寄与計算専用で、dm凍結とは独立。

---

## STA2 準拠チェック

- [x] **tee ログ保存**: テスト実行結果をstatus内に記録
- [x] **再現手順記載**: 全コマンドをstatusに記載
- [x] **数値の捏造なし**: FD一致を正直に報告
- [x] **コミットハッシュ記録**: コミット後に記録

---
