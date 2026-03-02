# status-046: Phase C5 — 幾何微分込み一貫接線 + slip consistent tangent + PDAS + 平行輸送フレーム

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-21

## 概要

status-045 の TODO を消化:
- **Phase C5: 幾何微分込み一貫接線 + semi-smooth Newton / PDAS 検討** — 本status
- **slip consistent tangent の実装（v0.2）** — 本status

テスト数 **993**（+35: 幾何剛性 10 + slip consistent tangent 8 + 平行輸送 7 + PDAS 4 + 統合テスト 6件）。

## 実施内容

### 1. 幾何剛性（Geometric Stiffness）

assembly.py に `_contact_geometric_stiffness_local()` を追加。

```
K_geo = -p_n / dist * G^T * (I₃ - n⊗n) * G
```

| パラメータ | 説明 |
|-----------|------|
| `p_n` | 法線接触反力 |
| `dist` | 中心線間距離（gap + r_A + r_B） |
| `G` | 3×12 ギャップ勾配行列（4節点×3成分） |
| `I₃ - n⊗n` | 法線面垂直射影 |

#### 性質

- **対称行列**: K_geo = K_geo^T
- **負半定値**: p_n > 0, dist > 0 の場合
- **法線方向ゼロ**: n 方向変位に対して K_geo の寄与なし
- **p_n に比例**: K_geo ∝ p_n
- **1/dist に比例**: K_geo ∝ 1/dist

`compute_contact_stiffness()` に `use_geometric_stiffness` パラメータを追加。デフォルト `True`。

### 2. Slip Consistent Tangent（v0.2）

law_friction.py の `friction_tangent_2x2()` を更新。

- **v0.1**: slip 時も `k_t * I₂`（stick と同じ近似）
- **v0.2**: 正確な return mapping ヤコビアン

```
D_t = (μ * p_n / ||q_trial||) * k_t * (I₂ - q̂ ⊗ q̂)
```

| パラメータ | 説明 |
|-----------|------|
| `q̂` | slip 方向単位ベクトル（= z_t / \|\|z_t\|\|） |
| `q_trial_norm` | 弾性予測の摩擦力ノルム（`friction_return_mapping` で記録） |

#### 性質

- **対称行列**: D_t = D_t^T
- **正半定値**: 全固有値 ≥ 0
- **ランク 1 不足**: (I₂ - q̂q̂^T) は slip 方向にゼロ固有値
- **stick と異なる**: ratio = μ*p_n / ||q_trial|| < 1 なので全体的にスケールダウン

### 3. 平行輸送（Parallel Transport）フレーム更新

geometry.py に `_parallel_transport()` を追加。

```
R(n_old → n_new) = I₃ + [v]_x + [v]_x² * (1-c) / s²
  (Rodrigues 公式、v = n_old × n_new)
```

`build_contact_frame()` を拡張:
- `prev_normal` パラメータ追加（平行輸送用）
- `prev_normal` あり → Rodrigues 回転で t1 を輸送 → Gram-Schmidt 直交化
- `prev_normal` なし → 従来の Gram-Schmidt フォールバック

#### 利点

- Gram-Schmidt より大きな法線回転に対してフレーム連続性が高い
- 複数ステップにわたるフレームの滑らかな追従を確認済み

### 4. PDAS（Primal-Dual Active Set）— 実験的

solver_hooks.py に PDAS active-set 更新を追加。

```python
if use_pdas:
    for pair in manager.pairs:
        if pair.state.status == INACTIVE:
            p_trial = λ_n + k_pen * (-g)
            if p_trial > 0 and g ≤ g_on:
                pair.state.status = ACTIVE
        else:
            p_trial = λ_n + k_pen * (-g)
            if p_trial ≤ 0:
                pair.state.status = INACTIVE
```

Inner NR loop 内で active-set を更新。従来は Outer loop でのみ更新。

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `use_pdas` | `False` | PDAS の有効化（実験的、デフォルト OFF） |
| `use_geometric_stiffness` | `True` | 幾何剛性の有効化（デフォルト ON） |

### 5. ContactConfig / ContactState 拡張

#### ContactState 追加フィールド

| フィールド | 型 | デフォルト | 説明 |
|-----------|---|-----------|------|
| `q_trial_norm` | float | 0.0 | 摩擦 trial force ノルム（slip consistent tangent 用） |

#### ContactConfig 追加フィールド

| フィールド | 型 | デフォルト | 説明 |
|-----------|---|-----------|------|
| `use_geometric_stiffness` | bool | True | 幾何微分込み一貫接線の有効化 |
| `use_pdas` | bool | False | PDAS Active-set 更新の有効化（実験的） |

### 6. テスト結果

#### 新規単体テスト（29件: test_consistent_tangent.py）

| テストクラス | テスト数 | 内容 |
|-------------|---------|------|
| TestGeometricStiffness | 10 | ゼロ(inactive/p_n=0)/対称性/法線方向ゼロ/接線非ゼロ/負半定値/p_nスケール/distスケール/有限差分/剛性含有 |
| TestSlipConsistentTangent | 8 | stick不変/slip異差/公式検証/対称性/正半定値/ランク不足/q_trial_norm記録(slip/stick) |
| TestParallelTransport | 7 | 恒等/小回転/90度/直交保存/build_contact_frame統合/フォールバック/連続性 |
| TestPDAS | 4 | pdas_default_off/geometric_stiffness_default_on/q_trial_norm存在/コピー |

#### 新規統合テスト（6件: test_solver_hooks.py TestContactSolverPhaseC5）

| テスト | 結果 |
|--------|------|
| 幾何剛性付き法線接触収束 | PASS |
| 幾何剛性無効化の後方互換 | PASS |
| PDAS 付き法線接触収束 | PASS |
| slip consistent tangent + 摩擦 | PASS |
| PDAS + 摩擦 | PASS |
| 全 C5 機能組み合わせ | PASS |

### 7. 既存テストへの影響

- **既存テスト**: 全パス — 回帰なし
  - `test_positive_semidefinite` / `test_rank_one` を `use_geometric_stiffness=False` に更新（幾何剛性は負半定値のため主項のみのプロパティを検証するよう変更）
- **新規テスト 35 件**: 単体 29 + 統合 6
- **合計 993 テスト**

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/pair.py` | ContactState に q_trial_norm 追加、ContactConfig に use_geometric_stiffness/use_pdas 追加、update_geometry で prev_normal を渡す |
| `xkep_cae/contact/geometry.py` | `_parallel_transport()` 追加、`build_contact_frame()` に prev_normal パラメータ追加 |
| `xkep_cae/contact/law_friction.py` | `friction_return_mapping` で q_trial_norm 記録、`friction_tangent_2x2` slip consistent tangent 実装 |
| `xkep_cae/contact/assembly.py` | `_contact_geometric_stiffness_local()` 追加、`compute_contact_stiffness` に use_geometric_stiffness パラメータ追加 |
| `xkep_cae/contact/solver_hooks.py` | use_geometric_stiffness/use_pdas 設定読取、幾何剛性フラグ伝播、PDAS active-set 更新ブロック |
| `tests/contact/test_consistent_tangent.py` | **新規** — 29件（幾何剛性10 + slip tangent 8 + 平行輸送 7 + PDAS 4） |
| `tests/contact/test_solver_hooks.py` | TestContactSolverPhaseC5 追加（統合テスト 6件） |
| `tests/contact/test_contact_assembly.py` | test_positive_semidefinite/test_rank_one を use_geometric_stiffness=False に更新 |

## テスト数

993（+35）

## 確認事項・懸念

1. **幾何剛性の効果**: 簡易テストケース（交差ビーム）では Newton 収束に大きな差はない。多数接触ペア・大変形問題での効果は Phase 4.7（撚線モデル）で検証予定
2. **PDAS の安定性**: 実験的機能としてデフォルト OFF。Inner loop での active-set 変更が NR 収束に悪影響を与える可能性あり。実問題での検証が必要
3. **Slip consistent tangent**: ランク不足（slip 方向にゼロ固有値）のため、一部の問題で収束性に影響する可能性。stick→slip 遷移時の不連続性に注意
4. **平行輸送の数値精度**: Rodrigues 回転は 180° 近傍で数値的に不安定。実用上は法線の急変は稀なため問題なしと想定
5. **テスト実行時間**: full test suite は一部の数値試験テストが長時間かかる。contact テスト（190件）は 12秒で完了

## TODO

- [ ] 摩擦ありの Abaqus バリデーション
- [ ] 撚線モデル（Phase 4.7）での幾何剛性・PDAS 効果検証
- [ ] PDAS の安定性改善（ダンピング付き active-set 更新等）
- [ ] 接触付き弧長法の検討

---
