# status-283: MPC変換行列T動的再構築 — MPC接触なし90度曲げ完走

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-03
- **ブランチ**: `claude/contact-baseline-check-LbO6E`
- **テスト数**: 606 passed + 1 slow テスト追加（MPC 90度曲げ完走テスト）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

MPC端部剛体結合の変換行列Tを**UL参照配置更新時に動的再構築**することで、MPC + 接触なし 90度曲げを **frac=0.14 → frac=1.0** に改善。

**根本原因**: MPC変換行列Tは初期配置の相対位置ベクトル `r = X_slave - X_master` で構築され、一度も更新されなかった。大回転時（~13°超）に `u_slave = u_master + [r₀]× θ_master` の線形化が破綻。

---

## 実装内容

### 1. MPC変換行列T動的再構築

| 項目 | 修正前 | 修正後 |
|------|--------|--------|
| T行列構築 | 初期配置で1回のみ | **各UL更新後に変形座標で再構築** |
| 相対位置ベクトルr | 初期座標r₀（不変） | **変形後座標r_current（更新）** |
| チェックポイント | Tなし | **_mpc_current_ckpt で保存/復元** |

### 2. _ExtendedULAssemblerWrapper拡張

| 項目 | 修正前 | 修正後 |
|------|--------|--------|
| update_reference | **なし**（hasattr=False） | **梁+参照点ノード座標を更新** |
| coords_ref | 梁ノードのみ（119ノード） | **梁+参照点（121ノード）** |
| checkpoint/rollback | 梁のみ | **参照点座標も保存/復元** |

### 3. BoundaryData拡張

`mpc_groups`フィールド追加（T再構築に必要なMPCグループ情報）。

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/constraints/mpc_elimination.py` | `rebuild_mpc_transform()` 関数追加 |
| `xkep_cae/contact/solver/process.py` | UL更新後にT再構築 + チェックポイント対応 |
| `xkep_cae/core/data.py` | `BoundaryData.mpc_groups` フィールド追加 |
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | _ExtendedULAssemblerWrapper 拡張 + mpc_groups渡し |
| `xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py` | MPC 90度曲げ完走テスト追加 |

---

## ベンチマーク結果

### MPC + 接触なし 90度曲げ（κ=π/200, θ=π/2）

| 構成 | frac | incr | cutback | 備考 |
|------|------|------|---------|------|
| 修正前（T固定） | 0.1367 | 70 | 1 | ~13°で線形化破綻 |
| **修正後（T動的再構築）** | **1.0000** | **142** | **5** | **完走** |
| free_end_mode（参考, status-281） | 1.0000 | 102 | 6 | MPC不使用 |

---

## 技術的要点（次の担当者向け）

### なぜMPCは13°で壊れたか

MPC変換行列Tは剛体結合を線形化した近似:
```
u_slave = u_master + [r]× θ_master
```

ここで `[r]×` は参照配置での相対位置ベクトルのスキュー行列。実際の剛体運動は `u_slave = u_master + (R(θ) - I) · r` で、回転行列Rが必要。線形化 `R ≈ I + [θ]×` は小回転でのみ有効。

### なぜ修正が効くか

UL参照配置更新後に、**変形後座標でrを再計算**してTを再構築する。各ステップの増分回転が小さく保たれるため、線形化近似が有効なまま。

### _ExtendedULAssemblerWrapperの問題

修正前は `update_reference` メソッドがなく、`hasattr(ul_assembler, "update_reference")` が False。UL更新が全く実行されておらず、status-281のUL参照配置更新はMPCモードでは無効だった。

さらに `coords_ref` が梁ノードのみ（119ノード）を返していたため、rollback時に `state.node_coords_ref` が119ノードに縮小される潜在バグがあった。

---

## 再現手順

```bash
git checkout claude/contact-baseline-check-LbO6E
pip install -e .

# MPC + 接触なし 90度曲げ（~3分）
python -c "
from xkep_cae.numerical_tests.strand_bending_oscillation import *
import math
cfg = StrandBendingOscillationConfig(
    n_strands=7, wire_radius=0.5, pitch_length=100.0,
    n_elements_per_pitch=16, n_pitches=1.0,
    E=130.0e3, nu=0.3, rho=8.96e-9,
    bending_curvature=math.pi/200.0, n_cycles=1,
    n_increments_per_cycle=40, rho_inf=0.9, mu=0.15,
    max_nr_attempts=50, tol_force=1e-8, max_increments=10000,
    exclude_same_strand=True,
    free_end_mode=False, contact_enabled=False,
    loading_mode='rotation',
)
result = StrandBendingOscillationProcess().process(cfg)
sr = result.solver_result
frac = sr.load_history[-1] if sr.load_history else 0.0
print(f'frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}')
" 2>&1 | tee /tmp/log-mpc-nocontact-90deg.log
# 期待値: frac=1.0000, incr≈142, cutback≈5

# 回帰テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour" 2>&1 | tee /tmp/log-regression.log
# 期待値: 606 passed

# 契約検証
python contracts/validate_process_contracts.py
```

---

## STA2 準拠チェック

- [x] **tee ログ保存**: `/tmp/log-mpc-nocontact-90deg-*.log`
- [x] **再現手順記載**: 全コマンドをstatusに記載
- [x] **数値の捏造なし**: ベースラインfrac=0.14 → frac=1.0
- [x] **回帰なし**: 606 passed, 0 failed

---

## TODO

- [x] MPC変換行列T動的再構築実装
- [x] _ExtendedULAssemblerWrapper update_reference/coords_ref修正
- [x] MPC + 接触なし 90度曲げ frac=1.0完走
- [x] 回帰テスト 606 passed
- [x] slow テスト追加（test_mpc_90deg_nocontact_completes）
- [ ] MPC + 接触あり 90度曲げの試行
- [ ] evaluate/tangent dm整合性回復（status-277 推奨手順）

---
