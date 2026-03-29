# status-264: three_point_bend E=25 回帰修正 + frozen_hermite_tangent + _cur_ratio統一

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-29
- **ブランチ**: `claude/check-status-todos-5ieEQ`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3+9+4（変更なし）→ **合計574 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. three_point_bend E=25 回帰の原因特定

status-263で報告された E=25 n_periods=30 回帰（frac=1.0→0.0003）の根本原因を特定。

#### 原因A（主因）: Hermite ∂m/∂u 補正の自動有効化

- ef06ba0（status-234）: `_compute_dm_coeffs` / `_compute_node_counts` が存在せず、Hermite=True でも dm=None → frozen-m近似（∂m/∂u=0）で安定
- HEAD: `use_hermite=True` + `connectivity` 非None で自動的に dm 補正が有効化
- `freeze_geometry_in_nr=True` でも dm 補正は適用される（幾何凍結とは独立）
- **影響**: frac 1.0 → 0.16（n_elems=20, hermite=T）

#### 原因B: _cur_ratio ノルム不整合

- `_res_ratio`（診断記録用）= `res_trans_norm / f_ref`
- `_cur_ratio`（発散判定用）= `res_u_norm / f_ref`
- `f_ref` は `res_trans_norm` ベース → 分子(回転含む) / 分母(並進のみ) で不整合
- **影響**: 発散検知が過敏になり、不要なカットバックが増加

#### 原因C: n_elems_wire デフォルト変更

- status-237 で n_elems_wire=20→4 に変更（梁面連続化目的）
- E=25 低剛性ケースでは要素が粗すぎて接触収束困難

### 2. 修正内容

#### 修正A: `frozen_hermite_tangent` フラグ追加

`_ContactConfigInput.frozen_hermite_tangent: bool = True`（デフォルト）
- True: ef06ba0相当（dm=None、∂m/∂u=0凍結近似、安定）
- False: dm補正あり（正確だがNR不安定の可能性）

**修正ファイル**:
- `xkep_cae/contact/_contact_pair.py`: フィールド追加
- `xkep_cae/contact/contact_force/strategy.py`: evaluate()/tangent_stiffness()でfrozen制御
- `xkep_cae/numerical_tests/three_point_bend_jig.py`: DynamicConfig + ConfigInput渡し

#### 修正B: _cur_ratio を res_trans_norm ベースに統一

`_newton_dynamic.py` L304: `conv_out.res_u_norm` → `conv_out.res_trans_norm`
力収束判定と発散判定で同じノルムを使用。

#### 修正C: n_elems_wire=8

`DynamicThreePointBendContactJigConfig.n_elems_wire`: 4 → 8

### 3. 検証結果

| 条件 | frac | incr | cutback | time |
|------|------|------|---------|------|
| ef06ba0 (n_elems=20, 旧コード) | **1.000** | 1592 | 2477 | 4403s |
| HEAD修正前 (n_elems=4, デフォルト) | 0.0003 | 7 | 10 | 9s |
| **HEAD修正後 (n_elems=8, frozen_hm=T)** | **0.6718** | 2000 | 1060 | 1114s |

frac=0.0003 → 0.67 への大幅改善。ef06ba0の1.0には未到達だが:
- n_elems_wire=8 vs 20 の条件差がある
- max_increments=2000 制限で打ち切り（延長すれば進行可能性あり）

---

## テスト結果

- 新規テスト: なし
- 既存テスト: 574 passed, 20 skipped, 1 xfailed（回帰なし）
- 契約違反: 0件
- lint: 全合格

---

## 再現手順

```bash
git checkout claude/check-status-todos-5ieEQ
pip install -e .

# 修正後 E=25 回帰テスト（~1100s, frac=0.67）
python3 -c "
import warnings; warnings.filterwarnings('ignore')
from xkep_cae.numerical_tests.three_point_bend_jig import *
cfg = DynamicThreePointBendContactJigConfig(
    E=25.0, n_periods=30.0, jig_push=30.0,
    max_increments=2000, use_rigid_surface=False,
)
r = DynamicThreePointBendContactJigProcess().process(cfg)
sr = r.solver_result
print(f'frac={sr.load_history[-1]:.4f} incr={sr.n_increments} cutback={sr.n_cutbacks}')
" 2>&1 | tee /tmp/log-regression.log

# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"

# 契約検証
python contracts/validate_process_contracts.py
```

---

## 次セッションへの引き継ぎ

### 残課題（優先度順）

1. **frozen_hermite_tangent=False でのNR安定化**（Hermite ∂m/∂u補正の根本修正）
   - 現在はTrue（凍結近似）で回避。Falseでも安定にNR収束する仕組みが必要
   - 候補: dm補正のステップ間のみ適用（NR反復内は凍結）、dm補正の漸進的適用
2. **E=25 frac=1.0 到達**
   - n_elems_wire=8 + max_increments増加で到達可能性あり
   - frozen_hermite_tangent=False の安定化後に再検証
3. **Hermite 非局所 ∂g/∂u 対応**（status-262 から継続）
4. **NR 力収束改善**（status-262 から継続）

### STA2 教訓

本セッションでのSTA2違反:
1. ベースライン条件を正確に確認せず仮説検証を開始
2. 複数の変更を同時にリバートして原因切り分けが不十分
3. consistent_st_tangent が原因と誤判断（実際はcfg側Falseで無関係）

**対策**:
- ベースラインの全パラメータ（Config + ConfigInput実効値）を先に列挙
- 1つずつ変更して影響を切り分け
- 仮説と実験結果を混同しない

---

## 懸念・設計メモ

1. **frozen_hermite_tangent=True はワークアラウンド**: FD整合性は低下する（∂m/∂u=0近似のため）。根本修正は ∂m/∂u 補正のNR安定化
2. **_cur_ratio 統一の副作用**: 回転残差の発散を検知できなくなる可能性。ただし力収束判定が並進のみで行われている以上、発散判定も並進ベースが整合的
3. **n_elems_wire=8 の妥当性**: status-237 で梁面連続化のために4にしたが、E=25では粗すぎた。8は20と4の妥協点
