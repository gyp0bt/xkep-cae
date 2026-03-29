# status-270: E=25 frac=1.0 回帰修正 — n_elems_wire=20 復元

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-29
- **ブランチ**: `claude/fix-frac1-regression-2lmER`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3+9+4+18（変更なし）→ **合計592 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### E=25 frac=1.0 回帰の根本原因特定

status-234 (commit ef06ba0) で frac=1.0 を達成していた E=25 三点曲げが、現在 frac=0.54 にとどまる回帰の原因を系統的に特定。

#### パラメータ bisect 結果

status-234 の条件と現デフォルトの差分を1パラメータずつ変更して50incr短縮テストで比較。

| テスト | n_elems_wire | use_rigid_surface | frac/50incr | cutback | 結論 |
|--------|-------------|-------------------|-------------|---------|------|
| A（ef06ba0条件） | 20 | False | **0.0590** | 39 | ベースライン |
| B（現デフォルト） | 8 | True | 0.0065 | 10 | **9x遅い** |
| C（n_elems変更のみ） | 20 | True | **0.0590** | 39 | rigid無影響 |
| D（rigid変更のみ） | 8 | False | 0.0065 | 10 | n_elemsが主因 |
| E（n_elems=16） | 16 | True | **0.0579** | 28 | 16以上で良好 |

#### 結論

**根本原因: `n_elems_wire: 20→8` のパラメータ変更が唯一の原因。**

- `use_rigid_surface: False→True` は**無影響**（テストA=C, B=D）
- コード変更（NR min restore, delta_h boost, Process化等）は**中立または改善**
- n_elems=8 は E=25 の低剛性大変形で要素が粗すぎ、接触解像度が不足

#### 回帰の経緯

1. status-234: n_elems=20 で frac=1.0 達成
2. status-237: n_elems=20→4 に変更（梁面連続化目的、E=200e3 高剛性用）
3. status-264: E=25 回帰発見、4→8 に修正（「妥協点」）
4. status-266-269: 8要素のまま NR 収束改善に注力（frac=0.47→0.54）

→ n_elems=8 は妥協であり本質的修正ではなかった。**20に復元が正解。**

### 修正内容

| ファイル | 変更内容 |
|----------|----------|
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | `n_elems_wire: int = 8` → `int = 20` |

1行のみの変更。

### 検証結果

#### 短縮テスト（50incr）

n_elems=20 で frac 進行率が 9x 改善（0.0065→0.0590/50incr）。

#### 500incr テスト

n_elems=20, rigid=True, max_increments=500 → frac=0.1831 (cutback=458)

初期接触確立フェーズ（frac<0.06）はカットバックが集中するが、
status-234 の dt 区間分析では frac=0.06 以降で加速し、frac=1.0 に到達。

#### Hermite OFF フルテスト（完走確認）

n_elems=20, rigid=True, use_hermite_centerline=False, max_increments=2000
→ **frac=1.0000 到達（incr=919, cutback=727）**

| 指標 | status-234 (ef06ba0) | status-270 (現コード) |
|------|---------------------|----------------------|
| frac | 1.0000 | **1.0000** |
| incr | 870 | 919 |
| cutback | 655 | 727 |

status-234 とほぼ同等の結果。n_elems_wire=20 復元で**リグレッション完全修正を確認**。

#### Hermite ON フルテスト（実行中）

n_elems=20, rigid=True, frozen_hermite_tangent=True, max_increments=2000
→ 実行中（Hermite ONの追加カットバックのため、Hermite OFFより遅い進行）

---

## テスト結果

- 新規テスト: なし
- 既存テスト: 592 passed, 20 skipped, 1 xfailed（回帰なし）
- 契約違反: 0件
- lint/format: 全合格

---

## 再現手順

```bash
git checkout claude/fix-frac1-regression-2lmER
pip install -e .

# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"

# 契約検証
python contracts/validate_process_contracts.py

# パラメータ bisect 短縮テスト（各~30秒）
python3 -c "
import warnings; warnings.filterwarnings('ignore')
from xkep_cae.numerical_tests.three_point_bend_jig import *
for label, ne, rs in [('A:ef06ba0', 20, False), ('B:current', 8, True), ('C:n20+rigid', 20, True), ('D:n8+norig', 8, False)]:
    cfg = DynamicThreePointBendContactJigConfig(
        E=25.0, n_periods=30.0, jig_push=30.0,
        n_elems_wire=ne, max_increments=50,
        use_rigid_surface=rs, frozen_hermite_tangent=True,
    )
    r = DynamicThreePointBendContactJigProcess().process(cfg)
    sr = r.solver_result
    print(f'[{label}] frac={sr.load_history[-1]:.4f} incr={sr.n_increments} cutback={sr.n_cutbacks}')
" 2>&1 | tee /tmp/log-bisect-270.log

# E=25 フルテスト（~73分、frac≈1.0 期待）
python3 -c "
import warnings; warnings.filterwarnings('ignore')
from xkep_cae.numerical_tests.three_point_bend_jig import *
cfg = DynamicThreePointBendContactJigConfig(
    E=25.0, n_periods=30.0, jig_push=30.0,
    max_increments=2000, frozen_hermite_tangent=True,
)
r = DynamicThreePointBendContactJigProcess().process(cfg)
sr = r.solver_result
print(f'frac={sr.load_history[-1]:.4f} incr={sr.n_increments} cutback={sr.n_cutbacks}')
" 2>&1 | tee /tmp/log-benchmark-270-full.log
```

---

## 次セッションへの引き継ぎ

### 残課題（優先度順）

1. **Hermite ON でのフルテスト確認**
   - Hermite OFF で frac=1.0 確認済み
   - Hermite ON（frozen_hermite_tangent=True）でも到達可能か検証
   - NR min restore + delta_h boost の相乗効果が n_elems=20 でどう作用するか

2. **frozen_hermite_tangent=False での検証**
   - n_elems=20 + frozen=False の組み合わせ効果

3. **NR 力収束改善**（status-269 から継続）
   - 力収束は依然 0/incr（全変位収束）

4. **Hermite 非局所 ∂g/∂u 対応**（status-262 から継続）

### 設計メモ

1. **n_elems_wire はモデル依存**: E=200e3 高剛性では 8 要素で十分だが、E=25 低剛性では 20 要素が必要。将来の汎用化時には E 依存の自動推定が望ましい。
2. **status-266-269 の改善は無駄ではない**: NR min restore, delta_h boost 等は n_elems=20 でもチャタリング帯域（frac>0.4）で有効な可能性がある。
3. **パラメータ bisect 手法の有効性**: 1パラメータずつ50incr短縮テストで比較する手法は、4000秒のフルテストを待たずに10分で原因特定できた。今後の回帰分析にも活用すべき。

### 開発運用メモ

- **回帰分析の鉄則**: まず「以前の成功条件を現コードで再現」する。コード変更とパラメータ変更を分離して原因切り分け。
- **短縮テスト（50incr）の活用**: frac 進行率の比較で十分な情報が得られる。フルテスト前にスクリーニングすべき。
- **n_elems_wire=8 の教訓**: 「妥協点」で中途半端に修正すると、以降のセッションが本質的でない NR 改善に注力してしまう。元の実績値に戻すべきだった。

---
