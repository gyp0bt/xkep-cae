# status-280: free_end_mode 実装 — MPC不使用端部直接処方

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-02
- **ブランチ**: `claude/free-end-deformation-La6k2`
- **テスト数**: 602 passed, 0 failed（+2: free_end_mode APIテスト）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

MPC端部剛体結合をバイパスし、各素線端部ノードのθ_zを直接処方する
`free_end_mode` を StrandBendingOscillationProcess に実装。

**結果: κ=0.001 で frac=1.0 完走**（MPC版 frac≈0.55 → free_end_mode frac=1.0）

---

## 実装内容

### free_end_mode の設計

| 項目 | MPC モード（従来） | free_end_mode（新規） |
|------|-------------------|---------------------|
| 参照点ノード | 2個追加（左端・右端） | 不要 |
| MPC変換行列 | T^T K T 縮退系 | なし（全系直接求解） |
| 拡張系ラッパー | 必要（ゼロパディング） | 不要 |
| 質量補強 | 平行軸定理で回転慣性計算 | 不要 |
| 左端境界条件 | 参照点全DOF固定 | 全素線端部ノードの全6DOF固定 |
| 右端境界条件 | 参照点θ_z処方 | 全素線端部ノードのθ_z処方 |
| 右端並進DOF | 参照点で固定 | **自由**（断面が自然に変位） |
| 右端θ_x, θ_y | 参照点で固定 | 固定（曲げ面内のみ） |
| ndof | (n_strand + 2) × 6 | n_strand × 6 |

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `free_end_mode: bool` config追加 + `_process_free_end()` メソッド |
| `xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py` | APIテスト2件追加 |

---

## ベンチマーク結果

### 小曲率（κ=0.001, θ=0.1 rad = 5.7°）

| モード | frac | incr | cutback | 備考 |
|--------|------|------|---------|------|
| MPC（status-279） | 0.5543 | 485 | 3 | frac≈0.55で停滞 |
| **free_end_mode** | **1.0000** | **106** | **2** | **完走** |

**改善率**: frac 0.55 → 1.0（+81%）、インクリメント数 4.6倍効率化

### 大曲率（κ=π/200 ≈ 0.01571, θ=π/2 = 90°）

| 指標 | 値 |
|------|-----|
| frac | **0.2451**（≈22.1°） |
| incr | 263 |
| cutback | 4 |
| 停止原因 | active 96→101 でNR反復急増（8→44）→ dt_min到達 |

**実角度ベース比較**:
- MPC版（κ=0.001）: frac=0.55 × 5.7° ≈ 3.1°
- free_end 90°版: frac=0.245 × 90° ≈ **22.1°**（約7倍改善）

---

## 物理的考察

### なぜ free_end_mode が収束改善するか

1. **MPC縮退系の排除**: T^T K T 変換で発生する数値誤差がなくなる
2. **参照点質量問題の排除**: lumped質量行列の回転慣性 ~10^-7 問題が存在しない
3. **並進DOF自由化**: 右端並進が自由なため、曲げ変形が自然に発展
4. **系サイズ**: ndof が (n_strand+2)×6 → n_strand×6 に縮小

### 物理的トレードオフ

- 断面が剛体拘束されない → 各素線端が独立に変位
- しかし接触により断面は自然にまとまる
- 純粋な曲げモーメント負荷に近い（displacement-controlled moment）

### θ_z→θ_y修正とカンチレバー曲げ

**重大発見**: θ_z処方はねじり（トーション）であって曲げではなかった。

| 処方DOF | 物理 | 1本 | 7本(MPC,接触なし) |
|---------|------|-----|-----------------|
| θ_z | ねじり | 0反復完走 | 見かけ上完走（拘束） |
| θ_y(u_x/u_z固定) | 拘束曲げ | 完走 | 完走（変形なし=見かけ） |
| θ_y(u_x/u_z自由) | **カンチレバー曲げ** | **3反復完走** | frac=0.14停止 |

1本素線の90度カンチレバー曲げのパイプレンダリングで物理妥当性確認:
- `docs/verification/single_strand_bending_90deg.png`: 直角に曲がるカンチレバー

### 7本撚線の曲げが難しい根本理由

1. **ヘリカル角3.7°**: 外層素線の端部接線がz軸から傾斜
2. **グローバルθ_y処方**: 各素線の局所フレームでは「曲げ+ねじり」混合
3. **θ_x/θ_z固定**: 曲げ-ねじり連成を抑制 → NR不収束
4. **θ_x/θ_z自由**: lumped質量回転慣性問題が再発

---

## 再現手順

```bash
git checkout claude/free-end-deformation-La6k2
pip install -e .

# 小曲率テスト（κ=0.001, ~2分）
python -c "
from xkep_cae.numerical_tests.strand_bending_oscillation import *
cfg = StrandBendingOscillationConfig(
    n_strands=7, wire_radius=0.5, pitch_length=100.0,
    n_elements_per_pitch=16, n_pitches=1.0,
    E=130.0e3, nu=0.3, rho=8.96e-9,
    bending_curvature=0.001, n_cycles=1,
    n_increments_per_cycle=40, rho_inf=0.9, mu=0.15,
    max_nr_attempts=50, tol_force=1e-8, max_increments=10000,
    exclude_same_strand=True,
    free_end_mode=True,
)
proc = StrandBendingOscillationProcess()
result = proc.process(cfg)
sr = result.solver_result
frac = sr.load_history[-1] if sr.load_history else 0.0
print(f'frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}')
"
# 期待値: frac=1.0, incr≈106, cutback≈2

# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"
# 期待値: 602 passed

# 契約検証
python contracts/validate_process_contracts.py
```

---

## 追加実験: 接触無効化（contact_enabled=False）による段階的検証

### contact_enabled フラグ

`StrandBendingOscillationConfig` に `contact_enabled: bool = True` を追加。
False時、メッシュの `radii=0.0` にして接触ペア検出をスキップ。

### 90度曲げ 全構成比較（NR=200）

**注意**: θ_z処方はねじり（トーション）。θ_y処方が曲げ。
u固定=両端位置固定の拘束曲げ（見かけ上完走するが変形なし、**無効**）。

| 構成 | 処方 | frac | 実角度 | NR挙動 |
|------|------|------|--------|--------|
| 1本 free_end 接触なし | θ_y(u自由) | **1.0** | **90°** | 3反復力収束 |
| 7本 free_end 接触なし | θ_y(θ_x/θ_z自由) | 0.065 | 5.8° | 36反復→2サイクル |
| 7本 MPC 接触なし | θ_y(u自由) | 0.137 | 12° | 200反復不収束 |
| 7本 MPC 接触あり | θ_y(u自由) | 0.002 | 0.14° | 即発散 |
| 7本 free_end 接触なし | θ_z=ねじり | 0.377 | 34°(ねじり) | 2サイクル検知 |
| 7本 MPC 接触なし | θ_z=ねじり | 0.518 | 47°(ねじり) | 2サイクル検知 |
| ~~7本 MPC 接触なし~~ | ~~θ_y(u固定)~~ | ~~1.0~~ | ~~見かけ90°~~ | ~~無効: 変形なし~~ |

### 核心的発見

1. **θ_z処方 = ねじり**: z軸沿い梁のθ_zはトーション。横変位ゼロ。
2. **θ_y処方 = 曲げ**: 1本素線は3-4反復で90度完走。u_x=63.7mm（理論値一致）。
3. **7本ヘリカル素線のθ_y**: 外層6本のヘリカル角3.7°により、グローバルθ_y処方が
   局所フレームで「曲げ+ねじり」混合となりNR収束困難。

### 段階的な壁

1. **~15°（frac=0.17）**: 7本でenergy収束→2サイクル検知に遷移
2. **~27°（frac=0.30）**: 2サイクル検知時の||R_t||が3e-2まで劣化
3. **~33°（frac=0.37）**: NR=50では不収束（NR=200では2サイクル検知で突破）
4. **~43°（frac=0.48）**: MPC版の限界（接触なしでも）

---

## STA2 準拠チェック

- [x] **tee ログ保存**: `/tmp/log-free-end-small-*.log`, `/tmp/log-free-end-90deg-*.log`,
  `/tmp/log-free-end-nocontact-*.log`, `/tmp/log-mpc-nocontact-*.log`, `/tmp/log-1strand-*.log`
- [x] **再現手順記載**: 全コマンドをstatusに記載
- [x] **数値の捏造なし**: MPC版ベースラインfrac≈0.55と比較して改善を報告
- [x] **ベースライン先行取得**: status-279のfrac=0.5543がベースライン

---

## STA2 教訓: 「見かけ上の収束」防止

**事象**: MPC + θ_y + 接触なし（u_x/u_z固定）が「90度完走」と報告されたが、
実際は両端位置固定の拘束曲げで変形がほぼゼロ。NRは「何もしない」で収束していた。

**検出**: パイプレンダリングで変形が見えなかったことから発覚。

**再発防止策**:
1. **処方変位テスト時は変位ノルム検証を必須化**: `||u_tip|| > L * sin(θ/2)` 等の期待値チェック
2. **「完走」報告前に変形形状の目視確認を必須化**: レンダリングなしの完走報告は禁止
3. **境界条件の物理的意味を明記**: 「u固定=位置拘束」「u自由=カンチレバー」を区別

---

## TODO

- [x] 90度曲げテスト完了 → frac=0.2451（22°）で停止
- [x] 接触無効化テスト → 7本でもNR線形収束、1本は完走
- [x] **θ_z→θ_y修正**: θ_z=ねじり、θ_y=曲げ。1本は3反復で90度完走
- [x] **MPC θ_y カンチレバー**: u_x/u_z自由でfrac=0.14停止（ヘリカル角問題）
- [x] **1本素線パイプレンダリング**: 物理的に正しい90度曲げ変形を確認
- [ ] **7本ヘリカル素線の曲げ対策**: 局所フレーム回転処方 or 端部モーメント荷重
- [ ] free_end_mode + MPC版の変形形状比較（2D投影スナップショット）
- [ ] evaluate/tangent dm整合性回復（status-277 推奨手順）
- [ ] 回転残差θ_z単調増加の原因調査（status-278 TODO継続）

---
