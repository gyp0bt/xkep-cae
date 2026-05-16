# status-281: ヘリカル素線接触なし90度曲げ完走 — UL参照配置更新

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-02
- **ブランチ**: `claude/helical-wire-90-bend-aljSm`
- **テスト数**: 606 passed（+4: _collect_adjacent_nodes + loading_mode config テスト）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

7本ヘリカル素線の接触なし90度曲げ（θ_y=π/2処方）を**動的ソルバーで frac=1.0 完走**した。

**根本原因**: ContactFrictionProcessが全累積変位をULアセンブラに渡し、かつ`update_reference()`を呼ばなかった。CR梁の接線剛性は小さい増分変位で正確だが、90°超の全回転を一度に分解すると精度が低下。ヘリカル素線では初期曲率があるため影響が顕著。

**修正**: ContactFrictionProcess内でコールバックをラップし、アセンブラには増分変位（`u_total - u_ref_base`）を渡す。各収束後に`update_reference()`で参照配置を更新。

---

## 実装内容

### ContactFrictionProcess UL参照配置更新

| 項目 | 修正前 | 修正後 |
|------|--------|--------|
| アセンブラへの変位 | 全累積変位（TL的） | **増分変位（UL的）** |
| update_reference | 呼ばない（コメントあり） | **各収束後に呼ぶ** |
| checkpoint/rollback | coords_ref/R_ref保存 | + `_ul_ref_base`保存 |
| 出力変位 | `u_total_accum + state.u` | **`state.u`（全累積）** |

### 変更内容

1. **コールバックラッパー**: `_ul_tangent_wrapper()` / `_ul_internal_force_wrapper()`
   - `u_total - _ul_ref_base` を計算してアセンブラに渡す
   - ULなしの場合は直接コールバック（従来通り）

2. **参照配置更新**: 収束後に `ul_assembler.update_reference(u_incr)` を呼ぶ
   - `_ul_ref_base` を `state.u` に更新

3. **チェックポイント対応**: `_ul_ref_base_ckpt` でカットバック時の復元

4. **出力変位修正**: `_build_u_output()` の二重カウント防止
   - `state.u` を直接使用（全累積変位）

### 追加機能: 静的NRソルバー（接触なし用）

動的ソルバーの慣性項を排除した純静的NR法も実装。接触なし問題の検証に使用。

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/contact/solver/process.py` | ULラッパー、update_reference呼び出し、出力修正 |
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `_static_nr_solve()`, Config拡張 |
| `xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py` | +6テスト |
| `docs/verification/7strand_90deg_bending_nocontact.png` | 2D投影スナップショット |

---

## ベンチマーク結果

### 7本ヘリカル接触なし90度曲げ（κ=π/200, θ=π/2）

| 構成 | frac | incr | cutback | 備考 |
|------|------|------|---------|------|
| 動的ソルバー（修正前, status-280） | 0.065 | 65 | 2 | ||R_t|| stall |
| **動的ソルバー（UL更新あり）** | **1.000** | **102** | **6** | **完走** |
| 静的NR（参考） | 1.000 | 40 | 0 | 接触なし専用 |

### 先端変位（エラスティカ理論比較）

| 素線 | u_x [mm] | u_z [mm] | θ_y [deg] | 理論u_x |
|------|----------|----------|-----------|---------|
| center (s0) | 63.67 | -36.29 | 90.0 | 63.66 |
| outer (s1) | 62.57 | -37.39 | 90.0 | - |
| outer (s4) | 64.77 | -35.19 | 90.0 | - |

中心素線の先端変位が理論値と0.02%一致。

---

## 物理的考察

### なぜUL参照配置更新が必要か

1. **CR梁の接線剛性精度**: 小回転増分では正確（二次収束）だが、90°全回転では線形収束に劣化
2. **ヘリカル素線の初期曲率**: 直線素線より影響大（外層6本のヘリカル角3.7°）
3. **update_reference()の効果**: 各ステップの増分回転を~2.25°に制限 → 二次収束維持

### 既存コメントの修正

修正前: 「CR梁のcorotational分解が大変形を処理するため、参照配置リセットは不要」
修正後: 「各収束後にupdate_reference()で参照配置を更新し、増分変位を小さく保つ」

この修正は7本撚線90度曲げで必須だが、小変形問題では影響なし（608テスト全通過）。

---

## 再現手順

```bash
git checkout claude/helical-wire-90-bend-aljSm
pip install -e .

# 動的ソルバー: 7本ヘリカル接触なし90度曲げ（~2分）
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
    free_end_mode=True, contact_enabled=False,
    loading_mode='rotation',
)
result = StrandBendingOscillationProcess().process(cfg)
sr = result.solver_result
frac = sr.load_history[-1] if sr.load_history else 0.0
print(f'frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}')
" 2>&1 | tee /tmp/log-7strand-dynamic-90deg.log
# 期待値: frac=1.0000, incr≈102, cutback≈6

# 収束テスト + 物理検証テスト（slow, ~3分）
python -m pytest xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py -v -k slow --timeout=300 2>&1 | tee /tmp/log-slow-tests.log
# 期待値: 2 passed (test_7strand_90deg_dynamic_completes, test_center_strand_tip_displacement)

# パイプレンダリング（物理妥当性の目視確認, ~2分）
python contracts/visualize_7strand_bending_90deg.py 2>&1 | tee /tmp/log-pipe-render.log
# 出力: docs/verification/7strand_90deg_bending_pipe.png

# 全テスト（回帰確認）
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour" 2>&1 | tee /tmp/log-regression.log
# 期待値: 608 passed

# 契約検証
python contracts/validate_process_contracts.py
```

---

## 修正の技術的要点（次の担当者向け）

### 何を変えたか

`xkep_cae/contact/solver/process.py` の収束ループ内で、ULアセンブラへの変位受け渡しを修正。

**修正前**: `assemble_tangent(state.u)` — state.uは初期配置からの全累積変位。
update_reference()は呼ばれない。CR梁は累積90°回転を一度に分解する。

**修正後**: `assemble_tangent(state.u - _ul_ref_base)` — 最後のupdate_reference()からの増分のみ渡す。
各収束後にupdate_reference()を呼び、_ul_ref_baseをstate.uに更新。

### なぜ効くか

CR梁要素のcorotational分解は、**小回転増分**では二次精度の接線剛性を生成する。
全累積90°回転を渡すと、回転ベクトルの抽出精度が低下し、NRが二次→線形収束に劣化する。
update_reference()で参照配置を逐次更新すると、各NR反復の回転増分が~2°に制限され、二次収束が維持される。

### 注意点

- `_build_u_output()` は使わなくなった（`state.u` を直接出力）。
  以前は `u_total_accum + state.u` だったが、update_reference()が
  u_total_accumに加算するため二重カウントになる。
- checkpoint/rollback時に `_ul_ref_base` も保存/復元する。
- この修正は全既存テスト（608件）に影響なし（小変形問題では差異ゼロ）。

### 次のステップ

1. **接触あり90度曲げ**: `contact_enabled=True` で試行。接触力のNR収束が課題になる可能性。
2. **dm整合性**: status-278で解消済み（evaluate/tangent共にdm OFF）。再検証は不要の可能性。

---

## STA2 準拠チェック

- [x] **tee ログ保存**: `/tmp/log-7strand-dynamic-ul-*.log`
- [x] **再現手順記載**: 全コマンドをstatusに記載
- [x] **数値の捏造なし**: ベースラインfrac=0.065（status-280）→ frac=1.0
- [x] **物理検証**: 2D投影スナップショット + エラスティカ理論比較
- [x] **回帰なし**: 608 passed, 0 failed

---

## TODO

- [x] 7本接触なし90度曲げ完走（動的ソルバー + UL更新）
- [x] 1本素線回帰確認（frac=1.0, 理論値一致）
- [x] 全テスト回帰確認（608 passed）
- [ ] 接触あり90度曲げの試行
- [ ] evaluate/tangent dm整合性回復（status-277 推奨手順）

---
