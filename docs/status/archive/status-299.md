# status-299: 90度曲げ+先端横変位±48mm揺動 完走

[← README](../../README.md) | [← status-index](status-index.md)

## 日付

2026-04-06

## 概要

90度曲げ後に先端横変位±48mm（2サイクル）の揺動を、**統合モード**（単一ソルバー + prescribed_func）で完走させた。

## 結果

| 項目 | 曲げのみ (status-298) | 曲げ+揺動 ±48mm |
|------|---------------------|-----------------|
| frac | 1.0000 | **1.0000** |
| increments | 535 | 1900 |
| cutbacks | 45 | 72 |
| 時間 | 752s | 1504s |

±5mm 統合モード: frac=1.0000, incr=987, cutback=49

## 設計判断

### 統合モード（prescribed_func方式）

曲げ+揺動を1回のソルバー実行で処理。`prescribed_func(frac)` が曲げフェーズ（frac ≤ frac_bend）とθ揺動フェーズ（frac > frac_bend）を連続的に処理。

- **prescribed_dofs**: θ_y（全フェーズ共通）
- **振幅変換**: 先端変位 δ → θ振幅 Δθ = δ / R_bend（R_bend = L/θ_bend）
- **遷移平滑化**: 曲げ→揺動遷移でC1連続ランプ（cos窓）を適用。遷移微分不連続による発散を防止。

### 2フェーズリスタート方式の断念

CR梁のUL assembler は `update_reference()` 後に `assemble_internal_force(u_incr=0) = 0` を返す。
新ソルバーを起動すると、曲げ応力が消失し接触力とのバランスが崩壊（初期残差 R_r ~ 10^4）。

検討した対策と結果:
- `skip_initial_detection=True`: _ul_ref_base=u0 → u_incr=0 → f_int=0
- `skip_initial_detection=False`: _ul_ref_base=0 だがアセンブラ座標更新済み → 不整合
- 新品アセンブラ（原点メッシュ）+ u0=_u_bend: R_r=7.2e+04、改善不十分
- **結論**: CR梁UL方式では2フェーズリスタートは構造的に困難。統合モード一択。

### 収束改善のポイント

1. **C1連続遷移ランプ**: 曲げ→揺動でsin波を直接使うと dθ/dfrac が 4.71→21.3 に急変（4.5倍）。最初の1/4周期にcos窓ランプを適用。
2. **dt_initial修正**: `_total_cycles = n_cycles + n_osc` で全サイクル数に基づくdt計算。
3. **カットバック深化**: 統合モードで `dt_min_fraction` の分母を64→256に拡大。

## 変更ファイル

- `xkep_cae/numerical_tests/strand_bending_oscillation.py`: 統合モード実装（prescribed_func、C1ランプ、dt修正）
- `xkep_cae/core/data.py`: SolverResultData に final_ul_ref_base, final_node_coords_ref 追加
- `xkep_cae/contact/solver/process.py`: UL参照配置エクスポート追加
- `contracts/verify_90deg_oscillation_48mm.py`: 検証スクリプト

## 再現手順

```bash
git checkout claude/check-status-baseline-8Kea2
python contracts/verify_90deg_oscillation_48mm.py 48 2>&1 | tee /tmp/log-osc48mm.log
# ±5mm: python contracts/verify_90deg_oscillation_48mm.py 5
```

## 次の課題

- cutback数削減（72→30以下）→ 計算効率改善
- 揺動フェーズの物理的妥当性検証（応力分布、接触力履歴）
- 変形メッシュの2D投影可視化
