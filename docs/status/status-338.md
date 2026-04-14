# status-338: 7本撚線 κ_cr 実測（ContactPairAnalysisProcess 初回運用）

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-14
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9（status-337 から変更なし — work/ スクリプト追加のみ）

## 概要

status-337 で新設した `ContactPairAnalysisProcess` を、接触が実際に活性化する
**7本撚線 90°曲げ条件**に適用し、κ_cr 分布・各ペア散逸・活性ペア数推移を
**初めて実測**した。

既存の 2本撚線統合テスト（`@slow`）では `bending_curvature=0.001` で接触が
活性化しないため（`n_active=0` 全ステップ）、物理的 κ_cr 分布の検証は本 status
に委ねられていた（status-337 次ステップ項目）。

## 実測結果

### 構成
- **`work/beam_hysteresis/09_kcr_measurement_7strand.py`** 新規追加
- `n_strands=7`, `wire_radius=0.5`, `pitch_length=100.0`, `n_pitches=1.0`,
  `n_elements_per_pitch=16`（= 96 梁要素/撚線）
- `bending_curvature=0.015`（90°相当、status-298 ベースライン）
- `mu=0.15`, `penalty_exponent=1.5`（Hertz型）, `free_end_mode=True`
- `n_increments_per_cycle=20`（ベースライン 40 より粗い — 実測時間節約）
- `track_contact_mk=True` + `track_contact_pairs=True`（status-333 基盤）
- 実行コマンド:
  ```
  uv run python work/beam_hysteresis/09_kcr_measurement_7strand.py 2>&1 \
      | tee /tmp/kcr_meas_$(date +%s).log
  ```

### ソルバー結果
| 項目 | 値 |
|------|-----|
| frac_completed | **1.0000** |
| converged | True |
| n_increments | 524 |
| n_cutbacks | 62 |
| elapsed | **281.15 s** |
| max \|u\| | 6.287e+01（mm スケール） |

※ 比較（status-298 ベースライン: frac=1.0, incr=535, cutback=45, 752s） —
同等規模で**2.7倍高速化**（status-321〜326 の K_st/culling/cache 効果累積）。

### ContactPairAnalysisProcess 抽出量
| 項目 | 値 |
|------|-----|
| n_steps | 524 |
| n_unique_pairs | **26** |
| n_slipped_pairs | **24**（92%） |
| total_dissipation | 1.713e-07 |
| κ_cr mean | **5.80e-03** |
| κ_cr std | 1.74e-03 |
| κ_cr min | 3.52e-03 |
| κ_cr max | 1.23e-02 |
| κ_cr CV | **0.30** |
| max active pairs | 15 |

### 活性ペア数推移
| step | load_frac | n_active |
|------|-----------|----------|
| 0 | 0.050 | 0 |
| 131 | 0.358 | 12 |
| 262 | 0.586 | 12 |
| 393 | 0.769 | 11 |
| 523 | 1.000 | 13 |

load_frac ≈ 0.23 付近から接触が活性化し始め、frac=0.36 以降はほぼ一定
（11〜15 ペア）で推移。

### per-pair dissipation トップ5
| ペア | dissipation | κ_cr |
|------|-------------|------|
| (82, 98) | 4.29e-08 | 6.84e-03 |
| (68, 84) | 3.12e-08 | 4.20e-03 |
| (4, 52) | 2.47e-08 | 6.27e-03 |
| (2, 18) | 2.15e-08 | 1.23e-02 |
| (4, 100) | 1.84e-08 | 5.25e-03 |

### κ_cr 10-bin ヒストグラム
```
[3.52e-03, 4.40e-03):  3 ###
[4.40e-03, 5.27e-03):  8 ########   ← ピーク
[5.27e-03, 6.15e-03):  6 ######
[6.15e-03, 7.02e-03):  3 ###
[7.02e-03, 7.89e-03):  2 ##
[7.89e-03, 8.77e-03):  1 #
[8.77e-03, 9.64e-03):  0
[9.64e-03, 1.05e-02):  0
[1.05e-02, 1.14e-02):  0
[1.14e-02, 1.23e-02):  1 #   ← 外れ値（端部ペア想定）
```

**右裾型分布**（対数正規様）。mean=5.80e-3 の 2倍を超える外れ値が 1 件。

## 物理的含意

### κ_cr=5.80e-3 の桁感覚確認
- Papailiou モデル（status-332）は素線毎の接触点配置から
  κ_slip = µ·N / (EI·r)（オーダー推定）
- E=130 GPa, r=0.5 mm, µ=0.15, 典型法線力 N~O(1e-4) → κ_slip~O(1e-3)
  で **オーダー整合**

### CV=0.30 の解釈
- Papailiou モデルは単一 κ_cr を仮定するが、実測は **30% の広がり**を示す
- 素線位置（外層 vs 芯）・接触点の長手位置（中央 vs 端）で κ_cr は変動
- ファイバー梁キャリブレーション時は mean κ_cr 単独ではなく**分布**の考慮が必要

### n_unique_pairs=26 の物理性
- `exclude_same_strand=True` の下、7本撚線で理論的接触候補は
  6 外側ペア × ~4 長手位置 ≈ 24 ペア + 外側-芯 6 ペア ≈ 30 前後
- 実測 26 ペアは**おおむね対称接触構造**と整合

## 変更ファイル
| ファイル | 変更内容 |
|---------|---------|
| `work/beam_hysteresis/09_kcr_measurement_7strand.py` | **新規** — 実測スクリプト |
| `docs/status/status-338.md` | **新規**（本ファイル） |
| `docs/status/status-index.md` | status-338 エントリ追加 |
| `docs/roadmap.md` | 7本撚線 κ_cr 実測完了行追加 |
| `README.md` | 現状行更新（status-338 追記） |

## 次のステップ

- [ ] **ピッチ依存性検証** — 同スクリプトを p=50/100/200 で掃引し、κ_cr 分布・
  mean・CV の変化を実測（Papailiou は「ピッチ非依存」を予測、CR梁実測で検証）
- [ ] **揺動サイクル測定** — `n_cycles=2` + `n_oscillation_cycles=1` で
  load+unload 時の κ_cr 履歴変化を観測（履歴依存性）
- [ ] **CSV 出力 + 可視化** — `work/beam_hysteresis/` に κ_cr 分布プロット、
  per-pair dissipation マップを生成するスクリプト追加
- [ ] **ファイバー梁キャリブレーション** — mean κ_cr=5.80e-3, CV=0.30 を
  `MultiLayerFrictionDegrading1D` のパラメータ推定に利用
- [ ] **端部外れ値の物理確認** — pair (2, 18) の κ_cr=1.23e-2 が端部境界条件
  由来かを `exclude_end_elements` で切り分け

## 開発運用メモ

- `bending_curvature=0.001`（元初版）では接触が活性化せず `n_active=0` 全ステップ
  となり、κ_cr 抽出不可。実測前の**接触活性化判定**が必須 — 今回は `max active` を
  ログで確認して再実行。同様のチェックは今後のピッチ/曲率掃引でも前提に。
- 281s は `@pytest.mark.slow` としても許容外（CI は ~30min まで）なので
  本検証は**work/ スクリプト固定**（pytest 非組込み）。
- `contact_pair_history` は 524 entry × ~40 ペア（推定）= 約 2万 snapshot で
  メモリは問題なし（ContactPairAnalysisProcess の O(n_steps × n_entries) 走査も
  1 秒未満）。
