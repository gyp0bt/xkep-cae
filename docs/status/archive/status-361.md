# status-361: 7本/19本挙動反転の幾何・Type 分布分析

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-22
**テスト数**: 変動なし

## 概要

status-360 で確認された「7本撚線で効く `smoothing_delta=1000` が 19本撚線では
逆効果（frac=0.37 退化）」の挙動反転を、**幾何モデル検証** + **Type 分布実測**
で原因切り分けした。

**結論**:

1. **幾何モデル自体は正常**: `_twisted_wire.py` で逆巻き（S/Z）は `layer_dir
   = lay_direction if (layer % 2 == 1) else -lay_direction` として実装済み。
   `bending_curvature=0.005`（軽荷重）で 19 本 **frac=1.0 完走**（302.98s,
   cb=14）→ 19 本の問題は**接触密度依存の数値的問題**。
2. **ただし非等ヘリックス角設計**: 全層同一 pitch のため外層ヘリックス角 α₂
   が内層 α₁ の 2 倍。19 本 (1,2) 層ペア相対角 **11.10°**（逆巻き加算）で
   7本の 3 倍、(1,2) ペアで κ_cr 最小 = スリップ集中という status-360 実測と整合。
3. **挙動反転の真因**: Type 分布で **mixed (C+D)** 系が 19 本重荷重でのみ
   **16.6% 突出**（他 3 ケースは 1-4%）。これは「低残差状態での接線不整合
   + active flip の同時発生」で、接触密度が増えると status-344 の K_c x/z
   カップリング不整合 (mat_only rel_err 44%) が前面化する。

## 1. 幾何モデル検証

`_make_strand_layout` 走査（`xkep_cae/mesh/_twisted_wire.py`）:

| 項目 | 7本 (1+6) | 19本 (1+6+12) |
|------|------|------|
| 内層半径 r₁ | 1.032 | 1.032 |
| 外層半径 r₂ | - | 2.064 |
| ヘリックス角 α₁ | **3.71°** | 3.71° |
| ヘリックス角 α₂ | - | **7.39°** |
| lay_direction | +1 (内層のみ) | 内層+1 / 外層-1（逆巻き実装済み） |
| 接触ペア相対角 | 3.71° (中心-内層) | **11.10°** (内-外、逆巻きで加算) |
| 接触ペア密度比 | 1× | 8.5× |

`pitch_length=100` を全層で使用するため非等ヘリックス角。等ヘリックス角
設計なら外層 pitch を 2 倍にする必要があるが、現行は固定。

## 2. 軽荷重検証: 19本 κ=0.005

`work/beam_hysteresis/18_light_load_19strand.py` 新設、`bending_curvature=0.005`
（90° の 1/3）で実測:

| 指標 | 19本 重荷重 (κ=0.015) | 19本 軽荷重 (κ=0.005) |
|------|------|------|
| frac | 0.3723 ❌ | **1.0000 ✓** |
| n_increments | 164 | 222 |
| n_cutbacks | 23 | 14 |
| elapsed | 365s | 303s |

軽荷重で完走 → **幾何モデル自体は正常**。多接触点化（接触ペア 51 件、
active pair 最大 ~40）が問題の発火トリガー。

## 3. Type 分布比較（全 4 ケース）

`work/beam_hysteresis/17_type_distribution_analyzer.py` 新設で NR 診断ログ
の `Type分布[...]` 行を集計:

| 分類 | 7本 baseline | 7本 (a') δ=1000 | 19本 軽 κ=0.005 | **19本 (a') 重** |
|------|------|------|------|------|
| frac | 1.0 ✓ | 1.0 ✓ | 1.0 ✓ | **0.3723 ❌** |
| 総 attempts | 3219 | 2845 | 242 | 673 |
| active_flip (C/E) | 49.6% | 45.7% | 38.4% | 29.7% |
| tangent (D 単独) | 46.6% | 49.7% | 37.2% | 44.3% |
| **mixed (C+D)** | **1.1%** | **1.2%** | **4.1%** | **16.6%** |
| A/B (通常) | 2.7% | 3.4% | 20.2% | 9.4% |

**決定的所見**: 19本重荷重でのみ **mixed (C+D) 16.6%**（他 3 ケースの 4〜15倍）。

### 数値的解釈

- 7本: active_flip (E) と tangent (D) が**別フェーズで発生し重ならない** →
  δ_h 拡大で E 系が抑制 → attempts 11.6% 減 → elapsed 短縮。
- 19本重荷重: **C+D 同時発生領域が支配的** → δ_h 拡大で接触遷移が緩くなり
  active ピン留めが弱まる → active flip 増加 → D 不整合との相乗で stall
  加速（frac=0.48→0.37）。

### δ_h の効果方向性

| 対象 Type | δ_h 拡大の効果 |
|-----------|------|
| active_flip (E) 単独 | ✓ 抑制（遷移帯が広がり瞬発振動が減る） |
| tangent (D) 単独 | ± 中立（K_c 不整合は変わらない） |
| **mixed (C+D)** | **✗ 悪化**（active ピン留め弱化で C+D が活性化） |

## 4. 結論と次手

1. **幾何モデルの実装バグなし**。逆巻きは正しく実装されている。
2. **非等ヘリックス角設計は数値問題の増幅要因**だが、実機撚線も同一 pitch
   の場合があり、モデル変更は慎重な判断が必要。`layer_pitch` パラメータ
   追加は次フェーズで検討可。
3. **本質対策は mixed (C+D) への直接対策**:
   - **(c) line search 強化**（NR 反復途中の過剰 active flip を backtracking
     で reject）→ mixed 領域を抑える第一候補。
   - K_c x/z カップリング不整合（status-344 の残余 44%）の直接是正 →
     mat_only 以外の項拡張を MCDD の枠組みで再検討。

## ファイル変更

| ファイル | 変更 |
|---------|------|
| `work/beam_hysteresis/17_type_distribution_analyzer.py` | **新規**: `[NR診断] Type分布[...]` ログ集計ツール |
| `work/beam_hysteresis/18_light_load_19strand.py` | **新規**: 19本軽荷重 (κ=0.005) Type 分布切り分け |
| `docs/status/status-361.md` | **新規**: 本ファイル |
| `docs/status/status-index.md` | status-361 行追加 |
| `README.md` | status-361 要約追記 |

実装本体（`xkep_cae/`、`tests/`、`contracts/`）は**無変更**。

## 引継ぎ（status-362 へ）

1. **仮説 C 候補 (c) line search 強化** — mixed (C+D) 領域の直接対策。
   `_newton_dynamic.py` に backtracking line search hook を追加、
   接触残差 / ペナルティ残差が増加する step を reject。
2. **層別 pitch API 追加検討** — `StrandMeshConfig.layer_pitches` を導入し
   等ヘリックス角設計で 19 本 frac の比較検証（機能追加は慎重に）。
