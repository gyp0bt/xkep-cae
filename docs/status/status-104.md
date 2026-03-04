# Status 104: 撚線メッシュ非貫入制約の実装

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-04
**ブランチ**: `claude/fix-wire-penetration-xgjbO`
**テスト数**: 1964（fast: 1585 / slow: 374 + 5）— 回帰なし

## 概要

撚線メッシュ生成モジュール（`twisted_wire.py`）に `strand_diameter`（外径）指定時の
非貫入制約アルゴリズムを実装。従来は `wire_diameter` から `r_lay = layer * d` で
配置半径を固定していたため、外径と素線径を独立に指定すると梁どうしが貫入する問題があった。

## 問題の分析

**従来の問題**:
- `make_strand_layout` は `r_lay = layer * (d + gap)` で配置半径を計算
- ユーザーが撚線外径（strand_diameter）と素線径（wire_diameter）を独立に指定する手段がない
- 外径が決まっている実ケーブルに対して、素線径を指定すると配置半径が合わず貫入が発生

**非貫入制約（2つの幾何学的条件）**:
1. 層間制約: `r_k - r_{k-1} >= d`（隣接層の素線中心間距離 >= 素線直径）
2. 層内制約: `2*r_k*sin(π/n_k) >= d`（同一層の隣接素線の弦距離 >= 素線直径）

## 実施内容

### 1. 新規関数の追加（`twisted_wire.py`）

| 関数 | 役割 |
|------|------|
| `_count_layers(n_strands)` | 撚構成の層数を計算 |
| `_compute_layer_structure(n_strands)` | 各層の素線数リストを返す |
| `minimum_strand_diameter(n_strands, wire_diameter)` | 非貫入最小外径を計算 |
| `validate_strand_geometry(n_strands, wire_diameter, strand_diameter)` | 実現可能性を検証 |

### 2. `make_strand_layout` の拡張

- `strand_diameter` オプション引数を追加
- 指定時: 外径から各層の配置半径を自動計算（非貫入制約を満たしつつ余剰スペースを均等配分）
- 未指定時: 従来の `r_lay = layer * (d + gap)` 方式（後方互換）

### 3. `make_twisted_wire_mesh` の拡張

- `strand_diameter` オプション引数を追加（`make_strand_layout` に転送）
- `strand_diameter` 指定時は `gap` 引数は無視される

### 4. テスト追加（`tests/mesh/test_wire_penetration.py`）

43テスト新規追加:

| テストクラス | テスト数 | 検証内容 |
|------------|--------|--------|
| `TestMinimumStrandDiameter` | 5 | 最小外径の計算値（1/3/7/19本 + 単調性） |
| `TestValidateStrandGeometry` | 7 | 実現可能性検証（正常/境界/異常） |
| `TestStrandDiameterLayout` | 18 | strand_diameter指定時の非貫入（3/7/19/37/61/91本 × 最小/余裕） |
| `TestStrandDiameterMesh` | 6 | 3Dメッシュでの非貫入検証 |
| `TestLegacyCompatibility` | 5 | 従来APIの後方互換性 |

## 配置アルゴリズム

```
入力: n_strands, wire_diameter, strand_diameter

1. 層構造を計算: layers = [1, 6, 12, 18, ...]
2. 各層の最小配置半径を計算:
   r_min[k] = max(
     r_min[k-1] + d,            # 層間非貫入
     r_wire / sin(π / n_k)      # 層内非貫入
   )
3. 余剰スペースの均等配分:
   surplus = (strand_diameter/2 - r_wire) - r_min[最外層]
   delta = surplus / n_layers
   r_lay[k] = r_min[k] + k * delta
```

## テスト結果

- 既存メッシュテスト: 32/32 パス（回帰なし）
- 新規非貫入テスト: 43/43 パス
- メッシュ全体: 224/224 パス

### 5. スクリプトのデフォルト変更

| スクリプト | 変更内容 |
|-----------|---------|
| `run_bending_oscillation.py` | `use_ncp=True`（NCP デフォルト有効）、`strand_diameter="auto"`（非貫入配置デフォルト） |
| `measure_wire_calculation_time.py` | `minimum_strand_diameter` で自動非貫入配置 |
| `wire_bending_benchmark.py` | `strand_diameter` 引数を追加 |

CLIオプション:
- `--no-ncp`: NCP を無効化し AL 法に切替
- `--strand-diameter auto|0|数値`: 撚線外径指定（`auto`=最小外径自動計算、`0`=従来配置）

### 6. Abaqus互換 .inp ライター（`write_abaqus_model`）

model/step レベルを正しく分離した新しい .inp ライターを追加。

**Model レベル**: *HEADING, *NODE, *ELEMENT, *NSET, *ELSET, *MATERIAL (*ELASTIC, *DENSITY), *BEAM SECTION, *INITIAL CONDITIONS

**Step レベル** (*STEP ～ *END STEP):
- `*STEP, INC=N, NLGEOM=YES, UNSYMM=YES` + ステップ名
- `*STATIC` / `*DYNAMIC` プロシージャ + 時間パラメータ
- `*BOUNDARY, TYPE=DISPLACEMENT/VELOCITY`
- `*CONTACT, ALGORITHM=NCP/AL`（独自拡張）
- `*OUTPUT, FIELD/HISTORY` → `*NODE OUTPUT`, `*ELEMENT OUTPUT`, `*ENERGY OUTPUT`
- `*ANIMATION`（独自拡張）
- `*CLOAD`, `*DLOAD`

データクラス: `InpStep`, `InpContactDef`, `InpOutputRequest`, `InpAnimationRequest`, `InpInitialCondition`

スクリプト export は Bending(STATIC) + Oscillation(DYNAMIC) の2ステップで出力。
密度 `*DENSITY 7850` を鋼線デフォルトとして追加。

## 次の課題

- [ ] 19本以上 NCP 収束のパラメータ最適化
- [ ] k_pen 自動スケーリング（EA/L ベース）
- [ ] 37/61/91本の段階的収束テスト
- [ ] 被膜付き撚線での strand_diameter 対応（coating.thickness を考慮）

## 確認事項

- 被膜（CoatingModel）がある場合、現在の `strand_diameter` は素線の芯線径ベースで配置を計算。
  被膜厚さを含めた有効直径での非貫入制約は今後の課題。
- 既存の呼び出し箇所（scripts/tests）は `strand_diameter` 未指定のため後方互換で影響なし。
