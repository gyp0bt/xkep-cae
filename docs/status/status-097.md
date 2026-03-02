# Status 097: 撚線規模別 曲げ揺動計算時間計測

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-02
**ブランチ**: `claude/measure-wire-calculation-time-SVG5G`
**テスト数**: 1886（fast: 1542 / slow: 344）※変更なし

## 概要

status-096 の TODO「1000本撚線での速度ベンチマーク実行」の前段階として、撚線規模を段階的に上げながら（7/19/37/61/91本）曲げ揺動計算を実施し、各工程の計算時間とスケーリング特性を計測・文書化した。

## 実施内容

### 1. 計測スクリプト作成

**新規ファイル**: `scripts/measure_wire_calculation_time.py`

- 撚線本数 [7, 19, 37, 61, 91] で曲げ揺動ベンチマークを順次実行
- 全規模で統一パラメータ（45°曲げ、±2mm揺動、1周期）
- `BenchmarkTimingCollector` による工程別計算時間の自動記録
- 結果を `docs/verification/wire_calculation_timing.md` に Markdown レポート出力

### 2. 計測結果サマリ

| 素線数 | 要素数 | 自由度数 | 計算時間(s) | 対7本比(時間) |
|---:|---:|---:|---:|---:|
| 7 | 28 | 210 | 92.1 | 1.00 |
| 19 | 76 | 570 | 239.1 | 2.60 |
| 37 | 148 | 1,110 | 501.4 | 5.44 |
| 61 | 244 | 1,830 | 902.6 | 9.80 |
| 91 | 364 | 2,730 | 1,476.3 | 16.02 |

### 3. 工程別ボトルネック分析

| 工程 | 7本(%) | 91本(%) | 傾向 |
|---|---:|---:|---|
| structural_tangent | 80.0 | 65.1 | 支配的だが相対的に減少 |
| line_search | 13.1 | 16.7 | 微増 |
| geometry_update | 0.2 | 6.5 | 規模増で急増（O(n²)的） |
| contact_stiffness | 1.7 | 5.4 | 接触ペア数増加に伴い増加 |
| structural_internal_force | 3.6 | 2.8 | 相対的に減少 |
| linear_solve | 0.8 | 1.5 | 微増 |

### 4. スケーリング分析

- 自由度比 13倍（7本→91本）に対し、計算時間比は 16倍
- スケーリング効率: 0.81（91本時点）
- 超線形スケーリング（O(n^1.2)程度）の原因は `geometry_update` と `contact_stiffness` の接触ペア数依存
- `structural_tangent` は NR反復あたりの平均時間が線形にスケール（115ms → 1479ms = 12.8倍、DOF比13倍とほぼ一致）

### 5. 収束性に関する所見

- 全規模で Phase 1/Phase 2 とも `converged=False`（outer_max=3 到達）
- ただし物理的には解は進行しており、発散はしていない
- 根本原因: 1ステップあたりの曲げ増分が大きい（45°/5step = 9°/step）
- inner loop の最初の outer=0 で収束せず、幾何更新後に収束する典型的パターン
- 改善策: ステップ数増加、n_outer_max 増加、あるいは修正 NR 法との組合せ

## 新規ファイル

- `scripts/measure_wire_calculation_time.py` — 計測スクリプト
- `docs/verification/wire_calculation_timing.md` — 計測結果レポート

## TODO

1. **S3パラメータチューニング**: n_outer_max 増加やステップ数増加で収束性改善の検討
2. **1000本撚線ベンチマーク**: geometry_update の O(n²) スケーリング対策が必要
3. **S4: 剛性比較ベンチマーク**
4. **xfail テストの根本対策**（n_outer_max=5 安定化、摩擦付きヒステリシス）

## 懸念・確認事項

- 45° 曲げで全規模 converged=False だが、外部比較のためパラメータ統一は維持すべき
- geometry_update が O(n²) 的にスケールする問題は、1000本で致命的になる可能性あり（推定: ~数千秒）
- structural_tangent は線形スケーリングしているため、疎行列化の効果は出ている

---
