# Status 092: S3ベンチマーク・タイミング計測基盤

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-01
**ブランチ**: `claude/add-benchmark-timing-L1gWP`
**テスト数**: 1875（fast: 1541 / slow: 334、新規 slow: 9）

## 概要

撚線ベンチマークの各工程処理時間を記録する `BenchmarkTimingCollector` を実装し、
`newton_raphson_with_contact` ソルバーに計測機能を組み込んだ。
7→19→37→61→91本の段階的ベンチマークで収束性と処理時間のボトルネックを可視化。

## 実施内容

### 1. BenchmarkTimingCollector の実装

**ファイル**: `xkep_cae/contact/solver_hooks.py`

- `TimingRecord`: 各計測ポイントのデータクラス（step, outer, inner, phase, elapsed_s）
- `BenchmarkTimingCollector`: タイミング収集器
  - `record()`: 計測データ追加
  - `phase_totals()`: 工程別合計時間
  - `phase_counts()`: 工程別呼び出し回数
  - `step_times()`: 荷重ステップ別合計時間
  - `summary_table()`: 表形式のサマリー出力

### 2. newton_raphson_with_contact へのタイミング計測挿入

**オプショナルパラメータ**: `timing: BenchmarkTimingCollector | None = None`

timing が None の場合、オーバーヘッドは `time.perf_counter()` 呼び出し1回（< 1μs）のみ。
計測対象の工程:

| 工程 | 場所 | 計測内容 |
|------|------|---------|
| `broadphase` | Outer loop | AABB候補ペア検出 |
| `geometry_update` | Outer loop | 最近接点パラメータ(s,t)更新 |
| `friction_mapping` | Inner loop | 摩擦 return mapping |
| `structural_internal_force` | Inner loop | 構造内力 f_int(u) 計算 |
| `contact_force` | Inner loop | 接触内力計算 |
| `structural_tangent` | Inner loop | 構造接線剛性 K_T(u) 組み立て |
| `contact_stiffness` | Inner loop | 接触接線剛性 K_c 組み立て |
| `bc_apply` | Inner loop | 境界条件適用 |
| `linear_solve` | Inner loop | 線形ソルバー（spsolve/GMRES） |
| `line_search` | Inner loop | merit line search |
| `outer_convergence_check` | Outer loop | (s,t)収束判定 + AL乗数更新 |

### 3. S3ベンチマークテスト

**ファイル**: `tests/test_s3_benchmark_timing.py`（新規、9テスト）

- `TestBenchmarkTiming`: 各規模の収束性 + タイミング計測
  - `test_small_strand_timing[7/19]`: 小規模ベンチマーク
  - `test_medium_strand_timing[37/61]`: 中規模ベンチマーク
  - `test_91_strand_timing`: 91本ベンチマーク
  - `test_timing_collector_phases`: 全主要工程の記録検証
  - `test_step_times`: ステップ別集計検証
  - `test_bending_load`: 曲げ荷重ベンチマーク
- `TestBenchmarkScaling`: スケーリング分析
  - `test_timing_scaling_report`: 7→19→37本の工程別スケーリング

## ベンチマーク実測結果

### 工程別ボトルネック（91本撚り、引張100N、2ステップ）

| 工程 | 合計(s) | 呼出回数 | 平均(ms) | 構成比 |
|------|---------|---------|---------|--------|
| contact_stiffness | 34.0 | 73 | 465.8 | **27.9%** |
| linear_solve | 22.2 | 73 | 304.8 | **18.2%** |
| geometry_update | 18.8 | 5 | 3753.7 | **15.4%** |
| line_search | 18.2 | 73 | 249.2 | **14.9%** |
| outer_convergence_check | 17.9 | 4 | 4482.6 | **14.7%** |
| broadphase | 5.7 | 5 | 1149.2 | 4.7% |
| contact_force | 4.2 | 73 | 57.2 | 3.4% |
| bc_apply | 0.6 | 73 | 7.9 | 0.5% |
| **合計** | **122.0** | | | |

### スケーリング（7→19→37本、引張100N、3ステップ）

| 素線数 | 要素数 | DOF | NR反復 | Outer | 活性ペア | 合計時間(s) |
|--------|--------|-----|--------|-------|---------|------------|
| 7 | 56 | 378 | 135 | 11 | 31 | 1.48 |
| 19 | 152 | 1026 | 59 | 4 | 306 | 4.60 |
| 37 | 296 | 1998 | 57 | 4 | 1002 | 17.09 |

### ボトルネック分析

1. **contact_stiffness (27-36%)**: 接触接線剛性 K_c の COO 組み立てが最大ボトルネック
   - 活性ペア数に比例してスケール（91本で4104ペア）
   - Gauss積分のバッチ化が有効な最適化候補
2. **linear_solve (11-22%)**: spsolve の計算コスト
   - DOF 増大に伴い auto → GMRES に切り替わる閾値を検討
3. **geometry_update + outer_convergence_check (30%@91本)**: 幾何更新が大規模で支配的
   - update_geometry 内の最近接点計算ループがボトルネック
4. **line_search (14-22%)**: merit 評価のための内力再計算コスト
5. **structural_tangent (~0%)**: Timo3D 線形の場合はキャッシュ済みで無視可能

### 収束に関する所見

- 7本: 5ステップでは発散（max_iter到達、contact_stiffnessのNR反復が150回）
- 19-37本: 3ステップでは一部発散（収束しないステップが存在）
- 91本: 2ステップでは収束するが、貫入比 15.6% で高め
- k_pen の auto tuning（beam_ei モード）は機能しているが、ステップ数・ペナルティ調整が必要

## TODO

- [ ] contact_stiffness の Gauss 積分バッチ化（最大ボトルネック27-36%）
- [ ] geometry_update のベクトル化（91本で15.4%、update_geometryのPythonループ）
- [ ] 荷重ステップ数・ペナルティパラメータの系統的チューニング（収束性改善）
- [ ] ILU ドロップ許容度のチューニング（linear_solve 高速化）
- [ ] S4: HEX8連続体 vs 素線被膜シース剛性比較ベンチマーク

## Rust化凍結

ユーザー指示により、接触ソルバーのRust化は凍結。
Python/numpy ベクトル化 + Gauss積分バッチ化で対応。

## 懸念事項

- 7本でも5ステップ・30反復では収束しない場合がある（パラメータ調整が必要）
- 91本の貫入比 15.6% はターゲット（<2%）に対して過大
  → ペナルティ増大率・ステップ数の調整が必要
- line_search の merit 評価で assemble_internal_force を再呼出しするため、
  line_search コスト = 接触力+構造内力の再計算 × backtracking回数

## 開発運用メモ

- **効果的**: status TODOに基づく作業は明確で即座に着手可能
- **効果的**: BenchmarkTimingCollector をオプショナルパラメータとして追加することで
  既存テスト・コードへの影響をゼロに抑制
- **注意点**: 接触問題の収束は荷重ステップ数・ペナルティパラメータに敏感。
  ベンチマークテストでは収束を assert しないのが安全
