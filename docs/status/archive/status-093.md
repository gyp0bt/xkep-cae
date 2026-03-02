# Status 093: 撚線曲げ+サイクル変位ベンチマーク追加

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-01
**ブランチ**: `claude/add-wire-benchmark-PgGdm`
**テスト数**: 1882（fast: 1541 / slow: 341、新規 slow: 7）

## 概要

z軸上に配置した撚線の一端を固定し、他端にモーメントで~90°曲げた後、
z方向にサイクリック力荷重を2周期与える本番計算用ベンチマークを追加。
1000本撚線の速度ベンチマーク（仮目標: 2時間、緩和: 6時間）の基盤となる。

## 実施内容

### 1. wire_bending_benchmark.py の実装

**ファイル**: `xkep_cae/numerical_tests/wire_bending_benchmark.py`

- `WireBendingBenchmarkResult`: ベンチマーク結果データクラス
- `run_wire_bending_benchmark()`: メインベンチマーク関数
  - Phase 1: CR梁モーメント荷重で大変形曲げ（M = E·I·θ/L, n_bending_steps分割）
  - Phase 2: 曲げ状態からz方向サイクリック力荷重（±F, n_cycles周期）
  - `BenchmarkTimingCollector` 連携で工程別計時
  - 接触マネージャ: 同層除外 + 中点距離プリスクリーニング + auto k_pen
- `run_scaling_benchmark()`: 複数素線本数でのスケーリング分析
- `print_benchmark_report()`: フォーマット済みレポート出力

### 2. テストファイル

**ファイル**: `tests/test_wire_bending_benchmark.py`（新規、7テスト）

- `TestWireBendingBenchmarkSmall`: 7本撚線の動作確認
  - `test_7_strand_bending_benchmark`: 軽量版（45°, 1周期）
  - `test_7_strand_full_benchmark`: フル版（90°, 2周期, 速度参照値）
- `TestWireBendingBenchmarkMedium`: 19/37本撚線ベンチマーク
- `TestWireBendingTimingReport`: タイミング記録・レポート形式の検証

### ロードパス設計

```
Phase 1: 曲げ
  f_ext = lam * f_ext_bend  (lam = step/n_bending_steps)
  f_ext_bend[DOF_rx] = M_per_strand  (x軸まわりモーメント)

Phase 2: サイクル（f_ext_base で曲げモーメントを維持）
  半周期1: amp 0→+1  (f_ext_base = f_ext_bend)
  半周期2: amp +1→-1  (f_ext_base = f_ext_bend + 1*f_cycle)
  半周期3: amp -1→+1  (f_ext_base = f_ext_bend - 1*f_cycle)
  半周期4: amp +1→-1  (f_ext_base = f_ext_bend + 1*f_cycle)
```

## ベンチマーク実測結果

### 7本撚線フル版（90°曲げ + ±5mm 2周期）

| 項目 | 値 |
|------|-----|
| 要素数 | 56 |
| 節点数 | 63 |
| 自由度数 | 378 |
| NR反復合計 | 150 |
| 総計算時間 | 42.3 s |

### 工程別ボトルネック

| 工程 | 合計(s) | 構成比 |
|------|---------|--------|
| structural_tangent | 34.0 | **80.6%** |
| line_search | 5.7 | 13.4% |
| structural_internal_force | 1.5 | 3.6% |
| contact_stiffness | 0.4 | 1.0% |
| linear_solve | 0.3 | 0.8% |

### ボトルネック分析

1. **structural_tangent (80.6%)**: CR梁のassemble_cr_beam3dが支配的
   - 密行列アセンブリ → COO/CSR化が必要
   - 7本撚線でも56要素×12DOF/要素のPythonループ
2. **line_search (13.4%)**: merit評価のための内力再計算
3. 接触関連（contact_stiffness, contact_force）は7本では1.2%と低い
   → 大規模（91本以上）で支配的になる見込み

### 収束状況

- 90°曲げ: 10ステップでは未収束（線形近似モーメントが過大）
  → ステップ数30-50に増加、またはモーメント値の調整が必要
- サイクル: Phase 1未収束状態からの開始のため発散
  → Phase 1収束後に再実行すべき

## TODO

- [ ] CR梁アセンブリのCOO/CSRベクトル化（structural_tangent 80% → 目標 <30%）
- [ ] 収束パラメータチューニング（n_bending_steps=50, tol調整）
- [ ] 19/37/61/91本での曲げベンチマーク計時
- [ ] 1000本撚線での速度ベンチマーク実行
- [ ] contact_stiffness の Gauss 積分バッチ化（status-092 TODO引き継ぎ）
- [ ] geometry_update のベクトル化（status-092 TODO引き継ぎ）

## 懸念事項

- CR梁アセンブリが密行列ベースで、スケーラビリティのボトルネック
  - 7本で80%のため、1000本でも支配的
  - COOベクトル化（status-090で構造要素に適用済み）をCR梁にも展開が必要
- 90°曲げのforce control収束には細かい荷重増分が必要
  - displacement control（変位制御）への切替も検討価値あり
  - 既存API（newton_raphson_with_contact）はforce controlのみ対応

## 開発運用メモ

- **効果的**: 既存のBenchmarkTimingCollector・newton_raphson_with_contactをそのまま活用
- **効果的**: run_contact_cyclicのパターンを参考にf_ext_base/f_ext_totalでマルチフェーズ荷重を実現
- **懸念**: 大変形の力制御は収束が難しく、多ステップが必要（計算時間増大）
