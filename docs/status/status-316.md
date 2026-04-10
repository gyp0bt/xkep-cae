# status-316: n_strands 掃引プロファイリング実測 + 撚線ボトルネック順位付け

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-10
- **ブランチ**: `claude/check-status-todos-t5Ckj`
- **テスト数**: 459+13+11（status-315 から変更なし、本 status はデータ取得のみ）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

---

## 概要

status-315 で整備した `ParameterSweepBenchmarkProcess` ×
`StrandBendingOscillationProcess` で、**初回の n_strands 実測データを取得**した。
status-315 の TODO 先頭「**実測実施**」を消化する status。

掃引対象は **n_strands = 7 / 19 / 37** の 3 ケース、軽量構成（n_pitches=0.25,
n_increments_per_cycle=4, 接触 ON、被膜 OFF）。1000 本に到達する前の
ボトルネック傾向の確定と、次の高速化項目の優先順位付けが目的。

実測データは `docs/benchmarks/` 配下に以下として保存済み:

| ファイル | 内容 |
|----------|------|
| [`ParameterSweepBenchmark_20260410T161115.yaml`](../benchmarks/ParameterSweepBenchmark_20260410T161115.yaml) | 集約サマリ |
| [`StrandBendingOscillationProcess_20260410T160832.yaml`](../benchmarks/StrandBendingOscillationProcess_20260410T160832.yaml) | n=7 ケース詳細 |
| [`StrandBendingOscillationProcess_20260410T160855.yaml`](../benchmarks/StrandBendingOscillationProcess_20260410T160855.yaml) | n=19 ケース詳細 |
| [`StrandBendingOscillationProcess_20260410T160949.yaml`](../benchmarks/StrandBendingOscillationProcess_20260410T160949.yaml) | n=37 ケース詳細 |
| [`status316_nstrands_sweep_analysis.md`](../benchmarks/status316_nstrands_sweep_analysis.md) | 集計テーブル + スケーリング分析 |
| [`status316_run_log.txt`](../benchmarks/status316_run_log.txt) | 完全 tee ログ（338 行） |

---

## 実施内容

### 1. 実測スクリプト `work/strand_profiling/status316_nstrands_sweep.py`

1 実行で 7/19/37 本の 3 ケースを直列掃引し、各ケースのマニフェスト YAML と
集約 YAML を `docs/benchmarks/` へ書き出す。`result_extractors` で
`n_increments / converged / total_ndof` を抽出。

設定:

```python
StrandBendingOscillationConfig(
    n_strands=7,                      # sweep で上書き
    n_pitches=0.25,
    n_elements_per_pitch=16,
    n_increments_per_cycle=4,
    bending_curvature=5e-4,
    contact_enabled=True,
    coating_stiffness=0.0,
    coating_barrier=False,
)
```

### 2. 実測結果

総実行時間: **162.74 s**（n=7: 22.39s、n=19: 53.85s、n=37: 85.86s）。
全 3 ケース converged=True、NR 不収束 0 件。

| n_strands | ndof | n_inc | 総 [s] | NR 反復 | LinearSolve [s] | TangentAssembly [s] | 接触剛性 [s] |
|-----------|------|-------|--------|---------|-----------------|---------------------|--------------|
| 7 | 222 | 4 | 22.39 | 15 | 20.08 | 0.36 | 0.05 |
| 19 | 582 | 4 | 53.85 | 32 | 45.15 | 4.39 | 1.39 |
| 37 | 1122 | 5 | 85.86 | 47 | 64.54 | 12.55 | 4.54 |

### 3. 主要知見（ボトルネック順位付け）

#### 知見 A: **LinearSolveProcess が現時点の支配的コスト**
- n=37 で elapsed の **75%** (64.5 s / 85.9 s)。
- ただし **avg/call が 1.34→1.41→1.37 s とほぼ定数**。
  pypardiso 直接法は DOF 1100 程度の疎行列では余裕あり。
- 総時間の伸びは **NR 反復数の伸び**（15→32→47）だけで説明できる。

#### 知見 B: **TangentAssembly / 接触剛性アセンブリが超線形スケール**
- `TangentAssembly 総時間`: 0.36 → 4.39 → 12.55 s
  （n=37/n=7 = **34.66 倍**、DOF 比 5.05 倍に対して超線形）
- `ContactForceStStiffnessProcess`: 0.05 → 1.39 → 4.54 s（**94.6 倍**）
- `ContactForceAssembly`: 0.39 → 1.39 → 4.10 s（**10.5 倍**）

現時点では TangentAssembly は全体の 3.58% に留まるが、
**n² オーダーの超線形性** があるため、1000 本モデルでは
**TangentAssembly が LinearSolve を抜いて支配的になる強い示唆**。

#### 知見 C: **NR 反復数が n_strands に対してほぼ線形成長**
15 → 32 → 47（~1.0/strand）。
各反復自体は pypardiso で安いので、1000 本で仮にこのペースが続くと
~1000 反復となり、LinearSolve がそれだけで数十分オーダーに膨らむ。
**短期の高速化は NR 反復削減が最大の余地**。

#### 知見 D: **dominant_process フィールドはネスト wrapper が取る**
`StrandBendingOscillationProcess`/`ContactFrictionProcess`/`NewtonDynamicProcess`
の 3 つが ~25% ずつで並ぶのは、全部同じ elapsed を入れ子で計上しているため。
dominant 判定には**葉 Process**（LinearSolve, TangentAssembly, ...）を見るべき。
`ParameterSweepBenchmarkProcess` の将来拡張案として、
`dominant_leaf_process` フィールド追加を TODO に積む（本 status では実装しない）。

---

## 変更ファイル

### 新規
- `work/strand_profiling/status316_nstrands_sweep.py`（実測エントリポイント）
- `docs/benchmarks/status316_nstrands_sweep_analysis.md`（分析レポート）
- `docs/benchmarks/status316_run_log.txt`（完全 tee ログ）
- `docs/benchmarks/ParameterSweepBenchmark_20260410T161115.yaml`（集約サマリ）
- `docs/benchmarks/StrandBendingOscillationProcess_20260410T16083[2|55|]*.yaml`（各ケース）
- `docs/status/status-316.md`（本ファイル）

### 更新
- `README.md`: 状態行と status-316 への参照追加
- `docs/status/status-index.md`: status-316 行 + テスト数推移 footer
- `docs/roadmap.md`: 「次」更新 + 実測完了行追加
- `CLAUDE.md`: TODO 消化マーク + 次課題更新

---

## 再現手順

```bash
# 前提: numpy / scipy / pypardiso / ruff インストール済み
git checkout claude/check-status-todos-t5Ckj

# 1. 契約チェック
python contracts/validate_process_contracts.py

# 2. 掃引実測
PYTHONPATH=. python work/strand_profiling/status316_nstrands_sweep.py \
    2>&1 | tee /tmp/log-status316-$(date +%s).log

# 3. lint / format
ruff check xkep_cae/ tests/ work/strand_profiling/
ruff format --check xkep_cae/ tests/ work/strand_profiling/
```

実測時間: 約 2.5 分。出力は `docs/benchmarks/` 配下。

### 実測環境

- Linux 4.4 / Python 3.11.15 / NumPy 2.4.4 / SciPy 1.17.1 / pypardiso 0.4.7
- git_commit: ba4305f08da5726b4347a4ec7f17c66b7ea1eb9d
- git_branch: claude/check-status-todos-t5Ckj

---

## TODO（次担当者向け）

### 直近

- [ ] **100 / 200 / 500 本への掃引拡張** — 計算リソース（推定 30 分～数時間）
  が確保でき次第、同じスクリプトを `SWEEP_VALUES = (7, 19, 37, 61, 91, 127)`
  で再実行し、**TangentAssembly が LinearSolve を抜く転換点**を特定する。
  1000 本到達の前提条件となる超線形項の特定。
- [ ] **被膜 ON でのプロファイル取得** — status-305 で被膜ありは incr 535→308
  (42%削減) を確認済み。profile_breakdown の構成が被膜の有無でどう変わるかは
  まだ測定していないので、`coating_barrier=True` で同スイープを実施する。
- [ ] **被膜幾何接線剛性（∂n/∂u, ∂s/∂u）の実装** — status-304 で FD 誤差 67%
  の主因と判明した項。フルバリア被膜の正確な接線剛性。
- [ ] **`ParameterSweepBenchmarkProcess.dominant_leaf_process`** — 集約 summary
  の dominant_process が wrapper を指す問題の改善。`profile_breakdown` から
  wrapper/nested を除外した葉 Process の先頭を別フィールドで出す拡張。

### 中期

- [ ] **リスタート解析方式への移行** — `(u, v, a, 接触ペア)` I/O 化（status-315 継続）
- [ ] **シース-素線接触統合** — 旧 SheathModel/HEX8 の Process 化
- [ ] **`ParameterSweepBenchmarkProcess` 並列実行モード** —
  `_profile_data` 競合を避けるスナップショット差分管理が必要（status-315 継続）

### 開発運用メモ

- **効果的**: status-315 の基盤整備があったおかげで、今回の実測 status は
  「スクリプト 1 本書く → 実行 → 分析」の 3 ステップで完了。status ごとの
  粒度分離（インフラ ↔ 実測）が想定通りワークした。
- **効果的**: `BenchmarkRunnerProcess` の `profile_breakdown` 自動キャプチャ
  （status-314）がそのまま生データになったので、後処理で YAML を読み返す
  だけでスケーリング分析ができた。手動 print を追加する必要なし。
- **非効果的 / 注意**: 初回実行時、本 status のスクリプトで
  `case.extracted` 参照 → `BenchmarkRunResult` の正しい属性は
  `case.manifest.results_summary`。API 齟齬に気付くのに 1 実測ぶん
  （2.5 分）浪費した。ScriptTemplate があれば防げた。次回 TODO:
  `numerical_tests/parameter_sweep_benchmark.py` の docstring に
  stdout 表示サンプルを追記する。
- **再現性**: n=37 ケースで Type D stall（FD 診断トリガー）発動の
  タイミングが実行ごとに微妙に異なる（PyPardiso のスレッド非決定性の
  影響と推定）。elapsed は ~10% の範囲でゆらぐが、profile_breakdown の
  順位は安定している。

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: 実 profile データを YAML から直接引用。
- [x] **再現手順記載**: 上記「再現手順」セクション。
- [x] **ベースライン維持**: status-315 で固まった 459+13+11 テスト数はそのまま。
- [x] **変更前計測**: status-315 と同じ git_commit でベースラインを走らせたのち
  実測。profile_breakdown は status-314/315 の設計がそのまま生きている。
- [x] **tee ログ出力**: `docs/benchmarks/status316_run_log.txt` に完全ログ保存。
