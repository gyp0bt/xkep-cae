# status-316: n_strands 掃引プロファイリング実測結果

[← README](../../README.md) | [← status-316](../status/status-316.md)

## 概要

status-315 で整備した `ParameterSweepBenchmarkProcess` ×
`StrandBendingOscillationProcess` を用いて、n_strands = 7 / 19 / 37 の
3 ケースで dominant Process の推移を実測した初回データ。

1000 本本番実測の準備として、まず小規模範囲で profile 傾向を捕捉するのが目的。
計算リソース確保後、`n_pitches` / `n_increments_per_cycle` を拡張して同一
スクリプト（`work/strand_profiling/status316_nstrands_sweep.py`）を再実行すれば
そのまま大規模版に展開できる。

## 実測構成

| 項目 | 値 |
|------|---|
| n_strands | 7, 19, 37 |
| n_pitches | 0.25 |
| n_elements_per_pitch | 16 |
| n_increments_per_cycle | 4 |
| bending_curvature | 5e-4 / mm |
| contact_enabled | True |
| coating | 無効（Hertz 型/バリア被膜 OFF） |
| 線形ソルバー | pypardiso（direct）|
| git_commit | ba4305f |
| git_branch | claude/check-status-todos-t5Ckj |
| 実行環境 | Linux / Python 3.11.15 / NumPy 2.4.4 / SciPy 1.17.1 |

## 集計テーブル

| n_strands | ndof | n_inc | 総時間 [s] | NR反復 | LinearSolve [s] | TangentAssembly [s] | ContactForceAssembly [s] | ContactSt Stiffness [s] | FrictionSt [s] |
|-----------|------|-------|-----------|--------|-----------------|---------------------|--------------------------|-------------------------|----------------|
| 7 | 222 | 4 | 22.394 | 15 | 20.078 | 0.362 | 0.391 | 0.048 | — |
| 19 | 582 | 4 | 53.849 | 32 | 45.152 | 4.386 | 1.387 | 1.392 | 1.373 |
| 37 | 1122 | 5 | 85.865 | 47 | 64.535 | 12.552 | 4.104 | 4.541 | 4.375 |

NR反復数は `LinearSolveProcess.n` 値。集計元 YAML:
- `ParameterSweepBenchmark_20260410T161115.yaml`（集約サマリ）
- `StrandBendingOscillationProcess_20260410T160832.yaml` (n=7)
- `StrandBendingOscillationProcess_20260410T160855.yaml` (n=19)
- `StrandBendingOscillationProcess_20260410T160949.yaml` (n=37)

## スケーリング分析

### 相対スケール（n=7 を 1.0 とする）

| 指標 | n=7 | n=19 | n=37 | n=37 / n=7 | DOF比(5.05x)との比較 |
|------|-----|------|------|-----------|---------------------|
| total ndof | 1.00 | 2.62 | 5.05 | 5.05 | — |
| NR反復数 | 1.00 | 2.13 | 3.13 | 3.13 | 劣スケール |
| 総時間 | 1.00 | 2.40 | 3.84 | 3.84 | 劣スケール |
| LinearSolve 総 | 1.00 | 2.25 | 3.22 | 3.22 | **劣スケール** |
| LinearSolve avg/call | 1.00 | 1.05 | 1.02 | ≈1.02 | **ほぼ定数** |
| TangentAssembly 総 | 1.00 | 12.11 | 34.66 | **34.66** | **超線形（~n²）** |
| ContactForceAssembly 総 | 1.00 | 3.55 | 10.49 | 10.49 | **超線形** |
| ContactForce St Stiffness 総 | 1.00 | 29.0 | 94.6 | **94.6** | **超超線形** |

### 所見

1. **LinearSolve は現時点でのボトルネック** (n=37 で全体の 75%, 64.5s/85.9s)。
   ただし pypardiso の avg/call が **1.34→1.41→1.37 s とほぼ定数**。
   コストはほぼ NR 反復数（15→32→47）だけでスケールしている。
2. **NR反復数が n_strands に対してほぼ線形成長**（15→32→47、k≈1.0/strand）。
   各反復自体は安いので、NR 反復削減が短期的な時短の最大余地。
3. **TangentAssembly / 接触剛性アセンブリ が超線形で伸びる**:
   n=7→n=37 (5.05x DOF) で 34.6x〜94.6x。
   1000 本モデルでは LinearSolve よりも **接触接線アセンブリが支配的**
   になる強い示唆。現時点では 3.58% 占有だが、n²成長で順位が逆転する。
4. **dominant_process フィールドは nested wrapper**:
   `StrandBendingOscillationProcess` (25%) ≈ `ContactFrictionProcess` (25%)
   ≈ `NewtonDynamicProcess` (24%) と並ぶ。これは全部入れ子で同じ区間を
   カウントしているため、dominant 判定には **葉ノードの子 Process**
   （LinearSolve, TangentAssembly, ...）を見るべき。

## 次アクション候補（status-316 で確定する順位付け）

1. **[短期]** NR反復数削減
   - チャタリング低減（既存: 接触凍結モード、Hertz 型ペナルティ）をフル装備で再測定
   - delta_h 最適化適用（status-262 の結果を反映）
2. **[中期]** TangentAssembly 超線形スケール緩和
   - 接触ペア検出の空間分割（status-308 KD-tree）効果を n=37 以上で再測定
   - K_st COO 構築の sparsity 事前確保（再割当削減）
3. **[長期]** LinearSolve 選択肢拡張
   - n_strands ≥ 100 で pypardiso 直接法が持たなくなった時点で AMG/Iterative 切替
   - factorization 再利用（NR内で K が大きく変わらない場合）

## 再現手順

```bash
# 前提: numpy / scipy / pypardiso インストール済み
PYTHONPATH=/home/user/xkep-cae python work/strand_profiling/status316_nstrands_sweep.py \
    2>&1 | tee /tmp/log-status316-$(date +%s).log
```

総実行時間: 約 2.5 分（このスイープ構成）。

## 注意

- **初回実行（当 status 起案時）**: 162.74 秒で完走。
- **2 回目以降のばらつき**: 37 本の NR 挙動で **Type D stall（FD 診断トリガー）**
  が発動するか否かで ~10% 変動する。FD 診断本体のオーバーヘッドは 1 秒以下
  （3 回で 1.02s）なので profile_breakdown への影響は軽微。
- **STA2 準拠**: elapsed は profile の pct と独立に `time.perf_counter()`
  で測定している（BenchmarkRunnerProcess の実装）。捏造なし。
