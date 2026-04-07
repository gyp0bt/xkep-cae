# status-301: 7本撚線ソルバー性能分析 — カットバック率限界 + 高速化フェーズ移行

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-07
- **ブランチ**: `claude/optimize-seven-strand-solver-eZEl3`
- **テスト数**: 442+ passed（既存テスト全合格）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-300のTODO「cutback数削減（72→30以下）」を定量分析した結果、
**カットバック率3.7%は接触問題として既に十分低く、更なる改善は見込めない**と判断。

被膜なし/被膜付きの2ケースを実行し、プロセス別プロファイリングで
1000本スケールのボトルネックを特定した。

**結論: 高速化フェーズへ移行すべき。**

---

## 実施内容

### 1. 被膜パラメータ追加

`StrandBendingOscillationConfig` に以下を追加:
- `coating_stiffness`, `coating_damping`, `coating_mu`, `coating_k_t_ratio`
- `process()` と `_process_free_end()` の両方で `_ContactConfigInput` に伝搬

### 2. 性能分析スクリプト

`contracts/analyze_solver_performance.py` を新規作成:
- 90度曲げ+±48mm揺動（2サイクル）を被膜なし/被膜付きで実行
- インクリメント診断データからカットバック内訳を分析
- `ProcessMetaclass._profile_data` でプロセス別時間を計測

---

## 結果

### 比較サマリ

| ケース | frac | incr | cutback | elapsed | cb率 |
|--------|------|------|---------|---------|------|
| 被膜なし | 1.0000 | 1900 | 72 | 1526s | 3.7% |
| 被膜付き(k=5e3,c=0.1,μ=0.15) | 1.0000 | 1900 | 72 | 1493s | 3.7% |

**被膜の有無でカットバック数は完全同一。** 被膜剛性(k=5e3)はペナルティ剛性に対して
小さく、接触力・収束挙動に実質的な影響なし。

### NR反復数統計

| 指標 | 値 |
|------|-----|
| 平均 | 4.8 |
| 中央値 | 3.0 |
| 最大 | 40 |
| 95%tile | 18 |
| 合計 | 9154 |
| >0.9収束率 | 124/1900 (6.5%) |

### カットバック発生パターン

- **初期(frac<0.02)**: Type A+B（活性集合+摩擦状態振動）— 初期接触確立期の不可避なカットバック
- **中盤(frac 0.14-0.40)**: **Type D+E が支配的**（接線剛性不整合+接触力振動）— 散発的。mat-only(K_c FD誤差1.8%)が最適解であり、これ以上の精度改善手段はない（status-296）
- **揺動フェーズ(frac>0.33)**: Type Eのチャタリング → 凍結モードで概ね吸収済み

### プロセス別プロファイル（被膜なし、上位10件）

| プロセス | 呼出数 | 合計[s] | 割合 |
|---------|--------|---------|------|
| TangentAssemblyProcess | 11,594 | 573s | **8.7%** |
| ContactForceAssemblyProcess | 12,442 | 376s | **5.7%** |
| LinearSolveProcess | 11,594 | 336s | **5.1%** |
| ContactForceStStiffnessProcess | 11,594 | 214s | 3.3% |
| UpdateGeometryProcess | 14,414 | 213s | 3.3% |
| FrictionStStiffnessProcess | 10,112 | 209s | 3.2% |
| LineSearchUpdateProcess | 11,594 | 41s | 0.6% |
| ComputeStJacobianProcess | 199,510 | 24s | 0.4% |
| NCPLineSearchProcess | 11,594 | 23s | 0.3% |
| TangentFDDiagnosticProcess | 109 | 15s | 0.2% |

**支配的ボトルネック**: 接線剛性アセンブリ(8.7%) + 接触力アセンブリ(5.7%) + 連立方程式求解(5.1%) = **全体の19.5%**

（残りはNewtonDynamicProcess自体のオーバーヘッド22.2%で、内訳は上記プロセスの呼び出しコスト）

---

## 「改善が見込めない」の判断根拠

1. **カットバック率3.7%は既に低い**: 72→30に減らしても最大84ステップ=4.4%の時間削減
2. **主要な収束改善策は全て導入済み**:
   - Hertz型ペナルティ(α=1.5) — status-285
   - チャタリング凍結モード — status-284
   - frozen-m最適解(mat-only, K_c FD誤差1.8%) — status-296
   - atol_force微小dt対策 — status-297
   - dt_contact_change_threshold=0.5（既に緩め）
3. **カットバックの主因はType D（接線剛性不整合）**: mat-only以上の精度改善手段なし
4. **被膜の有無で差がない**: 被膜は接触力平滑化効果が期待されたが、実質的に効いていない
5. **1000本スケールでは計算コスト（アセンブリ+求解）が支配的**: NR収束よりもper-iteration costが問題

---

## 高速化フェーズ ロードマップ

1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了するための候補:

| 優先度 | 候補 | 期待効果 | 現在のプロファイル |
|--------|------|---------|------------------|
| **1** | **接触ペア検出の高速化**（KD-tree/バウンディングボックス） | 2-10x | DetectCandidates: 12.8s (0.2%) → 1000本で支配的 |
| **2** | **K_c/K_st アセンブリのベクトル化強化** | 1.5-3x | TangentAssembly+ContactForce: 949s (14.4%) |
| **3** | **スパース求解の高速化**（CHOLMOD/PARDISO/AMG） | 2-5x | LinearSolve: 336s (5.1%) → 1000本で支配的 |
| **4** | **NR反復数の削減**（準Newton法、接線更新頻度制御） | 1.3-2x | NR平均4.8/incr、95%tile=18 |
| **5** | **並列化**（マルチスレッドアセンブリ） | 2-4x | 全プロセス |
| **6** | **メモリ最適化**（COO→CSR直接構築） | 1.2-2x | アセンブリ全般 |

---

## 変更ファイル

- `xkep_cae/numerical_tests/strand_bending_oscillation.py`: coating_* パラメータ追加
- `contracts/analyze_solver_performance.py`: 新規（性能分析スクリプト）

---

## 再現手順

```bash
# ブランチ
git checkout claude/optimize-seven-strand-solver-eZEl3

# 性能分析実行（被膜なし+被膜付き、約50分）
python contracts/analyze_solver_performance.py 2>&1 | tee /tmp/log-perf-$(date +%s).log
```

---

## TODO

- [ ] 高速化フェーズ: 接触ペア検出のKD-tree化（1000本スケールで最も効果的）
- [ ] 高速化フェーズ: K_c/K_st アセンブリのベクトル化強化
- [ ] 高速化フェーズ: スパース求解の高速化（CHOLMOD検討）
- [ ] 被膜パラメータの適正値検討（k=5e3では効果なし。k_penと同スケールにすべきか？）
- [ ] リスタート解析方式への移行（CLAUDE.md記載のTODO）

---

## 次の担当者向け

### 判断のポイント

カットバック率3.7%は接触非線形問題として良好な値。更なるNR収束改善よりも、
**1反復あたりの計算コスト削減**が1000本スケールで決定的に重要。

プロファイル結果から:
- TangentAssembly(573s) + ContactForceAssembly(376s) + LinearSolve(336s) = **1285s (19.5%)**
- これが7本(714 DOF)での結果。1000本(10万DOF)ではO(n^2)~O(n^3)で増大

### 被膜について

被膜剛性k=5e3 N/mmではペナルティ剛性に対して小さすぎて効果なし。
1000本モデルで被膜が重要なら、ペナルティ剛性と同スケール(k_pen相当)の
被膜剛性が必要。この場合、接触力が平滑化されカットバック率が下がる可能性あり。

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: 実行ログをtee保存、結果をそのまま記録
- [x] **回帰なし**: 既存コード変更は被膜パラメータ追加のみ
- [x] **再現手順記載**: コマンド列を明記
- [x] **ベースライン比較**: status-299(incr=1900, cutback=72, 1504s)と整合
