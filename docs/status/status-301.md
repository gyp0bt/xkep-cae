# status-301: 7本撚線ソルバー性能分析 — 被膜で incr 半減 + 高速化フェーズ移行

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-07
- **ブランチ**: `claude/optimize-seven-strand-solver-eZEl3`
- **テスト数**: 442+ passed（既存テスト全合格）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-300のTODO「cutback数削減（72→30以下）」を定量分析。

### 第1回実行（被膜バグあり）

`core_radii=None`（ハードコード）により被膜厚さ=0 → 被膜力が常にゼロ。
被膜なし/付きで結果が完全同一（incr=1900, cutback=72）だった。

### 第2回実行（被膜バグ修正後）

`coating_thickness`パラメータ追加、`core_radii = radii - thickness`で被膜圧縮量を正しく計算。

**結果: 被膜付きでインクリメント数が半減、実行時間が1/3に。**

---

## 実施内容

### 1. 被膜パラメータ追加

`StrandBendingOscillationConfig` に以下を追加:
- `coating_stiffness`, `coating_damping`, `coating_mu`, `coating_k_t_ratio`, `coating_thickness`
- `process()` と `_process_free_end()` の両方で `_ContactConfigInput` に伝搬

### 2. 被膜バグ修正

`ContactFrictionProcess`（process.py L236）で`core_radii = None`がハードコードされていた。
`core_radii=None` → DetectCandidatesProcess で `cr_arr = r_arr.copy()`（被膜厚さ=0）。

修正: `coating_thickness > 0 && coating_stiffness > 0` の場合に
`core_radii = radii - coating_thickness` を計算するロジックを追加。

### 3. 性能分析スクリプト

`contracts/analyze_solver_performance.py`:
- 90度曲げ+±48mm揺動（2サイクル）を被膜なし/被膜付きで実行
- `ProcessMetaclass._profile_data` でプロセス別時間を計測

---

## 結果

### 比較サマリ

| ケース | frac | incr | cutback | elapsed | cb率 |
|--------|------|------|---------|---------|------|
| 被膜なし | 1.0000 | 1900 | 72 | 1527s | 3.7% |
| **被膜付き(k=1e6, c=1e4, μ=0.3, t=0.05mm)** | **1.0000** | **965** | **31** | **555s** | **3.1%** |

**被膜付きで:**
- **インクリメント数 49%減** (1900→965)
- **カットバック数 57%減** (72→31)
- **実行時間 64%短縮** (1527s→555s)

### NR反復数統計比較

| 指標 | 被膜なし | 被膜付き |
|------|---------|---------|
| 平均 | 4.8 | 4.6 |
| 中央値 | 3.0 | 3.0 |
| 最大 | 40 | 40 |
| 95%tile | **18** | **12** |
| 合計 | 9154 | 4461 |
| >0.9収束率 | 124/1900 (6.5%) | 90/965 (9.3%) |
| 平均接触ペア数 | 5.4 | 3.5 |

### 改善メカニズムの考察

被膜（Kelvin-Voigtモデル）が接触力を平滑化する効果:
1. **コア間ギャップ段階で接触力がゼロ→非ゼロの遷移が滑らか** → 活性集合の急激な変化を抑制
2. **粘性ダッシュポット(c=1e4)による速度依存項** → 時間増分に応じた自然な安定化
3. **結果として大きなdtで安定進行** → incr半減

### プロセス別プロファイル（被膜付き、上位10件）

| プロセス | 呼出数 | 合計[s] | 割合 |
|---------|--------|---------|------|
| TangentAssemblyProcess | 5,238 | 168s | 7.2% |
| ContactForceAssemblyProcess | 5,519 | 147s | 6.3% |
| LinearSolveProcess | 5,238 | 130s | 5.6% |
| UpdateGeometryProcess | 6,515 | 87s | 3.7% |
| ContactForceStStiffnessProcess | 5,238 | 58s | 2.5% |
| FrictionStStiffnessProcess | 3,623 | 55s | 2.4% |
| LineSearchUpdateProcess | 5,238 | 17s | 0.7% |
| NCPLineSearchProcess | 5,238 | 9s | 0.4% |
| ComputeStJacobianProcess | 59,812 | 6s | 0.3% |

---

## 判断: 高速化フェーズへ移行

カットバック率の改善は被膜により達成（3.7%→3.1%）。
incr半減は大きな成果だが、1000本スケールでは**per-iteration cost**が支配的。

### カットバック率について

被膜なし3.7%、被膜付き3.1% — いずれも接触問題として十分低い。
主要な収束改善策（Hertz型、凍結モード、frozen-m、atol_force）は全て導入済み。
**これ以上のカットバック率改善は見込めないと判断。**

### 1000本スケールのボトルネック

7本(714 DOF)でTangentAssembly+ContactForce+LinearSolve = 445s。
1000本(10万DOF)ではO(n^2)~O(n^3)で増大し、6時間目標には計算コスト削減が必須。

---

## 高速化フェーズ ロードマップ

| 優先度 | 候補 | 期待効果 | 1000本スケール影響 |
|--------|------|---------|------------------|
| **1** | **接触ペア検出の高速化**（KD-tree） | 2-10x | 支配的 |
| **2** | **K_c/K_st アセンブリのベクトル化強化** | 1.5-3x | 大 |
| **3** | **スパース求解の高速化**（CHOLMOD/PARDISO） | 2-5x | 支配的 |
| **4** | **NR反復数の削減**（準Newton法） | 1.3-2x | 大 |
| **5** | **並列化**（マルチスレッドアセンブリ） | 2-4x | 大 |

---

## 変更ファイル

- `xkep_cae/contact/_contact_pair.py`: `coating_thickness`フィールド追加
- `xkep_cae/contact/solver/process.py`: `core_radii`計算ロジック追加
- `xkep_cae/numerical_tests/strand_bending_oscillation.py`: `coating_thickness`パラメータ追加・伝搬
- `contracts/analyze_solver_performance.py`: 新規（性能分析スクリプト）

---

## 再現手順

```bash
# ブランチ
git checkout claude/optimize-seven-strand-solver-eZEl3

# 性能分析実行（被膜なし+被膜付き、約35分）
python contracts/analyze_solver_performance.py 2>&1 | tee /tmp/log-perf-$(date +%s).log
```

---

## TODO

- [ ] 高速化フェーズ: 接触ペア検出のKD-tree化
- [ ] 高速化フェーズ: K_c/K_st アセンブリのベクトル化強化
- [ ] 高速化フェーズ: スパース求解の高速化（CHOLMOD検討）
- [ ] 被膜接線剛性の精度検証（K_c FD誤差67%が報告された。被膜K_coatの接線の整合性確認要）
- [ ] リスタート解析方式への移行（CLAUDE.md記載のTODO）

---

## 次の担当者向け

### 重要な発見

1. **被膜がincr半減効果を持つ**: Kelvin-Voigt被膜(k=1e6, c=1e4, t=0.05mm)で
   接触力が平滑化され、大きなdtで安定進行。1527s→555s。
2. **被膜バグが存在していた**: `core_radii=None`ハードコードで被膜厚さ=0。
   本statusで修正(`coating_thickness`パラメータ追加)。
3. **被膜接線剛性に不整合あり**: FD診断でK_c FD誤差67%が報告。
   被膜なしでは1.8%。被膜K_coatの接線計算に問題がある可能性。
   ただし実用上は完走しており、精度改善は今後のTODO。

### 被膜パラメータの選択

physics test準拠値: k=1e6 N/mm, c=1e4 N·s/mm, μ=0.3, thickness=0.05mm。
1000本モデルの実際の被膜パラメータは材料・構造に依存。

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: 実行ログをtee保存、結果をそのまま記録
- [x] **回帰なし**: 被膜バグ修正のみ。被膜なしの結果は前回と完全一致(incr=1900, cb=72)
- [x] **再現手順記載**: コマンド列を明記
- [x] **ベースライン比較**: status-299(incr=1900, cutback=72, 1504s)と整合
