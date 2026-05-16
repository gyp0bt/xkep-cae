# status-287: チャタリング内訳分析 — 活性集合振動ではなく接線剛性不整合が主因

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-04
- **ブランチ**: `claude/analyze-chattering-issue-1AmAM`
- **テスト数**: 621 passed（回帰なし）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-286のTODO「チャタリングという認識をもう一段深くして対策を決めたい」に対応。
NR反復レベルの詳細診断（`NRIterationSnapshot`）を実装し、90度曲げ（接触あり Hertz α=1.5）のチャタリング内訳を精査した。

**最重要発見: frac > 0.10 以降のNR不収束は「接触ペアのON/OFF振動」ではなく、「接線剛性の不整合による遅い収束（線形収束率 > 0.9）」が支配的。活性集合は完全に安定している。**

---

## チャタリング分類体系（新規）

| Type | 名称 | 判定基準 | 意味 |
|------|------|---------|------|
| **A** | 活性集合振動 | ペアON/OFF変化 + 接触DOF残差支配 | 真のチャタリング（ペアが離散的に切替） |
| **B** | 摩擦状態振動 | stick↔slide変化 + 接触DOF残差支配 | 摩擦状態の離散的切替 |
| **C** | 構造系不良 | 非接触DOF残差が支配 | UL/構造系の収束問題 |
| **D** | 接線剛性不整合 | 収束率 > 0.9 (att >= 2) | NR法の2次収束が得られない |
| **E** | 接触力値振動 | 活性集合固定 + 接触DOF残差支配 | ペナルティ力の非線形性による値の振動 |

---

## 分析結果: 7本90度曲げ（Hertz α=1.5, contact ON）

### ベンチマーク条件

- テスト: `contracts/analyze_chattering_breakdown.py`
- ブランチ: `claude/analyze-chattering-issue-1AmAM`
- 構成: 7本, κ=π/200, θ=π/2, penalty_exponent=1.5, μ=0.15
- 結果: **frac=0.9981, incr=551, cutback=60**（status-285再現）

### 全体Type分布（61不収束イベント、589 NR反復の分類）

| Type | 反復数 | 割合 |
|------|-------|------|
| **D** | 304 | **51.6%** |
| **-/E** | 182 | **30.9%** |
| C | 45 | 7.6% |
| A+B+D | 31 | 5.3% |
| A+D | 9 | 1.5% |
| その他 | 18 | 3.1% |

### frac区間別Type分布

| frac区間 | 主要Type | 特徴 |
|---------|---------|------|
| **0-0.10** | A+B+D (48%), C (17%) | 初期接触確立。ペアON/OFF + 摩擦切替 = **真のチャタリング** |
| **0.10-0.50** | **D (68%)**, - (24%) | 活性集合安定。接線剛性不整合 |
| **0.50-0.90** | **D (53%)**, - (38%) | 同上。Type A/Bは実質ゼロ |
| **0.90-1.00** | **- (51%)**, D (41%) | 同上 |

### 不収束時の典型パターン（frac > 0.10）

```
activated=0, deactivated=0, stick→slide=0, slide→stick=0
R_c ≈ 1e-5 ~ 1e-4, R_s ≈ 1e-6 ~ 1e-5
active=12~14（固定）, sliding=65~130（固定）
収束率 > 0.9（線形収束、2次収束なし）
```

---

## 解釈

### なぜ「チャタリング」と認識されていたか

凍結モードのトリガー条件が「残差の振動/停滞」（直近6反復のmax/min比 < 100）で検知されるため、**活性集合が変化していなくても**チャタリングと誤認されていた。実際には:

1. **接触力の値が反復ごとに微小に変動**（ペナルティの非線形性による）
2. **NRの収束率が0.9以上**（2次収束ではなく線形収束、あるいは停滞）
3. 凍結モードで接触力を固定しても、構造系が収束した後に再評価すると接触力が変わり、また不収束

### 根本原因の仮説

**接線剛性（K_c + K_st）が不正確で、NR法の2次収束が得られていない。**

裏付け:
- Type D（収束率 > 0.9）が全反復の52%を占める
- FD接線診断（status-258）では活性集合変化時に94-100%不整合が報告されたが、活性集合が安定した後も収束率が悪い
- Hertz型ペナルティ(α=1.5)の導関数整合性がFD検証されていない（status-285 TODO）

### 対策の方向性

| 優先度 | 対策 | 根拠 |
|--------|------|------|
| **高** | FD接線診断でHertz型整合性検証 | Type D主因 → K_c/K_st の不整合特定が最優先 |
| **高** | penalty_exponent=1.5 の ∂p/∂g FD検証 | 導関数補正の正確性を確認 |
| 中 | tol_force 緩和 (1e-8 → 1e-6) の検討 | 接触力値の微小振動が tol 到達を阻害 |
| 低 | 凍結モードの適用条件見直し | 活性集合安定時は凍結不要 |
| 低 | 初期フェーズ（frac < 0.10）のみ別のチャタリング対策 | Type A+Bは初期のみ |

---

## 実装内容

### 1. NRIterationSnapshot dataclass（チャタリング内訳診断）

`_diagnostics.py` に追加:

| 項目 | 内容 |
|------|------|
| `NRIterationSnapshot` | NR1反復の詳細スナップショット（15フィールド） |
| `classify_chattering_type()` | Type A/B/C/D/E 分類関数 |
| `ConvergenceDiagnosticsOutput.nr_iteration_snapshots` | NRスナップショット履歴フィールド |

### 2. NRソルバーへの計装

`_newton_dynamic.py` に追加:

| 項目 | 内容 |
|------|------|
| ペア状態遷移追跡 | `_prev_pair_statuses` で前反復と比較 |
| 接触/構造DOF残差分離 | `_contact_dofs()` マスクで R_u を分割 |
| ログ出力 | 検知時に `[Type]` + `R_c`, `R_s` を出力 |
| 不収束サマリ | 直近10反復のType分布を出力 |

### 3. 分析スクリプト

`contracts/analyze_chattering_breakdown.py`:
- 90度曲げ実行 + インクリメント診断分析 + NRスナップショット詳細分析

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/contact/solver/_diagnostics.py` | NRIterationSnapshot + classify_chattering_type + フィールド追加 |
| `xkep_cae/contact/solver/_newton_dynamic.py` | スナップショット記録 + Type付きログ出力 |
| `contracts/analyze_chattering_breakdown.py` | 新規: 分析スクリプト |

---

## 再現手順

```bash
git checkout claude/analyze-chattering-issue-1AmAM
python contracts/analyze_chattering_breakdown.py 2>&1 | tee /tmp/log-chattering-analysis.log
# 出力: frac=0.9981, incr=551, cutback=60
# 不収束ログに [D], [-], [A+B+D] 等のType分類が出力される
```

---

## TODO

- [ ] FD接線診断でHertz型（α=1.5）の ∂p/∂g 整合性を検証
- [ ] K_c + K_st の不整合箇所を活性集合安定状態で特定
- [ ] 収束率改善策（修正NR法 or quasi-Newton の検討）
- [ ] 揺動フェーズの同様の内訳分析（checkpoint復元後）
- [ ] tol_force 緩和の効果検証（1e-6 or 動的tol）
- [ ] 凍結モードの適用条件をType A/B検知時のみに限定する検討

---

## 次の担当者向け

### 最重要ポイント

**frac > 0.10 のNR不収束は「チャタリング」ではなく「接線剛性不整合」。** 
活性集合は安定しており、凍結モードは原理的に効かない。
NR法の2次収束を回復するために、接線剛性の正確性を高めるのが正攻法。

### ログの読み方

```
低残差チャタリング検知[D] → 接触凍結モード (att=15, ||R||/||f||=6.100e-05, R_c=4.03e-05, R_s=2.57e-06, ...)
```
- `[D]`: Type D（接線剛性不整合）
- `R_c`: 接触DOFの残差ノルム
- `R_s`: 非接触（構造）DOFの残差ノルム

```
不収束 チャタリング内訳(直近10) [-:4, C:1, D:6] R_c=..., R_s=..., activated=0, deactivated=0, stick→slide=0, slide→stick=0
```
- Type分布の括弧内が直近10反復の分類
- `activated=0, deactivated=0`: 活性集合変化なし

---
