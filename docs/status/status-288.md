# status-288: 収束診断ログ構造化 + Type D対策（FD自動トリガー・NR拡張）

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-04
- **ブランチ**: `claude/convergence-diagnosis-logging-iHaYP`
- **テスト数**: 621 passed（回帰なし、既知の2テスト失敗は変更前と同一）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-287で実装したNR反復レベルのチャタリングタイプ診断（Type A/B/C/D/E分類）を構造化し、**常にログ出力される**ように改善。さらにstatus-287で特定されたType D（接線剛性不整合、全反復の52%）に対する自動検知・対策メカニズムを実装。

---

## 実装内容

### 1. 収束診断ログの構造化

**問題**: タイプ分類はNR反復ごとに計算されていたが、ログ出力はチャタリング検知時と不収束サマリのみ。通常NR進捗（5反復ごと）にはタイプ情報が含まれず、不収束の原因追跡が困難だった。

**対策**:

| 変更 | 内容 |
|------|------|
| NR進捗ログにType付与 | 5反復ごとの進捗ログに `[D]`, `[A+B]` 等のType分類 + 収束率を追加 |
| NR診断サマリの常時出力 | 不収束時は**常に**Type分布を出力（`show_progress` に依存しない）。収束時も15反復超で出力 |
| 診断レポートにType分布追加 | `_format_diagnostics_report()` にNR Type分布 + 直近10反復の内訳 + 最終スナップショットを追加 |

**ログ出力例**:
```
  Incr 42 (frac=0.3500), attempt 15 [D], ||R_t||/||f|| = 3.5e-04, ||R_r|| = 1.2e-06, rate=0.932, active=12
  ...
  [NR診断] Incr 42 (frac=0.3500), 不収束 att=50, Type分布[D:30(60%), -:15(30%), E:5(10%)], 直近10[D:6, -:3, E:1], R_c=4.03e-05, R_s=2.57e-06, active=12, sliding=65, ...
```

### 2. Type D自動検知・FD接線診断トリガー

**問題**: Type D（収束率 > 0.9）が全反復の52%を占めるが、検知後のアクションがなかった。FD接線診断はストール検知(`_relax_active`)時のみトリガーで、Type D特化の応答がなかった。

**対策**:

| 設定パラメータ | デフォルト | 説明 |
|---------------|-----------|------|
| `type_d_auto_fd` | `True` | Type D連続検知時のFD診断自動トリガー |
| `type_d_consecutive_threshold` | `5` | FD診断トリガーまでの連続Type D回数 |
| `type_d_tangent_refresh_rate` | `0.85` | この収束率を超えたら接線リフレッシュ考慮 |
| `type_d_extra_attempts` | `15` | Type D検知時の追加NR反復上限 |

**動作フロー**:
1. NR反復ごとにType分類を追跡。連続Type D（活性集合安定）を `_consecutive_type_d` でカウント
2. 連続5回超過で自動的にFD接線診断をトリガー（1インクリメントにつき1回）
3. FD結果に基づきNR反復上限を拡張（線形収束を許容）
4. FD診断結果を `diag.type_d_fd_reports` に記録

### 3. 低残差チャタリング検知のType D分岐

**問題**: 低残差チャタリング検知（att>=15, 残差振動/停滞）で常に接触凍結モードに入っていたが、Type D支配的（活性集合安定）の場合は凍結が原理的に無効。

**対策**:
- Type D支配的（連続Type D >= 3 + 活性集合変化なし）を検知
- 凍結モード**ではなく**NR反復上限拡張を実施
- ログに `[Type D対策]` タグで対策内容を明示

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/contact/solver/_diagnostics.py` | `type_d_fd_reports`フィールド追加、`_format_diagnostics_report()`にNR Type分布出力追加 |
| `xkep_cae/contact/solver/_newton_dynamic.py` | Type D追跡変数追加、NR進捗ログにType+rate追加、FD診断自動トリガー条件拡張、低残差チャタリングのType D分岐、NR診断サマリ常時出力 |

---

## 設計判断

### なぜ凍結モードではなくNR拡張か

status-287の分析で、frac > 0.10 のNR不収束の52%がType D（接線剛性不整合）。活性集合は完全に安定しており、接触凍結（活性集合変化の抑制）は原理的に効かない。Type D の本質は「接線剛性が不正確→NRの2次収束が得られない→線形収束で反復数が不足」。対策として:

1. **NR反復上限拡張**: 線形収束率0.9でも反復を増やせば収束する可能性がある
2. **FD診断**: 不整合の程度を定量化し、今後のK_c/K_st改善の指針とする
3. **不要な凍結回避**: Type Dに凍結を適用すると、凍結→再評価→再凍結サイクルで無駄な反復が発生

### ログの設計思想

- **不収束時は常にType分布を出力**: `tee` でファイルに残れば後から分析可能
- **収束時は15反復超のみ**: 正常収束（2-5反復）でノイズを増やさない
- **NR進捗ログにrate追加**: 2次収束（rate << 1）か線形収束（rate ~ 0.9）かが即座にわかる

---

## 再現手順

```bash
git checkout claude/convergence-diagnosis-logging-iHaYP
# テスト
python -m pytest xkep_cae/contact/solver/tests/ -x -q
# lint
ruff format --check xkep_cae/ tests/ && ruff check xkep_cae/ tests/
# 90度曲げで収束診断ログを確認
python contracts/analyze_chattering_breakdown.py 2>&1 | tee /tmp/log-type-d-diagnosis.log
```

---

## TODO

- [ ] FD接線診断でHertz型（α=1.5）の ∂p/∂g 整合性を検証（実際の90度曲げで実行）
- [ ] K_c + K_st の不整合箇所を活性集合安定状態で特定（FD診断結果を解析）
- [ ] 収束率改善策の検討（修正NR法 or quasi-Newton）
- [ ] Type D時のdelta_hブースト効果検証
- [ ] tol_force 緩和の効果検証（1e-6 or 動的tol）
- [ ] 凍結モードの適用条件をType A/B検知時のみに限定する検討

---

## 次の担当者向け

### 最重要ポイント

status-287の分析結果を受けた対策の第一弾。ログの構造化により、今後のNR不収束の原因追跡が格段に容易になった。Type D自動検知+FD診断は「どこが不整合か」を定量的に示す基盤。

### ログの読み方（更新）

```
  Incr 42 (frac=0.3500), attempt 15 [D], ||R_t||/||f|| = 3.5e-04, rate=0.932, active=12
```
- `[D]`: Type D（接線剛性不整合）。`rate=0.932`: 線形収束率（1.0に近いほど悪い）

```
  [Type D対策] FD診断: 接線方向は有効 (dir_ratio=0.85), 線形収束許容→NR上限65
```
- FD診断の結果。`dir_ratio < 1.0`: Newton方向は残差を減少させている

```
  [NR診断] Incr 42 (frac=0.3500), 不収束 att=50, Type分布[D:30(60%), -:15(30%), E:5(10%)]
```
- インクリメント終了時の全体Type分布サマリ

---
