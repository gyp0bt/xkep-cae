# status-219: 動的三点曲げ接触 k_pen 適正化 + ペア別診断 + k_pen 上書きバグ修正

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-20
**ブランチ**: `claude/improve-three-point-bend-Qe3Xt`

---

## 概要

status-218 で特定した動的三点曲げ接触の k_pen 不足問題に対する根本対策を実施。
診断機能の高度化（ペア別接触診断）と、k_pen が AutoBeamEIPenalty に上書きされるバグを修正。

**0.5 周期の収束テスト PASSED**（2次収束、1-2 NR iter/step）。

## 変更内容

### 1. DynamicPenaltyEstimateProcess 新設

動的解析用 k_pen を c0*M_ii ベースで自動推定する Process を新設。

**推定式**: `k_pen = scale × c0 × m_ii`
- c0 = 1/(β×dt²) — Generalized-α の有効質量係数
- m_ii = ρ×A×L_elem/2 — 代表集中質量
- scale = 0.2（デフォルト）— dt cutback 1回で K_eff 正定値条件を保持

**配置**: `xkep_cae/contact/penalty/strategy.py`
**テスト**: 9件追加（API正確性 + 物理的妥当性 + 正定値マージン）

### 2. k_pen 上書きバグ修正（重要）

`ContactFrictionProcess.process()` の `k_pen = _penalty_strategy.compute_k_pen(0, 1)` が
`DynamicPenaltyEstimateProcess` で計算した動的 k_pen を **静的梁剛性ベースの値で上書き**していた。

**修正**: `contact_setup.k_pen > 0` の場合はそれを尊重し、`compute_k_pen` で上書きしない。
k_pen continuation も同様にスキップ。

### 3. exact_tangent 設定可能化

- `_ContactConfigInput` に `exact_tangent` フィールド追加
- `ContactFrictionProcess` のハードコード `False` を config 参照に変更
- `DynamicThreePointBendContactJigConfig` から制御可能に

### 4. ペア別接触診断

- `PairDiagnosticsEntry`: pair_id/elem_a/elem_b/gap/p_n/status を記録
- `ConvergenceDiagnosticsOutput` に `pair_snapshots` フィールド追加
- NR 反復ごとに全ペアの状態をスナップショット（動的・静的両方）
- 診断レポートにアクティブペアの詳細（最大10件）を表示

### 5. 発散早期検知の改善

- 残差フロア近傍（1%以内）での発散判定抑制
- softplus の ghost force で残差フロアが生じるため

## 変更ファイル

| ファイル | 変更内容 |
|----------|---------|
| `xkep_cae/contact/penalty/strategy.py` | DynamicPenaltyEstimateProcess 新設 |
| `xkep_cae/contact/penalty/__init__.py` | re-export 追加 |
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | k_pen推定Process化, scale=0.2, NRパラメータ調整 |
| `xkep_cae/contact/_contact_pair.py` | _ContactConfigInput に exact_tangent 追加 |
| `xkep_cae/contact/solver/process.py` | k_pen上書きバグ修正, exact_tangent config参照 |
| `xkep_cae/contact/solver/_diagnostics.py` | PairDiagnosticsEntry + pair_snapshots |
| `xkep_cae/contact/solver/_newton_uzawa_dynamic.py` | ペア別診断 + 発散検知改善 |
| `xkep_cae/contact/solver/_newton_uzawa_static.py` | ペア別診断収集 |
| `tests/contact/test_dynamic_penalty_estimate.py` | 新テスト9件 |
| `tests/contact/test_three_point_bend_jig.py` | n_uzawa_max テストヘルパー修正 |

## 収束特性

| 指標 | status-218 | status-219 |
|------|-----------|-----------|
| k_pen | AutoBeamEIPenalty(静的) | 0.2×c0×m_ii ≈ 11.2 N/mm |
| exact_tangent | False（ハードコード） | True（config設定可能） |
| NR 収束 | 非収束 | 2次収束（1-2 iter/step） |
| 0.5周期テスト | FAIL | PASS（36 steps, 38秒） |

## 診断で判明した数値

```
f1 = 396 Hz, T1 = 2.52e-3 s（固有振動数）
dt = T1/40 = 6.3e-5 s
beta = 0.277 (rho_inf=0.9)
c0 = 9.08e8, m_ii = 6.17e-8 ton
c0*m_ii = 56 N/mm（動的有効剛性スケール）
k_pen(20%) = 11.2 N/mm
(1-alpha_m)*c0*m_ii = 32 N/mm（exact_tangent 正定値限界）
k_static = 7.54 N/mm（48EI/L³）
K_eff正定値条件: 4ペア×k_pen×||g_shape||²_max < (1-α_m)×c0×m_ii
  → 4×11.2×2.0=89.6 > 32.4（元dt）→ dt cutback 1回で 89.6 < 130 ✓
```

## 発見した問題

### 1. k_pen 上書きバグ
`ContactFrictionProcess` が `_penalty_strategy.compute_k_pen()` で動的 k_pen を静的値に上書き。
**根本原因**: penalty strategy が暗黙に k_pen を決定する設計。全ロジックを明確な Input/Output の Process にすべき。

### 2. Uzawa + 変位/エネルギー収束の非互換
Uzawa 拡大ラグランジアンは **力収束**（R_u < tol）が前提。
現在の NR は変位/エネルギー収束で完了するため、力残差が大きいまま Uzawa に入り、
lambda 更新後の NR が収束しない。

### 3. 三点曲げ問題設定の誤り
**現在**: ワイヤに初期変位 → 自由振動 → ジグにバウンド（接触/離脱繰り返し）
**本来の意図**: ジグを変位制御で下に押し込む単純な三点曲げ

## TODO（次セッションへの引き継ぎ）

- [ ] **三点曲げ問題設定の修正**: ジグを変位制御で押し下げる三点曲げに変更。現在の自由振動バウンド方式は本来の意図と異なる
- [ ] **Uzawa 有効化（力収束前提の解決）**: NR 内ループが力収束でも完了するよう修正し、Uzawa 拡大ラグランジアンを有効化。純粋ペナルティ法（n_uzawa_max=1）は接触力精度が不十分で、エネルギー非保存（減衰率213×）を引き起こす
- [ ] **penalty strategy の Process 化**: k_pen 決定ロジックを明確な Input/Output の Process に統一。暗黙の上書き（AutoBeamEIPenalty vs DynamicPenaltyEstimate）を排除
- [ ] **ヘルパー関数の Process 化**: テストヘルパー `_dynamic_contact_config` 等もデフォルト上書きが見えにくい。Process として明示的なデータフローに
- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動

---
