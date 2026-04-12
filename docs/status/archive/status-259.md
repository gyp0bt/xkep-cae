# status-259: Huber smoothing_delta パイプライン貫通 + 自動推定有効化

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-28
- **ブランチ**: `claude/execute-status-todos-voRnF`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4（新規4件）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. smoothing_delta パイプライン貫通

status-258 で特定された「smoothing_delta=0.0 による接触ON/OFF境界の不連続性」を解消するため、smoothing_delta をパイプライン全体で設定可能にした。

#### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/setup/process.py` | `ContactSetupConfig` に `smoothing_delta: float = 0.0` を追加。`_ContactConfigInput` への受け渡し実装 |
| `xkep_cae/core/batch/strand_bending.py` | `StrandBatchConfig` に `smoothing_delta` を追加。`ContactSetupConfig` への受け渡し実装 |
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `StrandBendingOscillationConfig` に `smoothing_delta` を追加。自動推定ロジック（δ = 5000 / wire_radius）実装 |

#### パイプライン

```
StrandBendingOscillationConfig.smoothing_delta (0=自動推定)
  → 自動推定: 5000.0 / wire_radius
  → _ContactConfigInput.smoothing_delta
  → manager.config.smoothing_delta
  → default_strategies(smoothing_delta=...)
  → HuberContactForceProcess._smoothing_delta
  → delta_h = k_pen / smoothing_delta
  → Huber C1 平滑化
```

### 2. smoothing_delta 自動推定ロジック

`StrandBendingOscillationProcess` で `smoothing_delta=0.0`（デフォルト）の場合、`5000.0 / wire_radius` で自動推定する。`three_point_bend_jig.py` と同じ推定式。

手動指定（`smoothing_delta > 0`）の場合はそのまま使用。

### 3. テスト追加

| テストファイル | テスト名 | 内容 |
|--------------|---------|------|
| `xkep_cae/contact/setup/tests/test_process.py` | `test_smoothing_delta_default_zero` | デフォルト値 0.0 の確認 |
| `xkep_cae/contact/setup/tests/test_process.py` | `test_smoothing_delta_passthrough` | `_ContactConfigInput` まで値が貫通することの確認 |
| `xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py` | `test_smoothing_delta_auto_estimation` | 自動推定公式の確認 |
| `xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py` | `test_smoothing_delta_manual_override` | 手動指定の確認 |

---

## テスト結果

- 新規テスト: 4件（全通過）
- 既存テスト: 558 passed, 20 skipped, 1 xfailed（回帰なし）
- 契約違反: 0件
- lint: 全合格

---

## 再現手順

```bash
git checkout claude/execute-status-todos-voRnF
pip install -e .
# smoothing_delta パイプライン検証
python -m pytest xkep_cae/contact/setup/tests/test_process.py -v --timeout=30
python -m pytest xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py -v --timeout=30
# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"
# 契約検証
python contracts/validate_process_contracts.py
# lint
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## 次セッションへの引き継ぎ

### 残課題（優先度順）

1. **MPC+接触の曲げ揺動テストで smoothing_delta 効果検証**: `StrandBendingOscillationProcess` に smoothing_delta が貫通したので、実際の frac 改善を測定する。ベースライン（smoothing_delta=0）と比較してNR反復数・カットバック回数の変化を確認
2. **smoothing_delta の最適値チューニング**: 5000/wire_radius は three_point_bend_jig の経験値。MPC+撚線曲げでは最適値が異なる可能性がある。k_pen との比率（delta_h = k_pen / smoothing_delta）が物理的に意味のある範囲にあるか確認
3. **T1 Hermite atol 厳格化**: Hermite 非局所 ∂g/∂u 対応（4ノードペア外のDOF結合）が必要（status-258 から継続）
4. **FD診断の改善**: 活性集合変化を除外した K_c 精度評価（gap<0 ペアのみで診断）

### 設計メモ

- `ContactSetupConfig` 経由のパスは `smoothing_delta=0.0` がデフォルト（後方互換性維持）。0 の場合は既存の max(0,x) が使われる
- `StrandBendingOscillationProcess` は `smoothing_delta=0.0` → 自動推定（5000/r）。実質的に Huber C1 平滑化がデフォルト有効になる
- `three_point_bend_jig.py` は既に自動推定を実装済み（変更なし）

---

## 懸念・設計メモ

1. **後方互換性**: `ContactSetupConfig` と `StrandBatchConfig` のデフォルトは 0.0（既存動作と同一）。`StrandBendingOscillationProcess` のみ自動推定で smoothing_delta > 0 がデフォルト有効化
2. **smoothing_delta の物理的意味**: delta_h = k_pen / smoothing_delta はHuber関数の遷移幅。wire_radius=0.5, k_pen=1e6 の場合、delta_h = 1e6 / 10000 = 100。x_pen = k_pen * (-gap) なので gap ≈ delta_h / k_pen = 1e-4 mm の範囲で C1 平滑化される
3. **チューニングの必要性**: 最適な smoothing_delta は問題依存。frac 改善の実測データが必要
