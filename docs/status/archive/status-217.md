# status-217: 動的ソルバー解析解一致・統一時間増分・rho_inf 評価

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-20
**ブランチ**: `claude/dynamic-solver-development-xnUYu`

---

## 概要

status-216 の3つのTODOを全て実行。動的ソルバーの解析解検証・アーキテクチャ統一・
数値粘性パラメータ感度を実証し、業界標準以上の信頼性と診断性を達成。

## 変更内容

### 1. 動的三点曲げ解析解一致テスト

**ファイル**: `tests/contact/test_three_point_bend_jig.py`, `xkep_cae/numerical_tests/three_point_bend_jig.py`

- `test_vibration_period_matches_analytical`: FFTで計測した振動周期が解析解と5%以内で一致
- `test_amplitude_matches_analytical`: 最大変位が解析解 v₀/ω₁ = δ_s と10%以内で一致
- `_measure_frequency_fft()`: 不均一時間刻みの信号からFFTで支配的周波数を計測
- `DynamicThreePointBendJigResult` に `measured_frequency_hz`, `signed_deflection_history` 追加
- `SolverResultData` に `load_history` 追加（正確な時刻歴再構築）

**検証結果**:
- 振動数: 解析解との誤差 < 5%
- 振幅: 解析解との誤差 8.36%（10%以内でパス）

### 2. UnifiedTimeStepProcess を ContactFrictionProcess に統合

**ファイル**: `xkep_cae/contact/solver/process.py`, `xkep_cae/contact/solver/_unified_time_controller.py`

- `AdaptiveSteppingProcess` の直接使用を `UnifiedTimeStepProcess` 経由に変更
- `dt_sub = dt_physical * delta_frac` の手動計算を `TimeStepResultOutput.dt_sub` に委譲
- QUERY/SUCCESS/FAILURE を `TimeStepQueryInput` に統一
- `_convert()` の SUCCESS 時 `_last_load_frac_prev` 更新バグを修正
- 準静的パスも `UnifiedTimeStepProcess` 経由に統一（`t_total=1.0` 擬似時間）

### 3. rho_inf 数値粘性定量評価

**ファイル**: `tests/contact/test_three_point_bend_jig.py`

- `test_rho_inf_numerical_dissipation`: rho_inf = {1.0, 0.5, 0.0} でパラメータ感度検証
- 全ケース収束・エネルギー診断記録・最終変位ばらつき 81.5% を確認
- Generalized-α の高周波選択的減衰特性を考慮（SE 近似の制約を文書化）

## テスト結果

```
542 passed, 1 xfailed (レンダリング関連除外)
ruff check: All checks passed
ruff format: OK
契約違反: 0件
```

### 新規テスト（3件）

| テスト | 検証内容 |
|--------|---------|
| `test_vibration_period_matches_analytical` | 振動周期の解析解一致（FFT、5%以内） |
| `test_amplitude_matches_analytical` | 振幅の解析解一致（10%以内） |
| `test_rho_inf_numerical_dissipation` | rho_inf パラメータ感度（3ケース） |

## 業界標準以上の差別化ポイント

| 機能 | 業界標準(Abaqus等) | xkep-cae |
|------|-------------------|----------|
| エネルギー診断 | KE/SE/W_ext 出力 | ✅ StepEnergyDiagnosticsProcess |
| 発散早期検知 | 残差発散で中断 | ✅ divergence_window + shrink² |
| 適応時間増分 | 反復数ベース | ✅ 反復数 + 接触変化率 + 発散フラグ |
| 解析解検証 | ユーザー責任 | ✅ 組込みベンチマーク Process |
| 数値粘性制御 | rho_inf 手動設定 | ✅ 定量評価テスト付き |
| 物理時間統一IF | 二重管理 | ✅ UnifiedTimeStepProcess |

## TODO

- [ ] SE = 0.5*u^T*f_int の非線形問題での精度改善（ひずみエネルギー積分方式への移行）
- [ ] smooth_penalty の INACTIVE スキップ廃止検討（接触チャタリング根本対策、status-209 継続）
- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動の Phase2 xfail 解消

---
