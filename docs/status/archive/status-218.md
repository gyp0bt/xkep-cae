# status-218: 動的三点曲げ接触収束問題の診断・k_pen 根本原因特定

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-20
**ブランチ**: `claude/resume-three-point-bending-doVJN`

---

## 概要

動的三点曲げ接触ジグ（`DynamicThreePointBendContactJigProcess`）の収束問題を調査。
前セッション（status-217 TODO）から引き継いだ作業。

**収束失敗の根本原因を特定**: k_pen が動的解析に対して **6桁小さい**。

## 調査結果

### 1. k_pen 問題（根本原因）

静的梁剛性ベースの k_pen 推定: `k_pen = 0.5 * 48EI/L³ ≈ 3.77e-3 N/mm`

動的解析の有効剛性スケール: `c0 * M_ii ≈ 614 N/mm`（Generalized-α, dt=T1/40）

**接触力はワイヤの慣性力の ~10⁻⁶ 倍 → ワイヤがジグを貫通する。**

### 2. exact_tangent の効果検証

| 設定 | 結果 |
|------|------|
| `exact_tangent=False`（正定値近似）+ 小 k_pen | 線形収束 0.85/iter、残差フロア ~3%、200 反復でも非収束 |
| `exact_tangent=True`（厳密負定値）+ 小 k_pen | 初回で残差 1→223 爆発（Newton ステップが接触力方向に過大） |
| `exact_tangent=False` + 動的 k_pen（c0*M の 10%）| 線形収束は改善されたが、依然 200 反復で非収束 |

### 3. ユーザー質問への回答

1. **「スティックスリップが原因か？」** → 主因ではない。mu=0（摩擦なし）でも同じ収束失敗。根本原因は k_pen 不足。
2. **「押しジグ vs 抑えジグ？」** → 押しジグの2接触エッジのみが接触ペア。支持は境界条件（DOF固定）。
3. **「診断で判別できている？」** → 現状不可。ConvergenceDiagnosticsOutput は集約値のみ。

### 4. 接触遷移時の dt 制御バグ修正

`_adaptive_stepping.py`: `prev_n_active > 0` ガードが `0→N` の新規接触発生を捕捉できていなかった。修正済み。

## 変更ファイル

| ファイル | 変更内容 |
|----------|---------|
| `xkep_cae/contact/solver/process.py` | exact_tangent 検討（最終的に False に据え置き） |
| `xkep_cae/contact/solver/_adaptive_stepping.py` | 接触遷移 dt 制御: 0→N ケースを捕捉 |
| `xkep_cae/contact/solver/_newton_uzawa_dynamic.py` | Uzawa バグ修正 + 残差爆発検知（前セッション） |
| `xkep_cae/contact/solver/_newton_uzawa_static.py` | 同 Uzawa バグ修正（前セッション） |
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | 動的 k_pen 推定（c0*M ベース）、dt/grow パラメータ調整 |
| `xkep_cae/core/data.py` | du_norm_cap フィールド追加（前セッション） |
| `xkep_cae/contact/contact_force/strategy.py` | exact_tangent ドキュメント整備（前セッション） |
| `tests/contact/test_three_point_bend_jig.py` | mu=0 分離テスト追加 |

## TODO（次セッションへの引き継ぎ）

- [ ] **k_pen 適正化の本格実装**: 動的 k_pen = 0.1 * c0 * M_ii でも残差 ~0.4% フロアで非収束。k_pen スケーリングの再検討が必要。候補:
  - k_pen をさらに大きく（c0*M の 50-100%）
  - smoothing_delta も k_pen に連動して調整（遷移幅と力の整合性）
  - Uzawa 外ループ（n_uzawa_max > 1）で接触力精度を補強
- [ ] **exact_tangent の再検討**: k_pen が適正化された状態での exact_tangent=True の動作検証。k_pen が十分大きければ正定値近似でも2次収束する可能性がある
- [ ] **ペア別接触診断の追加**: ConvergenceDiagnosticsOutput にペア別 gap/p_n/status 履歴を追加
- [ ] **接触力残差の分析**: 残差フロアの原因を特定。接触力の非線形性（softplus）vs 接触幾何の近似誤差
- [ ] **NR 収束基準の再評価**: tol_force=1e-8 は接触問題では厳しすぎる可能性。エネルギー収束基準との併用
- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動

## 診断に有用な数値

```
L=100mm, d=2mm, E=200GPa, ρ=7.85e-9 ton/mm³
I = π*d⁴/64 ≈ 0.785 mm⁴, A = π mm²
f₁ ≈ 1585 Hz, T₁ ≈ 6.3e-4 s, ω₁ ≈ 9960 rad/s

静的梁剛性: 48EI/L³ = 7.54e-3 N/mm
動的有効剛性: c0*M_ii ≈ 614 N/mm (dt=T1/40, β=0.2525)
現 k_pen: 0.5 * 7.54e-3 ≈ 3.77e-3 N/mm ← 6桁小さい！
必要 k_pen: ~60-600 N/mm (c0*M の 10-100%)
```

## 設計上の懸念

1. **k_pen の自動推定は解析タイプ（静的/動的）で変えるべき**: 現在の `0.5 * k_beam_global` は静的専用。動的では c0*M スケールが必要。これは `DynamicThreePointBendContactJigProcess` 固有ではなく `ContactFrictionProcess` レベルで対応すべき可能性がある。

2. **smoothing_delta と k_pen の整合性**: softplus の遷移幅 1/δ は物理的なギャップスケールに対応。k_pen を大きくすると、接触力の勾配 `k_pen * sigmoid` が急峻になり、NR の非線形性が増す。δ と k_pen の同時チューニングが必要。

3. **近似接線の限界**: `weight = k_pen * abs(deriv)` は modified Newton の一種で、理論的に線形収束のみ。接触問題の NR で2次収束を得るには厳密接線が不可欠だが、K_eff の正定値性確保が前提条件。

---
