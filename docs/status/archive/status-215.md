# status-215: UL+GeneralizedAlpha 動的解析の振動修正

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-20
**ブランチ**: `claude/fix-stress-analysis-mpvzw`

---

## 概要

梁揺動解析で荷重・応力が振動しない問題（status-213 TODO）を修正。
2つの根本原因を特定し修正:

1. **UL参照配置リセットが復元力を消失させる問題**
2. **初速度のモーダル質量補正の欠如**

## 変更内容

### 1. UL参照配置更新の動的解析スキップ

**ファイル**: `xkep_cae/contact/solver/process.py` (line 535)

**問題**: UL（Updated Lagrangian）定式化では各ステップ収束後に参照配置を更新し
`state.u = 0` にリセットする。動的解析では、このリセットにより CR 梁の
`assemble_internal_force(u_incr≈0)` が近ゼロの復元力を返し、振動が再現されない。

**修正**: `if _ul:` → `if _ul and not _dynamics:`

CR 梁の corotational 分解が大変形を処理するため、動的解析では
UL 参照配置更新は不要。`state.u` を累積変位として保持することで
正しい復元力が計算される。

### 2. 初速度のモーダル質量補正

**ファイル**: `xkep_cae/numerical_tests/beam_oscillation.py`, `xkep_cae/numerical_tests/three_point_bend_jig.py`

**問題**: 初速度を `v₀ = ω₁ * δ` で計算していたが、これは単一DOF系の公式。
梁の集中質量では節点質量 `m_mid = ρAL/n_elems` がモーダル質量
`M₁ = ρAL/2` より小さいため、振幅が `2/n_elems` 倍に縮小される。

**修正**: `v₀ = ω₁ * δ * M₁ / m_mid` （モーダル質量補正）
- `M₁ = φ₁ᵀ M φ₁`（実際の質量行列から計算）
- `m_mid = M[mid_y_dof, mid_y_dof]`

### 3. エネルギー減衰率のピーク変位点比較

**ファイル**: `xkep_cae/numerical_tests/beam_oscillation.py`

**問題**: `energy_decay_ratio = E_final / E_init` で、E_init は集中初速度による
過大値、E_final は最終ステップの `E_kinetic=0` による過小値。

**修正**: 変位ピーク（KE≈0）でのひずみエネルギーを比較。

### 4. xfail マーカー除去

| テスト | 変更 |
|--------|------|
| `tests/test_beam_oscillation.py::test_small_oscillation_detected` | xfail 除去 → PASS |
| `tests/test_beam_oscillation.py::test_numerical_dissipation_rate` | xfail 除去 → PASS |
| `tests/contact/test_three_point_bend_jig.py::test_dynamic_response_has_oscillation` | xfail 除去 → PASS |
| `tests/contact/test_three_point_bend_jig.py::test_max_deflection_order` | xfail 除去 → PASS |

## 解析結果

### 小振幅（amplitude=0.1mm, 20要素, 2周期）

| 指標 | 修正前 | 修正後 |
|------|--------|--------|
| amplitude_ratio | 0.101 | **0.994** |
| 方向反転 | なし（単調減衰） | **4回**（2周期） |
| energy_decay_ratio | 0.000 | **0.734** |

### 大振幅（amplitude=5.0mm, 100要素, 3周期）

| フィールド | 最大値 | 単位 |
|-----------|--------|------|
| SK1 (曲率) | 1.93e-02 | 1/mm |
| LE11 (ひずみ) | 1.93e-02 | - |
| S11 (応力) | 1930 | MPa |

## 画像出力

出力先: `tmp/oscillation/`（25枚）— 振動が時刻歴プロットで確認可能

## テスト結果

```
468 passed, 1 xfailed (full suite, not slow)
14 passed (tests/test_beam_oscillation.py — slow 含む)
ruff check: All checks passed
ruff format: OK
契約違反: 0件
条例違反: 0件
```

## TODO

- [ ] 単線の剛体支えと押しジグによる動的三点曲げの解析解一致
- [ ] 数値粘性の定量評価: rho_inf 依存性の検証（振動修正済みで実施可能に）
- [ ] UL定式化の準静的→動的切替のスムーズ化（ハイブリッド方式の検討）

---
