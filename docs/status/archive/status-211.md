# status-211: smooth_penalty 正定値接線 + 動的三点曲げ Process 基盤

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-18
**ブランチ**: `claude/execute-status-todos-zWe36`

---

## 概要

status-210 の TODO を実行。3つの成果物:

1. **smooth_penalty 接触接線の正定値近似** — ゼロ接線 → sigmoid 重み付き正定値行列
2. **DynamicThreePointBendJigProcess** — 荷重制御動的三点曲げ Process 新規実装
3. **適応時間増分の dt_max_fraction 初期ステップ制限** — 動的解析の多ステップ分割

## 1. smooth_penalty 接触接線の正定値近似（v2.0.0 → v3.0.0）

### 変更

| ファイル | 変更 |
|---------|------|
| `xkep_cae/contact/contact_force/strategy.py` | `tangent()`: ゼロ行列 → `k_pen * sigmoid(-δg) * g_shape⊗g_shape` |
| `xkep_cae/contact/contact_force/tests/test_strategy.py` | テスト更新（正定値・対称性・非接触時ゼロ） |

### 技術的背景

softplus の厳密接線 `dp/dg = -k_pen * sigmoid(-δg)` は負定値。
直接 K_T に加算すると不定値化 → NR 不収束。
代わりに絶対値（正定値近似）`K_contact = k_pen * sigmoid * g_shape⊗g_shape` を使用。

### 検証

- **既存テスト 431 passed, 0 failed** — smooth_penalty 接線追加は既存梁–梁接触テストに影響なし
- **HEX8 接触ジグ**: 正定値接線だけでは不十分（k_pen/K_beam ミスマッチが根本原因）

## 2. DynamicThreePointBendJigProcess

### 新規実装

| ファイル | 内容 |
|---------|------|
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | `DynamicThreePointBendJigProcess` + Config/Result |
| `xkep_cae/numerical_tests/tests/test_three_point_bend_jig.py` | @binds_to テスト追加 |
| `tests/contact/test_three_point_bend_jig.py` | 動的収束・物理テスト5件追加 |

### 設計

- **荷重制御**: ステップ荷重 P = k_EB × δ を中央節点に適用（変位制御だと振動が処方値に強制される）
- **f_ext_base**: 定常荷重（ステップ荷重）
- **f_ext_total = 0**: 荷重増分なし → f_ext = f_ext_base（一定）
- **質量行列**: ULCRBeamAssembler.assemble_mass()（集中質量）
- **時間積分**: Generalized-α（dt_physical = 総計算時間）

## 3. 適応時間増分の改修

| ファイル | 変更 |
|---------|------|
| `xkep_cae/contact/solver/_adaptive_stepping.py` | `dt_max_fraction > 0` 時に初期ステップも制限 |

## テスト結果

```
431 passed, 3 xfailed
契約違反: 0件
条例違反: 0件
ruff check: All checks passed
ruff format: already formatted
```

xfail 内訳:
- HEX8 接触ジグ NR 収束問題（status-210 継続）
- 動的三点曲げ最終変位一致（dt_sub 計算問題）
- 動的三点曲げ振動検出（dt_sub 計算問題）

## TODO

- [ ] **動的時間積分の dt_sub 問題**: ContactFrictionProcess の `dt_sub = dt_physical * delta_frac` で
      UL + 適応ステップにおける実効 dt が意図の1/n_stepsになる。
      根本対策: dt_physical と load_frac の関係を再設計（dt_physical = 総時間 or ステップ時間の明確化）
- [ ] **HEX8 接触ジグ NR 収束**: k_pen/K_beam ミスマッチ（k_pen ≈ 10, K_beam ≈ 0.5）が根本原因。
      接触接線が構造剛性の20倍 → NR 補正が発散方向。
      対策案: k_pen を梁グローバル剛性ベースに設定 or 接触力の段階的活性化
- [ ] 動的三点曲げの物理テスト xfail 解消（dt_sub 修正後）
- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動の Phase2 xfail 解消

## 設計上の懸念

- **dt_sub の二重分割**: `dt_sub = dt_physical * delta_frac` の設計は、dt_physical がステップ幅か総時間かが
  曖昧。既存の動的テスト（numerical_tests/dynamic_runner.py）では dt_physical にステップ幅を渡し、
  load_frac = 1.0 で 1 ステップ実行しているため問題が表面化していなかった。
  多ステップ動的解析では dt_physical の意味を明確にする必要がある。

---
