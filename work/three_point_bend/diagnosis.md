# 三点曲げ動的接触 — 不収束診断

[README.md](../../README.md) | [status-index](../../docs/status/status-index.md)

## 概要

L=100mm, d=1.7mm, push=30mm, n_periods=30 の動的接触三点曲げにおいて、
全実行が frac≈0.89（push≈26.7mm）で不収束終了。30mm 押し切りに至っていない。

## 実行結果一覧

| # | ラベル | E [MPa] | k_pen 方式 | status filter | push到達 [mm] | 接触力 [N] | 計算時間 [s] | ログファイル |
|---|--------|---------|-----------|--------------|--------------|-----------|------------|------------|
| 1 | E100 動的kpen v1 | 100 | 動的(18.3) | あり | 26.7 | 527 | 926 | 01_*.log |
| 2 | E100 動的kpen v2 | 100 | 動的(18.3) | あり | 26.7 | 527 | 1281 | 02_*.log |
| 3 | E100 梁kpen | 100 | 梁(19.7) | あり | 26.5 | 526 | 984 | 03_*.log |
| 4 | E25 梁kpen noFilt | 25 | 梁(4.92) | なし | 3.4 | 14 | 1575 | 04_*.log |
| 5 | E100 梁kpen noFilt | 100 | 梁(16k) | なし | 3.4 | 56 | 1632 | 05_*.log |
| 6 | E25 動的kpen revert | 25 | 動的(4.56) | あり | 26.7 | 132 | 1313 | 06_*.log |

## 不収束メカニズム

### 直接原因
`AdaptiveSteppingProcess._on_failure()` において:
- dt_min = dt_initial/32 まで縮小しても NR 不収束
- `can_retry=False` が返され、ソルバーが即 `converged=False` で終了

### NR 不収束の状況（frac≈0.890 付近）
```
Incr 238 (frac=0.8904), attempt 0,  ||R_u||/||f|| = 1.000e+00, active=36
Incr 238 (frac=0.8904), attempt 25, ||R_u||/||f|| = 1.006e+00, active=36
  → Adaptive dt retry
Incr 238 (frac=0.8903), attempt 0, converged (du converged)
Incr 239 (frac=0.8904), attempt 0,  ||R_u||/||f|| = 1.000e+00, active=36
  → 同じパターンで再び不収束
```

- 残差 ||R||/||f|| が 1.0 付近で全く減少しない
- 接触数 active=36 で固定（チャタリングではない）
- 残差爆発（208倍）も散発

### 根本原因の仮説

1. **接触剛性と外力のバランス崩壊**: frac≈0.89 で大変形域に入り、
   接触剛性行列の条件数が急悪化。接線剛性が正しくても NR が進めない。

2. **dt_min の限界**: dt_initial/32 では frac 0.890→0.891 の微小ステップでも
   残差が下がらない。dt をさらに縮めても解が変わらない（物理的限界点）。

3. **動的項の寄与**: 動的解析で質量行列・減衰行列があるにもかかわらず、
   frac≈0.89 で NR が発散する。接触力（132〜527N）に対して慣性項が不十分か。

## 次のステップ候補

1. **dt_min をさらに縮小**（dt_initial/128 等）: 148回のリトライが全て dt_min 到達
   なら効果あり。ただし計算時間が大幅増加。
2. **接触剛性の再評価**: frac≈0.89 での条件数を診断出力。
3. **ラインサーチの導入**: NR の更新量をスケーリングして残差爆発を抑制。
4. **載荷パターン変更**: 揺動でなく単調載荷で 30mm 到達可否を確認。

## ファイル構成

```
work/three_point_bend/
├── diagnosis.md          ← 本ファイル
├── logs/
│   ├── 01_E100_dynamic_kpen_statusfilt.log
│   ├── 02_E100_dynamic_kpen_statusfilt_v2.log
│   ├── 03_E100_beam_kpen_statusfilt.log
│   ├── 04_E25_beam_kpen_nofilt.log
│   ├── 05_E100_beam_kpen_nofilt.log
│   └── 06_E25_dynamic_kpen_revert.log
├── results/
└── scripts/
```
