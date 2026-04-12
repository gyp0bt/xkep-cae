# status-224: 三点曲げワークスペース整備 + 不収束原因確定

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-22
**ブランチ**: `claude/verify-dynamic-bending-load-cmPwf`

---

## 概要

動的接触三点曲げの全6実行ログを `work/three_point_bend/` に集約。
30mm 押し切り不能の根本原因を `AdaptiveSteppingProcess._on_failure()` の
dt_min 到達判定と特定。

---

## 実行結果一覧

| # | E [MPa] | k_pen 方式 | status filter | push到達 [mm] | 接触力 [N] | 時間 [s] |
|---|---------|-----------|--------------|--------------|-----------|----------|
| 1 | 100 | 動的(18.3) | あり | 26.7 | 527 | 926 |
| 2 | 100 | 動的(18.3) | あり | 26.7 | 527 | 1281 |
| 3 | 100 | 梁(19.7) | あり | 26.5 | 526 | 984 |
| 4 | 25 | 梁(4.92) | なし | 3.4 | 14 | 1575 |
| 5 | 100 | 梁(16k) | なし | 3.4 | 56 | 1632 |
| 6 | 25 | 動的(4.56) | あり | 26.7 | 132 | 1313 |

---

## 不収束メカニズム

### 直接原因

`_adaptive_stepping.py` L170-178:

```python
def _on_failure(self, input_data):
    delta = input_data.load_frac - input_data.load_frac_prev
    if delta <= self._config.dt_min_fraction + 1e-15:
        return AdaptiveStepOutput(can_retry=False)  # 即終了
```

dt_min = dt_initial/32 まで縮小しても NR 不収束 → `can_retry=False` → ソルバー終了。

### NR 停滞の詳細

frac≈0.890 付近で:
- 残差 ||R||/||f|| が 1.0→1.006 で **全く減少しない**（30反復後も同じ）
- 接触数 active=36 で安定（チャタリングではない）
- 残差爆発（208倍）が散発 → dt 縮小 → 微小前進 → 再び停滞

### 根本原因

大変形域（push/L > 0.27）で接線剛性行列の精度が不足。
status-223 で追加した幾何剛性は法線回転のみで、
摩擦の接線方向幾何項（dn/du, dt1/du, dt2/du の連鎖微分）が不足している可能性。

---

## ワークスペース構成

```
work/three_point_bend/
├── diagnosis.md          — 詳細診断
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

---

## 次のステップ

1. dt_min 縮小（dt_initial/128）で押し切り可否を確認
2. frac=0.89 地点の条件数・スペクトル診断
3. 摩擦接線幾何項の追加
4. ラインサーチの導入検討

---

## テスト

**499 テスト** — 変更なし
