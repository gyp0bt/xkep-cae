# TimeIntegration Strategy

[← README](../../../../README.md)

## 概要

時間積分方法を Strategy として実装する。`TimeIntegrationStrategy` Protocol に従い、
準静的解析と動的解析（Generalized-α法）を統一的に扱う。

## 具象 Process

| クラス | 概要 |
|--------|------|
| `QuasiStaticProcess` | 準静的（荷重制御）— K_eff=K, R_eff=R |
| `GeneralizedAlphaProcess` | Generalized-α 動的解析（Chung & Hulbert 1993） |

## Protocol メソッド

- `predict(u, dt)`: 予測子 — Newmark β 法
- `correct(u, du, dt)`: 補正子 — Newton 反復更新
- `effective_stiffness(K, dt)`: 有効剛性行列
- `effective_residual(R, dt)`: 有効残差

## ファクトリ

```python
create_time_integration_strategy(
    mass_matrix=None,      # None → 準静的
    damping_matrix=None,
    dt_physical=0.0,       # >0 + mass_matrix → 動的
    rho_inf=0.9,           # スペクトル半径
)
```

## Generalized-α パラメータ

| rho_inf | 特性 |
|---------|------|
| 0.0 | 最大数値減衰（高周波完全減衰） |
| 0.9 | 推奨（適度な高周波減衰） |
| 1.0 | Newmark 平均加速度法（エネルギー保存） |

## 移行元

`xkep_cae_deprecated/process/strategies/time_integration.py` → status-179
