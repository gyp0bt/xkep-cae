# ContactForce Strategy

[← README](../../../../README.md)

## 概要

接触力の評価方法を Strategy として実装する。`ContactForceStrategy` Protocol に従い、
NCP 法と Smooth Penalty 法を統一的に扱う。

## 具象 Process

| クラス | 概要 |
|--------|------|
| `NCPContactForceProcess` | Alart-Curnier NCP + 鞍点系 |
| `SmoothPenaltyContactForceProcess` | softplus + Uzawa 外部ループ |

## Protocol メソッド

- `evaluate(u, lambdas, manager, k_pen)`: 接触力と NCP 残差を評価
- `tangent(u, lambdas, manager, k_pen)`: 接触接線剛性行列

## ファクトリ

```python
create_contact_force_strategy(
    contact_mode="ncp",        # "ncp" | "smooth_penalty"
    ndof=0,
    ndof_per_node=6,
    contact_compliance=0.0,    # NCP δ 正則化
    smoothing_delta=0.0,       # smooth penalty δ
)
```

## 推奨構成

- 摩擦なし: `contact_mode="ncp"`
- 摩擦あり: `contact_mode="smooth_penalty"`（status-147 推奨）

## 移行元

`xkep_cae_deprecated/process/strategies/contact_force.py` → status-179
