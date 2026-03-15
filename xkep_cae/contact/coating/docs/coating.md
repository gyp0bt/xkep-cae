# Coating Strategy

[← README](../../../../README.md)

## 概要

被膜接触モデルの Strategy 実装。Kelvin-Voigt 弾性+粘性ダッシュポットモデルにより、
撚線の被膜（コーティング）層を通じた接触力と摩擦力を計算する。

## Process 一覧

| Process | 説明 |
|---------|------|
| `NoCoatingProcess` | 被膜なし（全メソッドゼロ返却） |
| `KelvinVoigtCoatingProcess` | Kelvin-Voigt 被膜モデル（弾性+粘性） |

## ファクトリ関数

`_create_coating_strategy(coating_stiffness=0.0)`:
- `coating_stiffness <= 0` → `NoCoatingProcess`
- `coating_stiffness > 0` → `KelvinVoigtCoatingProcess`

## Kelvin-Voigt モデル

被膜圧縮量 δ = max(0, t_coat_total - gap_core) に対し:

```
f_coat = k × δ + c × δ̇
```

- k: 被膜弾性剛性 (`coating_stiffness`)
- c: 被膜粘性ダッシュポット係数 (`coating_damping`)
- δ̇ ≈ (δ - δ_prev) / dt （後退差分近似）

## 被膜摩擦

被膜法線力 p_n = k × δ に対して Coulomb return mapping を適用。
`return_mapping_core`（`contact/friction/law_friction.py`）を使用。

## 移行履歴

- status-169: `ContactManager.compute_coating_*()` → `CoatingStrategy`
- status-181: `xkep_cae_deprecated/process/strategies/coating.py` → `xkep_cae/contact/coating/strategy.py`
