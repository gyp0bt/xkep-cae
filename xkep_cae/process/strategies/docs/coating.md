# CoatingStrategy — 被膜接触モデル

[← README.md](../../../README.md)

## 概要

被膜接触モデル（Kelvin-Voigt弾性+粘性ダッシュポット）を Strategy として実装。
ContactManager.compute_coating_*() から移行（status-169）。

## Protocol

```python
class CoatingStrategy(Protocol):
    def forces(pairs, node_coords, config, dt) -> np.ndarray
    def stiffness(pairs, node_coords, config, ndof_total, dt) -> sp.csr_matrix
    def friction_forces(pairs, node_coords, config, u_cur, u_ref) -> np.ndarray
    def friction_stiffness(pairs, node_coords, config, ndof_total) -> sp.csr_matrix
```

## 実装

| クラス | 説明 |
|-------|------|
| `NoCoatingProcess` | 被膜なし（ゼロ返却） |
| `KelvinVoigtCoatingProcess` | Kelvin-Voigt弾性+粘性被膜モデル（status-137/140） |

## ファクトリ

```python
create_coating_strategy(coating_stiffness=0.0)
# coating_stiffness > 0 → KelvinVoigtCoatingProcess
# coating_stiffness == 0 → NoCoatingProcess
```
