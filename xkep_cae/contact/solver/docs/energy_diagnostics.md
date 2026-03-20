# エネルギー診断 Process

[← README](../../../../README.md)

## 概要

動的解析のステップごとのエネルギー収支を計算・記録する。

## エネルギー項

| 項目 | 式 | 説明 |
|------|------|------|
| 運動エネルギー (KE) | `0.5 * v^T M v` | 速度ベースの運動エネルギー |
| ひずみエネルギー (SE) | `0.5 * u^T f_int` | 内力仕事近似 |
| 外力仕事 (W_ext) | `f_ext · u` | 外力による仕事 |
| 接触仕事 (W_contact) | `f_c · u` | 接触力による仕事 |

## エネルギー保存性

- `energy_ratio = (KE + SE) / max(|W_ext| + |W_contact| + |KE + SE|, 1e-30)`
- 外力なし自由振動: `total / initial ≈ 1.0`（rho_inf < 1 で数値減衰あり）

## Process

- `StepEnergyDiagnosticsProcess`: 1ステップのエネルギー計算
- `EnergyHistory`: エネルギー履歴の蓄積器
- `EnergyHistoryEntry`: 1エントリのデータ

## 使用例

```python
from xkep_cae.contact.solver import StepEnergyDiagnosticsProcess, StepEnergyInput

proc = StepEnergyDiagnosticsProcess()
result = proc.process(StepEnergyInput(
    u=u, velocity=v, mass_matrix=M,
    f_int=f_int, f_ext=f_ext, f_c=f_c,
    dt=dt, step=step,
))
print(f"KE={result.kinetic_energy:.6e}, SE={result.strain_energy:.6e}")
```
