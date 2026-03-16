# VerifyProcess 群

[<- README](../../../README.md)

## 概要

ソルバー結果の検証を行う VerifyProcess 群。

### ConvergenceVerifyProcess

NR反復の収束履歴を検証（収束判定・反復数閾値・インクリメント妥当性）。

### EnergyBalanceVerifyProcess

エネルギー収支を検証（変位有限性・外力仕事・エネルギーバランス）。

### ContactVerifyProcess

接触状態を検証（最大貫入量・チャタリング比率）。

## 移行元

- `xkep_cae_deprecated/process/verify/convergence.py`
- `xkep_cae_deprecated/process/verify/energy.py`
- `xkep_cae_deprecated/process/verify/contact.py`

→ status-183
