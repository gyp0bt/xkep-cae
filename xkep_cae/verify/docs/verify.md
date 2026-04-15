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

### ContactKcComponentFDDiagnosticProcess

接触剛性 `K_c = K_mat - K_geo + K_st` を部分行列レベルで
FD（有限差分）と突合する診断 Process（status-343 で新設）。

- 入力: `u`, `du`, `compute_contact_force`, `K_mat`, `K_geo`, `K_st`, `eps`
- 出力: `full` / `mat_only` / `mat_geo` / `mat_st` の 4 組み合わせ FD 相対誤差
  + 各組の x/y/z/θx/θy/θz 成分別不整合シェア + 寄与率（share_mat/geo/st）

status-342 で観測された 19本撚線の f_c FD 相対誤差 115% / x 成分 68% 不整合が、
K_mat / K_geo / K_st のどの部分行列に由来するか切り分ける目的で設計。
既存の `TangentFDDiagnosticProcess`（単一 rel_err）を補完する。

## 移行元

- `__xkep_cae_deprecated/process/verify/convergence.py`
- `__xkep_cae_deprecated/process/verify/energy.py`
- `__xkep_cae_deprecated/process/verify/contact.py`

→ status-183
