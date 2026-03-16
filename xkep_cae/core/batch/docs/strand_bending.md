# StrandBendingBatchProcess

[<- README](../../../../README.md)

## 概要

撚線曲げ揺動ワークフロー全体をオーケストレーションする BatchProcess。
process-architecture.md §6 に準拠。

## 実行ツリー

```
StrandMeshProcess → ContactSetupProcess
  → [ExportProcess] → [BeamRenderProcess] → [ConvergenceVerifyProcess]
```

## Phase 3 実装状態

concrete プロセス移行完了:
- StrandMeshProcess（PreProcess）: xkep_cae/mesh/process.py
- ContactSetupProcess（PreProcess）: xkep_cae/contact/setup/process.py
- ExportProcess（PostProcess）: xkep_cae/output/export.py
- BeamRenderProcess（PostProcess）: xkep_cae/output/render.py
- ConvergenceVerifyProcess（VerifyProcess）: xkep_cae/verify/convergence.py
- EnergyBalanceVerifyProcess（VerifyProcess）: xkep_cae/verify/energy.py
- ContactVerifyProcess（VerifyProcess）: xkep_cae/verify/contact.py

## 移行元

`xkep_cae_deprecated/process/batch/strand_bending.py` → status-183
