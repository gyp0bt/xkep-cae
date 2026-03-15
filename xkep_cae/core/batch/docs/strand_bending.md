# StrandBendingBatchProcess

[← README](../../../../README.md)

## 概要

撚線曲げ揺動ワークフロー全体をオーケストレーションする BatchProcess。
process-architecture.md §6 に準拠。

## 実行ツリー（目標）

```
StrandMeshProcess → ContactSetupProcess → ContactFrictionProcess
  → [ExportProcess] → [RenderProcess] → [VerifyProcess]
```

## Phase 2 時点の状態

Strategy プロセスのみ移行済み。uses 宣言で依存関係を表明。
concrete プロセス（Mesh/Setup/Export/Render/Verify）は Phase 3 で追加予定。

## 移行元

`xkep_cae_deprecated/process/batch/strand_bending.py` → status-179
