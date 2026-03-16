# StrandMeshProcess

[<- README](../../../README.md)

## 概要

撚線メッシュ生成の PreProcess。TwistedWireMesh の機能を Process API として管理する。

## 入出力

- **入力**: `StrandMeshConfig` — 撚線パラメータ（本数、径、ピッチ、要素密度等）
- **出力**: `StrandMeshResult` — 節点座標、接続情報、半径、レイヤーID

## 依存

- `xkep_cae_deprecated.mesh.twisted_wire.make_twisted_wire_mesh`（importlib 経由）

## 移行元

`xkep_cae_deprecated/process/concrete/pre_mesh.py` → status-183
