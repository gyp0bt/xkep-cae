# 拘束条件パッケージ

剛体結合、MPC（Multi-Point Constraint）などの拘束条件 Process を提供する。

## RigidEdgeAssemblerProcess

剛体ジグ辺用のダミーアセンブラを生成する PreProcess。
剛性・内力はゼロを返し、座標管理のみ行う。

- **入力**: `RigidEdgeAssemblerConfig`（ジグ座標、節点数、オフセット、全DOF数）
- **出力**: `RigidEdgeAssemblerResult`（アセンブラインスタンス）
- **status-249**: `_RigidEdgeAssembler` の Process 化（脱法修正）

[← README](../README.md)
