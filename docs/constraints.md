# 拘束条件パッケージ

剛体結合、MPC（Multi-Point Constraint）などの拘束条件 Process を提供する。

## MPCEliminationProcess

DOF消去（静的凝縮）による剛体結合MPC。
梁端部のslave節点群をmaster参照点に結合する変換行列 T を構築する。

**制約式（6DOF/node）**:
- 並進: `u_slave = u_master + [r]× θ_master`
- 回転: `θ_slave = θ_master`

**変換**: `K_red = T^T K T`, `f_red = T^T f`, `du_full = T @ du_red`

- **入力**: `MPCEliminationConfig`（MPCグループ、全体DOF数）
- **出力**: `MPCEliminationResult`（変換行列T、slave/master/独立DOF配列）
- **status-253**: DOF消去MPC実装（7本撚線端部剛体結合）

## RigidEdgeAssemblerProcess

剛体ジグ辺用のダミーアセンブラを生成する PreProcess。
剛性・内力はゼロを返し、座標管理のみ行う。

- **入力**: `RigidEdgeAssemblerConfig`（ジグ座標、節点数、オフセット、全DOF数）
- **出力**: `RigidEdgeAssemblerResult`（アセンブラインスタンス）
- **status-249**: `_RigidEdgeAssembler` の Process 化（脱法修正）

[← README](../README.md)
