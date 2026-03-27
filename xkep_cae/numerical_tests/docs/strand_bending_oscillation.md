# 7本撚線曲げ揺動解析

端部剛体結合（MPC DOF消去）+ 曲げ処方変位 + 揺動サイクルの
撚線曲げ揺動を実行する BatchProcess。

## StrandBendingOscillationProcess

- **カテゴリ**: BatchProcess
- **入力**: `StrandBendingOscillationConfig`
- **出力**: `StrandBendingOscillationResult`
- **uses**: StrandMeshProcess, MPCEliminationProcess, ULCRBeamAssemblerProcess, ContactFrictionProcess

### パイプライン

1. StrandMeshProcess でメッシュ生成
2. 端部参照点ノード追加（左端・右端の重心位置）
3. MPCEliminationProcess で端部剛体結合
4. ULCRBeamAssemblerProcess でアセンブラ構築
5. 曲げ処方変位を境界条件に設定
6. ContactFrictionProcess で求解

### MPC剛体結合

各端面の全素線端部節点を参照点に結合:
- 並進: `u_slave = u_master + [r]x theta_master`
- 回転: `theta_slave = theta_master`

### 境界条件

- 左端参照点: 全DOF固定（固定端）
- 右端参照点: xyz固定 + theta_z 処方（曲げ揺動）

### status-253

DOF消去MPC実装 + 7本撚線曲げ揺動Process。

[← README](../../../README.md)
