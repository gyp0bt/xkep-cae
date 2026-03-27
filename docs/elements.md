# 要素モジュール

[← README](../README.md)

## 概要

梁要素・ソリッド要素・混合アセンブラを提供する。

## アセンブラ Process

### ULCRBeamAssemblerProcess

Updated Lagrangian + Corotational 定式化の梁アセンブラを生成する。
各収束ステップ後に参照配置を更新し、ヘリカル梁の大回転に対応。

- 入力: `ULCRBeamAssemblerInput` — 節点座標・要素接続・材料定数
- 出力: `ULCRBeamAssemblerOutput` — アセンブラインスタンス・節点数・DOF数

### Hex8AssemblerProcess

HEX8 ソリッド要素のアセンブラを生成する。
6 DOF/node レイアウト（梁要素との混合組立用）。回転 DOF はゼロ剛性。

- 入力: `Hex8AssemblerInput` — 節点座標・要素接続・E・ν・オフセット
- 出力: `Hex8AssemblerOutput` — アセンブラインスタンス・節点数・回転DOFリスト

### MixedAssemblerProcess

梁アセンブラと HEX8/剛体アセンブラを統合する混合アセンブラを生成する。

- 入力: `MixedAssemblerInput` — 梁アセンブラ・HEX8アセンブラ・全体DOF数
- 出力: `MixedAssemblerOutput` — 混合アセンブラインスタンス・全体DOF数

## 共通アセンブラインターフェース

全アセンブラは以下のメソッドを実装:

- `assemble_tangent(u)` → 接線剛性行列 (CSR)
- `assemble_internal_force(u)` → 内力ベクトル
- `update_reference(u_incr)` → UL参照配置更新
- `checkpoint()` / `rollback()` → 状態保存/復元
- `u_total_accum` → 累積変位
- `ndof` → 自由度数
