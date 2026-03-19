# BeamOscillationProcess — 接触なし梁揺動解析

[← README](../../../README.md)

## 概要

単純支持梁に初速度を与え、接触なしで自由振動させる動的解析 Process。
三点曲げの前準備として、非線形域の動的挙動を検証する。

## 物理モデル

- ワイヤ: x 軸方向直線梁（Timoshenko CR 3D）
- 支持: 左端=ピン（xyz+rx 固定）、右端=ローラー（yz 固定）
- 加振: 中央節点に初速度 v₀（y 方向下向き）

## 検証項目

1. 動的ソルバー（GeneralizedAlpha）の収束
2. 非線形域（大変形）での物理的妥当性
3. 数値粘性（高周波減衰率）の評価
4. 3D 応力コンター可視化

## 入出力

- 入力: `BeamOscillationConfig`
- 出力: `BeamOscillationResult`

## 解析解（小振幅線形）

- f₁ = π/(2L²) √(EI/ρA)
- δ(t) = (v₀/ω₁) sin(ω₁t)
