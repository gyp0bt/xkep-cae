# 数値試験フレームワーク

[← README](../README.md)

## 概要

梁要素の基本性能を検証する静的・動的数値試験フレームワーク。

## Process

### StaticBeamTestProcess

静的梁試験（3点曲げ・4点曲げ・引張・ねん回）を実行する。

- 入力: `NumericalTestConfig` — 試験種別・材料・形状・荷重
- 出力: `StaticTestResult` — 変位・解析解・相対誤差

### DynamicBeamTestProcess

動的梁試験（過渡応答解析）を実行する。

- 入力: `DynamicTestConfig` — 試験種別・材料・時間積分パラメータ
- 出力: `DynamicTestResult` — 時刻歴応答・解析解比較
