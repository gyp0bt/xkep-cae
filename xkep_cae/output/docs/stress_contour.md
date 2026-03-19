# StressContour3DProcess — 梁 3D 応力コンターレンダリング

[← README](../../../README.md)

## 概要

変形後の梁をチューブ状に 3D レンダリングし、
要素ごとの最大曲げ応力をカラーマッピングで可視化する PostProcess。

## 出力

- 3D ビュー PNG（正面・斜視）
- 時刻歴プロット（変位 + 応力）
- 連番 PNG（アニメーション用）

## 入出力

- 入力: `StressContour3DConfig`
- 出力: `StressContour3DResult`
