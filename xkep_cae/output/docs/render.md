# BeamRenderProcess

[<- README](../../../README.md)

## 概要

梁3Dレンダリングの PostProcess。変形後メッシュを2D投影で可視化する。

## 入出力

- **入力**: `RenderConfig` — ソルバー結果 + メッシュ + 出力先 + 描画パラメータ
- **出力**: `RenderResult` — 生成した画像ファイルパス一覧

## 移行元

`__xkep_cae_deprecated/process/concrete/post_render.py` → status-183
