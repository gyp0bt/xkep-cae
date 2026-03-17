# ExportProcess

[<- README](../../../README.md)

## 概要

結果出力の PostProcess。ソルバー結果を CSV/JSON 形式でエクスポートする。

## 入出力

- **入力**: `ExportConfig` — ソルバー結果 + メッシュ + 出力先 + フォーマット
- **出力**: `ExportResult` — エクスポートしたファイルパス一覧

## 移行元

`__xkep_cae_deprecated/process/concrete/post_export.py` → status-183
