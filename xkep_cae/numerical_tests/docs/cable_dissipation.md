# CableDissipationProcess — ケーブル曲げ散逸エネルギー計算

[← README](../../../README.md) | [← roadmap](../../../docs/roadmap.md)

## 概要

ファイバー梁モデルによるケーブルの曲げヒステリシス散逸エネルギーを計算する Process。
M-κ（モーメント-曲率）ヒステリシスループの面積から散逸エネルギーを定量評価する。

## 入力

| パラメータ | 型 | デフォルト | 説明 |
|-----------|---|---------|------|
| n_strands | int | 7 | 素線本数 |
| wire_radius | float | 0.5 | 素線半径 [mm] |
| pitch_length | float | 100.0 | ピッチ長 [mm] |
| bending_curvature | float | 0.001 | 曲げ曲率 κ_max [1/mm] |
| n_half_cycles | int | 2 | 半サイクル数（2=1ループ） |
| mu | float | 0.15 | 摩擦係数 |
| degrade_ratio | float | 0.25 | 接触剛性劣化比 β |

## 出力

| フィールド | 型 | 説明 |
|-----------|---|------|
| dissipation_energy | float | 散逸エネルギー [N·mm²] |
| mk_metrics | dict | M-κ ヒステリシス指標 |
| cable_info | dict | ケーブル幾何情報 |

## 散逸エネルギーの非線形依存性

- **撚線本数**: EI_max/EI_min 比の増大 → ループ幅増大 → 散逸増加
- **ピッチ**: 短ピッチでヘリックス角増大 → 素線間滑り増加 → 散逸増加
- **曲率**: 低κで弾性、高κで飽和のS字応答
- **劣化比**: β < 1 でティアドロップ形状の非対称性が出現

## status

- status-331 で新規作成（Phase F5 散逸エネルギー検証）
