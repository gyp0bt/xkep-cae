# 梁揺動解析 3D応力コンター — 画像一覧

[← README](../../README.md) | [← status-213](../../docs/status/status-213.md)

## 解析条件

| 項目 | 値 |
|------|-----|
| ワイヤ長 | 100 mm |
| ワイヤ径 | 2 mm |
| 要素数 | 100（メッシュサイズ ≈ 半径 1mm） |
| ヤング率 | 200 GPa |
| 密度 | 7.85e-9 ton/mm³ |
| 振幅（初速度等価） | 5 mm |
| 計算周期数 | 3 |
| rho_inf | 0.9 |
| 収束インクリメント | 303 |
| 最大変位 | 1.128 mm |
| 最大曲げ応力 | 304.7 MPa |
| 固有振動数 | 396.4 Hz |
| 固有周期 | 2.52 ms |

## 3D応力コンター（時系列）

| フレーム | 時刻 | 画像 |
|---------|------|------|
| 0 | 0.000 ms | [beam_osc_stress3d_000.png](beam_osc_stress3d_000.png) |
| 1 | 1.077 ms | [beam_osc_stress3d_001.png](beam_osc_stress3d_001.png) |
| 2 | 2.155 ms | [beam_osc_stress3d_002.png](beam_osc_stress3d_002.png) |
| 3 | 3.232 ms | [beam_osc_stress3d_003.png](beam_osc_stress3d_003.png) |
| 4 | 4.310 ms | [beam_osc_stress3d_004.png](beam_osc_stress3d_004.png) |
| 5 | 5.387 ms | [beam_osc_stress3d_005.png](beam_osc_stress3d_005.png) |
| 6 | 6.465 ms | [beam_osc_stress3d_006.png](beam_osc_stress3d_006.png) |
| 7 | 7.567 ms | [beam_osc_stress3d_007.png](beam_osc_stress3d_007.png) |

## 時刻歴プロット

| 内容 | 画像 |
|------|------|
| 中央変位 + 最大応力 | [beam_osc_time_history.png](beam_osc_time_history.png) |

## 考察

- 応力分布は中央部（高応力=赤）> 端部（低応力=青）で三点曲げの応力分布として妥当
- **梁が揺動せず単調に変位する問題あり** — UL+GeneralizedAlpha結合の既知バグ（status-213）
- 応力は時間とともに単調増加（304.7 MPa max）— 揺動すべきだが減衰+漸増挙動
- アスペクト比は各軸実寸比例で描画（`set_box_aspect([rx, ry, rz])`）
