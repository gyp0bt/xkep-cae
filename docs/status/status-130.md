# status-130: Updated Lagrangian (UL) CR梁アセンブラ実装 — ヘリカル梁の大回転収束達成

[← README](../../README.md) | [← status-index](status-index.md) | [← status-129](status-129.md)

**日付**: 2026-03-07
**テスト数**: 2271（変更なし: fast 1691 + 1 xfailed / slow 362 - 9 xfailed + deprecated 218）

## 概要

status-129 で診断されたCR梁ヘリカル要素の ~13° 収束障壁を **Updated Lagrangian (UL) アプローチ**で解消。
7本撚線で **45° および 90° 曲げが安定収束** することを確認した。

## 実装内容

### 1. ULCRBeamAssembler（beam_timo3d.py）

ヘリカル梁の大回転問題を解決するためのUpdated Lagrangian CR梁アセンブラ。

**設計思想**:
- 各収束ステップ後に参照配置（座標 + 回転）を更新
- CR要素の1ステップあたり回転を小さく保つ（~3°/step）
- 数値微分接線を使用（解析的接線のB行列にO(θ)誤差あり → status-129）

**主要メソッド**:
| メソッド | 内容 |
|---------|------|
| `assemble_tangent(u_incr)` | 更新済み参照座標で接線剛性を組み立て |
| `assemble_internal_force(u_incr)` | 更新済み参照座標で内力を組み立て |
| `update_reference(u_incr)` | 座標にΔu加算、回転にΔR乗算（Rodrigues） |

**回転更新の仕組み**:
```
R_ref_new[i] = R_incr(θ_x, θ_y, θ_z) @ R_ref_old[i]
```
ただし、CR要素は `coords_ref` の弦方向から自動的に R_0 を計算するため、
`_to_total_u` は単に `u_incr.copy()` を返すだけで十分（回転の二重カウントを回避）。

### 2. 曲げ揺動ベンチマーク UL統合（wire_bending_benchmark.py）

`run_bending_oscillation` に `use_updated_lagrangian: bool = False` パラメータを追加。

**Phase 1（曲げ）のUL実装**:
- 処方変位を `n_bending_steps` 個の増分に分割
- 各増分で NCP ソルバーを1ステップ実行
- 収束後に `ul_asm.update_reference(u_incr)` で参照更新
- 累積変位を追跡（`u_total_accum += u_incr`）

**Phase 2（揺動）のUL対応**:
- UL使用時は `ul_asm.coords_ref` を参照座標として使用
- 揺動変位は更新済み参照からの増分として適用
- 注: Phase 2のUL統合は単一ステップ揺動では動作確認済みだが、
  フルサイクル揺動では特異行列が発生するため xfail 設定

### 3. テスト更新（test_ncp_bending_oscillation.py）

| テスト | 変更 | 状態 |
|--------|------|------|
| `test_ncp_7strand_bending_45deg` | xfail除去、UL有効化 | ✅ PASS |
| `test_ncp_7strand_bending_90deg` | xfail除去、UL有効化、30 steps | ✅ PASS |
| `test_ncp_7strand_bending_oscillation_full` | xfail理由変更、UL有効化 | ⚠ xfail（Phase2未完了） |
| `test_ncp_19strand_bending_45deg` | UL有効化 | ✅ PASS（収束非必須） |
| `test_ncp_19strand_bending_oscillation` | xfail追加、UL有効化 | ⚠ xfail（Phase2未完了） |
| `test_tip_displacement_direction` | xfail除去、UL有効化 | ✅ PASS |
| `test_penetration_ratio_within_limit` | xfail理由変更、UL有効化 | ⚠ xfail（Phase2未完了） |

## 収束結果

### 7本撚線 45° 曲げ（UL, 15 steps）

```
UL Step 1/15 (θ=3.0°): iter=5, active=444
UL Step 6/15 (θ=18.0°): iter=5, active=444
UL Step 12/15 (θ=36.0°): iter=5, active=444
UL Step 15/15 (θ=45.0°): iter=5, active=446
Phase 1 完了: converged=True, NR=75
先端変位: dx=0.000 mm, dy=-7.459 mm, dz=-1.992 mm
最大貫入比: 0.016494
計算時間: 44.5s
```

### 7本撚線 90° 曲げ（UL, 30 steps）

```
UL Step 1/30 (θ=3.0°): iter=5, active=444
UL Step 15/30 (θ=45.0°): iter=5, active=444
UL Step 30/30 (θ=90.0°): iter=5, active=436
Phase 1 完了: converged=True, NR=150
先端変位: dx=0.000 mm, dy=-12.738 mm, dz=-7.262 mm
最大貫入比: 0.022906
計算時間: 90.6s
```

### 物理的妥当性

- **45° 曲げ先端変位**: dy=-7.46mm, dz=-1.99mm
  - 解析解（R=L/θ≈25.5mm → dy≈7.5mm, dz≈-2.0mm）と良好な一致
- **90° 曲げ先端変位**: dy=-12.74mm, dz=-7.26mm
  - 解析解（R=L/θ≈12.7mm → dy≈12.7mm, dz≈-7.3mm）と良好な一致
- **貫入比**: 0.016（45°）、0.023（90°）— いずれも2%台で物理的に妥当

## TL vs UL 比較

| 指標 | TL（従来） | UL（今回） |
|------|-----------|-----------|
| 最大収束角度 | ~13°（ヘリカル梁の限界） | 90°以上 |
| 1ステップあたり反復数 | 不収束 | 5反復（安定） |
| 接線剛性 | 解析的 or 数値微分 | 数値微分のみ |
| 参照配置 | 固定（初期） | ステップごとに更新 |

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status | 備考 |
|--------|--------|-----------|------|
| TL + NCP Phase1 | UL + NCP Phase1 | status-130 | ULがデフォルトではない（opt-in） |

## TODO

- [ ] UL Phase 2（揺動）の特異行列問題調査・修正
  - 原因仮説: UL更新後の参照配置での内力平衡状態が不整合
  - 特異行列 → NaN → GMRES無限ループのパターン
- [ ] 19本撚線の90° UL曲げ収束テスト追加
- [ ] 解析的接線剛性のB行列修正（Rodrigues drill spin成分の反映）
- [ ] UL Phase 1 完了後のGIF出力対応（総変位 vs 増分変位の座標系統一）
