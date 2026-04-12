# status-248: NR リラクゼーション早期脱出 + omega 回復試行とリバート

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-27
**テスト**: 200+10s | 契約違反 1件（pre-existing C3） | 条例違反 0件
**ブランチ**: `claude/execute-status-todos-PKrMH`

## 概要

status-247 の TODO に基づき、NR リラクゼーションの計算時間削減を実施。

1. **relax_max_iter パラメータ追加**: リラクゼーション有効後 25 反復で未収束の場合、
   NR ループを早期打ち切り → カットバック。無駄な反復を 6 回分削減。
2. **omega 回復スケジュール**: チャタリング解消後に omega を 1.0 に戻す機能を実装
   → **逆効果（frac=0.9846 で停止）**を確認しリバート。
   元の omega 減衰のみのロジック（status-247）が最適と判明。

## 変更内容

### 1. relax_max_iter パラメータ追加 (`_newton_dynamic.py`)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `relax_max_iter` | 25 | リラクゼーション有効化後の最大反復数 |

リラクゼーション有効化後、`relax_max_iter` 回以内に収束しなければ
`_diverged = True` で NR ループを打ち切り。adaptive stepping がカットバックを実施。

### 2. omega 回復スケジュール（実装→リバート）

**試行内容**: チャタリング解消（active set 安定化）を検知し、
omega を `relax_recovery_rate` 倍で段階的に 1.0 まで回復。

**結果**: frac=0.9846 で停止（status-247 の frac=1.0000 より悪化）。
原因: omega 回復により接触力ブレンドが弱まり、チャタリングが再発。
元の持続的リラクゼーション（omega 減衰のみ）が壁突破に必要。

**判断**: omega 回復は完全リバート。早期脱出のみ採用。

## ベンチマーク結果（STA2: tee ログ保存済み）

### n_periods=30 三点曲げ（E=200, freeze=F, Hermite=ON）

| 項目 | status-247 | omega回復版 | 早期脱出のみ |
|------|-----------|------------|------------|
| frac | **1.0000** | 0.9846 ❌ | **1.0000** ✅ |
| 計算時間 | 598.7s | 391.3s | **580.7s** |
| increments | 840 | 662 | 868 |
| cutbacks | 488 | 384 | 515 |
| 早期打切り | 0 | 0 | **6** |

**早期脱出により 6 回の無駄な NR 反復（各最大 25 反復 = 最大 150 反復）を回避。**
計算時間 3.0% 短縮（598.7s → 580.7s）で完走維持。

## 再現手順

```bash
# status-248: relax_max_iter=25 での早期脱出
python -c "
import time, warnings
warnings.filterwarnings('ignore')
from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
)
cfg = DynamicThreePointBendContactJigConfig(
    E=200.0, jig_push=30.0, n_periods=30.0, max_increments=10000,
    use_hermite_centerline=True, freeze_geometry_in_nr=False,
)
t0 = time.perf_counter()
result = DynamicThreePointBendContactJigProcess().process(cfg)
sr = result.solver_result
frac = sr.load_history[-1] if sr.load_history else 0.0
print(f'elapsed={time.perf_counter()-t0:.1f}s frac={frac:.4f} incr={sr.n_increments} cutbacks={sr.n_cutbacks}')
" 2>&1 | tee /tmp/log-np30-e200-status248.log
```

## 設計判断

1. **omega 回復は逆効果**: 直感的には「安定化後に元に戻す」のが正しいが、
   実際にはリラクゼーションが壁突破に必要。持続的な低 omega が接触力振動を抑制する。
2. **relax_max_iter=25**: max_attempts=50 のうち、チャタリング検知は概ね att=10-20 で
   発生するため、残り 25 反復は十分。25 回で未収束なら dt を縮小して再試行する方が効率的。
3. **早期脱出は安全**: `_diverged = True` で打ち切るため、adaptive stepping が
   発散時と同様にカットバック（shrink²）を適用。安全にリカバリーする。

## TODO

- [ ] デフォルト設定（E=200e3, freeze=True）での検証
- [ ] NR 力収束速度改善（中盤後～終盤で 25 反復が力収束に不足）
- [ ] Hermite 非局所 ∂g/∂u 対応（4ノードペア外の DOF 結合）
- [ ] 1000本スケールアップ向け反復ソルバー導入

## 開発運用メモ

- **omega 回復の試行 → リバートは有益な知見**: 「持続的リラクゼーションが壁突破に必須」
  という知見が得られた。今後の改善方針に影響。
- tee ログ 3 種を保存: baseline(`/tmp/log-np30-e200-status248.log`),
  omega回復版(`/tmp/log-np30-e200-status248-v1.log`弃),
  早期脱出のみ(`/tmp/log-np30-e200-status248-v2.log`)
