# status-247: NR 接触チャタリング対策 — 接触力リラクゼーションで n_periods=30 完走

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-27
**テスト**: 200+10s | 契約違反 1件（pre-existing C3） | 条例違反 0件
**ブランチ**: `claude/fix-three-point-bend-TPOGT`

## 概要

NR イテレーション内の接触チャタリング（active set 2-サイクル）を検知し、
接触力アンダーリラクゼーション + 接線剛性スケーリングで安定化。
n_periods=30 三点曲げ（E=200, freeze=F, Hermite=ON）が **frac=1.0000 完走**。

## 根本原因分析

frac≈0.9838 で NR 残差が ~2.0 に張り付き収束しない問題の原因:

1. **NR 2-サイクル**: active ペア数が 4↔5 で毎反復振動
2. **Line search の盲点**: 旧接触力 f_c_old で試行残差を評価するため、接触力変化による残差悪化を検知不能
3. **freeze_active_set の限界**: ペアのステータス（ACTIVE/INACTIVE）は凍結するが、力計算は gap ベースで毎反復再評価 → p_n が 0/非0 で振動
4. **Huber 遷移幅の狭さ**: smoothing_delta=5000/r により gap±0.0017mm が遷移域 → 遷移外のペアで離散的に力が変化

## 変更内容

### 1. NR ストール検知 (`_newton_dynamic.py`)

- 残差変化率 < 5% かつ active set が変化 → ストールカウント加算
- `stall_window`（デフォルト4）回連続でチャタリング検知 → リラクゼーション自動有効化

### 2. 接触力アンダーリラクゼーション (`_newton_dynamic.py`)

- `f_c_blend = ω * f_c_new + (1-ω) * f_c_prev`
- 収束時は f_c_new ≈ f_c_prev → 収束解は不変（固定点不変性）
- **漸進的 omega 減衰**: `ω = max(0.05, ω₀ * 0.7^(iter//2))`
  - iter 0-1: ω=0.50
  - iter 2-3: ω=0.35
  - iter 4-5: ω=0.245
  - iter 10+: ω→0.05（下限）

### 3. 接線剛性スケーリング (`_newton_steps.py`)

- リラクゼーション中: `K_T = K_struct + ω * K_contact`
- 残差の接触力寄与（ω倍）と接線剛性の整合性を保持
- Newton 方向が残差と一致 → 安定な収束

### 4. 新パラメータ (`core/data.py`, `_newton_dynamic.py`)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `contact_relax_omega` | 0.5 | 初期リラクゼーション係数 |
| `stall_window` | 4 | ストール検知に必要な連続回数 |

## ベンチマーク（STA2: tee ログ保存済み）

### n_periods=30 三点曲げ（E=200, freeze=F, K_st=OFF, Hermite=ON）

| 項目 | 変更前（status-246） | 変更後 | 差 |
|------|---------------------|--------|-----|
| frac | 0.9838（**不完走**） | **1.0000** | **完走達成** |
| 計算時間 | 396.9s | 598.7s | +51%（安定化のコスト） |
| increments | 650 | 840 | +190 |
| cutbacks | 374 | 488 | +114 |

**注**: 計算時間増加はリラクゼーション反復とカットバック増加による。
完走が最優先であり、高速化は Phase 2 以降で対処。

## 再現手順

```bash
# ベースライン（変更前: frac=0.9838 で停止）
git stash  # 変更前コードに戻す
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
" 2>&1 | tee /tmp/log-np30-baseline.log
git stash pop  # 変更を戻す

# 改善後（frac=1.0000）
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
" 2>&1 | tee /tmp/log-np30-relax.log
```

## 設計判断

1. **Process 外の NR ループ制御**: リラクゼーションは NR ループ制御の一部であり、`NewtonDynamicProcess` 内に実装。Contact Force Strategy の API は変更なし。
2. **固定点不変性**: ω < 1 のブレンドは f_c_new ≈ f_c_prev の固定点で f_c_blend = f_c_new。収束解は物理的に正しい。
3. **漸進的 omega**: 最初は穏やかに（ω=0.5）、収束しなければ積極的に（ω→0.05）。通常ステップではリラクゼーション無効（ω=1.0）で性能劣化なし。
4. **接線スケーリング**: `K_T = K_struct + ω*K_contact` は残差 `R = f_int + ω*f_c_new + (1-ω)*f_c_prev - f_ext` の ∂R/∂u と整合（f_c_prev は u 非依存）。

## TODO

- [ ] 計算時間削減: リラクゼーション有効時の NR 反復数削減（ストール早期脱出）
- [ ] デフォルト設定（E=200e3, freeze=True）での検証完了待ち
- [ ] NR 力収束速度改善（中盤後～終盤で 25 反復が力収束に不足）
- [ ] Hermite 非局所 ∂g/∂u 対応（4ノードペア外の DOF 結合）
- [ ] 1000本スケールアップ向け反復ソルバー導入
