# status-282: 接触あり90度曲げベースライン — frac=0.40停滞（チャタリング）

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-03
- **ブランチ**: `claude/contact-baseline-check-LbO6E`
- **テスト数**: 606 passed（status-281から変更なし）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-281で達成した「接触なし90度曲げ frac=1.0」を踏まえ、**接触あり**（`contact_enabled=True`）でベースラインを確認した。

**結果**: `frac=0.4014` で停滞。active=8-9の接触ペアで**2サイクルチャタリング**（残差振動）が発生し、カットバックを繰り返しても解消しない。

---

## ベンチマーク結果

### 7本ヘリカル撚線90度曲げ（κ=π/200, θ=π/2）

| 構成 | frac | incr | cutback | active | 停滞原因 |
|------|------|------|---------|--------|----------|
| 接触なし（status-281） | **1.000** | 102 | 6 | 0 | — |
| **接触あり** | **0.4014** | 234 | 15 | 8-9 | チャタリング |

### 小曲率（κ=0.001, θ=5.7°）

| 構成 | frac | incr | cutback | active | 備考 |
|------|------|------|---------|--------|------|
| 接触あり | **1.000** | 52 | 4 | 0 | 接触ほぼ不活性で完走 |

---

## 停滞分析

### frac推移とactive contact遷移

| frac 範囲 | active | 収束状況 |
|-----------|--------|----------|
| 0.00-0.22 | 0 | 正常収束（接触不活性）|
| 0.22-0.30 | 1-5 | 収束するが dt 縮小あり |
| 0.30-0.40 | 2-7 | 収束は維持、dt は小刻み |
| **0.40** | **8-9** | **チャタリング停滞** |

### チャタリングパターン（frac≈0.40 での典型）

```
attempt 25: ||R_t||/||f|| = 3.548e-04, active=8
attempt 30: ||R_t||/||f|| = 8.098e-04, active=8  ← 振動（2x増加）
attempt 35: ||R_t||/||f|| = 3.548e-04, active=8  ← 振動（戻る）
...45反復まで同パターン
```

- 残差は `3.5e-4 ↔ 8.1e-4` の **2サイクル振動**
- 回転残差（||R_r||）は `6.9e-6` で実質収束済み
- **並進残差のみが振動** — 接触力の活性集合変化が原因
- カットバックでdt縮小しても同じパターンが再発

### 根本原因

接触ペアが8-9個活性化すると、NR反復内で接触力のON/OFF切り替え（活性集合変化）が2サイクルで振動する。Huber平滑化で緩和されているが、frac≈0.40ではペナルティ力の変動が残差を収束させない。

---

## 物理的考察

### なぜfrac=0.40で停滞するか

1. **曲げ変形の進行**：frac=0.40はθ≈36°の曲げ角。ヘリカル素線間の相対滑りが増大
2. **接触ペアの急増**：frac=0.22で接触開始 → frac=0.40で8-9ペアに増加
3. **チャタリング閾値**：active≈8で残差振動が収束判定を超える

### 改善の方向性

1. **huber_delta_hの調整**: 平滑化遷移幅を広げてチャタリング振幅を抑制
2. **接触力リラクゼーション強化**: omega を下げて接触力更新を緩和
3. **チャタリング検知→収束判定緩和**: 2サイクル振動時に残差上限で判定

---

## 再現手順

```bash
git checkout claude/contact-baseline-check-LbO6E
pip install -e .

# 接触あり90度曲げベースライン（~5分）
python -c "
from xkep_cae.numerical_tests.strand_bending_oscillation import *
import math
cfg = StrandBendingOscillationConfig(
    n_strands=7, wire_radius=0.5, pitch_length=100.0,
    n_elements_per_pitch=16, n_pitches=1.0,
    E=130.0e3, nu=0.3, rho=8.96e-9,
    bending_curvature=math.pi/200.0, n_cycles=1,
    n_increments_per_cycle=40, rho_inf=0.9, mu=0.15,
    max_nr_attempts=50, tol_force=1e-8, max_increments=10000,
    exclude_same_strand=True,
    free_end_mode=True, contact_enabled=True,
    loading_mode='rotation',
)
result = StrandBendingOscillationProcess().process(cfg)
sr = result.solver_result
frac = sr.load_history[-1] if sr.load_history else 0.0
print(f'frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}')
" 2>&1 | tee /tmp/log-7strand-contact-90deg.log
# 期待値: frac≈0.40, incr≈234, cutback≈15

# 接触なし90度曲げ（参考、~2分）
python -c "
from xkep_cae.numerical_tests.strand_bending_oscillation import *
import math
cfg = StrandBendingOscillationConfig(
    n_strands=7, wire_radius=0.5, pitch_length=100.0,
    n_elements_per_pitch=16, n_pitches=1.0,
    E=130.0e3, nu=0.3, rho=8.96e-9,
    bending_curvature=math.pi/200.0, n_cycles=1,
    n_increments_per_cycle=40, rho_inf=0.9, mu=0.15,
    max_nr_attempts=50, tol_force=1e-8, max_increments=10000,
    exclude_same_strand=True,
    free_end_mode=True, contact_enabled=False,
    loading_mode='rotation',
)
result = StrandBendingOscillationProcess().process(cfg)
sr = result.solver_result
frac = sr.load_history[-1] if sr.load_history else 0.0
print(f'frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}')
" 2>&1 | tee /tmp/log-7strand-nocontact-90deg.log
# 期待値: frac=1.0, incr≈102, cutback≈6

# 契約検証
python contracts/validate_process_contracts.py
```

---

## STA2 準拠チェック

- [x] **tee ログ保存**: `/tmp/log-7strand-contact-baseline-small-*.log`, `/tmp/log-7strand-contact-90deg-*.log`
- [x] **再現手順記載**: 全コマンドをstatusに記載
- [x] **数値の捏造なし**: ベースラインfrac=0.4014を正直に報告
- [x] **ベースライン先行取得**: 接触なし（status-281: frac=1.0）→ 接触あり（frac=0.40）

---

## TODO

- [x] 接触あり90度曲げベースライン取得（frac=0.4014）
- [x] 接触あり小曲率ベースライン取得（frac=1.0）
- [ ] チャタリング対策の検討（huber_delta_h調整 or リラクゼーション強化）
- [ ] 改善後のfrac進行率計測

---
