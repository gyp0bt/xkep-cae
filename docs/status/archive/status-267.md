# status-267: チャタリング対策分析 + リラクゼーション diverged フラグ修正

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-29
- **ブランチ**: `claude/check-status-todos-kenbI`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3+9+4+18（変更なし）→ **合計592 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. E=25 カットバック詳細分析

ベースライン（frac=0.4837, 500incr, 293cutback）のログを徹底分析し、カットバックの根本原因を特定。

#### カットバック分類（ベースライン）

| 原因 | 件数 | frac 範囲 |
|------|------|-----------|
| chattering（リラクゼーション abort） | 91 | 0.4-0.5 に集中（90/91） |
| divergence（残差連続増加） | 66 | 0.0-0.1 に集中（66/66） |
| max_attempts（30反復未収束） | 136 | 0.0-0.1: 115, 0.3-0.5: 21 |
| **合計** | **293** | |

#### frac 帯域別カットバック

| frac 帯域 | cutback | 特徴 |
|-----------|---------|------|
| 0.0-0.1 | **181** | 接触確立フェーズ、発散+max_attempts |
| 0.1-0.4 | **0** | 完全安定（cutback なし） |
| 0.4-0.5 | **96** | チャタリング帯域、active set 振動 |

#### リラクゼーション有効性分析

**91回のリラクゼーション全てが abort（成功 0 回）**。

残差軌道は全イベントで同一パターン:
```
attempt 0: R_t/f = 0.853
attempt 5: R_t/f = 0.728
attempt 10: R_t/f = 0.622
attempt 15: R_t/f = 0.531
attempt 20: R_t/f = 0.453
→ 25 反復で未収束（abort）
```

**根本原因**: gap 振動による構造的残差。active set が毎反復変化し、リラクゼーション（f_c と f_c_prev のブレンド）では解消不能。残差は 0.45 に漸近するが、tol_force=1e-6 には到達しない。

### 2. リラクゼーション abort diverged フラグ修正

**問題**: リラクゼーション abort 時に `_diverged=True` を設定。adaptive stepping が `shrink²=0.25` （通常の 0.5 の2乗）で dt を縮小。91回の全失敗 × 4倍縮小 → チャタリング帯域で dt が枯渇。

**修正**: `_diverged=False` に変更。リラクゼーション abort は「停滞」であり「発散」ではない。通常の `shrink=0.5` で dt を縮小。

| 指標 | ベースライン | 修正後 | 変化 |
|------|-------------|--------|------|
| frac | 0.4837 | **0.4950** | **+2.3%** |
| chattering | 91 | **29** | **-69%** |
| divergence | 66 | 96 | +45% |
| max_attempts | 136 | 204 | +50% |
| total cutback | 293 | 329 | +12% |

**解釈**: dt が大きく維持されるため、チャタリング検知パターンに到達しにくくなった（91→29）。一方、より大きな dt で別の失敗モード（divergence, max_attempts）に遷移。全体 cutback は微増するが、dt が大きいため frac の進捗は改善。

### 3. 接線剛性スケーリング修正

**問題**: リラクゼーション時に `contact_tangent_scale=_current_omega`（0.5→0.05）で接線を縮小。残差の減衰（f_c_blend）と接線の縮小が不整合で、NR の探索方向が不正確。

**修正**: `contact_tangent_scale=1.0`（常時フルスケール）に変更。残差のみ減衰、接線は正しい Jacobian を維持。

**効果**: 単独では frac に変化なし（0.4837 → 0.4837）。リラクゼーション自体が無効（0/91成功）のため、接線精度が結果に反映されなかった。しかし今後リラクゼーション改善時に正しい基盤となる。

#### 修正ファイル

| ファイル | 変更内容 |
|----------|----------|
| `xkep_cae/contact/solver/_newton_dynamic.py` | リラクゼーション abort で `_diverged=False` + `contact_tangent_scale=1.0` |

---

## 試行して不採用だった変更

### dt 回復加速（adaptive stepping）
- カットバック直後の dt 成長を `grow_factor²` に加速
- **結果**: 逆効果。frac 0.0082→0.0059（50incr比較）。加速した dt が次のカットバック域に早く到達し、カットバック率が悪化（52%→74%）

### 常時接触力減衰
- NR の att>0 で常時 `f_c_blend = 0.8*f_c + 0.2*f_c_prev`
- **結果**: 致命的悪化。frac 0.0082→0.0016。減衰が初回 NR 更新の変位収束（||du||/||u|| < 1e-8）を阻害

### リラクゼーション max_iter 短縮（25→15）
- 無効なリラクゼーション反復の節約を意図
- **結果**: 50incr では影響なし（チャタリング未到達）。100incr+ で frac 0.0057 に悪化。別の adaptive stepping 変更との組合せが原因の可能性あり。単独では中立。

---

## テスト結果

- 新規テスト: なし
- 既存テスト: 592 passed, 20 skipped, 1 xfailed（回帰なし）
- 契約違反: 0件
- lint: 全合格

---

## 再現手順

```bash
git checkout claude/check-status-todos-kenbI
pip install -e .

# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"

# 契約検証
python contracts/validate_process_contracts.py

# E=25 ベンチマーク（~5min, frac≈0.4950）
python3 -c "
import warnings; warnings.filterwarnings('ignore')
from xkep_cae.numerical_tests.three_point_bend_jig import *
cfg = DynamicThreePointBendContactJigConfig(
    E=25.0, n_periods=30.0, jig_push=30.0,
    max_increments=500, use_rigid_surface=True,
    frozen_hermite_tangent=True,
)
r = DynamicThreePointBendContactJigProcess().process(cfg)
sr = r.solver_result
print(f'frac={sr.load_history[-1]:.4f} incr={sr.n_increments} cutback={sr.n_cutbacks}')
" 2>&1 | tee /tmp/log-benchmark-267.log
```

---

## 次セッションへの引き継ぎ

### 残課題（優先度順）

1. **E=25 frac=1.0 到達**（最優先）
   - 現状 frac=0.4950（+2.3%改善）、目標は 1.0
   - **チャタリング帯域（frac>0.4）の根本対策が必要**
   - リラクゼーション（f_c ブレンド）は構造的に無効（gap 振動による不可避残差）
   - 候補:
     a. **Frozen active set NR**: NR 反復中に active/inactive を凍結し、gap 振動を抑止
     b. **Huber delta_h 拡大**: gap 遷移幅を広げてチャタリング閾値を上げる（E=25 専用チューニング vs 汎用性のトレードオフ）
     c. **max_increments 拡大**: 2000+ で力技到達を試行（非効率だが frac=1.0 検証用）
     d. **接触確立フェーズ（frac<0.1）改善**: 181 cutback の削減でより大きな dt で中間帯域到達

2. **NR 力収束改善**
   - 現状: 力収束 0/500（全変位収束）
   - 接触活性集合変化により力残差が構造的に不連続

3. **Hermite 非局所 ∂g/∂u 対応**

### 設計メモ

1. **リラクゼーション diverged フラグ**: False が正しい。チャタリング = 停滞（残差振動）≠ 発散（残差増大）。dt の 4 倍縮小は過剰。
2. **接線スケーリング**: 常時 1.0 が正しい。リラクゼーションは残差のみの操作で、接線（Jacobian）は本来の状態を維持すべき。
3. **frac 帯域の二相構造**: 初期（frac<0.1）と終盤（frac>0.4）に問題集中、中間は安定。異なるメカニズムの対策が必要。

### 開発運用メモ

- 適応時間刻みの変更は慎重に。dt 成長加速は直感に反して悪化する場合がある（次のカットバック域に早く到達）。
- パラメータ変更は 50incr の短縮テスト → 500incr のフルテストの二段階で検証するのが効率的。

---
