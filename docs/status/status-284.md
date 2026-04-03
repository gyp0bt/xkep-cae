# status-284: チャタリング検知→接触凍結モード — frac 0.40→0.70改善

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-03
- **ブランチ**: `claude/chattering-detection-switch-aYbwF`
- **テスト数**: 606 passed（回���なし）
- **契約違反**: **1件**（C16: `rebuild_mpc_transform` 純粋関数 — status-283由来、���回未修正）
- **条例違反**: 0件

---

## 概要

チャタリング検知時に**接触力を凍結（��解法スイッチ）**し、構造系のみでNR収束させた後、接触を再評価する機構を実装。接触あり90度曲げで **frac=0.4014 → frac=0.7050**（75%改善）。

---

## 実装内容

### 1. 接触凍結モード（Explicit Contact Switch）

チャタリング検知後の処理フロー:
1. **凍結**: 接触力 `f_c` を外力として固定、接触剛��� `K_c` をゼロに（`contact_tangent_scale=0.0`）
2. **構造NR**: 凍結接触力のもとで構造平衡を収束
3. **再評価**: 収束後に接触を再計算し、残差が `tol_force * freeze_tol_factor` 以下なら成功
4. **再凍結**: 残���が大きければ新しい接触力で再凍結し、最大5サイクル繰り返し
5. **フォールバック**: サイクル上限超過時は緩���判定（10倍tol）または通常NR復帰

### 2. 低残差チャタリング検知

従来の検知条件 `_cur_ratio > 0.5` に加え、**低残差振動**の直接検知を追加:
- 条件: `att >= 15` かつ `n_active > 0` かつ直近6反復の残差が振動/停滞
- 振動判定: `max/min比 < 100` かつ `平均残差 > tol * 10`
- 典型パターン: `||R||/||f|| ≈ 3.5e-4 ↔ 8.1e-4`（active数一定でも検知）

### パラメータ（NewtonDynamicInput + ContactFrictionInputData）

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `chattering_freeze_enabled` | `True` | 接触凍結モード有効化 |
| `chattering_freeze_max_cycles` | `5` | 凍結→再評価の最大サイクル数 |
| `chattering_freeze_nr_max` | `15` | 凍結中の構造NR最大反復数 |
| `chattering_freeze_tol_factor` | `10.0` | 凍結中の収束判定緩和倍率 |

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/contact/solver/_newton_dynamic.py` | 凍結モードロ���ック全体 + 低残差チャタリング検知 |
| `xkep_cae/core/data.py` | `ContactFrictionInputData` にパラメータ4件追加 |
| `xkep_cae/contact/solver/process.py` | パラメータ転送追加 |

---

## ベンチマーク結果

### 7本ヘリカル撚線90度曲げ（κ=π/200, θ=π/2, contact_enabled=True）

| 構成 | frac | incr | cutback | 備考 |
|------|------|------|---------|------|
| ベースライン（status-282） | **0.4014** | 234 | 15 | チャタリング停滞 |
| **接触凍結モード** | **0.7050** | 570 | 66 | **75%改善** |
| 接触なし（参考, status-281） | 1.0000 | 102 | 6 | — |

### 凍結モードの動作パターン（frac≈0.62の典型例）

```
低残差チャタリング検知 → 接触凍結モード (att=15, ||R||/||f||=1.4e-4, cycle=1/5)
凍結解除(disp)→再凍結 ||R_t||/||f||=1.1e-4 (cycle=2)
凍結解除(disp)→再凍結 ||R_t||/||f||=4.1e-4 (cycle=3)
凍結解除(disp)→再凍結 ||R_t||/||f||=4.6e-3 (cycle=4)
凍結解除(disp)→再凍結 ||R_t||/||f||=2.9e-1 (cycle=5)
凍結上限超過→通常NR復帰
disp converged (9 active)
```

- 1-2サイクルで構造収束→再評価で残差減少 → 3-5サイクルで残差増加（活性集合が大きく変動）
- 5サイクル上限後に通��NRに復帰し、変位収束で抜ける
- 次のインクリメントで再チャタリングが起きても、小dtで��結モードが再発動

---

## frac=0.70停滞の分析

### 残存課題
- frac=0.70付近で凍結サイ���ル5回でも残差が許容範囲に入らないケースが発生
- 接触ペア数が10-11に増加し、凍結→解凍時の活性集合変動がより大きい
- カットバ���ク66回と多く、dt が非常に小さくなっている

### 次の対策候補
1. **Hertz型非線形ペナルティ** (`p_n ∝ δ^{1.5}`): 接触ON/OFF���界の物理的平滑化
2. **凍���サイクル数増加**: `chattering_freeze_max_cycles=10` で更なる収束待ち
3. **凍結中のtol緩和**をさらに大きく（`freeze_tol_factor=100`）

---

## 技術的要点（次の担当者向け）

### なぜ凍結が効くか

チャタリングの根本原因: NR反復内で接触ペ��がON/OFFし、接触力が2サイクル振動。
凍結モードは接触力を固定（外力化）するため、構造系は**単純な変位問題**になり確実に収束。
再評価で新���い平衡状態の接触力を取得し、凍結→再評価を繰り返すことで活性集合を安定化。

### なぜ完��に解消しないか

凍結解除時に接触力が大きく変化すると、構造変位が大きく更新され、
次の凍結で別の活性集合になる。この「マクロなチャタリング」は凍結サイクル間で起きるため、
NR内の凍結だけでは対処できない。物理モデルの改善（非線形ペナルティ等）が必要。

### ユーザへの回答：接触剛性の非線形化について

ユーザから「接触剛性に非線形性を持���せてチャタリング抑���」の提案が���った。
- Hertz型 `p_n = k_pen * (-gap)^{1.5}` + Huber平滑化が最も筋が良い
- 接触ON/OFF境界が連続的になり、活性集合の離散的切替が物理的に解消される
- ただし `∂p_n/∂δ ∝ δ^{0.5}` でgap=0付近の接線剛性がゼロ → NR初期収束が遅い懸念

---

## 再現手順

```bash
git checkout claude/chattering-detection-switch-aYbwF
pip install -e .

# 接触あり90度曲げ（凍結モード有効、~8分）
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
" 2>&1 | tee /tmp/log-freeze-contact-90deg.log
# 期待値: frac≈0.70, incr≈570, cutback≈66

# 回帰テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour" 2>&1 | tee /tmp/log-regression.log
# 期待値: 606 passed

# 契約検証
python contracts/validate_process_contracts.py
# 期待値: 1件（C16: rebuild_mpc_transform — status-283由来）
```

---

## STA2 準拠チェ��ク

- [x] **tee ログ保存**: `/tmp/log-freeze-v3-*.log`
- [x] **再現手順記載**: 全コマンドをstatusに記載
- [x] **数値の捏造なし**: ベースラインfrac=0.4014 → frac=0.7050
- [x] **ベースライン先行取得**: status-282でfrac=0.4014確認済み
- [x] **回帰なし**: 606 passed, 0 failed

---

## TODO

- [x] 接触凍結モード実装（NewtonDynamicInput + 凍結ロジック + パイプライン貫通）
- [x] 低残差チャタリング検知追加
- [x] ベンチマーク: frac=0.40→0.70改善確認
- [x] 回帰テスト 606 passed
- [ ] frac=0.70→1.0のための追加対策（Hertz型ペナルティ or パラメータチューニング）
- [ ] C16違反修正: `rebuild_mpc_transform` のProcess化（status-283由来）
- [ ] 凍結モードの単体テスト追加

---
