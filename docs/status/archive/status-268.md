# status-268: チャタリング対策 delta_h ブースト + NR反復動的拡張

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-29
- **ブランチ**: `claude/check-status-todos-lvxyN`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3+9+4+18（変更なし）→ **合計592 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. チャタリング時 Huber delta_h ブースト機構

#### 設計意図

status-267 で判明したチャタリングの根本原因（gap 振動→p_n 振動→残差振動）に対し、
リラクゼーション（力ブレンド）に代わる対策として **delta_h ブースト**を実装。

| 項目 | 力ブレンド（status-247） | delta_hブースト（status-268） |
|------|-------------------------|-------------------------------|
| 方式 | f_c = ω·f_c + (1-ω)·f_c_prev | Huber遷移幅δ_hを倍率で拡大 |
| 残差-Jacobian整合性 | **不整合**（残差のみ減衰） | **整合**（evaluate + tangent 両方に適用） |
| status-267実績 | 0/91成功 | — |

#### 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `xkep_cae/contact/contact_force/strategy.py` | `_delta_h_boost` 属性 + `set_delta_h_boost()` + `_resolve_delta_h` で倍率適用 |
| `xkep_cae/contact/solver/_newton_dynamic.py` | チャタリング検知時にブースト発動 + 早期abort回避 + NR反復動的拡張 |
| `xkep_cae/core/data.py` | `chattering_delta_h_boost`, `chattering_extra_attempts` 追加 |
| `xkep_cae/contact/solver/process.py` | パイプライン貫通 |

#### パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `chattering_delta_h_boost` | 4.0 | チャタリング時のdelta_h倍率 |
| `chattering_extra_attempts` | 20 | ブースト時の追加NR反復上限 |

### 2. NR反復上限の動的拡張

チャタリング検知＋ブースト発動時に `max_attempts + chattering_extra_attempts` まで
NR反復を許可。変位収束に到達するための時間を確保。

forループ → whileループに変換し、動的にmax拡張可能に。

### 3. ベンチマーク結果

E=25, n_periods=30, max_increments=500, frozen_hermite_tangent=True

| 指標 | ベースライン(status-267) | status-268 | 変化 |
|------|------------------------|------------|------|
| frac | 0.4950 | **0.4978** | **+0.6%** |
| cutback | 329 | **317** | **-3.6%** |
| delta_hブースト発動 | — | 7回 | — |
| divergence abort | 96 | 105 | +9% |
| relax abort | 29 | 0 | -100% |

改善は微小。根本原因分析で、delta_h ブーストの限界を特定。

---

## 根本原因分析

### delta_h ブーストが効かない理由

1. **深い貫入**: frac>0.4 の活性ペアは gap が大きく負（Huber遷移帯 ±delta_h の外側）。ブースト 4x でも遷移帯を拡大するだけで、深い貫入のペアには影響なし。

2. **収束率が不変**: ブースト有無で残差減衰率が同一（0.853→0.728→0.622... = 0.97/iter）。delta_h は収束率のボトルネックではない。

3. **真のボトルネック = frozen Hermite tangent**: `frozen_hermite_tangent=True` が NR を修正ニュートン法に降格させ、二次収束→線形収束（0.97/iter）に。力収束（tol_force=1e-8）到達には ~585反復必要（非現実的）。

### チャタリング帯域の二相挙動

ブースト＋拡張NRにより、失敗モードが変化:

| フェーズ | attempt 0-5 | attempt 5-25+ |
|---------|-------------|---------------|
| 旧（力ブレンド） | 残差0.85→停滞 | 停滞→abort（0/91成功） |
| 新（ブースト） | 残差1.0→0.09（急減） | 0.09→0.35（発散）→abort |

新パターン: NR は 5 反復で良好な近似解に到達するが、frozen tangent の不正確な探索方向で過修正→発散。

### frac>0.4 突破の必要条件

1. **接線剛性精度の回復**: frozen_hermite_tangent=False + 安定化。dm 補正の接線への反映が二次収束の鍵。
2. **Semi-smooth NR**: 活性集合変化を NR の枠組み内で適切に処理。
3. **残差最小値リストア**: NR が通過最小値を検知し、増加開始時にリストア + dt 縮小。

---

## 試行して不採用だった変更

なし（全変更を採用。効果は限定的だが将来のインフラとして有用）。

---

## テスト結果

- 新規テスト: なし
- 既存テスト: 592 passed, 20 skipped, 1 xfailed（回帰なし）
- 契約違反: 0件
- lint: 全合格

---

## 再現手順

```bash
git checkout claude/check-status-todos-lvxyN
pip install -e .

# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"

# 契約検証
python contracts/validate_process_contracts.py

# E=25 ベンチマーク（~5min, frac≈0.4978）
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
" 2>&1 | tee /tmp/log-benchmark-268.log
```

---

## 次セッションへの引き継ぎ

### 残課題（優先度順）

1. **E=25 frac=1.0 到達**（最優先）
   - 現状 frac=0.4978（+0.6%改善）、目標は 1.0
   - **ボトルネック確定: frozen_hermite_tangent による線形収束率（0.97/iter）**
   - delta_h ブーストは深い貫入では無効（遷移帯外）
   - **次のアプローチ候補**:
     a. **frozen_hermite_tangent=False の NR 安定化**: NR 5反復で良好近似に到達→過修正で発散。過修正防止（line search 強化 or du cap）で安定化できれば二次収束回復
     b. **NR 残差最小値リストア**: NR 中の残差を追跡し、増加開始時に最小値の u に戻してインクリメント成功とみなす（disp 収束基準を活用）
     c. **Semi-smooth NR with active set method**: 外側ループで活性集合を更新、内側 NR は固定活性集合で解く
     d. **max_increments=2000 力技**: 小さな dt で frac=1.0 到達を確認（非効率だが可能性検証）

2. **NR 力収束改善**
   - 現状: 力収束 0/500（全変位収束）
   - frozen tangent が原因で力残差に構造的下限

3. **Hermite 非局所 ∂g/∂u 対応**

### 設計メモ

1. **delta_h ブーストのインフラは有用**: set_delta_h_boost() は将来的に他の場面（初期接触確立フェーズ等）でも活用可能。
2. **for→while変換**: NR ループの動的制御基盤として活用可能。
3. **失敗モード遷移の知見**: ブースト＋拡張反復で chattering → divergence に遷移。これは NR が近似解に到達している証拠であり、過修正防止が鍵。

### 開発運用メモ

- delta_h ブーストの効果は問題依存。深い貫入が支配的な場合は効果薄い。
- NR 収束率 0.97/iter は frozen tangent の固有限界。unfrozen tangent の安定化が最優先課題。
- 50incr 短縮テストは初期フェーズ（frac<0.01）のみ検証。チャタリング帯域（frac>0.4）の検証には 500incr フルテスト必須。

---
