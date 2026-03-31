# status-276: NR収束改善調査 — 接線不整合の特定と対策方針

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-31
- **ブランチ**: `claude/check-status-todos-yKUof`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+3 → **合計600 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 現状の問題

7本撚線曲げ揺動ベンチマーク（`test_strand_bending_oscillation_converges`）が frac≈0.41 で停止。frac=1.0 に到達しない。

**ソルバー構成**: Huber ペナルティ（C1連続）+ Coulomb摩擦 + UL梁 + Generalized-α動的 + adaptive timestepping。

**理論的にはHuber C1連続ならNRは二次収束するはず。しかし実際は振動発散する。**

---

## NR反復の詳細観察（Incr 19, frac=0.4125）

```
att=0: ||R_t||/||f|| = 1.000, active=0   ← 初期（接触なし）
att=1: ||R_t||/||f|| = 0.123, active=54  ← 改善、だが54ペア同時活性化
att=2: ||R_t||/||f|| = 0.552, active=24  ← 悪化、active減少
att=3: ||R_t||/||f|| = 0.087, active=36  ← 最小値
att=4: ||R_t||/||f|| = 0.324, active=78  ← 爆発開始
att=5: ||R_t||/||f|| = 1.785, active=84  ← 発散
att=6: ||R_t||/||f|| = 5.197, active=112 ← 発散加速
att=7: ||R_t||/||f|| = 9.830, active=105 ← 発散
→ 発散検知、最小残差リストア (att=3 に戻す)
```

**特徴**: 残差とactive数が**反相関で振動**。NR更新が過修正→多数ペア貫入→次で戻しすぎ→ペア離脱→繰り返し。

**重要**: `freeze_active_set=True` はACTIVE/INACTIVEのステータスフラグを凍結するが、**p_n（ペナルティ力）はギャップから毎反復再計算**される。n_active（p_n > 0 のペア数）はNR反復中に自由に変動する。

---

## 試行した対策と結果

### ベースライン
- **status-260 (worktree)**: frac=0.5914 (224秒)
- **現HEAD**: frac=0.4125 (22秒)

### パラメータ調整
| 条件 | frac | 評価 |
|------|------|------|
| ベースライン (frozen_hm=False, tangent dm OFF) | 0.413 | 基準 |
| consistent_st_tangent=False (K_st除外) | 0.388 | 悪化 |
| frozen_hermite_tangent=True (全凍結) | 0.375 | 悪化 |
| use_hermite_centerline=False (Hermite OFF) | 0.213 | 大幅悪化 |
| n_increments_per_cycle=80 (細かい増分) | 0.334 | 悪化 |
| huber_delta_h=0.025 | 0.400 | 微悪化 |

### 構造的対策
| 対策 | frac | 評価 |
|------|------|------|
| NRインナー接触DOF凍結 | 0.337 | 悪化（物理的不整合） |
| サーフェスペアフィルタ | 0.413 | 変化なし（broadphaseが既に厳しい） |
| evaluate+tangent 両方dm ON | 0.388 | 悪化 |

### bisect結果
- `f7db2ae` (status-260): frac=0.591
- `60a6f3d` (active_contact_dofs): frac=0.591
- `d9c3758` (delta_h API): frac=0.591
- `7058453` (three_point_bend delta_h): frac=0.591
- **`7403aa2` (status-264)**: frac=0.375 ← **回帰コミット**
- 変更内容: frozen_hermite_tangent=True デフォルト追加 + _cur_ratio統一
- frozen=False にすると 0.413 に部分回復するが 0.59 には届かない

---

## 未解決の謎

1. **status-260 では evaluate/tangent 両方dm ON で frac=0.59 だった**。現在のコードで同じ設定にしても frac=0.39。dm以外の変更が影響。
2. **dt を小さくしても改善しない**。adaptive timestepping がカットバックしても同じ場所で停止。
3. **Huber C1連続なのにNR二次収束しない**。接線剛性と残差の不整合が疑われるが、tangent dm ON にしても悪化。

---

## 根本原因の仮説（次セッションで検証）

### 仮説A: NRソルバー内部ロジックの累積変更
`7403aa2` コミットには `_cur_ratio` 統一と `frozen_hermite_tangent` 追加の2つの変更が入っている。個別には改善しないが、**status-260→261→264 の累積で何かが壊れた**。worktree で status-261 各コミットを個別にテストして特定すべき。

### 仮説B: 接線剛性のK_st/K_c項の符号・スケール不整合
consistent_st_tangent=True がデフォルトで、K_st（滑り剛性）が接線行列に含まれる。K_st は ∂f/∂(s,t) · ∂(s,t)/∂u の項で、大きな値になりうる。**K_st の符号や大きさが不正確**なら、NR更新が過修正される。FD接線診断（TangentFDDiagnosticProcess）で K_st の精度を定量評価すべき。

### 仮説C: 動的項（慣性力・減衰力）との干渉
Generalized-α 時間積分の慣性力 c0*M*Δu が接触力と干渉。dt が小さいと c0 が大きくなり、慣性項が支配的になって接触力の寄与が相対的に小さくなる。これが dt を小さくしても改善しない原因かもしれない。**準静的（dt→∞）でのNR挙動**を比較すべき。

---

## 実装した変更（本セッション）

| 変更 | ファイル | 状態 |
|------|----------|------|
| frozen_hermite_tangent デフォルト True→False | `_contact_pair.py` | コミット済み |
| サーフェスペアフィルタ（デフォルトOFF） | `_manager_process.py`, `_contact_pair.py` | コミット済み |
| NRインナー接触DOF凍結（デフォルトOFF） | `_newton_dynamic.py` | コミット済み |
| テスト品質改善（非平行座標化等） | テストファイル2件 | コミット済み |

---

## 次セッションへの推奨アクション

1. **FD接線診断の実行**（最優先）
   - `TangentFDDiagnosticProcess` を frac=0.40 付近で実行
   - K_st, K_c, K_friction の各項の FD 一致率を定量評価
   - 不整合項を特定 → 修正

2. **準静的ソルバーでの検証**
   - dt を十分大きく（or 静的ソルバー）して慣性効果を除去
   - 純粋な接触NRの収束性を確認
   - 動的項が悪影響しているかどうかを切り分け

3. **status-260 との差分精査**
   - `git diff 7058453..7403aa2 -- xkep_cae/contact/` の変更を1行ずつレビュー
   - 特に `_cur_ratio` 統一の影響を frozen=False 状態で再検証

---

## 再現手順

```bash
git checkout claude/check-status-todos-yKUof
pip install -e .

# 全テスト
python -m pytest xkep_cae/ tests/ -q --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"

# 撚線ベンチマーク（~22秒）
python -c "
from xkep_cae.numerical_tests.strand_bending_oscillation import *
cfg = StrandBendingOscillationConfig(
    n_strands=7, wire_radius=0.5, pitch_length=100.0,
    n_elements_per_pitch=16, n_pitches=1.0,
    E=130.0e3, nu=0.3, rho=8.96e-9,
    bending_curvature=0.001, n_cycles=1,
    n_increments_per_cycle=40, rho_inf=0.9, mu=0.15,
    max_nr_attempts=50, tol_force=1e-8, max_increments=10000,
    exclude_same_strand=True,
)
r = StrandBendingOscillationProcess().process(cfg)
sr = r.solver_result
print(f'frac={sr.load_history[-1]:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}, {sr.elapsed_seconds:.1f}s')
"

# status-260 でのベースライン検証（~224秒）
git worktree add /tmp/wt-s260 f7db2ae
cd /tmp/wt-s260 && pip install -e . && python -c "同上" && cd -
git worktree remove /tmp/wt-s260

# 契約検証
python contracts/validate_process_contracts.py
```

---

## STA2 準拠チェック

- [x] **tee ログ保存**: 全ベンチマーク結果を /tmp/log-bench-*.log に保存
- [x] **再現手順記載**: 全コマンドをstatusに記載
- [x] **数値の捏造なし**: frac=0.413→0.388の悪化を正直に報告
- [x] **ベースライン先行取得**: frac=0.413（現HEAD）、frac=0.591（status-260 worktree）

---
