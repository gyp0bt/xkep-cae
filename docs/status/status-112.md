# status-112: 19本NCP収束達成（径方向圧縮テスト）

[← README](../../README.md) | [← status-index](status-index.md) | [← status-111](status-111.md)

日付: 2026-03-05

## 概要

1. **19本NCP収束達成**: 径方向圧縮荷重でアクティブ接触を含むNCP収束を確認
2. **収束パラメータの同定**: k_pen continuation + contact_force_ramp + adaptive omega の組合せ
3. **テスト追加**: `TestNCP19StrandRadialCompression` クラス

## 技術的知見: 19本撚線NCP収束の鍵

### 問題の本質

19本撚線NCP収束の困難さは以下の3点に集約される:

1. **荷重形態の問題**: 純引張・均一曲げでは全素線が同一変位するため層間の相対運動が生じず、接触が活性化しない
2. **gap=0での一斉活性化**: 密着配置（gap=0）では初期貫入ペアが552個同時に活性化し、鞍点系が不安定化
3. **k_penのメッシュ依存**: 16要素/ピッチで要素長L≈0.003m → k_pen = 12EI/L³ が構造剛性に対して過大

### 解決策: 径方向圧縮 + 段階的活性化

**荷重**: Layer 1素線（sid 1-6）の中間節点に中心向き径方向力（5N/節点）を付与。中心素線（sid=0）は全自由度拘束。

これにより:
- Layer 1素線が中心素線に押し付けられる
- 最初は少数ペア（6→18→24→30）が段階的に活性化
- 接触力と外力のバランスが物理的に明確

### 収束パラメータ

| パラメータ | 値 | 役割 |
|:---|:---:|:---|
| gap | 0.0001 m | 微小ギャップ（初期貫入なし） |
| g_on / g_off | 0.001 / 0.002 | 広い幾何的活性化バンド |
| k_pen_scale | 0.1 | EI/L³ ベース |
| k_pen_continuation | True | 0.1倍から5ステップで段階的増加 |
| k_pen_continuation_start | 0.1 | 初期k_pen = ターゲットの10% |
| contact_force_ramp | True | NR反復初期の接触力ランプ（5反復） |
| adaptive_omega | True | 0.3→0.02〜0.8の自動緩和 |
| adaptive_timestepping | True | 不収束時のΔt縮小 |
| residual_scaling | "rms" | RMS残差スケーリング |
| chattering_window | 3 | active-setチャタリング抑制 |
| tol_force / tol_ncp | 1e-4 | 収束公差 |

### 収束プロファイル

```
Step  1-6:  0 active, 収束 (k_pen continuation 増加中)
Step  7:    6 active, 収束 (初めて接触活性化)
Step  8:   18 active, 変位収束
Step  9:   18 active, エネルギー収束
Step 10:   18 active, 力残差収束 (R_u=4.8e-5, C_n=2.4e-5)
Step 11:   18 active, 力残差収束
Step 12:   24 active, 力残差収束
Step 13:   30 active, 力残差収束
Step 14:   30 active, エネルギー収束
Step 15:   24 active, 力残差収束 (R_u=8.5e-5, C_n=8.1e-8)
→ 全15ステップ収束, 172 NR反復, k_pen=7.5e6
```

### 引張テストが接触なしになる理由

純引張では接触が活性化しない原因:
- **線形梁モデル**: テストは線形Timoshenko梁を使用。幾何学的非線形がないため、ヘリカルワイヤの軸引張→径方向圧縮カップリングが存在しない
- **均一荷重**: 全素線に同一荷重 → 全素線が同一変位 → 相対変位ゼロ
- **NCP活性化条件**: p_n = λ + k_pen * (-g) > 0 が必要。g > 0（正ギャップ）かつλ=0では常に非活性

実際のワイヤロープ曲げ解析には幾何学的非線形（コロテーショナル等）が必要。

## 変更詳細

### test_ncp_convergence_19strand.py

- `TestNCP19StrandRadialCompression` クラス追加
- `_radial_load()` ヘルパー: Layer 1の中間節点に径方向力
- `_fixed_dofs_with_center()` ヘルパー: 中心素線全拘束

## テスト状況

- 新規テスト: `test_ncp_19strand_radial_with_active_contacts` — PASSED (354秒)
  - converged=True, 15ステップ, 172 NR反復, 24ペアアクティブ
- 既存テスト: 影響なし

## TODO

- [ ] 37/61/91本での径方向圧縮収束テスト
- [ ] 非線形梁（コロテーショナル）での引張・曲げ接触テスト
- [ ] k_penのL_avg依存問題の根本対策（構造剛性ベースのクランプ）
- [ ] 引張＋曲げ複合荷重での収束テスト
- [ ] 摩擦付き19本収束テスト

## 確認事項・設計上の懸念

1. **線形梁の限界**: 現在のテストは線形梁を使用しているため、実際のワイヤロープ挙動（幾何学的非線形）とは異なる。非線形梁での検証が今後必要。
2. **接触数の物理的妥当性**: 理想的な19本撚線では中心-Layer1間で6ペア、Layer1-Layer2間で12ペアの接触が期待される。現在の24ペアは妥当。
3. **エネルギー収束**: 一部ステップで力残差ではなくエネルギー収束判定を使用。力残差公差を緩めるか、NR反復数を増やすことで改善可能。
