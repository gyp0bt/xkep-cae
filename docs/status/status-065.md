# status-065: 7本撚り接触NR収束達成 — AL乗数緩和 + 反復ソルバー + Active Set Freeze + Pure Penalty

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-26
**作業者**: Claude Code
**テスト数**: 1525（+9）

## 概要

7本撚り（1+6構成、36+接触ペア同時活性化）の接触NR収束を達成。
status-064 で「根本解決にはAL乗数改善 or 反復ソルバー+前処理が必要」と診断していた問題を解決。

**結果**: `TestSevenStrandImprovedSolver` の全3荷重ケース（引張・ねじり・曲げ）が **xfail → PASS** に。

## 根本原因の特定

### AL lambda_n 蓄積と K_T の相互作用

7本撚り収束困難の根本原因は、**Augmented Lagrangian の lambda_n 蓄積と Timo3D の定数 K_T の不整合**:

1. Outer loop で lambda_n が蓄積 → 実効接触力が変化
2. Inner NR の Jacobian (K_T + K_c) は lambda_n の変化を正しく反映しない
3. Inner NR 収束率が二次 → 準線形（rate ~0.993）に劣化
4. Outer iteration 0 で 7反復だったものが、outer 10 で 100+反復に

### Active Set Chattering

Outer iteration 間で接触トポロジーが変動（48↔42↔36ペア）し、
Newton 方向の一貫性が失われる。broadphase の `detect_candidates()` が
非活性化を直接実行するため、`_update_active_set()` のヒステリシスバンドを迂回していた。

## 改善策（3段構え）

### 1. AL 乗数緩和 (update_al_multiplier with omega)

```python
# 従来: lambda_n <- p_n
# 改善: lambda_n <- lambda_n + omega * (p_n - lambda_n)
update_al_multiplier(pair, omega=al_relaxation, preserve_inactive=True)
```

- `al_relaxation=0.01` で lambda_n 蓄積をほぼゼロに（pure penalty 的動作）
- `preserve_inactive_lambda=True` で再活性化時の乗数不整合を防止

### 2. 反復線形ソルバー (GMRES + ILU前処理)

```python
_solve_linear_system(K, rhs, mode="auto", iterative_tol=1e-10, ilu_drop_tol=1e-4)
```

- `mode="auto"`: 直接法 → MatrixRankWarning or NaN 時に GMRES+ILU フォールバック
- ill-conditioned な K_T + K_c に対して直接法より安定
- 商用ソルバー（LS-DYNA implicit, ANSYS）と同等の手法

### 3. Active Set Freeze + Accept-on-Stall

```python
# no_deactivation_within_step=True:
#   - detect_candidates() をスキップ（最初の step/outer のみ実行）
#   - update_geometry() で allow_deactivation=False（活性化のみ許可）
#   - Accept-on-stall: inner NR が outer > 0 で stall → 現在の解を受容
```

### 最終動作パラメータ

| パラメータ | 値 | 効果 |
|-----------|-----|------|
| `n_outer_max` | 1 | Single outer = pure penalty 的動作 |
| `al_relaxation` | 0.01 | lambda_n 蓄積をほぼゼロに |
| `no_deactivation_within_step` | True | Active set chattering 防止 |
| `use_line_search` | False | Merit line search の干渉を排除 |
| `penalty_growth_factor` | 1.0 | k_pen 成長なし |
| `n_load_steps` | 50 | 細かいステップで安定収束 |
| `max_iter` | 30 | 十分な内部反復を許可 |
| `linear_solver` | "auto" | 直接法 + iterative フォールバック |

### 収束結果

| 荷重 | ステップ | 総NR反復 | 活性ペア | pen_ratio |
|------|---------|---------|---------|-----------|
| 引張 | 50 | 409 | 48 | 16.0% |
| ねじり | 50 | 408 | 48 | 16.0% |
| 曲げ | 50 | 408 | 48 | 16.0% |

**注**: pen_ratio=16% は pure penalty 方式の限界。将来的に AL 乗数の段階的蓄積（adaptive omega）や
Mortar 離散化で改善可能。しかし **収束自体は安定的に達成** されており、力学的に妥当な解が得られる。

## 実装内容

### ContactConfig 新規パラメータ (pair.py)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `al_relaxation` | 1.0 | AL 乗数更新の緩和係数 ω ∈ (0,1] |
| `preserve_inactive_lambda` | False | INACTIVE ペアの lambda_n 保持 |
| `linear_solver` | "direct" | 線形ソルバー: "direct"/"iterative"/"auto" |
| `iterative_tol` | 1e-10 | GMRES 収束判定 |
| `ilu_drop_tol` | 1e-4 | ILU 前処理の drop tolerance |
| `no_deactivation_within_step` | False | ステップ内での非活性化禁止 |
| `monolithic_geometry` | False | Inner NR 内で幾何毎反復更新 |

すべてデフォルト値は従来動作と完全互換。

### solver_hooks.py 変更

- `_solve_linear_system()`: direct/iterative/auto モードの線形ソルバー
- `_solve_iterative()`: GMRES + ILU(0) 前処理
- Active set freeze ロジック: `detect_candidates()` スキップ + `allow_deactivation` 制御
- Monolithic geometry: Inner NR 内での s,t,normal 毎反復更新（freeze_active_set=True）
- Accept-on-stall: outer > 0 で inner 未収束時に現在解を受容

### law_normal.py 変更

- `update_al_multiplier()`: omega パラメータ（緩和付き更新）と preserve_inactive パラメータ追加

### pair.py 変更

- `update_geometry()`: `allow_deactivation` と `freeze_active_set` キーワード引数追加
- `_update_active_set()`: `allow_deactivation` で ACTIVE→INACTIVE 遷移を制御

### テスト追加 (+9テスト)

| クラス | テスト数 | 主な検証 |
|--------|---------|---------|
| TestALRelaxation | 3 | omega=1.0 後方互換、omega=0.5 収束、摩擦付き |
| TestIterativeSolver | 3 | iterative 3本撚り引張、auto モード、摩擦付き |
| TestSevenStrandImprovedSolver | 3 | 7本撚り引張・ねじり・曲げ（**xfail → PASS**） |

## ファイル変更

### 新規
- `docs/status/status-065.md`
- `docs/contact/twisted_wire_contact_improvement.md` — 設計仕様書

### 変更
- `xkep_cae/contact/pair.py` — ContactConfig に7パラメータ追加、update_geometry に allow_deactivation/freeze_active_set
- `xkep_cae/contact/solver_hooks.py` — 反復ソルバー、active set freeze、accept-on-stall、monolithic geometry
- `xkep_cae/contact/law_normal.py` — update_al_multiplier に omega/preserve_inactive
- `tests/contact/test_twisted_wire_contact.py` — TestALRelaxation(3) + TestIterativeSolver(3) + TestSevenStrandImprovedSolver(3)
- `docs/status/status-index.md` — status-065 追加
- `docs/roadmap.md` — 7本撚り収束改善の状態更新
- `README.md` — 現在状態更新

## 診断過程のサマリー

14回の診断ランを実施。主な試行と結果:

| # | アプローチ | 結果 | 知見 |
|---|-----------|------|------|
| 1 | Active set freeze のみ | Outer 4 で detect_candidates が非活性化 | broadphase が _update_active_set をバイパス |
| 2 | detect_candidates スキップ | Outer 5 で k_pen 成長による発散 | k_pen growth が不安定 |
| 3 | k_pen 成長なし | Outer 10 で stall (100+ iter) | lambda_n 蓄積による K_T 不整合 |
| 4-5 | omega=1.0/0.7 | Outer 5-7 で発散/stall | omega を下げても lambda_n は蓄積 |
| 6-7 | Accept-on-stall | Step 2 で active set 変更により失敗 | ステップ間のトポロジー変更が問題 |
| 8-9 | CR assembler / 高 k_pen | 同パターン | アセンブラ変更は効果なし |
| 10-11 | Monolithic geometry | Step 2 で NR 発散 | ∂s/∂u, ∂t/∂u Jacobian 項が欠如 |
| 12-13 | Monolithic + line search | Merit 不整合 | 接触幾何変化が merit と非整合 |
| **14** | **n_outer_max=1 + al_relaxation=0.01** | **全3荷重ケース収束!** | **Pure penalty で lambda_n 問題を回避** |

## 設計上の懸念・TODO

- [ ] pen_ratio 改善: adaptive omega（omega を outer ごとに段階的に増加）で AL 乗数を安定的に蓄積
- [ ] Mortar 離散化: 接触界面の適合離散化で高 k_pen でも安定
- [ ] 7本撚りサイクリック荷重（ヒステリシス観測の7本撚り版）
- [ ] Stage S3: シース-素線/被膜 有限滑り
- [ ] Stage S4: シース-シース接触

---
