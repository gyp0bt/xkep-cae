# status-220: 動的三点曲げ押し下げ変換 + 準静的ソルバー自動検知 + 接触力符号問題特定

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-20
**ブランチ**: `claude/improve-three-point-bend-Qe3Xt`

---

## 概要

status-219 の TODO を引き継ぎ、三点曲げ問題設定を自由振動バウンスから変位制御押し下げに変更。
準静的ソルバー自動検知機構を追加。
**接触力符号問題を特定したが、収束修正は次セッションへ引き継ぎ。**

## 変更内容

### 1. 三点曲げ問題設定の変更（自由振動 → 変位制御押し下げ）

`DynamicThreePointBendContactJigProcess` の問題設定を変更:

**変更前**: ワイヤに初期変位 → 自由振動 → ジグにバウンド（接触/離脱繰り返し）
**変更後**: ジグ y-DOF を変位制御で下方に押し下げ → ワイヤを三点曲げ

- `initial_gap: 0.05 → 0.0`（ジグをワイヤに密着配置）
- `du_norm_cap: 0.1 → 0.0`（ステップクリッピングなし）
- `u0 = zeros`（ワイヤ静止状態から開始）
- ジグ y-DOF を変位処方（`prescribed_dofs`）
- ジグ x/z/回転 DOF は固定
- 結果出力: `dynamic_amplification` → `analytical_stiffness_eb`, `effective_stiffness`, `stiffness_error_eb`

### 2. 準静的ソルバー自動検知機構

テスト実行時に `StaticSolverWarning` を自動検知し、セッション終了時にレポート出力。

**目的**: 準静的ソルバーで通ったテストが「動的でも動く」と誤解されることを防ぐ。
準静的は動的ソルバーの足がかりに過ぎず、最終検証は動的ソルバーで行うべき。

**実装**:
- `tests/conftest.py` — `_detect_static_solver_usage` autouse fixture
- `pyproject.toml` — `static_solver_ok` マーカー登録
- `tests/contact/test_three_point_bend_jig.py` — 意図的な静的テストに `@pytest.mark.static_solver_ok` 付与

**動作**:
1. 全テストで `StaticSolverWarning` を監視
2. `@pytest.mark.static_solver_ok` 付きテストは検知対象外
3. セッション終了時に「Static Solver Detection Report」を出力

### 3. テスト更新

`TestDynamicContactJigPhysics` を押し下げ用に書き換え:
- `test_wire_deflection_matches_push` — ワイヤ変位が押し下げ量と一致
- `test_contact_force_order` — 接触力のオーダー検証
- `test_wire_deflects_downward` — ワイヤが下方に変位（現在FAIL: 符号問題）
- `test_energy_history_recorded` — エネルギー履歴記録

## 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | 押し下げ変位制御化 |
| `tests/contact/test_three_point_bend_jig.py` | テスト更新 + static_solver_ok マーカー |
| `tests/conftest.py` | 準静的ソルバー自動検知（新規） |
| `pyproject.toml` | static_solver_ok マーカー登録 |

## 発見した問題（未修正）

### 1. 接触力符号問題（最重要 — 次セッションで修正）

**症状**: ジグが下方に押しても、ワイヤが**上方**に0.534mm変位する。

**根本原因**: 残差式 `R_u = f_int + f_c - f_ext`（`_nuzawa_steps.py:126`）において:
- `f_c` は内力と同じ符号で加算される（= 内力として扱い）
- しかし softplus の接触力計算は `f_c` を **負** で返す（= 外力的な符号）
- 結果: NR 補正 `du = -K⁻¹R` がワイヤを上方に動かす

**調査すべき箇所**:
- `xkep_cae/contact/contact_force/strategy.py` — softplus 接触力の符号定義
- `xkep_cae/contact/solver/_nuzawa_steps.py:126` — `R_u = f_int + f_c - f_ext`
- 接触力ベクトル `f_c` の組み立て過程（形状関数 × 法線力）

**なぜ毎回再発するか**: softplus の力の符号規約と残差式の符号規約が文書化されておらず、
修正しても別の場所で「つじつま合わせ」の符号反転が入り、別の問題として再発する。
→ **統一的な符号規約の文書化が必要**。

### 2. f_ext_ref_norm = 0 問題

変位制御問題では `f_ext_total = 0`（外力なし）のため、力収束判定の参照値が 1.0 にフォールバック。
`||R_u|| < tol_force` が絶対値判定になり、tol_force=1e-6 では厳しすぎる。

**修正方針**: `f_ext_ref_norm = 0` の場合に `max(||K*u||, 1.0)` 等の別スケールを使う。

### 3. 線形収束（exact_tangent=True でも）

`exact_tangent=True` + `du_norm_cap=0` でも収束率 ~0.97/iter の線形収束。
原因候補: 接触接線剛性に幾何剛性項（d²g/du²）が欠落している可能性。

## TODO（次セッションへの引き継ぎ）

- [ ] **接触力符号規約の統一**: softplus 接触力の符号と残差式の符号規約を文書化し、統一。`f_c` が「反力（正 = 押し返す）」なのか「作用力（正 = 押す）」なのかを明確にし、全コードパスで整合させる
- [ ] **Uzawa 有効化**: 純粋ペナルティ法（n_uzawa_max=1）は接触力精度不足。n_uzawa_max=3 以上で拡大ラグランジアンを有効化。ただし符号問題の修正が先決
- [ ] **f_ext_ref_norm 修正**: 変位制御問題で力収束参照値がゼロになる問題の解決
- [ ] **線形収束の原因調査**: 接触接線剛性の幾何剛性項欠落を調査（d²g/du²）
- [ ] **符号規約ドキュメント**: 接触力・残差・NR補正の符号規約を一箇所に文書化
- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動

## テスト状態

- 新規追加テスト: 0件
- 既存テスト回帰: なし（動的接触ジグテストは既知のFAIL）
- 既存FAIL（本変更と無関係）:
  - `test_large_amplitude_converges` — 変更前からFAIL
  - `test_numerical_dissipation_rate` — 変更前からFAIL
  - `test_render_produces_images` — 変更前からFAIL

---
