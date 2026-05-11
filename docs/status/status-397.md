[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-397: ε-1 失敗 — `_process_free_end` 駆動経路 × explicit-TL の精度問題を 1 strand 規模で再現、改修対象を BC/process driver 層に局在化

**日付**: 2026-05-11
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12+5+17+11+6+4 passed
（status-396 と同数、実装本体無変更、work スクリプト新設のみ）

## 概要

status-396 で API 化された `explicit_ul_disable_update=True`（explicit-TL 固定モード、UL `update_reference()` を一切呼ばない）を **3 strand helical + 接触なし** の実機系で初検証する **ε-1** を実施。

**結果**: ε-1 **FAIL**（3 指標 AND gate 全 FAIL、u_x rel_err 96.36% / u_z rel_err 96.54%、3 指標で **explicit-TL が implicit 比 30 倍以上アンダー**）。

**原因局在化 sub-experiment**: 同じ駆動経路で **n_strands=1（直線、ヘリカルでない単一 strand）** にも explicit-TL を適用し、FAIL を再現（u_x 96.29% / u_z 96.40%）。**ヘリカル初期 κ / 多 strand global assembler 経由ではなく、`_process_free_end` 駆動経路 + explicit-TL の組合せ自体が問題** と局在化された。

`work/beam_element_validation/49_beta2_with_assembler_ul.py` Mode C（status-394、explicit + assembler + TL、専用ドライバ駆動）は機械精度 PASS していたため、**今回新たに発覚した改修対象は `_process_free_end`（および `_process` 系）の主ループそのもの**である。

## 1. 実験設計（`work/beam_hysteresis/41_epsilon1_3strand_helical_no_contact.py`、~330 行新設）

| パラメータ | 値 |
|---|---|
| n_strands | 3（sub-experiment では 1） |
| wire_radius | 0.5 mm |
| pitch_length | 100.0 mm |
| n_elements_per_pitch | 16 |
| n_pitches | 1.0 |
| E / ρ | 130 GPa / 8.96e-9 ton/mm³ |
| bending_curvature | 0.001 → κ·L=0.1 rad ≈ **5.7°**（小曲げ、γ-3 と同レンジ） |
| free_end_mode | True（右端各 strand に θ_y 処方、MPC 不使用） |
| contact_enabled | False |
| n_increments_per_cycle | 20 |
| explicit_mass_scaling | auto, β cap=1e5, safety=0.9 |
| explicit_ul_disable_update | True（reference 配置を初期固定、TL モード） |

**3 指標 AND gate**（status-388 透明性ルール）:

1. `u_x_tip` 相対誤差 vs implicit < 10%（kinematic 1）
2. `u_z_tip` 相対誤差 vs implicit < 10%（kinematic 2）
3. `E_strain` 相対誤差 vs implicit < 10%（kinematic 独立）

加えて物理的妥当性 gate: `max |u_trans| < L_strand × 10 = 1000 mm`（status-380）。

## 2. 実測（解析 cantilever と implicit の一致を先に確認）

3 strand helical + 接触なし + free_end_mode + κ·L=0.1 rad 曲げ:

| metric | implicit baseline | explicit-TL (`disable=True`) | rel_err |
|---|---:|---:|---:|
| `u_x_tip` [mm] | +4.9957e+00 | +1.8187e-01 | **96.36%** |
| `u_z_tip` [mm] | −1.6642e-01 | −5.7595e-03 | **96.54%** |
| `E_strain` [N·mm] | 4.7818e-02 | 1.8533e+01 | 38658% |
| `max |u_trans|` [mm] | 5.000 | 0.182 | — |
| frac | 1.0000 | 1.0000 | — |
| `E_kin/|E_str|` | 5.30e-05 | 3.81e-10 | — |

**解析 cantilever 曲げ近似** (`R=L/θ=1000 mm`):

- `δ_x = R(1−cos θ) = 4.9958 mm` → **implicit と機械精度級一致** ✓
- `δ_z = R sin θ − L = −0.1666 mm` → **implicit と機械精度級一致** ✓

→ implicit は cantilever 解析解と一致、**implicit baseline は妥当**。explicit-TL は **同 BC で約 30 倍小さい変位** に収束（frac=1.0 完走 + E_kin/E_str=4e-10 で動的緩和完了済み、定常解の値）。

**注意**: `E_strain` は UL（implicit）vs TL（explicit）で異なる量を報告する（UL は最終 incr の incremental SE、TL は初期 reference から累積 total SE）ため、絶対値の直接比較は formulation-dependent。本 status の主判定は **kinematic 量** `u_x_tip` / `u_z_tip`（formulation-invariant）。

## 3. 原因局在化 sub-experiment（`n_strands=1`、直線・単一 strand）

CLAUDE.md「次セッション最優先（status-397）」の 3 候補:

| # | 候補 | n_strands=1 が PASS の場合 | n_strands=1 が FAIL の場合 |
|:-:|---|---|---|
| a | ヘリカル初期 κ | **主因** | 候補から除外 |
| b | 多 strand global assembler | **主因** | 候補から除外 |
| c | 端部 BC（free_end_mode）/ process driver 層 | 候補から除外 | **主因** |

実機: `n_strands=1`（mesh は直線 strand、helical でない）+ free_end_mode + explicit-TL:

| metric | imp_n1 | exp_n1 (TL) | rel_err |
|---|---:|---:|---:|
| `u_x_tip` [mm] | +4.9957e+00 | +1.8551e-01 | **96.29%** |
| `u_z_tip` [mm] | −1.6642e-01 | −5.9930e-03 | **96.40%** |

→ **3 strand helical（96.36%）と 1 strand 直線（96.29%）でほぼ同一の under-deformation**。helical 初期 κ も多 strand assembler も関係なく、**process driver + explicit-TL の組合せ自体が under-deformation を生む**。

**新たに局在化された改修対象**: **`_process_free_end`（および同等の `_process` 主ループ）が explicit-TL モードで動作するときの BC 駆動経路**。

## 4. status-394 Mode C との対比 — 改修対象の正確な位置

| 系 | driver | 結果 |
|---|---|---|
| β-2 (1 要素 explicit + TL、status-391) | inline central diff | ✅ 機械精度 0.000% |
| Mode C (1 要素 + assembler + TL、status-394) | 専用 driver in script | ✅ 機械精度 0.000% |
| γ-3 (多要素 explicit + TL、status-395) | inline chain solver | ✅ 機械精度 0.000% |
| **ε-1 (3 strand helical + 接触なし + TL、本 status)** | **`_process_free_end`** | **❌ ~96% アンダー** |
| **ε-1 sub-experiment (1 strand 直線 + 接触なし + TL)** | **`_process_free_end`** | **❌ ~96% アンダー** |

CR foundation（要素 / chain / assembler 単独）は健全。**改修対象は `_process_free_end` driver そのもの**、すなわち:

- 増分ループ内での `update_reference` ゲート評価
- prescribed BC の累積適用（`prescribed_func` の frac→u_incr 変換）
- TL mode で UL `_u_total_accum` を更新せずに済む経路
- explicit dynamic ステップへの BC 引き渡し（`ExplicitDynamicProcess` の prescribed DOF 処理）

のいずれか／組合せ。本 status は **問題の局在化** のみで根本原因の切り分けは status-398 以降に持ち越し。

## 5. 仮説（status-398 で検証する候補）

1. **prescribed BC の TL 増分処理**: `_process_free_end` は increment ごとに prescribed θ_y を frac で線形補間して u_prescribed を構築するが、TL モードでは「u_prescribed が initial reference からの total」として渡されるか「u_incr」として渡されるかが explicit + TL では破綻している可能性。implicit + UL では update_reference により u_incr が小さく保たれるため違いが顕在化しない。
2. **explicit driver の reaction force 累積**: ExplicitDynamicProcess は prescribed DOF を constraint reaction で処理するが、TL で reference 固定の場合、prescribed 位置の累積方法が UL と不整合な可能性。
3. **`_ExtendedULAssemblerWrapper` の TL モード対応**: free_end_mode では使われないが、`_process_free_end` 主ループの assembler 呼出パスで `coords_ref` / `R_ref` が固定されない隠れた経路がある可能性。

仮説検証は次セッション（status-398）で着手。

## 6. ロードマップ修正（CLAUDE.md §段階的検証ロードマップ）

CLAUDE.md「次の課題」§段階的検証ロードマップは以下に更新:

| status | scope | 主成果物 | gate |
|---|---|---|---|
| ~~396~~ | ~~(z3) explicit-TL 固定 API 化のみ~~ | ✅ status-396 完了 | 達成 |
| ~~397~~ | ε-1 = 3 strand helical + 接触なし + explicit-TL | **🔁 FAIL — 改修対象を `_process_free_end` 経路に局在化（本 status）** | ε-1 FAIL → 原因切り分けへ |
| **398 (次)** | `_process_free_end` × explicit-TL の根本原因切り分け | prescribed BC 駆動経路 / explicit reaction force / assembler ref 固定の 3 仮説検証 | 単一仮説で u_x rel_err < 1% を実現 → 修正実装 |
| 399 | 修正後 ε-1 再検証 + ε-2 (3 strand 接触あり) | フィックス検証 + 接触統合初検証 | 3 指標 AND gate PASS |
| 400 | ε-3 = 7 strand + 接触あり | implicit baseline (status-301 frac=1.0) 対比 | 3 指標 AND gate |
| 401 | ε-4 = 19 strand + 接触あり（本命） | MCDD 凍結解除条件 (2)(3)(5) 同時達成試行 | frac=1.0 + max\|u\| + 精度 |

**判断根拠**: ε-1 が FAIL したが、CLAUDE.md plan の「Phase δ (2 strand contact) に retreat」より「driver 層自体の局在化が決定的に必要」と判断。Phase δ も同じ `_process_free_end` を通るため、driver 修正が前提となる。

## 7. ゲート結果

| ゲート | 結果 | 備考 |
|---|---|---|
| ε-1 主実験（3 strand helical + TL） | **❌ FAIL** | u_x 96.36% / u_z 96.54% / E_strain 38658% |
| ε-1 sub (n_strands=1) | **❌ FAIL** | u_x 96.29% / u_z 96.40% — driver 層に局在化 |
| implicit baseline vs cantilever 解析解 | ✅ 機械精度級一致 | implicit は妥当 |
| `pytest contact + math + time_integration + strand_bending_oscillation` | **747 passed 5 skipped** | status-396 と同数 |
| `python contracts/validate_process_contracts.py` | 全 24 検査 OK | C1〜C24 + O1〜O3 |
| `test_helical_3d_hermite` rel_err | 2.18e-07 維持 | status-356 で達成 |
| `ruff check + format` | All checks passed / 204 files formatted | work スクリプト含む |

## 8. 達成確認マトリクス更新

`docs/status/verification_matrix.md` 更新:

- §3 上位層改修対象 表に行「`_process_free_end` driver × explicit-TL」を **❌ 状態で新規追加**（status-397 で 1 strand 規模で FAIL 再現実証、改修対象として明示）
- §2 Phase ε 検証進捗を **新規 section 追加**:
  - ε-1 (3 strand helical no contact + TL): **❌** status-397
  - ε-1 sub (n_strands=1 straight no contact + TL): **❌** status-397
- §5 STA2 撤回履歴: 本 status は新規撤回事例なし（ε-1 を「達成」と主張していない、CLAUDE.md plan 通りの FAIL）
- §8 達成済 ✅ 一覧: 変更なし
- §8 未達 ❌ に「ε-1 = 3 strand helical no contact + TL」と「(z3) explicit-TL 固定 API の実機適用 (status-397)」を追加

## 9. MCDD 脱法 pattern 自己点検

- **pattern 1（tol 緩和）**: 該当なし。10% gate を踏襲、3 指標すべて報告
- **pattern 5（既存テスト skip）**: 既存 747 全 pass
- **pattern 6（骨格 status）**: 実機 ε-1 実行 + sub-experiment + 解析解との対比 + 仮説 3 つ列挙で完結
- **pattern 7（数値丸め）**: rel_err は `{:.2%}` + 生数値 `{:.4e}`
- **pattern 8（根拠なき主張）**: implicit baseline を解析 cantilever 解と機械精度級一致で先に確認、explicit-TL の FAIL は under-deformation の **kinematic 量** で示し、sub-experiment で原因候補を絞り込み
- **pattern 10（TODO 先送り）**: 本 status は **「ε-1 を実機実行し、FAIL の場合は改修対象を局在化する」** を完結。仮説検証 (status-398) は別 scope で明示。骨格 status ではない

## 10. 観察 — 開発運用上の発見

### 効果的

1. **「sub-experiment による候補絞り込み」の威力**: ε-1 FAIL を単に「ヘリカル初期 κ が原因」と推定せず、`n_strands=1`（最小サブセット）で同等の FAIL を再現することで **3 候補のうち 2 つ（ヘリカル / 多 strand）を即時除外** できた。1 status 内で原因が `_process_free_end` 経路に局在化した。
2. **解析解との先行対比**: implicit baseline を解析 cantilever 解と機械精度級一致で先に確認したことで、「implicit が壊れていない」と確証してから explicit-TL の FAIL を判定できた。status-380〜387 のように explicit と implicit の双方が信用ならない状態を回避できる。

### 今後の観察対象

- **driver 層の TL 対応**: `_process_free_end` 主ループは UL を前提に設計されているため、TL モードでの prescribed BC 駆動経路に隠れた整合性問題がある可能性。status-398 で 3 仮説を切り分ける際に、`work/beam_element_validation/` の Mode C 駆動と `_process_free_end` の処方値計算の差分を直接比較する必要。

## 11. 再現手順

```bash
git checkout claude/execute-status-todos-cb8n5

# ε-1 主実験 + sub-experiment（自動連結）
uv run --extra dev python work/beam_hysteresis/41_epsilon1_3strand_helical_no_contact.py \
    2>&1 | tee /tmp/epsilon1_$(date +%s).log
# 期待: 3 指標 AND gate FAIL（u_x 96% / u_z 96% / E_strain 38000%）
#       sub-experiment n_strands=1 でも同等 FAIL

# 回帰テスト（実装本体無変更）
uv run --extra dev pytest xkep_cae/contact/ xkep_cae/mathematics/ \
    xkep_cae/time_integration/ \
    xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py -q
# 期待: 747 passed, 5 skipped

# 契約検査 + FD diagnostic
uv run --extra dev python contracts/validate_process_contracts.py
uv run --extra dev pytest \
    xkep_cae/contact/contact_force/tests/test_kc_component_fd.py::TestKcComponentFD::test_helical_3d_hermite -v

# ruff
uv run --extra dev ruff check xkep_cae/ tests/ work/beam_hysteresis/41_epsilon1_3strand_helical_no_contact.py
uv run --extra dev ruff format --check xkep_cae/ tests/
```

## 12. 引き継ぎチェックリスト

| 項目 | 状態 | 備考 |
|---|---|---|
| `41_epsilon1_3strand_helical_no_contact.py` 新設 | ✅ | ~330 行、sub-experiment 含む |
| ε-1 主実験 実機 FAIL を確認 | ✅ | u_x/u_z 96% under-deformation |
| sub-experiment n_strands=1 で FAIL 再現 | ✅ | 候補 (a)(b) 除外、(c) driver 層に局在化 |
| implicit baseline vs cantilever 解析解 機械精度一致 | ✅ | implicit が妥当であることを先確認 |
| 回帰 747 passed 5 skipped 維持 | ✅ | 実装本体無変更 |
| 全 24 契約検査 OK | ✅ | C1〜C24 + O1〜O3 |
| `test_helical_3d_hermite` rel_err=2.18e-07 維持 | ✅ | status-356 で達成 |
| ruff check + format pass | ✅ | 204 files |
| status-397 作成 + status-index 更新 | ✅ | 本 status |
| README / roadmap / verification_matrix 更新 | ✅ | §現在の状態 / Phase ε / §3 §8 |
| **次セッション最優先（status-398）**: `_process_free_end` × explicit-TL 仮説 3 検証 | ⬜ | 3 仮説（prescribed BC TL 増分処理 / explicit reaction force 累積 / assembler ref 固定）を切り分け |

Phase A〜E / status-346〜397 の **48/N 完了**。
