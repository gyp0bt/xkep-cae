[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-395: Phase γ-3 完了 — 多要素 explicit + TL で circular arc 収束を O(1/n²) 再現実証（4/5 PASS、log-log slope=-2.000、γ-1 implicit と数値一致）

**日付**: 2026-05-08
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12+5+17+11+6 passed（status-394 と同数、新規 work スクリプトのみで実装本体無変更）

## 概要

ユーザー指示「implicit は完全凍結、解除想定なし」を受け、explicit 一本路線の foundation
確認として **Phase γ-3 (multi-element explicit + TL)** を実施。Phase β-2 (1 要素 explicit,
status-391) を multi-element に拡張し、Phase γ-1 (multi-element implicit static, status-392)
の対偶として「explicit + TL でも multi-element 規模で circular arc に O(1/n²) 収束するか」を
3 指標 AND gate (status-388 透明性ルール) で検証。

**結論**: n_elements ∈ {2, 4, 8, 16} すべてで 3 指標 AND gate PASS、**log-log slope=-2.000**
で γ-1 (implicit) と **数値レベルで一致**。多要素 explicit + TL モードで CR foundation
健全性確定。続く (z3) 「explicit モード TL 固定 API + 19 本撚線適用」へ進む準備が整った。

## 1. 実験設計

α-3 / β-2 / γ-1 と同 BC（左端 fix、右端 θ_y=0.15 rad 処方）を **多要素 chain explicit** で実行。
γ-1 (implicit) と γ-3 (explicit + TL) は対偶関係、解析解一致 + 数値挙動一致を期待。

| パラメータ | 値 |
|---|---|
| L_total | 10.0 mm |
| r | 0.5 mm |
| E | 130 GPa |
| ρ | 8.96e-9 ton/mm³ |
| n_elements | {1, 2, 4, 8, 16} |
| BC | 左端 6 DOF 固定、右端 θ_y(node n_nodes-1) 処方 |
| Loading | slow ramp 5·T_FE_1 + hold 5·T_FE_1 |
| Damping | 質量比例 Rayleigh α=2·ζ·ω_1, ζ=2 過減衰 |
| dt | 0.5·dt_critical_damped (Belytschko 6-178) |

**3 指標 AND gate**（γ-1 と同じ判定基準）:

1. `|u_x_tip|` ≈ circular arc 解 `R sin(θ) − L` (gate 10%)
2. `|u_z_tip|` ≈ circular arc 解 `R(1 − cos θ)` (gate 10%)
3. `L_chord_endpoints` ≈ `2R sin(θ/2)` (gate 10%)

実装本体無変更、新規 1 ファイル `work/beam_element_validation/51_gamma3_multi_element_explicit.py`
（~370 行）に explicit chain solver を inline 実装:

- `assemble_lumped_mass_diag`: 各要素 `timo_beam3d_lumped_mass_local` を global diag に accumulate
- `compute_natural_frequencies_chain`: u=0 で `K_aa φ = ω² M_aa φ` を解く（multi-element 拡張、`compute_natural_frequencies_fe` の chain 版）
- `solve_explicit_chain`: leap-frog Verlet on active DOFs + lumped mass + Rayleigh damping (TL モード固定、`update_reference` を呼ばない)

## 2. 実測結果

| n | err `|u_x|` [%] | err `|u_z|` [%] | err `L_chord` [%] | n_steps | KE/SE_proxy | gate |
|---:|---:|---:|---:|---:|---:|:-:|
|  1 | 24.9508 | 0.0938 | 0.0938 |  1,628 | 1.4e-27 | **FAIL** (期待通り) |
|  2 |  6.2346 | 0.0234 | 0.0234 |  3,118 | 2.0e-27 | **PASS** |
|  4 |  1.5585 | 0.0059 | 0.0059 |  5,941 | 3.2e-26 | **PASS** |
|  8 |  0.3896 | 0.0015 | 0.0015 | 16,339 | 3.6e-25 | **PASS** |
| 16 |  0.0974 | 0.0004 | 0.0004 | 57,594 | 8.2e-25 | **PASS** |

- **log-log slope of err(`|u_x|`) vs n (n≥2): −2.000**（理論値 O(1/n²) と完全一致、γ-1 と同値）
- **n=1 FAIL** は α-3 / γ-1 既知の **chord 長保存制約**（1 要素 CR は Hermite chord rotation
  α=θ/2 解を出す）による離散化誤差で **期待通り**、Phase γ で n_elements ↑ により消失
- **CR closed form 一致**（chord rotation φ_e=θ(e−1/2)/n の sum-to-product 解）は全 5 ケースで
  `|u_x|` / `|u_z|` / `L_chord` すべて **機械精度 0.000%**（実装が CR 多要素解析理論と完全整合）
- **polyline 長 = Σ L_elem**: 全ケース機械精度で 10.000 mm 保存（各要素 chord 長保存）
- **settle 残差 ||f_int_a||**: 5e-11〜9e-10 N（過減衰でほぼ完全 settle）
- **KE/SE 比**: 1e-27〜1e-25（quasi-static gate 完璧）

### 2.1 γ-1 (implicit) との数値一致

| n | γ-1 implicit err `|u_x|` [%] | γ-3 explicit + TL err `|u_x|` [%] | 差 |
|---:|---:|---:|---:|
|  1 | 24.95 | 24.95 | < 0.01% |
|  2 |  6.23 |  6.23 | < 0.01% |
|  4 |  1.56 |  1.56 | < 0.01% |
|  8 |  0.39 |  0.39 | < 0.01% |
| 16 |  0.097 |  0.097 | < 0.01% |

→ explicit + TL は implicit static と **同じ平衡解**に収束。foundation の対偶整合性確定。

## 3. 含意（次への道筋）

### 3.1 多要素 explicit + TL の foundation 健全性確定

Phase α (1 要素 implicit) → Phase β (1 要素 explicit) → Phase γ-1 (多要素 implicit) →
**Phase γ-3 (多要素 explicit + TL)** で **CR foundation の static / dynamic / multi-element /
explicit 全領域での健全性が定量実証**された。

status-394 で「Mode D (explicit + assembler + UL per step) のみ FAIL」だったのは UL update
タイミングの組合せ問題で、UL を呼ばない explicit + TL は **多要素規模でも機械精度級**で
動作する。

### 3.2 (z3) 19 本撚線適用への道筋

`status-394 §7.1` 「次セッション最優先」候補 (z3): explicit モード TL 固定 API 化 +
19 本撚線適用 への前提が整った:

1. ✅ 1 要素 explicit + TL: β-2 機械精度 (status-391)
2. ✅ assembler 経由 explicit + TL: Mode C 機械精度 (status-394)
3. ✅ **多要素 explicit + TL: γ-3 で n=2..16 で arc 解 O(1/n²) 収束** (本 status)
4. ⬜ 接触あり 2 本撚線（Phase δ、optional sanity check）
5. ⬜ 19 本撚線 + 接触 + explicit-TL 固定（本命検証、`explicit_ul_update_interval=0` で update_reference を一切呼ばない API 化）

→ 接触なし foundation は完全実証、残るは接触統合のみ。

### 3.3 Cosserat 路線

status-394 結論を裏付ける: γ-3 で多要素 explicit + TL が機械精度動作するため、SO(3) 直接
積分の Cosserat 移行は **absolute necessity ではない**。implicit 凍結方針下では Cosserat の
implicit + 大回転 plan B も scope 外（前セッション質疑応答の通り）。Cosserat は中期 plan B
としても優先度低下。

## 4. ゲート結果

| ゲート | 結果 | 備考 |
|---|---|---|
| `python work/beam_element_validation/51_gamma3_multi_element_explicit.py` | **n=1 FAIL（期待）/ n=2,4,8,16 PASS / slope=-2.000** | γ-1 と数値一致 |
| `pytest contact + math + time_integration + strand_bending_oscillation` | **743 passed 5 skipped** | status-394 と同数、新規 work スクリプトのみで本体無変更 |
| `python contracts/validate_process_contracts.py` | 全 24 検査 OK | C1〜C24 + O1〜O3 |
| `test_helical_3d_hermite` rel_err | 2.18e-07 維持 | status-356 で達成 |
| `ruff check work/beam_element_validation/` | All checks passed | |
| `ruff format --check work/beam_element_validation/` | already formatted | |

## 5. 達成確認マトリクス更新

`docs/status/verification_matrix.md` 更新:

- §2.3 Phase γ 表: γ-3 行を ⬜ → **✅** （n=2,4,8,16 全 PASS、slope=-2.000、γ-1 と数値一致）
- §8 達成済 ✅ 一覧に「Phase γ-3 (n=2,4,8,16) explicit + TL: status-395」+
  「Phase γ-3 O(1/n²) 収束 (slope=-2.000): status-395」+ 「assembler 経由 (implicit/explicit + TL): status-394」追加
- §8 未検証 ⬜ から「Phase γ-3 多要素 explicit」を削除（→ 達成済へ移行）

§5 STA2 撤回履歴: 本 status は新規撤回事例ではないため変更なし。

## 6. 次セッションへの引き継ぎ

### 6.1 最優先候補

- **候補 (z3) explicit モード TL 固定 API 化 + 19 本撚線適用**:
  `ContactFrictionInputData` に `explicit_ul_disable_update: bool = False` 追加（または
  `explicit_ul_update_interval=0` を「呼ばない」解釈に拡張）→ 19 本撚線 90° 曲げで
  frac=1.0 完走 + 解の精度 gate (5) 達成を試行。Mode C / γ-3 で foundation 機械精度
  実証済みなので、19 本でも有望。

### 6.2 副次候補

- **Phase δ 接触あり 2 本撚線** (`48_delta_2strand_contact.py`): (z3) 適用前の sanity check として有用。
  最小規模の接触統合で 3 指標一致を確認。
- **Phase γ-2 大 curvature 拡張**（θ=π/2、`50_gamma2_large_curvature.py`）: 1 ピッチ規模で
  「16 要素/ピッチ厳守」規範を再確認。
- **既存 validation の 3 指標 gate 化**（マトリクス §4 の 5 項目）

## 7. MCDD 脱法 pattern 自己点検

- **pattern 1（tol 緩和）**: 該当なし、status-388 透明性ルールの 10% gate を踏襲、γ-1 と同基準
- **pattern 5（既存テスト skip）**: 既存 743 test 全 pass、新規 work スクリプトのみ追加
- **pattern 6（骨格 status）**: 5 ケース全実測 + slope -2.000 で γ-1 と数値一致を実証、完結 status
- **pattern 7（数値丸め）**: `{:.4f}` (% 表示) は γ-1 と同フォーマット、生数値 `{:.6e}` で出力
- **pattern 8（根拠なき主張）**: γ-1 (implicit) との数値一致表で対偶整合性を視認可能化
- **pattern 10（TODO 先送り）**: 本 status は「explicit foundation 多要素確認」を完了、次は接触統合

## 8. 観察 — 開発運用上の発見

### 効果的

1. **「対偶対比」の威力**: γ-3 (explicit + TL) と γ-1 (implicit static) を同 BC + 同解析解で
   並べることで、**数値レベル一致**（< 0.01%）が確認でき、explicit 経路の信頼性を
   implicit と同じ基盤に置けた。今後の foundation 検証ではこの対偶対比を標準化したい。
2. **段階的 foundation building の有効性**: 1 要素 (β-2) → 1 要素 + assembler (Mode C) →
   多要素 + TL (γ-3) と段階を踏むことで、**どの組合せで何が起こっているか**が明確に局在化
   された。19 本適用前にこの段階を踏まなかった status-381〜387 は同じ問題を 7 status
   連続で誤判定していた。

### 今後の観察対象

- (z3) を 19 本撚線に適用したとき、foundation で機械精度だった挙動が **接触統合**でどう
  変化するか。Type D stall (status-344 K_c x/z カップリング不整合) は梁定式化と独立なので、
  explicit-TL では implicit AL n=2 (status-376 frac=0.5746) を超える可能性も低い可能性も
  ある — 実測必須。

## 9. 再現手順

```bash
git checkout claude/execute-status-todos-rMmcV

# Phase γ-3 多要素 explicit + TL 実機
uv run --extra dev python work/beam_element_validation/51_gamma3_multi_element_explicit.py \
    2>&1 | tee /tmp/gamma3_$(date +%s).log
# 期待: n=1 FAIL（24.95% chord 長保存制約）/ n=2,4,8,16 PASS / slope=-2.000

# 回帰テスト（実装本体無変更のため status-394 と同数期待）
uv run --extra dev pytest xkep_cae/contact/ xkep_cae/mathematics/ \
    xkep_cae/time_integration/ \
    xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py -q
# 期待: 743 passed, 5 skipped

# 契約検査
uv run --extra dev python contracts/validate_process_contracts.py
# 期待: 契約違反なし、条例違反なし
```

## 10. 引き継ぎチェックリスト

| 項目 | 状態 | 備考 |
|---|---|---|
| `51_gamma3_multi_element_explicit.py` 新設 | ✅ | 多要素 explicit + TL chain solver inline |
| n=2,4,8,16 で 3 指標 AND gate PASS | ✅ | 機械精度 〜 0.1% level |
| log-log slope = -2.000 | ✅ | 理論値 O(1/n²) と完全一致 |
| γ-1 implicit との数値一致 | ✅ | 全 n で < 0.01% 差 |
| n=1 FAIL（期待通り、chord 長保存制約） | ✅ | α-3 / γ-1 と整合 |
| status-395 作成 | ✅ | 本 status |
| status-index.md / README / roadmap 更新 | ✅ | エントリ追記 |
| `verification_matrix.md` §2.3 / §8 更新 | ✅ | γ-3 ✅ 化、達成済リスト追記 |
| 実装本体無変更 | ✅ | `xkep_cae/` 不変 |
| 回帰 743 passed 5 skipped | ✅ | status-394 と同数 |
| 全 24 契約検査 OK | ✅ | C1〜C24 + O1〜O3 |
| ruff check + format pass | ✅ | work/beam_element_validation/ |
| **次セッション最優先**: 候補 (z3) explicit-TL 固定 API 化 + 19 本撚線適用 | ⬜ | foundation 多要素確定で前提整う |

Phase A〜E / status-346〜395 の **46/N 完了**。
