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

### 6.1 段階的検証ロードマップ（ユーザー合意 5 段階）

implicit 完全凍結（解除想定なし）方針下、接触統合に向けた段階的検証 5 段階:

| status | scope | 検証対象 | 落ちる可能性 |
|:-:|---|---|---|
| **396 (次)** | (z3) explicit-TL 固定 API 化のみ（実機検証 scope 外） | API + 単体テスト + Default OFF 回帰 | 低（plumb のみ） |
| 397 | ε-1 = 3 strand helical + 接触なし + `disable=True` 適用 | ヘリカル初期 κ + 多 strand global assembler | **中-高**（直線 chain γ-3 から飛ぶ最初の難所） |
| 398 | ε-2 = 3 strand + 接触あり | 初の接触統合検証 | 中（K_c x/z カップリングは strand 数に依存しない可能性） |
| 399 | ε-3 = 7 strand + 接触あり | implicit baseline (status-301 frac=1.0) との対比 | 中-高 |
| 400 | ε-4 = 19 strand + 接触あり（本命） | MCDD 凍結解除条件 (2)(3)(5) 同時達成試行 | 高（Type D stall 領域、最終 gate） |

### 6.2 次 status (status-396) — explicit-TL 固定 API 化のみ

**設計確定**（本セッションでユーザー合意）: `explicit_ul_disable_update: bool = False`
**独立 field**。`explicit_ul_update_interval=0` 解釈拡張ではなく、意図明示で透明性高い。

**実装スコープ**:

- `ContactFrictionInputData.explicit_ul_disable_update: bool = False` 追加
  （default で既存挙動完全保持、status-383 で導入された `explicit_ul_update_interval`
   と独立、両者は AND で gate 評価）
- `StrandBendingOscillationConfig` 同 field + 3 経路 plumb-through
  （曲げ / 揺動 / free_end、status-383 配線を踏襲）
- `process.py` 主ループ update_reference 呼出箇所:
  `if not cfg.explicit_ul_disable_update and (_next_incr % interval == 0): ul.update_reference(...)`
- 単体テスト: `TestExplicitULDisableUpdate`（`_MockULAssembler` で `disable=True` 時に
  update_reference 呼出 0 回を直接計測、status-383 `TestExplicitULUpdateInterval` と
  並列配置で `xkep_cae/contact/solver/tests/test_process.py` 拡張）
- Default OFF 回帰: 743 passed 5 skipped 維持 + 全 24 契約検査 OK + 7 本 implicit
  frac=1.0 維持 + `test_helical_3d_hermite` rel_err=2.18e-07 維持

**scope 外**（status-397 ε-1 で実施）:

- 19 本 / 多 strand 実機検証
- ヘリカル初期 κ + 多 strand global assembler の foundation 検証
- 3 指標 AND gate 適用の実機ケース

本 status は API 化完結で documentation + 単体テスト + Default OFF 回帰のみ。

### 6.3 その次 (status-397) — ε-1 = 3 strand helical + 接触なし

`disable=True` を実機適用、`work/beam_hysteresis/` 系（既存 mesh / assembler / NR 経路）
で 3 本撚線曲げ揺動（接触なし、軽荷重 or 90° 曲げ）を実施。

**新たに出る要素**（γ-3 直線 chain で未検証、ε-1 で初検証）:

1. **初期 curvature 上の CR**: 直線 reference vs 曲線 reference で `R_0`（局所軸）の
   構築 + initial gap 等が変わる。`16 要素/ピッチ` 規範の妥当性は γ-3 small θ では未検証
2. **多 strand 並列 (no contact)**: 単独 strand と複数 strand の global assembler 振る舞い。
   `StrandBendingOscillationProcess` 経由で本物の mesh / connectivity を使うので、
   status-394 Mode C の 1 要素を超える
3. **端部 BC**: MPC + free_end_mode 等の組合せが explicit + TL モードで成立するか
   （implicit では status-280 で確立済）

**判定設計**:

- **ε-1 PASS** → ヘリカル foundation 健全、ε-2 (接触あり) へ進める
- **ε-1 FAIL** → ヘリカル初期 κ / 多 strand global / BC 経路で原因局在化、必要に応じて
  Phase δ (2 strand) に retreat して minimum 構成で原因切り分け

### 6.4 副次（並行可能）

- **Phase γ-2 大 curvature 拡張**（θ=π/2、`50_gamma2_large_curvature.py`）:
  γ-1/γ-3 は θ=0.15 rad の small-medium。full pitch (2π rad) レンジで
  「16 要素/ピッチ厳守」規範を再確認。
- **既存 validation の 3 指標 gate 化**（マトリクス §4 の 5 項目）

### 6.5 撤回済 / scope 外

- **(z2) Cosserat 路線**: implicit 凍結方針下で plan B も scope 外、status-394/395 で
  absolute necessity 消失。優先度低 (scope 外)。
- **凍結中 TODO**: 被膜圧縮 / リスタート / ファイバー梁キャリブレーション /
  7本ピッチ依存性 / 空間ブロック分離（MCDD 凍結解除後に再開）。

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
