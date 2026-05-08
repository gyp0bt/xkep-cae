[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-394: assembler / UL update_reference 1 要素再現実験 — 改修対象を **explicit + UL のみ** に局在化（4 モード比較で C/A/B PASS、D FAIL 99.85%）

**日付**: 2026-05-08
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12+5+17+11+6 passed（status-393 と同数、新規 work スクリプトのみで実装本体無変更）

## 概要

status-393 §6.1 で次セッション最優先候補として明示された **assembler / UL update_reference の
1 要素規模再現実験** を実施。Phase β-2 直接駆動（status-391, 機械精度 0.000%）と
Phase γ closed form 機械精度（status-392）の foundation 健全実証を踏まえ、status-381〜387
で発覚した精度問題（解析解の 50%〜99% アンダー）の改修対象を 1 要素規模で **再現** することで
**改修対象を局在化** する decisive 実験。

**結論**: 4 モード比較（implicit/explicit × TL/UL）で **3 モード（A/B/C）が機械精度 PASS、
Mode D（explicit + assembler + UL per step）のみ 99.85% / 96.14% アンダーで FAIL**。
status-381〜387 の精度問題は 1 要素規模で再現可能 + **改修対象は assembler の UL 機構そのもの
ではなく explicit との結合方式（update タイミング）** に局在することが定量実証された。

## 1. 実験設計

α-3 / β-2 と同 BC（θ_y=0.15 rad 処方）を 4 通りの実装パスで実行:

| モード | 実装パス | UL update_reference | 期待結果 | 仮説 |
|---|---|---|---|---|
| A | implicit + assembler | なし（TL モード） | PASS（直接駆動と機械精度一致） | assembler 自体は健全 |
| B | implicit + assembler | あり（増分ごと） | PASS or FAIL | UL が implicit でも問題か検証 |
| C | explicit + assembler | なし（TL モード） | PASS（β-2 直接駆動と機械精度一致） | assembler explicit パスは健全 |
| D | explicit + assembler | あり（毎 step） | FAIL（status-381〜387 再現） | UL update_reference + explicit が真の根本 |

**3 指標 AND gate**（α-3 と同じ Hermite 解、status-388 透明性ルール）:

1. `|u_x_tip|` ≈ Hermite 解 `L·(cos(θ/2) − 1) = -0.02811 mm` (gate 10%)
2. `|u_z_tip|` ≈ Hermite 解 `L·sin(θ/2) = 0.7493 mm` (gate 10%)
3. `L_chord` ≈ `L = 10.000 mm` (chord 保存 1 要素 CR、geometric, gate 10%)

実装本体は無変更、新規 1 ファイル `work/beam_element_validation/49_beta2_with_assembler_ul.py`
（~330 行）が `xkep_cae/elements/_beam_assembler.py::ULCRBeamAssembler` を直接 import して
4 モードを駆動する。共通条件: BC（左端固定、右端 θ_y 処方）、断面（L=10 mm、r=0.5 mm、
E=130 GPa、ν=0.3、ρ=8.96e-9 ton/mm³）、explicit は β-2 と同じ slow ramp（5T_1）+ hold（5T_1）+
質量比例減衰 ζ=2 過減衰。

## 2. 実測結果

| モード | 実装パス | u_x [mm] | u_z [mm] | L_chord [mm] | gate (3 AND) |
|---|---|---:|---:|---:|:-:|
| Hermite 解析解 | — | -2.811e-02 | +7.493e-01 | 10.0000 | — |
| **A** | implicit + assembler + TL | -2.811e-02 | -7.493e-01 | 10.0000 | **PASS** (0.000%) |
| **B** | implicit + assembler + UL（各 step） | -2.811e-02 | -7.493e-01 | 10.0000 | **PASS** (0.000%) |
| **C** | explicit + assembler + TL | -2.811e-02 | -7.493e-01 | 10.0000 | **PASS** (0.000%) |
| **D** | explicit + assembler + UL per step | -4.182e-05 | -2.895e-02 | 10.0000 | **FAIL** (99.85%/96.14%) |

（u_z 符号は実装の局所剛性 XZ 平面 `Ke[u_z, θ_y] = +6 EI/L²` 規約由来、kinematic 量は
`compare_abs=True` で吸収。Hermite 解と機械精度で絶対値一致）

実行ログ: `49_beta2_with_assembler_ul.py` から抜粋

```
[Mode D] explicit + assembler + UL (毎 step update_reference, status-381〜387 再現)
  [UL/per-step] dt=1.7068e-06 s, n_steps=1628, T_1=2.7787e-04 s, α=9.0447e+04
  |u_x_tip| -2.811e-02 → -4.182e-05 (99.85% 過小)
  |u_z_tip|  7.493e-01 → -2.895e-02 (96.14% 過小)
  L_chord   10.0000 → 10.0000 (chord 保存はそのまま)
```

## 3. 改修対象局在化の所見（最重要）

実測 A=PASS / B=PASS / C=PASS / D=FAIL のパターンは:

> **UL update_reference + explicit のみが問題**

であることを 1 要素規模で再現可能と確定。具体的に:

- **A=PASS** → assembler 自体は implicit static で健全（β-2 直接駆動と機械精度一致）
- **B=PASS** → UL update_reference は **implicit では正しく動作**（increment ごと reference 更新で
  rotation 累積されても 3 指標機械精度）。assembler の UL 機構そのものは健全。
- **C=PASS** → explicit + assembler の組合せも TL モード（update_reference 無し）なら健全。
  status-391 β-2 直接駆動と一致、explicit central diff + assembler-mass + assembler-internal_force の
  経路は健全。
- **D=FAIL** → explicit + assembler + UL per step **の組合せのみ** で 99.85% アンダー発生。
  status-381〜387 の精度問題（解析解 73.30 mm vs explicit 40.1 mm の 50% アンダー、status-382
  baseline 35.37 mm vs analytical 73.30 mm の 51% off など）は **1 要素規模で同パターンが再現**。

これは status-382 §3 で推定された「UL update_reference が各増分の dynamic lag を reference に
凍結 → `f_int(u_incr) ≈ 0`」が **正しい診断**であることを定量実証する。1 要素 12 DOF + θ_y
処方 + slow ramp という最小構成でも、毎 step `update_reference()` を呼び出すと変形が
reference に吸収されて elastic response がほぼ消失する。

### 物理的解釈

毎 step UL 更新の場合の挙動を追跡すると:

1. step k: `u_incr[10] = Δθ`（処方の incremental 角度 = 約 9e-5 rad/step）を設定
2. central diff で `u_incr[a]` を更新（ramp 速度に応じて非常に小さい変化）
3. `f_int(u_incr)` 計算 — `u_incr` がほぼゼロなので `f_int ≈ 0`
4. 加速度 `a = -f_int/M ≈ 0` + damping
5. step 末で `update_reference(u_incr)` → reference を更新、`u_incr ← 0` リセット

→ 4. の **加速度がほぼ常にゼロ**（damping のみ作用）→ deformation が elastic energy に
変換されない。reference が処方値に追従するだけで、elastic restoring force が発達しない。

これは status-382 §3 の解析と完全整合: UL の本来の目的（large rotation increment を分割
処理）は **静的解析のステップ間更新**であり、explicit dynamics の **time step 内更新**とは
スケールが 5 桁以上異なる。

## 4. 含意 — 改修パスの絞り込み

**(z2) Cosserat 路線は不要**: status-391 / 392 で foundation 健全実証 + 本 status で
explicit + UL（per step）のみが問題と確定。SO(3) 直接積分への移行は MCDD 凍結解除に
不要であり、改修候補は次の 3 つに絞れる:

### 候補 (z3) explicit モードでは update_reference を呼ばない（TL モード固定）

最も simple な解決策: `solver_mode="explicit"` のとき UL `update_reference()` を一切呼ばず、
TL モード固定で運用する。Mode C で機械精度 PASS が実証済み。

- **長所**: 実装最小、Mode C 通り動作確定
- **短所**: 撚線規模で UL の本来目的（参照配置更新で large rotation 蓄積を吸収）が effective
  かは別途検証必要。1 要素は累積 0.15 rad で TL 問題なし、撚線規模でヘリックス 1 ピッチ
  ≈ 360° は TL でも element-level に展開すれば各要素 0.15 rad × n_pitch 程度で線形化レンジ。
- **実装**: status-383 の `explicit_ul_update_interval` を `0` 解釈で「呼ばない」化、または
  別 field `explicit_ul_disable_update`。

### 候補 (z4) explicit + UL with sub-cycling（最有力）

Time step 内で multiple Verlet sub-step を走らせ、最後に **1 度だけ** `update_reference` を
呼ぶ。status-382 §6 候補 (q2) として recorded。1 要素規模では `update_per_step=False`
（hold 終了後に 1 度のみ update_reference）で**ほぼ同等の結果**になるはずで、追加実験で
確認可能（本 status では時間都合で省略）。

### 候補 (z5) explicit ramp 速度を物理 T1 ベースに縮小

mass scaling β を上げるのではなく、ramp 終了時刻 t_ramp を T_1 倍化（status-386 §6 で
方向間違いと判明、再考必要）。**z3 + z5** 組合せで撚線規模 explicit を robust 化する筋。

## 5. ゲート結果

| ゲート | 結果 | 備考 |
|---|---|---|
| `python work/beam_element_validation/49_beta2_with_assembler_ul.py` | **A/B/C PASS / D FAIL** | 想定通り、改修対象局在化 |
| `pytest contact + math + time_integration + strand_bending_oscillation` | **743 passed 5 skipped** | status-393 と同数、新規 work スクリプトのみで本体無変更 |
| `python contracts/validate_process_contracts.py` | 全 24 検査 OK | C1〜C24 + O1〜O3 |
| `test_helical_3d_hermite` rel_err | 2.18e-07 維持 | status-356 で達成 |
| `ruff check work/beam_element_validation/` | All checks passed | |
| `ruff format --check work/beam_element_validation/` | already formatted | |

## 6. 達成確認マトリクス更新

`docs/status/verification_matrix.md` §3「上位層改修対象」を更新:

- 「assembler 経由」行: ⬜ → ✅（implicit/explicit + TL は機械精度 PASS）
- 「UL `update_reference`」行: ⬜ → 🟡 部分達成（implicit では PASS、explicit per step で FAIL）

§5「STA2 撤回履歴」: 本 status は新規撤回事例ではないため変更なし。

## 7. 次セッションへの引き継ぎ

### 7.1 最優先候補（変更）

status-393 で「assembler / UL 1 要素再現実験」が最優先だったが本 status で完了。
新たな最優先は:

- **候補 (z3) explicit モード TL 固定の API 化**:
  `ContactFrictionInputData.explicit_ul_update_interval=0` で update_reference を一切
  呼ばない解釈を追加 + 19 本撚線で frac=1.0 完走 + 解の精度 gate (5) 達成を試行。
  status-383 で interval=10/20 は爆発したが、これは update を**遅延しすぎ**た結果。
  TL 固定（呼ばない）は Mode C で機械精度実証済みなので、撚線規模でも有望。

### 7.2 副次候補

- **候補 (z4) sub-cycling**: 1 要素規模で `update_per_step=False`（ramp 終了後 1 度のみ
  update_reference）の追加 mode E を試行 → 機械精度なら Mode C に近い実装で OK
- **Phase δ 接触あり 2 本撚線** (`48_delta_2strand_contact.py`)
- **Phase γ-2 大 curvature 拡張**（θ=π/2、`50_gamma2_large_curvature.py`）
- **既存 validation の 3 指標 gate 化**（マトリクス §4 の 5 項目）

## 8. MCDD 脱法 pattern 自己点検

- **pattern 1（tol 緩和）**: 該当なし、新規 gate threshold は status-388 透明性ルールの
  10% を踏襲
- **pattern 5（既存テスト skip）**: 既存 743 test 全 pass、新規 work スクリプトのみ追加
- **pattern 6（骨格 status）**: 4 モード実機 PASS/FAIL で改修対象を**定量的に局在化**、
  骨格ではなく完結 status
- **pattern 7（数値丸め）**: 全数値を `{:.6e}` で出力、`{:5.2f}` 丸めなし
- **pattern 8（根拠なき主張）**: 全モードの数値 + 解析解との相対誤差を併記、視認可能
- **pattern 10（TODO 先送り）**: 本 status は「次セッションの decisive 実験」を完了、
  改修候補（z3）に絞り込み済

## 9. 観察 — 開発運用上の発見

### 効果的

1. **「1 要素規模で問題を再現する」設計の威力**: status-381〜387 で 7 status 連続失敗した
   精度問題が、本 status の 1 要素 12 DOF + 4 モード対比で **30 秒以内に決定的に局在化**。
   問題の規模を最小化 + 4 モード対比で「どの組合せで失敗するか」を AND/OR 表で
   追跡可能にする手法は今後の MCDD 系統的検証で標準化すべき。
2. **マトリクス §3 「実装パス × UL on/off」の独立軸記録**: 本 status のように 4 モード
   結果を 1 表で並べることで、改修対象が「UL × explicit」のみと一目で確定。マトリクス
   設計（status-393）の「達成・未達・未検証を独立に記録」が運用面で機能している実例。

### 今後の観察対象

- 1 要素規模で再現できた問題が撚線規模でも同じ修正で解決するかは未検証。Mode C
  （TL 固定 explicit）は 1 要素では PASS だが、19 本 + 接触 + UL 由来の large rotation
  累積で TL がスケールするかは status-394 後継で検証。

## 10. 再現手順

```bash
git checkout claude/execute-status-todos-rMmcV

# 1 要素 4 モード再現実験
uv run --extra dev python work/beam_element_validation/49_beta2_with_assembler_ul.py \
    2>&1 | tee /tmp/beta2_assembler_ul_$(date +%s).log
# 期待: Mode A/B/C PASS、Mode D FAIL（u_x 99.85% / u_z 96.14% under）

# 回帰テスト（実装本体無変更のため status-393 と同数期待）
uv run --extra dev pytest xkep_cae/contact/ xkep_cae/mathematics/ \
    xkep_cae/time_integration/ \
    xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py -q
# 期待: 743 passed, 5 skipped

# 契約検査
uv run --extra dev python contracts/validate_process_contracts.py
# 期待: 契約違反なし、条例違反なし
```

## 11. 引き継ぎチェックリスト

| 項目 | 状態 | 備考 |
|---|---|---|
| `49_beta2_with_assembler_ul.py` 新設 | ✅ | 1 要素 4 モード対比 |
| Mode A/B/C 機械精度 PASS | ✅ | assembler / UL implicit / explicit-TL は健全 |
| Mode D 99.85% FAIL | ✅ | status-381〜387 の 1 要素規模再現確定 |
| 改修対象を **explicit + UL per step** に局在化 | ✅ | (z2) Cosserat 不要 |
| status-394 作成 | ✅ | 本 status |
| status-index.md / README / roadmap 更新 | ✅ | エントリ追記 |
| `verification_matrix.md` §3 更新 | ✅ | 上位層改修対象の状態更新 |
| 実装本体無変更 | ✅ | `xkep_cae/` 不変 |
| 回帰 743 passed 5 skipped | ✅ | status-393 と同数 |
| 全 24 契約検査 OK | ✅ | C1〜C24 + O1〜O3 |
| ruff check + format pass | ✅ | work/beam_element_validation/ |
| **次セッション最優先**: 候補 (z3) explicit-TL 固定 API 化 + 19 本撚線適用 | ⬜ | マトリクス §3 で追跡 |

Phase A〜E / status-346〜394 の **45/N 完了**。
