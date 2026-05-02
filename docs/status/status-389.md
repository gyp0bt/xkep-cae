[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-389: 引き継ぎ — **梁要素 1 つから系統的再検証** Phase 計画策定（透明性ルール下での foundation re-validation）

**日付**: 2026-05-02
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12+5+17+11+6 passed
（status-388 と同数、本 status は引き継ぎ計画 — 実装変更なし）

## 概要

status-388 で透明性ルール（独立解析解 3 個以上同時一致）を施行した結果、
**「梁が 2.3x に伸びる非物理解で max\|u\| が 90° 解析解と偶然一致」**という
status-387 の判定が**完全に偽**であることが暴露された。

これは「**過去の単一指標一致による判定すべてが信頼できない可能性がある**」
ことを示唆する。MCDD 凍結解除条件 (5)「精度 < 10%」を含む全ての過去判定で、
3 指標 AND gate を適用していなかったため、status-387 と同じパターンの
「非物理解だが特定指標が偶然一致」が潜んでいる懸念がある。

ユーザー指示に従い、**最も基礎的な単位（梁要素 1 つ）から系統的に再検証**する
方針で次セッションに引き継ぐ。

## 1. 再検証の対象範囲

### 1.1 既存 CR Timoshenko 3D 梁要素

`xkep_cae/elements/_beam_cr.py`:

- `timo_beam3d_cr_internal_force`: CR 内力ベクトル
- `timo_beam3d_cr_tangent_analytical`: Battini & Pacoste (2002) 解析的接線剛性
- `timo_beam3d_lumped_mass_local`: 集中質量行列
- `timo_beam3d_mass_local`: 整合質量行列

これらが**梁要素 1 つの単純荷重ケース**で正しく動作するかを、3 指標 AND gate で
**implicit / explicit 両モード**で検証する。

### 1.2 ファイバー梁（status-326〜333 で実装）

`xkep_cae/elements/fiber/strand_beam.py`:

- `StrandFiberBeamProcess` + `ULCRFiberBeamAssembler`
- 1 要素レベルで弾性内力・接線・FD 自己整合性が status-329 で検証済（rel_err < 0.2%）
- ただし大変形領域・動的領域での 3 指標一致は未検証

### 1.3 接触なし条件での既存 validation スクリプト

`work/beam_hysteresis/`:

- `30_implicit_vs_explicit_7strand.py` 〜 `40_explicit_n_inc_sweep.py` の **11 本**
- すべて単一指標（max\|u\|）または 2 指標で gate 判定していた可能性
- 3 指標 AND gate での再評価が必要

## 2. Phase 計画

### Phase α — 1 要素静的検証（最優先、~2 セッション）

CR Timoshenko 3D 梁要素 1 つに、**4 つの基礎荷重ケース**を **implicit static** で
適用し、3 指標 AND gate で検証する。

#### α-1: 純軸引張（小変形、線形弾性）

**設定**:
- L=10mm、wire_radius=0.5mm（A=0.7854 mm²）、E=130 GPa、ν=0.3
- BC: 左端 fix（全 6 DOF）、右端: F_x=100 N（軸方向引張）
- n_elements=1（節点 2 個）

**解析解 4 指標**:

| 指標 | 解析値 |
|------|---:|
| `u_x_tip` | F·L / (E·A) = 100 · 10 / (130000 · 0.7854) = **0.0009793 mm** |
| `\|u_z_tip\|` (横変位) | **0** |
| `θ_y_tip` (回転) | **0** |
| `SE = (1/2) F u_x` | 0.5 · 100 · 0.0009793 = **0.04897 N·mm** |
| `L_arc` | 10 + u_x_tip = **10.000979 mm** |

5 指標のうち独立な 3 個以上が同時 PASS で達成。

#### α-2: 純粋曲げ small κ（線形弾性、UL 凍結問題なし）

**設定**:
- L=10mm、wire_radius=0.5mm（I=π/4·r⁴=0.04909 mm⁴、EI=6.382 N·mm²）
- BC: 左端 fix、右端: M_y=0.01 N·mm（end moment）
- n_elements=1

**解析解 4 指標**（Euler-Bernoulli linear）:

| 指標 | 解析値 |
|------|---:|
| `u_x_tip` (圧縮成分、二次小) | ≈ **0** mm |
| `\|u_z_tip\|` | M·L²/(2·EI) = 0.01·100/(2·6.382) = **0.07835 mm** |
| `θ_y_tip` | M·L/EI = 0.01·10/6.382 = **0.01567 rad** |
| `SE = (1/2) M θ` | 0.5·0.01·0.01567 = **7.84×10⁻⁵ N·mm** |
| `L_arc` | ≈ **10.000 mm** (二次小) |

#### α-3: 純粋曲げ large κ（CR 定式化 large rotation テスト）

**設定**:
- L=10mm、wire_radius=0.5mm
- BC: 左端 fix、右端: prescribed θ_y=0.15 rad ≈ 8.6°（κ=0.015 で 1 要素相当）
- n_elements=1

**解析解 4 指標**（円弧、R=L/θ=66.67mm）:

| 指標 | 解析値 |
|------|---:|
| `u_x_tip` | R·sin(θ) − L = 66.67·sin(0.15) − 10 = **−0.0626 mm** |
| `\|u_z_tip\|` | R·(1 − cos(θ)) = 66.67·0.01124 = **0.7493 mm** |
| `θ_y_tip` | **0.15 rad**（処方値） |
| `SE = (1/2) EI κ² L` | 0.5·6.382·0.000225·10 = **7.180×10⁻³ N·mm** |
| `L_arc` | **10.000 mm**（不伸長） |

#### α-4: 純せん断（先端横荷重）

**設定**:
- L=10mm、E=130GPa、G=E/(2(1+ν))=50 GPa、A_s=A·κ_s（せん断補正係数 κ_s=6/7）
- BC: 左端 fix、右端: F_z=10 N（横荷重）
- n_elements=1

**解析解 4 指標**（Timoshenko cantilever）:

| 指標 | 解析値 |
|------|---:|
| `u_x_tip` | ≈ **0** |
| `\|u_z_tip\|` | F·L³/(3·EI) + F·L/(G·A_s) = 522.6 + 0.000297 = **522.6 mm** |
| `θ_y_tip` | F·L²/(2·EI) = 78.40 rad |
| `M_base = F·L` | **100 N·mm**（基部反力モーメント） |
| `SE = (1/2)·F·u_z` | 0.5·10·522.6 = **2613 N·mm** |

注: u_z=522.6 mm はラージスケール（>50x L）で linear 領域外。**Timoshenko theory の
線形領域は u_z << L** のため、F=0.01 N（u_z=0.52mm）等まで小さくしてテストする。

### Phase β — 1 要素動的検証（α 完了後）

α で implicit static が 3 指標一致を確認できたら、explicit 動的を 1 要素で検証:

#### β-1: 1 要素 自由振動（impulse → free vibration）

**設定**:
- 1 要素、左端 fix、右端: 初期速度 v_z=1 mm/s（または impulse）
- 解析解: SDoF Timoshenko cantilever 第 1 モード
- f₁ = (1.875²/2π) · √(EI/(ρAL⁴)) ≈ ... 計算
- T₁ = 1/f₁

**解析解 3 指標**:

| 指標 | 検証 |
|------|---|
| 周期 T₁ | 観測周期との比 < 5% |
| エネルギー保存 | KE+SE 5 周期で減衰 < 10% |
| L_arc 保存 | <1% で振動中も維持 |

#### β-2: 1 要素 prescribed θ_y (slow ramp → quasi-static)

α-3 と同 BC を **explicit + slow ramp + sufficient damping** で実施。
implicit α-3 と 3 指標で 10% 以内に一致するかを確認。

→ ここで「**1 要素 explicit + UL は静的解析解と一致できるのか**」を foundation
レベルで確定する。**できないなら multi-element の status-388 結果は当然破綻**、
これが foundation 起源と確定し (z2) Cosserat 移行の根拠が確固化する。

### Phase γ — multi-element 検証（β 完了後）

n_elements ∈ {2, 4, 8, 16} で α-3 を再実施し convergence を確認。
**16 要素/ピッチ厳守**（CLAUDE.md「プログラムテスト」基準）の妥当性を再確認する。

### Phase δ — 接触あり 2 本撚線（γ 完了後）

接触なし multi-element が PASS した後、最小規模の接触系（2 本撚線、平行配置、軽荷重）
で 3 指標一致を確認。`status-335` の 2 本撚線 M-κ 観測スクリプトが基盤になる。

## 3. 既存テストの 3 指標 gate 化 TODO

以下の既存テストは単一指標 / 2 指標で判定しているため、3 指標 AND gate に拡張する
（パラメータ調整は不要、追加検証のみ）:

| ファイル | 現在の gate | 必要な追加 |
|---|---|---|
| `xkep_cae/elements/tests/test_assembler_process.py` | TBD（要確認） | u_x/u_z/SE/L_arc から 3+ |
| `xkep_cae/elements/fiber/tests/test_strand_beam_physics.py` | TBD | 同上 |
| `xkep_cae/numerical_tests/tests/test_beam_oscillation.py` | TBD | 周期 + KE+SE + L_arc |
| `xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py::TestHelical90DegBendPhysics` | tip displacement | u_x + u_z + L_arc 多重集合 |
| `work/beam_hysteresis/30〜40_*.py` | 各種（多くは max\|u\|） | u_x + u_z + L_arc + SE if reliable |

## 4. (z2) Cosserat 路線との関係

本 Phase α〜δ で **既存 CR Timoshenko + UL の foundation を確定**することで、
(z2) Cosserat 着手の判断材料を得る:

- **Phase α-3 で 1 要素 implicit が 3 指標 PASS → CR は static 規模で妥当**
  - foundation 健全、status-388 の破綻は multi-element / dynamic / 接触の組合せ起源
  - (z2) Cosserat の主目的は「**explicit + 大回転の robust 化**」に絞れる
- **Phase β-2 で 1 要素 explicit が 3 指標 FAIL → 1 要素レベルで explicit + UL 破綻**
  - status-388 の (z2) Cosserat 移行根拠が absolute に確定
  - Cosserat 実装は急務、Phase γ/δ は CR では skip して Cosserat 後に実施

## 5. 推奨セッション開始手順

```bash
# セッション開始時の確認
1. /home/user/xkep-cae/docs/status/status-389.md（本ファイル）を読む
2. CLAUDE.md の「妥当性テストの透明性ルール」セクションを再読
3. python contracts/validate_process_contracts.py 実行で current state 確認
4. work/beam_hysteresis/40_explicit_n_inc_sweep.py のヘッダ docstring を読み、
   3 指標 gate の参考にする
```

新規 validation スクリプト作成位置:

```
work/beam_element_validation/
  41_alpha1_axial_tension.py       (Phase α-1)
  42_alpha2_pure_bending_small.py  (Phase α-2)
  43_alpha3_pure_bending_large.py  (Phase α-3)
  44_alpha4_pure_shear.py          (Phase α-4)
  45_beta1_free_vibration.py       (Phase β-1)
  46_beta2_explicit_quasistatic.py (Phase β-2)
  ...
```

各スクリプトは status-388 §3 の表形式（3 指標 multiset + L_arc + 診断 SE）を
踏襲する。`work/beam_hysteresis/40_explicit_n_inc_sweep.py` の `_summarize()`
ヘルパを参考に共通化を検討。

## 6. 引き継ぎチェックリスト

| 項目 | 状態 | 備考 |
|------|---|---|
| status-387 撤回通知 | ✅ | 冒頭ヘッダ追加済 |
| 透明性ルール CLAUDE.md 追記 | ✅ | 「STA2 防止ルール」末尾 |
| status-388 訂正記録 | ✅ | 14 ケース実機データ含む |
| **status-389 引き継ぎ計画** | ✅ | 本ファイル |
| Phase α 実装 | ❌ | **次セッションで着手** |
| Phase β 実装 | ❌ | α 完了後 |
| (z2) Cosserat 開始判断 | 保留 | β-2 結果次第 |

## 7. MCDD 脱法 pattern 回避（自己点検）

- **pattern 1（tol 緩和）**: 3 指標 gate を 10% で固定、変更不可
- **pattern 5（既存テスト skip）**: 既存 743 test 全 pass、新規 Phase α 実装で追加
- **pattern 6（骨格 status）**: 本 status は documentation status だが、
  Phase 計画 + 解析解 + 既存 TODO + 推奨手順を具体化、骨格ではなく実行可能計画
- **pattern 8（根拠なき主張）**: status-388 §3 の 14 ケース実機データを根拠に
  「foundation 再検証必要」と判定
- **pattern 10（TODO 先送り）**: 本 status は計画策定で完結、Phase α 実装は
  次セッションで完結する単位として独立 scope

## 8. 観察 — status-387/388 で得た教訓

### 透明性ルールの威力

「3 指標同時一致」を要求するだけで、status-387 の「sweet spot 達成」誤判定が
**1 sweep（11 分）で実機反証された**。status-381 から status-387 まで、
**8 status 連続**で「max\|u\| 一致」を見て判定していたが、L_arc=234mm（梁が 2.3x 伸び）
の異常を見落とし続けていた。

→ **今後あらゆる validation で 3 指標を先行設計する**。実装前に分析解 3 個を
    紙の上で確定してから実装に着手する規範を確立すべき。

### Foundation 再検証の必要性

status-387 の誤判定は、上位レベル（19 本撚線 / 7 本撚線 / 単梁 90°）で起きた。
しかし 3 指標 gate がなかった以上、**より基礎的な層（1 梁要素レベル）でも同じ
パターンが潜在している可能性がある**:

- 例: ファイバー梁の弾性内力検証（status-329）で「先端変位 0.02% 一致」と
  報告されているが、SE / L_arc / 内部応力分布が同時 PASS かは未検証
- 例: `test_beam_oscillation.py` で「周期一致」のみ確認しているなら、振動中の
  L_arc 維持 / KE+SE 保存 が同時 PASS かは未検証

これらを Phase α/β で 3 指標 gate に拡張することで、foundation 健全性を確定
してから上位（multi-element / 接触 / dynamic）に進む。

### 次セッション開始者へのメッセージ

> **「3 指標 PASS なし」と「3 指標 PASS あり」は天と地の差**です。
> status-381 から status-387 まで「explicit 50% under」「explicit 系統的アンダー」
> 「sweet spot 達成」と判定が二転三転したのは、すべて 3 指標 gate がなかったため
> でした。Phase α-1 から、**実装前に解析解 3 個を確定**してください。
>
> 1 要素 implicit が α-1 で PASS しないなら、CR Timoshenko 実装にバグがある
> 可能性すら否定できません。**foundation を疑うことから始めてください**。
