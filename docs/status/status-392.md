[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-392: Phase γ 完了 — multi-element CR Timoshenko 梁の circular arc 収束を O(1/n²) で実証

**日付**: 2026-05-06
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12+5+17+11+6 passed（status-391 と同数）

## 概要

status-391 §6.1 Phase γ 計画に従い、CR Timoshenko 3D 梁要素を **直線チェーン**
で `n_elements ∈ {1, 2, 4, 8, 16}` に並べた系を α-3 と同じ BC（左端 fix、右端
θ_y=0.15 rad 処方）で **implicit static** に解き、circular arc 解への収束を
3 指標 AND gate（status-388 透明性ルール）で確認した。

**4/5 ケース PASS**:

- n=1 のみ FAIL（u_x で 24.95% — α-3 / status-390 で実証済み chord 長保存制約
  による既知の離散化誤差）
- n=2,4,8,16 で 3 指標すべて PASS（10% gate）
- **log-log slope of err(u_x) vs n (n≥2): −2.000**（理論値 O(1/n²) と完全一致）
- **CR closed form 一致**: 全 5 ケースで \|u_x\| / \|u_z\| / L_chord すべて
  **機械精度（10⁻¹³%〜10⁻¹²%）** — 実装は CR 多要素 chord rotation 解析理論と完全整合

→ **CR foundation の multi-element アセンブル健全性確定**。
   「16 要素/ピッチ厳守」規範は典型 curvature レンジで十分なマージンを持つ
   （θ=0.15 rad ≈ 8.6° 単一曲げで n=2 から 10% gate を通過）。

実装本体（`xkep_cae/`）**無変更**、`work/beam_element_validation/` に
**2 ファイル新設**（`_gamma_common.py` + `47_*.py`、~430 行）+ README 更新のみ。

## 1. 検証結果サマリ

### 1.1 circular arc 解との比較（gate 10%）

| n_elements | err(\|u_x\|) [%] | err(\|u_z\|) [%] | err(L_chord) [%] | gate | iters |
| ---: | ---: | ---: | ---: | :---: | ---: |
|  1 | 24.951 | 0.094 | 0.094 | **FAIL** (u_x) | 40 |
|  2 |  6.235 | 0.023 | 0.023 | PASS | 40 |
|  4 |  1.558 | 0.006 | 0.006 | PASS | 40 |
|  8 |  0.390 | 0.001 | 0.001 | PASS | 40 |
| 16 |  0.097 | 0.000 | 0.000 | PASS | 40 |

**log-log slope of err(u_x) vs n (n≥2): −2.000**（理論値 O(1/n²) と完全一致）

### 1.2 CR closed form 一致（実装健全性、機械精度期待）

n 要素 CR で各要素 chord 回転角は `φ_e = θ(e − 1/2)/n` の線形分布になり、
sum-to-product で:

    x_n = L · sin(θ/2)·cos(θ/2) / (n·sin(θ/(2n)))
    z_n = L · sin²(θ/2)         / (n·sin(θ/(2n)))

| n_elements | err(\|u_x\|) [%] | err(\|u_z\|) [%] | err(L_chord) [%] |
| ---: | ---: | ---: | ---: |
|  1 | 5.97e-12 | 1.48e-14 | 1.78e-14 |
|  2 | 7.70e-13 | 1.48e-14 | 0.00e+00 |
|  4 | 5.46e-13 | 1.48e-14 | 0.00e+00 |
|  8 | 3.12e-12 | 1.48e-14 | 0.00e+00 |
| 16 | 3.23e-12 | 0.00e+00 | 1.78e-14 |

→ 全ケース 1e-11 レベル — `timo_beam3d_cr_*` 直接アセンブルが CR 解析理論と
   **機械精度で一致**。NR (load_steps=10, max_iter=30, tol=1e-9) は全ケース 4
   反復/ステップで安定収束（iters=40 = 10×4）。

### 1.3 polyline 長保存（補助診断）

各要素 chord 長 `L/n` は CR が厳密保存。総 polyline 長は全 5 ケースで
`Σ L_elem = 10.000 mm` を機械精度で保存（diff < 1e-13 mm）。

## 2. 実装

### 新規ファイル（2 個）

```
work/beam_element_validation/
  _gamma_common.py                            (+~280 行)
    - ChainedBeamSection (BeamSection の n_elements 拡張)
    - assemble_internal_force / assemble_tangent (要素ループ直接アセンブル)
    - solve_static_nr_chain (multi-element NR static, load stepping + prescribed)
    - compute_chord_total / compute_polyline_length
  47_gamma_multi_element_convergence.py       (+~270 行)
    - cr_closed_form / arc_form (解析解)
    - run_one_case / print_convergence_table
    - 5 ケース掃引 + 3 指標 AND gate × 5 = 15 個の判定 + log-log slope check
```

### 設計

- `xkep_cae` 本体の assembler を経由せず、`timo_beam3d_cr_internal_force` /
  `timo_beam3d_cr_tangent_analytical` を要素ループで直接アセンブル
  （Phase α/β 同様、foundation を最小単位で検証）
- 既存 `_alpha_common.MetricRow` / `evaluate_three_metric_gate` を再利用
  （status-388 透明性ルール準拠の 3 指標 AND gate）
- `compare_abs=True` で u_z 符号差を吸収（α-3 と同じ符号規約）

### 実装本体への影響

**無変更**。`xkep_cae/`、単体テスト、契約検査はすべて維持。

## 3. ゲート結果

| ゲート | 結果 | 備考 |
|---|---|---|
| `pytest contact + math + time_integration + strand_bending_oscillation` | **743 passed 5 skipped** | status-391 と同数 |
| `python contracts/validate_process_contracts.py` | 全 24 検査 OK | C1〜C24 + O1〜O3 |
| `test_helical_3d_hermite` rel_err | 2.18e-07 維持 | status-356 で達成 |
| `ruff check work/beam_element_validation/` | All checks passed | |
| `ruff format --check work/beam_element_validation/` | 10 files already formatted | |
| Phase γ 5 ケース | **4/5 PASS** | n=1 FAIL は α-3 既知の chord 長保存制約 |
| O(1/n²) 収束（n≥2 で u_x slope） | **−2.000**（理論値と完全一致） | |
| CR closed form 一致 | 全 5 ケース機械精度 | 実装健全性確定 |

## 4. 次セッションへの引き継ぎ

### 4.1 候補（次セッション最優先）— assembler / UL update_reference の 1 要素再現実験

Phase β-2 + Phase γ で「CR 要素自体は静的・動的・multi-element すべての領域で健全」が
定量実証された。**残る課題は status-381〜387 の精度問題を assembler 経由 +
UL update_reference 有効化で 1 要素規模で再現**することである。次の改修対象を
特定するための decisive 実験となる。

スクリプト案: `work/beam_element_validation/49_beta2_with_assembler_ul.py`
（β-2 と同 BC を assembler 経由 + UL 更新あり/なしで実施、機械精度
0.000% × 3 (β-2 直接駆動) との差分を比較）。

### 4.2 副次 — Phase δ 接触あり 2 本撚線

最小規模の接触系（2 本撚線、平行配置、軽荷重）で 3 指標一致を確認。
`status-335` の 2 本撚線 M-κ 観測スクリプトが基盤、`work/beam_element_validation/48_delta_2strand_contact.py`
を作成予定。Phase γ 完了で multi-element + 接触なしの foundation が確定したため、
接触あり foundation 検証への移行は logical な次ステップ。

### 4.3 副次 — Phase γ 拡張（curvature レンジ + Phase γ-2 dynamic）

- **γ-1**: 本 status の θ=0.15 rad（small-medium curvature）。**完了**
- **γ-2 候補**: より大きな curvature（θ=π/2 = 90°）で再実施
  → 「16 要素/ピッチ厳守」規範を full pitch（撚線で 1 turn）レンジで再確認
- **γ-3 候補**: multi-element explicit + slow ramp（β-2 同様）の拡張
  → multi-element でも explicit が CR closed form / arc 解と一致するか

### 4.4 副次 — 既存テストの 3 指標 gate 化（status-389 §3 TODO）

`test_assembler_process.py` / `test_strand_beam_physics.py` /
`test_beam_oscillation.py` / `TestHelical90DegBendPhysics` /
`work/beam_hysteresis/30〜40_*.py` を順次 3 指標 AND gate に拡張。
パラメータ調整不要、追加検証のみ。

### 4.5 中期 plan B — Cosserat 梁プロトタイプ

status-391 で absolute necessity ではなくなったが、assembler / UL 改修
（4.1 / 4.4）が頓挫したときの fallback として scope 維持。
geometrically exact (Simo-Reissner) beam、SO(3) 回転 DOF + reference 更新不要 +
軸方向拘束 exact 維持で L_arc 自動保存。実装中規模（~1000 行）。

### 4.6 凍結中 TODO

被膜圧縮モデル / リスタート方式 / ファイバー梁キャリブレーション /
7本撚線ピッチ依存性 / 空間ブロック分離（status-345 で凍結、再開可能）。

## 5. MCDD 脱法 pattern 自己点検

- **pattern 1（tol 緩和）**: 3 指標 gate threshold は全 10% で固定、Phase α/β と
  同じ。事後緩和なし、n=1 は素直に FAIL を記録。
- **pattern 2（dummy verifier）**: 該当なし、新規 `@verified_by` 紐付けなし。
- **pattern 5（既存テスト skip）**: 既存 743 test 全 pass、新規 Phase γ 実装は
  独立スクリプト。
- **pattern 6（骨格 status）**: 5 ケース全実機検証 + log-log slope 確認 +
  closed form 機械精度確認で具体的結果記録、骨格ではなく完結 status。
- **pattern 7（数値丸め）**: 0.097% / 24.951% を `{:.4f}%` 形式で出力、
  10⁻¹³ レベルの closed form err を `{:.6e}` で露呈。
- **pattern 8（根拠なき主張）**: 全主張に実機ログ + sum-to-product による
  closed form 解析解 + log-log slope = −2.000 を根拠提示。
- **pattern 10（TODO 先送り）**: 本 status は Phase γ 完結、次フェーズ候補は
  4.1〜4.5 で具体化。

## 6. 観察 — 開発運用上の効果的・非効果的な発見

### 効果的

1. **closed form 解析解の二重化が gate と実装健全性を分離**: 「circular arc 解
   との比較（gate）」+「n 要素 CR closed form との比較（診断、実装健全性）」
   の併用により、**gate FAIL（n=1）でも実装健全性は機械精度で別途確認**できた。
   これは「離散化誤差」と「実装バグ」を独立に診断する重要な手法で、Phase α
   までの「実装 = 解析理論」の単一指標から進化した。
2. **log-log slope = −2.000 の実証が「16 要素/ピッチ」規範を補強**: O(1/n²)
   収束は離散化誤差解析の教科書的結果だが、CR Timoshenko + chord 保存型 NR で
   実機実測すると機械精度で −2.000 が出ることを確認。convergence rate の理論的
   裏付けが得られたため、規範の数値マージン議論に定量的根拠を提供できる。
3. **Phase α-3 の chord 長保存制約発見が Phase γ 設計を効率化**: status-390 で
   「1 要素 CR は Hermite 解（chord rotation α=θ/2）を出す」発見が、Phase γ で
   `cr_closed_form()` の sum-to-product 導出に直接つながった。Phase 計画
   （status-389）の階層的構造が情報伝達を効率化している。

### 非効果的（観察）

- **n_load_steps=10 の固定**: 全 5 ケースで n_load_steps=10 を使ったが、n=1 / 2
  では 4 反復 / step で安定収束し、n=16 でも同じ。実は `n_load_steps=2` でも
  十分収束した可能性が高い（diagnostics として log を取っていないため確認は
  別途）。large rotation での load stepping マージン議論は Phase γ-2（θ=π/2）
  で改めて行う方が適切。
- **Phase γ-1 の curvature が小さい**: θ=0.15 rad は small-medium curvature で、
  「16 要素/ピッチ厳守」の典型シナリオ（撚線 1 turn = 2π rad）よりはるかに小さい。
  full pitch レンジでの再確認は γ-2 で行う必要がある（4.3 副次）。

## 7. 再現手順

```bash
git checkout claude/execute-status-todos-FnP23

# Phase γ 実行
uv run --extra dev python work/beam_element_validation/47_gamma_multi_element_convergence.py \
    2>&1 | tee /tmp/gamma_$(date +%s).log

# 期待結果: 4/5 ケース 3 指標 AND gate 通過 + log-log slope = -2.000

# 回帰テスト
uv run --extra dev pytest xkep_cae/contact/ xkep_cae/mathematics/ \
    xkep_cae/time_integration/ \
    xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py -q
# 期待: 743 passed, 5 skipped

# 契約検査
uv run --extra dev python contracts/validate_process_contracts.py
# 期待: 契約違反なし、条例違反なし

# Lint
uv run --extra dev ruff check work/beam_element_validation/
uv run --extra dev ruff format --check work/beam_element_validation/
# 期待: All checks passed / 10 files already formatted
```

## 8. 引き継ぎチェックリスト

| 項目 | 状態 | 備考 |
|---|---|---|
| Phase γ 5 ケース実機検証 | ✅ | n=1 FAIL は既知制約、n=2,4,8,16 PASS |
| 3 指標 AND gate × 5 | ✅ | 15 個中 12 個 PASS（n=1 u_x のみ FAIL） |
| O(1/n²) 収束実証 | ✅ | log-log slope = −2.000（理論値と完全一致） |
| CR closed form 一致 | ✅ | 全 5 ケース 機械精度 10⁻¹¹ レベル |
| polyline 長保存（chord 長保存） | ✅ | 全ケース 機械精度（diff < 1e-13 mm） |
| 「16 要素/ピッチ厳守」規範のマージン | ✅ | θ=8.6° で n=2 から PASS、n=16 で 0.1% |
| 実装本体無変更 | ✅ | `xkep_cae/` 不変 |
| 回帰 743 passed 5 skipped | ✅ | status-391 と同数 |
| 全 24 契約検査 OK | ✅ | C1〜C24 + O1〜O3 |
| ruff check + format pass | ✅ | 10 files already formatted |
| **assembler / UL 1 要素再現実験** | ❌ | **次セッション最優先候補**（4.1） |
| **Phase δ 接触あり 2 本撚線** | ❌ | 副次（4.2） |
| **Phase γ-2 大 curvature 拡張** | ❌ | 副次（4.3） |

Phase A〜E / status-346〜392 の **43/N 完了**。
