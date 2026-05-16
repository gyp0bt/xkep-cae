# status-354: Phase C-3 再定義実験 — K_hermite_adj フル項拡張の仮説 A 反証

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-20
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25（変動なし）

## 概要

status-353 で再定義された Phase C-3 仮説 A
（「`KcHermiteNonlocalStiffnessProcess` に `-w_geo * I_nn` の隣接ノード項を
追加すると 19 本撚線 `mat_only` rel_err mean=44% / comp_x max=98% が改善する」）を
**直接実験して反証**した。gate テスト `test_kc_component_fd.py::test_helical_3d_hermite`
の rel_err が **1.795% → 38.49% に 21 倍悪化**し、MCDD 脱法パターン 5
（「既存テスト 12 件を skip/xfail で pass させる」）に該当するため、変更を
全て revert し mat-only 形態（status-295）を継続する。

本 status は仮説 A の反証を数理的解釈付きで記録し、Phase C-3 を **Phase C-3'
再々定義** として hypothesis B/C/D に再配分する。計画骨格のみの status では
なく、**定量的な実験結果 + 数理台帳 03 章 §7 の訂正**を成果物として提示する。

## 実験内容

### 変更コード（revert 済）

`xkep_cae/contact/contact_force/strategy.py:941` の
`KcHermiteNonlocalStiffnessProcess.process()` 内部:

```python
# 旧（mat-only、status-295）
K_3x3_mat = w_mat[:, None, None] * nn

# 仮説 A 実装（フル項）
K_3x3_full = w_mat[:, None, None] * nn - w_geo[:, None, None] * I_nn
```

隣接ノード DOF 列への assembly は同一、`K_3x3` だけを入れ替えた。

### 測定結果

**ゲートテスト**: `tests/contact/contact_force/test_kc_component_fd.py::TestKcComponentFD::test_helical_3d_hermite`

| 構成 | `‖K_c‖` | `‖FD_Kc‖` | `‖diff‖` | rel_err | comp_x | comp_y | comp_z |
|------|---------|-----------|----------|---------|--------|--------|--------|
| mat-only（現行、ベースライン） | 5.488e+03 | 5.489e+03 | 9.852e+01 | **1.795%** | 45.0% | 44.8% | 77.3% |
| `mat + I_nn`（仮説 A 実装） | — | — | — | **38.49%** | — | — | — |

comp_x/y/z はそれぞれの方向の不整合シェア。

ログ: `/tmp/log-full-$(date +%s).log`

### 反証の数理的解釈

仮説 A のペア局所数理（status-353 §4 導出）は正しいが、隣接ノード拡張では
**2 経路の応答**を考慮する必要がある:

1. **直接経路**: $\boldsymbol{x}_{\mathrm{adj}}$ 摂動 → Hermite 接線
   $\boldsymbol{m}$ 変化 → $\boldsymbol{p}_A(s)$ 直接変化。
   これが `α_adj = H10*s * dm_ext` で計算される `K_hermite_adj` の寄与。
2. **s-tracking 補償**: min-distance 射影で $(s, t)$ が再決定され、
   closest point が restore される。`I_nn`（法線直交）方向の変動はこの
   s-tracking で**ほぼ相殺**される。

FD（有限差分）は (1) + (2) の合成応答を捕捉する。Process は (1) のみを
解析的に計算する。`w_mat * nn`（ギャップ方向 n⊗n）の寄与は s-tracking では
ほぼ補償されないため mat-only が FD と整合するが、`I_nn` 方向を追加すると
(2) で相殺される分も Process 側に載ってしまい FD との乖離が拡大する。

status-295 当時のコメント `strategy.py:1612-1614`:

> K_3x3_mat: 隣接ノード用。幾何剛性(I-n⊗n)を除外（status-295）。
> 理由: 隣接ノード変位→s追従により法線変化はほぼ相殺されるが、
> ギャップ変化(n⊗n項)は維持される。

は本 status の実験結果で定量的に検証された（38% vs 1.8%）。

## 成果物

### ドキュメント訂正

| ファイル | 変更内容 |
|---------|---------|
| `docs/math/03_huber_contact_penalty.md` §7 [#eq-hermite-pA] | 仮説 A 反証結果を表形式で追記、2 経路応答の数理的解釈を記述、Phase C-3 再々定義（hypothesis B/C/D）を明示 |
| 同 §3.1 表 `K_hermite_adj` 行 | 「I_nn 項は未拡張、x/z 残差候補に再設定」→「I_nn フル項拡張の仮説 A は実験反証済み — §7 参照」 |
| 同 §4 末尾 | 仮説 A 反証と Phase C-3 再々定義のポインタ |
| 同 §8 trace 表 [#eq-hermite-pA] 行 | 「x/z 残差候補」→「status-354 で仮説 A 反証 — mat-only 継続（§7 参照）」 |
| 同 関連 status | status-354 行を追加（「§7/§3.1/§4/§8 仲裁追記」） |

### コード側コメント訂正（実装変更なし）

| ファイル | 行 | 変更 |
|---------|----|------|
| `xkep_cae/contact/contact_force/strategy.py` | 31〜70 | `_K_C_TERM_EXPANSION_CONTRACT` モジュールコメントに status-354 仲裁セクションを追加、`description` に「K_hermite_adj の I_nn フル項拡張は実験反証済み（status-354）」を追記 |
| 同 | 904〜912 | `KcHermiteNonlocalStiffnessProcess` docstring に status-354 仲裁記述を追加（「status-354 で仮説 A = I_nn フル項拡張が反証されたため継続」） |
| 同 | 941〜944 | インラインコメントで仮説 A の実測結果（1.795% → 38.49%）を記録 |

### 新規実装

**なし**（仮説 A 反証により mat-only 継続）。Phase C-3 再々定義の hypothesis B/C/D
探索は次セッション（status-355）に引き継ぐ。

## Phase C-3 再々定義（hypothesis B/C/D）

19 本撚線 Type D stall (`mat_only` rel_err mean=44%, comp_x max=98%、status-344) の
真の原因候補を以下に再配分:

| 仮説 | 概要 | 優先度 |
|------|------|--------|
| **B**（新最有力） | `KcClosestPointStiffnessProcess` を隣接ノード DOF 列にも拡張。$\partial s/\partial \boldsymbol{u}_{\mathrm{adj}}$ / $\partial t/\partial \boldsymbol{u}_{\mathrm{adj}}$ を組み込み、仮説 A 反証で判明した s-tracking 補償経路 (2) を解析的に実装 | **★最有力** |
| C | NR 反復内の active set 凍結近似の弛緩（status-258 観察、「K_c 自体は正確、94-100% 不整合は活性集合変化」と整合） | 中 |
| D | 摩擦 `K_st` 隣接拡張での類似項不整合。仮説 A 反証と同様の 2 経路問題が発生する可能性 | 中 |

status-353 の仮説 A が reject されたことで仮説 B が論理的に昇格した（A が
扱っていた I_nn 経路の補償は s-tracking 経路で先に扱うべき）。

### 次セッション推奨アクション（status-355）

1. **仮説 B 検証**: `KcClosestPointStiffnessProcess` の `_assemble_term_coo`
   を精査し、`adj_node_map` 列への拡張が可能か評価。19 本撚線 K_c FD 再計測
   で `mat_geo`（K_st 追加前）rel_err が mean=44% → どの程度下がるか測定。
2. 仮説 B 実装前に、`test_kc_component_fd.py::test_helical_3d_hermite` の
   現状 comp_z 77% 不整合が s-tracking 不足由来であることを
   `KcClosestPointStiffnessProcess` の単体 Process FD で切り分け。

## ゲート

- ✅ `ruff check xkep_cae/ tests/`: All checks passed
- ✅ `ruff format --check xkep_cae/ tests/`: 191 files already formatted
- ✅ `pytest xkep_cae/contact/`: **421 passed, 5 skipped**（既存同数）
- ✅ `pytest xkep_cae/contact/contact_force/tests/test_kc_component_fd.py`: **7 passed** （`test_helical_3d_hermite` rel_err=1.795% 合格）
- ✅ **7 本撚線曲げ揺動回帰テスト**（弱曲げスモーク、status-353 同条件）:
  `tests/numerical_tests/test_strand_bending_convergence.py::TestStrandBendingConvergence::test_strand_bending_oscillation_converges`
  **frac=1.0000, incr=51, cutback=4, 10.54s, bending_angle=0.100 rad, max contact F=0.0 完走**
- ⚠️ **接触あり 90° 曲げ（status-298/299 系重量回帰）は本 status 未実行**:
  コメント / 台帳訂正および mat-only revert のみで数値挙動は status-353 と
  同一。重量回帰は仮説 B 実装後の status-355 以降で実施予定。

## 計画ロードマップ更新

| 旧（status-353 計画） | 新（status-354 以降） |
|------|------|
| status-354（Phase C-3 再定義）: `K_hermite_adj` フル項拡張 + 19 本 K_c FD 再計測 | **status-354: 仮説 A 実験反証 + Phase C-3 再々定義（hypothesis B/C/D）** |
| — | **status-355（Phase C-3' 着手）: 仮説 B = `KcClosestPointStiffnessProcess` の隣接ノード拡張** |
| status-355-356（Phase D）: `DiagnosticDispatcherProcess` 等 | status-356-357（Phase D）: 1 status 後ろ倒し |
| status-357（Phase E） | status-358（Phase E） |

MCDD Phase A〜E の進捗: **7/12 完了**（status-353 時点と同値、本 status は
実験反証という定量結果を伴う Phase C-3 の再々定義として追加）。

## 関連 status

- status-295: K_c_adj mat-only 確立（隣接ノード幾何項 `I_nn` を意図的に除外、本 status で根拠を実測再確認）
- status-342〜345: 19 本撚線 K_c 成分分解 FD 診断 + report 精度バグ訂正
- status-346〜351: MCDD Phase A〜C-2
- status-352: 計画書ロスト記録 + Phase C-3 前提疑義
- status-353: 数理台帳訂正（K_mat,ndir ≡ K_geo）+ Phase C-3 再定義（`K_hermite_adj` フル項拡張）
- **status-354（本 status）**: 仮説 A 実験反証 + Phase C-3 再々定義（hypothesis B/C/D）

## 懸念事項・引き継ぎメモ

- 本 status は**仮説 A の反証**という不成功実験の記録だが、MCDD 規範の
  「脱法実装禁止パターン 10: status ファイルに TODO として積む」には該当しない。
  定量的な実験結果（rel_err 数値）と数理的解釈（2 経路応答の分析）を伴う。
- 仮説 B の実装コストは仮説 A より高い可能性がある（`KcClosestPointStiffnessProcess`
  の既存 implementation が `active_idx` の 4 ノードのみを扱う設計のため、
  隣接ノード対応には `_st_jacobian.py` の入出力拡張が必要）。status-355 で
  まず FD 単体診断を取って効果見込みを確認してから実装着手を推奨。
- `test_helical_3d_hermite` の comp_z 77% 不整合は仮説 B 検証の direct target。
  現 rel_err 1.795% の内訳を理解することが仮説 B の効果見積もりの鍵。
- 計画書 `deep-wiggling-seal.md` は status-352 以降永久ロスト。MCDD 脱法パターン
  10 項は CLAUDE.md に転記済み、本 status も 10 項を遵守。

## コミット（予定）

1. `test(contact): Phase C-3 仮説 A 実験反証 + docstring 更新 — K_hermite_adj mat-only 継続（status-354）`
2. `docs(math): 03 章 §7/§3.1/§4/§8 仲裁追記 — 仮説 A 反証を数理台帳に反映（status-354）`
3. `docs(status): status-354 + README/status-index/roadmap/CLAUDE.md 更新（Phase C-3 再々定義）`
