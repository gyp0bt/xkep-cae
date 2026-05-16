# status-353: 数理台帳訂正 — K_mat,ndir ≡ K_geo の同一性確立 + Phase C-3 再定義

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-19
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25（変動なし）

## 概要

status-352 の中断スナップショットで提示した「Phase C-3 前提の数理的疑義」に対し、
**選択肢 A（数理台帳 §4 訂正）を実施**。`docs/math/03_huber_contact_penalty.md`
の §3 / §3.1 / §4 / §5 / §8 を訂正し、`KcGeoStiffnessProcess` が
$-p_n\,\partial\hat{\boldsymbol{n}}/\partial\boldsymbol{u}$ のペア局所形を担う
ことを明示。「`K_mat_ndir` 独立項の追加」という当初 Phase C-3 計画
（status-346〜352）を**撤回**し、5 項 `TermExpansionContract` で完結とする。

19本撚線 Type D stall（`mat_only` rel_err mean=44%, comp_x max=98%、status-344）
の真の原因候補は、ペア局所の完全項ではなく **`K_hermite_adj` の隣接ノード拡張が
mat-only（`w_mat * nn` のみ、`I_nn` 不含）であること**（status-295 で意図的に
除外）に再設定する。

## 成果物

### 数理台帳訂正

| ファイル | 変更箇所 | 内容 |
|---------|---------|------|
| `docs/math/03_huber_contact_penalty.md` | §3 [#eq-kc-full-decomposition] | 6 項 → **5 項**完全分解。`K_mat,ndir` 項を削除し、`-K_geo` が法線方向感度を表現することを明示 |
| 同 | §3 [#eq-kc-pair-block] | ペア局所形に隣接拡張・closest 項を追加表記、A-A 同側の導出（$\partial r/\partial x_j^{(A)} = -c_j I$ → $\partial p_n/\partial x = +c_j w_{\mathrm{mat}}\hat n^\top$、$\partial \hat n/\partial x = -(c_j/d) P_\perp$）を併記 |
| 同 | §3.1 表 | 6 行 → **5 行**。`K_mat_ndir` 行を削除、`K_geo` 行に「= $p_n\,\partial\hat n/\partial u$ と同値」の説明追加 |
| 同 | §4 [sec-ndir] | **全面書き直し**。`K_geo` ≡ `−p_n · ∂n̂/∂u` のペア局所同一性を導出付きで明示。「過去の主張 vs 訂正」表で status-346〜352 の誤診断を整理 |
| 同 | §5 [eq-kmat] / [sym-kmat] | 「完全項を持つ場合（K_mat,nn + K_mat,ndir）」→「`K_mat,nn − K_geo` の対称性」に訂正、対称性議論をペア局所行列で再構成 |
| 同 | §7 末尾 | `K_hermite_adj` の mat-only 制約を明示し、x/z 残差候補に再設定 |
| 同 | §8 trace 表 | `[#eq-kc-full-decomposition]` の「K_mat,ndir 欠落」記述を削除、`[#eq-dn-du]` を「既実装（`KcGeoStiffnessProcess`）」に更新 |
| 同 | 関連 status | status-348 を「初版（K_mat,ndir 独立項記述、後に訂正）」、status-352 を「数理検証」、status-353 を「本訂正」として整理 |

### コード側コメント訂正（実装変更なし）

| ファイル | 行 | 変更 |
|---------|----|------|
| `xkep_cae/contact/contact_force/strategy.py` | 31〜68 | `_K_C_TERM_EXPANSION_CONTRACT` 上部コメント: 「6 項のうち 5 項を確立、K_mat_ndir は status-352 本命修正で追加予定」→「5 項で完結。K_mat_ndir は K_geo と同一のため未追加（status-353）」 |
| 同 | 858〜880 | `KcNormalStiffnessProcess` docstring: 「K_mat_ndir (status-352 本命) は未実装」→「K_mat_ndir は KcGeoStiffnessProcess と同一のため追加 Process は不要（status-353）」 |
| 同 | 1013〜1031 | `KcGeoStiffnessProcess` docstring: 「本項は法線方向感度 $-p_n\\,\\partial\\hat n/\\partial u$ のペア局所形そのものであり、$1/d$ 因子は内在項」を追記 |

### 新規実装

**なし**（status-352 の「実装 Process 追加は二重計上リスク」判定に従い、ledger と
docstring のみ訂正）。

## 数理的論拠（status-352 の再掲 + 完成）

### A-A 同側ペア局所導出

接触力 $\boldsymbol{f}_c^{(A,i)} = -c_i p_n \hat{\boldsymbol{n}}$、残差規約
$\boldsymbol{K}_c = \partial(-\boldsymbol{f}_c)/\partial \boldsymbol{u}
= \partial(c_i p_n \hat{\boldsymbol{n}})/\partial \boldsymbol{x}_j$。

$\partial \boldsymbol{r}/\partial \boldsymbol{x}_j^{(A)} = -c_j \boldsymbol{I}$ から:

- $\partial(-g)/\partial \boldsymbol{x}_j^{(A)} = -\partial d/\partial \boldsymbol{x}_j^{(A)} = +c_j \hat{\boldsymbol{n}}^\top$
- $\partial p_n/\partial \boldsymbol{x}_j^{(A)} = w_{\mathrm{mat}} \cdot c_j \hat{\boldsymbol{n}}^\top$（$w_{\mathrm{mat}} = (\mathrm{d}p_n/\mathrm{d}x) k_{\mathrm{pen}}$）
- $\partial \hat{\boldsymbol{n}}/\partial \boldsymbol{x}_j^{(A)} = -(c_j/d) \boldsymbol{P}_\perp$

連鎖律:

$$
[\boldsymbol{K}_c]^{(i,j)}_{AA,ab}
= c_i [(p_n)_{,b} \hat n_a + p_n (\hat n_a)_{,b}]
= c_i c_j [w_{\mathrm{mat}} \hat n_a \hat n_b - (p_n/d) [P_\perp]_{ab}]
$$

これは `strategy.py:1595` の
`K_3x3 = w_mat[:, None, None] * nn - w_geo[:, None, None] * I_nn`
（$w_{\mathrm{geo}} = p_n/d$）と完全一致。

### 結論

`K_geo` の項そのものが $-p_n\,\partial\hat{\boldsymbol{n}}/\partial \boldsymbol{u}$ の
ペア局所形であり、$1/d$ 因子は $\hat{\boldsymbol{n}} = \boldsymbol{r}/d$ の
内在項として現れる（旧 status-348 の §4 表「$d$ で割らない」は誤り）。
新規 `KcNormalDirectionStiffnessProcess` の追加は二重計上で、`test_kc_component_fd.py`
19 件の rel_err 倍化 → fail を引き起こす。撤回が正解。

## status-344 残差の再原因仮説（status-353 更新）

| 仮説 | 根拠 | 優先度 |
|------|------|--------|
| **A**: `K_hermite_adj` mat-only（`w_mat * nn` のみ、`I_nn` 隣接拡張なし） | `strategy.py:1596-1598` 既存コメント「隣接ノード変位→s追従により法線変化はほぼ相殺されるが、ギャップ変化(n⊗n項)は維持される」（status-295 設計）。**真の数理対象**: `−p_n · ∂n̂/∂u` の隣接ノード成分は `−c_j (p_n/d) P_⊥ · ∂r_adj/∂u_adj` で、これが mat-only から欠落 | **★最有力** |
| B: `K_closest` の (s,t) 摂動残差（status-351 で抽出済み） | dpn_ds 自体の FD 整合性は status-351 で個別 Process FD 未取得 | 中 |
| C: 凍結 active set 近似（NR 反復内で freeze） | status-258 の「K_c 自体は正確、94-100% 不整合は活性集合変化」観察と整合 | 中 |
| D: K_st の摩擦接線剛性（μ > 0 で適用される項） | status-344 の `mat_only` (μ=0 相当) で 44% 残るため D は副次的 | 低 |

**次セッション推奨アクション**:

1. 仮説 A 検証: `KcHermiteNonlocalStiffnessProcess` に `I_nn` 隣接拡張を追加した
   実験ブランチで `test_kc_component_fd.py` 19 件の `mat_only` rel_err を再計測。
   下がれば仮説 A 確定 → Phase C-3 を「`K_hermite_adj` フル項拡張」に再定義。
2. status-344 の 19本撚線 K_c 成分分解 FD 診断 log（`scripts/run_19_strand_kc_fd.py`
   等）を再実行し、mat-only と full-K_c の rel_err 差分を comp 別に取得。

## ゲート

- ✅ `python contracts/validate_process_contracts.py`: 契約違反 0、条例違反 0
- ✅ `pytest xkep_cae/contact/`: **421 passed, 5 skipped**（既存と同数）
- ✅ `pytest xkep_cae/contact/contact_force/tests/test_kc_component_fd.py xkep_cae/verify/tests/test_kc_component_fd.py xkep_cae/mathematics/tests/`: **114 passed**
- ✅ **7本撚線曲げ揺動回帰テスト（弱曲げ・接触未活性スモーク）**: `tests/numerical_tests/test_strand_bending_convergence.py::TestStrandBendingConvergence::test_strand_bending_oscillation_converges` **frac=1.0000, incr=51, cutback=4, 10.20s, bending_angle=0.100 rad (≈5.73°), max contact F=0.0 完走**
  - **条件**: κ=0.001 1/mm, n_pitches=1.0（→ bending_angle=κ·L=0.1 rad≈5.73°）, Hertz α=1.5, free_end_mode=True, exclude_same_strand=True, μ=0.15
  - **注意**: **90° 曲げではなく、また実行中に `max contact F=0.0`（接触力はゼロ、全 51 incr で `active=0`）**。本テストはソルバーパイプライン整合性のスモーク回帰であり、**接触活性下の x/z カップリング検証ではない**。K_c FD 検証は別途 `test_kc_component_fd.py` 19 件（通常 pytest 経路）で実施済み
- ✅ **7本撚線 Hertz 型完走テスト（同条件、確認用）**: `test_strand_bending_full_completion_hertz` **frac=1.0000, incr=51, cutback=4, 9.96s, 接触未活性**
- ⚠️ **接触あり 90° 曲げ回帰は本 status 未実行**: status-298/299 系の重量テスト（κ=0.04, n_pitches=多、揺動 ±48mm）は標準 pytest 経路ではなく `scripts/` 系の手動実行。本訂正は数理台帳・コメントのみでコード数値挙動は無変更のため、重量回帰は status-354（Phase C-3 再定義、`K_hermite_adj` フル項拡張）で実施予定
- ✅ `ruff check xkep_cae/ tests/`: All checks passed
- ✅ `ruff format --check xkep_cae/ tests/`: 191 files already formatted

## 計画ロードマップ（CLAUDE.md / status 更新）

`CLAUDE.md` の MCDD ロードマップを以下に訂正:

| 旧（status-352 まで） | 新（status-353 以降） |
|------|------|
| status-352（Phase C-3）: `KcNormalDirectionStiffness` ★x/z 本命修正 | status-352: 計画書ロスト記録 + Phase C-3 前提検証（中断スナップショット） |
| status-353（Phase C-4）: K_mat_nn / K_st 再配線 + 項別 FD 整合性 | **status-353: 数理台帳訂正（K_mat,ndir ≡ K_geo 確立、5 項完結化）** |
| — | **status-354（Phase C-3 再定義）: 仮説 A 検証 — `K_hermite_adj` フル項拡張（`I_nn` 隣接拡張）+ 19本 K_c FD 再計測** |
| status-354-355（Phase D）: `DiagnosticDispatcherProcess` 等 | status-355-356（Phase D）: 同上、1 status 後ろ倒し |
| status-356（Phase E） | status-357（Phase E） |

## 関連 status

- status-289〜296: K_c FD 不整合の最初期追跡（s_unclamped、StJacobian、frozen-m、K_c_adj mat-only）
- status-295: `K_c_adj` mat-only 確立（隣接ノード幾何項 `I_nn` を意図的に除外、本 status の仮説 A の根)
- status-342〜345: 19本撚線 K_c 成分分解 FD 診断 + report 精度バグ訂正
- status-346〜347: MCDD Phase A
- status-348〜349: Phase B（数理台帳整備、当初 K_mat,ndir 独立項記述、本 status で訂正）
- status-350〜351: Phase C-1/C-2（5 項独立 Process 化）
- status-352: 計画書ロスト記録 + Phase C-3 前提の数理疑義提示（中断スナップショット）
- **status-353（本 status）**: 数理台帳訂正、Phase C-3 を「`K_hermite_adj` フル項拡張」に再定義

## コミット（予定）

1. `docs(math): 03 章 \u00a73/\u00a74/\u00a75/\u00a78 訂正 — K_mat,ndir \u2261 K_geo の同一性確立`
2. `refactor(contact): strategy.py コメント訂正 — K_mat_ndir 独立追加計画を撤回（status-353）`
3. `docs(status): status-353 + README/status-index/roadmap/CLAUDE.md 更新（Phase C-3 再定義）`
