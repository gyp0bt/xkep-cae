# status-357: 19 本撚線 K_c FD 再計測 + C5 解消 + C18/C19 契約検査追加（Phase E 着手）

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-21
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25（変動なし）

## 概要

status-356 で Phase C-3' 実装（仮説 A + 仮説 B 同時導入で
`test_helical_3d_hermite` rel_err 1.795% → 2.18e-07 達成）した結果を受け、
引継ぎで積まれた 4 TODO を実施:

1. **19 本撚線 K_c FD 再計測**（`work/beam_hysteresis/13_kc_component_fd_19strand.py`）
   — Phase C-3' 実装効果を「gate テスト外・実機規模」で定量測定
2. **19 本 Type D stall 再試行** — `frac=0.48 → 1.0` 目標
3. **接触 90° 曲げ重量回帰テスト**（status-298/299 系列の回帰なし確認）
4. **C18/C19 契約検査追加**（Phase E 着手、MCDD 脱法 pattern 2/4 対策）

副産物として、status-356 実装で混入していた **C5 違反**（`KcHermiteNonlocalStiffnessProcess.process()`
内で `_batch_dm_ext_coeffs` を `HuberContactForceProcess` のクラスメソッドとして
直接参照、`uses` 未宣言）を **モジュール関数化**で解消。

## 1. 19 本撚線 K_c FD 再計測（TODO 1）

### 実行コマンド

```bash
uv run python work/beam_hysteresis/13_kc_component_fd_19strand.py \
  2>&1 | tee /tmp/kc_component_fd_19strand_$(date +%s).log
```

### 結果（ログ `/tmp/kc_component_fd_19strand_1776779031.log`）

| 指標 | status-344（Phase C-3' 前） | status-357（Phase C-3' 後） | 判定 |
|------|---------------------------|---------------------------|------|
| frac_completed | 0.4839 | **0.3739**（-22.7%） | **退化** |
| n_increments | - | 177 | - |
| n_cutbacks | - | 19 | - |
| elapsed | - | 368.48 s | - |
| K_c 診断レコード件数 | 183 | 147 | - |
| **mat_only rel_err** mean | 0.44 | **0.508** | **+15% 悪化** |
| mat_only rel_err median | 0.43 | 0.426 | ほぼ同等 |
| mat_only 最良シェア | 183/183（100%） | 145/147（98.6%） | ほぼ同等 |
| share_geo mean | 1.02e-3 | 2.0e-3 | 同オーダー |
| share_st mean | 0.536 | 0.536 | 同一 |
| comp_x max（mat_only） | 0.98 | 0.772 | 若干改善 |

### 解釈

status-356 の gate テスト `test_helical_3d_hermite` では rel_err が
**1.795% → 2.18e-07（FD 機械精度）** に到達した一方、19 本撚線の
**広範な接触対（147 件）に対しては改善効果が確認できず、むしろ平均 rel_err が
15% 悪化**した。status-356 の 2 経路相殺定理（§7 数理台帳）は
`test_helical_3d_hermite` のような **単一接触対 + 安定 active 集合** 状況下では
解析的に成立するが、19 本撚線 Type D stall の断面では:

- NR Type 分布: `D+E:67%, E:28%, C:3%`（大半が active 集合変動）
- Last 10 反復: `D+E:9, E:1`（stall 直前も active 変動が継続）
- 414 ペアうち active=34, sliding=380

という **active 集合振動が支配的な領域**で FD 整合性が取れない。これは
Phase C-3' が解決したのは「active 集合固定下の解析的 K_c」であり、
status-352〜354 で提示されていた **仮説 C（active 集合振動）は未解決**であることを
改めて裏付ける。

### 次アクションへの示唆

- status-356 数理台帳 §7.1 の 2 経路解析は active 集合固定下で厳密
- 19 本 Type D stall の主要因は Phase C-3' の対象外領域（active 振動）
- 次は **仮説 C（active 集合振動対策）** を立案し status-358 以降で検証すべき
  候補: (a) smoothing_delta の遷移帯広げ、(b) active 判定の履歴平滑化、
  (c) line search 強化

## 2. 19 本 Type D stall 再試行（TODO 2）

TODO 1 の実行結果そのものが本 TODO の再試行データを兼ねる。
`frac=0.48 → 1.0` の目標に対し **`frac=0.3739` に退化**した。
Phase C-3' の gate テスト上の数値改善が実機規模には波及しないことが確認された。

## 3. 接触 90° 曲げ重量回帰テスト（TODO 3）

### 7 本撚線曲げ揺動回帰

```bash
uv run --with pytest python -m pytest \
  tests/numerical_tests/test_strand_bending_convergence.py::TestStrandBendingConvergence::test_strand_bending_oscillation_converges \
  -xvs 2>&1 | tee /tmp/regression_7strand_$(date +%s).log
```

結果: `1 passed in 10.69s`（status-356 の 10.18s と同等、回帰なし）

### 接触全体回帰

```bash
uv run --with pytest python -m pytest xkep_cae/contact/ \
  2>&1 | tee /tmp/contact_regression_$(date +%s).log
```

結果: **421 passed, 5 skipped in 39.65s**（status-356 と完全同数）

両テスト群とも status-356 ベースラインに対し回帰なし、Phase C-3' の
実装は後方互換性を保つ。

## 4. C18/C19 契約検査追加（TODO 4、Phase E 着手）

### C18: `@verified_by` 紐付け検査

`severity in {"hard", "nightly"}` の `MathematicalContract` ごとに
`ProcessContractRegistry` に検証 Process が紐付いているか静的検証。
`contracts/validate_process_contracts.py` に `check_c18_verified_by_binding()`
を追加。MCDD 脱法 pattern 2 **「dummy VerifyProcess を `@verified_by` に
紐付けて C18 を通す」** の前段防御（実体の dummy 検出は `ProcessContractRegistry.bind_verifier()`
の AST 検査で status-347 済み）。

### C19: `TermExpansionContract.providers` 実在検査

`TermExpansionContract.providers` の Process クラス名が `ProcessRegistry`
に実在するか静的検証。`check_c19_term_providers_exist()` を追加。
MCDD 脱法 pattern 4 **「`KcNormalDirectionStiffnessProcess` を rename で済ませる」**
対策 — 列挙した Process 名で typo や未登録をコミット前に捕捉。

### 5 Process への `@verified_by` バインディング

C18 を実際に有効化するため、`_K_C_TERM_EXPANSION_CONTRACT` を `contracts`
ClassVar に持つ 5 つの term-provider Process に `@verified_by("K_c_term_expansion",
ContactKcComponentFDDiagnosticProcess)` デコレータを付与:

- `ContactForceStStiffnessProcess`（status-350 新設）
- `KcNormalStiffnessProcess`（status-350 新設）
- `KcHermiteNonlocalStiffnessProcess`（status-351 新設、status-356 で `I_nn` 項追加）
- `KcGeoStiffnessProcess`（status-350 新設）
- `KcClosestPointStiffnessProcess`（status-351 新設、status-356 で active×adj 拡張）

`ContactKcComponentFDDiagnosticProcess`（`xkep_cae/verify/kc_component_fd.py`）は
4 組み合わせ FD 相対誤差で K_c = K_mat - K_geo + K_st 全項を検証する
既存 SolverProcess で、status-344/345 以来 19 本撚線 FD 診断で実働している。

### gate 検証

```
$ uv run python contracts/validate_process_contracts.py
...
--- C18: @verified_by 紐付け検査 ---
  OK
--- C19: TermExpansionContract.providers 実在 ---
  OK
...
契約違反なし、条例違反なし
```

## 5. C5 違反解消（副次作業）

status-356 で `_batch_dm_ext_coeffs` を `HuberContactForceProcess`
クラスの staticmethod として新設したが、`KcHermiteNonlocalStiffnessProcess.process()`
内で `HuberContactForceProcess._batch_dm_ext_coeffs(...)` を直接参照していたため
**C5 違反**（`KcHermiteNonlocalStiffnessProcess.uses` 未宣言で `HuberContactForceProcess`
依存）が発生していた（status-356 リリース時点では C18 未実装のため検出漏れ）。

### 修正内容

`_batch_dm_ext_coeffs` を **モジュールレベル関数**に昇格
（`xkep_cae/contact/contact_force/strategy.py:271`）。

```python
def _batch_dm_ext_coeffs(
    adj_node_counts: np.ndarray, nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """status-356: 共通ヘルパに抽出、status-357: C5 解消のためモジュール関数化."""
    ...
```

3 箇所の呼び出し点:
- `ContactForceStStiffnessProcess._process_batch_term`（strategy.py:552）
- `KcHermiteNonlocalStiffnessProcess.process`（strategy.py:1122）
- `HuberContactForceProcess.tangent`（strategy.py:1821）

いずれも `HuberContactForceProcess._batch_dm_ext_coeffs(...)` →
`_batch_dm_ext_coeffs(...)` に変更。Process クラス間依存が消滅し C5 遵守。

## gate（総合）

```
$ uv run python contracts/validate_process_contracts.py
契約違反なし、条例違反なし （C18 / C19 追加含む全 19 検査 OK）

$ uv run --with pytest python -m pytest xkep_cae/contact/
421 passed, 5 skipped in 39.65s

$ uv run --with pytest python -m pytest \
  tests/numerical_tests/test_strand_bending_convergence.py::...oscillation_converges
1 passed in 10.69s
```

## Phase A〜E 進捗更新

Phase A〜E / status-346〜358 の **9/13 完了**（status-357 で
Phase C-3' の実機規模検証 + Phase E 一部着手）。

- [x] Phase A-1（status-346）: `MathematicalContract` 型 5 種
- [x] Phase A-2（status-347）: `ProcessContractRegistry` + `@verified_by`
- [x] Phase B-1（status-348）: `docs/math/03_huber_contact_penalty.md`
- [x] Phase B-2（status-349）: 6 章 / 55 アンカー + `equation_index.py` + C15 拡張
- [x] Phase C-1（status-350）: `KcNormal` / `KcGeo` 抽出
- [x] Phase C-2（status-351）: `KcHermiteNonlocal` / `KcClosestPoint` 抽出
- [x] 数理台帳訂正（status-353）: `K_mat,ndir ≡ K_geo` 同一性
- [x] Phase C-3 仮説 A 実験（status-354）: 単独フル項化は反証
- [x] Phase C-3' 診断（status-355）: active×adj ブロック局在化
- [x] Phase C-3' 実装（status-356）: 2 経路同時導入で FD 機械精度
- [x] **status-357（本 status）**: 19 本 FD 再計測 + 回帰 + C5 解消 + C18/C19
- [ ] Phase E 完成（status-358）: 仮説 C（active 集合振動）立案 + 19 本 frac=1.0 再試行

## 引継ぎ（status-358 へ）

### 最優先 TODO

1. **仮説 C（active 集合振動対策）立案 + 19 本 frac=1.0 完走** — Phase C-3' は
   active 集合固定下で厳密だが、19 本 Type D stall は active 振動支配領域
   （D+E:67%, E:28%）。候補:
   - (a) `smoothing_delta` の遷移帯広げ（status-260 の δ=1000 延長）
   - (b) active 判定の履歴平滑化（low-pass）
   - (c) line search 強化（NR 反復途中の過剰 active flip 抑制）
   - (d) 接触凍結モード（status-284）の 19 本適用

2. **MCDD Phase E 仕上げ** — C20 以降の候補検討（`TermExpansionContract.term_names`
   と Process 出力の形状一致の静的検査、等）

### 凍結解除条件（MCDD 完了）

Phase E 完成 + 19 本 frac=1.0 完走 + `mat_only` rel_err < 1e-2 を満たした時点で
以下の凍結 TODO を再開可能:

- 7本撚線ピッチ依存性検証（p=50/100/200）
- ファイバー梁キャリブレーション
- リスタート解析方式
- 被膜圧縮モデル改善
- 空間ブロック分離 / ペアクラスタリング

## テスト数

459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25（status-356 と同数、
C18/C19 は既存 `validate_process_contracts.py` への追加のため独立テスト追加なし）

## コミット構成

本 status の変更は feature 単位で 3 コミットに分割:

1. `fix(contact): C5 解消 + 5 term-provider に @verified_by 付与（status-357 Phase E）`
   — `strategy.py` 一括（`_batch_dm_ext_coeffs` モジュール関数化 + インポート追加 + 5 Process への `@verified_by`）
2. `feat(contracts): C18 / C19 静的検査追加（status-357 Phase E 着手）`
3. `docs(status): status-357 + README/status-index/roadmap/CLAUDE.md 更新`
