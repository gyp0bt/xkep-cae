# status-356: Phase C-3' 仮説 B 実装 — K_closest/K_st 隣接拡張 + K_hermite_adj フル項化で FD 機械精度一致

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-21
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25（変動なし）

## 概要

status-355 で定量化された Phase C-3' 仮説 B 目標
（`test_helical_3d_hermite` の `||diff[ax]|| = 98.52 → <1e-3`、rel_err
1.795% → 1e-4 オーダー）を実装し、以下を達成した:

- **rel_err 1.795% → 2.18e-07**（5 桁改善、FD 機械精度水準）
- **||diff[ax]|| 98.52 → 4.75e-05**（6 桁改善、status-355 目標を約 5 桁オーバーシュート）
- **comp_z 77.3% → 1.16e-05**（adj 列 z 方向の s-tracking 未実装分を解消）

同時に status-354 の「mat-only が最良」という解釈も訂正した。
status-354 仮説 A（`K_hermite_adj` のフル項化 = `w_mat n⊗n - w_geo I_nn`）は
**数理的には正しい** が、単独で導入すると (ii) s-tracking 経路が未実装なため
相殺相手がなく 21 倍悪化していた。status-356 は仮説 A と仮説 B（K_closest/K_st
の active×adj 拡張）を**同時導入**して 2 経路 (i)(ii) の $P_\perp$ 成分を
解析的に相殺させた。

## 実装内容

### 1. `KcHermiteNonlocalStiffnessProcess.process()` フル項化

`xkep_cae/contact/contact_force/strategy.py:1064`:

```python
# status-355 以前（mat-only、(ii) 未実装時のワークアラウンド）
K_3x3_mat = w_mat[:, None, None] * nn

# status-356（フル項、(i) 直接経路の数理厳密形）
K_3x3_mat = w_mat[:, None, None] * nn - w_geo[:, None, None] * I_nn
```

docstring と version を 1.0.x → 1.1.0 に更新。

### 2. `ContactForceStStiffnessProcess._process_batch_term` active×adj 拡張

`xkep_cae/contact/contact_force/strategy.py:465-727`:

- 既存 `_batch_st_jacobian_hermite` は status-311 時点で既に `ds_du_adj`
  / `dt_du_adj`（N, 12）を返している（adj_node_map 系）
- `inp.adj_node_map` + `inp.adj_node_counts` + `term in {"closest","residual"}`
  の両経路で active×adj の COO ブロックを追加:
  ```python
  K_local_adj = -(
      df_ds_term[:, :, None] * ds_du_adj[:, None, :]
      + df_dt_term[:, :, None] * dt_du_adj[:, None, :]
  )
  ```
- row_idx は active gdofs（3 DOF）、col_idx は adj gdofs（隣接 4 ノード×3 DOF=12）

### 3. `ContactForceStStiffnessInput` 新フィールド

```python
adj_node_counts: np.ndarray | None = None
```

`HuberContactForceProcess.tangent` / `assemble_tangent` から
`_adj_node_counts` を貫通配線（5 箇所）。

### 4. `_batch_dm_ext_coeffs` ヘルパ抽出

`HuberContactForceProcess` staticmethod として新設（strategy.py:1492-1518）。
従来 `KcHermiteNonlocalStiffnessProcess.process()` 内と
`HuberContactForceProcess.tangent()` 内でインライン二重実装されていた
$\mathrm{d}m_{\mathrm{ext}}/\mathrm{d}u$ 係数計算を共通化
（MCDD 脱法パターン 3「類似コード二重実装」回避）。

### 5. モジュールコメント / docstring 更新

- `_K_C_TERM_EXPANSION_CONTRACT` モジュールコメントに status-356 注記を追加
- `KcHermiteNonlocalStiffnessProcess` docstring に status-354/355/356 経緯を追記

### コード差分規模

`git diff --stat`: `strategy.py` 192 insertions / 45 deletions。
status-355 見積もり「~45 行のコア変更」に対し実装は `_batch_dm_ext_coeffs`
ヘルパ抽出と `_process_batch_term` 両 term 対応で約 2〜3 倍に拡大したが、
ヘルパ化により正味の新規ロジックは ~60 行。

## 測定結果

### FD 診断（`work/beam_hysteresis/14_kc_closest_adj_diagnostic.py`）

```
── 全体 ──
  ||K_c||       = 5.4889e+03
  ||FD_Kc||     = 5.4889e+03
  ||diff||      = 1.1964e-03
  rel_err       = 2.1797e-07        ← status-355: 1.7949e-02

── ブロック分解 ──
  aa (active×active): ||diff||= 1.1954e-03 (99.8%, rel_err[aa]=2.19e-07)
  ax (active×adj)  : ||diff||= 4.7517e-05 ( 0.2%)   ← status-355: 9.8522e+01
  xa (adj×active)  : ||diff||= 0.0000e+00 ( 0.0%)
  xx (adj×adj)     : ||diff||= 0.0000e+00 ( 0.0%)

── adj 列 (行次元全体) comp 別 diff ──
  x: ||diff||=3.94e-05  ||FD||=4.59e+01   ← status-355: 4.43e+01
  y: ||diff||=2.39e-05  ||FD||=4.65e+01   ← status-355: 4.41e+01
  z: ||diff||=1.16e-05  ||FD||=5.98e+02   ← status-355: 7.61e+01
```

status-355 の実装計画で掲げた「||diff[ax]|| 98.52 → <1e-3」をほぼ 5 桁
オーバーシュートして機械精度に到達。comp_z の 76.11 → 1.16e-05 は
status-354 の「mat-only が最良」解釈の反例となり、適切な 2 経路実装で
完全解消されることを確認した。

### 数理的整合性

status-354 の仮説 A 単独実装（rel_err 38.49%）は (i) フル項を入れて
(ii) の相殺相手がない状態だった。status-356 の (i)+(ii) 同時実装で
$P_\perp$ 成分が相殺され、残るは FD が捉える $\hat{n}\hat{n}^\top$ 方向の
ギャップ変化のみ。詳細は `docs/math/03_huber_contact_penalty.md` §7.2/§7.3 参照。

## ゲート

- ✅ `ruff check xkep_cae/ tests/`: All checks passed
- ✅ `ruff format --check xkep_cae/ tests/`: 191 files already formatted
- ✅ `pytest xkep_cae/contact/`: **421 passed, 5 skipped**（status-355 と同数、回帰 0）
- ✅ `pytest xkep_cae/contact/contact_force/tests/test_kc_component_fd.py`: **7 passed**（`test_helical_3d_hermite` rel_err 1.795% → 2.18e-07、ゲート緩和なし）
- ✅ **7 本撚線曲げ揺動回帰**（`tests/numerical_tests/test_strand_bending_convergence.py::test_strand_bending_oscillation_converges`）: **passed**（10.18s、frac=1.0、bending_angle=0.100 rad 完走）
- ✅ 診断スクリプト `work/beam_hysteresis/14_kc_closest_adj_diagnostic.py` 実行で rel_err 5 桁改善を確認
- ⚠️ **19 本撚線 K_c FD 再計測 / 接触あり 90° 曲げ重量回帰は未実施**: 本 status の焦点は
  gate テスト FD 機械精度到達。19 本 `mat_only` rel_err mean=44% の改善測定は
  status-357 で `work/beam_hysteresis/13_kc_component_fd_19strand.py` を Phase C-3'
  設定で再実行して行う。

## ドキュメント訂正

| ファイル | 変更内容 |
|---------|---------|
| `docs/math/03_huber_contact_penalty.md` §7 | `§7.1 2 経路解析`、`§7.2 相殺定理`、`§7.3 status-354 反証 ⇒ status-356 解決`、`§7.4 診断裏付け` に再構成。status-354 の mat-only 擁護論を (ii) 未実装時のワークアラウンドとして訂正 |
| 同 §3.1 表 | `K_hermite_adj` 行を「(i) フル項（`w_mat n⊗n - w_geo I_nn`）」に、`K_st` 行に「status-356 で active×adj ブロックに拡張」を追記 |
| 同 §4.1 末尾 | status-354 反証 + status-356 解決の両論を併記 |
| 同 §8 trace 表 `[#eq-hermite-pA]` 行 | 実装位置に `KcHermiteNonlocalStiffnessProcess`（i）+ `ContactForceStStiffnessProcess._process_batch_term`（ii）の両 Process 列挙、状態を「status-356 で FD 機械精度一致」に |
| 同 関連 status | status-355 行・status-356 行を追加 |

## status-357 引き継ぎ

1. **19 本撚線 K_c FD 再計測**: `work/beam_hysteresis/13_kc_component_fd_19strand.py`
   を Phase C-3' 設定で実行し、`mat_only` rel_err mean=44% / comp_x max=98% の
   改善を定量計測。status-355 は active×adj 局在を 7-strand シナリオで確認した
   のみで、19-strand ではより複雑な adj パターン（1 active ペアあたり 4 隣接
   ノード × 複数 active 要素）が出現するため再検証が必要。
2. **19 本撚線 Type D stall 再試行**: frac=0.48 → 1.0 完走を目標
   （status-339 で frac=0.484 止まり、status-341 で n_incr=40 リトライも
   frac=0.1991 退化）。仮説 B 実装で Type D の真の原因が解消されたか確認。
3. **接触あり 90° 曲げ重量回帰**: status-298/299 の frac=1.0 / incr/cutback が
   status-356 で回帰していないこと（あるいは改善したこと）を確認。
4. **C18/C19 契約検査（Phase E 準備）**: status-350/351/356 で抽出した 5 項 Process
   が `TermExpansionContract.providers` の tuple 要素と一対一に紐付いていることの
   AST 静的検査を追加。

## 関連 status

- status-295: K_c_adj mat-only 確立（(ii) 未実装時の実装上のワークアラウンド、
  status-356 で置換）
- status-351: `KcClosestPointStiffnessProcess` 新設（Phase C-2）
- status-353: 数理台帳訂正（K_mat,ndir ≡ K_geo）
- status-354: 仮説 A 単独実装の反証（rel_err 1.795% → 38.49%、21 倍悪化）
- status-355: 仮説 B 診断（active×adj ブロック 100% 局在確認、実装コスト ~45 行）
- **status-356（本 status）**: 仮説 A + 仮説 B 同時実装 — FD 機械精度一致（rel_err 2.18e-07）
- status-357（次）: 19 本撚線 K_c FD 再計測 + Type D stall 再試行

## 懸念事項・引き継ぎメモ

- status-354 から status-356 への 2 回転換は**数理的仮説の単独検証 → 組合せ
  検証の反復**として MCDD 規範下で正当化される（脱法パターン 10「TODO として
  積む」に非該当、2 つの実装実験で仮説を精緻化）。status-354 時点で 2 経路
  同時実装の必要性を見抜けなかったのは数理台帳 §7 の 2 経路解析が未記述で
  あったためで、本 status で §7.1 / §7.2 に明示的に記載した（再発防止）。
- `_batch_dm_ext_coeffs` の抽出により、今後 `K_hermite_adj` 経路の数式を
  変更する際の修正箇所が 2 → 1 に集約された（`KcHermiteNonlocalStiffnessProcess`
  本体と `ContactForceStStiffnessProcess._process_batch_term` が同じヘルパを
  共有）。
- `term="residual"` 経路にも adj 拡張を適用したのは、摩擦 $K_{\mathrm{st}}$ の
  residual 項も同じ 2 経路構造を持つため。仮説 D（摩擦 K_st 隣接拡張）は
  仮説 B に自然吸収され、Phase C-3' は hypothesis C のみ残存。
- 本 status は gate テスト（7 ノード 2 セグメント pair）での FD 機械精度一致
  を示したのみで、19 本撚線スケールでの効果は status-357 で測定する。
  status-341 の「x 成分 72-97% 不整合」が K_c 不整合由来であれば frac=1.0
  完走が期待できるが、活性集合変化（仮説 C）由来の成分が残っていれば追加
  対処が必要。

## コミット（予定）

1. `feat(contact): Phase C-3' 仮説 B 実装 — K_closest/K_st active×adj 拡張 + K_hermite_adj フル項化で FD 機械精度一致（status-356）`
2. `docs(math): 03 章 §7 全面再構成 + §3.1/§4/§8 訂正 — 2 経路解析と status-356 解決を反映（status-356）`
3. `docs(status): status-356 + README/status-index/roadmap/CLAUDE.md 更新（Phase C-3' 完了、Phase D 着手準備）`
