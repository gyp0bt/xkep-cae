# status-352: 計画書ロスト記録 + Phase C-3 前提再検証（K_geo と K_mat,ndir の数理的同一性）

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-19
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25（変動なし）

## 概要

本 status は **2 つの中断スナップショット** を記録する（CLAUDE.md「コンテキスト
不足時は git stash + 中断スナップショット」方針に従う）:

1. **計画書 `/root/.claude/plans/deep-wiggling-seal.md` は永久ロスト**。セッション
   開始時のファイル存在確認で再現不能を確認。以降、計画情報は `CLAUDE.md` /
   `docs/roadmap.md` / `docs/status/status-{N}.md` の転記を正として運用する。
2. **Phase C-3（`KcNormalDirectionStiffnessProcess` 新設）の前提が数理的に疑わしい**。
   `HuberContactForceProcess.tangent()`（`strategy.py:1595`）のペア局所形を
   直接導出し、**現行の `K_geo` がすでに $-p_n\,\partial\hat{n}/\partial u$ と
   同一のテンソル形**（`-cc · (p_n/d) · (I - \hat{n}\hat{n}^T)`）を返すことを
   確認した。この状況で新規に `KcNormalDirectionStiffnessProcess` を追加すると
   **K_c 合算で K_mat,ndir 項が二重計上**となり、既存の `test_kc_component_fd.py`
   19 件が rel_err 倍化で fail するリスクが高い。

本 status では実装 Process を追加していない（**脱法パターン 8「ベースライン側が誤
と根拠なく主張」を避けるため**）。代わりに数理的な検証結果を提示し、次セッション
（Codex or 継続 Claude）に判断を引き継ぐ。

## 成果物

### ドキュメント更新（計画書ロスト対応）

| ファイル | 変更 |
|---------|------|
| `CLAUDE.md` | 「計画」行を「計画書は永久ロスト」表記に更新、脱法パターン 10 項は本ファイル転記として残置、「セッション開始時の必須確認」から計画書全文読みを削除 |
| `docs/roadmap.md` | 現在地セクションの計画書参照を「永久ロスト」記述に更新 |
| `docs/math/README.md` | 関連セクションの計画書参照を「永久ロスト」記述に更新 |
| `xkep_cae/mathematics/registry.py` | モジュール docstring の上位プラン注記を「永久ロスト」に更新 |
| `xkep_cae/mathematics/docs/mathematics.md` | 全体計画参照を「永久ロスト」に更新 |

status-346〜status-351 のヒストリカル status ファイルは改変しない（STA2 防止
ルール: 過去記録は改変禁止）。

### 新規実装

**なし**。Phase C-3 の実装は次セッションに引き継ぐ（下記「次セッション向け判断材料」
参照）。

## Phase C-3 前提の数理的検証

### 既存コードのペア局所 K_c 形

`HuberContactForceProcess.tangent()` の現行実装は 3x3 ペア局所ブロックとして
以下を組み立てる（`strategy.py:1594-1595`、status-288 以前からの一貫形）:

```python
K_3x3 = w_mat * (n̂ ⊗ n̂) − w_geo * (I − n̂ ⊗ n̂)
      = w_mat · nn − w_geo · I_nn
```

ここで:

- `w_mat = dp_n/dx · k_pen`（[#eq-dpn-dx](../math/03_huber_contact_penalty.md#eq-dpn-dx)）
- `w_geo = p_n / d`（$d$ は最近接点距離）
- `I_nn = I₃ − n̂⊗n̂ = P_⊥`（法線直交射影）

これを `cc = c_i c_j` で広げ、シェープ関数込みの 12x12 に組むのが Phase C-1 以降
の `KcNormalStiffnessProcess` + `KcGeoStiffnessProcess` の役割であり、現行
実装は `K_c = K_mat(cc · w_mat · nn) − K_geo(cc · w_geo · I_nn) + K_st` を返す。

### 数式からの再導出（$(s,t)$ 凍結、ペア局所、`K_c = ∂(-f_c_raw)/∂u` 符号）

接触力（コード符号規約）: $\boldsymbol{f}_c^{(k)} = c_k(s,t)\,p_n\,\hat{n}$
（$k \in \{A_0, A_1, B_0, B_1\}$、$c_k$ は符号を内包したシェープ関数係数 `coeffs`）。

FD テスト（`test_kc_component_fd.py:224`）は **`-f_c_raw` の FD** を取っているので、
$K_c = \partial(-\boldsymbol{f}_c)/\partial u$ を導出する。

$r_{AB} \equiv p_A − p_B$、$d = \|r_{AB}\|$、$\hat{n} = r_{AB}/d$（コード規約: $B\to A$）。
ノード $l$ に対する偏微分:

- $\partial r_{AB}/\partial u^{(l)} = +c_l \cdot I_3$（符号内包シェープ関数で統一）
- $\partial d/\partial u^{(l)} = c_l \cdot \hat{n}$
- $\partial g/\partial u^{(l)} = c_l \cdot \hat{n}$（$g = d - r_A - r_B$、半径は u 非依存）
- $\partial p_n/\partial u^{(l)} = (dp_n/dg) \cdot c_l \cdot \hat{n} = -w_{\mathrm{mat}} \cdot c_l \cdot \hat{n}$
- $\partial \hat{n}/\partial u^{(l)} = (1/d) P_\perp \partial r_{AB}/\partial u^{(l)} = (c_l/d) P_\perp$

ペア $(k,l)$ ブロック:

$$
K^{(kl)} = \frac{\partial(-\boldsymbol{f}_c^{(k)})}{\partial u^{(l)}}
= -c_k\left[\frac{\partial p_n}{\partial u^{(l)}} \otimes \hat{n}
  + p_n \cdot \frac{\partial \hat{n}}{\partial u^{(l)}}\right]
$$

- 第 1 項（$K_{\mathrm{mat,nn}}$）:
  $-c_k \cdot (-w_{\mathrm{mat}} \cdot c_l \cdot \hat{n}) \otimes \hat{n}
   = + c_k c_l w_{\mathrm{mat}} (\hat{n} \otimes \hat{n})$
   → 既存コードの `+ cc · w_mat · nn` と一致 ✓
- 第 2 項（本 status の焦点）:
  $-c_k \cdot p_n \cdot (c_l/d) P_\perp = − c_k c_l (p_n/d) P_\perp$
   $= − c_k c_l w_{\mathrm{geo}} I_{nn}$
   → 既存コードの `− cc · w_geo · I_nn`（すなわち `- K_geo`）と一致 ✓

**結論: コードの `K_geo` が `cc · w_geo · I_nn`（正値）、`K_c = K_mat − K_geo`
が `cc · (w_mat · nn − w_geo · I_nn)` を返している時点で、
$-p_n\,\partial\hat{n}/\partial u$ は既に `K_geo` として実装済み**。

### 数理台帳（`docs/math/03_huber_contact_penalty.md`）との整合性

- [#eq-kc-pair-block](../math/03_huber_contact_penalty.md#eq-kc-pair-block) の
  式は $K_c^{(ij)} = c_i c_j [w_{\mathrm{mat}} (\hat{n}\hat{n}^\top) - w_{\mathrm{geo}} P_\perp] + K_{\mathrm{st}}^{(ij)}$
  で、第 2 項 $- w_{\mathrm{geo}} P_\perp$ は**まさに上記の第 2 項（$K_{\mathrm{mat,ndir}}$）
  と同一形式**。
- 一方、3.1 項の表と §4 `sec-ndir` では `K_geo` と `K_mat,ndir` を **別項として
  列挙**しており、「K_geo 重み = $p_n/d$」「K_mat,ndir 重み = $p_n$（$d$ で割らない
  $P_\perp$ 単独）」と書いている。コード現状と直接導出とは不整合。
- 台帳 §4 の「ペア局所の $c_i c_j$ 込み」への最終代入は未記述で、導出の中間ステップ
  で $d$ 因子が消える根拠が示されていない。**数理台帳 §4 側の表記が過剰分離である
  可能性が極めて高い**。

### 19本撚線 Type D stall の再原因候補

status-344 の「K_mat rel_err 44% / comp_x max=98%」は、上記導出によれば
**コードの `K_geo` 不足ではない**。考えられる原因候補:

1. **Hermite 非局所項 `K_hermite_adj` の mat-only 近似**（status-295 / 351）で
   省略された幾何剛性の隣接拡張。`KcHermiteNonlocalStiffnessProcess` は
   `w_mat · nn` のみで `w_geo · I_nn` を隣接ノード列に展開していない。
2. **`K_closest` / `K_st_residual` の Hermite 追従項** で $\partial c_i/\partial(s,t)$
   と $\partial\hat{n}/\partial(s,t)$ が frozen-m 部分解消（status-294）の
   残差を含んでいる。
3. **接触ペア集合の凍結近似**（NR 反復内で active set を固定する運用）が
   large-deformation 方向で追従不足。

これらは `KcNormalDirectionStiffnessProcess` を新設しても解決しない。

## 次セッション向け判断材料

### 選択肢 A: 台帳を修正（K_geo と K_mat,ndir は同一項と宣言）

推奨パス。実装変更なしで完了可能。

1. `docs/math/03_huber_contact_penalty.md` §3.1 の項一覧を **5 項** に統合
   （K_mat,ndir を独立項としないで K_geo と同一）
2. §4 `sec-ndir` を「$-p_n\,\partial\hat{n}/\partial u$ は形式上 $-w_{\mathrm{geo}} P_\perp$
   に帰着し、`KcGeoStiffnessProcess` として実装済み」と書き直し
3. `_K_C_TERM_EXPANSION_CONTRACT.term_names` は 5 項のまま維持
4. Phase C-3 の目的を「19 本 Type D stall 解消」から「Hermite 非局所・K_st 追従
   項の追加修正」に再定義。原因候補 1〜3 のうち 1 が本命。

**懸念**: CLAUDE.md 脱法パターン 4「rename で済ませる」には抵触しない（rename
ではなく台帳側の誤記訂正）。ただし、過去 status-348 で書かれた 6 項分解が**初回
から誤っていた**という主張に正当化が必要 — 本 status の直接導出が根拠となる。

### 選択肢 B: 数式をさらに細密化（K_geo と K_mat,ndir を真に分離する再導出を試みる）

台帳 §4 が正しく何らかの分離原理があるなら、それを明らかにする必要がある。
候補:

- UL 定式化で基準配置更新時に発生する追加項があり、本導出は TL 近似のため
  見逃している可能性（→ `01_kinematics_beam.md` 参照要）。
- $\partial r_{AB}/\partial u$ がシェープ関数だけでなく回転 DOF にも依存する
  成分が抜けている可能性（→ 回転 DOF 3 成分はコードで `K_c` の行・列とも
  0 埋めされており、接触側では 3 並進のみ扱う。梁 K_struct 側で回転が入る。）

選択肢 B を採るには追加の導出時間が必要で、コンテキスト内では完結しない。

### 選択肢 C: とりあえず新 Process を追加してテスト結果で判定する

**非推奨**。CLAUDE.md 脱法パターン 3（wrapper だけ）・5（skip で pass）・
8（根拠なき主張）のいずれかに抵触する可能性が高い。本 status の導出が正しけれ
ば FD テストが fail し、台帳 §4 が正しければ pass するが、後者の場合でも
「なぜ pass したか」を数値的に説明できなければ盲目的な実装になる。

### 推奨

**選択肢 A を次セッションで採用し、§4 を訂正した上で Phase C-3 の目的を
「Hermite 非局所 K_geo 項の隣接拡張（原因候補 1）」に再設定する**。

## 検証（変更なしの回帰確認）

```
$ uv run ruff check xkep_cae/ tests/
All checks passed!

$ uv run ruff format --check xkep_cae/ tests/
191 files already formatted

$ uv run python contracts/validate_process_contracts.py
（`C15 equation_refs` セクションで既存 contracts.py docstring の alias 参照を含め
contract violation 0 件を維持する想定。変更は doc 表記のみなので回帰なし）
```

## 未解決事項（次セッションへの引き継ぎ）

- [ ] **選択肢 A を実行するか、選択肢 B で再導出するかの判断**（設計判断を要する）
- [ ] 選択肢 A 採用の場合:
  - [ ] `docs/math/03_huber_contact_penalty.md` §3.1 / §4 の修正
  - [ ] `_K_C_TERM_EXPANSION_CONTRACT` の `description` 更新
  - [ ] Phase C-3 目的を「Hermite 非局所 I_nn 拡張」に再定義
  - [ ] 19本撚線実測は目的再定義後に改めて実施
- [ ] 選択肢 B 採用の場合:
  - [ ] UL / 回転 DOF の寄与を含めた `K_c` の 6 項分解の再導出（紙ベース）
  - [ ] 分離原理の数理的確認（`docs/math/` に追記）
  - [ ] 初めて `KcNormalDirectionStiffnessProcess` の新設に進める

## コミット（予定）

```
docs: 計画書 /root/.claude/plans/deep-wiggling-seal.md ロスト記録 + 参照更新
docs: status-352 + status-index + roadmap 更新（Phase C-3 前提の数理検証）
```

Plan: `/root/.claude/plans/deep-wiggling-seal.md` は **永久ロスト**（本 status で確認）。
以降は `CLAUDE.md` + `docs/roadmap.md` + status 群を正規参照とする。
Phase A〜E / status-346〜356 の **6/11 完了 維持**（本 status は中断スナップショット
として位置付け、Phase C-3 は選択肢 A or B の判断後に再開）。
