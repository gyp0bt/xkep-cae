[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-393: 達成確認マトリクス導入 — STA2 連鎖撤回の構造的予防（documentation status）

**日付**: 2026-05-06
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12+5+17+11+6 passed（status-392 と同数、documentation status のため変動なし）

## 概要

ユーザー指示「STA2 を警戒するなら達成確認条件をマトリクスとかにまとめて、何が確認
できてできていないか整理すべき」を受け、`docs/status/verification_matrix.md` を
**永続ドキュメント**として新設。status-379 / 381 / 387 の連鎖撤回事例を踏まえ、
**達成・未達成・未検証**を独立な軸で可視化することで、STA2（数値の捏造 / 偽陽性 /
単一指標一致による誤判定）を**構造的**に予防する運用ルールを導入。

実装本体（`xkep_cae/`、`tests/`、`contracts/`）は **無変更**。documentation status。

## 1. 動機 — status-379 / 381 / 387 連鎖撤回の教訓

| 主張 status | 主張内容 | 撤回 status | 撤回理由 |
|:-:|---|:-:|---|
| 379 | 19 本 explicit frac=1.0 完走で凍結解除条件達成 | 380 | `max\|u\|=1.59×10⁸ mm` 数値発散発覚。frac=1.0 / E_kin/E_strain<5% gate は数学的構造由来で発散時にも PASS する盲点 |
| 381 | mass scaling bug 修正で発散停止、形式 gate 全 PASS | 381 自身 | ユーザー指摘で精査、解析解 73.3 mm に対し explicit 40 mm（50% 系統的アンダー）。形式 gate (1)〜(4) は under-relaxation 解でも PASS する盲点 |
| 387 | n_inc=8000 sweet spot で精度 gate (5) 達成（err 0.58%） | 388 | 3 指標 AND gate で再検証、L_arc=234 mm（梁が 2.3x ストレッチの非物理解）。「sweet spot」は座標値偶然交差 |

**3 件連続して達成主張 → 撤回**。これは個別 status の問題ではなく、**運用構造の
欠陥**: gate が形式的（数学的構造で常に PASS）/ 単一指標一致のみ / 達成主張記録の
散逸 によって、次セッションが過去の偽陽性を再演する素地が残っていた。

→ **達成・未達成・未検証を独立に記録するマトリクスを永続化**することで、運用面で
   STA2 を予防する。

## 2. 導入したマトリクス（`docs/status/verification_matrix.md`）

### 2.1 構造（8 セクション）

| § | 内容 | 行数（典型） |
|:-:|---|:-:|
| 0 | 状態凡例（✅/🟡/❌/⬜/⏸/🔁） | — |
| 1 | MCDD 凍結解除条件（5 条件） | 5 |
| 2 | Phase α/β/γ/δ 検証進捗 | 4 + 2 + 9 + 1 |
| 3 | status-381〜387 上位層改修対象 | 9 |
| 4 | 既存 validation の 3 指標 gate 化 | 5 |
| 5 | **STA2 撤回履歴（透明性ルール根拠記録）** | 3（履歴保持で増加） |
| 6 | 凍結中 TODO | 6 |
| 7 | 更新ルール | — |
| 8 | 凡例参照テーブル（一覧） | — |

### 2.2 状態凡例の設計

| 記号 | 意味 |
|:---:|---|
| ✅ | 達成（実機検証 + **3 指標 AND gate PASS**、撤回されていない） |
| 🟡 | 部分達成（条件の一部のみ満たす、または特定領域のみ） |
| ❌ | 未達（実機検証で **FAIL を実証**） |
| ⬜ | 未検証（実機実行未着手） |
| ⏸ | 凍結（MCDD 完了まで再開禁止） |
| 🔁 | 撤回（過去の達成主張が後に撤回された、**履歴保持**） |

**重要設計**: ❌「未達（実証済）」と ⬜「未検証（実行未着手）」を **明確に分離**。
status-379 のような「実証されていないが達成と主張」の偽陽性パターンを、状態記号
レベルで構造的に防ぐ。

### 2.3 STA2 撤回履歴の保持ルール

§5 に過去の撤回事例を **削除せず**履歴として保持（透明性ルール）。次セッションが
類似の達成主張をする前に、過去の同パターン撤回を視認できるようにする。

🔁 状態の項目（候補却下や撤回対象）も §3 に保持。改修候補の試行と却下履歴を一覧化
することで、再試行の必要性判断を効率化。

## 3. 運用ルール

### 3.1 CLAUDE.md「作業完了時の必須手順」§5 として組込

```
1. README.md 更新 → 2. status 新規作成/更新 → 3. status-index.md 更新
4. roadmap.md 更新
5. **`docs/status/verification_matrix.md` 該当行を更新**（達成 ✅ / 部分 🟡 /
   未達 ❌ / 撤回 🔁。撤回があれば §5 STA2 撤回履歴に履歴として追記、削除禁止）
6. 不整合はその場で修正 or TODO追加 → 7. feature ごとにコミット → push
```

### 3.2 CLAUDE.md「セッション開始時の必須確認」§3 として組込

```
1. ~~計画書を読む~~ → 永久ロスト、CLAUDE.md 参照
2. 最新 status を読む
3. **`docs/status/verification_matrix.md` を読む**（status-393 運用化）
4. MCDD 脱法 10 項自己チェック
5. その上で着手
```

「自分の作業がどの行を更新するか」を **先に把握**することで、達成主張の独立性 /
3 指標 AND gate 必須を運用面で担保する。

## 4. 現時点でのマトリクス内容サマリ

| カテゴリ | ✅ | 🟡 | ❌ | ⬜ | ⏸ | 🔁 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 凍結解除条件（5 条件） | 1 | 1 | 3 | 0 | 0 | 0 |
| Phase α（4 ケース） | 4 | 0 | 0 | 0 | 0 | 0 |
| Phase β（2 ケース） | 2 | 0 | 0 | 0 | 0 | 0 |
| Phase γ（5 + 集計 + 2 拡張） | 6 | 0 | 1 | 2 | 0 | 0 |
| Phase δ（接触あり） | 0 | 0 | 0 | 1 | 0 | 0 |
| 上位層改修対象（9 項目） | 0 | 0 | 0 | 2 | 1 | 6 |
| 既存 validation gate 化（5 項目） | 0 | 0 | 0 | 5 | 0 | 0 |
| 凍結中 TODO（6 項目） | 0 | 0 | 0 | 0 | 6 | 0 |

→ **Phase α/β/γ-1 で foundation 健全性は完全実証**（✅ 12 件）。
   **凍結解除に向けて未達 ❌ は 4 件**（19 本 frac=1.0 / max\|u\| / 解の精度 / γ-1 n=1
   は既知の離散化誤差）。
   **未検証 ⬜ は 10 件**（次セッション以降の対象、最優先は assembler / UL 1 要素再現）。

## 5. ゲート結果（documentation status）

| ゲート | 結果 | 備考 |
|---|---|---|
| `pytest contact + math + time_integration + strand_bending_oscillation` | **743 passed 5 skipped** | status-392 と同数、変動なし |
| `python contracts/validate_process_contracts.py` | 全 24 検査 OK | C1〜C24 + O1〜O3 |
| `test_helical_3d_hermite` rel_err | 2.18e-07 維持 | status-356 で達成 |
| `ruff check work/beam_element_validation/` | All checks passed | |
| `ruff format --check work/beam_element_validation/` | 10 files already formatted | |
| `verification_matrix.md` 新設 | ✅ | 8 セクション、初版 |
| CLAUDE.md 必須手順 + 開始確認に組込 | ✅ | §作業完了 §5 + §セッション開始 §3 |

## 6. 次セッションへの引き継ぎ

### 6.1 次 status の最優先候補（変更なし）

status-392 で確定した次セッション最優先候補は変わらず:

- **assembler / UL update_reference の 1 要素再現実験**
  (`work/beam_element_validation/49_beta2_with_assembler_ul.py`)
  → β-2 直接駆動（機械精度 0.000%）と assembler 経由 + UL 更新あり/なしの差分を
    比較し、status-381〜387 改修対象を 1 要素規模で局在化

新規追加された運用 task:
- マトリクス §3「上位層改修対象」の `assembler` 経由 + UL `update_reference` 行を
  状態 ⬜ → 検証完了に更新（実機 PASS なら ✅、実機 FAIL なら ❌）

### 6.2 副次（status-392 から継続）

- Phase δ 接触あり 2 本撚線（`48_delta_2strand_contact.py`）
- Phase γ-2 大 curvature 拡張（θ=π/2、`50_gamma2_large_curvature.py`）
- 既存 validation の 3 指標 gate 化（マトリクス §4 の 5 項目）

### 6.3 マトリクス運用の改善余地

本 status は初版。運用上の改善案:

1. **マトリクスの自動生成**: status から「達成・未達成」項目を抽出してマトリクスを
   自動更新する CLI を作る（status-393 範囲外、運用後に検討）。
2. **凍結解除条件の進捗バー**: 5 条件のうち何件が ✅ かを CI で可視化（CIPipeline
   との統合は実装本体への波及があり Phase E 完了後）。
3. **既存テストの 3 指標 gate 化チェックリスト**: §4 を実装タスクに展開。

## 7. MCDD 脱法 pattern 自己点検

- **pattern 1（tol 緩和）**: 該当なし、新規 gate threshold 設定なし。
- **pattern 5（既存テスト skip）**: 既存 743 test 全 pass、documentation のみ。
- **pattern 6（骨格 status）**: マトリクス 8 セクション全て埋め、運用ルール 2 箇所
  を CLAUDE.md に組込済。骨格ではなく完結 status。
- **pattern 8（根拠なき主張）**: 全エントリに根拠 status 列を付与。撤回履歴 §5 は
  数理的反証を併記。
- **pattern 10（TODO 先送り）**: 本 status はマトリクス導入 + 運用ルール組込で
  完結、運用は次 status 以降が継続。

## 8. 観察 — 開発運用上の発見

### 効果的

1. **STA2 連鎖撤回 3 件を §5 で履歴化**: 削除せず保持することで、次セッションが
   類似パターンを試行する前に過去の数理的反証を視認できる。これは **「失敗の
   再演を防ぐ予防接種」** として機能する設計。
2. **状態記号 ❌ と ⬜ の分離**: 「実証されていない」と「未検証」を明確に区別する
   ことで、status-379 系の偽陽性（未検証なのに達成主張）が記号レベルで防げる。
3. **マトリクス更新を必須手順化**: 「マトリクスを誰も更新していない status は
   受領しない」運用ルールにより、達成主張の透明性を制度化。

### 今後の観察対象

- マトリクスが肥大化したときの section 分割タイミング（典型 8 セクション → 12
  セクション程度で見直し）。
- 自動生成 CLI 化は便利だが、status から構造化 metadata を抽出する仕組みを別途
  整備する必要があり、優先度は MCDD 完了後。

## 9. 再現手順

```bash
git checkout claude/execute-status-todos-FnP23

# マトリクス確認
cat docs/status/verification_matrix.md

# CLAUDE.md 必須手順セクション確認
grep -A 8 "作業完了時の必須手順" CLAUDE.md
grep -A 12 "セッション開始時の必須確認" CLAUDE.md

# 回帰テスト（documentation status のため変動なし期待）
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
| `verification_matrix.md` 新設 | ✅ | 8 セクション、初版 |
| CLAUDE.md「作業完了時の必須手順」§5 追記 | ✅ | マトリクス更新を必須化 |
| CLAUDE.md「セッション開始時の必須確認」§3 追記 | ✅ | マトリクス読込を組込 |
| status-393 作成 | ✅ | 本 status |
| status-index.md / README / roadmap 更新 | ✅ | status-393 エントリ追記 |
| 実装本体無変更 | ✅ | `xkep_cae/` 不変 |
| 回帰 743 passed 5 skipped | ✅ | status-392 と同数 |
| 全 24 契約検査 OK | ✅ | C1〜C24 + O1〜O3 |
| ruff check + format pass | ✅ | 変更なし |
| **次セッション最優先（変更なし）— assembler / UL 1 要素再現実験** | ❌ | マトリクス §3 で追跡 |

Phase A〜E / status-346〜393 の **44/N 完了**（status-393 は documentation status）。
