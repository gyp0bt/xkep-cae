[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-396: explicit-TL 固定 API 化 — `explicit_ul_disable_update` 独立フィールド追加（候補 (z3) Phase 1、API 化完結 / 実機検証 scope 外）

**日付**: 2026-05-11
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12+5+17+11+6+4 passed（+4 = `TestExplicitULDisableUpdate` 4 ケース）

## 概要

status-395 §6.2 で確定した次セッション最優先項目 **(z3) explicit-TL 固定 API 化のみ**
を実施。`solver_mode="explicit"` でも UL `update_reference()` を一切呼ばない TL 固定
モードを **独立フィールド** `explicit_ul_disable_update: bool = False` で API 化し、
status-395 で多要素 explicit + TL の foundation 健全性が機械精度級で確定したことを
公開 API レベルで運用可能にした。

本 status は **API 化完結 + Default OFF 回帰**のみ。19 本撚線 / 多 strand 実機検証 /
ヘリカル初期 κ foundation 検証は status-397 (ε-1) で実施 (scope 外)。

## 1. 実装スコープ

| 変更点 | 行数 | 内容 |
|---|---:|---|
| `xkep_cae/core/data.py` | +9 | `ContactFrictionInputData.explicit_ul_disable_update: bool = False` 追加（既存 `explicit_ul_update_interval` の直下、status-396 コメント） |
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | +12 | `StrandBendingOscillationConfig.explicit_ul_disable_update` 追加 + 3 経路（曲げ / 揺動 / free_end）plumb-through |
| `xkep_cae/contact/solver/process.py` | +9 / -4 | `_explicit_ul_disable_update` 読出 + `_do_ul_update` ゲートを `(_solver_mode != "explicit") OR (not disable AND interval gate)` に変更 |
| `xkep_cae/contact/solver/tests/test_explicit_dynamic.py` | +148 | `TestExplicitULDisableUpdate` 4 ケース（disable=True 0 回 / interval override / default 既存挙動保持 / ゲート式直接検証） |

設計選択（status-395 §6.2 でユーザー合意）: **独立フィールド方式**を採用。
`explicit_ul_update_interval=0` 解釈拡張ではなく独立フィールドで意図を明示し、
透明性とテスト容易性を確保。

## 2. ゲート式（process.py L926-936）

```python
_next_incr = _incr_count + 1
_do_ul_update = _solver_mode != "explicit" or (
    not _explicit_ul_disable_update
    and (
        _explicit_ul_update_interval <= 1
        or (_next_incr % _explicit_ul_update_interval == 0)
    )
)
if _ul and hasattr(ul_assembler, "update_reference") and _do_ul_update:
    ...
```

- `_solver_mode != "explicit"`（implicit）: 常に True（implicit 経路は無変更）
- `_solver_mode == "explicit"` かつ `_explicit_ul_disable_update=True`: 常に False
- `_solver_mode == "explicit"` かつ `_explicit_ul_disable_update=False`: 既存 interval ゲート評価（status-383 挙動完全保持）

## 3. 単体テスト結果（`TestExplicitULDisableUpdate`）

| テスト | 検証内容 | 結果 |
|---|---|:-:|
| `test_disable_true_skips_all_update_reference` | `disable=True, interval=1, max_incr=4` で呼出 0 回 | PASS |
| `test_disable_true_overrides_interval` | `disable=True, interval=2`（通常 2 回呼出）で呼出 0 回 | PASS |
| `test_disable_false_default_preserves_interval_behavior` | `disable=False, interval=1` で呼出 4 回（既存挙動） | PASS |
| `test_gate_logic_disable_short_circuits` | 実装ゲート式を Python 関数で直接表現し implicit / disable / interval 組合せ全網羅 | PASS |

`_MockULAssembler` で `update_reference` 呼出回数を直接計測（status-383
`TestExplicitULUpdateInterval` と並列配置）。

## 4. ゲート結果

| ゲート | 結果 | 備考 |
|---|---|---|
| `pytest contact + math + time_integration + strand_bending_oscillation` | **747 passed 5 skipped** | status-395 の 743 + 新規 4 ケース |
| `pytest xkep_cae/contact/solver/tests/test_explicit_dynamic.py` | **54 passed** | status-395 の 50 + 新規 4 ケース |
| `python contracts/validate_process_contracts.py` | 全 24 検査 OK | C1〜C24 + O1〜O3 |
| `test_helical_3d_hermite` rel_err | **2.18e-07 維持** | status-356 で達成、無変更 |
| `ruff check xkep_cae/ tests/` | All checks passed | 203 files |
| `ruff format --check xkep_cae/ tests/` | already formatted | 203 files |

## 5. Default OFF 完全保持の検証

- `explicit_ul_disable_update: bool = False`（default）で `_do_ul_update` ゲート式は
  status-383 の挙動と数式的に等価（`disable=False` → 内側 AND の右辺は既存 interval gate と同じ）
- 既存 743 passed 5 skipped は無変更
- 7 本 implicit 90° 曲げ frac=1.0 維持（変更箇所 = explicit 経路のみ）
- `test_helical_3d_hermite` rel_err=2.18e-07 維持（接触接線無変更）

## 6. 達成確認マトリクス更新

`docs/status/verification_matrix.md` 更新:

- §3 上位層改修対象 表に行「explicit-TL 固定 API（`explicit_ul_disable_update`）」追加、
  「API 化 ✅ status-396 / 19 本実機 ⬜ status-397 ε-1 へ持ち越し」
- §2.4 / §8 未検証 ⬜ から「候補 (z3) explicit モード TL 固定 API 化」を分離: API 化部分は ✅、
  実機検証は status-397 で別行 ⬜
- §5 STA2 撤回履歴: 新規撤回事例なし、変更なし

## 7. 次セッションへの引き継ぎ（status-397 ε-1）

status-395 §6.3 で確定した **ε-1 = 3 strand helical + 接触なし + explicit-TL（`disable=True`）**
を `work/beam_hysteresis/` 系で実機適用。新たに検証される要素:

1. 初期 curvature 上の CR（直線 reference vs 曲線 reference の `R_0` 構築）
2. 多 strand 並列 (no contact) の global assembler 振る舞い
3. 端部 BC（MPC + free_end_mode）が explicit + TL モードで成立するか

判定: ε-1 PASS → ε-2（接触あり 3 strand）へ進行 / FAIL → Phase δ（接触あり 2 strand）に retreat。

## 8. MCDD 脱法 pattern 自己点検

- **pattern 6（骨格 status）**: 該当しない。API + 単体テスト 4 件 + Default OFF 回帰 +
  ゲート式の数式的等価性検証で完結
- **pattern 5（既存テスト skip）**: 既存 743 全 pass、追加 4 件も pass
- **pattern 10（TODO 先送り）**: 本 status は「API 化」を完結。実機検証は status-397 で
  独立 scope として明示（status-395 §6.2 ユーザー合意の段階分割に従う）
- **pattern 1（tol 緩和）**: 該当なし、新規テストは全て厳密一致（呼出回数 == 0 / == 4）
- **pattern 8（根拠なき主張）**: ゲート式の数式的等価性を §2 + §5 で明示、テストで実装一致を確認

## 9. 再現手順

```bash
git checkout claude/execute-status-todos-x4zIT

# 新規単体テスト
uv run --extra dev pytest \
  xkep_cae/contact/solver/tests/test_explicit_dynamic.py::TestExplicitULDisableUpdate -v
# 期待: 4 passed

# 回帰テスト
uv run --extra dev pytest xkep_cae/contact/ xkep_cae/mathematics/ \
  xkep_cae/time_integration/ \
  xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py -q
# 期待: 747 passed, 5 skipped

# 契約検査
uv run --extra dev python contracts/validate_process_contracts.py
# 期待: 契約違反なし、条例違反なし

# FD diagnostic（rel_err 2.18e-07 維持）
uv run --extra dev pytest \
  xkep_cae/contact/contact_force/tests/test_kc_component_fd.py::TestKcComponentFD::test_helical_3d_hermite -v -s

# ruff
uv run --extra dev ruff check xkep_cae/ tests/
uv run --extra dev ruff format --check xkep_cae/ tests/
```

## 10. 引き継ぎチェックリスト

| 項目 | 状態 | 備考 |
|---|---|---|
| `ContactFrictionInputData.explicit_ul_disable_update` field 追加 | ✅ | default `False` |
| `StrandBendingOscillationConfig.explicit_ul_disable_update` + 3 経路 plumb | ✅ | 曲げ / 揺動 / free_end |
| `process.py` ゲート式更新（AND 評価） | ✅ | implicit 経路完全無変更 |
| `TestExplicitULDisableUpdate` 4 ケース追加 | ✅ | 全 PASS |
| Default OFF 回帰 747 passed 5 skipped | ✅ | status-395 の 743 + 新規 4 |
| 全 24 契約検査 OK | ✅ | C1〜C24 + O1〜O3 |
| `test_helical_3d_hermite` rel_err=2.18e-07 維持 | ✅ | 接触接線無変更 |
| ruff check + format pass | ✅ | 203 files |
| status-396 作成 + status-index 更新 | ✅ | 本 status |
| README / roadmap / verification_matrix 更新 | ✅ | §現在の状態 / 撚線規模別 opt-in / §3 §8 |
| **次セッション最優先（status-397 ε-1）**: 3 strand helical + 接触なし + `disable=True` 実機適用 | ⬜ | foundation API 整備完了で前提整う |

Phase A〜E / status-346〜396 の **47/N 完了**。
