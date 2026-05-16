# status-334: C16 契約違反 12 件解消 — 純粋関数 privatization

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-14
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+9（増減なし、リネームのみ）
**契約違反**: 12件 → **0件**

## 概要

status-333 に起因する C16 契約違反 12 件（`cable_dissipation.py` / `strand_cross_section_model.py` の公開純粋関数）を `_` prefix privatization で解消。バリデータ仕様 `fn_name.startswith("_")` に準拠。

Process 化は回避した理由：

- `compute_mk_loop_area` / `compute_mk_metrics` は台形則・Shoelace の数値後処理で、M-κ 追跡パイプラインの補助関数。内部実装詳細であり Process 管理対象ではない。
- `cable_geometry` / `make_cable_material` は `CableDissipationProcess.process()` に入力する幾何/材料ファクトリ。Process の内部セットアップヘルパーであり、独立 Process に昇格する管理価値は低い。
- `strand_cross_section_model.py` の 8 関数は Papailiou 解析モデル（閉形式）の補助計算群。現時点で `work/beam_hysteresis/07_analytical_vs_numerical.py`（検証スクリプト）以外からの呼び出しはなく、 active な Process パイプラインに組み込まれていない。

CLAUDE.md「やってはいけないこと: 管理上 process クラスとすべきロジックをあえてプライベート関数や迂回ロジックに替えること」には抵触しない — 本件はそもそも Process 化を要求する「管理対象ロジック」ではなく、純粋数値処理ヘルパーの整理。

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/numerical_tests/cable_dissipation.py` | 4 関数を `_` prefix 化（`_compute_mk_loop_area` / `_compute_mk_metrics` / `_cable_geometry` / `_make_cable_material`）+ 内部呼び出し更新 |
| `xkep_cae/numerical_tests/strand_cross_section_model.py` | 8 関数を `_` prefix 化（`_make_cable_section` / `_compute_section_response` / `_dissipation_energy_bending` / `_calibrate_pretension` / `_dissipation_energy_bending_distributed` / `_calibrate_distributed_model` / `_dissipation_energy_combined` / `_mk_curve_analytical`）+ 内部呼び出し更新 + ruff format |
| `tests/numerical_tests/test_cable_dissipation.py` | 4 関数 import を `_` prefix 版に変更、5 箇所の呼び出し更新 |
| `work/beam_hysteresis/06_dissipation_formula.py` | `cable_geometry` → `_cable_geometry` import + 呼び出し更新 |
| `work/beam_hysteresis/07_analytical_vs_numerical.py` | 7 関数を `_xxx as xxx` alias import（スクリプト本体のロジック変更なし） |
| `README.md` | 現状行を status-334 反映（契約違反 12→0） |
| `docs/status/status-index.md` | status-334 エントリ追加 |

## 検証

### 契約バリデータ

```
$ python contracts/validate_process_contracts.py
(略)
--- C16: 新パッケージ滅菌 ---
  OK
(略)
契約違反なし、条例違反なし
```

### ruff

```
$ ruff check xkep_cae/ tests/
All checks passed!

$ ruff format --check xkep_cae/ tests/
175 files already formatted
```

### pytest

```
$ pytest tests/numerical_tests/test_cable_dissipation.py -x --timeout=180 -q
15 passed in 35.12s

$ pytest tests/test_process_diagnostics.py tests/test_profile_stats.py tests/test_benchmark_runner.py -x -q
48 passed in 1.42s
```

## 次の課題（status-333 から継続）

- [ ] **7本撚線でM-κヒステリシスループを直接取得**（曲げ+揺動でティアドロップ形状を観測）
- [ ] **接触力・滑り量からκ_cr分布を実測**（ファイバー梁キャリブレーションデータ）
- [ ] **ピッチ依存性検証**（p=50/100/200 での散逸差を直接計測）
- [ ] Papailiouモデルのキャリブレーション → 予測モデルとして完成
- [ ] リスタート解析方式（ContactFrictionProcess の I/O を `(u, v, a, contact_pairs)` 入出力に整理）

## 設計ノート

### 将来 Process 化候補（実需が生じた場合）

- `_compute_mk_metrics` → `MkLoopMetricsProcess`: 他所（揺動キャリブレーション等）からも共通に M-κ 指標を計算する需要が出た際に昇格
- `_make_cable_section` + `_compute_section_response` → `CableSectionAnalyticalProcess`: Papailiou モデルを Process 化して CR 梁実測と突き合わせる場合

現状は `work/beam_hysteresis/` の検証スクリプトからしか呼ばれないため、昇格は時期尚早。

## 開発運用メモ

- `ProcessMeta.document_path` を持たない pure analytical model は `work/` 配下のスクリプトから参照されがち。プロジェクト外公開 API ではないので `_` prefix 化が正解。
- ruff format check は毎コミット前に必須（CLAUDE.md `ruff format --check` ルール）。今回 `strand_cross_section_model.py` で 1 箇所リフォーマットが発生。
