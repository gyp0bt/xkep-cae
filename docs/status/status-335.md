# status-335: 2本撚線 M-κ ヒステリシスループ直接観測（infra検証）

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-14
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10（+1 テスト）

## 概要

status-333 で構築した M-κ 追跡基盤を用いて、2本撚線曲げ+揺動（`n_oscillation_cycles=1`）統合モードで **真の load+unload ヒステリシスループ**を観測することを目的とした軽量検証テストを追加。status-333 の既存テストは monotonic load（曲げのみ）だったため、closed-loop かつ散逸 > 0 の検証は初実現。

### 背景

status-333 の `test_mk_and_pairs_combined` は `n_cycles=1`, `n_oscillation_cycles=0` で κ 単調増加のみを検証していた。これでは

- load+unload 経路上のM-κ分岐
- ループ面積（= 1サイクル散逸エネルギー）

を評価できない。そこで `n_oscillation_cycles=1` 統合モードで曲げ+1サイクル揺動を回し、κ が最大値通過後に減少（除荷）→ 逆サイド最大 → 戻りを通過することを確認する軽量テストを追加。

### 意図的に軽量な範囲

7本撚線の load+unload 全体走行は CPU 分単位かかるため本 PR では範囲外。2本撚線で pipeline 健全性（M-κ 下降 + 非零ループ面積）を **CI 時間内（7 秒）** で検証する。7本以上での散逸エネルギー定量評価（ティアドロップ形状観測・ピッチ依存性）は後続 PR で実施。

## 実測値（本セッション）

```
=== M-κ ヒステリシスループ検証（load+unload）===
  M-κ entries: 41
  frac_completed: 1.0000
  n_decreases in κ: 14
  M_peak=6.8451e+00, κ_peak=2.0000e-03
  loop_area=1.1737e-02, elastic_scale=1.3690e-02
  dissipation_ratio = loop_area/(M_peak*κ_peak) = 8.5736e-01
```

- frac=1.0000 完走（incr=41, cutback=3, 6.88s）
- κ 下降イベント 14 回（揺動期間の sin 下降区間と整合）
- ループ面積 1.17e-2 > 0（接触摩擦散逸が捕捉されている）

`dissipation_ratio` は `loop_area / (M_peak × κ_peak)` の簡易指標。2本撚線の疎な接触では「弾性仕事スケールとほぼ同等」と出るが、これは M_peak・κ_peak を単純な外接矩形面積とみなした粗いスケール。実際の弾性仕事 ∫M dκ（load only）との比較は将来テストで計量化する。

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `tests/numerical_tests/test_mk_tracking.py` | `test_mk_hysteresis_loop_oscillation`（+1 テスト）を `TestMkTrackingConvergence` に追加。`n_oscillation_cycles=1` 統合モードで κ 単調増加の否定・loop_area 計算・NaN チェック |
| `CLAUDE.md` | 現状行を更新（テスト数 +9+10、契約違反 4→0） |
| `docs/status/status-index.md` | status-335 エントリ追加 |
| `docs/roadmap.md` | テスト数更新 |
| `README.md` | 現状行更新 |

## テスト結果

```
$ pytest tests/numerical_tests/test_mk_tracking.py -q
10 passed in 15.17s
```

`test_mk_hysteresis_loop_oscillation` 単体: `1 passed in 6.88s`

## 検証

- `ruff check xkep_cae/ tests/` → All checks passed
- `ruff format --check xkep_cae/ tests/` → 175 files already formatted
- `python contracts/validate_process_contracts.py` → 契約違反なし、条例違反なし

## 次のステップ

- [ ] **7本撚線 M-κ ヒステリシスループ実測**（frac=1.0 load+unload、`@pytest.mark.slow` + work/ スクリプト）
- [ ] **散逸エネルギー正確計算**: load-only 弾性仕事と loop_area の比で正しい散逸率を評価
- [ ] **ピッチ依存性検証**（p=50/100/200 での loop_area 差）
- [ ] **接触力/滑り量スナップショットの後処理** — `contact_pair_history` を可視化して κ_cr 分布を実測（ファイバー梁キャリブレーションデータ）
- [ ] リスタート解析方式（ContactFrictionProcess の I/O 整理）
- [ ] 被膜圧縮モデル改善（バリア関数 or 二層モデル）

## 契約違反

**0 件**（status-334 で 12→0 に解消済、本 PR で変動なし）

## 開発運用メモ

- `n_oscillation_cycles=1` 統合モードは CR梁 UL 参照配置不整合（CR梁の f_int=0 問題）を回避でき、load+unload 検証の軽量ケースとして有用。
- 2本撚線かつ `n_increments_per_cycle=8` で 6.88 秒は CI に十分乗るが、`slow` マーカー付き扱い。
- `loop_area` の `dissipation_ratio` 指標は外接矩形比で粗いため、**本質的な散逸評価には load-only 弾性仕事 `W_load = ∫_{load} M dκ` との比を用いるべき**（将来テスト）。
