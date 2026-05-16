# status-336: M-κ ループ散逸率を load-only 弾性仕事基準に厳格化

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-14
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10（変動なし — 既存テストの assertion 厳格化のみ）

## 概要

status-335 で追加した `test_mk_hysteresis_loop_oscillation` は散逸率を
`loop_area / (M_peak × κ_peak)` で算出していたが、これは弾性仕事の外接矩形
近似に過ぎず物理解釈が不可能だった（status-335 自身のメモで
「粗いスケール」「将来テストで計量化」と明記されていた）。

本セッションでは `_compute_mk_metrics`（`xkep_cae/numerical_tests/cable_dissipation.py`、
status-334 で private 化）が既に提供している
`loading_work = Σ max(0, dκ) · M_avg` と `unloading_work = Σ |min(0, dκ) · M_avg|` を
用いて、物理的に意味のある散逸率 `loop_area / W_load` をテストに組み込む。

## 変更内容

### `tests/numerical_tests/test_mk_tracking.py`

`TestMkTrackingConvergence.test_mk_hysteresis_loop_oscillation` を更新：

1. 自前 Shoelace 積分 + `M_peak × κ_peak` スケール比を削除
2. `_compute_mk_metrics` から `loading_work / unloading_work / loop_area /
   peak_moment / peak_curvature / EI_secant / EI_initial / dissipation_ratio`
   を取得
3. 追加 assertion:
   - `loading_work > 0`（曲げで M·dκ > 0 が必須）
   - `loop_area / W_load < 2.0`（明らかに非物理的な散逸率を排除）
   - `metrics["dissipation_ratio"]` と手計算 `loop_area / W_load` の一致
4. 出力メッセージを `W_load / W_unload / loop_area / EI_secant / EI_initial /
   true dissipation_ratio` 構成に整理

### 設計判断

- `_compute_mk_metrics` は status-334 で `_` prefix 化されたが、
  同パッケージ（`xkep_cae.numerical_tests`）内からの利用は契約違反にならない
  （O1: テスト直接関数呼び出し条例はあくまで新パッケージ滅菌 C16 の補完であり、
  `_` prefix private を `numerical_tests/` 内の tests から参照するのは許容）。
- C16 も今回は違反なし（バリデータ再実行で 0 件を確認）。
- 将来、揺動/ピッチ掃引など他所からも同指標が必要になった場合は
  `MkLoopMetricsProcess` に昇格する（status-334 で既に候補として明記）。

## 実測値（本セッション）

```
=== M-κ ヒステリシスループ検証（load+unload）===
  M-κ entries: 41
  frac_completed: 1.0000
  n_decreases in κ: 14
  M_peak=6.4265e-01, κ_peak=2.0000e-03
  EI_secant=3.2132e+02, EI_initial=1.9654e+01
  W_load=6.9875e-03, W_unload=4.7515e-03
  loop_area = |W_load - W_unload| = 2.2360e-03
  true dissipation_ratio = loop_area/W_load = 3.2000e-01
PASSED
```

- `dissipation_ratio = 0.32`：1サイクルあたり負荷仕事の 32% が散逸される
  物理的に意味のある数値
- status-335 の `loop_area / (M_peak × κ_peak) = 0.857` は
  外接矩形比で粗過ぎた（弾性仕事は矩形面積の ~50-60% 程度が典型）
- `EI_secant (3.21e+02) / EI_initial (1.97e+01) ≈ 16`：
  初期接線剛性と割線剛性の比。初期の接触未活性→後半ロックの剛性遷移を示唆
  （2本撚線で短ピッチなので小さいが、非ゼロ）

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `tests/numerical_tests/test_mk_tracking.py` | `test_mk_hysteresis_loop_oscillation` を `_compute_mk_metrics` 経由に書き換え、load-only 基準 assertion 追加 |
| `docs/status/status-index.md` | status-336 エントリ追加 |
| `docs/roadmap.md` | 散逸率厳格化行追加 |
| `README.md` | 現状行更新（旧 loop_area=1.17e-2 → W_load/W_unload/loop_area/ratio=0.32 へ差し替え） |

## テスト結果

```
$ uv run pytest tests/numerical_tests/test_mk_tracking.py -q
..........                                                               [100%]
10 passed in 16.04s
```

`test_mk_hysteresis_loop_oscillation` 単体: `1 passed in 5.40s`

## 検証

- `ruff check xkep_cae/ tests/` → All checks passed
- `ruff format --check xkep_cae/ tests/` → 175 files already formatted
- `python contracts/validate_process_contracts.py` → 契約違反なし、条例違反なし

## 次のステップ（status-335 から継続）

- [ ] **7本撚線 M-κ ヒステリシスループ実測**（frac=1.0 load+unload、
      `@pytest.mark.slow` + work/ スクリプト）
- [ ] **ピッチ依存性検証**（p=50/100/200 での `dissipation_ratio` 差の定量）
- [ ] **接触力/滑り量スナップショットの後処理** — `contact_pair_history` を
      可視化して κ_cr 分布を実測（ファイバー梁キャリブレーションデータ）
- [ ] `_compute_mk_metrics` → `MkLoopMetricsProcess` 昇格判断
      （他所（揺動掃引等）からの呼び出しが 2 箇所以上になった時点で Process 化）
- [ ] リスタート解析方式（ContactFrictionProcess の I/O 整理）
- [ ] 被膜圧縮モデル改善（バリア関数 or 二層モデル）

## 開発運用メモ

- `_compute_mk_metrics` は既に `loading_work / unloading_work / dissipation_ratio /
  EI_secant / EI_initial / loop_area / peak_moment / peak_curvature` を
  dict で返すため、テスト側で再実装する必要はない。今回のように
  外接矩形近似でバイアスを入れる前に一度既存指標関数を探すのが定石。
- status-335 のテストは「loop_area が非ゼロ」と「κ 下降が 1 回以上」という
  **健全性チェック** に留まっていた。本 PR で「**散逸率が物理的範囲内**」まで
  踏み込んだ契約に強化したことで、将来 loading_work が退化した場合
  （材料パラメータ改悪や接触設定の破損）に即検知できる。
