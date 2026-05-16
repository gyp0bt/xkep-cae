# status-340: ContactPairLayerClassifierProcess — 接触ペア層分類後処理

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-14
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+**8** passed（**+8**）

## 概要

status-339 で 19本撚線 κ_cr 分布が **バイモーダル気配**
（3.2e-3 付近と 5.0-5.3e-3 付近のダブルピーク）を示した。これは
**内層対 (layer 1) と外層対 (layer 2) で接触力・梃子比が異なるため
κ_cr が分岐する** 仮説を裏付けるが、status-339 時点では `(elem_a, elem_b)`
ペアを層構造で分類する手段が無かった。

本 status では status-339 の「次のステップ」に挙げられた

> [ ] **ペアインデックス→層分類** — elem_id から strand_id を逆引きし、
>      バイモーダル仮説（内層/外層）を検証するヘルパー追加

を、**新規 PostProcess `ContactPairLayerClassifierProcess`** として実装する。

19本撚線 frac=1.0 完走自体は引き続き次セッションの課題（status-339 の Type D
対策ガイド参照）。本 status では **完走後の解析資産を先に整備** し、完走データが
取れた瞬間に層別 κ_cr 分布を即座に切り出せる状態を作る。

## 実装内容

### 1. `ContactPairLayerClassifierProcess`（新規）

`xkep_cae/numerical_tests/contact_pair_layer_classifier.py`

- **入力** (`ContactPairLayerClassifierInput`):
  - `kappa_cr_per_pair: Mapping[(elem_a, elem_b), float]`
    （`ContactPairAnalysisResult.kappa_cr_per_pair` を直接渡す）
  - `per_pair_dissipation: Mapping[(elem_a, elem_b), float]`
  - `strand_ids: Sequence[int]` — elem_id → strand_id
  - `strand_layers: Sequence[int]` — strand_id → layer
- **出力** (`ContactPairLayerClassifierResult`):
  - `pair_layer_keys: dict[(elem_a, elem_b), (l_min, l_max)]`
  - `per_layer_pair_stats: dict[(l_min, l_max), LayerPairStats]`
    - `n_pairs / n_slipped`
    - `kappa_cr_mean / std / min / max`
    - `dissipation_sum / mean`

層分類ルールは「(elem_a, elem_b) → (layer_a, layer_b) を昇順正規化して
(l_min, l_max) に集約する」だけで責務分離。`PostProcess` カテゴリで
`uses = ()`。

### 2. `StrandMeshResult.strand_layers` 追加

`xkep_cae/mesh/process.py`

`StrandMeshProcess` が内部で持っていた `StrandInfoOutput.layer` を
`StrandMeshResult.strand_layers: tuple[int, ...]` として公開。
これにより外部から（status-339 で必要だった）「strand_id → layer」マッピングを
追加 Process を経由せずに取得可能。

19本撚線（1+6+12 構造）の例: `strand_layers = (0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)`

### 3. 19本撚線 work スクリプトに層分類組み込み

`work/beam_hysteresis/10_kcr_measurement_19strand.py`

`ContactPairAnalysisProcess` の出力をそのまま `ContactPairLayerClassifierProcess`
に渡す。出力例（19本撚線・期待される表示形式）:

```
ContactPairLayerClassifier 結果（バイモーダル仮説検証）
  n_unique_layer_pairs: 4

  (l_min,l_max)  n_pairs  n_slip   κ_cr mean   κ_cr std  κ_cr CV    diss_sum
  (0, 1)               6       6   X.XXe-03    X.XXe-04    X.XX    X.XXe-08
  (1, 1)             ...     ...        ...         ...     ...         ...
  (1, 2)             ...     ...        ...         ...     ...         ...
  (2, 2)             ...     ...        ...         ...     ...         ...
```

### 4. ドキュメント

- `xkep_cae/numerical_tests/docs/contact_pair_layer_classifier.md` 新規

### 5. テスト（8 件追加）

`tests/numerical_tests/test_contact_pair_layer_classifier.py`

| テスト | 検証内容 |
|--------|----------|
| `test_empty_input` | 空入力で全ゼロ結果 |
| `test_pair_layer_key_normalization` | (a,b) と (b,a) が同じ (l_min, l_max) に正規化 |
| `test_inner_inner_vs_outer_outer_separation` | 内層対 (1,1) と外層対 (2,2) で κ_cr 平均が分離（バイモーダル基盤） |
| `test_mixed_layer_pair` | 層跨ぎ対 (1,2) が独立カテゴリとして抽出 |
| `test_dissipation_aggregation` | 散逸合計・平均が層ペア単位で計算 |
| `test_kappa_and_dissipation_union` | kappa と dissipation の和集合が n_pairs にカウント |
| `test_std_computation` | 母集団標準偏差（n で割る）の数値整合 |
| `test_default_layer_pair_stats` | `LayerPairStats()` デフォルトが全ゼロ |

`@binds_to(ContactPairLayerClassifierProcess)` で C3 契約遵守。

## 検証

```bash
$ uv run python -m pytest tests/numerical_tests/test_contact_pair_layer_classifier.py -v
======================== 8 passed in 0.82s ========================

$ uv run python -m pytest xkep_cae/mesh/tests/ tests/numerical_tests/test_contact_pair_analysis.py
======================== 25 passed, 3 warnings in 4.28s ========================

$ uv run python -m pytest tests/ -q --timeout=180
======================== 312 passed, 11 skipped in 126.88s ========================

$ uv run ruff check xkep_cae/ tests/
All checks passed!

$ uv run ruff format --check xkep_cae/ tests/
179 files already formatted

$ uv run python contracts/validate_process_contracts.py
契約違反なし、条例違反なし
```

## 設計判断

### Q. なぜ `ContactPairAnalysisProcess` の拡張ではなく別 Process？

**A.** `ContactPairAnalysisProcess` は「履歴のみで完結する純粋集約処理」だが、
層分類は `MeshData.strand_ids` + `StrandInfoOutput.layer` を要する。
責務（履歴解析 vs メッシュ依存集約）が直交するため別 Process とした。
また、層分類は将来 `n_strands` 以外の軸（角度方向のセクター分類など）
にも拡張可能なので、独立 Process の方が変更影響が局所化される。

### Q. なぜ `StrandMeshResult.strand_layers` を新設？

**A.** `StrandMeshProcess` は内部で `strand_infos` を計算しているが、
これまで `StrandMeshResult` には `mesh: MeshData` と `core_radii` しか
公開していなかった。`mesh.strand_ids` から layer を逆引きするためだけに
`strand_infos` 全体を再計算するのは無駄なので、最小限 `strand_layers` のみ
追加で公開した。`StrandMeshResult` 生成箇所は 1 か所のみで他コードへの影響なし。

### Q. なぜ 19本撚線で実測しないか？

**A.** status-339 で **frac=0.484 で Type D stall** が確認済み。本 status は
**完走後の解析パイプライン整備** に集中し、Type D 対策（K_c FD 診断 /
n_incr=40 / 仮説 A）は別 PR で扱う方針。「次セッション向けに完走時の
レポートが即出る状態にしておく」ことが本 status の主目的。

## ファイル変更

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/numerical_tests/contact_pair_layer_classifier.py` | **新規** — `ContactPairLayerClassifierProcess` 本体 |
| `xkep_cae/numerical_tests/docs/contact_pair_layer_classifier.md` | **新規** — Process ドキュメント |
| `xkep_cae/mesh/process.py` | `StrandMeshResult.strand_layers` 追加 + StrandMeshProcess の出力に組み込み |
| `tests/numerical_tests/test_contact_pair_layer_classifier.py` | **新規** — 8 テスト |
| `work/beam_hysteresis/10_kcr_measurement_19strand.py` | 層分類実行 + 表形式出力追加 |
| `docs/status/status-340.md` | **新規**（本ファイル） |
| `docs/status/status-index.md` | status-340 エントリ追加 |
| `docs/roadmap.md` | テスト数更新・status-340 行追加 |
| `README.md` | 現状行更新（テスト数 +8、status-340 反映） |

## 次のステップ

- [ ] **19本撚線 frac=1.0 完走** — status-339 の Type D 対策ガイドに従い、
  推奨アクション 2（`n_increments_per_cycle=20→40`）から着手。完走後は
  本 status の `ContactPairLayerClassifierProcess` で内層対/外層対の
  κ_cr 平均が分離するか（バイモーダル仮説）を定量検証
- [ ] **7本撚線でも層分類実行** — `09_kcr_measurement_7strand.py` にも
  層分類出力を追加し、7本（1 core + 6 outer）の (0, 1) ペア vs (1, 1) ペアの
  κ_cr 平均を比較。「7本は単峰、19本はバイモーダル」が物理的妥当か検証
- [ ] **層別ヒストグラム可視化** — `matplotlib` で per_layer_pair_stats を
  カラー別ヒストグラムとして出力するヘルパー（`plot` extra 依存）
- [ ] **3 本撚線（layer 1 のみ）の検証** — 全ペアが (1, 1) になることで
  分類器が縮退時に正常動作することを実測で確認

## 開発運用メモ

- status-339 で「未完走の部分成果でも 57 ペアの κ_cr は有効」と書いた通り、
  本 status の層分類 Process は **完走を待たずに即座に活用可能**。次セッションで
  19本撚線が frac=0.484 のままでも、`(1, 1)` と `(2, 2)` の κ_cr 平均比較
  自体は実行できる
- `StrandMeshResult.strand_layers` を追加したことで、今後の層別解析全般
  （層別 dissipation 分布、層間摩擦エネルギー比など）の拡張が容易に
- バイモーダル仮説の物理的妥当性は、**梃子比** の解析的見積もり（layer 1 半径
  ~ 1.0 mm vs layer 2 半径 ~ 2.0 mm で曲率に対する接触ギャップ変化が 2x 異なる）
  と整合する。完走データが取れたら定量比較すべき
