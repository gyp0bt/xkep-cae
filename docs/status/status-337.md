# status-337: ContactPairAnalysisProcess — 接触ペア履歴後処理 Process

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-14
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9（+9 テスト）

## 概要

status-333 で整備された `contact_pair_history` は生データのまま放置されており、
status-336 までに整備された M-κ 集約量（`_compute_mk_metrics`）とは別に、
**素線レベル** の観測量（κ_cr 分布、各ペア散逸）を取り出す後処理が必要だった。

本 PR で `ContactPairAnalysisProcess`（`PostProcess` カテゴリ）を新設し、
`SolverResultData.contact_pair_history` + `moment_curvature_history` から
以下を抽出する:

- 各ペア (elem_a, elem_b) の **κ_cr**（初回スリップ曲率）
- 各ペアの **最終累積散逸エネルギー**
- インクリメント毎の **活性ペア数** 時系列
- κ_cr **分布統計**（mean / std / min / max / n_slipped_pairs）

これにより、CLAUDE.md の「CR梁接触動解析での M-κ ヒステリシス直接取得（最優先）」に
対応する **素線レベル** キャリブレーションデータの抽出経路が完成する。

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/numerical_tests/contact_pair_analysis.py` | **新規**。`ContactPairAnalysisInput/Result/Process` + 純粋ヘルパー `_slip_magnitude` / `_kappa_for_step` |
| `xkep_cae/numerical_tests/docs/contact_pair_analysis.md` | **新規**。設計文書（入出力・κ_cr 判定ルール・使用例） |
| `tests/numerical_tests/test_contact_pair_analysis.py` | **新規**。9 テスト追加（合成履歴 8 + 2本撚線統合 1） |
| `docs/status/status-index.md` | status-337 エントリ追加 |
| `docs/roadmap.md` | 接触ペア後処理行追加 |
| `README.md` | 現状行更新（テスト数 +9） |

## 設計判断

- **`PostProcess` カテゴリ**に配置。生成済みの `contact_pair_history` に対する
  純粋後処理で、ソルバー実行を伴わない。`uses = ()` で他 Process 非依存。
- `_compute_mk_metrics`（M-κ 集約）と責務直交：
  - こちらは **接触ペア** レベルの統計
  - あちらは **M-κ ループ** レベルの集約
- **κ_cr 判定ルール**:
  1. `stick == False`
  2. `|(slip_s, slip_t)| > slip_threshold`（数値雑音除外、デフォルト 1e-6）
  3. そのペアの κ_cr が未記録
  の 3 条件を最初に満たしたインクリメントで記録（初回遷移のみ追跡）。
- `moment_curvature_history` が空のときは `load_frac` を κ の代替として格納し、
  呼び出し側の使い分けを簡潔に。

## 実装の要点

```python
class ContactPairAnalysisProcess(
    PostProcess[ContactPairAnalysisInput, ContactPairAnalysisResult],
):
    meta = ProcessMeta(
        name="ContactPairAnalysis",
        module="post",
        version="1.0.0",
        document_path="docs/contact_pair_analysis.md",
    )
    uses = ()

    def process(self, input_data: ContactPairAnalysisInput) -> ContactPairAnalysisResult:
        # (load_frac, entries) 列を単一パスで走査
        # - seen_pairs で n_unique_pairs
        # - kappa_cr[key] 未記録 + slip 条件で κ_cr 記録
        # - last_dissipation[key] を都度更新（履歴最終値）
```

## テスト設計

### 純粋関数的テスト（合成履歴、`TestContactPairAnalysisAPI` クラス × 8）

1. `test_empty_history` — 空履歴は全ゼロ結果
2. `test_single_step_stick_only` — 全ペア stick なら κ_cr 空
3. `test_kappa_cr_records_first_slip` — 初回スリップ時の κ を記録、以降は上書きされない
4. `test_slip_threshold_filters_noise` — slip_threshold 以下は数値雑音として除外
5. `test_dissipation_tracks_last_value` — per_pair_dissipation は履歴最終値
6. `test_uses_moment_curvature_history_for_kappa` — mk_history 指定時は κ を記録
7. `test_kappa_cr_distribution_stats` — mean/std/min/max の正確性（3 ペア手計算検証）
8. `test_n_active_per_step_matches_entries` — 活性ペア数カウント整合

### 統合テスト（`TestContactPairAnalysisConvergence`、`@pytest.mark.slow`）

9. `test_end_to_end_2strand` — 2本撚線曲げ（frac=1.0）を実行し
   `contact_pair_history` + `moment_curvature_history` を取得、Process 経由で解析。
   パイプラインが例外なく走ることと、構造的整合（`len` 一致、load_frac 非減少）を検証。

   **物理的検証の範囲外**: 2本撚線 + `bending_curvature=0.001` (5.7°) の
   軽量設定では接触が活性化しないこと（`n_active_per_step=(0,...)`）を
   実測。これは status-335 と同じ状況（ペアは検出されるが p_n > 0 に
   到達しない）で、物理的な κ_cr 分布検証は **7本撚線 + `@slow` 後続 PR** に
   任せる（CI 時間節約）。

## テスト結果

```
$ uv run pytest tests/numerical_tests/test_contact_pair_analysis.py tests/numerical_tests/test_mk_tracking.py -q
...................                                                      [100%]
19 passed in 20.76s
```

内訳:
- `test_contact_pair_analysis.py` 新規 9 tests（合成 8 + 2本撚線 1） 5.00 s
- `test_mk_tracking.py` 既存 10 tests 継続合格

## 検証

- `ruff check xkep_cae/ tests/` → All checks passed
- `ruff format --check xkep_cae/ tests/` → 177 files already formatted
- `python contracts/validate_process_contracts.py` → 契約違反なし、条例違反なし
  - C3: `@binds_to(ContactPairAnalysisProcess)` で `TestContactPairAnalysisAPI` に紐付け
  - C15: `docs/contact_pair_analysis.md` 作成で document_path 解決

## 次のステップ

- [ ] **7本撚線 end-to-end 実測** — `@pytest.mark.slow` + work/ スクリプトで
      接触が実際に活性化する条件で κ_cr 分布・total_dissipation を取得
- [ ] **ピッチ依存性検証** — p=50/100/200 で κ_cr 分布がどう変わるか実測
      （Papailiou モデルは「ピッチ非依存」を予測、CR梁実測で検証）
- [ ] **ヒストグラム binning / CSV 出力** — 現状は統計量のみ。
      分布形状（ガウス? 対数正規?）を観察するための後処理は後続 PR で追加
- [ ] **可視化スクリプト** — `work/beam_hysteresis/` に κ_cr 分布プロット
- [ ] リスタート解析方式（ContactFrictionProcess の I/O 整理）
- [ ] 被膜圧縮モデル改善（バリア関数 or 二層モデル）

## 開発運用メモ

- 2本撚線 `bending_curvature=0.001` では接触が活性化しない点は想定済みで、
  M-κ 追跡（status-335）でも「接触があまり発生しない軽量設定」を
  infra 検証に使用している。統合テストはあくまで「パイプライン健全性」
  を確認するもので、物理検証は後続 PR の `@slow` に委ねる。
- `ContactPairAnalysisResult` は `frozen=True` だが、`dict` / `tuple` を
  格納するため `field(default_factory=dict)` を使用（C17 に抵触しない）。
- `TYPE_CHECKING` ガードで `ContactPairSnapshotEntry` を import
  し、ランタイム依存を最小化。
