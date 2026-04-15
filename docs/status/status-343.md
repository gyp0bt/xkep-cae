# status-343: K_c 成分分解 FD 診断 Process 新設 — K_mat/K_geo/K_st の 4 組み合わせで x 成分 68% 不整合の由来切り分け基盤整備

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-15
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+11（+11、本 status で `ContactKcComponentFDDiagnosticProcess` 単体テスト 11 件追加）

## 概要

status-342 推奨アクション 1（最優先）を実装。
19本撚線 Type D stall で観測された **f_c FD 相対誤差 mean=115% / x 成分 68% 不整合** の
由来を K_mat / K_geo / K_st の部分行列レベルで切り分けるため、
`xkep_cae/verify/kc_component_fd.py` に `ContactKcComponentFDDiagnosticProcess` を新設した。

既存の `TangentFDDiagnosticProcess`（status-256/290）は K_c = K_mat - K_geo + K_st の
**合成接線に対する単一の FD 相対誤差** しか報告しないため、x 成分不整合が
mat / geo / st のどれに由来するかまでは特定できなかった。本 Process は
`ContactForceStrategy.tangent_components()`（status-291 で追加された K_mat/K_geo/K_st 個別返却）
と組み合わせて、4 組み合わせ（`full` / `mat_only` / `mat_geo` / `mat_st`）で FD と突合する。

## 実装詳細

### 新規 Process

**ファイル**: `xkep_cae/verify/kc_component_fd.py`（~240 行）

**クラス**: `ContactKcComponentFDDiagnosticProcess(SolverProcess[In, Out])`

- `meta.name = "ContactKcComponentFDDiagnostic"`, `module = "verify"`, `version = "1.0.0"`
- `document_path = "docs/verify.md"`

**入力** (`ContactKcComponentFDDiagnosticInput`, frozen dataclass):

```python
u: np.ndarray                   # 現在の変位（shape (ndof,)）
du: np.ndarray                  # NR 方向（shape (ndof,)）
compute_contact_force: Callable # u → f_c
K_mat: sp.spmatrix              # 材料剛性
K_geo: sp.spmatrix              # 幾何剛性
K_st: sp.spmatrix               # 滑り剛性
eps: float = 1e-7               # FD 摂動幅
label: str = ""                 # 報告用ラベル
```

**出力** (`ContactKcComponentFDDiagnosticOutput`, frozen dataclass):

```python
full_rel_err:     float  # ||dfc_FD - (K_mat - K_geo + K_st) @ du|| / max(...)
mat_only_rel_err: float  # K_mat @ du 単独との比較
mat_geo_rel_err:  float  # K_mat - K_geo（K_st 除外）
mat_st_rel_err:   float  # K_mat + K_st（K_geo 除外）
share_mat/geo/st: float  # ||K_i @ du|| / ||K_c @ du|| の寄与率
comp_share_full/mat_only/mat_geo/mat_st: dict[str, float]  # 成分別 L2 シェア（%）
report: str              # 人間可読レポート
```

### 設計ポイント

1. **4 組み合わせで由来を特定**: `full` だけでなく `mat_only`（geo/st 除外）・
   `mat_geo`（st 除外）・`mat_st`（geo 除外）を比較することで、
   「K_st を足すと rel_err が悪化 → K_st が primary driver」のような
   論理的切り分けをレポート文字列で明示する（status-295 で手動で行った検証の自動化）。
2. **成分別不整合シェア**: 既存 `TangentFDDiagnosticProcess` の表記
   （x/y/z/θx/θy/θz 各 L2 ノルムを総ノルムで除算、% 単位）に揃え、
   parse 側の正規表現を共用可能にした。各成分 0〜100% の範囲。
3. **frozen dataclass**: Input/Output とも frozen=True で不変性担保。
   診断ログの再現性を保証する（CLAUDE.md STA2 防止ルール準拠）。
4. **線形系のセルフチェック**: 単体テストで `f_c(u) = K_c @ u` の線形系を
   構築し FD / 解析が完全一致（rel_err < 1e-5）を確認。

## テスト追加（11 件）

**ファイル**: `xkep_cae/verify/tests/test_kc_component_fd.py`

| テスト | 目的 |
|--------|------|
| `test_is_solver_process` | SolverProcess 継承確認 |
| `test_meta_name_and_module` | Process メタ情報確認 |
| `test_linear_full_agrees_with_fd` | 線形系で full rel_err ≈ 0（セルフチェック） |
| `test_mat_only_detects_missing_geo_st` | mat-only 単独では誤差を検出 |
| `test_st_primary_driver_isolation` | 真の K_st=0 に誤った K_st を渡すと mat_geo が最良を検証 |
| `test_comp_shares_are_bounded` | 成分別シェアが [0, 100]% に収まる |
| `test_zero_du_returns_default_output` | du=0 早期 return |
| `test_report_contains_key_sections` | label 挿入・ヘッダ検証 |
| `test_share_ratios_non_negative` | 寄与率の非負性 |
| `test_output_type` | 出力型確認 |
| `test_input_frozen` | frozen dataclass 検証 |

全 11 件 PASS（`uv run pytest xkep_cae/verify/tests/test_kc_component_fd.py`）。

## 検証・品質確認

- **単体テスト**: `xkep_cae/verify/` 33 件全 PASS（既存 22 + 新規 11）
- **ruff check**: 新規ファイル + `__init__.py` クリーン
- **ruff format**: 適用済み
- **契約違反**: 0 件 (`contracts/validate_process_contracts.py` GREEN)
- **回帰**: 既存 666 件全 PASS（`test_stress_contour.py::test_process_runs` は
  matplotlib レンダリング依存の環境 pre-existing failure、本変更と無関係)

## 次セッションへの推奨アクション

### 推奨アクション 1（本 Process の実運用）

`work/beam_hysteresis/` に新規スクリプト（例: `13_kc_component_fd_19strand.py`）を
追加し、status-342 と同様に 19本撚線の Type D stall 断面で本 Process を発火させる。
ソルバー内部からは `ContactFrictionProcess` の `tangent_components()` 経路で
K_mat/K_geo/K_st を受け取る必要があるため、以下の 2 パターンが考えられる:

- **パターン A（推奨）**: `ContactFrictionConfig` に `kc_component_fd_every_k_incr`
  フラグを追加し、`type_d_auto_fd` と同等のフックで本 Process を発火。
  レポートを `[K_c成分FD]` タグ付き stdout 行として吐き、
  work スクリプトで正規表現パース → CSV 化。
- **パターン B（軽量）**: Type D stall 終了後に直近 (u, du) を保存し、
  後段で本 Process を直接呼び出す。ただし manager 状態の再現が必要。

パターン A が status-342 の `tangent_fd_diagnostic` フローと整合的。

### 推奨アクション 2（理論整合性の検証）

本 Process を K_c 自己整合（内部整合）検証にも転用する。すなわち、
`f_c(u) = K_mat @ u` のように定義される **トイ問題** を contact_force strategy の
テスト fixture として追加し、`mat_only_rel_err < 1e-6` が常に保たれるかを
回帰テストとして担保する。これは status-291〜296 で特定された
`K_c_adj mat-only` / `K_st_adj` 判断の自動リグレッション防止になる。

### 推奨アクション 3（n=19 vs n=7 の切り分け）

status-342 が提案していた「7本撚線 FD 診断経路」（`tangent_fd_diagnostic_every_k_steps`）の
実装を前提に、本 Process を 7本 vs 19本で同一条件比較。x 成分 68% が
「19本特有」か「K_c の普遍的癖」かを判定する。

## 成果物

| ファイル | 内容 |
|---------|------|
| `xkep_cae/verify/kc_component_fd.py` | **新規**（~240 行）— `ContactKcComponentFDDiagnosticProcess` 本体 |
| `xkep_cae/verify/tests/test_kc_component_fd.py` | **新規**（~230 行、11 テスト） |
| `xkep_cae/verify/__init__.py` | `ContactKcComponentFDDiagnostic*` を公開 API に追加 |
| `xkep_cae/verify/docs/verify.md` | 新 Process の解説追加 |
| `docs/status/status-343.md` | **新規**（本ファイル） |
| `docs/status/status-index.md` | status-343 エントリ追加 |
| `docs/roadmap.md` | 進捗行更新（status-343 反映） |
| `README.md` | 現状行更新（テスト数 +11） |

## 開発運用メモ

- **Process 抽出原則の遵守**: status-295 で手動で行った
  「mat-only vs mat+st vs full の FD 比較」を Process 化することで、
  将来の類似検証を `contracts/` や `tests/` から再利用可能にした
  （CLAUDE.md「機能は可能な限り process クラスとして実装」準拠）。
- **I/O frozen + 再現性**: Input/Output とも frozen dataclass、
  `compute_contact_force` は callable 注入で副作用のない FD 計算を保証。
  status-342 で CSV 化された 166 レコードと同形式の成分別シェアを
  出力し、既存パースパイプラインと互換。
- **contrcts の整合**: `module="verify"` として `VerifyProcess` ではなく
  `SolverProcess` を継承している理由は、本 Process が pass/fail 判定ではなく
  数値診断（既存 `TangentFDDiagnosticProcess` と同カテゴリ）であるため。
  将来 verify 判定を追加する場合は別クラス `ContactKcComponentVerifyProcess` として
  分離予定。
