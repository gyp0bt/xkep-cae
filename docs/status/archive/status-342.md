# status-342: 19本撚線 K_c FD 診断取得 — f_c 自体の 115% 不整合 + x 成分支配を実測（仮説 A 再定義）

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-15
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8（変更なし — work/ スクリプト + 計測のみ）

## 概要

status-341 の推奨アクション 1（最優先・新規）を実装・実測。
`work/beam_hysteresis/12_kc_fd_diagnostic_19strand.py` を新規作成し、
`tangent_fd_diagnostic=True` + `type_d_auto_fd=True`（デフォルト）で
19本撚線の Type D stall 中に `TangentFDDiagnosticProcess` を自動発火させ、
**166 件の FD 診断レポートを収集**して stdout 捕捉 → 正規表現パース → CSV 化した。

**三現主義（現場/現物/現実）のもとで得られた実測結果**:

- **K_c 自体が大きく不正確**（`f_c` FD 相対誤差 mean=115%、median=110%、max=191%）
- 不整合方向は **x 成分が支配**（`f_c` comp別不整合 mean **x=68.3% / y=44.2% / z=41.5%**）
- 全体系残差は依然 z 成分支配（K@du comp 別 **z=89.7% / x=40.8%**）
- status-341 で示唆された「z 成分支配 → 広範な K_c 不整合」のうち、**K_c 自体は x が最も狂う**ことが判明

**仮説 A の再定義**: 「StJacobian **z 成分**カップリング不整合」ではなく、
**K_c の x 成分寄与（法線方向 or 幾何項）が primary driver**。
z 成分が全体系残差で支配的に見えるのは、曲げモード幾何（曲率軸 y → 変位主軸 z）
で x 成分不整合が z 方向 NR 修正の阻害に換算されるため（beam coupling）と推察される。

## 実測詳細

### ソルバー結果

```
frac_completed: 0.3743
converged:      False
n_increments:   175
n_cutbacks:     19
elapsed:        530.78 s
```

status-339（baseline）と status-341（n_incr=40 退化）との対比:

| 項目 | status-339 | status-341 (n_incr=40) | **status-342 (本試行, FD診断 ON)** |
|------|------:|------:|------:|
| frac_completed | 0.4839 | 0.1991 | **0.3743** |
| n_increments | 271 | 116 | 175 |
| n_cutbacks | 39 | 11 | 19 |
| elapsed [s] | 534.68 | 154.27 | **530.78** |
| FD 診断レコード数 | 0 | 0 | **166** |

※ FD 診断の発火自体がわずかにソルバー経路を変える可能性があるため、frac が
status-339 と完全一致しないのは想定内。stall 発生条件（Type D 連続 ≥5）は再現。

**再現コマンド**:
```bash
uv run python work/beam_hysteresis/12_kc_fd_diagnostic_19strand.py 2>&1 \
    | tee docs/measurements/kc_fd_diag_19strand_20260415T032250.log
```
（ブランチ: `claude/check-status-todos-X0Kxc`）

### FD 診断統計（n=166）

| キー | mean | min | max | median |
|------|------:|------:|------:|------:|
| `full_rel_err`（全体系 K@du） | 1.004 | 0.9995 | 1.355 | 1.000 |
| **`fc_rel_err`（f_c 単独）** | **1.155** | 0.431 | **1.909** | 1.099 |
| `directional_ratio`（\|\|R(u+εdu)\|\|/\|\|R(u)\|\|） | 1.000 | 0.9995 | 1.000 | 1.000 |
| `deriv_agreement`（\|FD−解析\|/max） | 0.785 | 0.00027 | 1.218 | 0.966 |

- **`full_rel_err ≈ 1.0` は NR 方向が残差を減らせていない**（directional_ratio=1.0 と整合）
- **`fc_rel_err mean=115%` — K_c 自体が 2 倍近く狂う瞬間がある**。これが直接 NR 方向精度を破壊している
- `deriv_agreement` の bimodal 分布（min=2.7e-4 vs median=0.97）は、
  **一部増分では解析/FD 一致するが大多数では極端に乖離**することを示す

### comp 別不整合（百分率シェア、n=166 の平均）

| comp | 全体系 K@du (%) | **f_c 単独 (%)** |
|------|------:|------:|
| x    | 40.8 (max 91) | **68.3 (max 98)** |
| y    |  8.3 (max 37) | 44.2 (max 94) |
| z    | **89.7 (max 97)** | 41.5 (max 99) |
| θx   |  0.7 (max 17) |  0.0 |
| θy   |  3.7 (max 66) |  0.0 |
| θz   |  0.0 (max  3) |  0.0 |

**解釈**:

1. **K_c 単独では x が最悪成分**（68.3%）— status-341 の stall 時 comp=x（72-97%）観察と整合
2. **全体系では z が支配**（89.7%）— 曲げ幾何による 2 次的応答（x の K_c 誤差 → z 方向 NR 修正阻害）
3. **θ（回転）成分は K_c で全てゼロ** — これは解析的に妥当（接触力は並進 DOF のみ直接駆動）
4. `full_comp_tz=0` / `fc_comp_z=41%` など、**K_c z 成分不整合は status-341 の「z 支配」の主因ではない**

## 仮説 A の再定義

### 従来仮説（status-341 まで）

> "StJacobian z 成分カップリング不整合"（status-295 の `K_c_adj mat-only` 化の再検討）

### 新仮説（本 status）

> **K_c の x 成分（法線力または幾何項の x 方向寄与）が primary driver**
>
> 全体系 z 支配は **beam coupling の 2 次効果**（曲率軸 y の曲げで
> 横変位方向 z と法線方向 x が強く連成する幾何）

### 検証ステップ（次セッション推奨）

1. **K_c を mat-only / geo-only / st-only に分解した FD 診断を取得**
   - status-295 で「mat-only が最適（1.8%）、K_st_adj 足すと 38.5% に悪化」と実証済み
   - 本実測の x 成分 68% 不整合が mat-only / geo-only のどちらに由来するか切り分け
2. **`ContactForceStStiffnessProcess` の `K_mat` x 行寄与を FD と突合**
   - `f_c = p_n · n_hat` の n_hat（法線方向） x 成分の微分が最も疑わしい
3. **7本撚線で同診断を取得 → n 依存スケーリング**
   - 7本は完走するため「Type D stall 時の FD 発火」が起きない。代替案:
     `tangent_fd_diagnostic_every_k_steps` のような強制トリガーを追加して
     完走経路中の FD データを取ること（ソルバー設定の小改修が必要）

## 成果物

| ファイル | 内容 |
|---------|------|
| `work/beam_hysteresis/12_kc_fd_diagnostic_19strand.py` | **新規**（265行）— FD 診断自動発火 + stdout 捕捉 + CSV 化 |
| `docs/measurements/kc_fd_diag_19strand_20260415T032250.log` | **新規**（508 KB）— tee full log |
| `docs/measurements/kc_fd_diag_19strand_20260415.csv` | **新規**（35 KB, 166 レコード）— 抽出 CSV |
| `docs/status/status-342.md` | **新規**（本ファイル） |
| `docs/status/status-index.md` | status-342 エントリ追加 |
| `docs/roadmap.md` | 進捗行更新（status-342 反映） |
| `README.md` | 現状行更新 |
| `work/beam_hysteresis/README.md` | 変更なし（script 12 は診断専用のため一覧外） |

### CSV カラム

```
trigger_idx, full_rel_err, fc_rel_err, directional_ratio, deriv_agreement,
full_comp_{x,y,z,tx,ty,tz}_pct,
fc_comp_{x,y,z,tx,ty,tz}_pct
```

## 次セッションへの推奨アクション

### 推奨アクション 1（最優先・新規）: K_c 分解 FD 診断

`ContactForceStStiffnessProcess` の出力を `K_mat` / `K_geo` / `K_st` ごとに
FD と突合する diagnostic Process を追加。x 成分 68% 不整合の由来を
mat/geo/st レベルで切り分ける。

```python
# xkep_cae/verify/ に `ContactKcComponentFDDiagnosticProcess` を新設
# 入力: K_mat, K_geo, K_st（既存の partial 行列）
# 出力: 各成分の FD 相対誤差 + comp 別シェア
```

所要: Process 新設 ~200 行、既存 FD 基盤流用、掃引実行 ~600s。リスク小。

### 推奨アクション 2（既存継続）: gap_cull_threshold 手動掃引

status-341 推奨アクション 2 そのまま継続。並列で走らせる価値あり。

### 推奨アクション 3（条件付き）: 7本撚線 FD 診断経路

完走経路中の `tangent_fd_diagnostic` 強制トリガー（`every_k_incr` オプション）を
`ContactFrictionConfig` に追加し、7本と 19本の同一条件比較を可能にする。
status-342 の x 成分 68% が「19本特有」なのか「K_c の普遍的癖」なのかを判定。

### 非推奨

- ~~n_incr=40 リトライ~~（status-341 で反証済み）
- ~~曲率減 (`bending_curvature`) ~~ — status-341 の C-反証-β メカニズム上、小曲率ほど相対誤差が効いて悪化予測

## 開発運用メモ

- **三現主義の実践**: 仮説 A（status-339 で挙げた「StJacobian z 成分」）は
  実データで検証され、**「z 成分は全体系の 2 次効果、primary は x 成分の K_c」**
  に再定義された。status-341 の「広範な K_c 不整合」示唆は定量化された
- **ソルバー付随機能の活用**: `TangentFDDiagnosticProcess` + `type_d_auto_fd=True`
  により、既存実装のみで 166 レコードの統計が取得できた。新 Process の実装不要で
  実測が完結したのは status-256〜258 / 288-290 の FD 基盤整備の成果
- **CSV と log の docs/measurements 配置**: Codex セッションが独立に再解析できるよう、
  一次データを `/tmp` 外の永続領域に配置。特にログは 508 KB と大きいが、
  `[NR診断]` / `接線剛性FD方向診断` の発火履歴がそのまま保存されており、
  追加解析（例: frac 依存の time series 分析）にも流用可能
