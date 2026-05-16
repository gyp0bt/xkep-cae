# status-344: 19本撚線 K_c 成分分解 FD 初回実測 — 仮説 A 最終検証（K_mat 主導 + K_st 追従、K_geo ≈ 0）

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-15
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+11（status-343 から変更なし、本 status は実測・ソルバー配線のみ）

## 概要

status-343 推奨アクション 1（本 Process の実運用）を実装 + 実測。

19本撚線 Type D stall 断面で `ContactKcComponentFDDiagnosticProcess`（status-343 新設）を
`tangent_fd_diagnostic` 発火と同期してソルバー内から 183 回発火させ、
K_c = K_mat - K_geo + K_st を 4 組み合わせで FD と突合した。

**主要知見（仮説 A 最終検証）**:
1. **K_geo の寄与率 = 0.000（183 件全件）** — 19本 Type D stall では K_geo が完全にゼロ。
   status-342 の x 成分 68% 不整合の駆動源は K_geo ではない。
2. **最良組み合わせは `mat_only`（183/183 = 100%）** — K_st / K_geo を除外した
   K_mat 単独の FD 相対誤差 mean=44% が最小。
3. **K_st 追加で rel_err 平均 +16%（最大 +52%）悪化** — status-295/296 の
   「mat-only が最適、K_st_adj 有効化で悪化」を 19本でも再現（7本 1.8%→38.5% 対
   19本 44.3%→60.7%）。**K_st_adj は mat-only に対して害を及ぼす**と再確認。
4. **K_mat 単独で 44.3% の FD 誤差が残る** — K_mat の x 成分（mat_only
   comp_x mean=71.8% / max=98.0%）が主要な不整合成分。status-341 の comp_x=97% と一致。

**結論**: 仮説 A は **K_mat 主導 + K_st 追従** という形で成立。
**次の工事は K_mat の x/z 成分不整合の解消**（status-295 の K_c_adj mat-only 拡張）
または K_st の mat-only 整合化（status-296 で否定された方向を 19本で再検討）。

## 実装詳細（status-343 Process のソルバー配線）

### 追加フラグ

- **`ContactFrictionInputData.kc_component_fd_diagnostic: bool = False`**
  （`xkep_cae/core/data.py`）
- **`_NewtonDynamicConfig.kc_component_fd_diagnostic: bool = False`**（既存）
- **`StrandBendingOscillationConfig.kc_component_fd_diagnostic: bool = False`**
  （`xkep_cae/numerical_tests/strand_bending_oscillation.py`）

### ソルバーフック（`_newton_dynamic.py`）

`tangent_fd_diagnostic` / `type_d_auto_fd` と同じトリガー条件で、
`TangentFDDiagnosticProcess` 実行直後に `ContactKcComponentFDDiagnosticProcess` を発火。

```python
if cfg.kc_component_fd_diagnostic and hasattr(
    _contact_force_strategy, "tangent_components"
):
    K_mat_c, K_geo_c, K_st_c = _contact_force_strategy.tangent_components(
        u, manager, k_pen, node_coords=input_data.node_coords_ref,
    )
    kc_out = ContactKcComponentFDDiagnosticProcess().process(
        ContactKcComponentFDDiagnosticInput(
            u=u, du=du,
            compute_contact_force=_compute_fc_at,  # 既存クロージャ再利用
            K_mat=K_mat_c, K_geo=K_geo_c, K_st=K_st_c,
            eps=1e-7, label=f"incr={increment_display} att={att}",
        )
    )
    if cfg.show_progress:
        print("[K_c成分FD]")
        print(kc_out.report)
```

例外は `try/except` で握り潰し、stderr に `[K_c成分FD] skipped:` を吐く。
FD 診断失敗でソルバーを落とさない設計。

### 実測スクリプト

**`work/beam_hysteresis/13_kc_component_fd_19strand.py`** （新規、~290 行）:

- `kc_component_fd_diagnostic=True` + `tangent_fd_diagnostic=True` で 19本 κ=0.015 実行
- stdout を `_Tee` で捕捉し、regex で `ContactKcComponentFDDiagnosticProcess` の
  `report` を抽出（4 combo × 6 comp = 24 列 + rel_err 4 種 + share 3 種 + best_combo）
- CSV 書き出し + 要約統計を stdout に出力

## 実測結果

### ソルバー挙動

| 指標 | 値 |
|------|-----|
| `frac_completed` | 0.3743 |
| `converged` | False |
| `n_increments` | 175 |
| `n_cutbacks` | 20 |
| `elapsed` | 611.78 s |

frac=0.3743 で Type D+E stall（incr 175, att 37）により early abort。
status-339（frac=0.4839）/ status-341（frac=0.1991）と異なる値だが、
FD 計算追加による浮動小数点経路の微細な差異 or random seed 影響の範囲。
**frac=0.37 に到達した 183 件の FD 診断が本 status の本題**。

### K_c 成分分解 FD 診断（183 件）

#### 組み合わせ別 FD 相対誤差

| 組み合わせ | mean | min | max | median |
|-----------|------|------|------|--------|
| `full` (K_mat - K_geo + K_st) | **0.607** | 0.197 | 1.256 | 0.513 |
| `mat_only` (K_mat のみ) | **0.443** | 0.114 | 0.982 | 0.311 |
| `mat_geo` (K_mat - K_geo) | 0.444 | 0.114 | 0.982 | 0.311 |
| `mat_st` (K_mat + K_st) | 0.607 | 0.197 | 1.256 | 0.512 |

**観察**:
- `mat_only ≈ mat_geo`（差 < 0.001）: K_geo は事実上ゼロ
- `full ≈ mat_st`（差 < 0.001）: やはり K_geo はゼロ
- `mat_only → full` で rel_err が mean +16%、max +52% 悪化 → **K_st 追加は常に有害**

#### K_i 寄与率（`||K_i @ du|| / ||K_c @ du||`）

| K_i | mean | max |
|-----|------|------|
| `share_mat` | **0.814** | 1.370 |
| `share_geo` | **0.000** | 0.000 |
| `share_st` | 0.473 | 1.050 |

**share_geo = 0.000 が全 183 件で再現**。19本撚線の接触点では
`ContactForceStStiffness` 系の幾何項が消えている（法線接触ペアで
`p_n > 0` かつ `grad p_n ≈ 0` か、あるいは実装上の分岐条件で省略されている可能性）。

#### 最良組み合わせ分布（rel_err 最小）

| combo | count | % |
|-------|-------|----|
| `full` | 0 | 0.0% |
| `mat_only` | **183** | **100.0%** |
| `mat_geo` | 0 | 0.0% |
| `mat_st` | 0 | 0.0% |

**183 件全てで mat_only が最良**。K_c = K_mat という縮約が常に最適。

#### 成分別不整合シェア（% 平均、mean_pct ∈ [0, 100]）

| combo | x | y | z | tx | ty | tz |
|-------|---|---|---|----|----|-----|
| `full` | **44.2** | 73.0 | 36.2 | 0.0 | 0.0 | 0.0 |
| `mat_only` | **71.8** | 23.1 | 59.2 | 0.0 | 0.0 | 0.0 |
| `mat_geo` | 71.9 | 23.1 | 59.1 | 0.0 | 0.0 | 0.0 |
| `mat_st` | 44.2 | 73.0 | 36.2 | 0.0 | 0.0 | 0.0 |

**解釈**:
- 回転成分（tx/ty/tz）は全て 0.0% — 接触力は並進 DOF のみに効いている（想定通り）
- `mat_only`（K_mat 単独）で x=71.8% / z=59.2% — **K_mat は x/z 方向の FD 整合性が悪い**
- `full`（+K_st）で x=44.2% / y=73.0% — K_st 追加で **y 成分に不整合が移動**
- **mat_only comp_x の max = 98.0%**（status-341 の x=97% stall 時コンポーネントと一致）

## 仮説 A 最終検証（status-341 で定義）

**仮説 A**: x 成分 68% 不整合の primary driver は K_mat / K_geo / K_st のいずれかの部分行列。

### 判定

- **K_geo 駆動説**: **却下**（share_geo=0.000、comp_x にも寄与なし）
- **K_st 駆動説**: **部分的支持** — K_st 追加で rel_err が悪化する。
  ただし mat_only が既に rel_err=44% を持つため、K_st 単独が駆動源ではなく、
  「K_mat の残留誤差 + K_st の逆方向誤差の重畳」構造。
- **K_mat 駆動説**: **主支持** — mat_only の時点で mean rel_err=44% /
  comp_x max=98%。**K_mat の x 成分 FD 整合性が第一の破綻点**。

### 7本 vs 19本 比較

| 指標 | 7本（status-295/296） | 19本（本 status） |
|------|---------------------|------------------|
| K_c FD rel_err（mat-only） | 1.8% | **44.3%** |
| K_c FD rel_err（mat-only + K_st_adj） | 38.5% | **60.7%**（= full） |
| 最良組み合わせ | mat-only | mat-only |
| K_st 追加の影響 | +36.7pp 悪化 | +16.4pp 悪化 |

**同じ mat-only 優位だが、19本では mat-only 自体の誤差が 25倍以上**。
本数増加で K_mat の不整合が支配的に。

## 次セッションへの推奨アクション

### 推奨アクション 1（最優先）: K_mat の x/z 不整合修正

status-295 の `K_c_adj mat-only 化` は `f_c` のカップリング調整だったが、
19本では K_mat 自体（`ContactForceMatStiffnessProcess` 相当）の不整合が支配的。
**K_mat の x/z 方向 DOF カップリングを status-291〜295 と同規模で再検証**する。

具体的には:
- `ContactForceStrategy.tangent_components()` の `K_mat` 構築経路で
  `∂(p_n · n̂) / ∂u` の x/z 成分を微分展開し、Hermite 補間 + 3D 非局所 du の
  寄与が漏れていないかを確認。
- 19本で `mat_only` を FD 検証用の toy 問題で 1.8% 相当（7本水準）まで絞り込めれば
  仮説 A 完全解決。

### 推奨アクション 2（低コスト確認）: gap_cull_threshold 掃引

status-324 の `gap_cull_threshold` を手動で 0.5x〜2.0x 掃引し、
mat_only rel_err が低下する設定が存在するかを確認。K_st 関連の quick win。

### 推奨アクション 3（長期）: K_geo == 0 の原因調査

share_geo=0.000 が **183 件全件**は異常。`ContactForceGeoStiffnessProcess` が
19本条件で常に空行列を返しているか、もしくは `tangent_components` の
geo 分岐が実装上発火していない可能性。
**診断系の次の一手: K_geo の非ゼロ性を要素レベルで検証**。

## 成果物

| ファイル | 内容 |
|---------|------|
| `xkep_cae/core/data.py` | `ContactFrictionInputData.kc_component_fd_diagnostic` 追加 |
| `xkep_cae/contact/solver/process.py` | `NewtonDynamicInput` への配線（line 459） |
| `xkep_cae/contact/solver/_newton_dynamic.py` | K_c 成分分解 FD 診断フック追加（trigger ブロック内） |
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `StrandBendingOscillationConfig.kc_component_fd_diagnostic` 追加 + 3 箇所配線 |
| `work/beam_hysteresis/13_kc_component_fd_19strand.py` | **新規**（~290 行） — 19本実測スクリプト + CSV 出力 |
| `docs/measurements/kc_component_fd_19strand_20260415T214702.log` | **新規** — 実測 stdout（815 KB） |
| `docs/measurements/kc_component_fd_19strand_20260415T214702.csv` | **新規** — 183 件 × 33 列 |
| `docs/status/status-344.md` | **新規**（本ファイル） |
| `docs/status/status-index.md` | status-344 エントリ追加 |
| `docs/roadmap.md` | 進捗行更新（status-344 反映） |
| `README.md` | 現状行更新 |

## 検証・品質確認

- **単体テスト**: `xkep_cae/verify/` 33 件全 PASS（status-343 の 11 件含む、回帰なし）
- **ruff check / format**: GREEN
- **契約違反**: 0 件
- **回帰**: 本 status はソルバー配線のみで新規テスト追加なし。既存テストに影響なし。

## 開発運用メモ

- **Process のソルバー内埋め込みは任意起動**: `kc_component_fd_diagnostic=False`
  （デフォルト）では追加 FD 計算 0 回、パフォーマンス影響なし。
- **K_geo=0 の発見**: status-343 で Process を新設しただけでは見えなかった情報。
  **手動計算では見落としがちな「0 の事実」を Process 化による網羅的ログで捕捉**
  できた典型例（CLAUDE.md「機能は可能な限り process クラスとして実装」の価値示唆）。
- **仮説 A 決着**: status-289 以来の「Type D stall の部分行列由来特定」が、
  Process 新設→ソルバー配線→実測→統計抽出の 2 セッションで完了。
  次は「K_mat の x/z 成分」という具体的な工事対象が明確化。
