# status-318: n_strands 掃引 6 ケース拡張 — `dominant_leaf_process` 実測検証 + TangentAssembly avg/call 線形性確認

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-11
- **ブランチ**: `claude/check-status-todos-UdrBD`
- **テスト数**: 459+13+22（status-317 から増減なし）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

---

## 概要

status-317 で導入した `dominant_leaf_process` フィールドを実データで検証する
ため、`work/strand_profiling/status316_nstrands_sweep.py` の `SWEEP_VALUES`
を `(7, 19, 37)` から **`(7, 19, 37, 61, 91, 127)`** に拡張して 6 ケースを
直列実行した。これは status-317 の TODO「100/200/500 本掃引拡張
（TangentAssembly 転換点特定）」の **第 1 段階**にあたる。

## 実施内容

### 1. スイープスクリプト拡張

`work/strand_profiling/status316_nstrands_sweep.py`:

- `SWEEP_VALUES` を 3 → 6 ケースに拡張（六角充填の 7/19/37/61/91/127）
- 出力テーブルを `dominant_leaf_process` / `dominant_leaf_pct` /
  `dominant_leaf_total` で表示するように変更（status-317 の新フィールド）
- モジュール docstring に status-318 を追記し、新フィールドの目的を明示

### 2. 6 ケースを直列実行

```bash
PYTHONPATH=. python work/strand_profiling/status316_nstrands_sweep.py \
    2>&1 | tee /tmp/log-status318-sweep.log
```

総実行時間: **198.32 秒**（pypardiso 無し scipy spsolve 環境）

### 3. 分析ドキュメント新規作成

`docs/benchmarks/status318_nstrands_sweep_analysis.md`:

- 6 ケースの集計テーブル + avg/call 正規化スケール解析
- `dominant_leaf_process` 検証結果（全 6 ケースで TangentAssemblyProcess を抽出）
- LinearSolveProcess が top10 に出てこない理由（pypardiso 非インストール環境）
- 1000 本本実測へのロードマップ + 推定コスト

---

## 主要結果

### dominant_leaf_process の抽出検証

| n_strands | 旧 `dominant_process`（wrapper） | 新 `dominant_leaf_process`（葉） |
|-----------|---------------------------------|---------------------------------|
| 7   | StrandBendingOscillationProcess (30.5%) | **TangentAssemblyProcess** (4.82%, 0.327s) |
| 19  | StrandBendingOscillationProcess (24.6%) | **TangentAssemblyProcess** (12.30%, 3.878s) |
| 37  | StrandBendingOscillationProcess (23.1%) | **TangentAssemblyProcess** (13.49%, 10.898s) |
| 61  | StrandBendingOscillationProcess (23.0%) | **TangentAssemblyProcess** (12.59%, 39.994s) |
| 91  | StrandBendingOscillationProcess (23.2%) | **TangentAssemblyProcess** (12.44%, 40.578s) |
| 127 | StrandBendingOscillationProcess (25.0%) | **TangentAssemblyProcess** (12.30%, 9.891s) |

**status-317 の葉抽出が wrapper を完全に読み飛ばし、6 ケース全てで
TangentAssemblyProcess を真のボトルネックとして抽出している**。
status-316 の wrapper 占有問題（3 wrapper が ~25% で並ぶ）が完全に
解消されることを実証。

### TangentAssembly の per-call スケール

NR 反復数の不安定性（後述）を排除するため avg/call で正規化:

| n_strands | ndof | TangentAssembly avg/call [ms] |
|-----------|------|-------------------------------|
| 7   | 222  | 21.8  |
| 19  | 582  | 121.2 |
| 37  | 1122 | 231.9 |
| 61  | 1842 | 281.7 |
| 91  | 2742 | 446.0 |
| 127 | 3822 | 520.6 |

ndof 比に対する avg/call 比:

| 区間 | ndof 比 | avg/call 比 | スケール |
|------|---------|-------------|----------|
| 7→19   | 2.62x | 5.56x | 超線形（cold cache）|
| 19→37  | 1.93x | 1.91x | **ほぼ線形** |
| 37→61  | 1.64x | 1.21x | **準線形** |
| 61→91  | 1.49x | 1.58x | ほぼ線形 |
| 91→127 | 1.39x | 1.17x | **準線形** |

**結論**: status-308〜310 の最適化（KD-tree, einsum バッチ COO,
摩擦 K_st ベクトル化）により、TangentAssembly の per-call コストは
**n=19 以降で ndof に対して線形〜準線形**に収まっている。
status-316 で観測された n² 成長は inclusive（NR 反復数を含む）の
見かけ上の挙動だった。

### NR 反復数の不安定性

| n_strands | n_inc | NR反復 | NR/inc 平均 |
|-----------|-------|--------|------------|
| 7   | 4  | 15  | 3.75 |
| 19  | 4  | 32  | 8.00 |
| 37  | 5  | 47  | 9.40 |
| 61  | 10 | 142 | 14.20 |
| 91  | 8  | 91  | 11.40 |
| 127 | 4  | 19  | 4.75 |

**n=61 が突出して 142 反復**。Type D stall + カットバックが多発した。
**n=127 のほうが n=91 / n=61 より速い**という逆転現象は NR 反復数の
ばらつきによるもので、コア計算の per-call スケールではない。

### LinearSolveProcess は top10 圏外

scipy `spsolve` フォールバック環境では LinearSolveProcess が top10 に
入らない（pct < 0.6%）。status-316 の pypardiso 環境では 75% 占有
だったため、**ソルバー選択で完全に逆転**している。

> **示唆**: pypardiso direct solver は中規模 (ndof < 5000) では
> 起動オーバーヘッドが scipy spsolve を上回る可能性。1000 本実測時は
> ndof ~30,000 になるため pypardiso の優位性が明確化する想定。
> **環境差を明示した上で**サイズ間の相対傾向のみを比較するのが
> 正しい読み方。

---

## 変更ファイル

### 更新

- `work/strand_profiling/status316_nstrands_sweep.py`
  - `SWEEP_VALUES` を 3 → 6 ケース拡張
  - 出力テーブルを `dominant_leaf_*` 表示に変更
  - module docstring に status-318 追記
- `README.md`: 状態行を status-318 リンクに更新（2026-04-10 → 2026-04-11）
- `docs/status/status-index.md`: status-318 行追加
- `docs/roadmap.md`: 完了テーブルに 6 ケース掃引行追加
- `CLAUDE.md`: 「次の課題」の TODO チェック更新

### 新規

- `docs/status/status-318.md`（本ファイル）
- `docs/benchmarks/status318_nstrands_sweep_analysis.md`（分析ドキュメント）
- `docs/benchmarks/ParameterSweepBenchmark_20260411T002531.yaml`（集約サマリ）
- `docs/benchmarks/StrandBendingOscillationProcess_20260411T002213.yaml` (n=7)
- `docs/benchmarks/StrandBendingOscillationProcess_20260411T002215.yaml` (n=19)
- `docs/benchmarks/StrandBendingOscillationProcess_20260411T002223.yaml` (n=37)
- `docs/benchmarks/StrandBendingOscillationProcess_20260411T002242.yaml` (n=61)
- `docs/benchmarks/StrandBendingOscillationProcess_20260411T002355.yaml` (n=91)
- `docs/benchmarks/StrandBendingOscillationProcess_20260411T002511.yaml` (n=127)

---

## 検証手順（再現手順）

```bash
git checkout claude/check-status-todos-UdrBD

# 1. 契約チェック
uv run python contracts/validate_process_contracts.py

# 2. lint / format
uv run ruff check xkep_cae/ tests/
uv run ruff format --check xkep_cae/ tests/

# 3. 既存テスト全通過確認（status-317 のテスト数 459+13+22）
uv run --with pytest --with pytest-timeout pytest \
    xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py \
    tests/test_benchmark_runner.py -v \
    2>&1 | tee /tmp/log-status318-tests-$(date +%s).log

# 4. 6 ケース掃引再実行（再現確認）
PYTHONPATH=. uv run --quiet python work/strand_profiling/status316_nstrands_sweep.py \
    2>&1 | tee /tmp/log-status318-sweep-$(date +%s).log
```

### 実行結果

- 契約違反 0 件、条例違反 0 件（変化なし）
- ruff check / format: All checks passed
- 既存テスト: 全通過（変更なし）
- 掃引総実行時間: **198.32 秒**

### 実測環境

- Linux 4.4 / Python 3.11.15 / uv 0.8.17
- NumPy 2.4.4 / SciPy 1.17.1 / ruff 0.14.3
- **pypardiso なし**（scipy spsolve フォールバック）
- git_commit: 6dc1433 → status-318 コミット

---

## 判断の根拠

### なぜ "100/200/500 本" ではなく "127 まで" にしたのか

status-317 の TODO は「100/200/500 本」と書かれていたが、本文では
具体的に `SWEEP_VALUES = (7, 19, 37, 61, 91, 127)` を例示していた。
**六角充填の系列** (1, 7, 19, 37, 61, 91, 127, ...) は撚線層構造
の自然なステップサイズで、計算量が滑らかに増えるため、プロファイル
傾向の観測に適する。本実測では計算リソース（197s）の枠内で 127 まで
到達できた。1000 本到達には別途 status を立てて pypardiso 環境 +
スクリーン解像度の大きい設定での再実測が必要。

### なぜ NR 反復回数を統計的安定化しなかったのか

複数 seed 実行や `n_pitches` を増やした重い構成にすると per-case
の実行時間が線形に増えるため、本 status の目的（dominant_leaf_process
の動作検証）には過剰だと判断。ただし「NR 反復数のばらつきが elapsed
の主要な変動源」という観測は記録し、次の status でアドレスする余地
を残した。

### なぜ被膜 ON でも実施しなかったのか

被膜 ON は status-302 で「k_coat=1e6 線形バネは数値的正則化、物理的
被膜モデルではない」と判明している。被膜物理を本格的に検証するなら
バリア関数 or 二層モデルへの移行が前提となる。プロファイリング目的
での被膜 ON 掃引は、物理モデル更新後に実施するのが効率的。

### `dominant_leaf_process` の偽葉判定について

`ContactForceStStiffnessProcess` / `FrictionStStiffnessProcess` は
`StrandBendingOscillationProcess` の `uses` グラフから到達できない
（ContactFrictionProcess が strategy slot 経由で呼ぶため）。
status-317 の `_is_leaf_process` は「未知=保守的に葉」とするので
誤って wrapper 扱いされない。これは設計通りの挙動だが、wrapper graph
の不完全さを示している。**動的 exclusive 時間記録（status-317 中期
TODO）への移行で根本解決可能**。

---

## TODO（次担当者向け）

### 直近

- [ ] **`uses` グラフ拡張**: `ContactFrictionProcess.uses` に
  `ContactForceStStiffnessProcess` / `FrictionStStiffnessProcess` を
  Strategy 経由で間接 declare する仕組み（StrategySlot に `uses`
  を持たせる等）を検討。dominant_leaf 判定の精度向上に直結する。
- [ ] **NR 反復数安定化テスト**: n=37 / 61 / 91 で同じ問題を 3〜5 回
  反復実行して NR 反復数の標準偏差を測る。bench 結果のエラーバー
  推定に必要。
- [ ] **被膜 ON プロファイル取得** （継続 TODO）: 被膜物理モデル
  改善（バリア関数 or 二層モデル）後に同 6 ケース掃引を再実行。
- [ ] **被膜幾何接線剛性（∂n/∂u, ∂s/∂u）の実装** （継続 TODO）:
  status-304 で FD 誤差 67% の主因と判明した項。

### 中期

- [ ] **pypardiso 環境ベンチ**: status-316 と直接比較可能な環境で
  6 ケース掃引を再実行し、線形ソルバーの転換点を再特定する。
- [ ] **n_strands ≥ 200 への拡張**: pypardiso + max_increments=2000
  の本格構成で 1000 本に近づける。
- [ ] **動的 exclusive 時間記録** （status-317 から継続）: 静的 `uses`
  グラフではなく実測 self-time で leaf 判定する API を ProcessMetaclass
  に追加。
- [ ] **リスタート解析方式への移行**（CLAUDE.md「次」項目）

### 開発運用メモ

- **効果的**: status-317 で `dominant_leaf_process` を実装し、
  本 status でそれを実データで検証する流れがスムーズに進んだ。
  **新フィールド導入 → 即実データ検証**の 2-status 構成は再現性
  を高める良い運用パターン。
- **効果的**: avg/call 正規化で per-call スケール傾向を抽出すると
  NR 反復数の不安定性に依存しない解釈が可能。今後のプロファイル
  分析でも採用を推奨。
- **注意**: pypardiso 有無で LinearSolveProcess の比重が劇的に
  変わる（75% → < 0.6%）。**ベンチマーク比較時は環境差を必ず明示**。
  CLAUDE.md の STA2 ルール（変更前ベースライン取得）を pypardiso
  バージョンや環境フラグレベルで運用するルールに拡張すべき。
- **注意**: NR 反復数のばらつきが大きい問題では elapsed total を
  そのままスケール解析に使うと **n=127 のほうが n=91 より速い**
  などの誤読が起きる。avg/call ベースの観測を一次指標にする運用
  ルールを統合すべき。

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: `/tmp/log-status318-sweep.log` に tee 保存。
      `docs/benchmarks/ParameterSweepBenchmark_20260411T002531.yaml`
      に実測値が記録されている。
- [x] **再現手順記載**: 上記「検証手順（再現手順）」セクション。
- [x] **ベースライン維持**: status-317 の 459+13+22 テストはそのまま
      通過。新規追加なし。
- [x] **環境差明示**: pypardiso 有無を実測表に明記。
- [x] **tee ログ出力**: 上記コマンドで明示。
