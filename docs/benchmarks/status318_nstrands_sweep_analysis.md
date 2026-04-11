# status-318: n_strands 掃引プロファイリング実測結果（6 ケース版）

[← README](../../README.md) | [← status-318](../status/status-318.md) | [← status-316 (3 ケース版)](status316_nstrands_sweep_analysis.md)

## 概要

status-316 で n_strands = 7 / 19 / 37 の 3 ケースで取得したプロファイル
データを、status-317 で追加した `dominant_leaf_process` フィールド込みで
**6 ケース (7, 19, 37, 61, 91, 127)** に拡張した。撚線本数を六角充填の
1 / 7 / 19 / 37 / 61 / 91 / 127 のうち 7 から 127 までを実測している
（7 = 1 + 6、19 = 7 + 12、…、127 = 91 + 36）。

目的:

1. status-317 の `dominant_leaf_process` フィールドが wrapper を読み飛ばして
   真のボトルネック（葉プロセス）を抽出できることを実データで検証
2. **TangentAssemblyProcess が LinearSolveProcess を抜く転換点**の所在を
   特定（status-316 で示唆された n² 成長の確認）

## 実測構成

| 項目 | 値 |
|------|---|
| n_strands | 7, 19, 37, 61, 91, 127 |
| n_pitches | 0.25 |
| n_elements_per_pitch | 16 |
| n_increments_per_cycle | 4 |
| bending_curvature | 5e-4 / mm |
| contact_enabled | True |
| coating | 無効（Hertz 型/バリア被膜 OFF） |
| 線形ソルバー | scipy `spsolve`（pypardiso 非インストール環境）|
| git_commit | 6dc1433 |
| git_branch | claude/check-status-todos-UdrBD |
| 実行環境 | Linux / Python 3.11.15 / NumPy 2.4.4 / SciPy 1.17.1 |

> **注意**: status-316 のベースラインは pypardiso direct ソルバーで実測
> されていたため、`LinearSolveProcess` が全体の 75% を占有していた。
> 本 status の実測は scipy `spsolve` フォールバック環境のため、
> 線形ソルブの占有率がそもそも違う。**サイズ間の相対傾向**を読むのが目的。

## 集計テーブル

| n_strands | ndof | n_inc | NR反復 | 総時間 [s] | TangentAssembly [s] | TangentAssembly avg/call [ms] | dominant_leaf_pct [%] |
|-----------|------|-------|--------|-----------|---------------------|-------------------------------|----------------------|
| 7   | 222  | 4  | 15  | 2.08  | 0.327  | 21.8  | 4.82  |
| 19  | 582  | 4  | 32  | 7.76  | 3.878  | 121.2 | 12.30 |
| 37  | 1122 | 5  | 47  | 18.66 | 10.898 | 231.9 | 13.49 |
| 61  | 1842 | 10 | 142 | 73.22 | 39.994 | 281.7 | 12.59 |
| 91  | 2742 | 8  | 91  | 75.65 | 40.578 | 446.0 | 12.44 |
| 127 | 3822 | 4  | 19  | 20.08 | 9.891  | 520.6 | 12.30 |

集計元 YAML:
- `ParameterSweepBenchmark_20260411T002531.yaml`（集約サマリ）
- `StrandBendingOscillationProcess_20260411T002213.yaml` (n=7)
- `StrandBendingOscillationProcess_20260411T002215.yaml` (n=19)
- `StrandBendingOscillationProcess_20260411T002223.yaml` (n=37)
- `StrandBendingOscillationProcess_20260411T002242.yaml` (n=61)
- `StrandBendingOscillationProcess_20260411T002355.yaml` (n=91)
- `StrandBendingOscillationProcess_20260411T002511.yaml` (n=127)

## TangentAssembly のスケーリング分析

NR 反復回数（カットバック含む）が n_strands ごとに大きく揺らぐため、
**`avg/call` で正規化**したほうがスケール傾向を読みやすい。

### avg/call スケーリング（ndof 増加比 / TangentAssembly avg 増加比）

| 区間 | ndof 比 | avg/call 比 | スケール |
|------|---------|-------------|----------|
| 7→19   | 2.62x | 5.56x | 超線形（cold cache 影響大） |
| 19→37  | 1.93x | 1.91x | **ほぼ線形** |
| 37→61  | 1.64x | 1.21x | **準線形（sub-linear）** |
| 61→91  | 1.49x | 1.58x | ほぼ線形 |
| 91→127 | 1.39x | 1.17x | **準線形** |

**結論**: TangentAssembly の per-call コストは、n=19 以降では **ndof に対して
ほぼ線形か準線形**に収まっている。status-316 で観測された n² 成長は
*total time* （NR 反復数を含む inclusive）の見かけ上の挙動であり、
**1 反復あたりは status-308〜310 の KD-tree + COO einsum バッチ化で
十分線形化されている**ことが確認できた。

### NR 反復数のばらつき（max_increments=200, lightweight 設定）

| n_strands | n_inc | NR反復 | NR/inc 平均 |
|-----------|-------|--------|------------|
| 7   | 4  | 15  | 3.75 |
| 19  | 4  | 32  | 8.00 |
| 37  | 5  | 47  | 9.40 |
| 61  | 10 | 142 | 14.20 |
| 91  | 8  | 91  | 11.40 |
| 127 | 4  | 19  | 4.75 |

NR 反復数は n=61 で 142 と突出している（カットバック + Type D stall 多発）。
n=91 / n=127 ではむしろ NR 反復が減っており、**収束パスのばらつきが
total elapsed の主要な変動源**になっている。1000 本本実測には
**収束テストの統計的安定性確保**（複数 seed / 平均化）が必要。

## `dominant_leaf_process` の検証結果

status-317 の `dominant_leaf_process` フィールドは **6 ケース全て
TangentAssemblyProcess を抽出**した。

| n_strands | 旧 `dominant_process`（wrapper） | 新 `dominant_leaf_process`（葉） |
|-----------|---------------------------------|---------------------------------|
| 7   | StrandBendingOscillationProcess (30.5%) | TangentAssemblyProcess (4.82%) |
| 19  | StrandBendingOscillationProcess (24.6%) | TangentAssemblyProcess (12.30%) |
| 37  | StrandBendingOscillationProcess (23.1%) | TangentAssemblyProcess (13.49%) |
| 61  | StrandBendingOscillationProcess (23.0%) | TangentAssemblyProcess (12.59%) |
| 91  | StrandBendingOscillationProcess (23.2%) | TangentAssemblyProcess (12.44%) |
| 127 | StrandBendingOscillationProcess (25.0%) | TangentAssemblyProcess (12.30%) |

**status-316 の wrapper 占有問題**（3 wrapper が ~25% ずつ並ぶ）が
本 6 ケースでも完全に再現しており、status-317 の葉抽出が正しく
**インフラストラクチャ wrapper を読み飛ばして真のボトルネック
(TangentAssembly) を特定**していることが確認できた。

### LinearSolveProcess が見えない理由

scipy `spsolve` フォールバック環境では `LinearSolveProcess` が
`profile_breakdown` の上位 5 に入らない。`top_n=10` でも記載が無い（pct < 0.6%）。
これは pypardiso direct ソルブよりも scipy `spsolve` が（小〜中規模問題で）
**TangentAssembly に対して十分速い**ことを意味する。

| 環境 | n=37 LinearSolve | n=37 TangentAssembly | 比 |
|------|------------------|---------------------|-----|
| pypardiso（status-316） | 64.535s (75%) | 12.552s | 5.14x |
| scipy spsolve（本実測） | < 0.5s（top10 圏外） | 10.898s | 0.05x 未満 |

> **示唆**: pypardiso は中規模 (n_strands < 100, ndof < 5000) では
> 起動オーバーヘッドが scipy `spsolve` を上回る可能性。1000 本実測時は
> ndof ~30,000 になるため pypardiso の優位性が明確化する想定。

## 観測上の注意点（次担当向け）

1. **NR 反復数の不安定性**: 同じ問題でも frac 推移により NR 反復数が
   3.75〜14.2 の幅で揺れる。total elapsed をそのままスケール解析に
   使うと **n=127 のほうが n=61 より速い**などの誤読が起きる。
   **avg/call ベースの分析を推奨**。
2. **`dominant_leaf_process` の偽葉判定**: ContactForceStStiffnessProcess /
   FrictionStStiffnessProcess は `target_process` の `uses` グラフから
   静的に到達できない（ContactFrictionProcess が strategy 経由で
   呼ぶため）。n=37 で偶然 5 位に登場したが、`_is_leaf_process` が
   「未知=保守的に葉扱い」とするため誤認はしていない。ただし
   wrapper グラフが不完全な状態であることは記録しておく価値あり。
3. **被膜 OFF 構成**: status-302 で k_coat=1e6 線形バネが数値的正則化
   と判明済み。コアワークロード比較が目的なので妥当だが、被膜 ON
   での挙動（contact + coating + barrier の総合接触剛性）は
   status-318 の TODO（status-302 → status-304 → 被膜幾何接線剛性）
   として残置。

## 1000 本実測へのロードマップ

本 6 ケースの結果から、1000 本（n_strands=1000、ndof ~30,000）の
推定コスト:

- **TangentAssembly**: avg/call が線形成長と仮定 → 1000/127 ≈ 7.87x
  → 0.521s × 7.87 ≈ 4.10s/iter
- **NR 反復数**: 1 サイクルあたり ~50 反復と仮定（保守的） → 200s/cycle
- **n_increments**: 1 サイクルあたり 4 → 800s/cycle = ~13 分
- **6 時間目標** に対する 1 サイクルあたりの余裕: ~28 サイクル分

ただし以下の前提が必要:

1. NR 反復数を <50 に抑える（status-264 以降のチャタリング対策強化）
2. LinearSolve が n_strands=1000 で爆発しない（pypardiso か iterative AMG）
3. メモリ制約: 30,000 × 30,000 sparse Jacobian + factorization

**次のアクション候補**:

1. **被膜 ON 6 ケース掃引** （TODO 残置）
2. **n_strands ≥ 200 で NR 反復数の収束検証**（max_increments=2000 で数 cycle）
3. **pypardiso 環境での再ベンチ**（status-316 の数値と直接比較可能）

## 再現手順

```bash
# 前提: numpy / scipy 必須、pypardiso オプショナル
PYTHONPATH=/home/user/xkep-cae \
    python work/strand_profiling/status316_nstrands_sweep.py \
    2>&1 | tee /tmp/log-status318-$(date +%s).log
```

総実行時間: 約 3.3 分（このスイープ構成、pypardiso 無し scipy spsolve）。

## STA2 準拠

- elapsed は `time.perf_counter()` で測定（profile pct と独立、捏造なし）
- 全ケースの個別 manifest YAML は上記パスで保存済み
- 集約 YAML には dominant_leaf_process / dominant_leaf_pct / dominant_leaf_total
  の 3 フィールドが書き出されている（status-317 で導入）
- ベースライン（status-316）との環境差分（pypardiso 有無）を本書で明示
