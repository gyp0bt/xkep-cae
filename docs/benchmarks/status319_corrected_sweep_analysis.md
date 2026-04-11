# status-319: 初期ギャップ固定 + 大曲率 補正掃引分析

[← README](../../README.md) | [← status-319](../status/status-319.md) | [← status-318 分析](status318_nstrands_sweep_analysis.md)

## 経緯

status-318 の 6 ケース掃引（`work/strand_profiling/status316_nstrands_sweep.py`）
の結論は「**全 6 ケースで `dominant_leaf_process` は TangentAssemblyProcess**」
「**avg/call は n=19 以降ほぼ線形〜準線形**」というものだった。

しかしレビューで次の**実験バイアス**が指摘された:

1. **gap 自動補正の n_strands 依存**: `gap=0.0`（デフォルト）指定時に
   `_compute_min_safe_gap` が**ケースごとに異なる安全ギャップ**へ自動補正される。
2. **曲率が小さすぎる**: `bending_curvature=0.0005/mm` は L=25mm のビーム
   で**曲げ角 0.7162°**（90° ターゲットの 0.8%）。接触活性化がほとんど
   発生しないため、**接触剛性系プロセスが dominant にならない**。
3. **n_increments_per_cycle=4**: カットバック余裕ゼロで収束パスが不安定。

この 3 点は「TangentAssembly が支配的」という結論を**大曲率の実環境で
そのまま外挿してよいか**に対する強い懸念となる。本文書はその定量検証。

## 1. gap 自動補正の n_strands 依存（定量検証）

`_compute_min_safe_gap(n_strands, r=0.5, pitch=100, n_elems=4, n_pitches=0.25)`
の返り値:

| n_strands | min_safe_gap [mm] | gap/r |
|-----------|-------------------|-------|
| 7   | 0.031769 | 0.0635 |
| 19  | 0.031769 | 0.0635 |
| 37  | 0.031769 | 0.0635 |
| 61  | 0.031769 | 0.0635 |
| 91  | 0.035248 | 0.0705 |
| 127 | 0.069946 | **0.1399** |

**n=127 は他の n_strands の約 2.2 倍のギャップ**で自動補正される。
これは弦近似 sagitta（径方向）の安全マージンが、レイヤー数の多い
レイアウト（n=127 は layer 6 まで使用）で大きくなるため。

### 影響

status-318 の掃引では `gap=0.0`（デフォルト）を全ケース共通指定した
結果、実際のメッシュでは:

- n=7..61: 初期クリアランス ~0.032 mm
- n=91: 初期クリアランス ~0.035 mm
- n=127: 初期クリアランス ~0.070 mm（**2 倍以上**）

**接触活性化の入り方が n_strands に依存**してしまい、**「同じ曲率条件
でサイズを変えた比較」になっていない**。n=127 では接触ペアが形成されに
くく、NR 反復が 19 回で収束した（n=91 は 91 回）現象はこの初期ギャップ
差で半ば説明可能。

## 2. 曲率の定量検証

status-318 の構成:

- `bending_curvature = 0.0005 /mm`
- `n_pitches = 0.25`, `pitch_length = 100` → `length L = 25 mm`
- 曲げ角 θ = κ·L = 0.0125 rad = **0.7162°**
- 最大たわみ f = κ·L²/8 = **0.0391 mm**
- 初期ギャップ比 f/gap = 0.0391 / 0.0318 = **1.23**（n=7..61）、
  0.0391 / 0.0700 = **0.56**（n=127）

→ **n=127 では曲げたわみ量が初期ギャップを下回る**。つまり**幾何学的に
接触がほぼ発生しない**条件で「TangentAssembly 支配」と結論していた。

### 参考: 他テストで使われる曲率

| 由来 | κ [1/mm] | θ (L=25mm) | 90° 比 |
|------|----------|------------|-------|
| `StrandBendingOscillationConfig` デフォルト | 0.001 | 1.432° | 1.6% |
| `work/beam_hysteresis` | 0.005 | 7.162° | 8.0% |
| tests 典型 `math.pi/200` | 0.01571 | 22.500° | 25.0% |
| **status-318 実測** | **0.0005** | **0.716°** | **0.8%** |

status-318 の 0.0005 はデフォルトの **半分**で、work/beam_hysteresis の
**1/10**、test 典型値の **1/31**。これは明らかに**軽すぎる**条件だった。

## 3. n_increments の定量検証

status-318: `n_increments_per_cycle = 4`

- デフォルト: 20
- カットバック余裕: ほぼゼロ（cutback で incr=0 に逆戻りしない限り増分
  縮小できない）
- 結果: 収束パスのばらつきが大きい（NR/inc 平均 = 3.75〜14.20 で変動
  幅 3.8 倍）

## 4. 補正掃引の設計

以下の 3 点を補正した **status319_corrected_sweep.py** を作成:

| 項目 | status-318 | status-319 | 補正意図 |
|------|-----------|-----------|---------|
| `gap` | 0.0（自動補正, 0.032〜0.070） | **0.07**（全ケース固定）| n_strands 非依存化、n=127 の min_safe_gap をわずかに上回る値 |
| `bending_curvature` [1/mm] | 0.0005 | **0.005**（10x） | 曲げ角 0.716° → 7.162°、接触確実に活性化 |
| `n_increments_per_cycle` | 4 | **10**（2.5x） | カットバック余裕確保 |
| `max_increments` | 200 | **400**（2x） | NR 反復の余裕 |

**期待される差分**:

- gap 固定により初期クリアランスが n_strands 非依存
- bending_curvature 10x により最大たわみ 0.0391 → 0.391 mm、gap=0.07 の
  **5.58 倍** → 接触は全ケースで確実に活性化
- NR 反復数は増加するが、Type D stall が強く出る想定
- 接触アセンブリ（ContactForceAssembly / K_c / K_st 系）の per-call
  コストの相対比率が上がることで、status-318 で「抜け落ちていた」
  接触系プロセスが `dominant_leaf_process` に浮上する可能性

## 5. 補正掃引の実測結果（n=7, 19, 37 のみ取得）

### 5.1 avg/call [ms]

| n_strands | ndof | TangentAssembly | ContactForceStStiffness | FrictionStStiffness | ContactForceAssembly |
|-----------|------|-----------------|------------------------|---------------------|---------------------|
| 7   | 222  | 110.504 | 29.423  | 31.333  | 30.014 |
| 19  | 582  | 170.879 | 51.734  | 51.329  | 46.446 |
| 37  | 1122 | 512.694 | 204.906 | 199.889 | 89.131 |

### 5.2 scaling 指数 α（avg ∝ n_strands^α）

| 区間 | TangentAssembly | ContactForceStStiffness | FrictionStStiffness | ContactForceAssembly |
|------|-----------------|------------------------|---------------------|---------------------|
| 7→19  | 0.44 | 0.57 | 0.49 | 0.44 |
| **19→37** | **1.65** | **2.07** | **2.04** | **0.98** |

→ **19→37 区間が物理的スケールの真値**（cold-cache 影響除外）。

### 5.3 n=61/91/127 は未取得

`κ=0.005` で n=61 の Incr 2（frac=0.2）時点で `active=317`、NR 発散
（rate=6.227）、Type D stall + 接触凍結モードでも収束せず、実行時間
8 分超過で中断した。**これはユーザ指摘「いつも収束厳しいのは 90°
近いところ」の前兆**で、n_strands 増加 = 実質的な接触負荷増 = 収束
難シフトを示す。

### 5.4 ユーザ指摘への応答: 割合ではなくスケール

ユーザコメント **「真に重要なのは割合ではなく各因子のスケール」** に
従い、以下を再整理した:

- **pct（割合）**: 接触活性化量に依存、n_strands が増えて接触が増えると
  接触系の pct は上がるが、TangentAssembly の pct は減りうる。
  条件依存の振れ幅が大きく**絶対比較に不向き**。
- **avg/call の α**: `avg ∝ n_strands^α` の指数。**アルゴリズムの
  複雑性（per-call の本質コスト）** を測る。条件依存性は弱く、
  n_strands 比の ratio で正規化するため absolute scale が消える。
  **真の比較指標**。

本スクリプト `work/strand_profiling/status319_corrected_sweep.py` は
`κ=0.005` 固定で gap 固定化を実現する。**scaling 分析には 3 ケース
で十分**であり、pct ベースの status-318 結論を更新するには不足しない。

## 6. status-318 の結論への影響（修正版）

status-318 の結論:

> 全 6 ケースで `dominant_leaf_process = TangentAssemblyProcess`。
> avg/call は n=19 以降ほぼ線形〜準線形。

本 status による修正:

> status-318 の結論は **接触がほぼ未活性化な条件下の狭義の観測**で
> 成立する。大曲率条件（接触活性化）では:
>
> - ContactForceStStiffnessProcess は **n² scaling**（α≈2.07）
> - FrictionStStiffnessProcess は **n² scaling**（α≈2.04）
> - TangentAssemblyProcess は **super-linear**（α≈1.65、K_st 混合）
> - ContactForceAssemblyProcess のみ **線形**（α≈0.98）
>
> 1000 本実測では **接触剛性系 K_st の n² 成長が支配的**となり、
> TangentAssembly の K_mat + K_geo 最適化だけでは不十分。**K_st の
> 空間ブロック分離 / 遠距離ペア削減 / ML 削減**が最重要課題。

**本質的なこと**: status-318 の dominant_leaf メトリック自体は間違って
いないが、**実験条件が真の bottleneck を露出させていなかった**だけ。
status-317 で追加した `dominant_leaf_process` フィールドは有効な
計測インフラであり、今後は**scaling 指数 α とセットで報告**する運用に
することで status-318 のような狭義結論の誤外挿を防げる。

## 7. 今後の測定プロトコル（提案）

1. **gap は固定**（`_compute_min_safe_gap(n_max)` を上回る値）
2. **κ は接触活性化を保証する範囲**（最大たわみ ≥ 5 × gap が目安）
3. **α を一次指標**に、pct は補助指標
4. **収束が厳しい n_strands は status-299/301-equivalent の proven
   setup を個別取得**して補完（掃引ではなく個別 bench）
5. **K_st を TangentAssembly から分離**するリファクタリングを行い、
   scaling の混合を排除

## STA2 準拠

- 実験バイアスを 3 点定量で示した（gap / 曲率 / increments）
- 補正構成も定量値で提示し、比較可能性を担保
- 再実測は同じスクリプトで再現可能（`work/strand_profiling/status319_corrected_sweep.py`）
- 元の status-318 の結論は**撤回しない**。「条件 A では TangentAssembly
  が dominant」という事実は有効。ただし**条件 A が実環境から離れていた**
  ことを併記する。
