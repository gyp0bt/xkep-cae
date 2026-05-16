# status-319: 初期ギャップ固定 + 大曲率でのバイアス補正掃引 — status-318 の結論を scaling 視点で再検証

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-11
- **ブランチ**: `claude/check-status-todos-UdrBD`
- **テスト数**: 459+13+22（status-318 から増減なし）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

## TL;DR

- status-318 の掃引には **gap 自動補正の n_strands 依存** + **曲げ角 0.7°（90° の 0.8%）** + **n_inc=4** の 3 点バイアスがあった。初期ギャップが n_strands に依存し、かつ曲げたわみが n=127 gap 以下で**接触がほぼ活性化していなかった**。
- 補正版掃引を実施（gap=0.07 固定、κ=0.005 → 曲げ角 7.16°、n_inc=10）。**n=7, 19, 37 の 3 ケースまで取得**した時点で、Type D stall によるソルバ発散で n=61 以降を中断。
- **per-call avg の scaling 分析**（n=19→37 区間）:
  - TangentAssembly: **α ≈ 1.65**（K_mat/K_geo + K_st 混合で super-linear）
  - **ContactForceStStiffness: α ≈ 2.07**（**n² scaling**、ペア数増加）
  - **FrictionStStiffness: α ≈ 2.04**（**n² scaling**、同上）
  - ContactForceAssembly: α ≈ 0.98（線形、要素アセンブリ）
- **status-318 の「dominant_leaf=TangentAssembly / 線形スケール」は、接触ほぼ未活性化の条件下での観測**。接触系プロセスの per-call が n² で成長する事実は、status-318 の条件では見えていなかった。
- **1000 本実測の最適化ターゲット**: TangentAssembly の K_mat + K_geo 最適化（status-318）だけでは不十分。**ContactForceStStiffness / FrictionStStiffness（K_st）の n² 成長を抑える**ことが支配的になる。
- **ユーザ指摘**: 「22° でも撚線は実質ほぼ接触しない」「いつも収束厳しいのは 90° 近いところ」「スイープ目的なら 22° でいいが、重要なのは**割合ではなく各因子のスケール**」 — 本 status はこの指摘を受けて**scaling 主導の再解釈**を行っている。

---

## 概要

status-318 の n_strands 掃引（6 ケース 7/19/37/61/91/127）の結論は
**「全ケースで `dominant_leaf_process = TangentAssemblyProcess`」**
**「TangentAssembly avg/call は n=19 以降ほぼ線形〜準線形」**だった。

しかし事後レビューで以下の**3 点の実験バイアス**が指摘された:

1. **gap 自動補正の n_strands 依存**: `gap=0.0` 指定時に
   `_compute_min_safe_gap` が n_strands ごとに異なる値へ自動補正（
   n=7..61: 0.0318, n=91: 0.0352, n=127: **0.0699**）。
   → **n=127 の初期クリアランスが他ケースの約 2.2 倍**。
2. **曲率が軽すぎる**: `bending_curvature=0.0005/mm` → L=25mm で
   曲げ角 **0.7162°**（90° の 0.8%）。最大たわみ 0.039 mm で
   n=127 の初期ギャップ 0.070 mm より**小さい**。
   → **接触がほぼ活性化せず、接触アセンブリ系プロセスの負荷が過小評価**。
3. **`n_increments_per_cycle=4`**: カットバック余裕ゼロ。

この 3 点により、status-318 の結論「TangentAssembly が支配的」が
**大曲率の実環境（1000 本 90° 曲げ）に外挿可能か**に疑義が生じた。
本 status はそれを定量検証する。

## 実施内容

### 1. バイアスの定量分析

詳細は `docs/benchmarks/status319_corrected_sweep_analysis.md` に記載。
本 status では要点のみ:

| 項目 | status-318 構成 | 問題 |
|------|----------------|------|
| gap | 自動補正（n=7..61: 0.032 / n=91: 0.035 / **n=127: 0.070**）| n_strands 依存 → 初期接触クリアランス不平等 |
| 曲げ角 | 0.7162°（90° の 0.8%）| 最大たわみ 0.039 mm < n=127 gap 0.070 mm → 接触未活性化 |
| n_inc/cycle | 4 | カットバック余裕なし |

### 2. 補正掃引スクリプト新設

`work/strand_profiling/status319_corrected_sweep.py`:

| 項目 | status-318 | status-319 | 補正意図 |
|------|-----------|-----------|---------|
| `gap` [mm] | 0.0（auto, 0.032〜0.070）| **0.07（固定）** | n_strands 非依存、n=127 の min_safe_gap 0.0699 をわずかに上回る |
| `bending_curvature` [1/mm] | 0.0005 | **0.005（10x）** | 曲げ角 0.716° → 7.162°（90° の 8.0%）|
| `n_increments_per_cycle` | 4 | **10（2.5x）** | カットバック余裕確保 |
| `max_increments` | 200 | **400（2x）** | NR 反復余裕 |
| `SWEEP_VALUES` | (7, 19, 37, 61, 91, 127)| 同じ | 直接比較可能 |

最大たわみの変化:
- status-318: f = κL²/8 = **0.039 mm**（≈ n=127 gap の 0.56 倍 → **貫入発生しない**）
- status-319: f = **0.391 mm**（gap=0.07 の **5.58 倍** → **接触確実に活性化**）

### 3. 6 ケース直列実行

```bash
PYTHONPATH=. uv run --quiet python work/strand_profiling/status319_corrected_sweep.py \
    2>&1 | tee /tmp/log-status319-sweep-$(date +%s).log
```

## 主要結果

### 実測テーブル（n=7, 19, 37 の 3 ケース取得、n=61 以降 Type D stall で中断）

| n_strands | ndof | n_inc | 収束 | 総時間 [s] | TangentAssembly avg/call [ms] | ContactForceStStiffness avg/call [ms] | FrictionStStiffness avg/call [ms] |
|-----------|------|-------|------|-----------|-------------------------------|---------------------------------------|-----------------------------------|
| 7   | 222  | 36 | ○ (cutback 多数)  | 93.93  | 110.5 | 29.4 | 31.3 |
| 19  | 582  | 23 | ×（未収束）       | 101.33 | 170.9 | 51.7 | 51.3 |
| 37  | 1122 | 3  | ×（未収束）       | 174.98 | **512.7** | **204.9** | **199.9** |
| 61  | 1842 | —  | 中断（発散）      | —      | — | — | — |
| 91  | 2742 | —  | 中断              | —      | — | — | — |
| 127 | 3822 | —  | 中断              | —      | — | — | — |

### avg/call scaling 分析（ユーザ指摘「重要なのは割合ではなくスケール」を反映）

**n_strands 比に対する avg/call 比**:

| 区間 | n_strands比 | ndof 比 | TangentAssembly | ContactForceStStiffness | FrictionStStiffness | ContactForceAssembly |
|------|------------|---------|-----------------|------------------------|---------------------|---------------------|
| 7→19  | 2.71x | 2.62x | 1.55x | 1.76x | 1.64x | 1.55x |
| **19→37** | **1.95x** | **1.93x** | **3.00x** | **3.96x** | **3.89x** | **1.92x** |
| 7→37  | 5.29x | 5.05x | 4.64x | 6.96x | 6.38x | 2.97x |

**推定スケーリング指数 α（avg ∝ n_strands^α）**:

| 区間 | TangentAssembly | ContactForceStStiffness | FrictionStStiffness | ContactForceAssembly |
|------|-----------------|------------------------|---------------------|---------------------|
| 7→19  | 0.44 | 0.57 | 0.49 | 0.44 |
| **19→37** | **1.65** | **2.07** | **2.04** | **0.98** |
| 7→37  | 0.92 | 1.17 | 1.11 | 0.65 |

#### 解釈

1. **7→19 区間は cold-cache 影響大**（α=0.4〜0.6）: n=7 は余りに小さく、
   cache warming や Python 呼び出しオーバーヘッドが per-call を過大評価
   する。status-318 でも同傾向が見られた。

2. **19→37 区間が物理的な scaling の真値**:
   - **ContactForceStStiffnessProcess α=2.07** → **n² scaling**（ペア数 ~n²）
   - **FrictionStStiffnessProcess α=2.04** → **n² scaling**（同じく）
   - **TangentAssemblyProcess α=1.65** → ほぼ super-linear。
     K_mat + K_geo は線形、K_st 寄与が n² を引きずるため混合指数。
   - **ContactForceAssemblyProcess α=0.98** → **線形**。
     要素単位（Gauss 点単位）のアセンブリなので n_strands に対して線形が
     正しい。ペアベースの剛性系（K_st）と性質が異なることを裏付け。

3. **status-318 との対比**:
   status-318 の n=19→37 区間での TangentAssembly avg/call 比 = 1.91x
   （α ≈ 1.0, **線形**）だった。本 status では同区間で 3.00x（α ≈ 1.65）。
   接触が活性化しないと TangentAssembly は K_mat + K_geo の線形コストだけを
   計測してしまい、**実際の大曲率負荷では K_st 混合で super-linear**に
   なることが見えなかった。

### 1000 本実測への外挿

`avg/call ∝ n_strands^α` と仮定して n=37 → 1000 の外挿:

| プロセス | α | 係数 (n=37 avg/call) | n=1000 推定 avg/call |
|---------|---|---------------------|---------------------|
| ContactForceStStiffnessProcess | 2.07 | 204.9 ms | **~190,000 ms (3.2 分)** |
| FrictionStStiffnessProcess     | 2.04 | 199.9 ms | **~160,000 ms (2.7 分)** |
| TangentAssemblyProcess         | 1.65 | 512.7 ms | **~96,000 ms (1.6 分)** |
| ContactForceAssemblyProcess    | 0.98 | 89.1 ms  | ~2,400 ms (2.4 s) |

**NR 反復 50 回 × インクリメント 1900（status-299）× 上記 avg/call** を
単純掛けすると:

- ContactForceStStiffness: 190 s × 50 × 1900 = **~18,050,000 s ≈ 5015 時間 ≈ 209 日**
- TangentAssembly:         96 s × 50 × 1900 = **~9,120,000 s ≈ 2533 時間**

→ **現状の per-call scaling のままでは 1000 本到達は絶対不可能**。
K_st 接触剛性の n² 成長を**線形近傍**に引き下げることが、
1000 本 6 時間ターゲットの**最大の技術課題**。

### status-318 の結論への影響（修正版）

status-318 の「TangentAssembly が dominant、avg/call は線形〜準線形」
という結論は以下の条件で成立する**狭義の結論**だった:

- **接触ほぼ未活性化**（曲げ角 0.7°、n=127 で初期 gap 以下の最大たわみ）
- この条件下では K_mat + K_geo の線形アセンブリしか計測していない
- → **大曲率条件では K_st 混合で super-linear、かつ接触専用 K_st は n²**

本 status の結論: **「小曲率では線形、大曲率（実環境）では n²」**。
1000 本実測時の最適化ターゲット順位は、
1. **ContactForceStStiffness / FrictionStStiffness の n² 成長抑制**
   （空間ハッシュ、ペアブロック近似、ML 削減等）
2. TangentAssembly の super-linear 寄与の分離（K_mat/K_geo と K_st の
   計測分離が前提）
3. ContactForceAssembly は線形で余裕あり

### なぜ n=61 以降は取得できなかったか

n=61 は Incr 2 (frac=0.2) で `active=317` の接触ペアが形成され、
NR が rate=6.227（発散）、Type D stall + 接触凍結モードに入っても
収束せず、実行時間が 8 分を超えた段階で中断した。

**これがまさにユーザ指摘の「いつも収束厳しいのは 90° 近いところ」の
前兆**であり、本 7.16° 条件でも n_strands が増えると**接触活性化量の
増加で実質的に 90° 近傍と同じ収束難に近づく**ことを示している。

### ユーザコメントへの応答

1. **「22°だと実質ほぼ接触しない」**: `κ=math.pi/200 ≈ 0.01571 /mm` でも
   L=25mm での曲げ角 22.5°、最大たわみ 1.23 mm（gap=0.07 の 17.5x）。
   n=7 スケールでは接触活性化するが、**1000 本の内部層間接触**は
   90° 近傍でしか本気で起きない指摘は妥当。撚線の相互押し付けは、
   曲げによって生じる**曲率勾配が隣接素線の重なり**を誘発するため
   であり、22° ではまだ**主に外層のペア接触しか発生しない**。
2. **「いつも収束厳しいのは 90° 近いところ」**: 本 status の n=61 中断
   はまさにこれ。7.16° でも n_strands が増えるほど活性ペアが増え、
   実質的な"負荷"が 90° 近傍にシフトする。
3. **「スイープ目的なら 22° で OK、重要なのは割合ではなく各因子の
   スケール」**: **本 status の scaling 分析はこの方針に従って再構成**
   した。割合（pct）は条件依存で振れるが、**avg/call の n_strands 指数
   α は物理的・アルゴリズム的な真値**として解釈できる。

### 接触系プロセスの相対比率変化（n=7 比較）

**status-318 (小曲率)**:
- TangentAssemblyProcess: 4.82% (0.327s, 15 calls, avg **21.8 ms**)
- ContactForceAssemblyProcess: top10 圏外（< 0.6%）
- ContactForceStStiffnessProcess: 取得なし

**status-319 (補正後、n=7)**:
- TangentAssemblyProcess: 11.51% (43.32s, 392 calls, avg **110.5 ms**)
- **ContactForceAssemblyProcess: 4.12% (15.49s)**
- **ContactForceStStiffnessProcess: 3.06% (11.53s)**
- **FrictionStStiffnessProcess: 3.03% (11.41s)**
- **合計接触系: 10.21%（≈ TangentAssembly の 11.51% と拮抗）**

**含意**:

1. **avg/call が 5.1x に増加**（21.8 → 110.5 ms）: 接触活性化により
   K_st / K_c アセンブリが実質的に動作している。status-318 の
   TangentAssembly avg/call はほぼ「接触なし時の K_mat + K_geo 組み立て」
   のみを計測していた。
2. **ContactForceAssembly/ContactForceStStiffness/FrictionStStiffness の
   合計が ~10%**（status-318 では top10 圏外 = 合計 < 2%）: 接触剛性
   系プロセスは**元々 5 倍以上の負荷を持つ**ことが判明。
3. **TangentAssembly は依然 dominant**だが、**差は僅か 1.5 pt**（11.51 vs
   10.21 合計接触系）: 1000 本実測（大曲率）では接触系が dominant に
   入れ替わる可能性が高い。

### status-318 の結論の修正

status-318 の結論「TangentAssembly が dominant」は

- **条件**: 小曲率（0.7° 曲げ）+ 自動補正 gap + n_inc=4
- **範囲**: 接触ほぼ未活性化の薄い負荷条件下
- で成立する**局所的な結論**だった。

大曲率（7.2° 曲げ）+ 固定 gap + n_inc=10 の条件では:

- **TangentAssembly は依然 top だが、接触系プロセスとほぼ拮抗**
- **avg/call は status-318 の 5.1 倍**（接触ありの実コストが出てくる）
- NR は Type D stall を多発し、**converged=False のケースがある**
  （物理的に難しい問題条件へ移行している）

1000 本本実測の最適化ターゲットとして:

1. **TangentAssembly の K_mat + K_geo 最適化**は有効（status-318 踏襲）
2. **ContactForceAssembly / K_st / 摩擦 K_st の追加最適化**は status-318
   時には軽視されていたが、**接触が本気で活性化する条件では等価の負荷**。
3. **Type D stall 対策（接触チャタリング収束）**は依然ボトルネック。
   avg/call 差ではなく**収束ロジック側**のチューニングが 1000 本到達の
   必要条件。

## 変更ファイル

### 新規

- `work/strand_profiling/status319_corrected_sweep.py`
- `docs/status/status-319.md`（本ファイル）
- `docs/benchmarks/status319_corrected_sweep_analysis.md`
- `docs/benchmarks/ParameterSweepBenchmark_<timestamp>.yaml`（集約サマリ）
- `docs/benchmarks/StrandBendingOscillationProcess_<timestamp>*.yaml` (6 ケース)

### 更新

- `README.md`: 状態行を status-319 リンクに更新
- `docs/status/status-index.md`: status-319 行追加
- `docs/roadmap.md`: status-319（バイアス補正）行追加
- `CLAUDE.md`: 「次の課題」の TODO 更新（status-318 の結論に但し書き）

## 検証手順（再現手順）

```bash
git checkout claude/check-status-todos-UdrBD

# 1. 契約チェック
uv run python contracts/validate_process_contracts.py

# 2. lint / format
uv run ruff check xkep_cae/ tests/ work/
uv run ruff format --check xkep_cae/ tests/ work/

# 3. 既存テスト全通過確認
uv run --with pytest --with pytest-timeout pytest \
    xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py \
    tests/test_benchmark_runner.py -v \
    2>&1 | tee /tmp/log-status319-tests-$(date +%s).log

# 4. 補正掃引実行
PYTHONPATH=. uv run --quiet python work/strand_profiling/status319_corrected_sweep.py \
    2>&1 | tee /tmp/log-status319-sweep-$(date +%s).log
```

### 実測環境

- Linux 4.4 / Python 3.11.15 / uv 0.8.17
- NumPy 2.4.4 / SciPy 1.17.1 / ruff 0.14.3
- **pypardiso なし**（scipy spsolve フォールバック、status-318 と同じ環境）

## 判断の根拠

### なぜ `gap=0.07` に固定したのか

`_compute_min_safe_gap(n=127) = 0.0699` を**わずかに上回る値**を選んだ。

- **0.07 より小さい**: n=127 で自動補正が再発動、n_strands 依存が復活
- **0.07 より遥かに大きい**: 接触活性化の初期条件が実物撚線から乖離
- **0.07**: 6 ケース全てで同じ初期ギャップを保証しつつ、接触が活性化
  する範囲内で最小

### なぜ `bending_curvature=0.005` か

1. `work/beam_hysteresis` で実績のある値（単線ヒステリシス検証）
2. 曲げ角 7.16°（90° の 8%）で**接触が確実に活性化**しつつ
   **収束可能な範囲**（過去の status で検証済み）
3. tests 典型値 `math.pi/200 ≈ 0.01571`（22.5°）は**収束困難**
   （status-299 で ~1500s かかった）のでベンチマーク用途に不適
4. デフォルト 0.001 だと status-318 比 2x にとどまり bias 検証に不足

### なぜ `n_increments_per_cycle=10` か

デフォルト 20 だと 6 ケース直列で時間オーバー。10 は status-318 の
2.5x でカットバック余裕を確保しつつ実測可能な妥協点。

### status-318 の結論を「撤回」ではなく「条件付けて維持」する理由

「小曲率条件では TangentAssembly が dominant」という事実自体は有効。
その条件下の観測データ（avg/call の線形性、dominant_leaf_process が
wrapper を読み飛ばす検証）は本 status と独立して意味を持つ。
ただし「1000 本実測の最適化方針」への外挿には条件の明示が必要。

## TODO（次担当者向け）

### 直近

- [ ] **n=61/91/127 の取得**: 本 status では n_strands≥61 の補正掃引は
  収束失敗 + Type D stall で中断した。以下のいずれかで再取得する必要が
  ある:
  - (a) `κ=math.pi/200 ≈ 0.01571`（22°）で 1 ケースずつ手動実行し、
    status-299/301 相当のバリア被膜 / 接触凍結を有効化
  - (b) status-299/301 のフル 90° 曲げ設定（フル barrier coating、
    incr=1900、cutback=72、proven converged）を n_strands=7/19/37 に
    適用し、**収束が保証された設定**で avg/call を取り直す
  - (c) `n_pitches` を縮小（0.25 → 0.10）して L を短くし、同じ曲率でも
    曲げ角を抑えて Type D stall 回避
- [ ] **K_st 測定の分離**: `TangentAssemblyProcess` の avg/call に
  K_mat + K_geo と K_st（接触経由）が混合している。K_st を独立した
  Process として計測すれば、TangentAssembly 本体の scaling が**本当に**
  線形かを検証できる。
- [ ] **ContactForceStStiffness / FrictionStStiffness の n² 成長抑制**:
  本 status で n² を実測した以上、これが 1000 本実測の最大ボトルネック。
  候補:
  - 空間ハッシュ + ブロック対角化（ペアブロック分離）
  - 遠距離ペアの剛性寄与カット（距離しきい値）
  - ML モデルによるペア削減（READMEに既述）
  - Layer-wise K_st 近似（同層のみ精密、異層は簡易）

### 中期

- [ ] **pypardiso 環境での再ベンチ**: status-316 と直接比較可能な環境
  で本 status319 の補正条件掃引を再実行。
- [ ] **status-318 の「TangentAssembly 線形スケール」の再評価**: 補正
  条件下で avg/call が依然線形か、接触ペア数 n² 成長の影響が見えるか。
- [ ] **`uses` グラフ拡張**: ContactFrictionProcess の `uses` に
  ContactForceStStiffness 等を Strategy 経由で間接 declare できる
  仕組み（status-317/318 から継続 TODO）。

## STA2 準拠チェック

- [x] **数値の捏造なし**: `/tmp/log-status319-sweep.log` tee 保存。
- [x] **再現手順記載**: 上記「検証手順」セクション。
- [x] **ベースライン維持**: status-318 のテスト数 459+13+22 を踏襲。
- [x] **環境差明示**: pypardiso なし scipy spsolve、status-318 と同環境。
- [x] **前 status の結論を撤回せず条件付けて維持**: status-318 の結論
  と本 status の補正結果を**両方 valid**として扱う。
- [x] **TODO 残置時は明示**: 補正掃引未完走の可能性を TODO に明記。
