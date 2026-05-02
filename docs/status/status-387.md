[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-387: (z2) Cosserat 梁 Phase 0 — 設計仕様 + SO(3) Lie 群ユーティリティ公開モジュール

**日付**: 2026-05-02
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12+5+17+11+6+38 passed
（status-386 比 +38 = `xkep_cae/mathematics/tests/test_so3.py`）

## 概要

`solver_mode="explicit"` + UL（更新ラグランジアン）+ CR-Timoshenko 梁の組合せが
status-382/383/385/386 の 4 status 連続検証で **原理的に成立しない** と確定:

- **UL `update_reference` 凍結**（status-382）: 各増分の dynamic lag を reference に
  凍結 → `_ul_internal_force_wrapper(state.u)` で `u_incr ≈ 0` → `f_int(0) = 0` で
  平衡駆動失敗。
- **mass scaling 波速減速**（status-386）: β 倍の質量増加が弾性波速度を `c → c/β`
  へ減速、波伝播時間 `β·L/c` が `t_cycle` を超過すると変形が梁長を横断できない。

UL を捨てた **幾何学的厳密 (geometrically exact / Simo-Reissner) Cosserat 梁** に
移行することで `update_reference` 不要 + 大回転を SO(3) 上のネイティブ更新で扱う
方針へ転換する。本 status はその Phase 0 = 設計仕様 + SO(3) Lie 群／代数の
公開ユーティリティモジュールの確立。

**MCDD 凍結解除条件 (5)** `|u_explicit - u_anal|/|u_anal| < 0.10` 達成への最終
本命路線着手。

## 1. 実装

### 1.1 設計仕様書 — `xkep_cae/elements/docs/cosserat_beam.md`

新設 (+170 行):

- 目的: UL を捨てた explicit + 大回転の本質整合
- 関連 status (z1a/b/c/d) との位置付け表
- DOF: $q_i = (u_i, \theta_i) \in \mathbb{R}^6$、$\Lambda_i \in SO(3)$ は
  $\theta_i$ から指数写像で復元
- ひずみ測度（material frame）: $\boldsymbol{\Gamma} = \Lambda^T r' - e_3$,
  $\boldsymbol{\Omega} = \mathrm{vee}(\Lambda^T \Lambda')$
- 構成則（Phase 0 弾性のみ）: 対角 $C_F = \mathrm{diag}(GA_1, GA_2, EA)$,
  $C_M = \mathrm{diag}(EI_1, EI_2, GJ)$
- 内力の弱形式 + 集中質量行列
- explicit 更新: $\Lambda_{n+1} = \exp_{SO(3)}(\Delta t\, w_{n+1/2})\,\Lambda_n$
- Phase 0〜5 進行計画表
- MCDD 脱法 pattern 10 項回避方針

### 1.2 公開モジュール — `xkep_cae/mathematics/so3.py`

新設 (+371 行):

| 関数 | 役割 |
|------|------|
| `skew(v)` | $\hat{v}$ 歪対称化 |
| `vee(S)` | 歪対称行列 → ベクトル |
| `exp_so3(theta)` | 指数写像（Rodrigues 公式） |
| `log_so3(R)` | 対数写像（四元数経由で安定化） |
| `dexp_so3(theta)` | 右ヤコビアン $T(\theta)$ |
| `dexp_inv_so3(theta)` | 右ヤコビアン逆 $T^{-1}(\theta)$ |
| `compose(R_a, R_b)` | 回転合成（API 明確化用） |
| `batch_skew` / `batch_exp_so3` / `batch_log_so3` / `batch_dexp_so3` / `batch_dexp_inv_so3` | バッチ版 (N, 3) → (N, 3, 3) |

**数値条件**:

- $\phi = \|\theta\| < 10^{-8}$ でテイラー展開（4 次まで）
  - exp: $a \approx 1 - \phi^2/6$, $b \approx 0.5 - \phi^2/24$
  - dexp: $a \approx 0.5 - \phi^2/24$, $b \approx 1/6 - \phi^2/120$
  - dexp_inv: $b \approx 1/12 + \phi^2/720$（Bernoulli 数）
- $\phi \to \pi$ で四元数経由 log（Shepperd 法 + $2\,\mathrm{atan2}(\|\vec{q}\|, q_w)$）

### 1.3 単体テスト — `xkep_cae/mathematics/tests/test_so3.py`

新設 (+289 行)、38 テスト:

| クラス | テスト数 | 検証項目 |
|--------|---------|---------|
| `TestSkewVee` | 8 | 対合 / 反対称 / cross product 整合 / 不正形状 reject |
| `TestExpLog` | 9 | zero/identity 不変 / roundtrip（一般・小角・π 近傍）/ SO(3) 性質 / 軸定位 / 不正形状 |
| `TestDexpAndInverse` | 6 | $T \cdot T^{-1} = I$ / zero・small angle / $T(\theta)\theta = \theta$（軸方向不変） |
| `TestCompose` | 3 | 単位元左右 + 逆元合成 |
| `TestBatchConsistency` | 7 | 単点 vs バッチ数値一致 + バッチ pair 互逆 + 不正形状 |
| `TestParityWithCRBeam` | 5 | `_beam_cr.py` private 関数との数値同値（Phase 4 委譲化への安全網） |

精度: atol=1e-12（roundtrip）/ atol=1e-10（dexp 互逆 batch）/ 機械精度。

### 1.4 CR 梁 private 関数の保持

`xkep_cae/elements/_beam_cr.py` の以下 5 関数は **意図的に変更せず**:

- `_skew`, `_rotvec_to_rotmat`, `_rotmat_to_rotvec`, `_tangent_operator`,
  `_tangent_operator_inv`

理由: status-356 で `test_helical_3d_hermite` の `rel_err = 2.18×10⁻⁷` 機械精度を
達成済の資産。Phase 0 で書き換えると回帰リスクが伴うため、Phase 4 以降の
DRY 化（薄い委譲化）まで現状維持。

`TestParityWithCRBeam` 5 テストが「公開 so3 と private CR 関数群が数値同値」を
保証するため、将来の委譲リファクタが安全に行える。

## 2. 検証

### 2.1 SO(3) 単体テスト

```
$ uv run --extra dev pytest xkep_cae/mathematics/tests/test_so3.py -v
38 passed in 0.28s
```

### 2.2 全体回帰

```
$ uv run --extra dev pytest xkep_cae/contact/ xkep_cae/mathematics/ \
       xkep_cae/time_integration/ \
       xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py
781 passed, 5 skipped, 4 warnings in 87.79s
```

status-386 比 +38（38 SO(3) 新規追加）。

### 2.3 契約検査

```
$ uv run --extra dev python contracts/validate_process_contracts.py
全 24 検査 OK / 契約違反 0 件 / 条例違反 0 件
```

### 2.4 主力 gate `test_helical_3d_hermite`

```
$ uv run --extra dev pytest xkep_cae/contact/contact_force/tests/test_kc_component_fd.py \
       -k helical_3d_hermite
1 passed
```

`rel_err = 2.18×10⁻⁷` 維持。

### 2.5 ruff

```
$ uv run --extra dev ruff check xkep_cae/ tests/
All checks passed!
$ uv run --extra dev ruff format --check xkep_cae/ tests/
205 files already formatted
```

## 3. MCDD 凍結解除条件 — 路線変更

| 条件 | 現状 |
|------|------|
| (1) Phase E 完了 | ✅ status-357 |
| (2) 19 本 frac=1.0 完走 | △ explicit 系で精度 (5) 不足のため事実上未達 |
| (3) max\|u_trans\| < L_strand × 10 | ✅ implicit / 一部 explicit |
| (4) `KcNormalDirectionStiffness` FD rel_err < 1e-2 | ✅ status-356（2.18×10⁻⁷） |
| (5) 解の精度 < 10% | **❌ Cosserat 路線で解決を狙う** |

(z1*) 全候補却下を踏まえ、Phase 0 から段階的に Cosserat 梁プロトタイプを
実装し、Phase 3 で単梁 90° 曲げの解析解 73.30mm に対し精度 gate < 10% を
最初の判定マイルストーンとする。

## 4. 実装変更まとめ

| ファイル | 変更 | 行数 |
|---------|------|-----|
| `xkep_cae/elements/docs/cosserat_beam.md` | 新規 | +170 |
| `xkep_cae/mathematics/so3.py` | 新規 | +371 |
| `xkep_cae/mathematics/tests/test_so3.py` | 新規 | +289 |
| `README.md` | 現在の状態 1 行更新 | (1 行修正) |
| `docs/status/status-index.md` | 387 エントリ追加 | +1 行 |
| `docs/status/status-387.md` | 本ファイル新規 | (本) |
| `docs/roadmap.md` | 現在地 / 次の課題ブロック更新 | (差分) |

実装本体に既存 Process / API の **挙動変更なし**、純粋な追加。
default 動作は完全に保持される。

## 5. 引継ぎ — 次 status の候補

### 5.1 最優先 — Phase 1: `CosseratBeamElementProcess`

**スコープ**:

- `xkep_cae/elements/cosserat/` サブパッケージ新設
- 1 要素 (2 ノード) の弾性内力ベクトル計算（Gauss 1 点積分）
  - 入力: ノード座標 $r_0, r_1$、ノード回転ベクトル $\theta_0, \theta_1$、
    剛性 $C_F, C_M$
  - 出力: 内力 12×1 ベクトル（並進 + 回転）
- 解析接線剛性 12×12 行列 + FD 検証（atol < 1e-6）
- 単体テスト: 直線曲げ / 純せん断 / ねじり / 大回転 + 微小ひずみ整合
- 設計仕様 §1.1〜§1.4 に対応する記述追記

**規模目安**: ~600 行（要素計算 + テスト + 仕様追記）

### 5.2 Phase 2: アセンブラ + explicit 配線

- マルチ要素アセンブラ（`xkep_cae/elements/_beam_assembler.py` 類似）
- `solver_mode="cosserat_explicit"` の追加
- 中央差分 + SO(3) 上の指数更新 `Λ_{n+1} = exp_so3(Δt·w_{n+1/2}) · Λ_n`
- 集中質量行列の構築

**規模目安**: ~500 行

### 5.3 Phase 3: 単梁 90° 曲げ精度 gate 達成判定

- `work/beam_hysteresis/40_cosserat_single_beam.py` 新設
- 解析解 73.30mm に対し `|u_explicit - u_anal|/u_anal < 0.10` 達成すれば
  **MCDD 凍結解除条件 (5) 達成**

### 5.4 副次 — t_cycle 据え置き + n_increments 大の追加実験

`status-386` (#11) で `n_inc=200` で max\|u\|=6.57mm（z1d 方向の 10x 改善）
だったので、`n_inc=2000` 等での precision exploration は短期で実施可能。
ただし UL 凍結の本質欠陥は不変なので gate 達成は楽観できない。

## 6. MCDD 脱法 pattern 回避

| Pattern | 回避策 |
|---------|--------|
| 1: tol 緩和 | `atol=1e-12` 等は Lie 群代数の閉形式整合に基づく数学的水準 |
| 4: rename で済ます | `_beam_cr.py` の private 関数は **未変更**、新規 `so3.py` は API 公開 + バッチ版 + テイラー展開 + Shepperd 法 + parity test を含む substantive 実装 |
| 5: 既存テスト skip | 既存 743 test 全 pass を維持、+38 追加 |
| 6: 骨格 status | Phase 0 = 設計仕様 + 公開モジュール + 38 単体テスト + parity 検証で完結 |
| 7: 数値丸め | テスト精度しきい値は `atol=1e-12` 等で機械精度水準明示 |
| 10: TODO 先送り | Phase 0 を完結させ、Phase 1 を次 status の自然な次ステップに位置付け（独立スコープなので別 status 化が適切） |

## 7. 引継ぎコマンド

```bash
# SO(3) 単体テスト
uv run --extra dev pytest xkep_cae/mathematics/tests/test_so3.py -v

# 全体回帰
uv run --extra dev pytest xkep_cae/contact/ xkep_cae/mathematics/ \
       xkep_cae/time_integration/ \
       xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py
uv run --extra dev python contracts/validate_process_contracts.py
uv run --extra dev ruff check xkep_cae/ tests/ \
   && uv run --extra dev ruff format --check xkep_cae/ tests/

# Phase 1 着手前の確認
ls xkep_cae/mathematics/so3.py
cat xkep_cae/elements/docs/cosserat_beam.md | head -60
```

## 8. 観察 — 開発運用

### 効果的だった点

- **CR 梁 private 関数との parity test 5 件**: status-356 の `rel_err=2.18e-07`
  資産を保護する安全網として機能。将来の DRY 化リファクタで `_beam_cr.py` の
  private 関数を `so3.py` に委譲する際、これらが回帰検出する。
- **テイラー展開の閾値統一**: 全関数で `_SMALL_ANGLE = 1e-8` を共有することで
  数値挙動の一貫性確保（個別関数で異なる閾値を選ぶと小角域で不連続になり得る）。
- **設計仕様で UL/CR 関係を明記**: Phase 0 の段階で「UL を捨てた帰結」を
  4 status 連続却下のサマリと共に文書化することで、Phase 1 以降で迷いなく
  実装方針を保てる。

### 学び — Lie 群アプローチへの移行コスト

Phase 0 の規模 (~600 行) は (z1c) や (g3) と同程度で、新規路線の
**初期投資としては小さい**。Cosserat 梁の各 Phase が ~500-1000 行
（Phase 1〜5 で計 ~3000〜4000 行）なので、(z1*)/(g*)/(p*) で消費した 4
status 分の労力に匹敵する。回り道に見えるが、UL 路線の根本欠陥が確定して
いる以上、ここで Lie 群基盤を整えるのは合理的判断。

### 観察 — 次セッション向け

- Phase 1 では `CosseratBeamElementProcess` 設計時に **解析接線の FD 検証**
  を `atol < 1e-6` で実装すること（status-356 の精神 — 早期に FD パリティを
  取るほどデバッグコストが小さい）。
- `xkep_cae/elements/cosserat/` サブパッケージは `xkep_cae/elements/fiber/` の
  Process 配置パターン（state/section/integrator/strand_beam）を踏襲すると
  良い。
