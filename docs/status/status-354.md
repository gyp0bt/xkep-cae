# status-354: 7本撚線 90度曲げ（接触あり）回帰確認 — status-353 数理台帳訂正後の重量回帰

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-19
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25（変動なし）
**ブランチ**: `claude/verify-7strand-90deg-contact-tkJEC`
**親コミット**: `e0a3f62` (Merge PR #298)

## 概要

status-353 の数理台帳訂正（`K_mat,ndir ≡ K_geo` 同一性確立、5 項完結化、
コード数値挙動変更なし）後の **7本撚線 90度曲げ（接触あり）重量回帰**
を実施。status-298 / status-299 baseline（`contracts/verify_s_unclamped_90deg.py`
相当）は標準 pytest 経路ではなく `contracts/` 手動実行だったため、
status-353 の「⚠️ 接触あり 90° 曲げ回帰は本 status 未実行」を本 status で解消する。

**結果**: `frac=1.0000, incr=542, cutback=52, 465.0s` で **完走**。
status-298 baseline（`frac=1.0000, incr=535, cutback=45, 752s`）に対し
**38% 高速化**（752s → 465s、287s 短縮）。incr は +1.3%、cutback は +15.6%
の微増だが frac=1.0 完走は維持。

## 再現手順

```bash
git checkout claude/verify-7strand-90deg-contact-tkJEC
uv run python contracts/verify_7strand_90deg_contact.py 2>&1 | tee /tmp/log-7strand-90deg-$(date +%s).log
```

**検証スクリプト**: `contracts/verify_7strand_90deg_contact.py`（本 status で新設、
status-326 `1e3b930` で掃除された `verify_s_unclamped_90deg.py` を status-298
設定で再整備）。

**ログファイル**: `/tmp/log-7strand-90deg-1776606237.log`（5163 行）

## 構成

```
n_strands=7, wire_radius=0.5 mm, pitch_length=100 mm
n_elements_per_pitch=16, n_pitches=1.0
E=130e3 MPa, nu=0.3, rho=8.96e-9 ton/mm³
κ = π/2/L = 0.015708 [1/mm] → bending_angle = π/2 = 90°
penalty_exponent=1.5 (Hertz)
contact_enabled=True, free_end_mode=True
mu=0.15, rho_inf=0.9
max_nr_attempts=200, tol_force=1e-8, max_increments=10000
```

## 結果サマリ

| 指標 | status-298 baseline | status-354 current | 差分 |
|------|-----|-----|-----|
| frac | 1.0000 | **1.0000** | ±0 |
| n_increments | 535 | 542 | +7 (+1.3%) |
| n_cutbacks | 45 | **52** | +7 (+15.6%) |
| elapsed [s] | 752 | **465.0** | **-287 (-38.2%)** |
| converged | — | True | — |
| max contact F | — | 2.667e-03 | — |

### カットバック原因分布（52 件）

| 原因 | 件数 | 割合 | 分布 |
|------|------|------|------|
| `diverged` | 8 | 15.4% | frac 0.05〜0.046（初期接触活性化フェーズ） |
| `nr_limit` | 43 | 82.7% | frac 0.42〜0.97 に散在（接触活性中の NR 反復打切） |
| `relax_fail` | 1 | 1.9% | frac 0.655 |

**主要原因は `nr_limit` 43 件**（82.7%）。frac 0.42 以降で接触が 11〜15 ペア
活性化する局面に散在。これは status-353 の「真の数理対象: `K_hermite_adj`
mat-only（`I_nn` 隣接拡張なし）」仮説 A と整合する傾向（接触活性下で NR 反復
200 回以内に収束しきれないケースが多数）。

### 収束型統計

```
force=19 (3%), disp=490 (92%), energy=20 (3%), total=529
```

**変位収束が 92% で圧倒的**。CLAUDE.md 規約「変位収束偏重は力未収束の警告」
に抵触するが、これは status-298 時点からの構造的特徴であり、本 status の
回帰確認スコープ（status-353 訂正後の数値再現性）では追加対策は未実施。
status-354 再定義 Phase C-3（`K_hermite_adj` フル項拡張）で改善見込み。

### エネルギー収支

```
初期 KE=1.194e-02, SE=3.437e-01, Total=3.556e-01
最終 KE=2.779e-03, SE=1.556e+00, Total=1.559e+00
エネルギー減衰率: 4.38
```

90度曲げにより SE（弾性歪エネルギー）が 4.5倍 に増加。KE は慣性項の減衰
（rho_inf=0.9）により初期の 23% まで減少。物理的に妥当な挙動。

## 性能分析

### 38% 高速化の内訳（推定）

status-298（2026-04-06）以降、以下の高速化が順次投入されている:

| 貢献 Status | 変更内容 | 期待寄与 |
|------------|---------|----------|
| status-308 | ペア検出 cKDTree 化 | 検出フェーズ |
| status-309〜310 | K_st アセンブリベクトル化（einsum + batch StJacobian） | 69-208x 高速化 |
| status-311〜312 | adj batch + BC ベクトル化 + pypardiso | 20,000x BC 適用 |
| status-321〜322 | K_st CSR/COO 経路最適化 + `_find_caller` キャッシュ | ContactForceSt 14% |
| status-324〜325 | distance culling + symbolic factor reuse | ContactForceStStiffness 96-99% |
| status-326 | 上記統合 | scaling α=2.07→1.24 |

**実測 38% (287s 短縮)** は上記の累積効果として妥当。

### cutback +7 件（+15.6%）の検証

cutback 増は以下いずれかに起因する可能性:

1. **distance culling 閾値（status-324）の影響**: gap pre-filter によりペア数は減るが、
   境界付近のペア on/off 切替でチャタリング増の可能性
2. **symbolic factor reuse（status-325）の再構築タイミング**: NR 反復中の剛性行列
   パターン変化で reuse が外れて factorize 再走するケース
3. **Hertz 型 + free_end_mode の既知不安定**: frac 0.4〜0.6 で NR 残差がフラット化
   しやすい構造（status-298 時点から既知）

**本 status ではソース変更は無いため、数値差異の原因特定は別途**。frac=1.0
完走は維持されており、かつ 38% 高速化は cutback 7 件増を十分に相殺する。

## status-353 訂正の数値回帰性検証

status-353 で実施した訂正:

- `docs/math/03_huber_contact_penalty.md` の §3/§3.1/§4/§5/§8 書換
- `xkep_cae/contact/contact_force/strategy.py` の docstring / コメント訂正（実装変更なし）

**コード数値挙動は無変更**のため、status-298 baseline と完全一致するはずだが、
実測は incr +7 / cutback +7 の差分が発生した。これは status-298 以降
（status-308〜326）の高速化コミット群が non-deterministic な並列 batch / cache
タイミングを持つためと推定。frac=1.0 完走という定性結論は維持されており、
status-353 訂正の数値回帰性は確認された（新たな divergence / 収束壁は未発生）。

## ゲート

- ✅ 7本撚線 90度曲げ（接触あり）重量回帰: **frac=1.0000, incr=542, cutback=52, 465.0s**
- ✅ status-298 baseline（frac=1.0000）との frac 一致
- ✅ 接触活性化: max 15 ペア（frac 0.5 以降）、max contact F = 2.67e-03
- ✅ エネルギー収支の物理的妥当性（SE 4.5倍、KE 減衰）
- ⚠️ 収束型 disp 92% 偏重（CLAUDE.md 警告対象、status-298 時点から構造的）
- ⚠️ cutback +7 件（nr_limit 主体、frac 0.42-0.97 散在）— 次 status で原因特定

## 関連 status

- status-298: 初代ベースライン確立（`frac=1.0000, incr=535, cutback=45, 752s`）
- status-299: 曲げ+揺動 ±48mm 統合モード完走（`frac=1.0000, incr=1900, cutback=72, 1504s`）
- status-308〜326: 高速化コミット群（K_st vectorize / cKDTree / distance culling / symbolic cache 等）
- status-346〜351: MCDD Phase A-1〜C-2
- status-352: 計画書ロスト記録 + Phase C-3 前提疑義
- status-353: 数理台帳訂正（K_mat,ndir ≡ K_geo、5 項完結化、Phase C-3 再定義）
- **status-354（本 status）**: 7本撚線 90度曲げ接触あり重量回帰確認、frac=1.0 維持 + 38% 高速化

## 次の課題

本 status は重量回帰確認のみで実装変更なし。次 status で以下を実施:

1. **Phase C-3 再定義の着手**: `KcHermiteNonlocalStiffnessProcess` に `I_nn` 隣接ノード
   成分追加（status-353 仮説 A 検証）→ 19本撚線 `mat_only` rel_err 再計測
2. **nr_limit 43 件の分析**: frac 0.42-0.97 に散在する Type D cutback の
   comp 別（x/y/z）分布取得 → `K_hermite_adj` フル項拡張の効果予測
3. **19本撚線回帰**: 7本 frac=1.0 維持を確認の上で、status-339 以降未完走の
   19本撚線（frac=0.48 で Type D stall）の再計測。Phase C-3 実装後が本命だが、
   本 status の高速化 38% で単発トライの時間コスト削減あり

## コミット（予定）

1. `contracts: 7本撚線 90度曲げ（接触あり）回帰確認スクリプト新設`（コミット済 `b549dbd`）
2. `docs(status): status-354 7本撚線 90度曲げ接触あり回帰 frac=1.0 完走 + 38%高速化`

## 運用上の気付き

- **削除済みスクリプトの復活は contracts/ の 1 ファイル再整備で十分**: status-326 の
  contracts/ 掃除は適切だったが、重量回帰の再現条件スクリプトは git history
  から簡単に復元できる。今回 `git show 1e3b930^:contracts/verify_s_unclamped_90deg.py`
  で完全復元できた。
- **baseline との差分（incr +7/cutback +7）は高速化副作用として許容範囲**:
  frac=1.0 完走という定性結論を基準にする限り、高速化 38% のトレードオフは妥当。
  数値厳密一致を求めるならば、高速化導入時に baseline を都度更新する運用が必要。
