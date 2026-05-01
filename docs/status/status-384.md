[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-384: 候補 (z1a) 要素ごと波速 Δt + (z1b) selective mass scaling — 実装完了、validation で「2 段階スケーリング」要件を発見

**日付**: 2026-05-01
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12+5+17 passed（status-383 比 +17 = z1a 6 + z1b detect 5 + z1b apply 6）

## 概要

ユーザーから「応力波の速度と要素サイズから dt 目安を決められる（Abaqus 様式）」+
「Cosserat 梁の大回転ネイティブ特性が注目される」という根本的指摘を受け、
status-383 までの "explicit + UL の組合せは原理的に成立しない" 結論を踏まえて
**Abaqus/Explicit 標準アプローチ**への移行を着手:

- **(z1a) 要素ごと波速ベース Δt 推定**: `dt_e = L_e / √(E/ρ)` を要素ごとに計算し、
  Gerschgorin 全体上界と min を取る。Gerschgorin の過大評価を物理的下限で
  補強。
- **(z1b) selective mass scaling**: β² 倍化を「stiff DOF」に限定。Gerschgorin
  row-sum / M_lump が median × `threshold_ratio` を超える DOF のみ自動検出し、
  梁 DOF は β=1 を維持する。status-381 §5 「explicit 解 50% アンダー」の主因
  と推定される β² 過剰減衰を回避。

両 API の実装は完了し、**17 単体テスト全 pass**。一方で 7 本撚線実機検証で
**新たな限界**が判明: selective scaling 単独では 7 本でも frac=1.0 完走不可。

## 1. 実装

### 1.1 候補 (z1a): 要素ごと波速 Δt 推定

`xkep_cae/contact/solver/_explicit_dynamic.py`:

```python
def _estimate_critical_dt_per_element(
    connectivity, node_coords, beam_E, beam_rho,
) -> float:
    """各 beam 要素で dt_e = L_e / √(E/ρ) を計算し最小値を返す."""
    if connectivity is None or beam_E <= 0 or beam_rho <= 0:
        return float("inf")
    c_a = float(np.sqrt(beam_E / beam_rho))
    L_e = np.linalg.norm(p2 - p1, axis=1)
    return float(L_e[L_e > 0].min() / c_a)
```

`ExplicitDynamicProcess.process()` で:
```python
dt_c_gers = _estimate_critical_dt(K, M_lump_inv, fixed_dofs)  # 既存 Gerschgorin
dt_c_beam = _estimate_critical_dt_per_element(
    connectivity, node_coords_ref, cfg.beam_E, cfg.beam_rho,
)
if np.isfinite(dt_c_beam) and time_strategy._mass_scaling_dof_mask is None:
    dt_c_beam *= time_strategy.mass_scaling_beta  # β は Gerschgorin 側に既反映
dt_c = min(dt_c_gers, dt_c_beam)
```

### 1.2 候補 (z1b): selective mass scaling

新ヘルパ `_detect_stiff_dofs(K, M_lump, fixed_dofs, threshold_ratio=10.0)`:
- Gerschgorin row-sum / M_lump で各 DOF の ω² を計算
- median を計算（fixed・M=0 を除外）
- `ω² > median × threshold_ratio` を stiff DOF として返す

`ExplicitCentralDifferenceProcess` の拡張:
- `_compute_scaled_mass(beta)` ヘルパで mask 反映
- `set_mass_scaling_dof_mask(mask)` API 追加（None で全 DOF 一律へリセット）
- `set_mass_scaling_beta()` の v/a rescale を mask に従って selective 化

`ExplicitDynamicProcess.process()` で初回 Courant 検査時に検出 → set_mask:
```python
if cfg.mass_scaling_selective and time_strategy._mass_scaling_dof_mask is None:
    stiff_mask = _detect_stiff_dofs(K, M_lump, fixed_dofs, threshold_ratio=...)
    time_strategy.set_mass_scaling_dof_mask(stiff_mask)
```

### 1.3 plumb-through

| 層 | field |
|----|-------|
| `_ContactConfigInput` | `beam_rho: float = 0.0` |
| `ExplicitDynamicInput` | `beam_E`, `beam_rho`, `mass_scaling_selective`, `mass_scaling_stiff_threshold_ratio` |
| `ContactFrictionInputData` | `explicit_mass_scaling_selective`, `explicit_mass_scaling_stiff_threshold_ratio` |
| `StrandBendingOscillationConfig` | 同 2 field |
| `strand_bending_oscillation.py` 3 経路 | `cfg.rho` を `_ContactConfigInput.beam_rho` に伝搬、selective 関連も plumb |

### 1.4 単体テスト追加（+17）

**`test_explicit_dynamic.py`**:

`TestPerElementCriticalDt` (+6):
- 不正入力で inf 返却（zero E / rho / None connectivity）
- 単一要素 L=1, E=1, ρ=1 → dt = 1.0
- 複数要素で最短要素が支配
- 現実的鋼梁 E=130 GPa, ρ=8.96e-9, L=6.25mm → dt ≈ 1.64 μs

`TestDetectStiffDofs` (+5):
- 一様 K で 検出ゼロ
- 1 DOF 100× outlier を検出
- 固定 DOF が median 計算から除外
- M=0 DOF は対象外
- threshold_ratio で感度制御

`TestSelectiveMassScaling` (+6):
- default mask=None で全 DOF β² 倍化（既存挙動）
- mask 適用で True DOF のみ β² 倍化
- mask 設定後 β 上昇で selective に追従
- mask=None で全 DOF 一律へリセット
- 不正 shape で ValueError
- v/a rescale が mask 内に限定

## 2. 実機検証

### 2.1 単梁 90° 曲げ（status-381〜382 と同条件）

`work/beam_hysteresis/37_z1ab_accuracy_validation.py`:

| ケース | frac | max\|u\| [mm] | 解析解誤差 | gate |
|--------|------|--------------|-----------|------|
| implicit_baseline | 1.000 | 70.45 | 3.90% | PASS |
| exp_baseline (status-382 同等) | 1.000 | 35.37 | 51.74% | FAIL |
| exp_z1b_selective_beta1000 | 0.000 | DIVERGED | — | FAIL |
| exp_z1ab_selective_beta100 | 0.000 | DIVERGED | — | FAIL |
| exp_z1ab_selective_strict_threshold | 0.000 | DIVERGED | — | FAIL |

**観察**: 単梁は K がほぼ一様で、selective stiff DOF 検出が「outlier なし」と
判定 → β² 倍化対象が空集合 → 実質 β=1 → dt_physical=1.0 s に対し 60万 step
要求 → Courant cap で発散判定。**実装の bug ではなく、selective scaling が
heterogeneous K を要求する性質** に由来。

### 2.2 7 本撚線 90° 曲げ（接触あり）

```
[SELECTIVE_MASS] Incr 1 stiff DOFs detected: 112/714 (threshold ratio 10.0)
[MASS_SCALE] Incr 1 β: 1.000e+00 → 1.000e+03 (target 4.737e+04, abs cap 1.000e+03)
[COURANT] Incr 1 β cap reached → cutback (#1)
[COURANT] Incr 1 β cap reached (target=8.819e+06 > 1.000e+03) → cutback (#2)
[CUTBACK:courant_cap] frac 0.0125 → frac 0.0008 → 不収束
```

**stiff DOF 検出は機能している**（112/714 = 15.7% が contact ref / MPC 関連）が、
**残り 84% (beam DOF) が β=1 のまま dt 制限を支配**。target β=8.8×10⁶ は
beam dt （~1.6 μs）で dt_sub=0.05s を満たすために必要な値 → 物理的整合性。

→ **selective scaling 単独では「beam も β=1 では dt が物理 wave 速度に縛られる」
ため、loading rate (dt_physical) を物理時間スケールへ縮小しなければ frac=1.0
不可能**。これは Abaqus/Explicit でも同じ制約で、quasi-static explicit の
標準 recipe は **(loading rate up) + (modest mass scaling)** の組合せ。

## 3. 真の解決経路 — 2 段階質量スケーリング

status-381 §5「explicit 解 50% アンダー」の真の構造:

| 全 DOF β=1000 一律 (status-381〜382) | β² rescale で全 DOF v/a 過剰減衰 |
|--------------------------------------|----------------------------------|
| selective β_stiff=1000、β_beam=1（本 status） | beam DOF dt 1.6μs に縛られ frac<<1.0 |
| **2 段階: β_stiff=1000、β_beam=10** | beam dt 1.6μs × 10 = 16μs、過剰減衰回避 |

つまり「梁 DOF にも `modest` mass scaling を許容しつつ、stiff DOF にはより
aggressive に倍化する」 **2 段階スケーリング** が解。これは **per-DOF β
配列** （現在の binary mask ではなくスカラー配列）への API 拡張で実装可能。

加えて **loading rate reduction**（`t_cycle = max(10·T1, 1.0)` の下限を下げる）
を併用すれば、β_beam を更に小さく抑えて梁 dynamics の物理精度を完全保持できる。

## 4. 実装変更まとめ

- `xkep_cae/contact/solver/_explicit_dynamic.py`:
  - `_estimate_critical_dt_per_element()` 新設（+50 行）
  - `_detect_stiff_dofs()` 新設（+45 行）
  - `ExplicitDynamicInput` に 4 field 追加（beam_E/beam_rho/mass_scaling_selective/threshold_ratio）
  - `process()` 内 selective 検出 + per-element dt 統合（+30 行）
- `xkep_cae/time_integration/strategy.py`:
  - `_compute_scaled_mass()` ヘルパ抽出（+10 行）
  - `set_mass_scaling_dof_mask()` API 追加（+25 行）
  - `set_mass_scaling_beta()` の rescale を selective 対応（+5 行）
- `xkep_cae/contact/_contact_pair.py`:
  - `_ContactConfigInput.beam_rho: float = 0.0` 追加
- `xkep_cae/core/data.py`:
  - `ContactFrictionInputData` に selective 2 field 追加
- `xkep_cae/contact/solver/process.py`:
  - 主ループ + relax 両方の `ExplicitDynamicInput` 構築箇所で plumb
- `xkep_cae/numerical_tests/strand_bending_oscillation.py`:
  - `StrandBendingOscillationConfig` に 2 field 追加
  - `_ContactConfigInput(beam_rho=cfg.rho)` を 2 経路で plumb
  - 3 経路の `ContactFrictionInputData` 構築で plumb
- 単体テスト +17（`TestPerElementCriticalDt` 6 + `TestDetectStiffDofs` 5 + `TestSelectiveMassScaling` 6）
- 検証スクリプト `work/beam_hysteresis/37_z1ab_accuracy_validation.py` 新設（+220 行）

回帰: 全 24 契約検査 OK / **726 passed 5 skipped**（status-383 比 +17）/
`test_helical_3d_hermite` rel_err=2.18×10⁻⁷ 維持 / 7 本 implicit frac=1.0 / ruff pass。

## 5. **MCDD 凍結解除条件 — 条件 (5) 未達**

| 条件 | 状態 |
|------|------|
| (1) Phase E 完了 | ✅ status-357 |
| (2) 19 本 frac=1.0 完走 | △ explicit 系は本 status で再評価対象 |
| (3) max\|u_trans\| < L_strand × 10 | ✅ implicit / N/A explicit 発散ケース |
| (4) `KcNormalDirectionStiffness` FD rel_err < 1e-2 | ✅ status-356（2.18×10⁻⁷） |
| **(5) 解の精度 < 10%** | **❌ selective 単独では 7 本でも frac<<1.0** |

## 6. 引継ぎ — 次 status の候補

### 6.1 候補 (z1c) 最有力 — 2 段階質量スケーリング + loading rate 縮小

API 拡張: `set_mass_scaling_dof_mask(mask)` を `set_mass_scaling_per_dof_beta(beta_array)`
に置換。あるいは現 mask + 「外側の β」も別パラメータ化（`beta_outside`）。

実装:
```python
def __init__(..., mass_scaling_beta_outside: float = 1.0):
    # mask 内: β² 倍、外: β_outside² 倍
def _compute_scaled_mass(beta):
    scaling = (beta_outside**2) * np.ones_like(M_raw)
    scaling[mask] = beta**2
    return scaling * M_raw
```

検証:
- 単梁 90°: β_outside=10, β_stiff=N/A（stiff DOF 不在）→ dt 10× 拡大
  + KE rescale が全 DOF に作用するが β_outside=10 程度なら過剰減衰小
- 7 本: β_outside=10, β_stiff=1000 で frac=1.0 + 解析解一致目標
- 19 本: 同様に検証、frac=1.0 + 解析解 70mm 近傍を目標

### 6.2 候補 (z2) — Cosserat 梁プロトタイプ

UL を捨てて explicit + 大回転を本質解決。中期的に最もクリーンだが実装コスト
中（~1000 行オーダー）。先に (z1c) で 2 段階スケーリングが解析解一致に
十分か検証してから判断。

### 6.3 候補 (z1d) — `t_cycle` 下限緩和

現在 `t_cycle = max(10·T1, 1.0)` で下限 1 秒を強制。物理的に必要な loading
時間は `T1 = 6.7ms` なので、下限を `0.1·T1 = 0.67ms` 程度まで下げれば
（z1c）が更に効きやすい。設計仕様への影響大なので別 status。

## 7. MCDD 脱法 pattern 回避

- **pattern 1（tol 緩和）**: 精度 gate 0.10 を変更せず、未達と明記
- **pattern 5（既存テスト skip）**: 既存 709 test 全 pass、+17 追加
- **pattern 6（骨格 status）**: API 実装 + 17 単体テスト + 5 ケース実機検証 +
  数理的観察（uniform K で stiff 検出ゼロ、heterogeneous K でも beam DOF 制約）で完結
- **pattern 8（根拠なき主張）**: 7 本実機ログ「stiff DOF 112/714」「target β=8.8e6」を
  実証根拠として提示
- **pattern 10（TODO 先送り）**: (z1ab) infrastructure は完了、次の (z1c) 2 段階
  スケーリングは API 拡張を要するため別 status が適切

## 8. 引継ぎコマンド

```bash
# z1ab 検証
uv run --extra dev python work/beam_hysteresis/37_z1ab_accuracy_validation.py \
    2>&1 | tee /tmp/z1ab_$(date +%s).log

# 回帰
pytest xkep_cae/contact/ xkep_cae/mathematics/ xkep_cae/time_integration/ \
       xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py
python contracts/validate_process_contracts.py
ruff check xkep_cae/ tests/ && ruff format --check xkep_cae/ tests/
```

## 9. 観察 — 開発運用

### 効果的だった点

- **ユーザーからの「Cosserat / 要素ごと dt」指摘**: status-383 で頭打ちになった
  「explicit + UL 不整合」を、より物理に根ざした観点（Abaqus/Explicit 標準
  recipe）で再構成できた。
- **Unit test ファースト**: 17 単体テストを先に通したことで、validation で
  発散しても「実装の bug」と「物理 / パラメータ要件の限界」を明確に切り分け可能。

### 学び — selective mass scaling の前提

selective mass scaling が有効に機能する条件:
1. **K が heterogeneous** であること（contact / MPC ref node 等）
2. **stiff DOF と非 stiff DOF の dt 比が β cap 以内** であること
3. もし条件 2 を満たさない場合、**非 stiff (beam) DOF にも modest mass scaling
   を許容**しなければならない（2 段階スケーリング）

19 本撚線で K_c x/z カップリング不整合が支配的な場合、stiff DOF に絞った
β 倍化は「症状緩和」にすぎない。**真の解は (z1c) 2 段階 + (z2) Cosserat の
並行検討**。
