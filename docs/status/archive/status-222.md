# status-222: 接触ソルバー完全一本化 — Huber ペナルティ統一

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-21
**ブランチ**: `claude/fix-ci-diagnostics-LM2ko`

---

## 概要

接触ソルバーを **Huber ペナルティ法 + Coulomb 摩擦必須** に完全一本化。
後方互換なし。以下の機構を削除し、コードパス・設定パラメータを大幅に簡素化。

---

## 削除対象と復元情報

### 1. Uzawa 拡大ラグランジアン（status-221 で凍結済み → 本 status で完全削除）

**設計意図**: penalty 法の精度不足を Lagrange 乗数 λ で補償する拡大ラグランジアン法。
内部 NR ループ収束後に λ = max(0, λ + k_pen*(-g)) で乗数を更新し、外部 Uzawa ループで繰り返す。

**凍結理由 (status-221)**: n_uzawa_max=1 で純粋ペナルティと等価。λ 更新は実質無効。
k_pen の自動推定が十分であり、拡大ラグランジアンの追加精度が不要だった。

**削除範囲**:
- `lam_all: np.ndarray` — 全ペアの λ 配列（SolverStateOutput, checkpoint 含む）
- `UzawaUpdateProcess` / `UzawaUpdateInput` / `UzawaUpdateOutput` — _nuzawa_steps.py
- `n_uzawa_max`, `tol_uzawa` — 全箇所（_ContactConfigInput, default_strategies, ファクトリ関数）
- Uzawa 外部ループ (`for _uzawa_iter in range(n_uzawa_max)`) — _newton_uzawa_dynamic.py, _newton_uzawa_static.py

**復元手順**: status-221 時点の commit から `lam_all` 管理と Uzawa ループを復元。
`_nuzawa_steps.py` の UzawaUpdateProcess が自己完結しているので、
戻す場合はこのクラスと外部ループの `for _uzawa_iter` を復活させれば良い。

---

### 2. SmoothPenaltyContactForceProcess（softplus 接触力）

**設計意図**: max(0, -g) を softplus = (1/δ)log(1+exp(-δg)) で C∞ 近似。
INACTIVE 判定不要（softplus は自然に 0 へ漸近）。

**Huber との比較**:
| | Huber (残存) | softplus (削除) |
|---|---|---|
| 滑らかさ | C1 | C∞ |
| サポート | コンパクト (x < -δ で厳密 0) | 無限（指数減衰） |
| INACTIVE チェック | 必要（性能向上のため） | 不要 |

**凍結理由**: Uzawa 凍結で λ=0 となり、p_n = huber(-k_pen*g) と p_n = k_pen*softplus(g) は
同等の精度。Huber の C1 は Newmark 動的解析では十分。

**削除範囲**:
- `SmoothPenaltyContactForceProcess` クラス全体 — strategy.py
- `SmoothPenaltyFrictionProcess` クラス全体 — friction/strategy.py

**復元手順**: status-221 時点の strategy.py から SmoothPenaltyContactForceProcess を
コピーし、ファクトリ関数に `contact_mode="smooth_penalty"` 分岐を戻す。

---

### 3. 準静的接触ソルバー (NewtonUzawaStaticProcess)

**設計意図**: 慣性項なしの Newton-Raphson 接触解析。静的荷重の接触問題向け。

**剥奪理由**: 接触は動的ソルバーの c0*M 正則化に依存して K_eff の正定値性を確保している。
準静的では K_eff = K_struct + K_c のみで、接触 on/off で固有値が急変し発散しやすい。
実用上、動的ソルバーで十分小さい dt を使えば準静的に近い解が得られる。

**削除範囲**:
- `NewtonUzawaStaticProcess` / `NewtonUzawaStaticInput` / `StaticStepOutput` — _newton_uzawa_static.py
- `ContactFrictionProcess` 内の `if not _dynamics:` 分岐

**復元手順**: _newton_uzawa_static.py は自己完結ファイル。
ContactFrictionProcess の分岐を戻せば復元可能。

---

### 4. contact_mode / use_friction パラメータ

**削除対象**:
- `contact_mode: str` — ContactSetupData, _ContactConfigInput, ContactSetupInput, default_strategies, ファクトリ関数
- `use_friction: bool` — 同上（動的接触は摩擦必須に統一）
- `NoFrictionProcess` — friction/strategy.py

**統一後の構成**:
- 接触力: `HuberContactForceProcess`（旧 NCPContactForceProcess から λ を除去）
- 摩擦: `CoulombReturnMappingProcess`（常時有効）
- ソルバー: `NewtonDynamicProcess`（旧 NewtonUzawaDynamicProcess から Uzawa を除去）

---

## 推奨ソルバー構成（統一後）

```python
# 全パラメータ
contact_force = HuberContactForceProcess(ndof, smoothing_delta=5000/r_min)
friction = CoulombReturnMappingProcess(ndof, k_pen=k_pen, k_t_ratio=1.0)
solver = NewtonDynamicProcess()  # 動的のみ、摩擦必須
```
