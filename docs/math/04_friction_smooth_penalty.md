# 04 — Coulomb 摩擦（smooth penalty）+ 接線剛性

[← README](../../README.md) | [← roadmap](../roadmap.md) | [← 数理台帳](README.md) | [← MCDD 設計仕様](../../xkep_cae/mathematics/docs/mathematics.md)

> **本台帳の責務**: smooth penalty 型 Coulomb 摩擦の接線力 $\boldsymbol{f}_t$、
> 摩擦限界 $|\boldsymbol{f}_t|\le \mu p_n$、return mapping による弾塑性滑り
> 分解、および接線剛性 $\boldsymbol{K}_t$ の項構成を正規記述する。
>
> **本台帳の非責務**: NCP 系（Fischer-Burmeister）は摩擦接線剛性の符号問題で
> 発散するため凍結中（status-147）。本台帳は `contact_mode="smooth_penalty"` 系のみを
> 対象とする。

## 表記

| 記号 | 意味 |
|---|---|
| $\mu$ | Coulomb 摩擦係数 |
| $p_n$ | 法線ペナルティ力（03 章 [#eq-pn](03_huber_contact_penalty.md#eq-pn)） |
| $\hat{\boldsymbol{n}}$ | 接触法線 |
| $\boldsymbol{P}_\perp = \boldsymbol{I}-\hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top$ | 接線面射影 |
| $\boldsymbol{u}^{\mathrm{rel}}_t$ | 接触点での接線方向相対変位 |
| $\boldsymbol{f}_t$ | 摩擦接線力（$A$ に作用） |
| $k_t$ | 接線ペナルティ剛性（弾性スティック剛性） |
| $\phi$ | 滑り関数 $\phi = \lVert\boldsymbol{f}_t\rVert - \mu p_n$ |
| $\lambda$ | 塑性滑り乗数（$\lambda\ge 0$） |

→ 実装: `xkep_cae/contact/friction/strategy.py`,
`xkep_cae/contact/friction/law_friction.py`, `xkep_cae/contact/friction/_assembly.py`

---

<a id="eq-coulomb"></a>

## 1. Coulomb 摩擦則

スティック / スリップの境界:

$$
\lVert\boldsymbol{f}_t\rVert \;\le\; \mu\,p_n
$$

スティック ($\phi < 0$): $\dot{\boldsymbol{u}}^{\mathrm{rel}}_t = 0$。
スリップ ($\phi = 0$): $\dot{\boldsymbol{u}}^{\mathrm{rel}}_t \parallel \boldsymbol{f}_t$、
かつ $\boldsymbol{f}_t \cdot \dot{\boldsymbol{u}}^{\mathrm{rel}}_t \ge 0$（エネルギー散逸）。

<a id="inv-friction-bound"></a>

不変量:

$$
\phi(\boldsymbol{f}_t, p_n) \;=\; \lVert\boldsymbol{f}_t\rVert - \mu p_n \;\le\; 0
$$

→ 契約: `InequalityContract(name="friction_coulomb_bound", expr="||f_t||",
kind="leq", bound="mu * p_n", equation_ref="04_friction_smooth_penalty.md#inv-friction-bound")`

---

<a id="eq-return-mapping"></a>

## 2. Return mapping（predictor–corrector）

トライアル（elastic predictor）:

$$
\boldsymbol{f}_t^{\mathrm{trial}}
\;=\; \boldsymbol{f}_t^{\,n} \;+\; k_t\,\Delta \boldsymbol{u}^{\mathrm{rel}}_t
$$

ここで $\Delta \boldsymbol{u}^{\mathrm{rel}}_t = \boldsymbol{P}_\perp\,
(\Delta \boldsymbol{u}_B - \Delta \boldsymbol{u}_A)$。

<a id="eq-slip-correction"></a>

判定:

$$
\boldsymbol{f}_t^{\,n+1}
\;=\;
\begin{cases}
\boldsymbol{f}_t^{\mathrm{trial}},
& \lVert\boldsymbol{f}_t^{\mathrm{trial}}\rVert \le \mu p_n\quad\text{(stick)}\\[6pt]
\mu p_n \dfrac{\boldsymbol{f}_t^{\mathrm{trial}}}{\lVert\boldsymbol{f}_t^{\mathrm{trial}}\rVert},
& \lVert\boldsymbol{f}_t^{\mathrm{trial}}\rVert > \mu p_n\quad\text{(slip, radial return)}
\end{cases}
$$

滑り乗数は $\lambda = (\lVert\boldsymbol{f}_t^{\mathrm{trial}}\rVert - \mu p_n)/k_t \ge 0$。

→ 実装: `law_friction.py:CoulombReturnMappingProcess.process`

---

<a id="eq-ft-smooth"></a>

## 3. smooth penalty 正則化

スリップ判定の $\max(0,\cdot)$ 不連続を遷移幅 $\delta_t$ で平滑化する
（status-147 が NCP 系を凍結した背景）:

$$
\boldsymbol{f}_t
\;=\;
-\mu p_n \,\boldsymbol{s}_{\mathrm{smooth}}\!\big(\boldsymbol{u}^{\mathrm{rel}}_t,\delta_t\big)
\;=\;
-\mu p_n \,\frac{\boldsymbol{u}^{\mathrm{rel}}_t}{\sqrt{\lVert\boldsymbol{u}^{\mathrm{rel}}_t\rVert^2 + \delta_t^2}}
$$

この regularization は $\delta_t \to 0$ で厳密 Coulomb に収束し、
$\boldsymbol{f}_t$ と $\boldsymbol{K}_t$ が $C^\infty$ になる。

→ 実装: `strategy.py:_compute_smooth_friction_force`（Process: `FrictionTangentStiffnessProcess`）

---

<a id="eq-kt-tangent"></a>

## 4. 接線剛性 $\boldsymbol{K}_t$ の項分解

$\boldsymbol{K}_t = \partial \boldsymbol{f}_t/\partial \boldsymbol{u}$ は以下の項で構成される:

$$
\boldsymbol{K}_t
\;=\;
\underbrace{\frac{\partial \boldsymbol{f}_t}{\partial \boldsymbol{u}^{\mathrm{rel}}_t}
\cdot \frac{\partial \boldsymbol{u}^{\mathrm{rel}}_t}{\partial \boldsymbol{u}}}_{\boldsymbol{K}_{t,\mathrm{mat}}}
\;\;\underbrace{+\;\boldsymbol{f}_t\,\otimes\,\frac{\partial\,(-\mu p_n)}{\partial \boldsymbol{u}}\,\frac{1}{\mu p_n}}_{\boldsymbol{K}_{t,p_n\text{-coupling}}}
\;\;\underbrace{+\;\boldsymbol{K}_{t,\mathrm{geo}}}_{\partial \boldsymbol{P}_\perp/\partial \boldsymbol{u}}
\;\;\underbrace{+\;\boldsymbol{K}_{t,\mathrm{st}}}_{(s,t)\,\text{追従}}
$$

<a id="eq-kt-mat"></a>

接線材料項（stick 近傍）:

$$
\boldsymbol{K}_{t,\mathrm{mat}}
\;=\;
-\mu p_n\,\frac{1}{\lVert\boldsymbol{u}^{\mathrm{rel}}_t\rVert_\delta}
\left(\boldsymbol{I} - \frac{\boldsymbol{u}^{\mathrm{rel}}_t(\boldsymbol{u}^{\mathrm{rel}}_t)^\top}{\lVert\boldsymbol{u}^{\mathrm{rel}}_t\rVert_\delta^2}\right)\,\boldsymbol{P}_\perp
$$

$\lVert\cdot\rVert_\delta = \sqrt{\lVert\cdot\rVert^2+\delta_t^2}$。

$\boldsymbol{K}_{t,\mathrm{st}}$ は $\partial s/\partial \boldsymbol{u}$ を介して
`FrictionStStiffnessProcess` が担当（02 章 [#eq-st-jacobian](02_contact_geometry.md#eq-st-jacobian)）。

→ 実装:
- $\boldsymbol{K}_{t,\mathrm{mat}}$ : `strategy.py:FrictionTangentStiffnessProcess`
- $\boldsymbol{K}_{t,\mathrm{geo}}$ : `strategy.py:FrictionGeometricStiffnessProcess`
- $\boldsymbol{K}_{t,\mathrm{st}}$ : `strategy.py:FrictionStStiffnessProcess`
- アセンブリ : `_assembly.py`（status-256 で Process 化、status-310 でベクトル化）

---

<a id="eq-kt-fd"></a>

## 5. FD 整合性

接線剛性 $\boldsymbol{K}_t$ と摩擦力 $\boldsymbol{f}_t$ の FD:

$$
\boldsymbol{K}_t\,\delta\boldsymbol{u}
\;\approx\;
\frac{\boldsymbol{f}_t(\boldsymbol{u}+\varepsilon\,\delta\boldsymbol{u})
-\boldsymbol{f}_t(\boldsymbol{u})}{\varepsilon}
$$

→ 契約: `FDConsistencyContract(name="K_t_fd_consistency", vector_name="f_t",
jacobian_name="K_t", equation_ref="04_friction_smooth_penalty.md#eq-kt-fd",
severity="nightly")`

---

## 6. 既存実装との trace

| 数式 | 実装位置 | 備考 |
|---|---|---|
| [#eq-coulomb](#eq-coulomb) | `law_friction.py:CoulombReturnMappingProcess` | 摩擦則 |
| [#eq-return-mapping](#eq-return-mapping) | `law_friction.py:_return_map` | predictor–corrector |
| [#eq-slip-correction](#eq-slip-correction) | `law_friction.py:_radial_return` | radial return |
| [#eq-ft-smooth](#eq-ft-smooth) | `strategy.py:_compute_smooth_friction_force` | 平滑化 |
| [#eq-kt-tangent](#eq-kt-tangent) | `strategy.py:FrictionTangentStiffnessProcess` + 兄弟 Process | 項分解 |
| [#eq-kt-mat](#eq-kt-mat) | `strategy.py:FrictionTangentStiffnessProcess.process` | 材料項 |
| [#eq-kt-fd](#eq-kt-fd) | `xkep_cae/verify/` 系（未整備、status-356 予定）| FD 検証 |

---

## 関連 status

- status-147: NCP 摩擦の鞍点系符号問題 → smooth_penalty 必須
- status-256: 摩擦アセンブリ Process 化（B1–B4）
- status-274: 摩擦 K_st 隣接ノード拡張
- status-310: Hermite dpA/dpB バッチ化 + 摩擦 K_st ベクトル化
- status-324: K_st distance culling（Huber 遷移幅ベース）
- status-335〜336: M–κ ヒステリシスループ観測（摩擦散逸検証）
