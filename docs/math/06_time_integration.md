# 06 — 時間積分（Generalized-α / Newmark）と疑似時間

[← README](../../README.md) | [← roadmap](../roadmap.md) | [← 数理台帳](README.md) | [← MCDD 設計仕様](../../xkep_cae/mathematics/docs/mathematics.md)

> **本台帳の責務**: 動的解析における Generalized-α 時間積分、Newmark-β
> 予測子・補正子、有効剛性行列 $\boldsymbol{K}_{\mathrm{eff}}$、疑似時間
> （準静的荷重比例）を正規記述する。
>
> **本台帳の非責務**: 増分制御（増分分割・カットバック・$f_{\mathrm{ref}}$ 設定）
> および収束診断ログ規約は CLAUDE.md「ソルバー診断ログ規約（status-307）」を参照。

## 表記

| 記号 | 意味 |
|---|---|
| $\boldsymbol{u}^{\,n},\,\boldsymbol{v}^{\,n},\,\boldsymbol{a}^{\,n}$ | 時刻 $t_n$ の変位・速度・加速度 |
| $\Delta t$ | 時間刻み |
| $\rho_\infty$ | スペクトル半径（$0\le\rho_\infty\le 1$）|
| $\alpha_m,\alpha_f,\beta,\gamma$ | Chung-Hulbert パラメータ |
| $\boldsymbol{M},\,\boldsymbol{C}$ | 質量・減衰行列 |
| $\boldsymbol{K}_T$ | 接線剛性（内力 + 接触 + 摩擦の合計） |
| $\boldsymbol{K}_{\mathrm{eff}}$ | NR 反復用有効剛性 |
| $\boldsymbol{R}_{\mathrm{eff}}$ | 有効残差 |
| $\tau \in [0,1]$ | 疑似時間（準静的の荷重係数 `frac`） |

→ 実装: `xkep_cae/time_integration/strategy.py`, `xkep_cae/solve/_newton_dynamic.py`

---

<a id="eq-chung-hulbert"></a>

## 1. Chung-Hulbert パラメータ

$\rho_\infty$ から:

$$
\alpha_m = \frac{2\rho_\infty - 1}{\rho_\infty + 1},\qquad
\alpha_f = \frac{\rho_\infty}{\rho_\infty + 1}
$$

$$
\gamma = \tfrac{1}{2} - \alpha_m + \alpha_f,\qquad
\beta = \tfrac{1}{4}\,(1 - \alpha_m + \alpha_f)^2
$$

$\rho_\infty=1$ で Newmark 平均加速度法（エネルギー保存）、$\rho_\infty=0$ で
最大数値減衰。推奨 $\rho_\infty = 0.9\text{–}1.0$（status-283）。

→ 実装: `strategy.py:GeneralizedAlphaProcess.__init__`

---

<a id="eq-newmark-predictor"></a>

## 2. Newmark 予測子

$$
\boldsymbol{u}^{\mathrm{pred}}
\;=\; \boldsymbol{u}^{\,n} + \Delta t\,\boldsymbol{v}^{\,n}
+ \tfrac{1}{2}\Delta t^{\,2}(1 - 2\beta)\,\boldsymbol{a}^{\,n}
$$

$$
\boldsymbol{v}^{\mathrm{pred}}
\;=\; \boldsymbol{v}^{\,n} + \Delta t\,(1 - \gamma)\,\boldsymbol{a}^{\,n}
$$

→ 実装: `strategy.py:GeneralizedAlphaProcess.predict`

---

<a id="eq-newmark-corrector"></a>

## 3. Newmark 補正子（Newton 反復内）

NR 反復で $\Delta \boldsymbol{u}$ を得た後:

$$
\boldsymbol{a}^{\,n+1} \;=\; \frac{1}{\beta\,\Delta t^{\,2}}
\big(\boldsymbol{u}^{\,n+1} - \boldsymbol{u}^{\mathrm{pred}}\big)
$$

$$
\boldsymbol{v}^{\,n+1} \;=\; \boldsymbol{v}^{\mathrm{pred}} + \Delta t\,\gamma\,\boldsymbol{a}^{\,n+1}
$$

→ 実装: `strategy.py:GeneralizedAlphaProcess.correct`

---

<a id="eq-effective-stiffness"></a>

## 4. 有効剛性行列

$$
\boldsymbol{K}_{\mathrm{eff}}
\;=\; \boldsymbol{K}_T
\;+\; (1 - \alpha_m)\,\frac{1}{\beta\,\Delta t^{\,2}}\,\boldsymbol{M}
\;+\; (1 - \alpha_f)\,\frac{\gamma}{\beta\,\Delta t}\,\boldsymbol{C}
$$

→ 実装: `strategy.py:GeneralizedAlphaProcess.effective_stiffness`

---

<a id="eq-effective-residual"></a>

## 5. 有効残差（中間点定義）

$$
\boldsymbol{R}_{\mathrm{eff}}
\;=\; \boldsymbol{R}
\;+\; \boldsymbol{M}\,\boldsymbol{a}_{\mathrm{mid}}
\;+\; \boldsymbol{C}\,\boldsymbol{v}_{\mathrm{mid}}
$$

$$
\boldsymbol{a}_{\mathrm{mid}} = (1 - \alpha_m)\boldsymbol{a}^{\,n+1} + \alpha_m\,\boldsymbol{a}^{\,n},
\qquad
\boldsymbol{v}_{\mathrm{mid}} = (1 - \alpha_f)\boldsymbol{v}^{\,n+1} + \alpha_f\,\boldsymbol{v}^{\,n}
$$

$\boldsymbol{R} = \boldsymbol{f}_{\mathrm{int}} - \boldsymbol{f}_{\mathrm{ext}}$
は内力・外力・接触力・摩擦力の総和。

→ 実装: `strategy.py:GeneralizedAlphaProcess.effective_residual`

---

<a id="eq-pseudo-time"></a>

## 6. 準静的疑似時間

準静的解析では $\boldsymbol{M}=\boldsymbol{C}=\boldsymbol{0}$、$\Delta t$ は
疑似時間 $\tau\in[0,1]$ の増分 $\Delta\tau$ と同一視される:

$$
\boldsymbol{f}_{\mathrm{ext}}(\tau)
\;=\; \tau\,\boldsymbol{f}_{\mathrm{ext}}^{\,\mathrm{target}},
\qquad
\boldsymbol{K}_{\mathrm{eff}} = \boldsymbol{K}_T,\quad
\boldsymbol{R}_{\mathrm{eff}} = \boldsymbol{R}
$$

→ 実装: `strategy.py:QuasiStaticProcess`（predict/correct は identity）

---

<a id="inv-energy-balance"></a>

## 7. エネルギー収支（不変量）

外力入力仕事 = 内部エネルギー + 運動エネルギー + 散逸:

$$
W_{\mathrm{ext}}
\;=\; \Delta U_{\mathrm{int}}
\;+\; \Delta T_{\mathrm{kin}}
\;+\; D_{\mathrm{diss}}
$$

$\rho_\infty=1$（Newmark 平均加速度法）のとき $D_{\mathrm{diss}}=0$ で
エネルギー保存。$\rho_\infty<1$ では高周波成分が数値減衰される。

→ 契約: `InequalityContract(name="energy_dissipation_nonneg",
expr="D_diss", kind="geq", bound="0",
equation_ref="06_time_integration.md#inv-energy-balance", severity="soft")`

→ 実装: `xkep_cae/verify/`（散逸エネルギー検証は status-331
`CableDissipationProcess` で導入済み）

---

## 8. 既存実装との trace

| 数式 | 実装位置 | 備考 |
|---|---|---|
| [#eq-chung-hulbert](#eq-chung-hulbert) | `strategy.py:GeneralizedAlphaProcess.__init__` | $\rho_\infty$ 依存 |
| [#eq-newmark-predictor](#eq-newmark-predictor) | `strategy.py:GeneralizedAlphaProcess.predict` | u/v 予測 |
| [#eq-newmark-corrector](#eq-newmark-corrector) | `strategy.py:GeneralizedAlphaProcess.correct` | 加速度更新 |
| [#eq-effective-stiffness](#eq-effective-stiffness) | `strategy.py:GeneralizedAlphaProcess.effective_stiffness` | K + βM + γC |
| [#eq-effective-residual](#eq-effective-residual) | `strategy.py:GeneralizedAlphaProcess.effective_residual` | mid-point |
| [#eq-pseudo-time](#eq-pseudo-time) | `strategy.py:QuasiStaticProcess` | identity |
| [#inv-energy-balance](#inv-energy-balance) | `xkep_cae/verify/`（status-331）| 散逸検証 |

---

## 関連 status

- status-281: 動的ソルバーでヘリカル素線 90 度曲げ完走
- status-283: `rho_inf=0.9–1.0` 推奨（数値減衰と安定性のバランス）
- status-297: 微小 $\Delta t$ 耐性改善（dt snap + `atol_force`）
- status-299: 90 度曲げ + 先端横変位 ±48 mm 揺動の統合モード完走
- status-307: ソルバー診断ログ規約（$f_{\mathrm{ref}}$、CUTBACK 原因タグ）
- status-331: `CableDissipationProcess`（M–κ ヒステリシスの散逸エネルギー追跡）
