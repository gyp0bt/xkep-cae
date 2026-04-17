# 01 — 梁要素の運動学（CR / TL / UL + Hermite）

[← README](../../README.md) | [← roadmap](../roadmap.md) | [← 数理台帳](README.md) | [← MCDD 設計仕様](../../xkep_cae/mathematics/docs/mathematics.md)

> **本台帳の責務**: Co-Rotational（CR）/ Total-Lagrangian（TL）/ Updated-Lagrangian（UL）
> の 3 定式化における梁要素の離散化式と Hermite 補間式を正規記述する。
> `MathematicalContract.equation_ref` から `01_kinematics_beam.md#eq-...` 形式で参照される。
>
> **本台帳の非責務**: 要素行列の数値、収束 tol、ベンチマーク値は status ファイル側で管理。
> 本台帳は「式そのもの」と「定式化の網羅性」のみ責任を持つ。

## 表記

| 記号 | 意味 |
|---|---|
| $\boldsymbol{X}_I$ | ノード $I$ の初期配置座標（$I=0,1$ の 2 端点） |
| $\boldsymbol{x}_I = \boldsymbol{X}_I + \boldsymbol{u}_I$ | 現配置座標 |
| $\boldsymbol{u}_I,\,\boldsymbol{\theta}_I$ | ノード変位 / 回転（Timoshenko 梁では独立） |
| $L_0,\,L$ | 初期 / 現配置の弦長 |
| $\boldsymbol{R}_e$ | 要素 CR フレーム（$3\times 3$ 直交行列） |
| $\boldsymbol{u}_{\mathrm{loc}},\,\boldsymbol{\theta}_{\mathrm{loc}}$ | CR フレーム内の局所変位・回転 |
| $\boldsymbol{f}_{\mathrm{int}}$ | 要素内力ベクトル |
| $\boldsymbol{K}_{\mathrm{mat}},\,\boldsymbol{K}_{\mathrm{geo}}$ | 材料 / 幾何剛性 |
| $H_{00}, H_{10}, H_{01}, H_{11}$ | 三次 Hermite 基底関数（$s\in[0,1]$） |
| $\xi \in [-1,+1]$ | 要素自然座標（Gauss 積分で使用） |

→ 実装: `xkep_cae/elements/_beam_cr.py`, `xkep_cae/elements/_beam_section.py`,
`xkep_cae/elements/_beam_assembler.py`

---

<a id="eq-cr-frame"></a>

## 1. Co-Rotational（CR）フレーム

要素の剛体回転を分離するため、現配置の弦方向・初期配置のノーダル三脚から
回転行列 $\boldsymbol{R}_e$ を構築する:

$$
\boldsymbol{R}_e \;=\; \boldsymbol{R}_e\big(\boldsymbol{x}_0,\boldsymbol{x}_1,
\boldsymbol{\theta}_0,\boldsymbol{\theta}_1\big)
$$

局所変位（CR フレーム内）は:

<a id="eq-cr-local-disp"></a>

$$
\boldsymbol{u}_{\mathrm{loc}} \;=\; \boldsymbol{R}_e^\top(\boldsymbol{x}_1-\boldsymbol{x}_0)
- (\boldsymbol{X}_1-\boldsymbol{X}_0),
\qquad
\boldsymbol{\theta}_{\mathrm{loc},I} \;=\; \mathrm{Rot}^{-1}\!\big(\boldsymbol{R}_e^\top \boldsymbol{R}_I\big)
$$

→ 実装: `_beam_cr.py:_build_cr_frame` / `_compute_local_disp`

---

<a id="eq-tl-greenlagrange"></a>

## 2. Total-Lagrangian（TL）ひずみ

初期配置を基準とする Green-Lagrange ひずみ（1D 繊維方向成分）:

$$
E_{xx}(\xi) \;=\; \frac{1}{2}\Big(\lVert \partial_X\boldsymbol{x}(\xi)\rVert^2 - 1\Big)
$$

断面積分後の一般化ひずみ（軸・曲げ・剪断）:

<a id="eq-beam-generalized-strain"></a>

$$
\boldsymbol{\varepsilon} \;=\;
\big(\varepsilon_{\mathrm{axial}},\,\kappa_y,\,\kappa_z,\,\gamma_y,\,\gamma_z,\,\tau\big)^\top
$$

ここで $\varepsilon_{\mathrm{axial}} = u_{\mathrm{loc},x}/L_0$、
$\kappa_y,\kappa_z$ は曲率、$\gamma$ は剪断ひずみ、$\tau$ は捻じり。

→ 実装: `_beam_section.py:compute_strains`（fiber integrator の弾塑性では
`elements/fiber/integrator.py:FiberSectionIntegratorProcess`）

---

<a id="eq-ul-objective-rate"></a>

## 3. Updated-Lagrangian（UL）対象率

現配置を基準とする率形表現。CR 組込時は参照配置更新（`update_reference`）で
実現し、対象率は Jaumann 率を採用:

$$
\overset{\nabla}{\boldsymbol{\sigma}}
\;=\; \dot{\boldsymbol{\sigma}} - \boldsymbol{W}\boldsymbol{\sigma} + \boldsymbol{\sigma}\boldsymbol{W},
\qquad \boldsymbol{W} = \tfrac{1}{2}(\nabla\boldsymbol{v} - \nabla\boldsymbol{v}^\top)
$$

UL + CR 統合では対象率の回転項が $\boldsymbol{R}_e$ の更新で吸収される。

→ 実装: `_beam_cr.py:update_reference`（ステップ後の参照配置更新）。
UL アセンブラは `_beam_assembler.py:ULCRBeamAssemblerProcess`。

---

<a id="eq-hermite-basis"></a>

## 4. Hermite 三次補間基底

弧長パラメータ $s\in[0,1]$ に対する 4 基底:

$$
H_{00}(s) = 2s^3 - 3s^2 + 1,\qquad
H_{10}(s) = s^3 - 2s^2 + s,
$$

$$
H_{01}(s) = -2s^3 + 3s^2,\qquad
H_{11}(s) = s^3 - s^2
$$

要素内の中間点補間:

<a id="eq-hermite-interp"></a>

$$
\boldsymbol{p}(s) \;=\;
H_{00}(s)\,\boldsymbol{x}_0 + H_{10}(s)\,\boldsymbol{m}_0
+ H_{01}(s)\,\boldsymbol{x}_1 + H_{11}(s)\,\boldsymbol{m}_1
$$

ここで $\boldsymbol{m}_I$ は端点タンジェント（隣接ノードからの有限差分、
`_st_jacobian.py` で構築）。接触最近接点 $\boldsymbol{p}_A(s)$ もこの補間式を使う
（03 章 [#eq-hermite-pA](03_huber_contact_penalty.md#eq-hermite-pA)）。

→ 実装: `xkep_cae/contact/geometry/_st_jacobian.py`（タンジェント構築 + 微分）、
補間自体は `_compute.py:_closest_point_hermite_refine` 内で展開。

---

<a id="eq-beam-stiffness"></a>

## 5. 要素接線剛性（CR 展開）

CR フレームにおける局所接線剛性 $\boldsymbol{K}_{\mathrm{loc}}$ を大域に引き戻す:

$$
\boldsymbol{K}_e
\;=\;
\boldsymbol{B}^\top \boldsymbol{K}_{\mathrm{loc}}\,\boldsymbol{B}
\;+\; \boldsymbol{K}_{\mathrm{geo}}
$$

$\boldsymbol{B}$ は局所自由度↔全体自由度の変換行列。$\boldsymbol{K}_{\mathrm{geo}}$
は CR フレーム回転の 1 次感度（内力 $\boldsymbol{f}_{\mathrm{loc}}$ と $\partial
\boldsymbol{R}_e/\partial \boldsymbol{u}$ の積）から生じる幾何剛性。

→ 実装: `_beam_cr.py:tangent_stiffness` / `_beam_assembler.py::ULCRBeamAssemblerProcess._assemble_element`

---

## 6. 既存実装との trace

| 数式 | 実装位置 | 備考 |
|---|---|---|
| [#eq-cr-frame](#eq-cr-frame) | `_beam_cr.py:_build_cr_frame` | 三脚構築 |
| [#eq-cr-local-disp](#eq-cr-local-disp) | `_beam_cr.py:_compute_local_disp` | CR 内局所量 |
| [#eq-tl-greenlagrange](#eq-tl-greenlagrange) | `_beam_section.py:compute_strains` | 1D 繊維ひずみ |
| [#eq-beam-generalized-strain](#eq-beam-generalized-strain) | `elements/fiber/integrator.py:FiberSectionIntegratorProcess` | 断面積分 |
| [#eq-ul-objective-rate](#eq-ul-objective-rate) | `_beam_cr.py:update_reference` | UL 参照更新 |
| [#eq-hermite-basis](#eq-hermite-basis) | `contact/geometry/_st_jacobian.py` | H_00..H_11 |
| [#eq-hermite-interp](#eq-hermite-interp) | `contact/geometry/_compute.py` | 最近接点補間 |
| [#eq-beam-stiffness](#eq-beam-stiffness) | `_beam_cr.py:tangent_stiffness` | K_mat + K_geo |

---

## 関連 status

- status-279〜281: CR フレーム UL 参照配置更新（接触なし 90 度曲げ完走）
- status-329〜330: ファイバー梁 Phase F4/F5（`StrandFiberBeamProcess` 統合）
- status-271〜274: Hermite 非局所 $\partial g/\partial \boldsymbol{u}$ 隣接ノード拡張
- status-348（03 章）: 接触 $\boldsymbol{p}_A(s)$ が本章 Hermite 補間を経由する点で
  [#eq-hermite-pA](03_huber_contact_penalty.md#eq-hermite-pA) と相互参照
