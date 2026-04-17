# 02 — 接触ペア構築と最近接点（$s,t$）射影・StJacobian

[← README](../../README.md) | [← roadmap](../roadmap.md) | [← 数理台帳](README.md) | [← MCDD 設計仕様](../../xkep_cae/mathematics/docs/mathematics.md)

> **本台帳の責務**: 梁–梁接触ペアの最近接点 $(s,t)$ 射影、ギャップ $g$ の定義、
> および最近接点パラメータの変位感度 $\partial s/\partial \boldsymbol{u}$ /
> $\partial t/\partial \boldsymbol{u}$（StJacobian）を正規記述する。
> 03 章（Huber 接触ペナルティ）の前提条件となる幾何量を提供する。
>
> **本台帳の非責務**: broadphase 性能・KD-tree 実装詳細は `contact/_broadphase.py`
> の docstring を参照（台帳対象外）。

## 表記

| 記号 | 意味 |
|---|---|
| $\boldsymbol{p}_A(s),\,\boldsymbol{p}_B(t)$ | 要素 $A,B$ 上の候補最近接点（Hermite 補間）|
| $s,t \in [0,1]$ | 要素内弧長パラメータ |
| $\boldsymbol{r} = \boldsymbol{p}_B(t) - \boldsymbol{p}_A(s)$ | 距離ベクトル |
| $d = \lVert\boldsymbol{r}\rVert$ | 最近接距離 |
| $\hat{\boldsymbol{n}} = \boldsymbol{r}/d$ | 単位法線（$A\to B$）|
| $r_A,\,r_B$ | 梁半径（被膜含む場合は `radius_a/b`）|
| $g = d - (r_A + r_B)$ | 接触ギャップ（貫入時 $g<0$）|
| $\boldsymbol{P}_\perp = \boldsymbol{I} - \hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top$ | 法線直交射影 |

→ 実装: `xkep_cae/contact/geometry/_compute.py`,
`xkep_cae/contact/geometry/_st_jacobian.py`,
`xkep_cae/contact/_contact_pair.py`

---

<a id="eq-closest-point"></a>

## 1. 最近接点条件（梁–梁 line-to-line）

$d^2(s,t) = \lVert \boldsymbol{p}_B(t)-\boldsymbol{p}_A(s)\rVert^2$ の最小化:

<a id="eq-closest-residual"></a>

$$
\frac{\partial d^2}{\partial s} = -2\,\boldsymbol{r}\cdot \boldsymbol{p}_A'(s) = 0,
\qquad
\frac{\partial d^2}{\partial t} = +2\,\boldsymbol{r}\cdot \boldsymbol{p}_B'(t) = 0
$$

直交条件:

$$
\boldsymbol{r}\cdot\boldsymbol{t}_A(s) \;=\; 0,
\qquad
\boldsymbol{r}\cdot\boldsymbol{t}_B(t) \;=\; 0
$$

ここで $\boldsymbol{t}_A(s) = \boldsymbol{p}_A'(s)$ は要素接線。$s,t\in[0,1]$
範囲外のときは端点（$s\in\{0,1\}$）で clamp して再評価する。

→ 実装: `_compute.py:_closest_point_segments_batch`（初期推定）→
`_closest_point_hermite_refine`（Hermite Newton 反復）

---

<a id="eq-gap"></a>

## 2. ギャップ定義と符号

$$
g \;=\; d \;-\; (r_A + r_B)
$$

| 状態 | 判定 |
|---|---|
| $g > 0$ | 離反（接触力ゼロ） |
| $g = 0$ | 接触境界 |
| $g < 0$ | 貫入（ペナルティ力発生） |

<a id="inv-gap-definition"></a>

不変量: $d \ge 0$ より $g \ge -(r_A+r_B)$。実装上は被膜付き半径 $r$ が時間変化
しない（撚線モデル）ため、$\partial g/\partial \boldsymbol{u} = \partial d/\partial \boldsymbol{u}$。

→ 実装: `_contact_pair.py:_evolve_pair`（ペア評価）

---

<a id="eq-dr-du"></a>

## 3. 距離ベクトルの変位感度

$\boldsymbol{r}$ は $(s,t)$ と各ノード変位の両方に依存:

$$
\frac{\partial \boldsymbol{r}}{\partial \boldsymbol{u}}
\;=\;
\frac{\partial \boldsymbol{p}_B}{\partial \boldsymbol{u}}
- \frac{\partial \boldsymbol{p}_A}{\partial \boldsymbol{u}}
\;+\;\boldsymbol{p}_B'(t)\,\frac{\partial t}{\partial \boldsymbol{u}}
\;-\;\boldsymbol{p}_A'(s)\,\frac{\partial s}{\partial \boldsymbol{u}}
$$

最近接条件 [#eq-closest-residual](#eq-closest-residual) が満たされる点では
$\boldsymbol{r}\cdot \boldsymbol{p}_A'(s) = 0$ なので、$s$ 微分が $\hat{\boldsymbol{n}}$
方向成分に寄与しない:

<a id="eq-dd-du"></a>

$$
\frac{\partial d}{\partial \boldsymbol{u}}
\;=\; \hat{\boldsymbol{n}}^\top \frac{\partial \boldsymbol{r}}{\partial \boldsymbol{u}}
\;=\; \hat{\boldsymbol{n}}^\top \Big(\frac{\partial \boldsymbol{p}_B}{\partial \boldsymbol{u}}
- \frac{\partial \boldsymbol{p}_A}{\partial \boldsymbol{u}}\Big)
$$

この恒等式により、接触力計算で $\partial s/\partial \boldsymbol{u},\,\partial t/\partial
\boldsymbol{u}$ を露に扱わずに済む。

---

<a id="eq-st-jacobian"></a>

## 4. StJacobian（$\partial s/\partial \boldsymbol{u}$, $\partial t/\partial \boldsymbol{u}$）

最近接条件 [#eq-closest-residual](#eq-closest-residual) を $\boldsymbol{u}$ で
微分して陰関数の定理を適用すると、$2\times 2$ 系:

$$
\begin{pmatrix}
\boldsymbol{t}_A\!\cdot\!\boldsymbol{t}_A + \boldsymbol{r}\!\cdot\!\boldsymbol{t}_A'
& -\,\boldsymbol{t}_A\!\cdot\!\boldsymbol{t}_B \\[4pt]
-\,\boldsymbol{t}_A\!\cdot\!\boldsymbol{t}_B
& \boldsymbol{t}_B\!\cdot\!\boldsymbol{t}_B + \boldsymbol{r}\!\cdot\!\boldsymbol{t}_B'
\end{pmatrix}
\begin{pmatrix}\partial s/\partial \boldsymbol{u}\\ \partial t/\partial \boldsymbol{u}\end{pmatrix}
\;=\;
\begin{pmatrix}
+\,\boldsymbol{t}_A^\top\,\partial \boldsymbol{r}/\partial \boldsymbol{u} \\[2pt]
-\,\boldsymbol{t}_B^\top\,\partial \boldsymbol{r}/\partial \boldsymbol{u}
\end{pmatrix}
$$

を解く（status-292 の $2\times 2$ カップリング修正）。端部 clamp 時は
対応する行を $\partial s / \partial \boldsymbol{u} = 0$ に置き換える
（`unclamped` 遷移帯は status-291/293 で smooth 化）。

→ 実装: `_st_jacobian.py:_compute_st_jacobian` / `_compute_st_jacobian_batch`。
隣接ノード $\partial \boldsymbol{m}_I/\partial \boldsymbol{u}_{\mathrm{adj}}$ 拡張は
status-271〜274 で実装。

---

<a id="sec-exclude-same-strand"></a>

## 5. 同素線除外・端部除外

同一撚線素線内の自己接触と、素線端部要素の接触を除外する:

- **同素線除外**: `exclude_same_strand=True`（status-146 推奨構成）
- **端部除外**: `exclude_end_elements` で素線両端 N 要素を contact candidate から除外
  （status-296 の MPC+contact 安定化）

これらは候補ペア構築時のフィルタであり、ペア確定後の $(s,t)$ 射影式は不変。

→ 実装: `_broadphase.py`, `_contact_pair.py:_pair_passes_filter`

---

## 6. 既存実装との trace

| 数式 | 実装位置 | 備考 |
|---|---|---|
| [#eq-closest-point](#eq-closest-point) | `_compute.py:_closest_point_segments_batch` | 線形 2 次式初期推定 |
| [#eq-closest-residual](#eq-closest-residual) | `_compute.py:_closest_point_hermite_refine` | Hermite Newton |
| [#eq-gap](#eq-gap) | `_contact_pair.py:_evolve_pair` | ギャップ評価 |
| [#eq-dr-du](#eq-dr-du) | `_st_jacobian.py:_compute_dr_du_batch` | 距離ベクトル微分 |
| [#eq-dd-du](#eq-dd-du) | `strategy.py:_assemble_normal_force`（contact_force）| ノーマル射影 |
| [#eq-st-jacobian](#eq-st-jacobian) | `_st_jacobian.py:_compute_st_jacobian_batch` | 2×2 陰関数解 |

---

## 関連 status

- status-147/148: line-to-line Gauss 積分 + 同素線除外
- status-271〜274: Hermite 非局所 StJacobian 隣接ノード拡張
- status-291〜293: `s_unclamped` smooth 遷移（Hermite 誤差 20%→0.0001%）
- status-292: StJacobian 2×2 カップリング修正（K_st FD 94%→0.0001%）
- status-308: broadphase KD-tree 化
- 03 章 [#eq-dn-du](03_huber_contact_penalty.md#eq-dn-du): 本章 [#eq-dr-du](#eq-dr-du)
  を $\boldsymbol{P}_\perp$ で射影した法線方向感度
