# 03 — Huber 法線ペナルティ + Hertz 非線形 + K_c 項展開

[← README](../../README.md) | [← roadmap](../roadmap.md) | [← 数理台帳](README.md) | [← MCDD 設計仕様](../../xkep_cae/mathematics/docs/mathematics.md)

> **本台帳の責務**: 接触法線ペナルティ系の離散化方程式・項展開・不変量を
> TeX 文字列で正規記述する。`MathematicalContract.equation_ref` から
> `03_huber_contact_penalty.md#eq-...` 形式で参照される。
>
> **本台帳の非責務**: tol / 実測値 / 誤差は status ファイル側で管理する。
> 台帳は「式そのもの」と「項の網羅性」のみに責任を持つ。

## 表記

| 記号 | 意味 |
|---|---|
| $\boldsymbol{u}$ | 全体変位ベクトル（DOF 並び順は梁要素章 `01_kinematics_beam.md` 参照） |
| $g$ | 接触ペアのギャップ（貫入時 $g<0$）|
| $\hat{\boldsymbol{n}}$ | 接触法線（ペア $A\to B$ で長さ 1） |
| $d = \lVert\boldsymbol{p}_B-\boldsymbol{p}_A\rVert$ | 最近接点間距離 |
| $\boldsymbol{p}_A(s),\,\boldsymbol{p}_B(t)$ | Hermite 補間による最近接点（弧長パラメータ $s,t\in[0,1]$） |
| $k_{\mathrm{pen}}$ | ペナルティ係数（`PenaltyStrategy` で推定） |
| $\delta_h$ | Huber 遷移幅（`huber_delta_h` 直接指定 or $k_{\mathrm{pen}}/$`smoothing_delta`） |
| $\alpha$ | ペナルティ指数（$\alpha=1.0$ 線形 / $\alpha=1.5$ Hertz、status-285） |
| $p_n$ | 法線ペナルティ力スカラ（$p_n\ge 0$） |
| $\boldsymbol{f}_c$ | 接触力ベクトル（$\boldsymbol{f}_c=-p_n\hat{\boldsymbol{n}}$、`A` 側に作用） |
| $\boldsymbol{K}_c=\partial \boldsymbol{f}_c/\partial \boldsymbol{u}$ | 接触接線剛性 |
| $\boldsymbol{P}_\perp = \boldsymbol{I}-\hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top$ | 法線直交射影 |

→ 実装: `xkep_cae/contact/contact_force/strategy.py::HuberContactForceProcess`

---

<a id="eq-pn-huber"></a>
<a id="eq-pn"></a>

## 1. 法線ペナルティ力 $p_n$

### 1.1 Huber 平滑化 $\max(0,\cdot)$

$$
\operatorname{huber}(x,\delta_h) =
\begin{cases}
0, & x < -\delta_h \\
\dfrac{(x+\delta_h)^2}{4\delta_h}, & -\delta_h \le x \le \delta_h \\
x, & x > \delta_h
\end{cases}
$$

導関数（$C^0$ 連続）:

$$
\operatorname{huber}'(x,\delta_h) =
\begin{cases}
0, & x < -\delta_h \\
\dfrac{x+\delta_h}{2\delta_h}, & -\delta_h \le x \le \delta_h \\
1, & x > \delta_h
\end{cases}
$$

→ 実装: `HuberContactForceProcess._huber` / `_huber_deriv`（およびバッチ版 `_huber_batch` / `_huber_deriv_batch`）

### 1.2 線形ペナルティ（$\alpha=1$、status-222）

<a id="eq-pn-linear"></a>

$$
p_n = \operatorname{huber}(k_{\mathrm{pen}}\,(-g),\,\delta_h)
$$

### 1.3 Hertz 型非線形ペナルティ（$\alpha=1.5$、status-285）

<a id="eq-pn-hertz"></a>

$h := \operatorname{huber}(k_{\mathrm{pen}}\,(-g),\,\delta_h)$ とおき、

$$
p_n = \frac{h^\alpha}{k_{\mathrm{pen}}^{\alpha-1}}
\;\;=\;\;
k_{\mathrm{pen}}\left(\frac{h}{k_{\mathrm{pen}}}\right)^{\alpha}
$$

導関数（$h$ に関する）:

<a id="eq-dpn-dx"></a>

$$
\frac{\mathrm{d}p_n}{\mathrm{d}x}
= \alpha\,\left(\frac{h}{k_{\mathrm{pen}}}\right)^{\alpha-1}\operatorname{huber}'(x,\delta_h),
\quad x = k_{\mathrm{pen}}\,(-g)
$$

$\alpha=1.5$ は接触 ON/OFF 境界で力が緩やかに立ち上がり活性集合切替を平滑化する。

→ 実装: `HuberContactForceProcess._apply_power_law` / `_apply_power_law_deriv`

### 1.4 不変量

<a id="inv-pn-nonneg"></a>

$$
p_n \ge 0 \quad \forall\,g\in\mathbb{R},\;\delta_h>0,\;k_{\mathrm{pen}}>0,\;\alpha\ge 1
$$

→ 契約: `InequalityContract(name="p_n_nonneg", expr="p_n", kind="geq", bound="0", equation_ref="03_huber_contact_penalty.md#inv-pn-nonneg")`

---

<a id="eq-fc"></a>

## 2. 接触力ベクトル $\boldsymbol{f}_c$

ペア $A\to B$ で:

$$
\boldsymbol{f}_c^{(A)} = -\,p_n\,\hat{\boldsymbol{n}},
\qquad
\boldsymbol{f}_c^{(B)} = +\,p_n\,\hat{\boldsymbol{n}}
$$

DOF 全体への組み立てでは `s, t` の Hermite 形状係数 $c_i(s)$（または線形係数）で
分配する:

<a id="eq-fc-assembly"></a>

$$
\boldsymbol{f}_c[\,\mathrm{DOF}_{Ai}\,] \;{+}{=}\; -\,c_i(s)\,p_n\,\hat{\boldsymbol{n}},
\qquad
\boldsymbol{f}_c[\,\mathrm{DOF}_{Bj}\,] \;{+}{=}\; +\,c_j(t)\,p_n\,\hat{\boldsymbol{n}}
$$

→ 実装: `HuberContactForceProcess.evaluate`（`_extract_pair_arrays` のバッチ経路）

---

<a id="eq-kc-full-decomposition"></a>
<a id="eq-kc"></a>
<a id="eq-kc-def"></a>

## 3. 接触接線剛性 $\boldsymbol{K}_c$ の完全項展開

接触力は $\boldsymbol{f}_c^{(A)} = -p_n\hat{\boldsymbol{n}}$（A 側、ペナルティ符号）。
NR ソルバーは残差 $\boldsymbol{R} = \boldsymbol{f}_{\mathrm{int}} + \boldsymbol{f}_c - \boldsymbol{f}_{\mathrm{ext}}$
の接線として $\boldsymbol{K}_c \equiv \partial(-\boldsymbol{f}_c)/\partial \boldsymbol{u}
= \partial(p_n\hat{\boldsymbol{n}})/\partial \boldsymbol{u}$ を組む（符号規約、
`test_kc_component_fd.py:224` で FD 比較値は `-f_c`）。

これを連鎖律で展開:

$$
\boldsymbol{K}_c
\;=\;
\underbrace{\frac{\partial p_n}{\partial \boldsymbol{u}}\otimes\hat{\boldsymbol{n}}}_{\boldsymbol{K}_{\mathrm{mat,nn}}}
\;\;\underbrace{-\;\boldsymbol{K}_{\mathrm{geo}}}_{\displaystyle=\,p_n\,\partial \hat{\boldsymbol{n}}/\partial \boldsymbol{u}\;\text{（法線方向感度）}}
\;\;\underbrace{+\;\boldsymbol{K}_{\mathrm{closest}}}_{(s,t)\,\text{追従}}
\;\;\underbrace{+\;\boldsymbol{K}_{\mathrm{hermite,adj}}}_{\text{隣接ノード非局所}}
\;\;\underbrace{+\;\boldsymbol{K}_{\mathrm{st}}}_{(s,t)\,\partial s/\partial \boldsymbol{u}\;\text{残差}}
$$

**重要**: $-\boldsymbol{K}_{\mathrm{geo}}$ の項**そのもの**が
$p_n\cdot\partial\hat{\boldsymbol{n}}/\partial \boldsymbol{u}$（法線方向感度）を表現する。
従来 status-344 以前の「$\boldsymbol{K}_{\mathrm{mat,ndir}}$ 欠落」という診断は
**重複カウント誤認**だった（詳細 [#sec-ndir](#sec-ndir)）。`TermExpansionContract`
は 5 項で完結する。

ペアの組み立て前行列形（$3\times 3$ ブロック、Hermite 形状係数 $c_i,c_j$ 込み、A-A カップリング）:

<a id="eq-kc-pair-block"></a>

$$
\boldsymbol{K}_c^{(ij)} \;=\;
c_i\,c_j\Big[\,
\underbrace{w_{\mathrm{mat}}\,(\hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top)}_{\boldsymbol{K}_{\mathrm{mat,nn}}}
\;-\;
\underbrace{w_{\mathrm{geo}}\,\boldsymbol{P}_\perp}_{\boldsymbol{K}_{\mathrm{geo}}\;=\;p_n\,\partial\hat{\boldsymbol{n}}/\partial \boldsymbol{u}}
\,\Big]
\;+\; \boldsymbol{K}_{\mathrm{hermite,adj}}^{(ij)} \;+\; \boldsymbol{K}_{\mathrm{closest}}^{(ij)} \;+\; \boldsymbol{K}_{\mathrm{st}}^{(ij)}
$$

ここで:

$$
w_{\mathrm{mat}} = \frac{\mathrm{d}p_n}{\mathrm{d}x}\cdot k_{\mathrm{pen}}
\quad\text{（[#eq-dpn-dx](#eq-dpn-dx) 参照）},
\qquad
w_{\mathrm{geo}} = \frac{p_n}{d}
$$

導出（A-A 同側、$\partial \boldsymbol{r}/\partial \boldsymbol{x}_j^{(A)} = -c_j\boldsymbol{I}$ を使用）:

- $\partial(-g)/\partial \boldsymbol{x}_j^{(A)} = +c_j \hat{\boldsymbol{n}}^\top \Rightarrow \partial p_n/\partial \boldsymbol{x}_j^{(A)} = +c_j w_{\mathrm{mat}}\hat{\boldsymbol{n}}^\top$
- $\partial \hat{\boldsymbol{n}}/\partial \boldsymbol{x}_j^{(A)} = -(c_j/d)\boldsymbol{P}_\perp$

を $[\boldsymbol{K}_c]^{(i,j)}_{AA} = c_i\,[\partial p_n/\partial \boldsymbol{x}_j^{(A)}\cdot \hat{\boldsymbol{n}} + p_n\,\partial \hat{\boldsymbol{n}}/\partial \boldsymbol{x}_j^{(A)}]$ に代入すると
$c_i c_j [w_{\mathrm{mat}}\,\hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top - (p_n/d)\,\boldsymbol{P}_\perp]$ が得られ、`strategy.py:1595` の
`w_mat * nn - w_geo * I_nn` と一致する。

→ 実装: `HuberContactForceProcess.assemble_tangent`（`strategy.py:1595` 付近、バッチ経路）— **5 項で完結**。

### 3.1 項一覧（`TermExpansionContract.term_names` と一対一対応）

| `term_name` | 数式（ペア局所形） | 実装 Process | 状態 |
|---|---|---|---|
| `K_mat_nn` | $+c_i c_j\,w_{\mathrm{mat}}\,(\hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top)$ | `KcNormalStiffnessProcess` | ✅ status-350/351（K_hermite_adj 分離後はペア局所のみ）|
| `K_geo` | $-\,c_i c_j\,w_{\mathrm{geo}}\,\boldsymbol{P}_\perp$（$= p_n\,\partial\hat{\boldsymbol{n}}/\partial \boldsymbol{u}$ と同値）| `KcGeoStiffnessProcess` | ✅ status-350、**法線方向感度 $\partial\hat{\boldsymbol{n}}/\partial \boldsymbol{u}$ を内包**（status-353 訂正）|
| `K_closest` | $-\,p_n\,\hat{\boldsymbol{n}}\otimes\partial s/\partial \boldsymbol{u}$ 等（`dpn_ds * g_shape`）| `KcClosestPointStiffnessProcess` | ✅ status-351 で K_st 残差から分離 |
| `K_hermite_adj` | 隣接ノード $\partial \boldsymbol{p}_A/\partial \boldsymbol{u}_{\mathrm{adj}}$（mat-only、`w_mat * nn` のみ） | `KcHermiteNonlocalStiffnessProcess` | ✅ status-351（status-295 mat-only、**status-354 で I_nn フル項拡張の仮説 A は実験反証済み** — §7 参照） |
| `K_st` | $\partial \boldsymbol{f}_{\mathrm{raw}}/\partial s\;\otimes\;\partial s/\partial \boldsymbol{u}$ ほか（K_closest 分離後の残差項）| `ContactForceStStiffnessProcess` | ✅ status-350/351 |

→ 契約: `TermExpansionContract(name="K_c_term_expansion", total_name="K_c", term_names=("K_mat_nn","K_closest","K_hermite_adj","K_geo","K_st"), providers=(...), combinator="add_sub", equation_ref="03_huber_contact_penalty.md#eq-kc-full-decomposition")`

`combinator="add_sub"` は $\boldsymbol{K}_c = (\boldsymbol{K}_{\mathrm{mat,nn}}+\boldsymbol{K}_{\mathrm{closest}}+\boldsymbol{K}_{\mathrm{hermite,adj}}+\boldsymbol{K}_{\mathrm{st}}) - \boldsymbol{K}_{\mathrm{geo}}$ の符号付き加算を表す。

---

<a id="sec-ndir"></a>

## 4. 法線方向感度 $\partial \hat{\boldsymbol{n}}/\partial \boldsymbol{u}$ と $\boldsymbol{K}_{\mathrm{geo}}$ の同一性（status-353 訂正）

最近接点距離ベクトル $\boldsymbol{r} = \boldsymbol{p}_B(t)-\boldsymbol{p}_A(s)$ から
$\hat{\boldsymbol{n}} = \boldsymbol{r}/d$、$d=\lVert\boldsymbol{r}\rVert$。

<a id="eq-dn-du"></a>

$$
\frac{\partial \hat{\boldsymbol{n}}}{\partial \boldsymbol{u}}
\;=\;
\frac{1}{d}\,\boldsymbol{P}_\perp\,\frac{\partial \boldsymbol{r}}{\partial \boldsymbol{u}}
\;=\;
\frac{1}{d}\left(\boldsymbol{I}-\hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top\right)\,
\frac{\partial \boldsymbol{r}}{\partial \boldsymbol{u}}
$$

A-A 同側（$\partial \boldsymbol{r}/\partial \boldsymbol{x}_j^{(A)} = -c_j\boldsymbol{I}$）で
$\partial \hat{\boldsymbol{n}}/\partial \boldsymbol{x}_j^{(A)} = -(c_j/d)\boldsymbol{P}_\perp$。
[#eq-kc-full-decomposition](#eq-kc-full-decomposition) の連鎖律で外側の $p_n$ と
組み立て係数 $c_i$ を掛けると:

$$
p_n\cdot \frac{\partial \hat{\boldsymbol{n}}}{\partial \boldsymbol{u}}\;\Big|_{\text{ペア局所 A-A}}
\;=\;
-\,c_i c_j\,\frac{p_n}{d}\,\boldsymbol{P}_\perp
\;\equiv\;
-\,\boldsymbol{K}_{\mathrm{geo}}^{(ij)}
$$

**すなわち $\boldsymbol{K}_{\mathrm{geo}} = -\,p_n\,\partial\hat{\boldsymbol{n}}/\partial \boldsymbol{u}$ のペア局所形
そのもの**であり、両者は符号・係数とも同一。`strategy.py:1595` の `-w_geo * I_nn`
項がこれを実装しており、$w_{\mathrm{geo}} = p_n/d$ の $1/d$ 因子は
$\partial \hat{\boldsymbol{n}}/\partial \boldsymbol{u}$ 由来（`[#eq-dn-du]` の $1/d$）である。

### 4.1 status-344〜352 の「K_mat,ndir 欠落」診断誤りの訂正（status-353）

status-289〜344 の K_c FD 調査で「x/z 成分カップリング欠落」が観測された際、
旧 Phase C-3 計画（status-346 当初の mathematics.md / status-352 の B 候補）は
これを「$\boldsymbol{K}_{\mathrm{mat,ndir}}$ が独立の項として未実装」と診断した。
この診断は **$\boldsymbol{K}_{\mathrm{geo}}$ が既に法線方向感度を含んでいる事実を
見落とした** もので、status-353 の数理再検証により以下に訂正する:

| 過去の主張（status-346〜352） | 訂正（status-353）|
|---|---|
| $\boldsymbol{K}_{\mathrm{geo}}$ と $\boldsymbol{K}_{\mathrm{mat,ndir}}$ は別項、重み $p_n/d$ vs $p_n$ | **同一項**、重み $p_n/d$（$1/d$ は $\hat{\boldsymbol{n}} = \boldsymbol{r}/d$ 由来の内在項） |
| `KcNormalDirectionStiffnessProcess` の新設で x/z 欠落解消 | `K_geo` は既実装のため新設は**二重計上**となる |
| Phase C-3 で 6 項 `TermExpansionContract` 化 | **5 項で完結**（status-351 で既達） |

status-344 で観測された x/z カップリング残差（`mat_only` rel_err mean=44%, comp_x max=98%）
について、status-353 で提示された仮説 A（「`K_hermite_adj` の `I_nn` 隣接拡張
追加で改善」）は **status-354 の直接実験で反証された**（`test_helical_3d_hermite`
rel_err **1.795% → 38.49%**、§7 参照）。mat-only 近似は s-tracking 補償の
実装上の要請であり、真の残差源は他の 3 経路（hypothesis B/C/D）に再配分する。
詳細は status-354 および §7 を参照。

→ 実装: **既存 `KcGeoStiffnessProcess`** が $-p_n\,\partial\hat{\boldsymbol{n}}/\partial \boldsymbol{u}$ のペア局所形を担う。新規 Process 追加は不要。

---

<a id="eq-kmat"></a>

## 5. $\boldsymbol{K}_{\mathrm{mat,nn}} - \boldsymbol{K}_{\mathrm{geo}}$ の対称性

ペア局所の $3\times 3$ 表現 [#eq-kc-pair-block](#eq-kc-pair-block) の材料項
（法線剛性 + 法線方向感度）は対称:

$$
\big(\boldsymbol{K}_{\mathrm{mat,nn}} - \boldsymbol{K}_{\mathrm{geo}}\big)^{(ij)}
\;=\;
c_i c_j\,\big[\,w_{\mathrm{mat}}\,\hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top
\;-\;w_{\mathrm{geo}}\,\boldsymbol{P}_\perp\,\big]
$$

$\hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top$ と $\boldsymbol{P}_\perp$ は
いずれも $3\times 3$ 対称行列であり、$(i,j)\leftrightarrow(j,i)$ で $c_i c_j$
対称性から:

<a id="sym-kmat"></a>

$$
\big(\boldsymbol{K}_{\mathrm{mat,nn}} - \boldsymbol{K}_{\mathrm{geo}}\big)^{(ij)}
\;=\;
\big[\big(\boldsymbol{K}_{\mathrm{mat,nn}} - \boldsymbol{K}_{\mathrm{geo}}\big)^{(ji)}\big]^\top
$$

→ 契約: `SymmetryContract(name="K_mat_symmetric", matrix_name="K_mat_nn_minus_geo", kind="symmetric", equation_ref="03_huber_contact_penalty.md#sym-kmat")`（status-353 で `matrix_name` を `K_mat` から `K_mat_nn_minus_geo` に訂正予定）

注意: 現行実装の $\boldsymbol{K}_c$ は $\boldsymbol{K}_{\mathrm{st}}$（摩擦・滑り
追従）と $\boldsymbol{K}_{\mathrm{closest}}$ を含むため**全体としては非対称**。
本契約は対称部分（`KcNormalStiffnessProcess` + `KcGeoStiffnessProcess`）にのみ
適用する。$\boldsymbol{K}_{\mathrm{hermite,adj}}$ は隣接ノード DOF の非対称展開で
あるため別契約で扱う（status-353 で検討）。

---

<a id="eq-kc-fd"></a>

## 6. FD 整合性

$\boldsymbol{K}_c$ の全体（または項別合計）が $\boldsymbol{f}_c$ の有限差分と一致する:

$$
\boldsymbol{K}_c\,\delta\boldsymbol{u}
\;\approx\;
\frac{\boldsymbol{f}_c(\boldsymbol{u}+\varepsilon\,\delta\boldsymbol{u})-\boldsymbol{f}_c(\boldsymbol{u})}{\varepsilon}
$$

→ 契約: `FDConsistencyContract(name="K_c_fd_consistency", vector_name="f_c", jacobian_name="K_c", equation_ref="03_huber_contact_penalty.md#eq-kc-fd", severity="nightly")`

→ 検証 Process: `ContactKcComponentFDDiagnosticProcess`（status-343/344/345）。
`@verified_by("K_c_fd_consistency", ContactKcComponentFDDiagnosticProcess)`
で Phase E（status-356）に紐付け予定。

### 6.1 項別 FD 整合性（status-352 以降）

各 `K_term_k` 単独についても FD 整合性を要求する:

<a id="eq-kc-term-fd"></a>

$$
\boldsymbol{K}_{\mathrm{term},k}\,\delta\boldsymbol{u}
\;\approx\;
\frac{\boldsymbol{f}_{\mathrm{term},k}(\boldsymbol{u}+\varepsilon\,\delta\boldsymbol{u})-\boldsymbol{f}_{\mathrm{term},k}(\boldsymbol{u})}{\varepsilon}
$$

ここで $\boldsymbol{f}_{\mathrm{term},k}$ は当該項のみの「仮想力」分解
（`KcNormalStiffness` なら $-p_n\hat{\boldsymbol{n}}$ で $\hat{\boldsymbol{n}}$ を
凍結したもの、等）。tol は status-353 で項ごとに決定する。

---

## 7. Hermite 隣接ノード非局所項（status-271〜274）

最近接点 $\boldsymbol{p}_A(s)$ は Hermite 補間により**当該要素の 2 端点ノードと
そのタンジェント** $\boldsymbol{m}_0,\boldsymbol{m}_1$ に依存する。タンジェント
は隣接要素の端点座標から有限差分で構成されるため、$\boldsymbol{p}_A$ は実質的に
**隣接ノードの座標にも依存**する:

<a id="eq-hermite-pA"></a>

$$
\boldsymbol{p}_A(s) =
H_{00}(s)\,\boldsymbol{x}_0 + H_{10}(s)\,\boldsymbol{m}_0
+ H_{01}(s)\,\boldsymbol{x}_1 + H_{11}(s)\,\boldsymbol{m}_1
$$

$$
H_{00} = 2s^3-3s^2+1,\;\; H_{10} = s^3-2s^2+s,\;\;
H_{01} = -2s^3+3s^2,\;\; H_{11} = s^3-s^2
$$

frozen-m 部分解消（status-294）後は

$$
\boldsymbol{m}_0 \approx \tfrac{1}{2}(\boldsymbol{x}_1-\boldsymbol{x}_{-1}),\quad
\boldsymbol{m}_1 \approx \tfrac{1}{2}(\boldsymbol{x}_2-\boldsymbol{x}_{0})
$$

から $\partial \boldsymbol{p}_A/\partial \boldsymbol{u}_{\mathrm{adj}}$ が
$H_{10},H_{11}$ を介して非ゼロとなる。

→ 実装: `xkep_cae/contact/geometry/_st_jacobian.py`、隣接ノード DOF 拡張は
`HuberContactForceProcess.assemble_tangent` の `_adj_node_map`/`_node_counts`
分岐（status-273/274）。

→ 契約: status-351 で `KcHermiteNonlocalStiffnessProcess` が抽出され、
`TermExpansionContract.term_names="K_hermite_adj"` で網羅性を要求している。

**mat-only 形態の正当性（status-354 実験検証）**:
status-353 で提示された仮説 A（「K_hermite_adj に `-w_geo * I_nn` の隣接
ノード項を追加すると x/z 残差が改善する」）は **status-354 の実験で反証**
された。`KcHermiteNonlocalStiffnessProcess` の `K_3x3_mat` を
`w_mat * nn - w_geo * I_nn`（ペア局所と同形）に拡張して
`test_kc_component_fd.py::test_helical_3d_hermite` を実行した結果:

| 構成 | rel_err | 備考 |
|------|---------|------|
| mat-only（現行、status-295） | **1.795%** | 既存ゲートテスト合格 |
| `mat + I_nn`（仮説 A 実装） | **38.49%** | 21 倍悪化、ゲートテスト fail |

この反証の数理的解釈:  隣接ノード $\boldsymbol{x}_{\mathrm{adj}}$ の摂動は
(i) Hermite 接線 $\boldsymbol{m}$ 経由で $\boldsymbol{p}_A$ を直接動かす
成分と、(ii) min-distance 射影で $(s,t)$ を再決定する s-tracking
補償成分の 2 経路を持つ。`I_nn` 方向（法線直交）の変動は (ii) により
ほぼ相殺されるのに対し、`n⊗n` 方向（ギャップ方向）の変動は (i) から (ii) への
漏出が小さく維持される。Process は (i) のみを計算するため、数学的に
純粋な chain-rule（フル項）よりも **mat-only のほうが FD（実測）と一致する**
（status-295 の設計意図の実証）。

**Phase C-3 再々定義**: 19 本撚線 Type D stall の真の原因候補は以下に再配分:

1. **B**: `KcClosestPointStiffnessProcess` を隣接ノード DOF 列にも拡張
   （$\partial s/\partial \boldsymbol{u}_{\mathrm{adj}}$ / $\partial t/\partial \boldsymbol{u}_{\mathrm{adj}}$
   を組み込み、上記 (ii) 経路を解析的に実装）
2. **C**: NR 反復内の active set 凍結近似の弛緩（status-258 観察と整合）
3. **D**: 摩擦 `K_st` 隣接拡張での類似項不整合

status-354 はコードの mat-only を維持し、数理台帳と実装の整合性を確認した。

---

## 8. 既存実装との trace（status-353 時点）

| 数式 | 実装位置 | 備考 |
|---|---|---|
| [#eq-pn-huber](#eq-pn-huber) | `strategy.py:_huber` / `_huber_batch` | C¹ 連続 |
| [#eq-pn-hertz](#eq-pn-hertz) | `strategy.py:_apply_power_law` | $\alpha=1.5$ |
| [#eq-dpn-dx](#eq-dpn-dx) | `strategy.py:_apply_power_law_deriv` | tangent 用 |
| [#eq-fc](#eq-fc) | `strategy.py:HuberContactForceProcess.evaluate` | バッチ経路 |
| [#eq-kc-full-decomposition](#eq-kc-full-decomposition) | `strategy.py:tangent_components`（orchestrator）| 5 項 Process 統合、status-350/351 で抽出 |
| [#eq-kc-pair-block](#eq-kc-pair-block) | `strategy.py:1595`（`w_mat * nn - w_geo * I_nn`）| `KcNormalStiffnessProcess` + `KcGeoStiffnessProcess` で分解済み |
| [#eq-dn-du](#eq-dn-du) | `strategy.py:1595` の `-w_geo * I_nn` 項（`KcGeoStiffnessProcess`）| **既実装。status-353 で $\boldsymbol{K}_{\mathrm{geo}}$ との同一性を確立** |
| [#sym-kmat](#sym-kmat) | — | 静的契約のみ、Phase E で C18 検査 |
| [#eq-kc-fd](#eq-kc-fd) | `xkep_cae/verify/kc_component_fd.py` | `@verified_by` 紐付け予定 |
| [#eq-kc-term-fd](#eq-kc-term-fd) | — | status-353 以降の項別 FD 整合性、tol 未決定 |
| [#eq-hermite-pA](#eq-hermite-pA) | `_st_jacobian.py` | status-271〜274 拡張、隣接 I_nn 非拡張は **status-354 で仮説 A 反証** — mat-only 継続（§7 参照） |

---

## 関連 status

- status-222: NCP 除去・純粋ペナルティ確立
- status-259〜261: Huber `smoothing_delta` パイプライン貫通 + `huber_delta_h` 直接指定
- status-285: Hertz 型非線形ペナルティ ($\alpha=1.5$)
- status-271〜274: Hermite 非局所 $\partial g/\partial \boldsymbol{u}$ 隣接ノード拡張
- status-289〜296: K_c FD 不整合追跡（s_unclamped、StJacobian、frozen-m、K_c_adj mat-only）
- status-342〜345: 19 本撚線 K_c 成分分解 FD 診断 + report 精度バグ訂正
- status-346〜347: MCDD Phase A（`MathematicalContract` + `ProcessContractRegistry`）
- status-348（本台帳初版）: Phase B-1 — 03 章先行整備（K_mat,ndir を独立項として記述、後に訂正）
- status-349: Phase B-2 — 他 5 章 + `equation_index.py` + C15 拡張
- status-350〜351: Phase C-1/C-2 — 項別 Process 抽出（5 項独立 Process 化）
- status-352: 計画書ロスト記録 + Phase C-3 前提の数理検証（K_mat,ndir ≡ K_geo の指摘）
- **status-353（§3/§4/§5/§8 訂正）**: K_mat,ndir と K_geo の同一性を確立し、6 項記述を 5 項に訂正。x/z カップリング残差の再原因候補を K_hermite_adj mat-only に再設定
- **status-354（§7/§3.1/§4/§8 仲裁追記）**: 仮説 A（K_hermite_adj フル項拡張）を実験反証（`test_helical_3d_hermite` rel_err 1.795% → 38.49%）、mat-only 継続を実証。Phase C-3 再々定義 = hypothesis B/C/D 探索へ
