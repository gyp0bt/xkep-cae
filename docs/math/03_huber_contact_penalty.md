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
| `K_hermite_adj` | 隣接ノード $\partial \boldsymbol{p}_A/\partial \boldsymbol{u}_{\mathrm{adj}}$ 直接経路 (i) フル項（`w_mat n⊗n - w_geo I_nn`）| `KcHermiteNonlocalStiffnessProcess` | ✅ status-356（status-354 の I_nn 単独追加反証を受けて s-tracking adj と同時導入で FD 機械精度一致）— §7 参照 |
| `K_st` | $\partial \boldsymbol{f}_{\mathrm{raw}}/\partial s\;\otimes\;\partial s/\partial \boldsymbol{u}$ ほか（K_closest 分離後の残差項、status-356 で active×adj ブロックに拡張）| `ContactForceStStiffnessProcess` | ✅ status-350/351/356 |

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
追加で改善」）は status-354 の単独実装で反証された（rel_err 1.795%→38.49%）
が、**status-356 で仮説 A と仮説 B（`K_closest` / `K_st` の active×adj 拡張）を
同時導入して 2 経路を同時に実装すると $P_\perp$ 成分が相殺し、
`test_helical_3d_hermite` の rel_err は 1.795% → 2.18e-07 に 5 桁改善した**
（§7.3 参照）。すなわち仮説 A は数理的には正しく、単独では過剰計上だが
仮説 B と同時適用することで機械精度に達する。

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

### 7.1 2 経路解析（status-354 実験 + status-356 解決）

隣接ノード $\boldsymbol{x}_{\mathrm{adj}}$ の摂動に対する $\boldsymbol{f}_c$ の全微分は
**chain-rule で 2 経路**に分解される:

$$
\frac{\mathrm{d}\boldsymbol{f}_c}{\mathrm{d}\boldsymbol{u}_{\mathrm{adj}}}
\;=\;
\underbrace{\frac{\partial \boldsymbol{f}_c}{\partial \boldsymbol{u}_{\mathrm{adj}}}\bigg|_{s,t}}_{\text{(i) 直接経路}}
\;+\;
\underbrace{\frac{\partial \boldsymbol{f}_c}{\partial s}\,\frac{\mathrm{d}s}{\mathrm{d}\boldsymbol{u}_{\mathrm{adj}}}
+\frac{\partial \boldsymbol{f}_c}{\partial t}\,\frac{\mathrm{d}t}{\mathrm{d}\boldsymbol{u}_{\mathrm{adj}}}}_{\text{(ii) s-tracking 補償経路}}
$$

ここで:

- **(i) 直接経路**: $(s,t)$ 固定で $\boldsymbol{x}_{\mathrm{adj}}$ が Hermite 接線
  $\boldsymbol{m}$ 経由で $\boldsymbol{p}_A$ を動かし、$p_n$ と $\hat{\boldsymbol{n}}$ の
  両者を変化させる。`K_hermite_adj` が担当すべき経路。フル展開は
  [#eq-kc-pair-block](#eq-kc-pair-block) と同じ $c_i c_j[w_{\mathrm{mat}}\,\hat{n}\hat{n}^\top - w_{\mathrm{geo}}\,P_\perp]$。
- **(ii) s-tracking 補償経路**: min-distance 射影の停留条件
  $\partial d^2/\partial s = 0$、$\partial d^2/\partial t = 0$ を
  $\boldsymbol{u}_{\mathrm{adj}}$ で微分すると $\mathrm{d}s/\mathrm{d}\boldsymbol{u}_{\mathrm{adj}}$、
  $\mathrm{d}t/\mathrm{d}\boldsymbol{u}_{\mathrm{adj}}$ が決まる。`KcClosestPointStiffnessProcess`
  および `ContactForceStStiffnessProcess`（residual 項）が担当すべき経路。

### 7.2 相殺定理（$P_\perp$ 成分の消去）

停留条件から $\hat{\boldsymbol{n}}\perp \partial \boldsymbol{p}_A/\partial s$、
$\hat{\boldsymbol{n}}\perp \partial \boldsymbol{p}_B/\partial t$ が成り立つため、
(ii) の寄与は $\partial \boldsymbol{f}_c/\partial s$ と
$\partial \boldsymbol{f}_c/\partial t$ を通じて **$P_\perp$ 方向にのみ** $\boldsymbol{f}_c$ を
変化させる（$n\otimes n$ 方向は $\mathrm{d}g/\mathrm{d}\boldsymbol{u}_{\mathrm{adj}}$ に含まれ (i) 側に残る）。
(i) のフル項 $w_{\mathrm{mat}}\,\hat{n}\hat{n}^\top - w_{\mathrm{geo}}\,P_\perp$ の
うち $-w_{\mathrm{geo}}\,P_\perp$ 成分は (ii) の $P_\perp$ 寄与と**符号が逆で同じ
オーダー**になり、(i)+(ii) 合計で $P_\perp$ が相殺し、FD で観測されるのは
$w_{\mathrm{mat}}\,\hat{n}\hat{n}^\top$ に近い値になる。

### 7.3 status-354 反証 ⇒ status-356 解決

この数理は 2 回の実装実験で検証された:

| 構成 | `K_hermite_adj` | `K_closest` / `K_st` adj | `test_helical_3d_hermite` rel_err | 観察 |
|------|-----------------|---------------------------|-----------------------------------|------|
| status-295〜353（ベースライン） | mat-only（`w_mat n⊗n` のみ）| 未拡張（0）| **1.795%** | (i) 部分のみ・(ii) 未実装で近似一致 |
| status-354（仮説 A 単独）| フル項（`w_mat n⊗n - w_geo I_nn`） | 未拡張 | **38.49%**（21x 悪化） | (i) の $P_\perp$ を入れたが (ii) で相殺する相手がない |
| status-356（仮説 A + 仮説 B 同時導入）| **フル項** | **active×adj 拡張** | **2.18e-07**（5 桁改善）| (i)+(ii) の $P_\perp$ が相殺し FD 機械精度一致 |

status-354 の「mat-only が最良」という観察は **(ii) 未実装のワークアラウンド**
であり、理論的には (i) のフル項を入れて (ii) と相殺させるのが正しい。
status-356 は `KcHermiteNonlocalStiffnessProcess.process()` の `K_3x3_mat` を
`w_mat * nn - w_geo * I_nn` に戻し、同時に `ContactForceStStiffnessProcess`
の COO 構築で `ds_du_adj` / `dt_du_adj` 経由の active×adj ブロックを追加
することで、両経路を同時に Process 側に実装した。

### 7.4 診断裏付け（status-355/356）

`work/beam_hysteresis/14_kc_closest_adj_diagnostic.py` で `test_helical_3d_hermite`
シナリオの $\boldsymbol{K}_c$ を (active, adj) × (active, adj) の 4 ブロックに
分解した FD 診断:

| ブロック | status-355 実測（ベースライン）| status-356 実測（両経路実装後）|
|---|---|---|
| active×active | $\lVert\mathrm{diff}\rVert = 1.20\mathrm{e}{-3}$（rel_err 2.2e-7）| 1.20e-3（不変） |
| **active×adj** | $\lVert\mathrm{diff}\rVert = \mathbf{98.52}$（rel_err 16.4%）| **4.75e-05**（6 桁改善）|
| adj×active | 0（f_c は active 行のみ出力）| 0（不変）|
| adj×adj | 0 | 0（不変）|
| **全体** | rel_err **1.795%** | **2.18e-07** |

status-355 が予言した「active×adj ブロックに diff が 100% 局在」の仮説 B
目標 `||diff[ax]|| 98.52 → <1e-3` は **4.75e-05 で約 6 桁オーバーシュート**
で達成された（機械精度水準）。

→ 実装: `strategy.py::KcHermiteNonlocalStiffnessProcess.process()` で
`K_3x3_mat = w_mat * nn - w_geo * I_nn`（フル項）、
`strategy.py::ContactForceStStiffnessProcess._process_batch_term` の
`term in {"closest","residual"}` 両経路で
$K_{\mathrm{local,adj}} = -(\partial \boldsymbol{f}/\partial s \otimes \mathrm{d}s/\mathrm{d}\boldsymbol{u}_{\mathrm{adj}} + \partial \boldsymbol{f}/\partial t \otimes \mathrm{d}t/\mathrm{d}\boldsymbol{u}_{\mathrm{adj}})$
を `adj_gdofs` に COO 追加。`adj_node_counts` は
`ContactForceStStiffnessInput` 新フィールドで `HuberContactForceProcess.tangent`
から貫通配線、`_batch_dm_ext_coeffs` ヘルパで 2 箇所の dm_ext 計算を共通化
（脱法実装 pattern 3「類似コード二重実装」回避）。

---

## 9. Augmented Lagrangian 動機と Uzawa 外側ループ（status-221 凍結 / status-376 限定再導入）

<a id="sec-al"></a>

### 9.1 古典的 Augmented Lagrangian と Uzawa 反復

純粋ペナルティ法は $k_{\mathrm{pen}} \to \infty$ の漸近で接触制約 $g \ge 0$ を厳密化する
が、有限 $k_{\mathrm{pen}}$ では微小貫入 $g < 0$ が残る。Augmented Lagrangian
（拡大ラグランジアン）は **per-pair Lagrange 乗数** $\lambda_k \ge 0$ を導入し、
ペナルティ力を補強する:

<a id="eq-al-pn"></a>

$$
p_n^{\mathrm{AL}}(g, \lambda) \;=\; \max\bigl(0,\; \lambda + p_n^{\mathrm{huber}}(g)\bigr)
$$

ここで $p_n^{\mathrm{huber}}(g) = \operatorname{huber}(k_{\mathrm{pen}}(-g), \delta_h)$
は §1 のペナルティ力。NR 内側ループでは $\lambda$ を**固定値**として扱い、$p_n^{\mathrm{AL}}$ を
$f_c$ アセンブリと $K_c$ 線形化の両方で使用する。NR 収束後、外側ループで
**Uzawa 更新**を行う:

<a id="eq-uzawa"></a>

$$
\lambda_k \;\leftarrow\; \max\bigl(0,\; p_n^{\mathrm{AL},*}_k\bigr) \;=\; \max\bigl(0,\; \lambda_k + p_n^{\mathrm{huber}}(g^*_k)\bigr)
$$

ここで $g^*_k$ は内側 NR が収束したときの gap。AL の漸近収束（$\lambda \to$ 真の
Lagrange 乗数 = 接触圧）により、有限 $k_{\mathrm{pen}}$ で残存していた貫入が
反復ごとに減少する。

### 9.2 K_c の整合性（modified Newton 不要）

$\lambda$ は NR 内側で定数（$\partial\lambda/\partial \boldsymbol{u} = \boldsymbol{0}$）
なので、AL を導入しても $K_c$ の形は変わらず、ただし重み係数が:

| 項 | 重み（純ペナルティ） | 重み（AL 適用時） |
|---|---|---|
| $K_{\mathrm{mat,nn}}$ | $w_{\mathrm{mat}} = h'(x) k_{\mathrm{pen}}$ | 不変（$\partial p_n^{\mathrm{huber}}/\partial \boldsymbol{u}$ のみ捕捉） |
| $K_{\mathrm{geo}}$ | $w_{\mathrm{geo}} = p_n^{\mathrm{huber}}/d$ | $w_{\mathrm{geo}} = p_n^{\mathrm{AL}}/d$（$\lambda$ 寄与込み）|

実装上は `pair.state.p_n` を $p_n^{\mathrm{AL}}$ で更新するだけで $K_{\mathrm{geo}}$ が
自動的に整合する（[#eq-dn-du](#eq-dn-du) と $K_{\mathrm{geo}}$ の同一性、§4 参照）。
$K_{\mathrm{mat,nn}}$ は $h'(x)$ 由来の penalty 接線そのままで正しく、$\lambda$ の
$\partial/\partial \boldsymbol{u}$ がゼロであることを忠実に反映する。

### 9.3 status-221 における凍結根拠

status-219〜221 で動的接触三点曲げが収束しないバグの原因究明過程で、Uzawa 外側
ループ (`n_uzawa_max=5`) の効果を実測した結果、以下が判明した:

1. `n_uzawa_max=1`（純粋ペナルティ）と `n_uzawa_max≥2`（AL 反復）の最終解はほぼ
   同等（接触力 132N、変位差 < 1%）
2. $k_{\mathrm{pen}}$ の自動推定（`AutoEALPenalty` の `c0·M_ii·0.2`）が十分大きく、
   有限 $k_{\mathrm{pen}}$ 由来の貫入が既に許容範囲（< 0.01·radius）
3. AL ループの追加コスト（NR 反復 ×2〜5）に見合う精度向上が観測されない
4. **摩擦接線剛性の符号問題**（status-147、§9.5 参照）が NCP 鞍点系で再発しやすい

これら 4 点から status-221 で `n_uzawa_max=1` をデフォルトに固定し、status-222 で
Uzawa 関連の `lam_all` 管理 / `UzawaUpdateProcess` / `n_uzawa_max` パラメータを
**完全削除**した（凍結ではなく削除）。

### 9.4 status-376 限定再導入の動機

status-357〜375 で 19 本撚線 90° 曲げの Type D stall（K_c x/z カップリング不整合
領域）に対して候補 (a)/(a')/(c)/(d)/(e)/(g1)/(g3) を全て却下した結果、
**候補 (g) サブライン最後の (g2) AL 限定再導入**が残存した。status-373/374/375
の引継ぎでは:

- 内側 NR（純粋ペナルティ） + **AL 外側ループ最大 1〜2 サイクル** の二重ループ化
- $k_{\mathrm{pen}}$ 自動推定値の周辺で $\lambda$ を 1〜2 回更新することで「19 本
  Type D stall 断面（active 集合振動 + tangent 不整合）」を escape できるか検証
- 法線成分のみ AL 適用（摩擦は §9.5 の符号問題回避のため対象外）

を gate `19 本 frac ≥ 0.6`（baseline 0.3739 比 +60%）で評価する。

### 9.5 摩擦接線剛性の符号問題（status-147 残存リスク）

status-147 で NCP 鞍点系の摩擦接線剛性 `K_t` の符号が以下の二者択一であることが判明:

- **正符号** $K_t = +k_t (g_t \otimes g_t)$: 正定値で線形ソルブは安定だが、
  Newton が **slip 平衡に収束**して stick が成立しない
- **負符号** $K_t = -k_t (g_t \otimes g_t)$（解析的に正しい）: 不定値となり Schur
  complement 分解が不安定化

NCP では Alart-Curnier 拡大鞍点系での解決が必要だが実装が複雑。一方
**smooth penalty + Uzawa 外側ループ**（本章 §9.1 形式）では NCP 鞍点系を経由しないため
この符号問題は構造的に回避される（status-147 の `verify_smooth_penalty_friction_bend.py`
で実証済）。

status-376 の AL 再導入は **法線成分 $\lambda$ のみ更新**し、摩擦の rate-mapping
は status-225 で確立した `K_t` 構成（$dq/du \otimes G_{t\alpha} + q_\alpha \cdot dG_{t\alpha}/du$）を
そのまま使用する。これにより status-147 の摩擦接線符号問題は再活性化しない設計とする。

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
| [#eq-hermite-pA](#eq-hermite-pA) | `_st_jacobian.py` + `strategy.py::KcHermiteNonlocalStiffnessProcess`（直接経路 i）+ `ContactForceStStiffnessProcess._process_batch_term`（s-tracking 経路 ii、active×adj COO）| status-271〜274 拡張、**status-356 で (i) フル項化 + (ii) adj 拡張を同時導入し FD 機械精度一致**（§7 参照） |

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
- **status-355（診断 + 実装計画）**: `K_c_analytical vs FD` を (active/adj) 4 ブロックに分解、rel_err 1.795% の **100% が active×adj ブロックに局在**することを確認。仮説 B の定量目標 `||diff[ax]|| 98.52 → <1e-3` と実装パス（~45 行）を確立
- **status-356（§7 全面再構成、§3.1/§4/§8 訂正）**: 仮説 A + 仮説 B を同時導入し 2 経路 (i)(ii) を Process 側に実装。`K_hermite_adj` をフル項化 + `K_closest`/`K_st` を active×adj に拡張することで $P_\perp$ 相殺が成立し、`test_helical_3d_hermite` rel_err **1.795% → 2.18e-07**（5 桁改善）、`||diff[ax]||` **98.52 → 4.75e-05**（6 桁改善）。status-354 の「mat-only 最良」解釈は (ii) 未実装時のワークアラウンドであったと訂正
