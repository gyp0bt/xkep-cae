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

$\boldsymbol{f}_c = -p_n\hat{\boldsymbol{n}}$ を全体変位 $\boldsymbol{u}$ で微分:

$$
\boldsymbol{K}_c
\;=\;\frac{\partial \boldsymbol{f}_c}{\partial \boldsymbol{u}}
\;=\;
\underbrace{-\frac{\partial p_n}{\partial \boldsymbol{u}}\otimes\hat{\boldsymbol{n}}}_{\boldsymbol{K}_{\mathrm{mat,nn}}}
\;\;\underbrace{-\;p_n\,\frac{\partial \hat{\boldsymbol{n}}}{\partial \boldsymbol{u}}}_{\boldsymbol{K}_{\mathrm{mat,ndir}}\;(\text{未実装、status-352 本命})}
\;\;\underbrace{+\;\boldsymbol{K}_{\mathrm{closest}}}_{(s,t)\,\text{追従}}
\;\;\underbrace{+\;\boldsymbol{K}_{\mathrm{hermite,adj}}}_{\text{隣接ノード}}
\;\;\underbrace{-\;\boldsymbol{K}_{\mathrm{geo}}}_{\text{幾何補正}}
\;\;\underbrace{+\;\boldsymbol{K}_{\mathrm{st}}}_{(s,t)\,\partial s/\partial \boldsymbol{u}}
$$

ペアの組み立て前行列形（$3\times 3$ ブロック、Hermite 形状係数 $c_i,c_j$ 込み）:

<a id="eq-kc-pair-block"></a>

$$
\boldsymbol{K}_c^{(ij)} \;=\;
c_i\,c_j\Big[\,
\underbrace{w_{\mathrm{mat}}\,(\hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top)}_{\text{法線剛性}}
\;-\;
\underbrace{w_{\mathrm{geo}}\,\boldsymbol{P}_\perp}_{\text{幾何補正}}
\,\Big]
\;+\; \boldsymbol{K}_{\mathrm{st}}^{(ij)}
$$

ここで:

$$
w_{\mathrm{mat}} = \frac{\mathrm{d}p_n}{\mathrm{d}x}\cdot k_{\mathrm{pen}}
\quad\text{（[#eq-dpn-dx](#eq-dpn-dx) 参照）},
\qquad
w_{\mathrm{geo}} = \frac{p_n}{d}
$$

→ 実装: `HuberContactForceProcess.assemble_tangent`（`strategy.py:960` 付近）— 現行の式網羅は **5 項 + $\boldsymbol{K}_{\mathrm{mat,ndir}}$ 欠落**。

### 3.1 項一覧（`TermExpansionContract.term_names` と一対一対応）

| `term_name` | 数式 | 実装 Process（Phase C で抽出予定） | 状態 |
|---|---|---|---|
| `K_mat_nn` | $-\,\frac{\partial p_n}{\partial \boldsymbol{u}}\otimes\hat{\boldsymbol{n}}$ | `KcNormalStiffnessProcess` | ✅ status-350 で抽出（K_hermite_adj も暫定包含、status-353 で分離予定） |
| `K_mat_ndir` | $-\,p_n\,\frac{\partial \hat{\boldsymbol{n}}}{\partial \boldsymbol{u}}$ | **`KcNormalDirectionStiffnessProcess`** | **未実装。status-352 本命修正**（[#sec-ndir](#sec-ndir)） |
| `K_closest` | $-\,p_n\,\hat{\boldsymbol{n}}\otimes\partial s/\partial \boldsymbol{u}$ 等 | `KcClosestPointStiffnessProcess` | status-351 で分離 |
| `K_hermite_adj` | 隣接ノード $\partial \boldsymbol{p}_A/\partial \boldsymbol{u}_{\mathrm{adj}}$ | `KcHermiteNonlocalStiffnessProcess` | status-353 で抽出（status-271〜274 で C 実装済、status-350 時点では `KcNormalStiffnessProcess` に暫定包含） |
| `K_geo` | $-\,(p_n/d)\,\boldsymbol{P}_\perp \cdot c_i c_j$ | `KcGeoStiffnessProcess` | ✅ status-350 で抽出 |
| `K_st` | $\partial \boldsymbol{f}_{\mathrm{raw}}/\partial s\;\otimes\;\partial s/\partial \boldsymbol{u}$ ほか | `ContactForceStStiffnessProcess`（status-351 で `KcStStiffnessProcess` へ rename 予定） | ✅ status-350 で `_K_C_TERM_EXPANSION_CONTRACT` 宣言済 |

→ 契約: `TermExpansionContract(name="K_c_term_expansion", total_name="K_c", term_names=("K_mat_nn","K_mat_ndir","K_closest","K_hermite_adj","K_geo","K_st"), providers=(...), combinator="add_sub", equation_ref="03_huber_contact_penalty.md#eq-kc-full-decomposition")`

`combinator="add_sub"` は $\boldsymbol{K}_c = (\boldsymbol{K}_{\mathrm{mat,nn}}+\boldsymbol{K}_{\mathrm{mat,ndir}}+\boldsymbol{K}_{\mathrm{closest}}+\boldsymbol{K}_{\mathrm{hermite,adj}}) - \boldsymbol{K}_{\mathrm{geo}} + \boldsymbol{K}_{\mathrm{st}}$ の符号付き加算を表す。

---

<a id="sec-ndir"></a>

## 4. 法線方向感度 $\partial \hat{\boldsymbol{n}}/\partial \boldsymbol{u}$（K_mat,ndir、status-344 本命）

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

これを [#eq-kc-full-decomposition](#eq-kc-full-decomposition) の第 2 項に
代入することで、status-344 で観測された **$\boldsymbol{K}_{\mathrm{mat}}$ の x/z 成分カップリング欠落**
（`mat_only` rel_err mean=44%, comp_x max=98%）が解消される見込み。

注意: $\boldsymbol{P}_\perp$ 自体は [#eq-kc-pair-block](#eq-kc-pair-block) の
**幾何項** $\boldsymbol{K}_{\mathrm{geo}}$ にも現れるが、両者は意味も符号も異なる:

| 項 | 出所 | 符号 | 重み |
|---|---|---|---|
| $\boldsymbol{K}_{\mathrm{geo}}$ | $\partial(c_i c_j \boldsymbol{p}_A)/\partial \boldsymbol{u}$ の幾何補正 | $-$ | $p_n/d$ |
| $\boldsymbol{K}_{\mathrm{mat,ndir}}$ | $\partial \hat{\boldsymbol{n}}/\partial \boldsymbol{u}$ そのもの | $-$ | $p_n$（$d$ で割らない $\boldsymbol{P}_\perp$ 単独） |

両者を混同して「$\boldsymbol{K}_{\mathrm{geo}}$ で代用済み」と誤読する経路が
status-289〜344 を浪費させた根本要因。Phase C で **異なる Process に分離**
することで構造的に再発を防ぐ（[#eq-kc-full-decomposition](#eq-kc-full-decomposition)）。

→ 実装: **未配置**。status-352（Phase C-3）で `KcNormalDirectionStiffnessProcess` を新設予定。

---

<a id="eq-kmat"></a>

## 5. $\boldsymbol{K}_{\mathrm{mat}}$ の対称性

完全項を持つ場合（$\boldsymbol{K}_{\mathrm{mat,nn}}+\boldsymbol{K}_{\mathrm{mat,ndir}}$）、
法線方向の合成として:

$$
\boldsymbol{K}_{\mathrm{mat}}\,\partial \boldsymbol{u}
\;=\;
-\,\partial(p_n\hat{\boldsymbol{n}})/\partial \boldsymbol{u}\cdot \partial \boldsymbol{u}
$$

このペア局所の $3\times 3$ 表現は対称（共役勾配系の前提）:

<a id="sym-kmat"></a>

$$
\boldsymbol{K}_{\mathrm{mat}}^{(ij)} = \big(\boldsymbol{K}_{\mathrm{mat}}^{(ji)}\big)^\top
$$

→ 契約: `SymmetryContract(name="K_mat_symmetric", matrix_name="K_mat", kind="symmetric", equation_ref="03_huber_contact_penalty.md#sym-kmat")`

注意: 現行実装の $\boldsymbol{K}_c$ は $\boldsymbol{K}_{\mathrm{st}}$（摩擦・滑り
追従）を含むため**全体としては非対称**。本契約は項分解後の $\boldsymbol{K}_{\mathrm{mat}}$ にのみ適用する。

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

→ 契約: status-353 で `KcHermiteNonlocalStiffnessProcess` を抽出後、
`TermExpansionContract.term_names="K_hermite_adj"` で網羅性を要求する。

---

## 8. 既存実装との trace（status-348 時点）

| 数式 | 実装位置 | 備考 |
|---|---|---|
| [#eq-pn-huber](#eq-pn-huber) | `strategy.py:_huber` / `_huber_batch` | C¹ 連続 |
| [#eq-pn-hertz](#eq-pn-hertz) | `strategy.py:_apply_power_law` | $\alpha=1.5$ |
| [#eq-dpn-dx](#eq-dpn-dx) | `strategy.py:_apply_power_law_deriv` | tangent 用 |
| [#eq-fc](#eq-fc) | `strategy.py:HuberContactForceProcess.evaluate` | バッチ経路 |
| [#eq-kc-full-decomposition](#eq-kc-full-decomposition) | `strategy.py:assemble_tangent` | **K_mat,ndir 欠落** |
| [#eq-kc-pair-block](#eq-kc-pair-block) | `strategy.py:tangent_components`（旧）| Phase C で項別 Process に分解 |
| [#eq-dn-du](#eq-dn-du) | **未実装** | status-352（Phase C-3）|
| [#sym-kmat](#sym-kmat) | — | 静的契約のみ、Phase E で C18 検査 |
| [#eq-kc-fd](#eq-kc-fd) | `xkep_cae/verify/kc_component_fd.py` | `@verified_by` 紐付け予定 |
| [#eq-hermite-pA](#eq-hermite-pA) | `_st_jacobian.py` | status-271〜274 拡張 |

---

## 関連 status

- status-222: NCP 除去・純粋ペナルティ確立
- status-259〜261: Huber `smoothing_delta` パイプライン貫通 + `huber_delta_h` 直接指定
- status-285: Hertz 型非線形ペナルティ ($\alpha=1.5$)
- status-271〜274: Hermite 非局所 $\partial g/\partial \boldsymbol{u}$ 隣接ノード拡張
- status-289〜296: K_c FD 不整合追跡（s_unclamped、StJacobian、frozen-m、K_c_adj mat-only）
- status-342〜345: 19 本撚線 K_c 成分分解 FD 診断 + report 精度バグ訂正
- status-346〜347: MCDD Phase A（`MathematicalContract` + `ProcessContractRegistry`）
- **status-348（本台帳）**: Phase B-1 — 03 章先行整備
- status-349（予定）: Phase B-2 — 他 5 章 + `equation_index.py` + C15 拡張
- status-350〜353: Phase C — 項別 Process 抽出 + `KcNormalDirectionStiffnessProcess` 新設（本命修正）
