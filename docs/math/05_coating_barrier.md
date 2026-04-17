# 05 — バリア関数被膜モデル

[← README](../../README.md) | [← roadmap](../roadmap.md) | [← 数理台帳](README.md) | [← MCDD 設計仕様](../../xkep_cae/mathematics/docs/mathematics.md)

> **本台帳の責務**: 被膜層の圧縮モデル（バリア関数 $f = k\delta/(1-\delta/\delta_{\max})$）の
> 正規記述。芯線貫入防止の境界条件 $\delta < \delta_{\max}$ と、被膜エネルギー、
> 接線剛性を網羅する。
>
> **本台帳の非責務**: 線形バネモデル（status-302 で数値的正則化として機能する
> ことが判明）は過去モデルとして残すが、bar function 以降を主力とする。
> 粘性被膜（Kelvin-Voigt）は本章で式のみ触れ、詳細は status-137/140 参照。

## 表記

| 記号 | 意味 |
|---|---|
| $k_{\mathrm{coat}}$ | 被膜剛性係数 |
| $\delta$ | 被膜圧縮量（$\delta = \max(0,\,t_{\mathrm{coat}}-d_{\mathrm{coat}})$） |
| $\delta_{\max}$ | 被膜総厚さ $(r_A - r_{A,\mathrm{core}}) + (r_B - r_{B,\mathrm{core}})$ |
| $r_{A,\mathrm{core}}$ | 芯線（core）半径 |
| $r = \delta/\delta_{\max}$ | 正規化圧縮率（$0\le r < 1$） |
| $p_n^{\mathrm{coat}}$ | 被膜ペナルティ力 |
| $k_{\mathrm{eff}}^{\mathrm{coat}}$ | 被膜接線剛性 $\mathrm{d}p_n^{\mathrm{coat}}/\mathrm{d}\delta$ |
| $E_{\mathrm{coat}}$ | 被膜弾性エネルギー |
| $\epsilon_B$ | バリア特異点回避定数 `_BARRIER_CLAMP = 1e-3` |

→ 実装: `xkep_cae/contact/coating/strategy.py`

---

<a id="eq-barrier-pn"></a>

## 1. バリアペナルティ力

$$
p_n^{\mathrm{coat}}(\delta)
\;=\;
\frac{k_{\mathrm{coat}}\,\delta}{\,1 - \delta/\delta_{\max}\,},
\qquad 0 \le \delta < \delta_{\max}
$$

数値安定化のため分母を $\max(1-r,\,\epsilon_B)$ で clamp する:

$$
p_n^{\mathrm{coat}}
\;=\; \frac{k_{\mathrm{coat}}\,\delta}{\max(1 - r,\,\epsilon_B)}
$$

$\delta\to\delta_{\max}$ で $p_n^{\mathrm{coat}}\to k_{\mathrm{coat}}\delta_{\max}/\epsilon_B$
（clamp 上限）であり、芯線貫入を構造的に防ぐ。

→ 実装: `strategy.py:_barrier_p_n`

---

<a id="eq-barrier-keff"></a>

## 2. バリア接線剛性

$$
k_{\mathrm{eff}}^{\mathrm{coat}}(\delta)
\;=\;
\frac{\mathrm{d}p_n^{\mathrm{coat}}}{\mathrm{d}\delta}
\;=\;
\frac{k_{\mathrm{coat}}}{\,(1 - \delta/\delta_{\max})^2\,}
$$

clamp 経路:

$$
k_{\mathrm{eff}}^{\mathrm{coat}}
\;=\; \frac{k_{\mathrm{coat}}}{\max(1 - r,\,\epsilon_B)^{\,2}}
$$

→ 実装: `strategy.py:_barrier_k_eff`

---

<a id="eq-barrier-energy"></a>

## 3. 被膜弾性エネルギー

解析積分:

$$
E_{\mathrm{coat}}(\delta)
\;=\; \int_0^{\delta} p_n^{\mathrm{coat}}(x)\,\mathrm{d}x
\;=\; k_{\mathrm{coat}}\,\delta_{\max}^{2}\,\Big[-\ln(1-r)-r\Big]
$$

$\delta\to\delta_{\max}$ で $E_{\mathrm{coat}}\to\infty$（芯線貫入はエネルギー
無限のペナルティ）。clamp 経路では $r_{\mathrm{clamped}}=\min(r,1-\epsilon_B)$
で有限値に収束する。

$\delta_{\max}\to 0$ で線形フォールバック $E = \tfrac{1}{2}k_{\mathrm{coat}}\delta^2$。

→ 実装: `strategy.py:_barrier_energy`

---

<a id="inv-barrier-bounded"></a>

## 4. 不変量（芯線貫入防止）

$$
\delta < \delta_{\max}\quad\Longleftrightarrow\quad p_n^{\mathrm{coat}} < \infty
$$

解析上の不等式。数値実装では $\epsilon_B$-clamp により $p_n^{\mathrm{coat}}$ は
有限値に制限されるが、$r\to 1$ で十分大きなペナルティが働き、NR 反復内で
$r$ が減少方向に収束する。

→ 契約: `InequalityContract(name="coating_no_core_penetration", expr="delta",
kind="lt", bound="delta_max", equation_ref="05_coating_barrier.md#inv-barrier-bounded",
severity="soft")`

---

<a id="eq-barrier-linear-fallback"></a>

## 5. 線形フォールバックと Kelvin-Voigt 粘性（参照）

**線形モデル**（status-302 で数値的正則化と判定）:

$$
p_n^{\mathrm{coat,lin}} \;=\; k_{\mathrm{coat}}\,\delta
$$

`delta_max <= 1e-30` のとき `_barrier_p_n` が自動的に本式にフォールバックする。

**Kelvin-Voigt 粘性**（status-137/140）:

$$
p_n^{\mathrm{coat,KV}}
\;=\; k_{\mathrm{coat}}\,\delta + c_{\mathrm{coat}}\,\dot{\delta}
$$

`KelvinVoigtCoatingProcess`（`strategy.py:133`）が担当。接触力全体が速度に
依存するため動的解析でのみ有効。

---

## 6. 既存実装との trace

| 数式 | 実装位置 | 備考 |
|---|---|---|
| [#eq-barrier-pn](#eq-barrier-pn) | `strategy.py:_barrier_p_n` | `_BARRIER_CLAMP=1e-3` |
| [#eq-barrier-keff](#eq-barrier-keff) | `strategy.py:_barrier_k_eff` | 接線剛性 |
| [#eq-barrier-energy](#eq-barrier-energy) | `strategy.py:_barrier_energy` | $-\ln(1-r)-r$ |
| [#inv-barrier-bounded](#inv-barrier-bounded) | — | 契約のみ、Phase E で C18 検査 |
| [#eq-barrier-linear-fallback](#eq-barrier-linear-fallback) | `strategy.py:_barrier_p_n`（`delta_max<=1e-30` 分岐）| 退化時線形 |

---

## 関連 status

- status-137/140: Kelvin-Voigt 粘性被膜モデル実装
- status-301〜302: 被膜貫入量診断（平均 54% 圧縮、8.6% 芯線貫入）
- status-303: バリア関数被膜モデル新設（芯線貫入防止）
- status-304: 被膜接線剛性 FD 精度検証
- status-305: バリア被膜 90 度曲げ収束性（incr 42% 削減）
- status-306: 被膜エネルギー比診断
