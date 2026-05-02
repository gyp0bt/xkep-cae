# Cosserat 梁プロトタイプ 設計仕様（status-387 / Phase 0）

[← README](../../../README.md) | [← 設計文書索引](../../../docs/design/README.md) | [← roadmap](../../../docs/roadmap.md)

## 目的

`solver_mode="explicit"` + UL（更新ラグランジアン）+ CR-Timoshenko 梁の
組合せが **status-382/383/385/386** で原理的に成立しないと確定したため、
**幾何学的厳密 (geometrically exact / Simo–Reissner) Cosserat 梁** を新規実装し、
explicit 中央差分 + 大回転 + 大変位を本質的に整合させる。

UL を捨てた帰結:

- 各増分で `update_reference()` を呼ぶ必要がない（reference 配置 = 初期配置）。
- $\boldsymbol{f}_\mathrm{int}(\boldsymbol{u})$ は $\boldsymbol{u}$ 全体（変位 + 回転）から
  直接計算され、線形化レンジ仮定（"u_incr 微小"）に依存しない。
- 大回転は SO(3) 上の指数写像で更新されるため、回転 DOF をベクトル空間で直線的に
  足し合わせる近似（CR-Timoshenko の Battini & Pacoste 局所線形化）に頼らない。

---

## 関連 status とアプローチの位置付け

| 候補 | 結果 | 関連 status |
|------|------|------------|
| (z1a) 要素ごと波速 Δt 推定 | infrastructure 完成 / selective が単梁で機能せず | status-384 |
| (z1b) selective mass scaling | API 完成 / heterogeneous K 必要が判明 | status-384 |
| (z1c) 2 段階 β（β_stiff + β_outside） | API 完成 / β_stiff cap 支配 | status-385 |
| (z1d) `t_cycle` 下限緩和 | 方向自体が逆と単梁実機で実証 | status-386 |
| **(z2) Cosserat 梁プロトタイプ** | **本仕様** — UL 凍結を根本解決 | status-387 〜 |

**MCDD 凍結解除条件 (5)** `|u_explicit - u_anal|/|u_anal| < 0.10` を達成するための
最終本命路線。

---

## 数学的枠組み

### 1. 配置と運動学

3D Cosserat 梁の配置は以下の 2 場で表される:

| 量 | 記号 | 値域 |
|----|------|------|
| 中心軸位置 | $\boldsymbol{r}(s,t)\in\mathbb{R}^3$ | 弧長 $s\in[0,L_0]$、時刻 $t$ |
| 断面方向 | $\boldsymbol{\Lambda}(s,t)\in SO(3)$ | 回転行列 |

DOF 配置（ノード $i$）:

```
q_i = (u_i, θ_i)  ∈ ℝ⁶
```

ここで $\boldsymbol{\theta}_i$ は **回転ベクトル**（軸角度表現の 3 成分）で、
SO(3) 上の更新は $\boldsymbol{\Lambda}_i = \exp_{SO(3)}(\boldsymbol{\theta}_i)\,\boldsymbol{\Lambda}_i^{ref}$ 形式で行う。

### 2. ひずみ測度（material frame）

Simo (1985) / Simo-Vu Quoc (1986) 定義:

$$
\boldsymbol{\Gamma}(s) = \boldsymbol{\Lambda}^T \frac{\partial \boldsymbol{r}}{\partial s} - \boldsymbol{e}_3
$$

$$
\boldsymbol{\Omega}(s) = \mathrm{vee}\!\left(\boldsymbol{\Lambda}^T \frac{\partial \boldsymbol{\Lambda}}{\partial s}\right)
$$

- $\boldsymbol{\Gamma}$: 軸ひずみ $\Gamma_3$ + せん断 $\Gamma_1, \Gamma_2$
- $\boldsymbol{\Omega}$: ねじれ $\Omega_3$ + 曲げ $\Omega_1, \Omega_2$
- $\boldsymbol{e}_3 = (0,0,1)^T$: 初期断面法線（reference）

両方とも **対象配置間の相対量** であり、UL の `update_reference()` に依存しない。

### 3. 構成則（Phase 0 では弾性のみ）

```
n = C_F · Γ      (断面力ベクトル)
m = C_M · Ω      (モーメントベクトル)
```

Phase 0 では対角:

```
C_F = diag(GA_1, GA_2, EA)
C_M = diag(EI_1, EI_2, GJ)
```

撚線ファイバー化（弾塑性 + 摩擦）は Phase 3 以降で `xkep_cae/elements/fiber/` を再利用する。

### 4. 内力の弱形式

$$
\delta W_\mathrm{int} = \int_0^{L_0} \left(\boldsymbol{n}^T \delta \boldsymbol{\Gamma}_{||} + \boldsymbol{m}^T \delta \boldsymbol{\Omega}_{||}\right) ds
$$

成分は $\boldsymbol{\Lambda}$ で物質座標から空間座標へ回転される。

### 5. 質量行列と運動方程式

集中質量近似（explicit 中央差分との相性のため）:

$$
M_{tt} = \rho A,\qquad M_{rr} = \rho \, \mathrm{diag}(I_1, I_2, I_1+I_2)
$$

運動方程式（ノード $i$）:

$$
M_{tt}\,\ddot{\boldsymbol{u}}_i + \boldsymbol{f}^{(\mathrm{int},u)}_i = \boldsymbol{f}^{(\mathrm{ext},u)}_i
$$

$$
M_{rr}\,\ddot{\boldsymbol{w}}_i + \boldsymbol{w}_i \times M_{rr}\boldsymbol{w}_i + \boldsymbol{f}^{(\mathrm{int},\theta)}_i = \boldsymbol{f}^{(\mathrm{ext},\theta)}_i
$$

ここで $\boldsymbol{w}_i$ は body-fixed 角速度。**explicit 更新は SO(3) 上の指数写像** で行う:

```
Λ_{n+1} = exp_so3(Δt · w_{n+1/2}) · Λ_n
```

---

## SO(3) ユーティリティ — Phase 0 スコープ

`xkep_cae/mathematics/so3.py` に集約。

| 関数 | 役割 | 公式 |
|------|------|------|
| `skew(v)` | $\hat{v}$ 歪対称化 | $\hat{v}_{ij} = -\epsilon_{ijk} v_k$ |
| `vee(S)` | 歪対称行列 → ベクトル | inverse of `skew` |
| `exp_so3(theta)` | 指数写像（Rodrigues） | $\exp(\hat{\theta}) = I + \frac{\sin\phi}{\phi}\hat{\theta} + \frac{1-\cos\phi}{\phi^2}\hat{\theta}^2$ |
| `log_so3(R)` | 対数写像 | $\theta = \frac{\phi}{2\sin\phi}\,\mathrm{vee}(R-R^T)$ |
| `dexp_so3(theta)` | 右ヤコビアン $T(\theta)$ | $I + \frac{1-\cos\phi}{\phi^2}\hat{\theta} + \frac{\phi-\sin\phi}{\phi^3}\hat{\theta}^2$ |
| `dexp_inv_so3(theta)` | 右ヤコビアン逆 $T^{-1}(\theta)$ | 級数展開 + 解析閉形式 |

**数値条件**: $\phi \to 0$ でテイラー展開（$\phi < 10^{-8}$ 程度で切替）、$\phi \to \pi$ で
quaternion 経由の log を使用（status-356 と整合する精度）。

**バッチ版**: `_batch_*` プレフィックスで $(N,3)$ 入力を $(N,3,3)$ 出力に拡張。

---

## CR 梁との関係（既存資産の保持）

`xkep_cae/elements/_beam_cr.py` には既に以下の private 関数が存在する:

- `_skew`, `_rotvec_to_rotmat` (= exp_so3), `_rotmat_to_rotvec` (= log_so3)
- `_tangent_operator` (= dexp_so3), `_tangent_operator_inv` (= dexp_inv_so3)
- バッチ版 `_batch_*`

**Phase 0 ではこれらに変更を加えない**（CR 梁は status-356 で `rel_err = 2.18×10⁻⁷` の
機械精度まで仕上がっており、回帰リスクを避ける）。Cosserat 路線が成熟した時点で、
`_beam_cr.py` の private 関数を `mathematics.so3` への薄い委譲に切り替える DRY 化を
検討する（Phase 4 以降）。

---

## Phase 進行計画

| Phase | 内容 | 規模目安 | 関連 status |
|-------|------|---------|-------------|
| **Phase 0** | 設計仕様 + `mathematics/so3.py` + 単体テスト | ~600 行 | **status-387（本 status）** |
| Phase 1 | `CosseratBeamElementProcess` — 1 要素弾性内力 + 解析接線 | ~600 行 | 次 status |
| Phase 2 | アセンブラ + `CosseratExplicitProcess` 配線 | ~500 行 | |
| Phase 3 | 単梁 90° 曲げで解析解 73.30mm に対し精度 gate < 10% 達成 | 検証 | MCDD 凍結解除条件 (5) 達成判定 |
| Phase 4 | 接触結合 + 7 本撚線 frac=1.0 完走 | | |
| Phase 5 | ファイバー化（`xkep_cae/elements/fiber/` 再利用）+ 19 本検証 | | |

---

## MCDD 脱法 pattern 回避方針

| Pattern | 回避策 |
|---------|--------|
| 1: tol 緩和 | 単体テストの精度しきい値（exp/log 往復 1e-12、dexp 整合 1e-10）は数学的根拠あり |
| 4: rename で済ます | `_beam_cr.py` の private 関数群はそのまま。新規 module `so3.py` は **公開 API + 拡張 (composition / hat_inv 同値性等)** を追加 |
| 5: 既存テスト skip | 既存 743 test 全 pass を維持、Phase 0 では追加のみ |
| 6: 骨格 status | Phase 0 = 設計仕様 + 実装 + 単体テスト + 既存資産との整合検証で完結。Phase 1 以降は別 status |
| 7: 数値丸め | `rel_err` は `{:.3e}` 形式で報告 |
| 10: TODO 先送り | Phase 0 を完結させ、Phase 1 を次 status の自然な次ステップに位置付け |

---

## 引継ぎ

- Phase 0 完了で SO(3) 演算が公開モジュールとして利用可能になる。
- Phase 1 は `xkep_cae/elements/cosserat/` サブパッケージを新設し、
  `CosseratBeamElementProcess` を実装する。
- 既存 `_beam_cr.py` の private SO(3) 関数は **意図的に残置**（回帰防止）。
