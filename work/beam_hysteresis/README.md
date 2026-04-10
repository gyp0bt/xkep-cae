# work/beam_hysteresis — 撚線ファイバー梁モデル 概念検証

[← README](../../README.md) | [← 設計仕様](../../xkep_cae/elements/docs/fiber_beam_strand.md)

撚線の曲げヒステリシスを **1本のファイバー梁要素** で再現するための
数値実験スクリプト群。本ディレクトリは本番コードではなく
**設計・キャリブレーションの裏付け資料**である。

正式な設計は
[`xkep_cae/elements/docs/fiber_beam_strand.md`](../../xkep_cae/elements/docs/fiber_beam_strand.md)
を参照のこと。

---

## 結論（要点）

1. **移動硬化則 ≡ 撚線摩擦**: 1D kinematic hardening と 1D 撚線摩擦モデルは
   数学的に同型（`01_kh_vs_friction_equivalence.py`）。
2. **傾き非対称には接触剛性劣化が必要**: 純粋 KH は平行四辺形で対称になる。
   スリップ後の剛性低下 $\beta = 0.25$ でリアル U/L = 0.77 を再現
   （`02_slope_asymmetry_degradation.py`, `03_multilayer_degradation.py`）。
3. **滑らかなティアドロップ**: 対数分布した $N = 150$ 摩擦層＋
   外層ソフト・内層スティフ勾配＋繊維断面で角のない丸い履歴を得る
   （`05_smooth_teardrop.py`）。
4. **ジグ摩擦 vs 内部摩擦の分離**: 三点曲げのローラー摩擦は
   乗算的（形状不変の幅拡大）、素線摩擦は加算的（ティアドロップ）
   （`06_jig_friction.py`）。
5. **サイクル動作**: 劣化ありモデルは Cycle 2 以降でシェイクダウンし
   U/L < 1 を保つ（`07_cyclic_hysteresis.png`）。
6. **陽接触モデルとの整合**: 7本撚線の `StrandBendingOscillationProcess`
   実解析で散逸ループを確認（`08_strand_hysteresis.png`,
   `run_strand_hysteresis.py`）。これがファイバー梁キャリブレーション目標。

---

## スクリプト一覧

| ファイル | 内容 | 主な図 |
|---------|------|-------|
| `01_kh_vs_friction_equivalence.py` | KH ≡ 摩擦 同型性の証明 | `verify_kh_asymmetry.png` |
| `02_slope_asymmetry_degradation.py` | 接触剛性劣化で傾き非対称を生む | `02_slope_asymmetry_degradation.png` |
| `03_multilayer_degradation.py` | 多層摩擦 + 劣化の実装 | `03_multilayer_degradation.png` |
| `04_fat_teardrop.py` | 30 層で太いティアドロップ | `04_fat_teardrop.png` |
| `05_smooth_teardrop.py` | **メインモデル**: $N=150$ 滑らかティアドロップ | `05_smooth_teardrop.png` |
| `06_jig_friction.py` | 三点曲げローラー摩擦の影響 | `06_jig_friction.png`, `06_jig_friction_span_compare.png` |
| `beam_bending_hysteresis.py` | 30 層ベースライン比較 | `hysteresis_comparison.png` |
| `run_strand_hysteresis.py` | 7本撚線陽接触ベンチマーク | `07_cyclic_hysteresis.png`, `08_strand_hysteresis.png` |

---

## キー方程式

### 1D 撚線摩擦モデル（`01_*`）

$$\sigma = E_{\text{base}} \varepsilon + \sum_{i=1}^{N} f_i(\varepsilon)$$

各層 $f_i$ は

$$f_i = k_i (\varepsilon - u_i^{\text{slip}}), \qquad |f_i - f_i^{\text{locked}}| \leq f_{y,i}$$

閾値超過で $u_i^{\text{slip}}$ と $f_i^{\text{locked}}$ を return mapping で更新。

### 接触剛性劣化（`02_*`, `05_*`）

$$k_i = \begin{cases} k_i^{\text{virgin}} & \text{（初回スリップ前）} \\ \beta k_i^{\text{virgin}} & \text{（それ以降）}\end{cases}$$

$\beta \in [0.1, 0.3]$ が実ケーブル計測値に整合（status-121 NCP ヒステリシスと同等）。

### ジグ摩擦補正（`06_*`）

三点曲げの荷重–モーメント関係を

$$P_{\text{load}} = \frac{4 M}{L - 2\mu(r_s + r_l)}, \qquad
P_{\text{unload}} = \frac{4 M}{L + 2\mu(r_s + r_l)}$$

に分ける。内部摩擦が**加算**、ジグ摩擦が**乗算**で作用するため
実験値からの両寄与の分離同定が可能。

---

## キャリブレーション対象パラメータ

`05_smooth_teardrop.py` の数値:

| 記号 | 意味 | デフォルト | 自由度 |
|------|------|----------|--------|
| $E_{\text{base}}$ | 素線個体曲げ剛性 | 1500 MPa | 同定対象 |
| $k_{\text{contact,total}}$ | 摩擦層合計剛性 | 7500 MPa | 同定対象 |
| $N$ | 摩擦層数 | 150 | 固定（プリセット） |
| $k_i$ 重み | $\text{linspace}(0.5, 1.5)$ | 固定 | 固定 |
| $\varepsilon_{y,i}$ | $\log(0.002) \to \log(0.30)$ | 固定 | 固定 |
| $\beta$ | 劣化比 | 0.25 | 同定対象 |

自由度 3 に制約することで、7本撚線陽接触モデルの $\theta$–$M$ ループに
対して過パラメータを回避する（設計仕様「既知のリスク 3」参照）。

---

## 実行方法

全スクリプトは単独で実行可能。出力は同ディレクトリの PNG。

```bash
cd work/beam_hysteresis
python 05_smooth_teardrop.py
python 06_jig_friction.py
python run_strand_hysteresis.py    # 本番ソルバーを呼ぶ（時間かかる）
```

> これらは本番テスト（`tests/`）には含めない。
> 新機能の収束検証は正式な `tests/` + `TestFiberSection*` クラスで行う
> （CLAUDE.md「新機能の収束検証フロー」参照）。
