# status-118: Cosserat Rodが必要/有効なケースの調査記録

[← README](../../README.md) | [← status-index](status-index.md) | [← status-117](status-117.md)

日付: 2026-03-06

## 概要

CR-Timo（corotational Timoshenko）とCosserat Rodの使い分け指針を文献調査+コード解析で整理。
「いつCosseratが必要になるか」「いつCRでは不十分か」を具体的シナリオとともに記録する。

## 1. CR定式化の弱点（具体的な破綻条件）

### 1.1 要素あたり変形回転の限界

CR定式化は **corotatedフレーム内で線形理論を使う** ため、要素あたりの **変形回転**（剛体回転ではなく歪みを生む回転）が小さい前提で成立する。

- **閾値**: 文献上 ~10-15°/要素が実用限界（Hsiao et al., 1987; Crisfield, 1997）
- **剛体回転は制限なし**: CR更新で任意大回転を扱えるが、これは「要素が曲がらずに回る」場合のみ
- **対処法**: メッシュ細分化で要素あたり回転を小さく保つ → 計算コスト増

> **xkep-cae 現状**: `beam_timo3d.py` L988-991 にこの制約が明記されている。
> 撚線では16要素/ピッチ以上を使用しており、1要素あたり ~22.5° の **剛体回転**。
> 曲げ揺動の変形回転は通常これより遥かに小さいので、現状のメッシュ密度では問題ない。

### 1.2 客観性（objectivity）の欠如

**CRは客観性を自然に満たすが、増分回転ベクトル補間には注意が必要。**

- Crisfield & Jelenić (1999, Proc. R. Soc. Lond. A): 回転の空間補間が客観性（剛体回転に対する不変性）を満たさないケースがある
- Magisano et al. (2020, CMAME): 増分回転ベクトルの補間は **非客観的かつ経路依存** → メッシュ細分化で回復するが、粗いメッシュでは問題
- **繰り返し荷重での誤差蓄積**: 「最初のサイクルには十分なメッシュが、後続サイクルでは不十分になる」現象が報告

> **xkep-cae への影響**: 曲げ揺動は繰り返し荷重そのもの。CR+粗メッシュで長サイクル計算すると
> 誤差蓄積のリスクがある。ただし現状のメッシュ密度（16要素/ピッチ）では十分細かい可能性が高い。

### 1.3 Wagner効果の近似

- `beam_timo3d.py` L838-866: 捻り-曲げ連成（Wagner効果）が **近似形** で実装されている
- 完全なWagner連成: `Mx * (δθ₂·θ₃' - δθ₃·θ₂')` の全非線形項
- 薄肉開断面（I形鋼等）では重要だが、**円形断面ワイヤでは影響は限定的**

### 1.4 接線剛性の対称性

- CR定式化: 保存力の場合でも接線剛性が非対称になりうる（回転パラメータの非可換性に起因）
- Cosserat: 四元数表現で同様の問題があるが、歪み-構成関係が直截なので連成項が見えやすい

## 2. Cosseratが必要になる具体的シナリオ

### シナリオA: 超大変形（要素あたり変形回転 > 15°）

| 条件 | 具体例 | xkep-cae関連度 |
|------|--------|----------------|
| 結び目形成 | ケーブルの結び目引き締め | ★☆☆ 低 |
| 座屈後挙動 | 細長い棒の座屈後大変形 | ★☆☆ 低 |
| コイルばね圧縮 | ばね素線の大曲率変化 | ★☆☆ 低 |

> 撚線の曲げ揺動ではこのレベルの変形は発生しない。

### シナリオB: 長時間繰り返し荷重での精度保証 ← **ターゲット範囲内**

> **ユーザー確認（2026-03-06）**: サイクル曲げでのヒステリシス、速度依存、疲労破断は
> すべてターゲット範囲内。このシナリオはCosserat移行の最も強い動機。

| 条件 | 具体例 | xkep-cae関連度 |
|------|--------|----------------|
| 疲労寿命予測 | 10⁶サイクル以上の応力精度 | ★★★ 高 |
| ヒステリシス定量評価 | 摩擦散逸エネルギーの精密計算 | ★★★ 高 |
| 速度依存性 | 載荷速度による応答変化（粘弾性・慣性効果） | ★★★ 高 |
| 疲労破断 | ワイヤ断線進展のシミュレーション | ★★★ 高 |
| フレッティング摩耗 | 接触面での微小往復すべり | ★★★ 高 |

> **これがCosserat移行の最も強い動機。** 曲げ揺動の繰り返しで:
>
> - **ヒステリシスループ面積**（散逸エネルギー）の定量評価にはサイクル間の誤差蓄積が致命的
> - **疲労寿命予測**には局所応力の高精度が必要 → CRの近似的歪み測度では不十分な可能性
> - **速度依存性**の正確な評価には時間積分の客観性が必要 → 経路依存の誤差蓄積が結果を汚す
> - **疲労破断**（ワイヤ断線）シミュレーションではトポロジー変化後の再平衡計算が多数発生
>   → 各ステップの精度が蓄積するため、定式化の客観性が重要
>
> **定量的にどの程度影響するかは実際に比較しないとわからない**が、
> ターゲット要件として明確であるため、**Phase 1比較テストの優先度を引き上げる**。

### シナリオC: ヘリカル形状の要素数削減（1000本目標との関連）

| 条件 | 具体例 | xkep-cae関連度 |
|------|--------|----------------|
| ヘリカル形状の厳密表現 | IGA+Cosserat: 1セグメントで一定曲率ヘリックス表現可能 | ★★★ 高 |
| メッシュ粗化による高速化 | CR: 16要素/ピッチ → Cosserat/IGA: 4-8要素/ピッチで同等精度の可能性 | ★★★ 高 |
| 1000本計算のDOF削減 | 要素数半減 → DOF半減 → 計算時間大幅短縮 | ★★★ 高 |

> **1000本6時間目標に対して最も実用的な動機**。CRでは各要素の局所回転を小さく保つため
> ピッチあたり多数の要素が必要。Cosserat/IGAではヘリカル形状関数を用いて少ない要素で
> 正確に表現できる（Zhang 2020, Weeger et al. 2019）。
> ただし、少ない要素数で接触精度が維持できるかは別途検証が必要。

### シナリオD: Cosserat rodとしての直接的な接触定式化

| 条件 | 具体例 | xkep-cae関連度 |
|------|--------|----------------|
| IGA + Cosserat接触 | Weeger et al. (2022) | ★★☆ 中 |
| 弾塑性Cosserat rod | Smriti et al. (2021) | ★☆☆ 低 |
| 均質化RVE | Stainier et al. (2021) | ★★☆ 中 |

> Cosserat rodの断面回転を接触法線の定義に直接使うことで、
> 接触フレームの連続性が自然に保証される。CRでは追加の並行移送が必要。

### シナリオE: 数学的/ソフトウェア工学的利点

| 利点 | 説明 | 影響 |
|------|------|------|
| 特異点なし | 四元数でギンバルロック回避 | 現状CR+回転ベクトルで問題なし |
| 歪み量の直截性 | Γ, κ が構成則と直結 | コード可読性向上 |
| 接線剛性の解析的導出 | B行列が構造的にクリーン | NR収束改善の鍵（現状は数値微分） |

## 3. CRベースの先行成功事例

**Foti et al.** は撚線の曲げ振動にCR梁要素を適用し成功している:
- マクロスケールで断面構成則（モーメント-曲率のヒステリシス則、ワイヤ間のstick-slip遷移）をCR梁に埋め込む手法
- 単層撚線のグローバル応答（曲げ剛性・ヒステリシスループ）を実験と良好に一致

> **含意**: CRベースでも撚線ヒステリシスの定量評価は可能。
> ただしFotiモデルはマクロ断面構成則（個別ワイヤの接触を均質化済み）であり、
> xkep-caeのようにワイヤ個別にモデル化+接触解析するアプローチとは異なる。

## 4. 結論: いつCosseratに切り替えるべきか

### CRで当面十分な場合

- 初期のS3-S4段階（収束改善・剛性比較）で定性的挙動の確認が主目的
- メッシュが十分細かい（16要素/ピッチ以上）
- 計算時間が支配的制約でヒステリシス精度は二次的

### Cosserat移行が必要になるタイミング

ヒステリシス・速度依存・疲労破断がターゲット範囲内である以上、
**Cosserat移行は「するかどうか」ではなく「いつするか」の問題**。

移行の前提条件（ブロッカー解消順）:
1. **解析的接線剛性の実装** ← 最重要ブロッカー（NR収束ストール解消）
2. **COO/CSR疎行列アセンブリ** ← 大規模計算の前提
3. **質量行列の実装** ← 動的解析の前提
4. **Phase 1比較テストで定量的影響を確認** ← 移行の緊急度判断

### 推奨アクション（段階的検証）— 優先度引き上げ

```
Phase 1（S4並行、優先度高）: CR vs Cosserat 単線比較
  → 同一問題で曲げ揺動100サイクル、先端変位・エネルギー履歴を比較
  → ヒステリシスループ面積の差を定量評価
  → 差が有意（>5%）なら移行タイムラインを前倒し

Phase 2（解析的接線実装後、S5並行）: 接触付き7本比較
  → Cosserat+NCPで7本撚り曲げ揺動
  → CR版との収束速度・精度・ヒステリシス比較
  → 疲労寿命予測に必要な応力精度の評価

Phase 3（Phase 2で差が確認された場合）: 接触パイプライン統合
  → DOFマッピング改修（6→7 DOF/node対応）
  → broadphase/geometry/NCPの四元数対応
  → 疲労破断（要素除去）機能のCosserat対応
```

## 5. 文献リスト

### CR定式化の限界に関する文献

- Hsiao, K.M. et al. (1987) "A corotational procedure that handles large rotations of spatial beam structures", *Comput. Struct.* — CR要素あたり回転制限の原論文
- Crisfield, M.A. & Jelenić, G. (1999) "Objectivity of strain measures in geometrically exact 3D beam theory and its finite-element implementation", *Proc. R. Soc. Lond. A* — 客観性の本質的問題を指摘
- Magisano, D. et al. (2020) "A large rotation finite element analysis of 3D beams by incremental rotation vector and exact strain measure", *CMAME* — 増分回転補間の非客観性・経路依存性を解析、p-refinementで回復
- Hsiao, K.M. (1994) "Effects of coordinate system on the accuracy of corotational formulation for Bernoulli-Euler's beam", *Int. J. Solids Struct.* — 座標系選択と精度の関係

### Cosserat rod + 接触/撚線に関する文献

- Weeger, O. et al. (2022) "An isogeometric finite element formulation for frictionless contact of Cosserat rods with unconstrained directors", *Comput. Mech.* — IGA + Cosserat rod接触の最新定式化
- Stainier, L. et al. (2021) "Solid and 3D beam finite element models for the nonlinear elastic analysis of helical strands within a computational homogenization framework", *Comput. Struct.* — 撚線RVE均質化にbeam FE+接触使用
- Smriti et al. (2021) "A finite element formulation for a direct approach to elastoplasticity in special Cosserat rods", *IJNME* — Cosserat rodの弾塑性直接定式化
- Bosten, A. et al. (2023) "A beam contact benchmark with analytic solution", *ZAMM* — 梁接触の解析解ベンチマーク（Mortar法比較）
- Lang, H. et al. (2011) "Multi-body dynamics simulation of geometrically exact Cosserat rods", *Multibody Syst. Dyn.* — Cosserat rodの動的シミュレーション

### ワイヤロープモデリング

- Costello, G.A. "Theory of Wire Rope" — 撚線理論の基礎
- Foti, F. & Martinelli, L. (2016) "Hysteretic bending of spiral strands" — 撚線ヒステリシスの実験+理論
- Foti, F. et al. "A corotational finite element to model bending vibrations of metallic strands" — CRベースの撚線曲げ振動（マクロ断面構成則）
- Zhang, J. (2020) "High-Efficiency Dynamic Modeling of Helical Spring Based on GEB Theory", *Shock Vib.* — GEB（幾何学的厳密梁）によるヘリカル構造の高効率モデリング

### CR定式化の精度に関する追加文献

- Iura, M. et al. (2003) "Accuracy of corotational formulation for 3D Timoshenko beam", *Comput. Mech.* — CR精度の系統的評価
- Wang, T. et al. (2025) "Efficient 3D corotational beam for nonlinear dynamics", *Comput. Struct.* — CR動的解析の最新高精度定式化
- Meier, C. et al. (2017) "Kirchhoff-Love vs Simo-Reissner theory", *Arch. Comput. Methods Eng.* — 梁理論の体系的比較レビュー（MIT）

## 確認事項

- **Cosserat移行は「するかどうか」ではなく「いつするか」**: ヒステリシス・速度依存・疲労破断がターゲット範囲であることが確認された。長期的にはCosserat移行が必要になる可能性が高い
- **「試さないとわからない」は依然として正しい**: CRで精度不足になるかの定量的判断にはPhase 1比較テストが必要。CRでも16要素/ピッチ以上なら客観性は回復する（Magisano 2020）
- **最大のブロッカーは解析的接線**: Cosseratの数値微分接線（eps=1e-7）はNR残差~1e-7にストールする。これが解消されないとCosserat移行は実用的に困難
- **短期戦略は変更なし**: S3-S4はCR-Timoで進める。Phase 1比較テストをS4並行で実施し、移行タイムラインを判断
- **Fotiモデルとの違い**: Foti et al.はCRベースで撚線ヒステリシスに成功しているが、マクロ断面構成則（均質化済み）。xkep-caeのワイヤ個別モデル+接触解析では、CRの経路依存誤差がより顕在化する可能性がある

---
