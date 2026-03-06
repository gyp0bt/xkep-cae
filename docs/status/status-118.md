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

### シナリオB: 長時間繰り返し荷重での精度保証

| 条件 | 具体例 | xkep-cae関連度 |
|------|--------|----------------|
| 疲労寿命予測 | 10⁶サイクル以上の応力精度 | ★★★ 高 |
| ヒステリシス定量評価 | 摩擦散逸エネルギーの精密計算 | ★★☆ 中 |
| フレッティング摩耗 | 接触面での微小往復すべり | ★★☆ 中 |

> **これが最もCosserat移行の動機になりうる。** 曲げ揺動の繰り返しで
> 内部エネルギー散逸（ヒステリシスループ面積）を定量的に評価する場合、
> CR定式化の誤差蓄積が結果を汚す可能性がある。
> ただし、定量的にどの程度汚れるかは **実際に比較しないとわからない**。

### シナリオC: Cosserat rodとしての直接的な接触定式化

| 条件 | 具体例 | xkep-cae関連度 |
|------|--------|----------------|
| IGA + Cosserat接触 | Weeger et al. (2022) | ★★☆ 中 |
| 弾塑性Cosserat rod | Smriti et al. (2021) | ★☆☆ 低 |
| 均質化RVE | Stainier et al. (2021) | ★★☆ 中 |

> Cosserat rodの断面回転を接触法線の定義に直接使うことで、
> 接触フレームの連続性が自然に保証される。CRでは追加の並行移送が必要。

### シナリオD: 数学的/ソフトウェア工学的利点

| 利点 | 説明 | 影響 |
|------|------|------|
| 特異点なし | 四元数でギンバルロック回避 | 現状CR+回転ベクトルで問題なし |
| 歪み量の直截性 | Γ, κ が構成則と直結 | コード可読性向上 |
| 接線剛性の解析的導出 | B行列が構造的にクリーン | NR収束改善の鍵（現状は数値微分） |

## 3. 結論: いつCosseratに切り替えるべきか

### 切り替え不要（CRで十分）な場合

- 曲げ揺動のサイクル数が少ない（~100サイクル以下）
- ヒステリシスの定性的挙動のみが必要（面積の高精度定量は不要）
- メッシュが十分細かい（16要素/ピッチ以上）
- 計算時間が支配的制約（1000本6時間目標）

### 切り替えを検討すべきトリガー

1. **ヒステリシスループの定量評価が必要になった場合**: CRとCosseratで散逸エネルギーを比較し、有意差があれば移行
2. **メッシュ粗化が必要になった場合**: 1000本計算で計算コスト削減のためメッシュを粗くすると、CRの精度劣化が顕在化
3. **解析的接線剛性が実装された場合**: Cosseratの最大弱点（数値微分接線→NR収束ストール）が解消されたら、切り替えコストが大幅減
4. **接触法線/フレームの不連続が問題になった場合**: 並行移送で対処しきれない接触フレームジャンプが発生したら

### 推奨アクション（段階的検証）

```
Phase 1（今すぐ可能）: CR vs Cosserat 単線比較
  → 同一問題で曲げ揺動100サイクル、先端変位・エネルギー履歴を比較
  → 差が5%未満ならCR継続で十分

Phase 2（解析的接線実装後）: 接触付き7本比較
  → Cosserat+NCPで7本撚り曲げ揺動
  → CR版との収束速度・精度比較

Phase 3（差が有意な場合のみ）: 接触パイプライン統合
  → DOFマッピング改修（6→7 DOF/node対応）
  → broadphase/geometry/NCPの四元数対応
```

## 4. 文献リスト

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

## 確認事項

- **「試さないとわからない」は正しい**: CRで精度不足になるかは、xkep-caeの具体的なメッシュ密度・荷重条件・評価指標に依存。文献から一般論は得られるが、定量的判断には Phase 1 比較テストが必要
- **最大のブロッカーは解析的接線**: Cosseratの数値微分接線（eps=1e-7）はNR残差~1e-7にストールする。これが解消されないとCosserat移行は実用的に困難
- **1000本目標との整合**: 現段階ではCR-Timoに集中し、S4-S5でCosserat比較を並行検証するのが最も効率的

---
