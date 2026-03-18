# ThreePointBendJigProcess — 単線の剛体支え＋押しジグ三点曲げ

[← README](../../../README.md)

## 概要

単線ワイヤを剛体支持点（ピン＋ローラー）で支持し、
剛体押しジグ（変位制御梁要素）で中央から押し下げる三点曲げ試験。
接触力伝達を経由した変位–荷重応答を Euler-Bernoulli / Timoshenko 解析解と比較する。

## 物理モデル

```
        ジグ（変位制御）
          ↓ δ_push
   ───────●───────
         ///
  ─────────────────────  ← ワイヤ（単線）
  △                  ○
  ピン              ローラー
```

- **ワイヤ**: x軸方向直線梁（Timoshenko CR 3D）
- **ジグ**: z軸方向短梁（全DOF変位制御 → 剛体）
- **接触**: ワイヤ–ジグ間ポイント接触（smooth penalty）
- **支持**: 左端=ピン（xyz並進固定）、右端=ローラー（y並進固定）

## 解析解

| 理論 | 中央変位 |
|------|---------|
| Euler-Bernoulli | δ = PL³/(48EI) |
| Timoshenko | δ = PL³/(48EI) + PL/(4κGA) |

## 入出力

- 入力: `ThreePointBendJigConfig`
- 出力: `ThreePointBendJigResult`

## パイプライン

1. ワイヤ + ジグメッシュ生成
2. 接触設定（DetectCandidatesProcess）
3. UL CR 梁アセンブラ構築
4. 境界条件設定（支持 + ジグ変位制御）
5. ContactFrictionProcess で求解
6. 解析解比較
