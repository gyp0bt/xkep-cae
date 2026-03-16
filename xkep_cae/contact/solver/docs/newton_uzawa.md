# NewtonUzawaProcess — 1荷重増分の NR+Uzawa 反復

[← contact_friction](contact_friction.md)

## 概要

1荷重増分に対して Newton-Raphson + Uzawa 乗数更新を実行する SolverProcess。
ContactFrictionProcess の内部プロセスとして使用される。

## 入出力

- **入力**: `NewtonUzawaStepInput`（frozen dataclass）
  - 変位・乗数・外力・固定DOF・コールバック・Strategy 群
- **出力**: `StepResult`（frozen dataclass）
  - 収束フラグ・反復数・接触力・診断情報

## アルゴリズム

1. Uzawa 外部ループ（乗数更新）
2. Newton-Raphson 内部ループ
   - 接触力評価（ContactForceStrategy）
   - 摩擦力評価（FrictionStrategy）
   - 被膜力評価（CoatingStrategy）
   - 接線剛性組立 + 線形ソルブ
   - Line search + 変位更新
3. 収束判定（力残差 / 変位 / エネルギー）
