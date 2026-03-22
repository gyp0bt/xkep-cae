# NewtonUzawaProcess — 1荷重増分の NR 反復

[← contact_friction](contact_friction.md)

## 概要

1荷重増分に対して Newton-Raphson 反復を実行する SolverProcess。
ContactFrictionProcess の内部プロセスとして使用される。

**注意**: Uzawa 外ループは凍結（n_uzawa_max=1、status-221）。
現在は Huber（Fischer-Burmeister NCP）が主力接触力評価。

## 入出力

- **入力**: `NewtonUzawaStepInput`（frozen dataclass）
  - 変位・乗数・外力・固定DOF・コールバック・Strategy 群
- **出力**: `StepResult`（frozen dataclass）
  - 収束フラグ・反復数・接触力・診断情報

## アルゴリズム

1. Newton-Raphson 内部ループ
   - 接触力評価（ContactForceStrategy）
   - 摩擦力評価（FrictionStrategy）
   - 被膜力評価（CoatingStrategy）
   - 接線剛性組立 + 線形ソルブ
   - Line search + 変位更新
2. 収束判定（力残差 / 変位 / エネルギー）
