# 3点曲げ

## 前提

- yz対称1/2モデル
- 線長100mm
- 曲げ治具、抑え治具ともにR5
- ストローク30mm
- 準静的安定化動解析(APPLICATION=QUASI-STATIC)
- 幾何学非線形あり(NLGEOM)
- 剛性行列の非対称性を考慮(UNSYM)

## 計算ファイル

- go_idx{index}.inp: 計算インプット
- go_idx{index}_RF.csv: 曲げ治具の時系列データ
- go_idx{index}_SK.csv: 各要素の曲率の時系列データ
- go_idx{index}.sta: 収束履歴(参考)
- go_idx{index}.msg: 収束履歴詳細(参考)

## 条件一覧

- idx1:
    - 分類: 弾性動解析
    - 線直径1mm
    - ヤング率: 100 [GPa]
    - ポアソン比: 0.3 [-]
    - 密度: 8.96E-7 [ton/mm^3]

- idx2:
    - 分類: 等方弾塑性動解析
    - 線直径1mm
    - ヤング率: 100 [GPa]
    - ポアソン比: 0.3 [-]
    - 密度: 8.96E-7 [ton/mm^3]
    - 塑性変形あり(等方硬化)
