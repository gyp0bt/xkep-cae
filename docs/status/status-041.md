# status-041: 三点曲げ非線形動解析スクリプト + GIFアニメーション出力

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-19

## 概要

status-040 の TODO「三点曲げテストにxkep-cae側でも非線形動解析実施」を実行。
Abaqus三点曲げモデル（idx1: 弾性）と同条件で非線形動解析を実施し、
GIFアニメーション・荷重-変位比較プロット・サマリーを出力するスクリプトを追加。

テスト数 865（変更なし。既存テスト全パス確認済み）。

## 実施内容

### 1. `examples/run_nonlinear_bend3p.py` 新設 — 三点曲げ非線形動解析スクリプト

Abaqus三点曲げモデル（`assets/test_assets/Abaqus/1-bend3p/` idx1）と同一パラメータで
xkep-caeの非線形動解析（`solve_nonlinear_transient`）を実行するスタンドアロンスクリプト。

#### モデル仕様（Abaqus idx1 準拠）

| パラメータ | 値 |
|-----------|-----|
| モデル | yz対称1/2モデル |
| 半モデル長 | 50.0 mm |
| 要素数 | 100（B31相当 3D Timoshenko梁） |
| 断面 | 円形 d=1.0mm |
| ヤング率 | 100 GPa (100,000 MPa) |
| ポアソン比 | 0.3 |
| 密度 | 8.96e-9 ton/mm³ |
| 支持位置 | x=25mm (uy=0) |
| 対称BC | x=0: ux=0, uz=0, θx=0, θy=0, θz=0 |

#### 解析条件

| 項目 | 値 |
|------|-----|
| ソルバー | `solve_nonlinear_transient` (Newmark-β + NR) |
| 時間積分 | 平均加速度法 (β=0.25, γ=0.5) |
| 荷重 | ランプ荷重 -5.0 N (100s かけて線形増加) |
| 減衰 | Rayleigh 剛性比例 (β=1e-3) |
| 質量行列 | 集中質量 (HRZ法) |
| 時間刻み | 0.5 s |
| ステップ数 | 200 |

#### 解析結果

| 指標 | 値 |
|------|-----|
| 全ステップ収束 | OK (各ステップ NR 2回反復) |
| 最終中央変位 | -5.310 mm |
| 最終支持点反力 | 5.000 N |
| 線形剛性 | 0.9416 N/mm |
| Abaqus 線形剛性 | 0.9520 N/mm |
| **剛性相対差異** | **1.09%** |

#### 出力ファイル

| ファイル | 内容 |
|---------|------|
| `examples/output/nonlinear_bend3p/animation_xy.gif` | 変形アニメーション（40フレーム, xy ビュー） |
| `examples/output/nonlinear_bend3p/force_displacement.png` | 荷重-変位曲線比較 (xkep-cae vs Abaqus) |
| `examples/output/nonlinear_bend3p/displacement_time.png` | 変位・荷重時刻歴 |
| `examples/output/nonlinear_bend3p/summary.txt` | 解析サマリー |

### 2. Abaqus比較の知見

- Abaqusモデルは変位制御（準静的動解析, NLGEOM=YES, APPLICATION=QUASI-STATIC）
- xkep-caeは力制御（ランプ荷重）+ 線形梁剛性
- **初期線形領域（小変形）での剛性一致は良好（1.09%差異）**
- Abaqusの支持点反力は動的ピーク（t≈44s, ~10N）を経て定常状態（t=100s, ~2.44N）に収束
  - 大ストローク時（30mm）はNLGEOM効果で剛性が大幅に低下（Abaqus RF2最終=2.44N, 線形外挿=28N）
- xkep-cae側は線形梁のため幾何学的非線形効果は反映されない
  - 大変形比較には Cosserat rod の非線形モード活用が必要

### 3. 既存テストへの影響

- **既存テスト 865 件**: 全パス（破壊なし）
- 新規テスト追加なし（スクリプトのみ追加）

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `examples/run_nonlinear_bend3p.py` | **新規** — 三点曲げ非線形動解析スクリプト |
| `examples/output/nonlinear_bend3p/` | **新規** — GIF/PNG/TXT 出力（gitignore 対象） |
| `README.md` | テスト数・機能反映 |
| `docs/roadmap.md` | 非線形動解析の三点曲げスクリプト追加記載 |
| `docs/status/status-index.md` | status-041 行追加 |

## テスト数

865（変更なし）

## 確認事項・懸念

1. **大変形比較の限界**: xkep-caeの線形梁では幾何学的非線形効果を捉えられない。Abaqusとの大ストローク比較には Cosserat rod 非線形モード or TL定式化梁が必要
2. **変位制御の非線形動解析**: 現在のソルバーは力制御のみ。Abaqus準拠の変位制御（velocity BC）を直接再現するには、ソルバー拡張が必要
3. **CJKフォント**: ヘッドレス環境ではCJKフォントが不可用。プロットラベルは英語で出力
4. **出力ディレクトリ `examples/output/`**: .gitignore に追加推奨（バイナリ出力）

## TODO

- [ ] **Timoshenko 3D 梁の UL（Updated Lagrangian）定式化** — Q4要素で実装済みのULを梁要素に拡張。幾何学的非線形（大変形・大回転）を Timoshenko 3D 梁で扱えるようにし、Abaqus NLGEOM 三点曲げとの大ストローク比較を可能にする
- [ ] Cosserat rod 非線形モードでの大変形三点曲げ比較（既存の `assemble_cosserat_beam()` + `solve_nonlinear_transient()` で実現可能）
- [ ] `apply_dirichlet` のスパース行列非ゼロ規定変位バグの修正
- [ ] Abaqus弾塑性三点曲げ（idx2）のバリデーション
- [ ] 要素単体剛性のAbaqus比較（接触フェーズ完了後に入念に実施）
- [ ] Phase C3: 摩擦 return mapping + μランプ
- [ ] Phase C4: merit line search + 探索/求解分離の運用強化

---
