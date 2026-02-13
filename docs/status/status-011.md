# status-011: 2D断面力ポスト処理 & せん断応力 & 数値試験ロードマップ

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-010](./status-010.md)

**日付**: 2026-02-13
**作業者**: Claude Code
**ブランチ**: `claude/add-testing-implementations-nlXq7`

---

## 実施内容

status-010 の短期TODO 2項目を実施し、ロードマップに数値試験フレームワーク（Phase 2.6）を追加。

### 1. 2D梁の断面力ポスト処理

3D版（`BeamForces3D`, `beam3d_section_forces()`）のミラー実装として、2D版を実装。

#### 新規クラス・関数（`beam_eb2d.py`）

| 名前 | 種別 | 概要 |
|------|------|------|
| `BeamForces2D` | dataclass | 断面力データ (N, V, M) |
| `eb_beam2d_section_forces()` | 関数 | EB梁の要素両端の断面力を計算 |
| `beam2d_max_bending_stress()` | 関数 | 断面力から最大曲げ応力を推定 |
| `beam2d_max_shear_stress()` | 関数 | 断面力から最大横せん断応力を推定 |
| `EulerBernoulliBeam2D.section_forces()` | メソッド | クラスインタフェース経由の断面力計算 |

#### 新規関数（`beam_timo2d.py`）

| 名前 | 種別 | 概要 |
|------|------|------|
| `timo_beam2d_section_forces()` | 関数 | Timoshenko梁の要素両端の断面力を計算 |
| `TimoshenkoBeam2D.section_forces()` | メソッド | クラスインタフェース経由の断面力計算 |

#### 2D断面力の計算アルゴリズム

```
1. 全体座標系の変位 → 局所座標系に変換: u_local = T @ u_global
2. 局所剛性行列で要素端力を計算: f_local = Ke_local @ u_local
3. 断面力の抽出:
   - 節点1: (N, V, M) = -f_local[0:3]
   - 節点2: (N, V, M) = f_local[3:6]
```

### 2. せん断応力のポスト処理

2D/3Dの最大せん断応力推定関数を実装。

#### 2D: `beam2d_max_shear_stress()`

断面形状に応じた横せん断応力の最大値を計算:
- 矩形断面: τ_max = 3V/(2A)
- 円形断面: τ_max = 4V/(3A)
- 一般断面: τ_max = V/A（フォールバック）

#### 3D: `beam3d_max_shear_stress()`

ねじりせん断応力と横せん断応力の保守的な和を計算:
- ねじり: τ_torsion = |Mx|·r_max/J
- 横せん断: 形状依存（上記2Dと同じ公式、Vy/Vz の大きい方を使用）
- 合計: τ_max = τ_torsion + max(τ_Vy, τ_Vz)

注意: ねじりと横せん断の最大応力発生点は一般に異なるため、保守的推定。

### 3. 数値試験フレームワーク（ロードマップ Phase 2.6 追加）

以下の4種類の材料試験を一括・部分実行できるフレームワークを Phase 2.6 としてロードマップに追加:

| 試験種別 | 荷重条件 | 主要な断面力 |
|---------|---------|-------------|
| **3点曲げ試験** | 中央集中荷重、両端単純支持 | V, M |
| **4点曲げ試験** | 2点対称荷重、両端単純支持 | V, M（純曲げ区間あり） |
| **引張試験** | 一端固定、他端軸方向荷重 | N |
| **ねん回試験** | 一端固定、他端ねじりモーメント | Mx |

設計方針: `NumericalTest`/`TestResult` データクラスによる統一インタフェース、
`run_all_tests()` / `run_tests()` による一括・部分実行API。

---

## テスト結果

**193 passed, 2 deselected (external)**（前回174 → 19テスト増加）

### テスト増加の内訳

| ファイル | 増加数 | 内容 |
|---------|--------|------|
| `tests/test_beam_eb2d.py` | +11 | 2D EB断面力テスト (4) + 最大曲げ応力 (3) + 最大せん断応力 (3) + 力の釣り合い (1) |
| `tests/test_beam_timo2d.py` | +4 | 2D Timo断面力テスト (4) |
| `tests/test_beam_timo3d.py` | +5 | 3Dせん断応力テスト (5) |

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `xkep_cae/elements/beam_eb2d.py` | 更新 — `BeamForces2D`, `eb_beam2d_section_forces()`, `beam2d_max_bending_stress()`, `beam2d_max_shear_stress()`, `EulerBernoulliBeam2D.section_forces()` 追加 |
| `xkep_cae/elements/beam_timo2d.py` | 更新 — `timo_beam2d_section_forces()`, `TimoshenkoBeam2D.section_forces()` 追加 |
| `xkep_cae/elements/beam_timo3d.py` | 更新 — `beam3d_max_shear_stress()` 追加 |
| `tests/test_beam_eb2d.py` | 更新 — 2D断面力テスト 4件 + 応力テスト 7件追加 |
| `tests/test_beam_timo2d.py` | 更新 — 2D Timoshenko断面力テスト 4件追加 |
| `tests/test_beam_timo3d.py` | 更新 — 3Dせん断応力テスト 5件追加 |
| `docs/roadmap.md` | 更新 — Phase 2.6 数値試験フレームワーク追加、現在地更新 |
| `docs/status/status-011.md` | **新規** — 本ステータス |
| `README.md` | 更新 — 現在の状態・リンク更新 |

---

## TODO（次回以降の作業）

### 短期（Phase 2 残り）

- [ ] Phase 2.6: 数値試験フレームワーク実装（`NumericalTest`, `TestResult` データクラス）
- [ ] 3点曲げ試験の実装（単純支持 + 中央荷重メッシュ生成）
- [ ] 4点曲げ試験の実装（単純支持 + 2点荷重メッシュ生成）
- [ ] 引張試験の実装
- [ ] ねん回試験の実装（3Dのみ）
- [ ] 一括実行API `run_all_tests()` / 部分実行API `run_tests()`
- [ ] pytest マーカーによる試験種別選択実行

### 中期（Phase 2.5 / Phase 3）

- [ ] Phase 2.5: Cosserat rod の設計仕様書作成
- [ ] SO(3) 回転パラメトライゼーションの選定と実装
- [ ] Cosserat rod の線形化バージョン（テスト用）
- [ ] Phase 3: Newton-Raphson ソルバーフレームワーク

### 長期（Phase 4.6 撚線モデル）

- [ ] θ_i 縮約の具体的手法検討（曲げ主目的に最適化）
- [ ] δ_ij 追跡＋サイクルカウント基盤の設計
- [ ] Level 0: 軸方向素線＋penalty接触＋正則化Coulomb＋δ_ij追跡
- [ ] Level 1: θ_i 未知量化＋被膜弾性ばね（G_c, K_c）
- [ ] Level 2: 素線Cosserat rod化＋接触ペア動的更新

### 残存する不確定事項

- [ ] 接触ペア更新頻度 N_update の適切な値（数値実験で決定）
- [ ] 接触活性化閾値 g_threshold の設定方針
- [ ] 大変形時のペア更新に伴う力の不連続の許容度
- [ ] 疲労評価のサイクルカウント手法（雨流計数法？他？）
