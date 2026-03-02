# status-010: 3Dアセンブリテスト & 内力ポスト処理 & ワーピング検討

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-009](./status-009.md)

**日付**: 2026-02-13
**作業者**: Claude Code
**ブランチ**: `claude/check-status-todos-7yI2K`

---

## 実施内容

status-009 の短期TODO 3項目を実施。

### 1. 3D梁のアセンブリレベルテスト（`test_protocol_assembly.py` に追加）

Protocol ベースの `assemble_global_stiffness` を使った3D Timoshenko梁の統合テストを5件追加。
全テストで円形断面（`BeamSection.circle(d=10.0)`）をデフォルトとして使用。

#### 追加テスト

| テスト名 | 内容 | 検証 |
|---------|------|------|
| `test_assembly_beam_timo3d_single` | 単一要素アセンブリ | 対称性, 半正定値性, 6剛体モード |
| `test_assembly_beam_timo3d_cantilever_y` | 片持ち梁 y方向荷重 (20要素) | δ_y = PL³/(3EIz) + PL/(κGA) |
| `test_assembly_beam_timo3d_cantilever_z` | 片持ち梁 z方向荷重 (20要素) | δ_z = PL³/(3EIy) + PL/(κGA) |
| `test_assembly_beam_timo3d_torsion` | 片持ち梁 ねじり (10要素) | θ = TL/(GJ) |
| `test_assembly_beam_timo3d_inclined` | 45度傾斜梁 y方向荷重 (20要素) | 座標変換の正確性、解析解比較 |

全テストで解析解との相対誤差 < 1e-8 を確認。

### 2. 3D梁の応力・内力ポスト処理の基盤実装

要素変位から断面力（内力）を計算するポスト処理基盤を `beam_timo3d.py` に実装。

#### 新規クラス・関数

| 名前 | 種別 | 概要 |
|------|------|------|
| `BeamForces3D` | dataclass | 断面力データ (N, Vy, Vz, Mx, My, Mz) |
| `beam3d_section_forces()` | 関数 | 要素両端の断面力を計算 |
| `beam3d_max_bending_stress()` | 関数 | 断面力から最大曲げ応力を推定 |
| `TimoshenkoBeam3D.section_forces()` | メソッド | クラスインタフェース経由の断面力計算 |

#### 断面力の計算アルゴリズム

```
1. 全体座標系の変位 → 局所座標系に変換: u_local = T @ u_global
2. 局所剛性行列で要素端力を計算: f_local = Ke_local @ u_local
3. 断面力の抽出:
   - 節点1: (N, Vy, Vz, Mx, My, Mz) = -f_local[0:6]
   - 節点2: (N, Vy, Vz, Mx, My, Mz) = f_local[6:12]
```

#### 符号規約

- N 正 = 引張
- Vy/Vz 正 = 局所y/z方向の正のせん断
- Mx 正 = 正のねじり
- Mz 正 = xy面内で凸下（sagging）

#### 最大曲げ応力の推定

```
σ_max = |N/A| + |Mz|·y_max/Iz + |My|·z_max/Iy
```

ねじりによるせん断応力は含まない（将来拡張）。

#### テスト（8件追加）

| テスト名 | 検証内容 |
|---------|---------|
| `TestSectionForces::test_axial_tension` | 軸引張で N = P, 他成分ゼロ |
| `TestSectionForces::test_torsion` | ねじりで Mx = T, 他成分ゼロ |
| `TestSectionForces::test_cantilever_bending_y` | Vy = P, Mz(x) = P·(L-x) |
| `TestSectionForces::test_section_forces_via_class` | クラスメソッドと関数版の一致 |
| `TestSectionForces::test_equilibrium` | 要素両端の力の釣り合い |
| `TestMaxBendingStress::test_pure_axial` | σ = N/A |
| `TestMaxBendingStress::test_pure_bending_mz` | σ = Mz·y/Iz |
| `TestMaxBendingStress::test_combined_loading` | 複合荷重の重ね合わせ |

### 3. ワーピング（薄肉断面用）の検討

設計検討の結果、**現時点では実装不要**と判断。

#### ワーピングとは

薄肉開断面（I形、溝形など）のねじりにおいて、断面のワーピング（面外変形）を
拘束した場合に発生する追加応力。Vlasov理論に基づく。

#### 理論的背景

- St. Venant ねじり（現行実装）: θ' = T/(GJ)、ワーピング自由
- Vlasov ねじり（ワーピング拘束）: GJ·θ'' - E·C_w·θ'''' = m_t
  - C_w: ワーピング定数（断面形状依存）
  - 追加のDOF（θ' = ワーピングパラメータ）が必要 → 7 DOF/node

#### 実装が不要な理由

| 理由 | 説明 |
|------|------|
| ターゲット断面 | 撚線電線・単線・ロッドは**円形/パイプ断面**が主 |
| 円形断面のワーピング | **厳密にゼロ**（軸対称断面ではワーピングが発生しない） |
| パイプ断面のワーピング | **実質ゼロ**（薄肉閉断面ではワーピングは無視可能） |
| Phase 2.5 (Cosserat rod) | 大回転を扱う定式化に移行予定。ワーピングは別途検討 |

#### 将来必要になる場合

- I形鋼、溝形鋼などの薄肉開断面を扱う場合
- 電線以外の構造物（建築鉄骨など）に適用する場合

その場合の実装方針:
1. `BeamSection` に `C_w`（ワーピング定数）を追加
2. 7 DOF/node 版の梁要素を新規作成（`TimoshenkoBeam3DWarping`）
3. ワーピング自由度用の剛性マトリクス項を追加

**結論**: 現在のターゲット（円形/パイプ断面）ではワーピングは不要。
ロードマップの Phase 2.3 ワーピング項目を「スキップ（薄肉開断面用、現行ターゲット外）」に更新。

---

## テスト結果

**174 passed, 2 deselected (external)**（前回161 → 13テスト増加）

### テスト増加の内訳

| ファイル | 増加数 | 内容 |
|---------|--------|------|
| `test_protocol_assembly.py` | +5 | 3D梁アセンブリテスト |
| `test_beam_timo3d.py` | +8 | 断面力テスト (5) + 最大応力テスト (3) |

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `xkep_cae/elements/beam_timo3d.py` | 更新 — `BeamForces3D`, `beam3d_section_forces()`, `beam3d_max_bending_stress()`, `section_forces()` 追加 |
| `tests/test_protocol_assembly.py` | 更新 — 3D梁アセンブリテスト 5件追加 |
| `tests/test_beam_timo3d.py` | 更新 — 断面力テスト 8件追加 |
| `docs/status/status-010.md` | **新規** — 本ステータス |
| `docs/roadmap.md` | 更新 — ワーピング項目更新、ポスト処理追記 |
| `README.md` | 更新 — 現在の状態・リンク更新 |

---

## TODO（次回以降の作業）

### 短期（Phase 2 残り）

- [ ] 2D梁の断面力ポスト処理（`beam2d_section_forces()` — 3D版のミラー実装）
- [ ] せん断応力のポスト処理（ねじり τ = Mx·r/J、横せん断 τ = VQ/(Ib)）

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
