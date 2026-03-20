# status-214: 複数フィールドコンター（S11/LE11/SK1） + C16修正 + 銅材料

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-20
**ブランチ**: `claude/fix-stress-analysis-mpvzw`

---

## 概要

揺動解析の可視化を応力コンターからAbaqus準拠のフィールドラベル体系に変更。
S11（応力）、LE11（対数ひずみ）、SK1（曲率）の3フィールドを同時出力。
材料パラメータを鉄鋼から銅（E=100GPa, ν=0.3, ρ=8.96e-9）に変更。
C16契約違反（純粋関数 `compute_element_bending_stress`）を修正。

## 変更内容

### 1. C16契約違反修正: `ElementBendingStrainProcess`

- `compute_element_bending_stress()` (純粋関数) → `ElementBendingStrainProcess` (PostProcess)
- 出力に `element_curvature` (κ) と `element_strain` (ε=κR) の両方を含める
- `ElementBendingStrainInput` / `ElementBendingStrainOutput` (frozen dataclass)

### 2. 複数フィールドコンター対応

| ファイル | 操作 |
|---------|------|
| `xkep_cae/output/stress_contour.py` | **変更** — `ContourFieldInput` で複数フィールド受入 |
| `xkep_cae/output/tests/test_stress_contour.py` | **変更** — 3フィールドテスト |

**フィールドラベル規約（Abaqus準拠）:**
- **S**: 応力 (S11=曲げ応力 [MPa])
- **LE**: 対数ひずみ (LE11=曲げひずみ [-])
- **SK**: 曲率 (SK1=曲率 [1/mm])

**StressContour3DConfig 変更:**
- `element_strain_snapshots` → `contour_fields: list[ContourFieldInput]`
- 各フィールドに `name` + `snapshots` を指定
- 各フィールドごとに独立した3D画像セットを生成
- 時刻歴プロットは全フィールドを1枚に統合（変位 + SK1 + LE11 + S11）
- **変形倍率** 自動計算 + **アスペクト比固定**

### 3. BeamOscillationResult 拡張

- `contour_fields: dict[str, list[np.ndarray]]` 追加（S11, LE11, SK1）
- `element_strain_history` は後方互換で残留（= contour_fields["LE11"]）
- 材料デフォルト: E=200e3→100e3, ρ=7.85e-9→8.96e-9 (銅)

### 4. 画像生成スクリプト

| ファイル | 操作 |
|---------|------|
| `contracts/generate_oscillation_images.py` | **新規** — tmp/oscillation/に画像出力 |

## 解析結果

| フィールド | 最大値 | 単位 |
|-----------|--------|------|
| SK1 (曲率) | 1.52e-03 | 1/mm |
| LE11 (ひずみ) | 1.52e-03 | - |
| S11 (応力) | 152.4 | MPa |

整合性: S11 = E × LE11 = 100e3 × 1.52e-3 = 152 MPa (一致)

## 画像出力

出力先: `tmp/oscillation/`（毎回上書き、25枚）

| ファイル | 内容 |
|---------|------|
| `beam_osc_SK1_000〜007.png` | 曲率コンター（8フレーム） |
| `beam_osc_LE11_000〜007.png` | ひずみコンター（8フレーム） |
| `beam_osc_S11_000〜007.png` | 応力コンター（8フレーム） |
| `beam_osc_time_history.png` | 変位 + SK1 + LE11 + S11 時刻歴 |

## テスト結果

```
518 passed, 5 xfailed
ruff check: All checks passed
ruff format: OK
契約違反: 0件
条例違反: 0件
```

## TODO

- [ ] **UL + 動的時間積分の結合修正**: state.u の増分/累積管理を明確化し、振動を正しく再現
- [ ] **エネルギー計算の修正**: ULステップごとの増分仕事累積方式に変更
- [ ] **数値粘性の定量評価**: 振動修正後に rho_inf 依存性を検証
- [ ] 単線の剛体支えと押しジグによる動的三点曲げの解析解一致

---
