# status-214: 応力→ひずみコンター変更 + C16修正 + 材料を銅に変更

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-20
**ブランチ**: `claude/fix-stress-analysis-mpvzw`

---

## 概要

揺動解析の可視化を応力コンターからひずみコンターに変更。
変位が小さいのに300MPaは不自然 → ひずみ表示に切り替え。
材料パラメータを鉄鋼から銅（E=100GPa, ν=0.3, ρ=8.96e-9）に変更。
C16契約違反（純粋関数 `compute_element_bending_stress`）を修正。

## 変更内容

### 1. C16契約違反修正: `ElementBendingStrainProcess`（新規）

| ファイル | 操作 |
|---------|------|
| `xkep_cae/numerical_tests/beam_oscillation.py` | **変更** — 純粋関数を PostProcess に変換 |
| `xkep_cae/numerical_tests/tests/test_beam_oscillation.py` | **変更** — binds_to テスト追加 |

- `compute_element_bending_stress()` (純粋関数) → `ElementBendingStrainProcess` (PostProcess)
- 応力(σ = E*κ*R)ではなくひずみ(ε = κ*R)を計算するように変更
- `ElementBendingStrainInput` / `ElementBendingStrainOutput` (frozen dataclass) 追加

### 2. StressContour3DProcess をひずみコンターに変更

| ファイル | 操作 |
|---------|------|
| `xkep_cae/output/stress_contour.py` | **変更** — ひずみコンター + 変形倍率 + アスペクト比固定 |
| `xkep_cae/output/tests/test_stress_contour.py` | **変更** — ひずみデータ対応 |

**変更点:**
- `element_stress_snapshots` → `element_strain_snapshots`
- カラーバーラベル: `Max Bending Stress [MPa]` → `Max Bending Strain [-]`
- **変形倍率**（`deformation_scale`）: 自動計算（梁長の5%を目安）or 手動指定
- **アスペクト比固定**: 初期形状基準でxyz軸範囲を固定し、`set_box_aspect` で1:1:1維持
- `tmp/oscillation/` は毎回上書き

### 3. BeamOscillationProcess の材料・結果変更

| ファイル | 操作 |
|---------|------|
| `xkep_cae/numerical_tests/beam_oscillation.py` | **変更** — 材料→銅、strain_history |
| `tests/test_beam_oscillation.py` | **変更** — 材料→銅、ひずみベースに更新 |

- `BeamOscillationConfig` デフォルト: E=200e3→100e3, ρ=7.85e-9→8.96e-9
- `BeamOscillationResult.element_stress_history` → `element_strain_history`
- `BeamOscillationProcess.uses` に `ElementBendingStrainProcess` 追加

## テスト結果

```
506 passed, 3 xfailed
ruff check: All checks passed
ruff format: OK
契約違反: 0件（C16修正完了）
条例違反: 0件
```

## 画像出力仕様

- 出力先: `tmp/oscillation/`（毎回上書き）
- アスペクト比: 初期形状基準で固定（xyz軸範囲を変更しない）
- 変形倍率: 自動（梁長の5%をターゲットに算出）、手動指定可
- カラーマップ: jet（ひずみ 0〜ε_max）
- フレーム: 等間隔6フレーム + 時刻歴プロット

## TODO

- [ ] **UL + 動的時間積分の結合修正**: state.u の増分/累積管理を明確化し、振動を正しく再現
- [ ] **エネルギー計算の修正**: ULステップごとの増分仕事累積方式に変更
- [ ] **数値粘性の定量評価**: 振動修正後に rho_inf 依存性を検証
- [ ] 単線の剛体支えと押しジグによる動的三点曲げの解析解一致

---
