# status-328: ファイバー梁 Phase F3 — CircularFiberSection + FiberSectionIntegratorProcess

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-13
- **ブランチ**: `claude/check-status-todos-kMWwI`
- **テスト数**: 459+13+22+5+8+12+12+25（Section API 9 + Integrator API 6 + Integrator Physics 10 テスト追加）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

## TL;DR

status-327 TODO「ファイバー梁 Phase F3 着手」を実行。`CircularFiberSection` frozen dataclass（strip/polar 2 生成メソッド）と `FiberSectionIntegratorProcess`（断面積分 Process）を実装。FD 接線検証を Elastic/BilinearKH/MultiLayerFriction の 3 材料で合格。弾性 EI 誤差 < 1%。25 テスト全合格。

---

## 1. CircularFiberSection 実装

### 概要

円形断面をファイバー離散化する frozen dataclass。

### ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/elements/fiber/section.py` | `CircularFiberSection` + `_strip_area()` 新規作成 |
| `xkep_cae/elements/fiber/__init__.py` | エクスポート追加 |
| `xkep_cae/elements/fiber/tests/test_section_api.py` | `TestCircularFiberSectionAPI` 9 件 |

### 設計の要点

- **frozen dataclass**: diameter, n_fiber, y, z, area（全て tuple）
- **`strip()` classmethod**: y 方向ストリップ分割（`05_smooth_teardrop.py::FiberSection._area` と同一面積計算）
- **`polar()` classmethod**: 極座標格子分割（n_radial × n_theta）、3D 二軸曲げ対応
- **面積精度**: strip/polar ともに面積総和が解析値（πR²）と 1% 以内で一致

### テスト（9 件）

| テスト | 内容 |
|--------|------|
| `test_strip_returns_frozen_dataclass` | frozen 性 |
| `test_strip_lengths` | y/z/area 長さ |
| `test_strip_z_all_zero` | 平面曲げで z=0 |
| `test_strip_y_symmetry` | y 座標原点対称 |
| `test_strip_area_positive` | 面積正値 |
| `test_strip_total_area_convergence` | 面積総和 < 1% 誤差 |
| `test_polar_returns_correct_count` | ファイバー数 |
| `test_polar_total_area_convergence` | 面積総和 < 1% 誤差 |
| `test_strip_matches_prototype` | 05_smooth_teardrop.py 一致 |

---

## 2. FiberSectionIntegratorProcess 実装

### 概要

ファイバーループで逐次 `material.evaluate()` を呼び出し、断面力 (N, M_y, M_z) と接線行列 C_section (3×3) を積算する Process。

### ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/elements/fiber/integrator.py` | `FiberIntegratorConfig`/`FiberIntegratorResult`/`FiberSectionIntegratorProcess` 新規作成 |
| `xkep_cae/elements/fiber/__init__.py` | エクスポート追加 |
| `xkep_cae/elements/fiber/tests/test_integrator_api.py` | `TestFiberSectionIntegratorAPI` 6 件（`@binds_to` C3 準拠） |
| `xkep_cae/elements/fiber/tests/test_integrator_physics.py` | `TestFiberSectionConvergence` 10 件 |

### 接線行列

$$\mathbf{C}_{\text{sec}} = \sum_{i=1}^{n_f} E_t^{(i)} A_i \begin{bmatrix} 1 & -y_i & z_i \\ -y_i & y_i^2 & -y_i z_i \\ z_i & -y_i z_i & z_i^2 \end{bmatrix}$$

- 弾性均質断面: 対角 = EA, EI、非対角 = 0（対称性）
- 部分塑性: 非対角 ≠ 0（軸-曲げカップリング）

### FD 接線検証

| 材料 | FD h | 検証方法 | 結果 |
|------|------|---------|------|
| Elastic1D | 1e-6 | atol = scale × 1e-5 | **合格** |
| BilinearKH（塑性後弾性域） | 1e-6 | atol = scale × 1e-5 | **合格** |
| MultiLayerFriction（スリップ後弾性域） | 1e-6 | atol = scale × 1e-5 | **合格** |

注: 対称断面の弾性域では off-diagonal が数値的にゼロ。FD の数値キャンセレーション回避のため、max(|C|) × 1e-5 をスケーリングした atol を使用。

### テスト（16 件 = API 6 + Physics 10）

| テスト | 分類 | 内容 |
|--------|------|------|
| `test_returns_result_type` | API | 戻り値型チェック |
| `test_result_has_expected_fields` | API | フィールド存在 |
| `test_c_section_shape` | API | C_section 3×3 対称 |
| `test_state_new_length` | API | 新状態ファイバー数 |
| `test_frozen_result` | API | frozen 性 |
| `test_zero_strain_gives_zero_forces` | API | ゼロ入力→ゼロ出力 |
| `test_elastic_ei_convergence` | Physics | EI < 1% 誤差 |
| `test_elastic_ea_convergence` | Physics | EA < 1% 誤差 |
| `test_elastic_coupling_zero` | Physics | 弾性対称 → カップリング=0 |
| `test_n_fiber_convergence` | Physics | n=20/40/60/80 収束 |
| `test_tangent_fd_elastic` | Physics | **FD 接線 Elastic** |
| `test_tangent_fd_bilinear_kh` | Physics | **FD 接線 BilinearKH** |
| `test_tangent_fd_multilayer_friction` | Physics | **FD 接線 MultiLayerFriction** |
| `test_pure_bending_moment_sign` | Physics | M_y 符号 |
| `test_plastic_coupling_nonzero` | Physics | 部分降伏カップリング |
| `test_moment_matches_prototype` | Physics | 05_smooth_teardrop.py 一致 |

---

## TODO（次担当者向け）

### 直近

- [ ] **ファイバー梁 Phase F4 着手** — `StrandFiberBeamProcess` + `_beam_assembler` 配線（弾性 EI 一致 < 0.1%）
- [ ] **被膜 ON プロファイル + pypardiso 環境再ベンチ** — status-326 TODO 継続
- [ ] **n=61+ Type D stall 対策** — 大 n_strands での接触活性化時に Type D stall が発生

### 中期

- [ ] **リスタート解析方式への移行**: ContactFrictionProcess の I/O を `(u, v, a, 接触ペア)` 入出力に整理
- [ ] **空間ブロック分離 or ペアクラスタリング**: 物理的接触ペア数の n² 成長を抑制する構造的対策

---

## 再現手順

```bash
# Phase F3 テスト
pytest xkep_cae/elements/fiber/tests/ -v

# 契約検証
python contracts/validate_process_contracts.py
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: pytest -v 出力 49 件全合格（F1:12 + F2:12 + F3:25）
- [x] **再現手順記載**: 上記
- [x] **テスト数記載**: 459+13+22+5+8+12+12+25
- [x] **契約違反 0 件維持**: validate_process_contracts.py 実行済み
- [x] **lint/format 検証**: ruff check + ruff format --check OK
