# status-329: ファイバー梁 Phase F4 — StrandFiberBeamProcess + ULCRFiberBeamAssembler 配線

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-13
- **ブランチ**: `claude/check-status-todos-qZNp5`
- **テスト数**: 459+13+22+5+8+12+12+25+26（Phase F4: API 8 + Physics 10 + Assembler 8 テスト追加）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

## TL;DR

status-328 TODO「ファイバー梁 Phase F4 着手 — StrandFiberBeamProcess + _beam_assembler 配線」を実行。CR（Corotational）定式化の Timoshenko 梁要素にファイバー断面積分を統合した `StrandFiberBeamProcess` と、マルチ要素アセンブラ `ULCRFiberBeamAssembler` を実装。弾性材料で線形梁と **内力 < 0.2%、接線剛性対角 < 1%** で一致。FD 自己整合性検証合格。26 テスト全合格、契約違反 0 件。

---

## 1. StrandFiberBeamProcess 実装

### 概要

CR 定式化の Timoshenko 梁要素で、断面応答をファイバー積分で評価する要素レベル Process。

### ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/elements/fiber/strand_beam.py` | `StrandFiberBeamConfig`/`StrandFiberBeamResult`/`StrandFiberBeamProcess` + `_timo_ke_from_section`/`_CRKinematics`/`_extract_cr_kinematics`/`_compute_cr_tangent` 新規作成 |
| `xkep_cae/elements/fiber/__init__.py` | エクスポート追加 |
| `xkep_cae/elements/fiber/tests/test_strand_beam_api.py` | `TestStrandFiberBeamAPI` 8 件（`@binds_to` C3 準拠） |
| `xkep_cae/elements/fiber/tests/test_strand_beam_physics.py` | `TestStrandFiberBeamPhysics` 10 件 |

### 設計の要点

- **CR キネマティクス**: `_beam_cr.py` の既存関数群（`_beam3d_length_and_direction`, `_build_local_axes`, `_rodrigues_rotation`, `_rotmat_to_rotvec`, `_rotvec_to_rotmat`）を再利用
- **ファイバー→要素変換**: `FiberSectionIntegratorProcess` で得た C_section 対角 (EA_eff, EI_y_eff, EI_z_eff) から Timoshenko 12×12 剛性行列を構築。せん断補正 (Φ 項) を保存
- **Battini & Pacoste 解析的接線**: K_mat（材料剛性）+ K_geo（幾何剛性）の完全な接線を実装。`_skew`, `_tangent_operator`, `_tangent_operator_inv` を `_beam_cr.py` から import
- **frozen dataclass**: `StrandFiberBeamConfig`（入力）、`StrandFiberBeamResult`（出力）ともに frozen
- **Process アーキテクチャ**: `uses = [FiberSectionIntegratorProcess]` 宣言

### _timo_ke_from_section の設計

標準 `timo_beam3d_ke_local(E, G, A, Iy, Iz, J, L, kappa_y, kappa_z)` と異なり、セクション積分結果（EA, EI_y, EI_z, GJ, kGA_y, kGA_z）を直接受け取る。ファイバー積分で得た有効剛性を「E×A」に分解せずにそのまま利用可能。テストで標準関数と完全一致を確認（rtol=1e-12）。

---

## 2. ULCRFiberBeamAssembler 実装

### 概要

Updated Lagrangian + Corotational のファイバー梁要素アセンブラ。マルチ要素メッシュに対して内力・接線剛性・質量行列をアセンブルし、要素ごとの section_state を管理する。

### ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/elements/_beam_assembler.py` | `ULCRFiberBeamAssembler`/`ULCRFiberBeamAssemblerInput`/`ULCRFiberBeamAssemblerProcess` 追加 |
| `xkep_cae/elements/fiber/tests/test_fiber_assembler.py` | `TestULCRFiberBeamAssemblerAPI` 5 件 + `TestULCRFiberBeamAssemblerPhysics` 3 件（`@binds_to` C3 準拠） |

### 設計の要点

- **要素ループ**: 各要素で `StrandFiberBeamProcess` を呼び出し、CSR スパース行列にアセンブル
- **状態管理**: `section_states: list[SectionState]`（要素ごと）を保持
- **checkpoint/rollback**: `coords_ref` + `section_states` のスナップショット・復元
- **update_reference(u)**: UL 参照配置更新 + section_state リセット
- **assemble_mass(rho, lumped)**: 断面直径から A, Iy, Iz を計算し、線形梁と同一の質量行列
- **Process ラッパー**: `ULCRFiberBeamAssemblerProcess`（`@binds_to` 付きテスト）

---

## 3. 物理テスト結果

### 要素レベル（StrandFiberBeamProcess）

| テスト | 断面 | 精度 | 結果 |
|--------|------|------|------|
| 軸剛性一致 | strip n=200 | < 0.2% | **合格** |
| y 軸曲げ一致 | strip n=200 | < 0.2% | **合格** |
| z 軸曲げ一致 | polar 12×24 | < 1% | **合格** |
| 接線剛性対角一致 | polar 12×24 | < 1% | **合格** |
| EI 収束（n=200→400） | strip | < 0.2% | **合格** |
| 複合変形一致 | polar 12×24 | < 1% | **合格** |
| FD 自己整合性 | polar 12×24 | atol=scale×1e-4 | **合格** |
| ねじり一致 | strip n=200 | < 1e-10 | **合格**（ファイバー非関与） |

### アセンブラレベル（ULCRFiberBeamAssembler）

| テスト | 精度 | 結果 |
|--------|------|------|
| 内力一致（5 要素片持ち梁） | < 1% | **合格** |
| 接線剛性対角一致（3 要素） | < 1% | **合格** |
| 質量行列一致（3 要素） | rtol=1e-10 | **合格** |

### ファイバー離散化精度

| 離散化 | EA 誤差 | EI_y 誤差 | EI_z 誤差 |
|--------|---------|-----------|-----------|
| strip n=200 | 0.037% | 0.139% | N/A（z=0） |
| polar 12×24 | — | 0.347% | 0.347% |

---

## 4. テスト一覧（26 件 = API 8 + Physics 10 + Assembler API 5 + Assembler Physics 3）

### test_strand_beam_api.py（8 件）

| テスト | 分類 | 内容 |
|--------|------|------|
| `test_returns_result_type` | API | 戻り値型チェック |
| `test_f_int_shape` | API | f_int (12,) |
| `test_k_elem_shape` | API | K_elem (12,12) |
| `test_k_elem_symmetric` | API | K_elem 対称 |
| `test_zero_displacement_zero_force` | API | ゼロ入力→ゼロ出力 |
| `test_state_preserved_at_zero` | API | 状態保存 |
| `test_axial_force_sign` | API | 引張正符号 |
| `test_uses_declaration` | API | FiberSectionIntegratorProcess in uses |

### test_strand_beam_physics.py（10 件）

| テスト | 分類 | 内容 |
|--------|------|------|
| `test_ke_from_section_matches_standard` | Physics | `_timo_ke_from_section` ↔ `timo_beam3d_ke_local` |
| `test_elastic_axial_stiffness_match` | Physics | 軸剛性 < 0.2% |
| `test_elastic_bending_y_match` | Physics | y 曲げ < 0.2% |
| `test_elastic_bending_z_polar_match` | Physics | z 曲げ < 1% |
| `test_elastic_tangent_matches_linear` | Physics | 接線対角 < 1% |
| `test_elastic_ei_convergence` | Physics | n=200→400 < 0.2% |
| `test_combined_deformation_polar_match` | Physics | 軸+曲げ+ねじり < 1% |
| `test_fd_tangent_self_consistency` | Physics | FD 接線自己整合 |
| `test_large_rotation_nonzero_force` | Physics | 大回転非ゼロ |
| `test_torsion_matches_linear` | Physics | ねじり完全一致 |

### test_fiber_assembler.py（8 件）

| テスト | 分類 | 内容 |
|--------|------|------|
| `test_assembler_creation` | API | 生成＋ndof |
| `test_zero_displacement_zero_force` | API | 変位ゼロ→力ゼロ |
| `test_tangent_returns_sparse` | API | sparse 行列 |
| `test_process_creation` | API | Process ラッパー動作 |
| `test_checkpoint_rollback` | API | checkpoint/rollback |
| `test_internal_force_matches_linear` | Physics | 内力 < 1% |
| `test_tangent_diagonal_matches_linear` | Physics | 接線対角 < 1% |
| `test_mass_matrix_matches_linear` | Physics | 質量行列完全一致 |

---

## TODO（次担当者向け）

### 直近

- [ ] **ファイバー梁 Phase F5 着手** — `ContactFrictionProcess` へのファイバー梁統合（アセンブラ差し替え + 非線形材料での収束テスト）
- [ ] **被膜 ON プロファイル + pypardiso 環境再ベンチ** — status-326 TODO 継続
- [ ] **n=61+ Type D stall 対策** — 大 n_strands での接触活性化時に Type D stall が発生

### 中期

- [ ] **Phase F6（ヒステリシス付き揺動シミュレーション）**: BilinearKH / MultiLayerFriction での揺動サイクル
- [ ] **リスタート解析方式への移行**: ContactFrictionProcess の I/O を `(u, v, a, 接触ペア)` 入出力に整理
- [ ] **空間ブロック分離 or ペアクラスタリング**: 物理的接触ペア数の n² 成長を抑制する構造的対策

---

## 再現手順

```bash
# Phase F4 テスト
pytest xkep_cae/elements/fiber/tests/test_strand_beam_api.py \
       xkep_cae/elements/fiber/tests/test_strand_beam_physics.py \
       xkep_cae/elements/fiber/tests/test_fiber_assembler.py -v

# 全ファイバーテスト（Phase F1-F4: 75 件）
pytest xkep_cae/elements/fiber/tests/ -v

# 契約検証
python contracts/validate_process_contracts.py
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: pytest -v 出力 75 件全合格（F1:12 + F2:12 + F3:25 + F4:26）
- [x] **再現手順記載**: 上記
- [x] **テスト数記載**: 459+13+22+5+8+12+12+25+26
- [x] **契約違反 0 件維持**: validate_process_contracts.py 実行済み
- [x] **lint/format 検証**: ruff check + ruff format OK
