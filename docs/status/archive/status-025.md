# status-025: von Mises 3D塑性テスト計画策定

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-024](./status-024.md)

**日付**: 2026-02-17
**作業者**: Claude Code
**ブランチ**: `claude/setup-project-docs-loGXw`

---

## 概要

Phase 4.3（von Mises 3D弾塑性）の**実装は完了**しているが、**テストが一切ない**状態。
テスト要件が大きく担当者がフリーズしているため、テスト計画を細粒度に分割し、
次回以降の作業で段階的に実装できるようにした。

### 実装済みコード

| ファイル | 行数 | 内容 |
|---------|------|------|
| `xkep_cae/core/state.py` | — | PlasticState3D（3成分 Voigt + 将来6成分設計）|
| `xkep_cae/materials/plasticity_3d.py` | 339行 | return mapping, consistent tangent, 硬化モデル |
| `xkep_cae/assembly_plasticity.py` | 400行 | Q4, Q4_EAS, TRI3 弾塑性アセンブリ |

### テスト現状

- `test_plasticity_3d.py` **未作成**
- `test_assembly_plasticity.py` **未作成**
- 検証図 **未生成**

---

## テスト計画

5つの大ステップに分け、各ステップをさらに3サブステップに分解。
各サブステップは **テスト3本程度** で、1回の作業セッションで完結する粒度。

---

### Step 1: 構成則単体テスト — 基本動作

**ファイル**: `tests/test_plasticity_3d.py` (Part A)
**依存**: `plasticity_3d.py` のみ。アセンブリ不要。

#### Step 1-1: 弾性応答テスト（3テスト）

1Dテストの `TestPlasticity1DElastic` に相当。降伏未満の基本動作確認。

- [ ] `test_elastic_stress_uniaxial`: 単軸引張（ε_xx のみ）で σ = D_e @ ε を検証
- [ ] `test_elastic_tangent`: 降伏未満で `tangent == D_e` を検証
- [ ] `test_elastic_state_unchanged`: 降伏未満で `eps_p`, `alpha`, `beta` が全てゼロのまま

**解析解**: σ = D_e @ ε（平面ひずみ弾性テンソル）

#### Step 1-2: 降伏境界・完全弾塑性テスト（3テスト）

- [ ] `test_yield_boundary_uniaxial`: 単軸引張で f_trial ≈ 0 の境界ケース
- [ ] `test_perfectly_plastic_uniaxial`: H_iso=0, C_kin=0 で降伏後の応力が von Mises 降伏面上に留まる
- [ ] `test_state_immutability`: return_mapping に渡した state が変更されないこと

**注意**: 平面ひずみでの単軸引張は σ_zz ≠ 0 なので、解析解は 1D と異なる。

#### Step 1-3: 単軸引張・降伏後テスト（3テスト）

- [ ] `test_uniaxial_tension_bilinear`: 等方硬化の bilinear 応答解析解比較
- [ ] `test_uniaxial_eps_p_direction`: 塑性ひずみの方向確認（偏差応力方向）
- [ ] `test_uniaxial_alpha_increment`: 等価塑性ひずみ α の増分が正

---

### Step 2: 構成則単体テスト — Consistent tangent & 硬化モデル

**ファイル**: `tests/test_plasticity_3d.py` (Part B)

#### Step 2-1: Consistent tangent 有限差分検証（3テスト）★最重要

- [ ] `test_ctangent_fd_elastic`: 弾性域での FD 検証（3x3 全9成分）
- [ ] `test_ctangent_fd_plastic_isotropic`: 等方硬化での FD 検証（相対誤差 < 1e-5）
- [ ] `test_ctangent_fd_plastic_kinematic`: Armstrong-Frederick での FD 検証

**FDヘルパーテンプレート**:
```python
def _tangent_fd_3d(plasticity, strain, state, h=1e-7):
    D_fd = np.zeros((3, 3))
    for j in range(3):
        e_j = np.zeros(3)
        e_j[j] = h
        r_plus = plasticity.return_mapping(strain + e_j, state)
        r_minus = plasticity.return_mapping(strain - e_j, state)
        D_fd[:, j] = (r_plus.stress - r_minus.stress) / (2.0 * h)
    return D_fd
```

#### Step 2-2: 除荷・逆載荷テスト（3テスト）

- [ ] `test_unloading_elastic_tangent`: 引張降伏後に除荷 → tangent = D_e に戻る
- [ ] `test_bauschinger_uniaxial`: 移動硬化で引張→圧縮。逆方向の降伏応力低下
- [ ] `test_reload_after_unload`: 除荷後に再載荷 → 拡大した降伏面から再び塑性化

#### Step 2-3: Voce硬化・複合硬化テスト（3テスト）

- [ ] `test_voce_hardening_saturation`: Voce硬化で応力が σ_y0 + Q_inf に漸近
- [ ] `test_combined_iso_kin`: 等方硬化 + 移動硬化の組合せ
- [ ] `test_ctangent_fd_voce_kin`: Voce + Armstrong-Frederick の FD 検証

---

### Step 3: 構成則単体テスト — 多軸応力パス

**ファイル**: `tests/test_plasticity_3d.py` (Part C)

#### Step 3-1: 純せん断テスト（3テスト）

- [ ] `test_pure_shear_yield_stress`: 純せん断での降伏。τ_y = σ_y0 / √3
- [ ] `test_pure_shear_return_mapping`: 純せん断で降伏後の応力
- [ ] `test_pure_shear_ctangent_fd`: 純せん断状態での tangent FD 検証

**解析解**: q = √3 * |μ * γ_xy|,  γ_y = σ_y0 / (√3 * μ)

#### Step 3-2: 二軸応力テスト（3テスト）

- [ ] `test_equibiaxial_yield`: 等二軸引張 ε = [ε, ε, 0] の降伏判定
- [ ] `test_biaxial_stress_ratio`: 比例載荷 ε = [2ε, ε, 0] の応力比
- [ ] `test_yield_surface_locus`: 降伏曲面の複数点（6〜8方向）チェック

#### Step 3-3: 非比例載荷・ロバスト性テスト（3テスト）

- [ ] `test_nonproportional_loading`: 2段階載荷（単軸引張→せん断追加）
- [ ] `test_large_strain_increment`: 降伏ひずみの100倍でも破綻しない
- [ ] `test_multi_step_incremental`: 多ステップ（10〜20ステップ）増分解析

---

### Step 4: 要素アセンブリテスト

**ファイル**: `tests/test_assembly_plasticity.py` (Part A)

#### Step 4-1: 弾性一致テスト（3テスト）

- [ ] `test_q4_elastic_match`: Q4要素で降伏未満 → 弾性アセンブリと一致（rtol < 1e-12）
- [ ] `test_tri3_elastic_match`: TRI3要素で同様
- [ ] `test_q4_eas_elastic_match`: Q4_EAS要素で同様

#### Step 4-2: パッチテスト（3テスト）

- [ ] `test_q4_patch_uniaxial`: Q4要素 2x2メッシュで一様引張パッチテスト
- [ ] `test_tri3_patch_uniaxial`: TRI3要素でパッチテスト
- [ ] `test_q4_eas_patch_uniaxial`: Q4_EAS要素でパッチテスト

#### Step 4-3: 全体接線剛性の有限差分検証（3テスト）

- [ ] `test_q4_global_tangent_fd`: Q4要素の K_T FD 検証
- [ ] `test_tri3_global_tangent_fd`: TRI3要素の K_T FD 検証
- [ ] `test_q4_eas_global_tangent_fd`: Q4_EAS要素の K_T FD 検証

---

### Step 5: 構造レベルテスト & 検証図

**ファイル**: `tests/test_assembly_plasticity.py` (Part B) + `tests/generate_verification_plots.py`

#### Step 5-1: NR収束テスト（3テスト）

- [ ] `test_nr_single_elem_tension`: 単要素引張での NR 収束（3〜5反復）
- [ ] `test_nr_quadratic_convergence`: 二次収束率 ||r_{k+1}|| ≈ C * ||r_k||^2
- [ ] `test_nr_load_increments`: 多ステップ荷重増分解析

#### Step 5-2: 多要素構造テスト（3テスト）

- [ ] `test_multi_elem_uniform_tension_q4`: Q4 4x1メッシュで bilinear 解析解比較
- [ ] `test_multi_elem_uniform_tension_tri3`: TRI3メッシュで同様
- [ ] `test_strip_bending_plastic`: 曲げ荷重での塑性域進展

#### Step 5-3: 検証図生成（3タスク）

- [ ] 降伏曲面プロット（σ_xx-σ_yy 平面、解析解 = 楕円 + 数値解マーカー）
- [ ] 応力パスプロット（単軸・二軸・純せん断の応力パス）
- [ ] 荷重-変位曲線（bilinear 解析解と数値解の重ね描き）

---

## 実施優先順位

| 順位 | ステップ | テスト数 | 重要度 | 理由 |
|------|---------|---------|--------|------|
| 1 | Step 1 (基本動作) | 9 | ★★★ | 構成則の基本が動かないと何もできない |
| 2 | Step 2-1 (tangent FD) | 3 | ★★★ | NR収束の前提。最重要テスト |
| 3 | Step 3-1 (純せん断) | 3 | ★★★ | 多軸応力の最も基本的なケース |
| 4 | Step 4-1 (弾性一致) | 3 | ★★☆ | アセンブリ配線の正確性 |
| 5 | Step 4-3 (K_T FD) | 3 | ★★☆ | NR収束のアセンブリレベル検証 |
| 6 | Step 2-2, 2-3 | 6 | ★★☆ | 硬化モデルの網羅 |
| 7 | Step 3-2, 3-3 | 6 | ★★☆ | 多軸応力の網羅 |
| 8 | Step 4-2 (パッチ) | 3 | ★☆☆ | 要素性能検証 |
| 9 | Step 5-1, 5-2 | 6 | ★☆☆ | 構造レベル統合 |
| 10 | Step 5-3 (検証図) | 3 | ★☆☆ | ドキュメント用 |

**合計: 45テスト + 3検証図**

---

## 次回作業への引き継ぎTODO（バッチ単位）

各バッチは**コミット1回**の粒度。1セッション ≈ 1〜2バッチ。

### バッチ1: 基本動作テスト（最優先）

- [ ] **TODO-A1**: Step 1-1 実装 — `test_plasticity_3d.py` 新規作成。弾性応答テスト3本 + ヘルパー関数定義。lint → コミット
- [ ] **TODO-A2**: Step 1-2 実装 — 降伏境界・完全弾塑性テスト3本。lint → コミット
- [ ] **TODO-A3**: Step 1-3 実装 — 単軸引張・降伏後テスト3本。lint → コミット

### バッチ2: Consistent tangent 検証（最重要）

- [ ] **TODO-B1**: Step 2-1 実装 — tangent FD検証3本（弾性・塑性等方・塑性移動）。lint → コミット

### バッチ3: 多軸テスト

- [ ] **TODO-C1**: Step 3-1 実装 — 純せん断テスト3本。lint → コミット
- [ ] **TODO-C2**: Step 3-2 実装 — 二軸応力テスト3本。lint → コミット

### バッチ4: 硬化・逆載荷テスト

- [ ] **TODO-D1**: Step 2-2 実装 — 除荷・逆載荷テスト3本。lint → コミット
- [ ] **TODO-D2**: Step 2-3 実装 — Voce硬化・複合硬化テスト3本。lint → コミット

### バッチ5: 非比例載荷・ロバスト性

- [ ] **TODO-E1**: Step 3-3 実装 — 非比例載荷・大変形・多ステップテスト3本。lint → コミット

### バッチ6: アセンブリテスト前半

- [ ] **TODO-F1**: Step 4-1 実装 — `test_assembly_plasticity.py` 新規作成。弾性一致テスト3本。lint → コミット
- [ ] **TODO-F2**: Step 4-2 実装 — パッチテスト3本。lint → コミット

### バッチ7: アセンブリテスト後半

- [ ] **TODO-G1**: Step 4-3 実装 — 全体接線FD検証3本。lint → コミット

### バッチ8: 構造テスト

- [ ] **TODO-H1**: Step 5-1 実装 — NR収束テスト3本。lint → コミット
- [ ] **TODO-H2**: Step 5-2 実装 — 多要素構造テスト3本。lint → コミット

### バッチ9: 検証図・ドキュメント・仕上げ

- [ ] **TODO-I1**: Step 5-3 実装 — `generate_verification_plots.py` に3プロット追加
- [ ] **TODO-I2**: `docs/verification/` に検証図と説明文を追加
- [ ] **TODO-I3**: roadmap Phase 4.3 チェックボックス更新、README更新、最終status作成

---

## 平面ひずみ単軸引張の解析解メモ

次回実装者への参考として、テストで使う主要な解析解を記す。

### 弾性域

```
ε = [ε_xx, 0, 0]
σ = [(λ+2μ)ε_xx, λε_xx, 0]
σ_zz = λε_xx（拘束応力）
```

### 降伏ひずみ（単軸引張）

平面ひずみの単軸引張では σ_zz ≠ 0:
```
σ = [(λ+2μ)ε, λε, 0],  σ_zz = λε
p = (σ_xx + σ_yy + σ_zz) / 3 = (λ + 2μ/3)ε
s_xx = (4μ/3)ε, s_yy = (-2μ/3)ε, s_zz = (-2μ/3)ε
q = √(3/2) * √((4μ/3)² + 2*(2μ/3)²) * ε = 2μ * ε

∴ ε_y = σ_y0 / (2μ)
```

### 純せん断

```
ε = [0, 0, γ_xy]
σ = [0, 0, μ*γ_xy]
q = √3 * |μ*γ_xy|

∴ γ_y = σ_y0 / (√3 * μ)
```

---

## 確認事項・懸念

1. **Voigt規約の一貫性**: `plasticity_3d.py` では γ_xy = 2ε_xy を使用。B行列も同じ規約か要確認（パッチテストで発覚するはず）
2. **eps_zz_e の計算**: `return_mapping` L186 で `eps_zz_e = state.eps_p[0] + state.eps_p[1]`。体積不変条件（J2塑性）の仮定に依存。テストで確認したい
3. **Q4_EAS のα=0近似**: `assembly_plasticity.py` L161 で EAS 自由度を無視。NR収束テスト（Step 5-1）で確認必要
4. **テスト数**: 45テスト + 3図。15バッチに分割済み。1セッション1〜2バッチで進めること

---

## 運用メモ（Codex/Claude 2交代制）

- **本PRはテスト計画のみ。テストコードの変更なし**
- Phase 4.3 のコード実装（plasticity_3d.py, assembly_plasticity.py）は前回までに完了済み
- テストは TODO-A1 から順に進めること
- テストでバグ発見時は、テストと実装を同一コミットで修正
- lint: `ruff check xkep_cae/ tests/ && ruff format --check xkep_cae/ tests/`
