# status-021: Phase 4.1 1次元弾塑性 完了

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-020](./status-020.md)

**日付**: 2026-02-14
**作業者**: Claude Code
**ブランチ**: `master`

---

## 実施内容

Phase 4（材料非線形）の最初のサブフェーズ Phase 4.1 を完了。
Cosserat rod の**軸方向のみ**に 1D 弾塑性構成則（return mapping）を実装。
等方硬化（線形）+ 移動硬化（Armstrong-Frederick）、consistent tangent modulus を提供。
既存の 407 テストを維持しつつ 28 テスト追加、合計 435 テストパス（2 skipped）。

### Step 1: PlasticState1D データクラス

**ファイル**: `xkep_cae/core/state.py`（新規）

- `PlasticState1D` — 1D 塑性状態（eps_p, alpha, beta）
- `CosseratPlasticState` — Cosserat 要素の塑性状態（axial: PlasticState1D）
  - Phase 4.2 でファイバーモデル用に bending_y, bending_z 等を追加予定
- `xkep_cae/core/__init__.py` にエクスポート追加

### Step 2-3: Plasticity1D 構成則

**ファイル**: `xkep_cae/materials/plasticity_1d.py`（新規）

- `IsotropicHardening` — 線形等方硬化パラメータ（sigma_y0, H_iso）
- `KinematicHardening` — Armstrong-Frederick 移動硬化パラメータ（C_kin, gamma_kin）
- `Plasticity1D` — 1D 弾塑性構成則クラス
  - `return_mapping(strain, state)` → `ReturnMappingResult(stress, tangent, state_new)`

#### Return mapping アルゴリズム

1. 弾性試行: `sigma_trial = E * (eps - eps_p_n)`
2. 有効応力: `xi_trial = sigma_trial - beta_n`
3. 降伏関数: `f = |xi_trial| - (sigma_y0 + H_iso * alpha_n)`
4. `f ≤ tol` → 弾性（`D_ep = E`、状態変更なし）
5. `f > tol` → 塑性修正:
   - `gamma_kin = 0`: closed-form `dg = f / (E + H_iso + C_kin)`
   - `gamma_kin > 0`: Newton 反復（AF 回復項あり）
6. Consistent tangent: `D_ep = E * H_bar / (E + H_bar)`

**重要な知見**: 降伏面上の判定トレランスを `f_trial <= 1e-10 * sigma_y_n` とする必要がある。
`f_trial <= 0.0` では浮動小数点丸め誤差で `f_trial ≈ +epsilon` となった場合に
D_ep が E から急激に変化し、NR 法の発散を引き起こす。
Simo & Hughes (1998) でも推奨される標準的な手法。

### Step 4: 弾塑性アセンブリ関数

**ファイル**: `xkep_cae/elements/beam_cosserat.py`（関数追加、既存変更なし）

- `_compute_generalized_stress_plastic()` — 軸方向(index 0)のみ return mapping、他は弾性
  - `stress[0] = sigma * A`（一般化応力 = 応力 × 断面積）
  - `C_tangent[0,0] = D_ep * A`（接線も同様）
- `assemble_cosserat_beam_plastic()` — 弾塑性アセンブリ
  - uniform / SRI 両方対応
  - SRI: 非せん断成分は 2 点ガウス（各点に独立な PlasticState1D）

### Step 5: テスト

**ファイル**: `tests/test_plasticity_1d.py`（新規、30テスト）

#### 構成則単体テスト
- 降伏未満で弾性（sigma = E*eps, D = E）
- 降伏点近傍の境界テスト
- 等方硬化の単調載荷（bilinear 解析解と一致）
- 除荷で弾性勾配復帰
- consistent tangent 値の検証
- 完全弾塑性（H=0）
- 移動硬化（バウシンガー効果）
- Armstrong-Frederick 繰返し載荷
- consistent tangent の有限差分検証
- 状態不変性（state_n が return_mapping で変更されない）

#### 要素・構造レベルテスト
- 弾性一致テスト（降伏未満で plastic 版 = 通常版）
- 弾塑性棒の荷重-変位曲線（bilinear 解析解）
- 接線剛性の有限差分検証（全体 K_T）
- NR 二次収束（consistent tangent による）
- 多要素一様引張（全要素同一歪み）
- SRI 版テスト

### Step 6: 検証図

**ファイル**: `tests/generate_verification_plots.py`（新規）
**出力**: `docs/verification/`（4枚の PNG）

1. `stress_strain_isotropic.png` — 等方硬化の応力-歪み曲線（解析解 vs 数値解）
2. `hysteresis_loop.png` — 繰返し荷重のヒステリシスループ
3. `bauschinger_comparison.png` — バウシンガー効果（等方 vs 移動硬化）
4. `load_displacement_bar.png` — 弾塑性棒の荷重-変位（NR 結果 vs 解析解）

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `xkep_cae/core/state.py` | **新規**: PlasticState1D, CosseratPlasticState |
| `xkep_cae/core/__init__.py` | エクスポート追加 |
| `xkep_cae/materials/plasticity_1d.py` | **新規**: Plasticity1D, return mapping, consistent tangent |
| `xkep_cae/elements/beam_cosserat.py` | 関数追加: `_compute_generalized_stress_plastic()`, `assemble_cosserat_beam_plastic()` |
| `tests/test_plasticity_1d.py` | **新規**: 28件テスト |
| `tests/generate_verification_plots.py` | **新規**: 検証図生成スクリプト |
| `docs/verification/*.png` | **新規**: 検証図 4枚 |
| `CLAUDE.md` | 検証図規約追加 |
| `docs/roadmap.md` | Phase 4.1 チェックリスト更新 |
| `README.md` | 現在の状態・テスト数・ステータスリンク・使用例更新 |
| `docs/status/status-021.md` | **新規**: 本ステータス |

---

## テスト結果

```
435 passed, 2 skipped (199.05s)
```

新規追加テスト内訳:
- `test_plasticity_1d.py` — 構成則 + 要素 + NR 28件
- **合計 28件追加**（407 → 435）

---

## 技術的知見（デバッグ記録）

### 降伏面上の浮動小数点問題

除荷後の再載荷で NR 法が発散する問題が発生。原因は降伏面上（f_trial ≈ 0）で
float64 の丸め誤差により f_trial が微小正値となり、塑性ブランチに入ること。
このとき D_ep ≈ E*H/(E+H) ≈ 995 となるが、弾性ブランチなら D_ep = E = 200000。
約 200 倍の剛性差が NR の振動・発散を引き起こした。

**対策**: `f_trial <= 1e-10 * sigma_y_n`（相対トレランス）に変更。
計算塑性力学の標準的手法。

### 弾性一致テストの幾何剛性差

`assemble_cosserat_beam()` は `rod.tangent_stiffness()` 経由で幾何剛性 Kg を含むが、
`assemble_cosserat_beam_plastic()` は B^T·C·B（材料剛性のみ）。
ゼロ変位（Kg=0）でのK比較と、非ゼロ変位での f_int 比較に分離して解決。

---

## 設計判断

- **軸方向のみ**: Phase 4.1 では軸力の塑性のみ。曲げの塑性は Phase 4.2（ファイバーモデル）で対応。
- **既存関数は変更なし**: `assemble_cosserat_beam()` はそのまま残し、新関数 `assemble_cosserat_beam_plastic()` を追加。
- **状態管理パターン**: `newton_raphson(n_load_steps=1)` を外部ループで回し、コールバック内クロージャで states_trial を管理。収束後に states を確定。
- **SRI 対応**: 非せん断成分の 2 点ガウス各点に独立な PlasticState1D を配置。

---

## 次作業（TODO）

### 優先度A（Phase 4 続き）
- [ ] Phase 4.2: ファイバーモデル（曲げの塑性化）
- [ ] Phase 4.3: 3D von Mises 降伏関数
- [ ] Phase 4.4: 3D 一般化材料構成則

### 優先度B（Phase 5）
- [ ] Phase 5: 動的解析（Newmark-β、lumped mass）

### 優先度C
- [ ] Phase C: 梁–梁接触

---

## 引き継ぎメモ（Codex/Claude 2交代運用）

- **Phase 4.1 完了**。1D return mapping + consistent tangent + 弾塑性アセンブリが使える状態。
- `assemble_cosserat_beam_plastic()` は NR のコールバックとして使う。使用パターンは `tests/test_plasticity_1d.py` の `TestPlasticBarAnalytical` を参照。
- `tests/generate_verification_plots.py` で検証図を再生成可能（`python tests/generate_verification_plots.py`）。
- Armstrong-Frederick の `gamma_kin > 0` は Newton 反復で解く。`gamma_kin = 0` は closed-form。
- 降伏判定のトレランス `1e-10 * sigma_y_n` は重要。変更するとNR発散の可能性あり。
