# status-014: Cosserat rod Phase 2.5 完成 & 数値試験フレームワーク拡張

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-013](./status-013.md)

**日付**: 2026-02-14
**作業者**: Claude Code
**ブランチ**: `claude/execute-status-todos-Sf0AC`

---

## 実施内容

status-013 の TODO に基づき、Phase 2.5 後半（内力・幾何剛性・初期曲率）と
Phase 2 の残り（数値試験フレームワーク拡張）を完了。

### 1. Cosserat rod 内力ベクトル `internal_force()`

Phase 3（幾何学的非線形）への準備として、Cosserat rod の内力ベクトルを実装。

```
f_int = ∫₀ᴸ Bᵀ · σ ds
σ = C · (ε - ε₀)    ε₀ = 初期曲率
```

| 関数 | 説明 |
|------|------|
| `cosserat_internal_force_local()` | 局所内力ベクトル（ガウス求積、初期曲率対応） |
| `cosserat_internal_force_global()` | 全体座標系変換付き内力ベクトル |
| `CosseratRod.internal_force()` | クラスメソッド |

**検証**: 線形領域では `Ke · u` と一致（相対誤差 < 1e-10）。

### 2. 幾何剛性行列 `geometric_stiffness()`

座屈解析・非線形解析に必要な幾何剛性行列を導出・実装。

```
Kg = ∫₀ᴸ N·(∂v₁/∂x · ∂v₂/∂x + ∂w₁/∂x · ∂w₂/∂x) ds
   + ∫₀ᴸ Mx·(∂θy₁/∂x · ∂θz₂/∂x - ∂θz₁/∂x · ∂θy₂/∂x) ds
```

| 関数 | 説明 |
|------|------|
| `cosserat_geometric_stiffness_local()` | 軸力N + ねじりMxによる幾何剛性 |
| `cosserat_geometric_stiffness_global()` | 全体座標系変換付き |
| `CosseratRod.geometric_stiffness()` | クラスメソッド |

**特徴**:
- 軸力Nによる横変位微分の二次形式
- ねじりMxによる回転DOF連成
- 対称行列（Kg = Kgᵀ）
- 引張時にKg正定値（剛性増加）

### 3. 初期曲率 `kappa_0` サポート

ヘリカル構造の基盤として、ストレスフリー配位での初期曲率を導入。

```python
rod = CosseratRod(section=sec, kappa_0=np.array([κ_x, κ_y, κ_z]))
# κ_x: 初期ねじり率、κ_y/κ_z: 初期曲率
```

- 構成則レベルで歪みから初期曲率を差し引く: `ε_eff = ε - ε₀`
- 初期曲率配位で変位ゼロ → 内力ゼロ（ストレスフリー）
- `CosseratRod.kappa_0` プロパティで取得

### 4. 数値試験フレームワークへの Cosserat rod 統合

`beam_type="cosserat"` で Cosserat rod を数値試験フレームワークから利用可能に。

| 変更箇所 | 内容 |
|---------|------|
| `core.py` | `BeamType` に `"cosserat"` 追加、バリデーション修正 |
| `runner.py` | `_get_ke_func()` に cosserat 分岐追加 |
| `runner.py` | `_compute_section_forces()` に cosserat 分岐追加 |
| `runner.py` / `frequency.py` | `is_3d` 判定を `in ("timo3d", "cosserat")` に統一 |

**検証テスト**: 引張・ねん回・3点曲げ・4点曲げ・断面力の5テストが全パス。

### 5. pytest マーカー対応

試験種別ごとの選択実行を可能にした。

```bash
pytest -m bend3p     # 3点曲げのみ
pytest -m cosserat   # Cosseratのみ
pytest -m "bend3p or tensile"  # 複合
```

| マーカー | 対象テスト |
|---------|----------|
| `bend3p` | 3点曲げ試験テスト |
| `bend4p` | 4点曲げ試験テスト |
| `tensile` | 引張試験テスト |
| `torsion` | ねん回試験テスト |
| `freq_response` | 周波数応答試験テスト |
| `cosserat` | Cosserat rod 数値試験テスト |

`pyproject.toml` にマーカー定義を追加済み。

### 6. 周波数応答試験の固有振動数検証

FRF ピーク検出による固有振動数推定を、Euler-Bernoulli 梁のカンチレバー解析解と比較。

```
解析解: f₁ = (β₁L)² / (2πL²) · √(EI/(ρA))
β₁L = 1.8751（カンチレバー第1モード）
```

20要素EB2D梁 + 加速度励起FRFで、相対誤差 5%以内を確認。
※ 断面二次モーメントの規約（b/h方向）を `_build_section_props` と整合させた。

### 7. 非一様メッシュサポート

荷重点周辺で細分割する非一様メッシュ生成関数を追加。

| 関数 | 説明 |
|------|------|
| `generate_beam_mesh_2d_nonuniform()` | 2D非一様メッシュ |
| `generate_beam_mesh_3d_nonuniform()` | 3D非一様メッシュ |

**特徴**: 荷重点座標リストと `refinement_factor` を指定し、荷重点周辺の要素密度を自動増加。

---

## テスト結果

**345 passed, 2 skipped**（前回 314 → 31テスト増加）

### テスト増加の内訳

| テストファイル | 増加数 | 内容 |
|-------------|-------|------|
| `test_beam_cosserat.py` | +20 | 内力・幾何剛性・初期曲率テスト |
| `test_numerical_tests.py` | +11 | Cosserat数値試験・周波数応答解析解・非一様メッシュ |

### 新規テスト詳細

#### test_beam_cosserat.py（36 → 56テスト）

| テストクラス | テスト数 | 内容 |
|------------|---------|------|
| `TestInternalForce` | 7 | 線形等価性（軸/ねじり/曲げ）、全体座標系、クラスメソッド、ゼロ変位、2点ガウス |
| `TestGeometricStiffness` | 8 | 形状、対称性、ゼロ応力、引張、比例性、全体変換、クラスメソッド、正定値性 |
| `TestInitialCurvature` | 5 | 属性設定、デフォルトNone、初期曲率下の内力、相殺、ヘリカル |

#### test_numerical_tests.py（新規テスト）

| テストクラス | テスト数 | 内容 |
|------------|---------|------|
| `TestCosseratNumerical` | 5 | tensile, torsion, bend3p, bend4p, section_forces |
| `TestFrequencyResponseAnalytical` | 1 | EB2D固有振動数 vs 解析解（5%以内） |
| `TestNonUniformMesh` | 4 | 2D/3D基本、複数ポイント、要素数増加確認 |
| `TestValidation` | +1 | cosserat beam_type バリデーション |

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `xkep_cae/elements/beam_cosserat.py` | 更新 — internal_force, geometric_stiffness, kappa_0 追加 |
| `xkep_cae/numerical_tests/core.py` | 更新 — cosserat対応、非一様メッシュ関数追加 |
| `xkep_cae/numerical_tests/runner.py` | 更新 — cosserat分岐追加 |
| `xkep_cae/numerical_tests/frequency.py` | 更新 — cosserat対応 |
| `xkep_cae/numerical_tests/__init__.py` | 更新 — 非一様メッシュ関数エクスポート |
| `tests/test_beam_cosserat.py` | 更新 — 20テスト追加 |
| `tests/test_numerical_tests.py` | 更新 — 11テスト追加、pytestマーカー追加 |
| `pyproject.toml` | 更新 — pytestマーカー定義追加 |
| `docs/status/status-014.md` | **新規** — 本ステータス |
| `docs/roadmap.md` | 更新 — Phase 2.5/2.6 チェックボックス更新 |
| `README.md` | 更新 — 状態・テスト数・リンク更新 |

---

## TODO（次回以降の作業）

### 短期（Phase 3: 幾何学的非線形）

- [ ] Newton-Raphson ソルバーフレームワーク
- [ ] 四元数の増分更新ロジック: q_{n+1} = quat_from_rotvec(Δθ) ⊗ q_n
- [ ] Cosserat rod の非線形歪み計算
- [ ] テスト: 大変形片持ち梁（Euler elastica）

### 短期（Phase 2 残り）

- [ ] 一般断面（任意形状、メッシュベース数値積分）

### 中期（Phase 4.6 撚線モデル）

- [ ] θ_i 縮約の具体的手法検討（曲げ主目的に最適化）
- [ ] δ_ij 追跡＋サイクルカウント基盤の設計
- [ ] Level 0: 軸方向素線＋penalty接触＋正則化Coulomb＋δ_ij追跡

### 確認事項（ユーザーへの質問）

- [ ] Phase 3（幾何学的非線形）の着手タイミング
- [ ] Euler elastica ベンチマークの優先度
