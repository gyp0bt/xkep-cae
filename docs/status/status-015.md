# status-015: Cosserat rod SRI & Phase 3 幾何学的非線形開始

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-014](./status-014.md)

**日付**: 2026-02-14
**作業者**: Claude Code
**ブランチ**: `claude/cosserat-sri-nonlinear-Y4T6Q`

---

## 実施内容

status-014 の TODO に基づき、以下を実施:

1. Cosserat梁にせん断のみ低減積分（SRI）バリエーションを追加
2. Phase 3（幾何学的非線形）の実装開始: Newton-Raphsonソルバー、接線剛性、大変形梁テスト

### 1. SRI（選択的低減積分）バリエーション

Cosserat rod に `integration_scheme="sri"` オプションを追加。
せん断項（Γ₂, Γ₃）のみ1点ガウス求積（低減積分）、
それ以外（Γ₁軸伸び、κ₁ねじり、κ₂κ₃曲率）は2点ガウス求積（完全積分）。

```
Ke_sri = ∫₀ᴸ Bᵀ · C_full · B ds   （2点ガウス: 軸・ねじり・曲率）
       + ∫₀ᴸ Bᵀ · C_shear · B ds  （1点ガウス: せん断のみ）
```

| 関数 | 説明 |
|------|------|
| `cosserat_ke_local_sri()` | SRI版 局所剛性行列 |
| `cosserat_ke_global_sri()` | SRI版 全体座標系剛性行列 |
| `cosserat_internal_force_local_sri()` | SRI版 局所内力ベクトル |
| `cosserat_internal_force_global_sri()` | SRI版 全体座標系内力ベクトル |

**CosseratRodクラス対応**:
- `integration_scheme` パラメータ追加: `"uniform"`（デフォルト）or `"sri"`
- `local_stiffness()`, `internal_force()` がスキームに応じて自動分岐

**特徴**:
- 軸力・ねじりは uniform/1点 と完全一致（構造が同じため）
- 曲げの収束が高速（2点完全積分の恩恵）
- せん断ロッキング回避は1点低減積分で維持

### 2. Newton-Raphson ソルバーフレームワーク

Phase 3 の基盤となる非線形ソルバーを `solver.py` に実装。

```python
result = newton_raphson(
    f_ext_total, fixed_dofs,
    assemble_tangent,      # u → K_T(u) コールバック
    assemble_internal_force,  # u → f_int(u) コールバック
    n_load_steps=10,
    max_iter=30,
)
```

| 機能 | 説明 |
|------|------|
| 荷重増分 | 全荷重を n_load_steps に等分割 |
| Newton反復 | 各ステップで残差 R = f_ext - f_int を0に収束 |
| 接線剛性 | K_T = K_material + K_geometric |
| 収束判定 | 力ノルム / 変位ノルム / エネルギーノルムの3基準 |
| 結果 | `NonlinearResult` データクラス（変位履歴・荷重履歴） |

**コールバック設計**: アセンブリロジックはコールバック関数として外部から注入。
要素種別やメッシュに依存しない汎用フレームワーク。

### 3. 接線剛性行列

CosseratRod クラスに `tangent_stiffness()` メソッドを追加。

```
K_T = K_material + K_geometric
```

- `K_material`: `local_stiffness()` で計算済みの材料剛性
- `K_geometric`: `geometric_stiffness()` で計算済みの幾何剛性

### 4. 非線形梁アセンブリヘルパー

`assemble_cosserat_beam()` 関数を追加。
直線梁（x軸方向）の接線剛性と内力を一括アセンブリする。

```python
K_T, f_int = assemble_cosserat_beam(
    n_elems, L, rod, material, u,
    stiffness=True, internal_force=True,
)
```

### 5. 大変形片持ち梁テスト

Newton-Raphson + 接線剛性の検証として、以下のテストを実装:

| テスト | 内容 |
|--------|------|
| `test_large_deflection_converges` | 大荷重（L/5相当たわみ）でNRが収束 |
| `test_nonlinear_stiffer_than_linear` | 引張下で幾何剛性効果が出現（非線形 < 線形変位） |
| `test_sri_large_deflection` | SRIスキームでも大変形解析が収束 |
| `test_load_history_recorded` | 荷重増分の履歴が正しく記録される |

---

## テスト結果

**374 passed, 2 skipped**（前回 345 → 29テスト増加）

### テスト増加の内訳

| テストファイル | テスト数 | 内容 |
|-------------|---------|------|
| `test_cosserat_sri.py` | 18 | SRI剛性行列・内力・クラス統合テスト |
| `test_nonlinear.py` | 11 | Newton-Raphson線形/非線形・大変形テスト |

### 新規テスト詳細

#### test_cosserat_sri.py（18テスト）

| テストクラス | テスト数 | 内容 |
|------------|---------|------|
| `TestSRIStiffnessMatrix` | 6 | 形状・対称性・正半定値性・軸/ねじりuniform一致・全体変換 |
| `TestSRIAxialTorsion` | 2 | 1要素軸引張・ねじり厳密 |
| `TestSRIBendingConvergence` | 2 | SRI vs uniform精度比較・32要素精度確認 |
| `TestSRIInternalForce` | 2 | f_int=Ke·u等価・ゼロ変位ゼロ内力 |
| `TestCosseratRodSRIClass` | 6 | スキーム選択・バリデーション・剛性/内力/接線剛性 |

#### test_nonlinear.py（11テスト）

| テストクラス | テスト数 | 内容 |
|------------|---------|------|
| `TestNonlinearResult` | 1 | データクラス生成 |
| `TestNewtonRaphsonLinear` | 3 | 軸引張・曲げ・荷重増分（線形→NR一致） |
| `TestAssembleCosseratBeam` | 3 | 剛性のみ・内力のみ・ゼロ変位ゼロ力 |
| `TestLargeDeformationCantilever` | 4 | 大変形収束・非線形剛性効果・荷重履歴・SRI大変形 |

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `xkep_cae/elements/beam_cosserat.py` | 更新 — SRI関数4本、tangent_stiffness()、assemble_cosserat_beam()、integration_schemeパラメータ |
| `xkep_cae/solver.py` | 更新 — NonlinearResult、newton_raphson() 追加 |
| `tests/test_cosserat_sri.py` | **新規** — SRI 18テスト |
| `tests/test_nonlinear.py` | **新規** — 非線形 11テスト |
| `docs/status/status-015.md` | **新規** — 本ステータス |
| `docs/roadmap.md` | 更新 — Phase 3 チェックボックス更新 |
| `README.md` | 更新 — 状態・テスト数・リンク更新 |

---

## TODO（次回以降の作業）

### 短期（Phase 3 幾何学的非線形 残り）

- [ ] 弧長法（Arc-length / Riks法）: スナップスルー・座屈追跡用
- [ ] ラインサーチ（収束加速）
- [ ] 四元数の非線形歪み計算: Γ = R(q)ᵀ r' - e₁（線形化なし）
- [ ] 四元数増分更新: q_{n+1} = quat_from_rotvec(Δθ) ⊗ q_n をNR内で使用
- [ ] Euler elastica ベンチマーク（解析解との定量比較）
- [ ] Lee's frame 等の標準非線形ベンチマーク

### 短期（Phase 2 残り）

- [ ] 一般断面（任意形状、メッシュベース数値積分）
- [ ] 数値試験フレームワークの cosserat-sri 対応

### 中期（Phase 4.6 撚線モデル）

- [ ] θ_i 縮約の具体的手法検討
- [ ] δ_ij 追跡＋サイクルカウント基盤の設計
- [ ] Level 0: 軸方向素線＋penalty接触＋正則化Coulomb＋δ_ij追跡

### 設計上の注記

- 現在の Newton-Raphson は接線剛性 K_T = K_m + K_g（材料+幾何）で線形化した
  updated-Lagrangian 的アプローチ。大変形で回転が大きい場合、四元数の明示的な
  非線形歪み計算への移行が必要（次回以降）。
- SRI は Cosserat rod の1次要素で特に効果的。高次要素を将来実装する場合は
  積分スキームの見直しが必要。
