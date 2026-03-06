# status-117: 動的解析物理テスト + Generalized-α法実装 + 梁要素主体決定

[← README](../../README.md) | [← status-index](status-index.md) | [← status-116](status-116.md)

日付: 2026-03-06

## 概要

1. **動的解析の物理テスト**: エネルギー保存・大変形・対称性・周波数・安定性を検証する13テスト作成
2. **Generalized-α法実装**: Chung & Hulbert (1993) に基づく時間離散化。パラメータ自動計算、API・収束・物理テスト14件
3. **CR-Timo vs Cosserat Rod比較**: 物理・収束性・計算コストの3観点で定量比較
4. **梁要素主体決定**: 短期CR-Timo（実用的）、中長期Cosserat移行（解析的接線・質量行列が必要）

## 動的解析物理テスト（test_dynamics_physics.py）

### TestDynamicsEnergyPhysics（3テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_cantilever_free_vibration_energy_linear` | 線形梁Newmark-βでエネルギー保存（誤差<1%） |
| `test_cantilever_free_vibration_energy_cr_nonlinear` | CR非線形梁でエネルギー保存（誤差<10%、数値微分起因） |
| `test_hht_alpha_numerical_dissipation_beam` | HHT-α（α=-0.1）で全エネルギー減少を確認 |

### TestDynamicsLargeRotationPhysics（2テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_cantilever_large_deflection_cr_converges` | CR-Timoでδ/L≈10%の大変形が動的に収束 |
| `test_cantilever_large_deflection_cosserat_converges` | CosseratでもΔ/L≈10%の大変形が動的に収束 |

### TestCRvsCosseratPhysics（3テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_small_deformation_match` | 小変形で先端変位差<5%（10要素・動的解析） |
| `test_large_deformation_convergence_comparison` | 大変形（δ/L≈10%）で両者の差<1%（20要素・静的NR） |
| `test_cosserat_nr_convergence_stall_documented` | Cosserat NR残差ストール（~1e-7）の記録テスト |

### TestDynamicsSymmetryPhysics（1テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_simply_supported_symmetric_response` | 単純支持梁の対称荷重→対称変形（誤差<1%） |

### TestDynamicsFrequencyPhysics（2テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_cantilever_fundamental_frequency` | FEM第1固有周波数 vs 理論解（β₁L=1.8751）差<10% |
| `test_geometric_stiffening_increases_frequency` | 軸引張で固有振動数が増加することを固有値解析で確認 |

### TestDynamicsStabilityPhysics（2テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_newmark_unconditional_stability` | Δt=10×Δt_crでも安定（無条件安定性） |
| `test_central_difference_instability_detection` | Δt=1.5×Δt_crで不安定フラグが立つ |

## Generalized-α法実装（dynamics.py + test_generalized_alpha.py）

### 実装内容
- `generalized_alpha_params(rho_inf)`: Chung-Hulbert (1993) の公式でα_m, α_f, γ, β を計算
  - α_m = (2ρ∞-1)/(ρ∞+1), α_f = ρ∞/(ρ∞+1)
  - γ = 1/2 - α_m + α_f, β = (1-α_m+α_f)²/4
- `GeneralizedAlphaConfig`: ρ∞指定でパラメータ自動計算、Newmark互換パラメータ
- `solve_generalized_alpha()`: 非線形NR反復、中間時刻での慣性/減衰評価、減衰行列C対応
- `GeneralizedAlphaResult`: 時間・変位・速度・加速度・反復数履歴

### テスト（test_generalized_alpha.py）

#### TestGeneralizedAlphaAPI（6テスト）
- ρ∞=1.0→Newmark等価（α_m=α_f=0.5）
- ρ∞=0.0→最大減衰（α_m=-1, α_f=0）
- ρ∞=0.9→中間値、無効入力バリデーション、Config自動計算

#### TestGeneralizedAlphaConvergence（3テスト）
- 線形自由振動の振幅保存
- ρ∞=1.0でNewmark結果と一致（atol=1e-8）
- Duffing振動子（硬化ばね）で全ステップ収束

#### TestGeneralizedAlphaPhysics（5テスト）
- ρ∞=1.0: Duffing振動子のエネルギー保存（<0.5%）
- ρ∞=0.5: 数値減衰でエネルギー単調減少
- 減衰のρ∞単調性: ρ∞ small→large で減衰量が減少
- HHT-α比較: 同等パラメータで両者が合理的な応答
- 減衰+静的荷重→定常値への収束（<2%誤差）

## CR-Timo vs Cosserat Rod 比較結果

### 物理的妥当性
| 項目 | CR-Timo | Cosserat |
|------|---------|----------|
| 小変形精度 | 2要素で解析解 | 要素数依存（O(h²)収束） |
| 大変形（δ/L=10%） | 20要素で差<1% | 20要素で差<1% |
| 回転表現 | 3パラメータ（回転ベクトル） | 4元数（特異点なし） |

### 収束性
| 項目 | CR-Timo | Cosserat |
|------|---------|----------|
| NR収束限界 | ~1e-7〜1e-8（数値微分起因） | ~1e-7〜1e-8（数値微分起因） |
| 解析的接線 | なし（中心差分eps=1e-7） | なし（中心差分eps=1e-7） |
| 動的解析tol | 1e-6実用限界 | 1e-6実用限界 |

### 計算コスト
| 項目 | CR-Timo | Cosserat |
|------|---------|----------|
| NR反復あたり | やや遅い | ~1.7x高速 |
| アセンブリ | COO/CSR疎行列 ✅ | 密行列のみ ❌ |
| 質量行列 | あり ✅ | なし ❌ |
| 接触テスト | 全テスト済み ✅ | 接触テストなし ❌ |

### 結論: 段階的移行方針
- **短期（〜S4）**: CR-Timoを主体（疎行列、質量行列、接触テスト全対応）
- **中期（S5〜）**: Cosseratに解析的接線・疎行列アセンブリ・質量行列を追加
- **長期**: Cosserat Rodを主体（4元数の特異点なし、幾何学的精密性）

### 課題（Cosserat移行のための必要改良）
1. 解析的接線剛性（数値微分からの脱却）→NR収束精度向上
2. COO/CSR疎行列アセンブリ（現在はnp.ix_で密行列のみ）
3. 一貫した質量行列の実装
4. 接触テストの移行

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status | 状態 |
|--------|--------|-----------|------|
| Newmark-β/HHT-α のみ | Generalized-α法追加 | status-117 | 共存（ρ∞=1.0でNewmark等価） |

## テスト結果

- **追加**: 27テスト（13 + 14）
- **全体**: 2197テスト（fast: 1682 / slow: 297 / deprecated: 218）
- lint: ruff check + format 通過

## 影響ファイル

### 新規
- `tests/test_dynamics_physics.py` — 動的解析物理テスト13件
- `tests/test_generalized_alpha.py` — Generalized-α法テスト14件

### 変更
- `xkep_cae/dynamics.py` — Generalized-α法（generalized_alpha_params, GeneralizedAlphaConfig, GeneralizedAlphaResult, solve_generalized_alpha）追加

## TODO

- Cosserat Rodの解析的接線剛性実装
- Cosserat Rodの疎行列アセンブリ実装
- Cosserat Rodの質量行列実装
- Cosserat Rodの接触テスト移行
- NCP版摩擦バリデーションテスト（status-116からの継続TODO）
- NCP版ヒステリシステスト（status-116からの継続TODO）
- NCP版曲げ揺動ベンチマークテスト（status-116からの継続TODO）

## 確認事項

- Cosseratの数値微分接線はeps=1e-7で残差~1e-7〜1e-8にストールする。解析的接線が中長期の最重要課題
- Generalized-α法はρ∞=1.0でNewmark-βと完全等価。ρ∞∈[0,1]で連続的に数値減衰を制御可能
- 動的解析の物理テストにより、エネルギー保存・大変形・対称性・周波数精度・安定性の全てを検証済み

---
