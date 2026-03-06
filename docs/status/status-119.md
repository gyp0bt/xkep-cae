# status-119: 動的解析・幾何学非線形基盤の物理的正しさ検証

[← README](../../README.md) | [← status-index](status-index.md) | [← status-118](status-118.md)

日付: 2026-03-06

## 概要

status-117のTODOに基づき、動的解析と幾何学非線形基盤の物理的正しさを検証。
物理テスト19件追加、検証プロット3種を新規作成。

## 1. 幾何学非線形物理テスト（test_geometric_nonlinear_physics.py — 14件）

### TestCRStressPhysics（3テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_tip_load_stress_order_of_magnitude` | CR大変形後の最大曲げ応力がPL/Zのオーダーと一致（ratio 0.5-2.0） |
| `test_moment_monotonic_decrease_from_root` | CR corotated断面力のモーメントが固定端→先端で単調減少 |
| `test_axial_stress_uniform_under_tension` | 軸引張で全要素の軸応力がP/Aの5%以内で一様 |

### TestCRCurvaturePhysics（3テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_curvature_linear_under_tip_load` | 小変形域（δ/L=2%）で曲率分布がほぼ線形（隣接差ratio<2.0） |
| `test_curvature_sign_consistency` | 一方向曲げで全有意要素の曲率符号が一致 |
| `test_curvature_zero_at_free_end` | 先端要素の曲率が固定端の15%未満 |

### TestCRLoadOrderPhysics（2テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_reaction_force_order` | 固定端反力が外力と5%以内で平衡 |
| `test_tip_displacement_order` | 先端変位が線形解析解（PL³/3EI）の0.8-1.2倍 |

### TestCRSymmetryPhysics（2テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_simply_supported_symmetric_load` | 単純支持梁の中央荷重で対称変形（誤差<2%） |
| `test_opposite_loads_give_opposite_displacements` | 正負逆荷重で正負逆変位（誤差<5%） |

### TestCRDeformationPhysics（4テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_large_deformation_stiffening` | 大変形（δ/L=15%）で幾何学的剛性効果（NL変位 < 線形変位） |
| `test_deformed_shape_smooth` | 変形形状が滑らか（隣接差分のmax/mean < 3.0） |
| `test_load_path_independence_small_deformation` | 小変形で荷重ステップ数に非依存（差<1%） |
| `test_strain_energy_equals_external_work` | エネルギーバランス（U_int/W_ext が0.8-1.2） |

### 技術的ポイント
- `beam3d_section_forces` は線形定式化のためCR変形後の応力抽出には不適切な場合がある
- CR断面力はcorotatedフレームでの `K_local @ d_cr` から直接抽出（`_extract_cr_element_moments`）
- これによりCR定式化の内部整合性を正確に検証

## 2. 動的解析+非線形複合物理テスト（test_dynamics_nonlinear_physics.py — 5件）

### TestDynamicLoadOrderPhysics（2テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_static_load_converges_to_static_solution` | ゆっくりランプ荷重+減衰→静的解に収束（誤差<15%） |
| `test_impulse_response_decays_with_damping` | 減衰あり衝撃→振幅が時間とともに減少 |

### TestDynamicDisplacementPhysics（3テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_free_vibration_oscillatory` | 非減衰自由振動で符号反転≥4回（振動的） |
| `test_cr_nonlinear_dynamic_bounded` | CR非線形動的で変位が有界（< 10L） |
| `test_velocity_consistent_with_displacement` | Newmark速度と中心差分近似が5%以内で整合 |

## 3. 検証プロット3種

### cr_stress_curvature_contour.png
- **変形メッシュ**: δ/L=3%, 10%, 20%の3段階、初期形状との比較
- **モーメント分布**: 固定端→先端で単調減少、荷重増大に比例
- **曲率分布**: モーメントと整合、線形解析解との比較

### dynamics_energy_history.png
- **線形自由振動**: ΔE/E₀ = 7.06e-13（機械精度レベルの保存）
- **CR非線形**: max|ΔE/E₀| = 2.12e-03（数値微分接線起因）
- **HHT-α散逸**: 3.25%の単調エネルギー散逸

### dynamics_displacement_response.png
- **自由振動**: 非減衰で振幅一定の正弦波
- **ランプ荷重**: 動的応答が静的解に振動しながら追従
- **減衰振動**: 理論的エンベロープ exp(-ζωt) と良好一致

## テスト結果

- **追加**: 19テスト（14 + 5）
- **全体**: 2233テスト（fast: 1672 / slow: 343 / deprecated: 218）
- lint: ruff check + format 通過

## 影響ファイル

### 新規
- `tests/test_geometric_nonlinear_physics.py` — CR幾何学非線形物理テスト14件
- `tests/test_dynamics_nonlinear_physics.py` — 動的+非線形複合物理テスト5件
- `docs/verification/cr_stress_curvature_contour.png` — CR応力/曲率コンター図
- `docs/verification/dynamics_energy_history.png` — 動的エネルギー時刻歴
- `docs/verification/dynamics_displacement_response.png` — 動的変位応答

### 変更
- `tests/generate_verification_plots.py` — 3プロット関数追加
- `README.md` — テスト数更新（2197→2233）
- `CLAUDE.md` — テスト数更新
- `docs/roadmap.md` — テスト数・到達点更新
- `docs/status/status-index.md` — status-119追加

## TODO（継続）

- Cosserat Rodの解析的接線剛性実装
- Cosserat Rodの疎行列アセンブリ実装
- Cosserat Rodの質量行列実装
- NCP版摩擦バリデーションテスト
- NCP版ヒステリシステスト
- NCP版曲げ揺動ベンチマークテスト
- CR vs Cosserat 単線比較（Phase 1比較テスト）

## 確認事項

- CR定式化の応力・曲率・エネルギーバランスが全て物理的に妥当であることを確認
- 動的解析のエネルギー保存は線形で機械精度、CR非線形で0.2%（数値微分起因）
- HHT-α/Generalized-αの数値減衰は理論通りに動作
- `beam3d_section_forces`はCR変形後の応力抽出には注意が必要（corotated断面力を直接使うべき）
- 検証プロット9種（6→9に増加）で視覚的妥当性も確認済み

---
