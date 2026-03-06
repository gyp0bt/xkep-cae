# status-121: S3残りTODO消化 — NCP版摩擦/ヒステリシス移行 + 37本NCP収束

[← README](../../README.md) | [← status-index](status-index.md) | [← status-120](status-120.md)

日付: 2026-03-06

## 概要

status-120のTODOに基づき、S3フェーズの残りタスクを消化。
NCP版摩擦バリデーションテスト（16件）、NCP版ヒステリシステスト（9件）、37本NCP収束テスト（2件）を新規作成。
旧ソルバー(deprecated)テストのNCP移行を完了。

## 1. NCP版摩擦バリデーションテスト（test_friction_validation_ncp.py — 16件）

旧ソルバー `test_friction_validation.py` (DEPRECATED) のNCP移行版。

### TestNCPFrictionCoulombCondition（4テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_coulomb_limit_satisfied` | 全ペアで ||q_t|| <= μ·p_n |
| `test_slip_friction_equals_mu_pn` | slip時の摩擦力 ≈ μ·p_n |
| `test_stick_condition_small_tangential_load` | 小接線荷重でstick保持 |
| `test_friction_cone_two_axes` | 2軸接線荷重でCoulomb円錐内 |

### TestNCPFrictionForceBalance（2テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_normal_force_positive_at_contact` | 接触中の法線力が正値 |
| `test_lambda_nonneg` | NCP解でλ >= 0 |

### TestNCPStickSlipTransition（2テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_increasing_tangential_load_causes_larger_displacement` | 接線荷重増加→変位増加 |
| `test_large_tangential_displacement_exceeds_small` | 大荷重>小荷重の変位 |

### TestNCPFrictionEnergyDissipation（2テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_tangential_load_causes_dissipation` | 接線荷重で散逸が非負 |
| `test_dissipation_nonnegative` | 全ペア散逸 >= 0 |

### TestNCPFrictionSymmetry（2テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_opposite_tangential_load_gives_opposite_displacement` | 反対荷重→反対変位 |
| `test_no_tangential_load_gives_zero_tangential_displacement` | 接線荷重なし→接線変位ゼロ |

### TestNCPFrictionMuDependence（3テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_zero_friction_no_tangential_resistance` | μ=0で摩擦力ゼロ |
| `test_higher_mu_gives_less_tangential_displacement` | 高μで接線変位減少 |
| `test_higher_mu_gives_less_or_equal_tangential_displacement` | 高μ <= 低μの変位 |

### TestNCPFrictionContactPenetration（1テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_friction_does_not_increase_penetration` | 摩擦有無で法線変位ほぼ同一 |

## 2. NCP版ヒステリシステスト（test_hysteresis_ncp.py — 9件）

旧ソルバー `test_hysteresis.py` (DEPRECATED) のNCP移行版。
NCPソルバーの逐次呼び出しによるサイクリック荷重でヒステリシスを検証。

### TestNCPCyclicBasic（3テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_single_phase_ncp` | 1フェーズ（0→1）でNCP収束 |
| `test_two_phase_forward_reverse` | 2フェーズ（0→1→0）荷重係数推移 |
| `test_full_cycle_load_history` | 3フェーズ（0→+1→-1→0）各フェーズ収束 |

### TestNCPTwistedWireHysteresis（3テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_tension_hysteresis_converges` | 引張往復荷重でNCP+摩擦収束 |
| `test_bending_hysteresis_converges` | 曲げ往復荷重でNCP+摩擦収束 |
| `test_torsion_hysteresis_converges` | ねじり往復荷重でNCP+摩擦収束 |

### TestNCPHysteresisPhysics（3テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_max_displacement_at_peak_load` | 最大荷重時 >= 除荷後の変位 |
| `test_displacement_returns_near_zero_after_unload` | 除荷で変位縮小 |
| `test_peak_displacement_positive` | 正荷重で正変位 |

## 3. 37本NCP収束テスト（test_ncp_convergence_19strand.py — 2件追加）

19本径方向圧縮パターン(status-112)を37本(1+6+12+18)に拡張。

### TestNCP37StrandRadialCompression（2テスト）
| テスト | 検証内容 | 結果 |
|--------|----------|------|
| `test_ncp_37strand_radial_layer1` | Layer 1のみ径方向圧縮 | **収束達成**（~3.5分） |
| `test_ncp_37strand_radial_layer1_2` | Layer 1+2複合圧縮 | NR実行確認（収束は未達、~2.5分） |

### 技術的ポイント
- `_radial_load` のlayer判定を `strand_infos[sid].layer` ベースに修正（37本以上対応）
- Layer 1のみの37本圧縮は19本と同様に収束達成
- Layer 1+2は接触ペア数が大幅増加（36ペアアクティブ）のため、現行パラメータでは力残差が~3%で停滞

## テスト結果

- **追加**: 27テスト（16 + 9 + 2）
- **全体**: 2261テスト（fast: 1689 / slow: 354 / deprecated: 218）
- lint: ruff check + format 通過

## 影響ファイル

### 新規
- `tests/contact/test_friction_validation_ncp.py` — NCP版摩擦バリデーション16件
- `tests/contact/test_hysteresis_ncp.py` — NCP版ヒステリシス9件
- `docs/status/status-121.md` — 本statusファイル

### 変更
- `tests/contact/test_ncp_convergence_19strand.py` — 37本テスト2件追加 + _radial_load修正
- `README.md` — テスト数更新（2251→2261）
- `CLAUDE.md` — テスト数更新
- `docs/roadmap.md` — S3 TODO更新
- `docs/status/status-index.md` — status-121追加

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status | 状態 |
|--------|--------|-----------|------|
| `test_friction_validation.py`（旧ソルバー） | `test_friction_validation_ncp.py`（NCP） | status-121 | NCP移行完了 |
| `test_hysteresis.py`（旧ソルバー） | `test_hysteresis_ncp.py`（NCP） | status-121 | NCP移行完了 |

## TODO（継続）

- 37本Layer1+2圧縮の収束達成（力残差 3%→0.01%）
- 61/91本の段階的収束テスト
- NCPソルバー版S3ベンチマーク（AL法との計算時間比較）
- Cosserat Rodの解析的接線剛性実装
- Cosserat Rodの疎行列アセンブリ実装
- 3D表面レンダリング（Axes3D pipe rendering）の実装検討

## 確認事項

- NCP+摩擦の収束は大きな接線荷重（f_t > μ·f_n）で困難な場合がある
  - z_sep=0.035（初期接触あり）+ g_on=0.01/g_off=0.02 で安定収束
  - z_sep=0.041（初期ギャップあり）では接触活性化時に不安定
- 37本Layer1径方向圧縮はS3改良パラメータで収束達成
- ヒステリシスのNCP実装はrun_contact_cyclic不要で、NCPソルバー逐次呼び出し方式で実現

---
