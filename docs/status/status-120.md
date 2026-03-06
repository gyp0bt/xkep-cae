# status-120: 被膜/シース物理テスト + 撚線断面3D可視化 + 断面繊維応力コンター

[← README](../../README.md) | [← status-index](status-index.md) | [← status-119](status-119.md)

日付: 2026-03-06

## 概要

status-119のTODOおよびユーザー要望に基づき、被膜(CoatingModel)/シース(SheathModel)の物理テスト18件を新規作成。
さらに撚線断面の3D構造可視化（素線+被膜+シース）と、円形断面内の繊維応力コンター図を追加。

## 1. 被膜/シース物理テスト（test_coating_sheath_physics.py — 18件）

### TestCoatingPhysics（6テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_coating_always_increases_stiffness` | どんな材料でも被膜は剛性を増加させる |
| `test_thicker_coating_stiffer` | 厚い被膜ほど剛性が大きい（単調増加） |
| `test_coating_stiffness_scales_with_modulus_ratio` | 被膜寄与はE_coat/E_wire比に概ね比例 |
| `test_coating_contact_radius_geometry` | 接触半径 = 素線半径 + 被膜厚（厳密） |
| `test_annular_area_positive_and_bounded` | 環状断面積が正かつ外接円面積より小さい |
| `test_strain_energy_additivity` | EA_total = EA_wire + EA_coat（エネルギー加法性） |

### TestSheathPhysics（8テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_sheath_encloses_all_wires` | 3/7/19本全てでシースが全素線を囲む（ギャップ≥0） |
| `test_sheath_encloses_coated_wires` | 被膜付きでもシースが全ワイヤを囲む |
| `test_larger_strand_needs_larger_sheath` | 素線本数増加→シース内径増加 |
| `test_sheath_radius_geometry_consistency` | r_inner = r_envelope + clearance（厳密） |
| `test_sheath_stiffness_proportional_to_modulus` | EA比がヤング率比に一致 |
| `test_sheath_thicker_is_stiffer` | 厚いシースほど剛性大 |
| `test_clearance_increases_sheath_radius` | クリアランス増加→内径増加 |
| `test_sheath_section_area_ring_formula` | A = π(r_out² - r_in²)で幾何学的に正確 |

### TestCoatingSheathCombinedPhysics（4テスト）
| テスト | 検証内容 |
|--------|----------|
| `test_coating_shifts_envelope_outward` | 被膜によりエンベロープが被膜厚だけ外に |
| `test_sheath_with_coating_larger_than_without` | 被膜ありシース > 被膜なしシース |
| `test_sheath_stiffness_with_coating_differs` | 被膜でシース内径変化→EI変化 |
| `test_19wire_sheath_stiffer_than_7wire` | 19本のシースEI > 7本のシースEI |

## 2. 検証プロット2種

### twisted_wire_cross_section.png
- **断面図(z=0)**: 7本素線（青円）+ 被膜環（橙）+ シース内外面（赤）+ エンベロープ（緑点線）
- **断面図(z=L/2)**: 中央断面、素線がヘリカル回転している様子を確認
- **側面図(xz平面)**: 全7素線のヘリカル中心線 + シース外面境界

### fiber_stress_cross_section.png
- **固定端(z=0)**: My=-5000Nm、上下に引張/圧縮の対称分布（最大±5400MPa）
- **中央(z=0.5)**: My=-2500Nm、応力振幅が半分
- **先端(z=1.0)**: My=-500Nm、応力振幅が1/10
- FiberSection.circle(d, nr=8, nt=16) による128ファイバー分割

## テスト結果

- **追加**: 18テスト
- **全体**: 2251テスト（fast: 1690 / slow: 343 / deprecated: 218）
- lint: ruff check + format 通過
- 既存テスト1件（test_displacement_consistency）は本変更と無関係の既知失敗

## 影響ファイル

### 新規
- `tests/test_coating_sheath_physics.py` — 被膜/シース物理テスト18件
- `docs/verification/twisted_wire_cross_section.png` — 撚線断面3D構造図
- `docs/verification/fiber_stress_cross_section.png` — 断面繊維応力コンター図
- `docs/status/status-120.md` — 本statusファイル

### 変更
- `tests/generate_verification_plots.py` — 2プロット関数追加
- `README.md` — テスト数更新（2233→2251）
- `CLAUDE.md` — テスト数更新
- `docs/roadmap.md` — テスト数・到達点更新
- `docs/status/status-index.md` — status-120追加

## TODO（継続）

- Cosserat Rodの解析的接線剛性実装
- Cosserat Rodの疎行列アセンブリ実装
- Cosserat Rodの質量行列実装
- NCP版摩擦バリデーションテスト
- NCP版ヒステリシステスト
- NCP版曲げ揺動ベンチマークテスト
- CR vs Cosserat 単線比較（Phase 1比較テスト）
- 3D表面レンダリング（Axes3D pipe rendering）の実装検討

## 確認事項

- 被膜は常に剛性を増加させ、その寄与はE比に比例する（物理的に妥当）
- シースは全撚線本数(3/7/19)で素線を正しく囲む
- 断面繊維応力分布は曲げにより上下対称な引張/圧縮分布を示す
- 検証プロット11種（9→11に増加）で視覚的妥当性も確認済み
- CJKフォント未インストール環境では日本語タイトルが表示されない（機能に影響なし）

---
