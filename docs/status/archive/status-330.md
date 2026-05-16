# status-330: ファイバー梁 Phase F5 — StrandBendingOscillationProcess に use_fiber_beam フラグ統合

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-14
- **ブランチ**: `claude/check-status-todos-8TINW`
- **テスト数**: 459+13+22+5+8+12+12+25+26+10（Phase F5: API 4 + Physics 6 テスト追加）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

## TL;DR

status-329 TODO「ファイバー梁 Phase F5 着手 — ContactFrictionProcess へのファイバー梁統合」を実行。`StrandBendingOscillationProcess` に `use_fiber_beam: bool = False` フラグを追加し、`True` のとき素線メッシュの代わりに1本のファイバー梁として解くモードを実装。弾性材料で先端変位が理論値と **0.02% 一致**、BilinearKH / MultiLayerFriction 材料で **NR 収束合格**。10 テスト全合格、契約違反 0 件。

---

## 1. use_fiber_beam フラグ実装

### 概要

`StrandBendingOscillationConfig` にファイバー梁モード用パラメータを追加し、`process()` メソッドで `use_fiber_beam=True` のとき `_process_fiber_beam` メソッドに分岐する。

### 追加パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|---|---------|------|
| `use_fiber_beam` | bool | False | ファイバー梁モード有効化 |
| `fiber_material` | object\|None | None | 1D材料則（None=Elastic1D） |
| `fiber_section_type` | str | "strip" | 断面離散化方式 |
| `fiber_n_fiber` | int | 60 | ファイバー数 |
| `fiber_n_theta` | int | 16 | polar離散化の周方向分割数 |

### ファイル変更

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | Config にフラグ追加、`_process_fiber_beam` メソッド新規、`_static_nr_solve` に TL モード追加 |
| `xkep_cae/elements/_beam_assembler.py` | `ULCRFiberBeamAssembler.__init__` で `material.initial_state()` 対応 |
| `tests/numerical_tests/test_fiber_beam_integration.py` | 10テスト新規作成 |

---

## 2. _process_fiber_beam メソッド

### パイプライン

1. **直線梁メッシュ生成**: n_elements 個の2ノード要素（x軸方向）
2. **ファイバー断面生成**: `CircularFiberSection.strip()` or `.polar()`
3. **ULCRFiberBeamAssemblerProcess** でアセンブラ構築
4. **境界条件**: 左端全DOF固定 + 右端θ_y処方 + 全ノード u_y/θ_z 拘束（x-z面曲げ）
5. **_static_nr_solve** で静的NR求解

### 設計判断

- **接触なし**: ファイバー梁の内部摩擦はセクション積分で処理。ContactFrictionProcess は不要
- **x-z 面拘束**: strip 断面では `EI_z = 0`（z座標ゼロ）のため u_y/θ_z の剛性がゼロ。全ノードで拘束して特異行列を回避
- **TL 定式化**: 非線形材料では `update_reference` を呼ばない。CR梁ULのf_int=0問題（eps_p/alpha の参照枠不整合）を回避

---

## 3. _static_nr_solve TL モード

### 変更内容

`use_ul` パラメータを追加（デフォルト True = 従来の UL 動作）。

| | UL モード (`use_ul=True`) | TL モード (`use_ul=False`) |
|---|---|---|
| 変位追跡 | 増分 u_incr（毎ステップゼロリセット） | 全変位 u_total（累積） |
| 処方変位 | 増分: `(frac_target - frac) * values` | 全量: `frac_target * values` |
| 収束後 | `update_reference(u_incr)` | `u_total = u_incr` |
| カットバック | `rollback()` | `rollback()` + `u_total` 復元 |
| 出力変位 | `assembler.u_total_accum` | `u_total` |

### 弾性材料は UL、非線形材料は TL

```python
is_nonlinear = not isinstance(material, Elastic1D)
use_ul = not is_nonlinear
```

---

## 4. ULCRFiberBeamAssembler 初期状態修正

### 問題

`MultiLayerFrictionDegrading1D` は `n_layers` 個の `slip/slipped` タプルを状態に持つ。
従来の `Fiber1DState()` デフォルト（空タプル）では `slipped[i]` でインデックスエラー。

### 修正

```python
if hasattr(material, "initial_state"):
    fiber_init = material.initial_state()
else:
    fiber_init = Fiber1DState()
```

材料に `initial_state()` メソッドがあれば使用。MultiLayerFriction では `n_layers` 分の初期状態を生成。

---

## 5. テスト結果

### API テスト（4件）

| テスト | 内容 | 結果 |
|--------|------|------|
| `test_fiber_beam_returns_result` | use_fiber_beam=True で結果が返る | **合格** |
| `test_fiber_beam_mesh_is_single_beam` | メッシュが単一直線梁 | **合格** |
| `test_fiber_beam_bending_angle` | 曲げ角度 = κ×L | **合格** |
| `test_fiber_beam_polar_section` | polar 離散化で動作 | **合格** |

### Physics テスト（6件）

| テスト | 精度 | 結果 |
|--------|------|------|
| `test_elastic_fiber_beam_converges` | frac=1.0, NR 1回/step | **合格** |
| `test_elastic_tip_displacement_matches_theory` | 理論値 0.02% 誤差 | **合格** |
| `test_elastic_fiber_matches_contact_mode` | 接触なしモードと一致 | **合格** |
| `test_bilinear_kh_fiber_beam_converges` | frac=1.0, NR 2-6回/step | **合格** |
| `test_multilayer_friction_fiber_beam_converges` | frac=1.0 完走 | **合格** |
| `test_nonlinear_tip_displacement_less_than_elastic` | ||diff|| = 7.2mm | **合格** |

### 回帰チェック

- Phase F1-F4 テスト 75 件: **全合格**
- 契約違反: **0 件**
- lint/format: **OK**

---

## TODO（次担当者向け）

### 直近

- [ ] **Phase F5 散逸エネルギー検証** — 7本撚線接触モデルとファイバー梁モデルの散逸エネルギー一致 < 10% を確認（Phase F5 の完了判定基準）
- [ ] **Phase F6 キャリブレーション Process 着手** — MultiLayerFriction のパラメータを接触モデルの M-κ ループから自動推定
- [ ] **被膜 ON プロファイル + pypardiso 環境再ベンチ** — status-326 TODO 継続
- [ ] **n=61+ Type D stall 対策** — 大 n_strands での接触活性化時に Type D stall が発生

### 中期

- [ ] **リスタート解析方式への移行**: ContactFrictionProcess の I/O を `(u, v, a, 接触ペア)` 入出力に整理（CR梁ULのf_int=0問題の根本解決）
- [ ] **空間ブロック分離 or ペアクラスタリング**: 物理的接触ペア数の n² 成長を抑制する構造的対策

### 設計上の懸念

- **TL定式化の大回転制限**: 非線形材料でTL（update_referenceなし）を使うため、大回転問題ではCR定式化の精度が劣化する可能性。現状のテスト（κ=0.001〜0.002、小変形範囲）では問題なし。大曲率での検証が必要
- **f_int=0問題**: UL+非線形材料の根本解決はリスタート解析方式への移行が必要。TLは暫定対策

---

## 再現手順

```bash
# Phase F5 テスト
pytest tests/numerical_tests/test_fiber_beam_integration.py -v

# 全ファイバーテスト（Phase F1-F5: 85 件）
pytest xkep_cae/elements/fiber/tests/ tests/numerical_tests/test_fiber_beam_integration.py -v

# 契約検証
python contracts/validate_process_contracts.py
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: pytest -v 出力 10 件全合格（API 4 + Physics 6）
- [x] **再現手順記載**: 上記
- [x] **テスト数記載**: 459+13+22+5+8+12+12+25+26+10
- [x] **契約違反 0 件維持**: validate_process_contracts.py 実行済み
- [x] **lint/format 検証**: ruff check + ruff format OK
