# status-137: gap_offset廃止 + 被膜厚考慮メッシュ配置 + 被膜接触モデル再構築

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-08
**テスト数**: 2271（変更なし）

---

## 概要

status-136で指摘した `gap_offset`（LS-DYNA IGNORE=1相当）の物理的妥当性問題を解決。
根本原因はメッシュ生成時に被膜厚を考慮していなかったこと。
以下3つの対策を実施:

1. **メッシュ生成の被膜厚考慮**: `make_strand_layout` に `coating_thickness` パラメータを追加し、素線配置を `d_eff = 2*(wire_radius + coating_thickness)` で計算
2. **gap_offset手法の完全廃止**: `store_initial_offsets()` を `check_initial_penetration()` に置換。初期貫入はエラーとして検出する（隠蔽しない）
3. **被膜接触モデルの再構築**: コア半径ベースのギャップ定義 + 被膜弾性スプリングモデル（案1実装）

---

## 変更内容

### 1. メッシュ生成: 被膜厚考慮配置（根本原因修正）

**ファイル**: `xkep_cae/mesh/twisted_wire.py`

問題: `make_strand_layout` が `d = 2*wire_radius` で素線間距離を計算していたため、被膜付き素線で初期貫入 = `2*coating_thickness` が発生していた。

修正:
```python
def make_strand_layout(
    n_strands, wire_radius, *, gap=0.0, lay_direction=1,
    strand_diameter=None, coating_thickness=0.0,  # ← 追加
):
    d = 2.0 * wire_radius
    d_eff = 2.0 * (wire_radius + coating_thickness)  # ← 被膜厚考慮
    # 以下全ての配置計算で d_eff を使用
```

3本/伝統的配置/strand_diameter配置の全モードで `d_eff` を適用。

### 2. gap_offset完全廃止

**ファイル**: `xkep_cae/contact/pair.py`

| 旧 | 新 |
|----|-----|
| `ContactPair.gap_offset: float` | 削除 |
| `store_initial_offsets()` | `check_initial_penetration()` — 貫入数を返すのみ |
| `update_geometry()` でオフセット減算 | オフセットなし（物理ギャップをそのまま使用） |

### 3. 被膜接触モデル（案1: 被膜弾性スプリング）

**ファイル**: `xkep_cae/contact/pair.py`, `xkep_cae/contact/solver_ncp.py`

- `ContactPair` に `core_radius_a`, `core_radius_b` フィールド追加
- `ContactState` に `coating_compression` フィールド追加
- `ContactConfig` に `coating_stiffness` フィールド追加
- ギャップ定義: `g = dist - (r_core_a + r_core_b)` （コア半径のみ）
- 被膜圧縮: `coat_comp = max(0, coat_total - gap_core)`
- 被膜スプリング力: `f_coat = coating_stiffness * coat_comp`
- NCP: コア同士の接触のみ（被膜は弾性層として別計算）

### 4. テスト・スクリプト修正

| ファイル | 変更内容 |
|---------|---------|
| `tests/contact/test_friction_validation_ncp.py` | z_sep=0.035→0.04（初期貫入解消） |
| `xkep_cae/numerical_tests/wire_bending_benchmark.py` | `store_initial_offsets` → `check_initial_penetration` |
| `scripts/diagnose_7wire_convergence.py` | 同上 |
| `scripts/verify_coating_gap.py` | 新規: 被膜厚考慮メッシュ配置の物理的妥当性検証 |

---

## 検証結果

`scripts/verify_coating_gap.py` による4ケース検証:

| ケース | 結果 |
|--------|------|
| 被膜なし gap=0 | 弦近似貫入のみ（0.48%、32要素/ピッチ） ✅ |
| 被膜あり coat指定 | 弦近似貫入のみ（0.48%） ✅ |
| 被膜あり coat未指定（旧動作） | 0.4000mm貫入 = 2*coat_t ✅（問題を正しく検出） |
| 配置半径比較 | r_with - r_no = 2*coat_t ✅ |

---

## status-136 TODO消化状況

- [x] gap_offset手法の完全廃止（LS-DYNA IGNORE=1相当の撤去）
- [x] 被膜層を陽に考慮した接触ギャップ定義の実装
- [x] 被膜の弾性構成則に基づく接触力計算
- [x] 被膜付き撚線テストの物理的妥当性再検証（verify_coating_gap.py）
- [ ] status-098, 120の被膜パラメータスタディの再実施（次回TODO）

---

## TODO

- [ ] status-098, 120の被膜パラメータスタディの再実施（被膜厚考慮メッシュ + 被膜スプリングモデルで再計測）
- [ ] 被膜スプリングモデルの実問題での収束検証（7本被膜付き曲げ揺動）
- [ ] CI重量テスト（mortar_twisted_wire, ncp_convergence_19strand）の通過確認

---

## 前回status

- [status-136](status-136.md): 初期貫入オフセット手法の物理的妥当性問題（要やり直し）

---
