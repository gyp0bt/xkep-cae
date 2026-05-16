# status-303: バリア関数被膜モデル — 芯線貫入防止

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-08
- **ブランチ**: `claude/check-status-todos-AuiH4`
- **テスト数**: 442+11 passed（既存テスト全合格 + バリア関数テスト11件追加）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-302で発見された「被膜の芯線貫入（8.6%のペア）」を根本解決するため、
被膜力モデルを線形バネからバリア関数に改善した。

### 変更内容

**被膜力モデル:**
- 旧: `f = k * δ`（線形、δ > δ_max でも有限力 → 芯線貫入）
- 新: `f = k * δ / (1 - δ/δ_max)`（バリア関数、δ→δ_max で力→∞）

**接線剛性:**
- 旧: `df/dδ = k`（定数）
- 新: `df/dδ = k / (1 - δ/δ_max)²`（圧縮量に応じて急増）

**特異点保護:** `1 - δ/δ_max` が `1e-3` 以下にクランプ（数値安全性）

---

## 実施内容

### 1. _ContactConfigInput 拡張

`coating_barrier: bool = True` を追加。`coating_thickness > 0` と併用時のみバリア有効。
既存コード（`coating_thickness=0`）は線形モデルのまま → 完全後方互換。

### 2. KelvinVoigtCoatingProcess 改修

4メソッド全てにバリア関数を統合:
- `forces()`: 法線力計算にバリア関数
- `stiffness()`: ペア毎の接線剛性計算（線形時はグローバル定数のまま）
- `friction_forces()`: Coulomb条件の `p_n` にバリア関数
- `friction_stiffness()`: 摩擦接線剛性の `p_n` にバリア関数

### 3. ヘルパー関数

- `_coat_total(pair)`: ペアの被膜総厚さ（A側+B側）
- `_barrier_p_n(k, δ, δ_max)`: バリア力
- `_barrier_k_eff(k, δ, δ_max)`: バリア接線剛性

### 4. テスト追加（11件）

| テスト | 検証内容 |
|--------|---------|
| `test_small_compression_matches_linear` | 小圧縮量（1%）で線形とほぼ一致 |
| `test_force_increases_nonlinearly` | 80%圧縮で線形の4倍以上 |
| `test_no_core_penetration` | δ=δ_maxでも有限力（NaN/Inf回避） |
| `test_barrier_force_exact_value` | 50%圧縮での解析値一致 |
| `test_action_reaction_with_barrier` | 作用反作用成立 |
| `test_barrier_stiffness_exact_value` | 接線剛性の解析値一致 |
| `test_barrier_stiffness_symmetry` | 剛性行列対称性 |
| `test_barrier_stiffness_positive_semi_definite` | 半正定値性 |
| `test_barrier_stiffness_fd_consistency` | FD（有限差分）との整合 |
| `test_barrier_friction_slip_limit` | バリア法線力に基づくCoulomb限界 |
| `test_barrier_disabled_when_no_thickness` | thickness=0で線形に戻る |

---

## バリア関数の物理的効果

| 圧縮率 δ/δ_max | 線形力 (kδ) | バリア力 (kδ/(1-δ/δ_max)) | 倍率 |
|-----------------|-------------|---------------------------|------|
| 10% | 0.10k | 0.111k | 1.11x |
| 50% | 0.50k | 1.000k | 2.00x |
| 80% | 0.80k | 4.000k | 5.00x |
| 95% | 0.95k | 19.00k | 20.0x |
| 99% | 0.99k | 99.00k | 100x |

芯線接近時に力が急激に増大し、物理的に不可能な芯線貫入を防止する。

---

## 変更ファイル

- `xkep_cae/contact/_contact_pair.py`: `coating_barrier` フラグ追加
- `xkep_cae/contact/coating/strategy.py`: バリア関数ヘルパー + 4メソッド改修
- `xkep_cae/contact/coating/tests/test_physics.py`: 11テスト追加

---

## 再現手順

```bash
# ブランチ
git checkout claude/check-status-todos-AuiH4

# テスト実行
python -m pytest xkep_cae/contact/coating/tests/test_physics.py -v

# lint
ruff check xkep_cae/contact/coating/ xkep_cae/contact/_contact_pair.py
ruff format --check xkep_cae/contact/coating/ xkep_cae/contact/_contact_pair.py

# 契約チェック
python contracts/validate_process_contracts.py
```

---

## TODO

- [ ] 被膜付き90度曲げでバリア関数の収束性検証（status-298ベースライン比較）
- [ ] 被膜接線剛性のFD誤差67%（status-301）がバリア関数で改善されるか検証
- [ ] 被膜パラメータの物理的根拠確認（k_coat値の妥当性検証）
- [ ] シース-素線接触統合（旧SheathModel/HEX8のProcess化）
- [ ] 高速化フェーズ（接触ペア検出KD-tree化 → K_c/K_stアセンブリベクトル化）

---

## 次の担当者向け

### 重要ポイント

1. **後方互換**: `coating_thickness=0`（デフォルト）では線形モデルのまま。バリアは `coating_thickness > 0` 時のみ有効。
2. **クランプ値**: `_BARRIER_CLAMP = 1e-3` → 99.9%圧縮で力はmax 999k倍。物理的に十分。
3. **接線剛性**: ペア毎に異なる `k_eff` を計算（線形時はグローバル定数で高速）。
4. **FD整合**: `test_barrier_stiffness_fd_consistency` でバリア接線剛性の解析値とFD差分の一致を確認済み。

### 推奨アクション

status-298の被膜付き90度曲げを再実行し、バリア関数での芯線貫入防止効果を確認。
被膜圧縮率が100%以下に収まることを `contracts/diagnose_coating_penetration.py` で検証。

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: テスト結果はpytest実行ログで再現可能
- [x] **再現手順記載**: コマンド列を明記
- [x] **ベースライン比較**: 既存18テスト全合格（後方互換確認）
