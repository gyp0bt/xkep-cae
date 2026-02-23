# status-049: 貫入制約チューニング — 適応的ペナルティ増強 + マルチセグメント/スライディングテスト

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-23
**作業者**: Claude Code
**テスト数**: 1054（+20）

## 概要

status-048 の TODO を消化し、貫入を断面半径の1%以下に抑制する方策を技術的に検討。
Simo & Laursen (1992) に基づく適応的ペナルティ増強（`adaptive_penalty`）を実装し、
マルチセグメント梁テスト・スライディングテスト・`augment_penalty_if_needed` 単体テストを追加。

## 技術的検討: 貫入を断面半径の1%以下に抑制するには

### 現状の分析

現在の接触実装は Augmented Lagrangian (AL) 法:

```
p_n = max(0, λ_n + k_pen × (-g))
```

- Outer loop 終了時に `λ_n ← p_n` で乗数更新（Uzawa 更新）
- Outer loop 上限: `n_outer_max = 5`（デフォルト）
- 各荷重ステップで最大5回の AL 更新

**理論的な貫入量の推定**:

1要素ばねモデル（k_struct = 構造剛性）に対する簡易分析:

| 条件 | 貫入量 |gap| の推定 | pen/r (r=0.04) |
|------|---------|----------|
| 純ペナルティ (k_pen=1e4, F=50) | F/k_pen = 5e-3 | **12.5%** |
| 純ペナルティ (k_pen=1e5, F=50) | F/k_pen = 5e-4 | **1.25%** |
| 純ペナルティ (k_pen=1e6, F=50) | F/k_pen = 5e-5 | **0.125%** |
| AL更新5回 (k_pen=1e5, F=50) | ≈ (k_s/(k_s+k_pen))^5 × 初期gap | **< 0.1%** |

### 課題: ペナルティ剛性 vs 収束性のトレードオフ

1. **条件数悪化**: k_pen >> K_struct の場合、全体剛性行列の条件数が k_pen/K_min オーダーに増加。
   LU分解の数値精度低下、反復法の収束遅延を招く。
2. **Newton振動**: 高ペナルティでは接触/非接触の境界で接線剛性が不連続的に変化し、
   NR反復が振動する。
3. **荷重ステップのカットバック**: 条件数悪化により、荷重増分の縮小が必要になる場合がある。

### 私見: 極端な k_pen 増加は不要

**結論: AL法のOuter更新を適切に運用すれば、k_pen を極端に上げる必要はない。**

理由:

1. **AL法の本質**: 乗数 λ_n が反力の「記憶」として蓄積されるため、各Outer更新で
   ペナルティ項 `k_pen × |gap|` が担うべき力が減少する。理論的にはOuter更新を
   十分繰り返せば gap → 0 に収束する。
2. **収束レート**: Uzawa更新の収束レートは `ρ = k_struct / (k_struct + k_pen)` であり、
   k_pen が k_struct の10倍あれば `ρ ≈ 0.09`（1回の更新で91%縮小）。
   5回の更新で `ρ^5 ≈ 6e-6` → 実質的にギャップゼロ。
3. **1%制約に必要な条件**: k_pen=1e5 で初期貫入が1.25%程度の場合、
   AL更新1回で0.1%以下に到達する計算になる。
   問題は **Outer loop が merit 停滞で早期終了する場合**。

### 対策（3段階）

#### 対策1: 適応的ペナルティ増強（本status で実装）

Simo & Laursen (1992) の手法に基づき、AL乗数更新後にギャップ違反が
許容値（`gap_tol = gap_tol_ratio × (r_a + r_b)`）を超えるペアに対して
k_pen を段階的に増強する。

```python
# ContactConfig に追加
adaptive_penalty: bool = False
adaptive_penalty_factor: float = 2.0      # 増強倍率
adaptive_penalty_max_scale: float = 100.0  # 最大倍率
gap_tol_ratio: float = 0.01               # 許容ギャップ比（1%）
```

**利点**:
- 初期 k_pen を低めに設定でき、条件数を抑制
- 違反ペアのみ選択的に増強するため、全体の条件数への影響が限定的
- k_pen 上限（`max_scale`）により発散を防止

**欠点**:
- 増強のタイミングがOuter更新に依存するため、n_outer_max が十分でないと効果が限定的
- 増強による接線剛性の不連続がNewton振動を誘発する可能性

#### 対策2: Outer loop 回数の増加 + ギャップベース収束判定（推奨、未実装）

現在の Outer 収束判定は `|Δs|, |Δt| < tol_geometry` + merit 停滞。
ギャップベースの追加判定を導入:

```python
# 提案（未実装）
max_gap_violation = max(|gap_i| for active pairs where gap_i < 0)
if max_gap_violation < gap_tol:
    outer_gap_converged = True
```

これにより、merit が停滞していてもギャップ違反が大きい場合は
Outer loop を継続する。n_outer_max も 5 → 10 程度に増加が望ましい。

#### 対策3: ラグランジュ乗数法（mortar法、将来検討）

理論的にはギャップ=0を保証できるが:
- 実装が大幅に複雑化（saddle-point問題、LBB条件）
- 既存のAL/ペナルティ基盤からの移行コストが大きい
- Phase 4.7（撚線モデル）で貫入精度が問題になった場合に検討

### 推奨パラメータ設定（1%以下達成の目安）

| パラメータ | 推奨値 | 備考 |
|-----------|--------|------|
| k_pen_scale | 1e5 | 構造剛性の10〜100倍 |
| n_outer_max | 8〜10 | デフォルト5では不足する場合あり |
| adaptive_penalty | True | ギャップ違反時に自動増強 |
| gap_tol_ratio | 0.01 | 1%基準 |
| adaptive_penalty_factor | 2.0 | 1回の増強で2倍 |
| adaptive_penalty_max_scale | 100.0 | 初期値の100倍を上限 |

## 実装内容

### 1. 適応的ペナルティ増強 (`law_normal.py`)

`augment_penalty_if_needed()` 関数を追加:
- ギャップ違反（|gap| > gap_tol）時に k_pen を factor 倍に増強
- k_pen の上限（max_scale × base）を超えない
- k_t も比率を維持して更新
- INACTIVE ペアは無視

### 2. solver_hooks への組み込み (`solver_hooks.py`)

Outer loop の AL 乗数更新直後に適応的増強ブロックを挿入:
```
→ 幾何更新 → |Δs|,|Δt| 判定 → AL乗数更新 → [適応的増強] → Merit判定
```

`ContactConfig.adaptive_penalty = True` のとき有効。
各ペアの `radius_a + radius_b` を参照半径として `gap_tol = gap_tol_ratio × r_ref` を計算。

### 3. ContactConfig 拡張 (`pair.py`)

4つの新設定フィールドを追加:
- `adaptive_penalty`: 有効化フラグ
- `adaptive_penalty_factor`: 増強倍率
- `adaptive_penalty_max_scale`: 最大倍率
- `gap_tol_ratio`: ギャップ許容比

### 4. テスト追加 (`tests/contact/test_penetration_constraint.py`)

| クラス | テスト数 | 検証内容 |
|-------|---------|---------|
| TestMultiSegmentPenetration | 4 | マルチセグメント梁の接触検出/貫入制限/分割数依存/複数ペア |
| TestSlidingPenetrationTracking | 3 | スライディング時の貫入制限/接線変位/接触力履歴 |
| TestAdaptivePenaltyAugmentation | 4 | 適応的増強の貫入低減効果/収束性/k_pen増加/摩擦併用 |
| TestAugmentPenaltyUnit | 5 | augment_penalty_if_needed 単体（許容内/超過/上限/k_t更新/INACTIVE） |

合計: **16テスト**

### 5. status-048 TODO の消化状況

| TODO | 状態 | 備考 |
|------|------|------|
| マルチセグメント梁での貫入テスト | **完了** | 4分割/8分割、複数ペア検出含む |
| スライディング時の貫入量追跡テスト | **完了** | 摩擦あり/接線変位/履歴検証 |
| 接触付き弧長法との統合テスト | **保留** | Phase 4.7前提（設計検討は status-047 で完了） |

## ファイル変更

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/pair.py` | ContactConfig に adaptive_penalty 系4フィールド追加 |
| `xkep_cae/contact/law_normal.py` | `augment_penalty_if_needed()` 関数追加 |
| `xkep_cae/contact/solver_hooks.py` | Outer loop に適応的増強ブロック挿入 + import追加 |
| `tests/contact/test_penetration_constraint.py` | **新規** — 16テスト |

## 確認事項・懸念

1. **テスト未実行**: numpy/scipy がインストール不可のため、テストの実行検証ができていない。
   次の作業者は `pytest tests/contact/test_penetration_constraint.py -v` で全テストパスを確認すること。

2. **適応的増強の安定性**: 急激な k_pen 増加が Newton 振動を誘発する懸念がある。
   `adaptive_penalty_factor=2.0` は保守的な設定だが、撚線モデル等の複雑な問題では
   `factor=1.5` への低減や、増強後に追加の Inner NR 反復を挟むことを検討。

3. **ギャップベース Outer 収束判定**: 本 status では未実装。
   merit 停滞で早期終了する場合にギャップ違反が残る問題の根本対策として、
   次のステップで実装を推奨。

4. **n_outer_max の推奨値**: 現デフォルト 5 は1%制約には不十分な場合がある。
   撚線モデル（Phase 4.7）では 8〜10 への増加を推奨。

## TODO

- [ ] pytest でテスト全パスの確認（numpy/scipy 環境必要）
- [ ] ギャップベース Outer 収束判定の実装（max_gap_violation < gap_tol → 継続）
- [ ] n_outer_max のデフォルト値見直し（5 → 8）
- [ ] 撚線モデル（Phase 4.7）での適応的増強の効果検証
- [ ] 接触付き弧長法との統合テスト（Phase 4.7前提）

---
