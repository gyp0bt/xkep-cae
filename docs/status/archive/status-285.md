# status-285: C16修正 + 凍結テスト + Hertz型非線形ペナルティ — frac 0.70→0.998

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-03
- **ブランチ**: `claude/check-status-todos-eunyp`
- **テスト数**: 621 passed（+15: RebuildMPC 3件 + 凍結Config 4件 + Hertz 8件）
- **契約違反**: **0件**（C16修正完了）
- **条例違反**: 0件

---

## 概要

status-284のTODO3件を対応:
1. **C16契約違反修正**: `rebuild_mpc_transform` 純粋関数を `RebuildMPCTransformProcess` に変換
2. **凍結モードの単体テスト追加**: パラメータ設定テスト + パイプライン検証
3. **Hertz型非線形ペナルティ実装**: `penalty_exponent` パラメータ追加（frac=0.70→1.0対策の基盤）

---

## 実装内容

### 1. C16契約違反修正: `rebuild_mpc_transform` Process化

`rebuild_mpc_transform()` 純粋関数を `RebuildMPCTransformProcess` に変換。

| 変更 | 詳細 |
|------|------|
| `RebuildMPCTransformInput` | frozen dataclass: mpc_groups, node_coords, ndof_total, ndof_per_node |
| `RebuildMPCTransformProcess` | PreProcess として実装 |
| `ContactFrictionProcess.uses` | `RebuildMPCTransformProcess` を追加 |
| `__init__.py` | エクスポート追加 |

### 2. 凍結モード単体テスト

| テストクラス | 内容 |
|-------------|------|
| `TestRebuildMPCTransformProcessAPI` | C3紐付け + API検証 3件 |
| `TestChatteringFreezeConfig` | 凍結パラメータデフォルト値・カスタマイズ・パイプライン 4件 |

### 3. Hertz型非線形ペナルティ

既存Huber平滑化ペナルティに **べき乗則** を追加:

```
現在（α=1.0）: p_n = huber(k_pen × δ_pen, δ_h)
Hertz（α=1.5）: p_n = huber(k_pen × δ_pen, δ_h)^α / k_pen^{α-1}
```

- α=1.0 で完全に既存と同一（デフォルト）
- α=1.5 で Hertz 接触モデル（p_n ∝ δ^{1.5}）
- Huber C1 平滑化を維持しつつ、接触ON/OFF境界で力が緩やかに立ち上がる

**導関数**: `dp/dg = α × (h/k_pen)^{α-1} × h'(x) × (-k_pen)`

### パイプライン貫通

| レイヤー | パラメータ |
|---------|----------|
| `ContactFrictionInputData` | `penalty_exponent: float = 1.0` |
| `default_strategies()` | `penalty_exponent` 引数追加 |
| `_create_contact_force_strategy()` | `penalty_exponent` 転送 |
| `HuberContactForceProcess.__init__()` | `penalty_exponent` 保持 |
| `evaluate()` | `_apply_power_law()` 適用 |
| `tangent()` | `_apply_power_law_deriv()` 適用 |
| `ContactForceStStiffnessInput` | `penalty_exponent` フィールド追加 |
| K_st 計算 | スカラー版導関数補正 |

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/constraints/mpc_elimination.py` | `RebuildMPCTransformProcess` 新規 |
| `xkep_cae/constraints/__init__.py` | エクスポート追加 |
| `xkep_cae/contact/solver/process.py` | uses追加 + import変更 |
| `xkep_cae/contact/contact_force/strategy.py` | Hertz型ペナルティ実装 |
| `xkep_cae/core/data.py` | `penalty_exponent` パイプライン貫通 |
| `xkep_cae/constraints/tests/test_mpc_elimination.py` | テスト追加 |
| `xkep_cae/contact/solver/tests/test_process.py` | 凍結テスト追加 |
| `xkep_cae/contact/contact_force/tests/test_strategy.py` | Hertzテスト追加 |

---

## 技術的要点（次の担当者向け）

### Hertz型ペナルティの使い方

```python
cfg = StrandBendingOscillationConfig(
    ...,
    penalty_exponent=1.5,  # Hertz型
)
```

または `ContactFrictionInputData(penalty_exponent=1.5)` で設定。

### ベンチマーク結果

#### 7本ヘリカル撚線90度曲げ（κ=π/200, θ=π/2, contact_enabled=True）

| 構成 | frac | incr | cutback | 備考 |
|------|------|------|---------|------|
| ベースライン（status-284, α=1.0） | **0.7050** | 570 | 66 | チャタリング凍結停滞 |
| **Hertz型（α=1.5）** | **0.9981** | 551 | 60 | **事実上完走（42%改善）** |
| 接触なし（参考, status-281） | 1.0000 | 102 | 6 | — |

**Hertz型ペナルティで接触あり90度曲げが事実上完走。**

- incr=551 はベースライン（570）より少ない → 効率的
- cutback=60 もベースライン（66）より少ない → チャタリング低減
- frac=0.9981（99.8%）は θ ≈ 89.8° に相当

#### なぜ効果的か

1. Hertz型 `p_n ∝ δ^{1.5}` により、接触ON/OFF境界で力が連続的に立ち上がる
2. 活性集合のON/OFF離散切替が物理的に平滑化 → NR内チャタリングが大幅減
3. 凍結モードとの組み合わせで、残存するチャタリングも対処
4. gap=0付近の接線剛性ゼロの懸念は、Huber C1平滑化で相殺

### 次のステップ

- FD接線診断でHertz型の接線剛性が正しいか検証
- α最適値探索（1.0, 1.2, 1.5 比較）
- penalty_exponent=1.5 をデフォルトにするかの検討

---

## 再現手順

```bash
git checkout claude/check-status-todos-eunyp
pip install -e .

# 回帰テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 \
  --ignore=tests/contact/test_st_jacobian.py \
  -k "not slow and not stress_contour" 2>&1 | tee /tmp/log-regression-285.log
# 期待値: 621 passed

# 契約検証
python contracts/validate_process_contracts.py
# 期待値: 契約違反なし

# Hertz型テストのみ
python -m pytest xkep_cae/contact/contact_force/tests/test_strategy.py -q -k "Hertz" 2>&1 | tee /tmp/log-hertz-test.log
# 期待値: 8 passed
```

---

## STA2 準拠チェック

- [x] **tee ログ保存**: `/tmp/log-hertz-1.5-90deg.log`, `/tmp/log-baseline-1.0-90deg.log`
- [x] **再現手順記載**: 全コマンドをstatusに記載
- [x] **数値の捏造なし**: テスト 621 passed、frac=0.9981 は tee ログと一致
- [x] **ベースライン先行取得**: status-284 frac=0.7050 をベースライン使用
- [x] **回帰なし**: 621 passed, 0 failed

---

## TODO

- [x] Hertz型ペナルティのベンチマーク（90度曲げ penalty_exponent=1.5）→ **frac=0.9981**
- [x] penalty_exponent の StrandBendingOscillationConfig パイプライン貫通
- [ ] FD接線診断でHertz型の接線剛性整合性検証
- [ ] α最適値探索（1.0, 1.2, 1.5 比較）
- [ ] penalty_exponent=1.5 をデフォルトにするか検討
- [ ] frac=0.9981→1.0 の最後の0.2%を詰めるか判断

---
