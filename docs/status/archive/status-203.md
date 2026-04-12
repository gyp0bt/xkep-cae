# status-203: C17 例外リスト廃止 + replace() 検知追加

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-18
**テスト数**: ~2260 + 374 新パッケージテスト
**契約違反**: C17: 3件（non-frozen dataclass）| O2: 2件（警告）

---

## 概要

C17 の `_KNOWN_NON_FROZEN` 例外リストを廃止。
違反は違反として正直に報告する方式に変更。
併せて `dataclasses.replace()` の使用を C17 違反として検知するルールを追加。

**方針転換**: frozen 化に `replace()` を使うのは構造的な不変設計ではない。
mutable な dataclass を frozen + replace() に書き換えるだけでは本質的な解決にならない。
Process が Input を受け取り Output を返す構造改革が必要。

## 変更内容

### 1. `_KNOWN_NON_FROZEN` 例外リスト廃止

`scripts/validate_process_contracts.py` の C17 チェックから既知例外スキップを完全除去。
3件の non-frozen dataclass が正規の契約違反として報告される。

### 2. `dataclasses.replace()` 検知ルール追加

C17 チェックにプライベートモジュール内の `dataclasses.replace()` 使用検知を追加:

- `dataclasses.replace(...)` 直接呼び出し
- `from dataclasses import replace` 経由の `replace(...)` 呼び出し

frozen 化の代替としての `replace()` は構造的な不変設計にならないため違反とする。

### 3. status-202 の「C17 0件（既知例外3件あり）」表記是正

実態は違反3件。例外で隠していただけ。

## 現在の C17 違反一覧（3件）

| # | ファイル | クラス | 違反内容 | 変異箇所数 | 解消方針 |
|---|---------|--------|---------|-----------|---------|
| 1 | `contact/_contact_pair.py` | `_ContactStateOutput` | non-frozen | 50+ | Process 出力として新インスタンスを返す設計に変更 |
| 2 | `contact/_contact_pair.py` | `_ContactPairOutput` | non-frozen | ~10 | state フィールドの再代入を Process 出力に変換 |
| 3 | `contact/_contact_pair.py` | `_ContactManagerInput` | non-frozen | ~5 | pairs リスト変異を Process API に移行 |

### 変異箇所の詳細

#### `_ContactStateOutput` 変異箇所（50+件）

| ファイル | 変異内容 |
|---------|---------|
| `contact/_contact_pair.py:406-414` | s, t, gap, normal, tangent1, tangent2, coating_compression（update_geometry） |
| `contact/_contact_pair.py:431-436` | status（_update_active_set） |
| `contact/_contact_pair.py:443-444` | k_pen, k_t（initialize_penalty） |
| `contact/coating/strategy.py:234-237` | coating_z_t, coating_stick, coating_q_trial_norm, coating_dissipation（リセット） |
| `contact/coating/strategy.py:269-272` | coating_z_t, coating_stick, coating_q_trial_norm, coating_dissipation（更新） |
| `contact/solver/process.py:500` | coating_compression_prev |
| `contact/solver/process.py:539-540` | lambda_n, p_n |
| `contact/friction/_assembly.py:199-200` | k_pen, k_t |
| `contact/friction/_assembly.py:211-218` | z_t, stick, q_trial_norm, dissipation, status |
| `contact/friction/strategy.py:196` | p_n |
| `contact/geometry/strategy.py:69-72` | status（_update_active_set_hysteresis） |
| `contact/geometry/strategy.py:278-285` | s, t, gap, normal, tangent1, tangent2, coating_compression（_batch_update_geometry） |
| `contact/contact_force/strategy.py:257` | p_n（SmoothPenalty） |

#### `_ContactPairOutput` 変異箇所

| ファイル | 変異内容 |
|---------|---------|
| `contact/_contact_pair.py:246` | `pair.state = _ContactStateOutput()`（reset_all） |
| `contact/_contact_pair.py:331` | `self.pairs[idx].state.status = ...`（detect_candidates） |

#### `_ContactManagerInput` 変異箇所

| ファイル | 変異内容 |
|---------|---------|
| `contact/_contact_pair.py:240` | `self.pairs.append(pair)`（add_pair） |

## 解消方針（次回セッション向け）

**原則**: `replace()` は使わない。Process が Input を受け取り新しい Output を返すパターンに統一。

### Step 1: `_ContactStateOutput` の frozen 化

最も変異箇所が多い。以下の Process/Strategy が新しい state を返すように改修:

1. **GeometryProcess** → `update_geometry()` が `list[_ContactStateOutput]` を返す
2. **ActiveSetProcess（新設）** → status 更新結果を返す
3. **PenaltyInitProcess** → k_pen/k_t 初期化済み state を返す
4. **FrictionAssembly** → 摩擦状態更新済み state を返す
5. **CoatingStrategy** → 被膜状態更新済み state を返す
6. **ContactForceStrategy** → p_n 更新済み state を返す
7. **SolverProcess** → lambda_n/p_n 更新済み state を返す

### Step 2: `_ContactPairOutput` の frozen 化

Step 1 完了後、state の再代入が不要になるため frozen 化が可能。

### Step 3: `_ContactManagerInput` の frozen 化

pairs リストの append を外部化（Process が新しいリストを返す）。

## 検証結果

```
$ python scripts/validate_process_contracts.py
契約違反: 3 件
  C17: contact/_contact_pair.py の _ContactStateOutput は non-frozen
  C17: contact/_contact_pair.py の _ContactPairOutput は non-frozen
  C17: contact/_contact_pair.py の _ContactManagerInput は non-frozen
条例違反: 2 件（警告）
  O2: numerical_tests/_backend.py BackendRegistry（2件）
```

## 互換ヒストリー

| 旧 | 新 | 備考 |
|---|---|------|
| C17: `_KNOWN_NON_FROZEN` 例外リスト | 廃止（違反は正規報告） | status-203 |
| C17: `dataclasses.replace()` 未検知 | プライベートモジュール内で検知 | status-203 |
