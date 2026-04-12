# status-227: 条件数スペクトル分析 + NR内幾何凍結検証

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-22
**ブランチ**: `claude/fix-focus-guard-bending-RlaJe`

---

## 概要

frac=0.60 の壁の根本原因を多角的に調査。
条件数スペクトル分析、NR内幾何凍結、s,t relaxation、摩擦除去、k_pen変更を系統的に検証。

**結論**: 壁の直接原因は frac=0.60 付近での **2-cycle 残差振動**（接線剛性の非正定値性に起因）。
試みた全ての対策（幾何凍結、relaxation、K_st、摩擦除去）で改善なし。
status-224 ブランチの成功（frac=0.89）は異なるコードベースで達成されたもので、再現不可能。

---

## 実装内容

### 1. 条件数スペクトル分析（`_newton_dynamic.py`）

`compute_condition_number` フラグを NR ループに追加。
各反復で K_T（境界条件適用後）の固有値分解を実行:
- `condition_number_history`, `min_eigenvalue_history`, `max_eigenvalue_history` を ConvergenceDiagnosticsOutput に追加
- `ContactFrictionInputData.compute_condition_number` フラグで制御

### 2. NR内幾何凍結（`_manager_process.py`, `_contact_pair.py`）

- `freeze_geometry_in_nr: bool = False` フラグを `_ContactConfigInput` に追加
- `freeze_st: bool = False` を `UpdateGeometryInput` に追加
- freeze_st=True 時: 既存 s,t を保持し、現在座標での gap/normal のみ再計算

### 3. s,t under-relaxation（`_manager_process.py`）

- `st_relaxation: float = 1.0` を `_ContactConfigInput` と `UpdateGeometryInput` に追加
- s_new = α*s_computed + (1-α)*s_old のブレンディング
- 緩和済み s,t で gap/normal を再計算

---

## 診断結果

### スペクトル分析

| 項目 | 値 | 備考 |
|------|-----|------|
| λ_max | ≈ 8.16e+04 | 構造剛性（一定） |
| λ_min | -1.0 〜 +0.8 | **振動**（接触依存） |
| n_neg | 0 or 1 | 1個の負の固有値 |
| 条件数 | 1e4 〜 2e5 | 中程度 |

**K_T は非正定値**（負の固有値 1 つ）。n_neg=1 の反復では線形収束、n_neg=0 の反復では2次収束。

### 2-cycle 残差振動パターン（frac=0.6067）

```
att  5: ||R||/||f|| = 8.170e-01, active=15
att 10: ||R||/||f|| = 8.131e-01, active=15
att 15: ||R||/||f|| = 8.170e-01, active=15
att 20: ||R||/||f|| = 8.131e-01, active=15
att 25: ||R||/||f|| = 8.170e-01, active=15
```

完全な 2-cycle（0.8170 ↔ 0.8131）。active ペア数は安定（チャタリングではない）。

### 対策実験結果

| # | 対策 | frac到達 | 時間(s) | cutback | 判定 |
|---|------|---------|---------|---------|------|
| 1 | Baseline (α=1.0) | 0.6011 | 78 | 17 | 基準 |
| 2 | freeze_st_nr=True (幾何凍結) | 0.38 | 245 | 54 | 悪化 |
| 3 | st_relaxation α=0.5 | 0.19 | 289 | 83 | 大幅悪化 |
| 4 | st_relaxation α=0.3 | 0.60 | 245 | 54 | 基準同等（遅い） |
| 5 | mu=0 (摩擦なし) | 0.59 | 170 | 44 | 摩擦は無関係 |
| 6 | k_pen=5000 (高ペナルティ) | 0.00 | 30 | 10 | 発散 |
| 7 | max_incr=500 | 0.60 | ~200 | 17 | 壁は超えない |

### 摩擦は無関係

mu=0（摩擦完全除去）でも frac=0.59 で停滞。diagnosis.md の結論と一致。
**問題は接触法線力の NR 自体にある。**

### status-224 との乖離

status-224 (Run 06): frac=0.890, fc=131.9N（別ブランチ `claude/verify-dynamic-bending-load-cmPwf`）。
同一条件（E=25, k_pen=auto, max_incr=500）で現在のコードベースでは frac=0.60 で停止。
別ブランチとの差分は特定困難（マージ後に多数の変更あり）。

---

## 分析

### 根本原因

1. **K_T の非正定値性**: 接触幾何剛性 K_geo が負の固有値を導入
2. **NR 内 s,t 更新**: 反復ごとに接触点が移動し、残差と接線剛性の整合性が崩れる
3. **2-cycle 振動**: frac > 0.60 で残差が2値間で振動し、力収束も変位収束も達成できない

### 試行錯誤で判明した制約

- s,t 凍結: 精度低下で悪化
- s,t 緩和: 収束速度低下で悪化
- 摩擦除去: 効果なし
- k_pen 増大: 発散
- max_incr 増大: 壁は超えない

---

## 次のステップ（TODO）

1. **status-224 ブランチのコード差分調査**: `git diff f3c835d..HEAD -- xkep_cae/contact/` で変更点を特定
2. **NR ループ構造の見直し**: status-224 時点では NR内幾何更新なしで frac=0.60 到達。
   現在は NR内幾何更新ありで frac=0.60。根本的な NR ループ設計の再検討が必要。
3. **外側幾何ループ**: NR を「内部ループ(s,t 固定) + 外部ループ(s,t 更新)」に分離
4. **Levenberg-Marquardt 正則化**: K_T に小さな正の対角シフトを加え正定値化
5. **ギャップベースのペナルティ更新**: NR 内で gap のみ更新し s,t,normal は固定

---

## 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
|---------|---------|------|
| `xkep_cae/contact/_contact_pair.py` | 変更 | freeze_geometry_in_nr, st_relaxation フラグ追加 |
| `xkep_cae/contact/_manager_process.py` | 変更 | freeze_st, st_relaxation 対応 |
| `xkep_cae/contact/solver/_newton_steps.py` | 変更 | freeze_st, st_relaxation の伝搬 |
| `xkep_cae/contact/solver/_newton_dynamic.py` | 変更 | compute_condition_number 追加 |
| `xkep_cae/contact/solver/_diagnostics.py` | 変更 | condition_number_history 等追加 |
| `xkep_cae/contact/solver/process.py` | 変更 | compute_condition_number 伝搬 |
| `xkep_cae/core/data.py` | 変更 | compute_condition_number フラグ追加 |
| `contracts/check_spectral_freeze.py` | 新規 | スペクトル分析 + 幾何凍結検証 |
| `contracts/check_st_relaxation.py` | 新規 | s,t relaxation 検証 |
| `contracts/check_wall_cause.py` | 新規 | 壁原因切り分け（摩擦/k_pen） |
| `contracts/check_reproduce_224.py` | 新規 | status-224 再現テスト |

---

## テスト

**186+10 passed** — 契約違反 1件（既存）、条例違反 0件
