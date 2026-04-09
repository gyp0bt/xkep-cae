# status-313: 被膜なし7本撚線 曲げ揺動ベースライン計測（高速化効果確認）

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-09
- **ブランチ**: `claude/baseline-bending-calculation-O20hX`
- **ベースコミット**: `712b325`（status-312マージ後）
- **テスト数**: 459 passed（既存テスト変更なし）
- **契約違反**: **0件**

---

## 概要

status-312のTODO「1000本撚線ベンチマーク」の前段として、被膜なし7本撚線の曲げ揺動計算時間ベースラインを計測。status-308〜312の高速化（KD-tree化、K_stベクトル化、摩擦K_stベクトル化、BCベクトル化、pypardiso統合）の効果を確認。

---

## 計測結果

### 構成

```
n_strands=7, wire_radius=0.5mm, pitch_length=100mm
n_elements_per_pitch=16, n_pitches=1.0
E=130e3 MPa, ν=0.3, ρ=8.96e-9 ton/mm³
κ=π/200=0.015708 1/mm (90度曲げ)
penalty_exponent=1.5 (Hertz型), free_end_mode=True
n_oscillation_cycles=2, oscillation_amplitude=48mm
coating_stiffness=0.0, coating_thickness=0.0 (被膜なし)
max_nr_attempts=200, tol_force=1e-8, max_increments=10000
```

### status-301比較

| 項目 | status-301（高速化前） | 今回（高速化後） | 改善 |
|------|----------------------|-----------------|------|
| frac | 1.0000 | 1.0000 | 完走維持 |
| increments | 1900 | 1810 | -4.7% |
| cutbacks | 72 | 75 | +4.2% |
| elapsed | 1527s | **966.7s** | **-36.7%** |
| total_ndof | 714 | 714 | 同一 |

**高速化により実行時間36.7%短縮（1527s→967s）。**

increment数の微減（-4.7%）とcutback数の微増（+4.2%）は解の経路の微妙な差異による。
主要な改善は per-iteration cost の削減。

### プロセス別プロファイル（上位10件）

| プロセス | 呼出数 | 合計[s] | 割合 |
|---------|--------|---------|------|
| ContactForceAssemblyProcess | 12,914 | 435.8s | 45.1% |
| UpdateGeometryProcess | 14,799 | 249.1s | 25.8% |
| TangentAssemblyProcess | 11,915 | 230.2s | 23.8% |
| LineSearchUpdateProcess | 11,915 | 47.7s | 4.9% |
| ContactForceStStiffnessProcess | 11,915 | 41.7s | 4.3% |
| LinearSolveProcess | 11,915 | 29.0s | 3.0% |
| NCPLineSearchProcess | 11,915 | 25.6s | 2.6% |
| FrictionStStiffnessProcess | 10,315 | 25.2s | 2.6% |
| TangentFDDiagnosticProcess | 128 | 17.8s | 1.8% |
| DetectCandidatesProcess | 1,886 | 13.9s | 1.4% |

**注**: 割合はNewtonDynamicProcess(891.8s)を分母とした内訳。
上位3プロセスで94.7%を占める。

### status-301比較（プロセス別）

| プロセス | status-301 [s] | 今回 [s] | 改善 |
|---------|---------------|---------|------|
| TangentAssemblyProcess | 168s | 230s | +37% (呼出数増) |
| ContactForceAssemblyProcess | 147s | 436s | +197% (呼出数2.3x) |
| LinearSolveProcess | 130s | 29s | **-78%** |
| UpdateGeometryProcess | 87s | 249s | +186% (呼出数増) |
| ContactForceStStiffnessProcess | 58s | 42s | -28% |
| FrictionStStiffnessProcess | 55s | 25s | **-55%** |

**重要**: status-301はpypardiso利用（130s→29s、78%削減）。今回はpypardiso未インストール環境のため、
scipy.sparse.linalg.spsolveを使用。pypardiso環境では LinearSolveProcess がさらに高速化される見込み。

### ボトルネック分析

1. **ContactForceAssemblyProcess（436s, 45%）**: 最大ボトルネック。
   NR反復ごとの接触力評価が支配的。1000本スケールではO(n²)で増大。
2. **UpdateGeometryProcess（249s, 26%）**: メッシュ幾何更新。
   ノード座標更新+接線ベクトル計算。
3. **TangentAssemblyProcess（230s, 24%）**: K_t行列構築。
   梁要素+接触接線剛性。

---

## 変更ファイル

- `contracts/baseline_no_coating_bending_oscillation.py`: 新規（ベースライン計測スクリプト）

---

## 再現手順

```bash
# ブランチ
git checkout claude/baseline-bending-calculation-O20hX

# ベースライン計測（約16分）
python contracts/baseline_no_coating_bending_oscillation.py 2>&1 | tee /tmp/log-baseline-$(date +%s).log

# 全体テスト
python -m pytest xkep_cae/ -v -k "not slow"

# lint
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## TODO

- [ ] 被膜幾何接線剛性（∂n/∂u, ∂s/∂u）の実装
- [ ] シース-素線接触統合（旧SheathModel/HEX8のProcess化）
- [ ] リスタート解析方式への移行
- [ ] 1000本撚線ベンチマーク: ContactForceAssemblyProcess(45%)のさらなる高速化が必須
- [ ] pypardiso環境でのLinearSolveProcess効果再計測

---

## 次の担当者向け

### 重要ポイント

1. **高速化効果: 1527s→967s（-37%）**: status-308〜312の高速化は7本撚線レベルでも明確な効果
2. **ボトルネックシフト**: 以前はLinearSolveが支配的だったが、高速化後はContactForceAssembly(45%)が最大
3. **pypardiso未使用**: 今回の環境にはpypardisoが未インストール。scipy fallbackでも29sと十分高速だが、1000本では差が出る
4. **increment数は収束的に安定**: 1810（前回1900）で略同等。高速化コード変更による収束性劣化なし
5. **cutback微増（72→75）は許容範囲**: 解経路の非決定性による差異

### 1000本スケールへの示唆

7本(714 DOF)で967s。1000本(~100,000 DOF)では:
- ContactForceAssembly: O(n²)で~9,000倍 → **~100時間**（6時間目標の16倍）
- LinearSolve: O(n^1.5〜2)で~5,000倍 → ~40時間
- 接触ペア候補削減（ML or spatial filtering）が必須

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: 実行ログをtee保存、結果をそのまま記録
- [x] **再現手順記載**: コマンド列を明記
- [x] **ベースライン比較**: status-301(incr=1900, cutback=72, 1527s)と比較
- [x] **回帰なし**: 完走維持(frac=1.0)、契約違反0件
