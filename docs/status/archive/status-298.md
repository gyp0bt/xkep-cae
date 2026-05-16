# status-298: Hertz型+atol_force frac=1.0完走確認（ベースライン検証）

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-06
- **ブランチ**: `claude/check-status-baseline-8Kea2`
- **テスト数**: 442+ passed（既存テスト全合格）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-297のTODO「Hertz型+atol_forceで frac=1.0 完走確認」を実行。
`contracts/verify_s_unclamped_90deg.py`（7本撚線90度曲げ、Hertz型α=1.5、接触あり）で
**frac=1.0000 完走を達成**。

---

## 検証結果

### 構成

```
n_strands=7, n_elements_per_pitch=16, penalty_exponent=1.5 (Hertz)
contact_enabled=True, free_end_mode=True, loading_mode="rotation"
max_nr_attempts=200, tol_force=1e-8, max_increments=10000
κ = π/(2×100) = 0.015708 [1/mm] → θ_target = 90°
```

### 結果比較

| 指標 | status-285 ベースライン | status-297実装後（今回） | 改善 |
|------|----------------------|--------------------------|------|
| frac | 0.9981 | **1.0000** | **完走達成** |
| n_increments | 551 | 535 | 3%削減 |
| n_cutbacks | 60 | **45** | **25%削減** |
| elapsed | - | 752.3 sec | - |

### カットバック分析

- **合計45回**のカットバック
- **初期（frac<0.05）**: 7回 — 接触の初期安定化に伴う活性集合変化（Type A+B+D.div主因）
- **中盤（frac 0.34〜0.97）**: 37回 — 広範囲に分散、チャタリング的集中なし
- **終盤（frac>0.97）**: 1回 — atol_forceにより微小dtでも確実に収束

### 主要な改善要因

1. **atol_force**（status-297）: 微小dt時の力収束保証 → 最終ステップ不収束を解消
2. **dt snap改善**（status-297）: 端数dt発生防止
3. **s_unclamped修正**（status-291）: Hermite導関数整合性 → K_c誤差低減
4. **Hertz型ペナルティ**（status-285）: 非線形ペナルティでNR収束性改善

---

## 再現手順

```bash
# ブランチ: claude/check-status-baseline-8Kea2
# コミット: 12dcdd9 (Merge pull request #252)
python contracts/verify_s_unclamped_90deg.py 2>&1 | tee /tmp/log-hertz-atol-$(date +%s).log
```

---

## TODO

- [ ] cutback数削減（45→20以下）: 初期安定化の改善（接触初期条件最適化 or 初期dt自動調整）
- [ ] 中盤カットバック37回の削減: NR収束率改善（K_c残余1.8%誤差のさらなる低減 or 適応的delta_h）
- [ ] MPC+contact: ローカルMPC（ワイヤ単位の端部結合）の検討

---

## 次の担当者向け

### 現在のベースライン

```
Hertz型(α=1.5) + atol_force + dt snap
frac=1.0000, incr=535, cutback=45, elapsed=752.3s
```

### 課題: cutback 45→20以下

カットバック分布:
- 初期7回: 接触安定化。初期dt縮小 or 初期接触softening で改善可能性
- 中盤37回: 散発的。Type D.div（接線剛性不整合）が多い。K_c mat-only(1.8%誤差)の残余が根本原因
- 終盤1回: atol_forceで解消済み

**優先度**: 中盤の散発的カットバック削減が最大のインパクト。

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: verify_s_unclamped_90deg.py の出力結果をそのまま記録
- [x] **回帰なし**: 既存テスト全合格（pytest 3 passed for atol_force/snap tests）
- [x] **ベースライン確認**: status-285(frac=0.9981, cutback=60)に対する改善を確認
- [x] **再現手順記載**: ブランチ、コミットハッシュ、実行コマンドを記録
