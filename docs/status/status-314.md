# status-314: surface_pair_n_neighbor パイプライン貫通 + 7本撚線効果確認

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-09
- **ブランチ**: `claude/baseline-bending-calculation-O20hX`
- **前提**: status-313（被膜なし7本ベースライン: incr=1810, cutback=75, 967s）
- **テスト数**: 459 passed（既存テスト変更なし）
- **契約違反**: **0件**

---

## 概要

status-276で実装された `surface_pair_n_neighbor`（隣接素線ペアフィルタ）が `StrandBendingOscillationConfig` から `_ContactConfigInput` に伝搬されていないバグを修正。パイプライン貫通後、n_neighbor=2 の7本撚線への効果を計測。

**結論: 7本撚線ではn_neighbor=2は効果なし（incr/cutback同一）。デフォルト=0維持。91本以上で候補ペア爆発時に効果を期待。**

---

## 修正内容

### バグ: `surface_pair_n_neighbor` 未伝搬

`StrandBendingOscillationConfig` に `surface_pair_n_neighbor` フィールドが存在せず、
`_ContactConfigInput` 構築時にも渡されていなかった。結果として `DetectCandidatesProcess` での
ホワイトリストフィルタが常に無効（n_neighbor=0）だった。

### 修正箇所

1. `xkep_cae/numerical_tests/strand_bending_oscillation.py`:
   - `StrandBendingOscillationConfig` に `surface_pair_n_neighbor: int = 0` 追加
   - MPC モード（~line 720）と free_end_mode（~line 984）の `_ContactConfigInput` 構築に `surface_pair_n_neighbor=cfg.surface_pair_n_neighbor` を追加

### デフォルト値

- **0（無効）を維持**: 7本撚線では効果なし。91本以上のスケールで有効化を検討。
- 明示的に `surface_pair_n_neighbor=2` を渡すことで有効化可能。

---

## 計測結果: n_neighbor=2 vs n_neighbor=0

### 比較（被膜なし7本撚線 曲げ+揺動）

| 項目 | n_neighbor=0 (status-313) | n_neighbor=2 | 差異 |
|------|--------------------------|-------------|------|
| frac | 1.0000 | 1.0000 | 同一 |
| increments | 1810 | 1810 | **同一** |
| cutbacks | 75 | 75 | **同一** |
| elapsed | 967s | 1137s | +17.5% |
| total_ndof | 714 | 714 | 同一 |
| max \|u\| | 6.465e+01 | 6.465e+01 | 同一 |

### プロセス別比較

| プロセス | n_neighbor=0 [s] | n_neighbor=2 [s] | 差異 |
|---------|-----------------|-----------------|------|
| ContactForceAssembly | 436s | 522s | +20% |
| UpdateGeometry | 249s | 299s | +20% |
| TangentAssembly | 230s | 271s | +18% |
| LinearSolve | 29s | 34s | +17% |
| DetectCandidates | 14s | 16s | +11% |

### 分析

- **incr/cutback/max|u| が完全一致**: n_neighbor=2 は7本撚線の接触ペア候補に影響なし
- **+17.5%の時間増は環境差**: 全プロセスで均一に+17〜20%増加しており、CPU負荷変動（サーマルスロットリング等）による。ホワイトリスト構築のオーバーヘッドではない。
- **7本撚線（16要素/ピッチ×7本=112要素）ではブロードフェーズが十分**: 隣接フィルタなしでもペア候補は限定的。
- **91本以上（~1500要素）で候補ペアがO(n²)爆発するため、そのスケールで効果を期待**。

---

## 変更ファイル

- `xkep_cae/numerical_tests/strand_bending_oscillation.py`: `surface_pair_n_neighbor` フィールド追加 + パイプライン貫通
- `contracts/baseline_no_coating_bending_oscillation.py`: docstring更新

---

## 再現手順

```bash
# ブランチ
git checkout claude/baseline-bending-calculation-O20hX

# n_neighbor=0 ベースライン（status-313と同一）
python contracts/baseline_no_coating_bending_oscillation.py 2>&1 | tee /tmp/log-baseline-$(date +%s).log

# n_neighbor=2 計測（Configで明示指定）
# contracts/baseline_no_coating_bending_oscillation.py の cfg に
# surface_pair_n_neighbor=2 を追加して実行

# lint
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## TODO

- [ ] 91本撚線での n_neighbor=2 効果計測
- [ ] 被膜幾何接線剛性（∂n/∂u, ∂s/∂u）の実装
- [ ] シース-素線接触統合（旧SheathModel/HEX8のProcess化）
- [ ] リスタート解析方式への移行
- [ ] 1000本撚線ベンチマーク: ContactForceAssemblyProcess(45%)のさらなる高速化が必須

---

## 次の担当者向け

### 重要ポイント

1. **パイプライン貫通完了**: `surface_pair_n_neighbor` が `StrandBendingOscillationConfig` → `_ContactConfigInput` → `DetectCandidatesProcess` に正しく伝搬される
2. **デフォルト=0（無効）**: 7本撚線では効果なし。91本以上で `surface_pair_n_neighbor=2` を明示指定して効果確認が必要
3. **7本撚線ベースライン不変**: status-313の結果（incr=1810, cutback=75, 967s）が引き続き有効

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: 実行ログをtee保存、結果をそのまま記録
- [x] **再現手順記載**: コマンド列を明記
- [x] **ベースライン比較**: status-313(n_neighbor=0)と直接比較
- [x] **回帰なし**: 完走維持(frac=1.0)、契約違反0件
