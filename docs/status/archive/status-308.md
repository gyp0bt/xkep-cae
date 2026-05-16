# status-308: 収束型統計デッドコード修正 + 接触ペア検出KD-tree化

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-08
- **ブランチ**: `claude/check-status-todos-xYiua`
- **テスト数**: 442+20+14 passed（既存テスト全合格 + broadphaseテスト14件追加）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-307のTODO「slowテスト実機実行でログ出力フォーマット確認」を実行した結果、`[収束型統計]`・`[coat]`・エネルギー診断が出力されないバグを発見・修正。並行して高速化フェーズ第1弾として接触ペア検出のKD-tree化を実施。

---

## 実施内容

### 1. 収束型統計デッドコード修正（バグ修正）

**問題**: `process.py` の lines 643-710 が `if not step_result.converged:` ブロック内に配置されており、到達不可能なデッドコードになっていた。

- `if fail_out.can_retry:` → `continue`
- `else:` → `return`
- その後の「ステップ完了」セクション（収束型統計、被膜圧縮統計、エネルギー診断）には到達しない

**修正**: 該当コードブロック全体を4スペースデデント（`if not step_result.converged:` ブロックの外へ移動）。

**効果**: 修正前後のテスト結果比較（2本撚線ベースライン）:

| 項目 | 修正前 | 修正後 |
|------|--------|--------|
| `[収束型統計]` | 出力なし | `force=8(18%), disp=4(9%), energy=31(72%), total=43` |
| エネルギー収支サマリ | 出力なし | 正常出力 |
| テスト結果 | PASSED | PASSED |

7本撚線テスト: `[収束型統計] force=16(31%), disp=10(19%), energy=25(49%), total=51`。frac=1.0000完走。

### 2. 接触ペア検出KD-tree化（高速化）

**変更**: `_broadphase_aabb` の内部実装を空間ハッシュグリッドから `scipy.spatial.cKDTree` に置換。

| 項目 | 旧実装（空間ハッシュ） | 新実装（KD-tree） |
|------|----------------------|-------------------|
| アルゴリズム | 均一格子ビニング + セル走査 | cKDTree.query_pairs + AABB フィルタ |
| 内部ループ | Pure Python（3重ネストfor） | C/Cython（scipy内部） |
| 計算量 | O(n * cells) | O(n log n) |
| 外部依存 | なし | scipy.spatial.cKDTree |

**API互換性**: 関数シグネチャ完全互換。`cell_size` パラメータは後方互換のため受け入れるが無視。

**テスト**: 14件追加（API 8件 + Physics 6件）

| テスト | 検証内容 |
|--------|---------|
| `test_empty_returns_empty` | 空入力 |
| `test_single_segment_returns_empty` | 1セグメント |
| `test_two_overlapping_segments` | 近接ペア検出 |
| `test_two_distant_segments_no_pair` | 遠方ペア除外 |
| `test_pair_order_i_less_j` | ペア順序 |
| `test_scalar_radii` | スカラー半径 |
| `test_array_radii` | 配列半径 |
| `test_margin_expands_search` | マージン拡張 |
| `test_parallel_helical_wires` | ヘリカル素線配置 |
| `test_no_false_negatives_for_touching` | 接触ペア偽陰性なし |
| `test_cell_size_backward_compat` | cell_size後方互換 |
| `test_scaling_consistency[50/100/200]` | スケーリング整合性 |

---

## 変更ファイル

- `xkep_cae/contact/solver/process.py`: 収束型統計・被膜圧縮統計・エネルギー診断のインデント修正
- `xkep_cae/contact/_broadphase.py`: 空間ハッシュグリッド → cKDTree 置換
- `xkep_cae/contact/tests/test_broadphase.py`: 新規（broadphaseテスト14件）

---

## 再現手順

```bash
# ブランチ
git checkout claude/check-status-todos-xYiua

# broadphaseテスト
python -m pytest xkep_cae/contact/tests/test_broadphase.py -v

# 既存テスト（geometry + manager + solver）
python -m pytest xkep_cae/contact/geometry/tests/ xkep_cae/contact/tests/ xkep_cae/contact/solver/tests/ -v

# slowテスト（ログ出力確認）
python -m pytest tests/numerical_tests/test_strand_bending_convergence.py::TestTwoStrandBendingConvergence::test_two_strand_bending_baseline -v -s -m slow 2>&1 | tee /tmp/log-308-$(date +%s).log

# lint
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/

# 契約チェック
python contracts/validate_process_contracts.py
```

---

## TODO

- [ ] KD-tree broadphaseの大規模ベンチマーク（1000本撚線での性能比較）
- [ ] K_c/K_stアセンブリベクトル化（高速化フェーズ第2弾）
- [ ] スパース求解高速化（高速化フェーズ第3弾）
- [ ] 被膜幾何接線剛性（∂n/∂u, ∂s/∂u）の実装
- [ ] シース-素線接触統合（旧SheathModel/HEX8のProcess化）
- [ ] リスタート解析方式への移行

---

## 次の担当者向け

### 重要ポイント

1. **収束型統計が正常出力されるようになった**: `[収束型統計]`・エネルギー収支サマリ・`[coat]`被膜統計が解析完了時に正しく出力される
2. **broadphaseがKD-treeベースに**: `scipy.spatial.cKDTree` を使用。Pure Pythonのセル走査ループが排除され、大規模メッシュで高速化が期待される
3. **cell_sizeパラメータは無視される**: 後方互換のためシグネチャに残置しているが、KD-treeでは不要。呼び出し側の `broadphase_cell_size` 設定は影響なし
4. **高速化フェーズの次ステップ**: K_c/K_stアセンブリのベクトル化が最も効果的な次の改善候補

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: slowテスト実行ログをteeで保存
- [x] **再現手順記載**: コマンド列を明記
- [x] **回帰なし**: 既存テスト全合格、契約違反0件
