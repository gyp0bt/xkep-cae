# status-113: メッシュ密度 ValueError 恒久対策 + 物理テスト思想

[← README](../../README.md) | [← status-index](status-index.md) | [← status-112](status-112.md)

日付: 2026-03-05

## 概要

1. **メッシュ密度検査の厳格化**: `make_twisted_wire_mesh` の密度不足を `UserWarning` → `ValueError` に変更
2. **`min_elems_per_pitch` パラメータ追加**: デフォルト16（厳格）、0で検査スキップ（明示的opt-out）
3. **物理テスト追加**: 弦近似貫入量の物理的妥当性テスト `TestChordApproximationPenetration`
4. **CLAUDE.md 更新**: テストの2分類（プログラムテスト vs 物理テスト）+ 視覚的妥当性検証思想を恒久規約化

## 背景

初期貫入問題が約20PR検知されずに潜伏。原因は:
- テストが `n_elems_per_strand=4`（4要素/ピッチ）で粗すぎるメッシュを使用
- 警告（`UserWarning`）は無視されやすい
- プログラムの正しさ（収束するか）のテストはあったが、物理的正しさ（貫入量が妥当か）のテストがなかった

## 実装詳細

### 1. `make_twisted_wire_mesh` の変更

- **新パラメータ**: `min_elems_per_pitch: int = 16`
  - デフォルト16: 粗いメッシュは `ValueError` で拒否
  - 0を指定: 検査スキップ（テスト用の明示的opt-out）
- 旧 `UserWarning` → 新 `ValueError`（見逃し防止）
- `n_strands == 1` の場合は検査しない（中心線のみ）

### 2. テストファイル更新（22ファイル、86箇所）

粗いメッシュを使うプログラムテストには `min_elems_per_pitch=0` を明示追加。
これにより「このテストは意図的に粗メッシュを使っている」ことがコードレベルで可視化。

### 3. 物理テスト追加

`tests/mesh/test_wire_penetration.py` に2クラス追加:

- **`TestMeshDensityValidation`** (5テスト): ValueError発生、opt-out、カスタム閾値、1本撚り除外
- **`TestChordApproximationPenetration`** (3テスト):
  - 16要素/ピッチ → 貫入 < 2% wire_d ✓
  - 32要素/ピッチ → 貫入 < 1% wire_d ✓
  - 4要素/ピッチ → 貫入 > 5% wire_d（粗メッシュの問題を物理的に証明）✓

### 4. CLAUDE.md 恒久規約

テストの2分類を制度化:
- **プログラムテスト**: 収束・API・エラーハンドリング（粗メッシュOK、`min_elems_per_pitch=0`）
- **物理テスト**: 貫入量・応力連続性・荷重オーダー・対称性（物理的に当然の性質を検証）
- **視覚的妥当性検証**: 四元数による2D投影、コンター図の定量判定（将来実装）

## 影響ファイル

- `xkep_cae/mesh/twisted_wire.py` — `min_elems_per_pitch` パラメータ追加
- `tests/mesh/test_wire_penetration.py` — 物理テスト8件追加
- テスト22ファイル — `min_elems_per_pitch=0` 追加（86箇所）
- `CLAUDE.md` — テスト2分類 + 視覚検証思想

## テスト結果

- 修正対象ファイルのテスト: **337 passed** (fast)
- 新規テスト: **51 passed** (test_wire_penetration.py)
- 既存テストへの影響: なし（pre-existing failure 2件は無関係）

## 確認事項

- [ ] 視覚的妥当性検証の具体的実装（四元数2D投影 + matplotlib コンター図生成）は次status以降
- [ ] `test_block_preconditioner::test_block_vs_direct_displacement_consistency` は pre-existing failure（本PRと無関係）

## TODO

- 視覚的妥当性検証スクリプトの実装（`tests/generate_verification_plots.py` に追加）
- 応力コンター・曲率コンターの自動判定テスト（隣接要素間変化率チェック）
