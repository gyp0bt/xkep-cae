# status-124: 16要素/ピッチ最低密度の厳格化

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-06
**テスト数**: 2263（fast: 1640 / deprecated除外済）

## 概要

全テストで16要素/ピッチ未満でのメッシュ生成を厳格に禁止。
`min_elems_per_pitch=0` による密度チェックバイパスを完全廃止し、
`make_twisted_wire_mesh` に16要素/ピッチの強制下限を設けた。

## 変更内容

### 1. `make_twisted_wire_mesh` の密度チェック厳格化

- `min_elems_per_pitch < 16` の指定を `ValueError` で拒否するよう変更
- `min_elems_per_pitch=0` による検査スキップ機能を完全削除
- エラーメッセージから「`min_elems_per_pitch=0` で検査をスキップ」の案内を削除

### 2. CLAUDE.md コーディング規約更新

- プログラムテストの規約から `min_elems_per_pitch=0` の記述を削除
- 全テスト共通で16要素/ピッチ以上を厳守する規約に変更

### 3. テストファイル一括修正（約30ファイル、90箇所以上）

修正対象ファイル:

| カテゴリ | ファイル数 | 主な変更 |
|---------|----------|---------|
| tests/mesh/ | 6 | `min_elems_per_pitch=0` 削除、`_N_ELEM` 定数を16以上に |
| tests/contact/ | 15 | 同上 + `n_elems_per_strand=4` → 16 |
| tests/ (root) | 6 | ベンチマーク・物理テストの要素数修正 |
| xkep_cae/tuning/ | 1 | executor.py の default を16に |
| scripts/ | 1 | run_bending_oscillation.py の DEFAULT_PARAMS |

### 4. ノード数アサーション修正

要素数変更に伴い、ハードコードされたノード数アサーションを修正:
- `test_sheath_contact.py`: `TestSheathSheathMergedCoords` — 22→34, 33→51
- `test_inp_metadata_validation.py`: `test_validate_records_node_info` — 12→48, 15→51
- `test_wire_bending_benchmark.py`: `result.n_elems == 7 * 4` → `7 * 16`

## 設計判断

- 16要素/ピッチは弦近似による初期貫入をワイヤ直径の2%以内に抑える最低限の分割数
- テスト実行時間は増加するが、物理的精度を優先する方針
- deprecated テスト（旧ソルバー比較）は既存の失敗がありそのまま

## 確認事項

- ruff check: 全パス
- pytest (not slow, not deprecated): 1640 passed, 56 skipped

---

[← README](../../README.md)
