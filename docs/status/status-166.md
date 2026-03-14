# status-166: ProcessRegistry 導入 — _registry dict のクラス昇格

[← README](../../README.md) | [← status-index](status-index.md) | [← status-165](status-165.md)

**日付**: 2026-03-14
**テスト数**: 2477 + 327 process テスト（+13 registry テスト）

## 概要

`AbstractProcess._registry`（クラス変数 dict）を `ProcessRegistry` クラスに昇格。
レジストリ操作の一元化、テスト時の隔離、依存グラフクエリ、カテゴリフィルタリングを実現。

後方互換: `AbstractProcess._registry` は `RegistryProxy` 経由で `ProcessRegistry.default()` に委譲されるため、既存コードは変更不要。

## 設計判断

ChatGPT 断片 A（ProcessRunner / ExecutionContext）は Phase 8 で実装済み。
残る構造的課題として `_registry` がただの dict であり:
- テスト時のレジストリ隔離が困難（グローバル状態を直接操作）
- 依存グラフの横断検索 API がない
- 重複登録の警告がない

これらを解決するために `ProcessRegistry` クラスを導入した。

## 実施内容

### ProcessRegistry クラス（registry.py 新規作成）

```
ProcessRegistry
├── default() → シングルトン
├── register(cls) → 登録（重複警告付き）
├── get(name) → 名前検索
├── items() / keys() / values() → dict 互換
├── filter_by_category(name) → カテゴリフィルタ
├── filter_by_stability(stability) → stability フィルタ
├── non_deprecated() → deprecated 除外
├── concrete_processes() → テストフィクスチャ除外
├── dependencies_of(name) → uses 正引き
├── dependants_of(name) → used_by 逆引き
└── isolate() → テスト用スナップショットコピー
```

### RegistryProxy（後方互換）

`AbstractProcess._registry` を `RegistryProxy(ProcessRegistry.default)` に置換。
dict を継承し、全操作を `ProcessRegistry.default()._store` に委譲。
既存の `name in AbstractProcess._registry` 等はそのまま動作。

### 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/process/registry.py` | **新規**: ProcessRegistry + RegistryProxy |
| `xkep_cae/process/base.py` | `_registry` を RegistryProxy に変更、register() 呼び出し |
| `xkep_cae/process/runner.py` | `_validate_deps` を ProcessRegistry 経由に変更 |
| `xkep_cae/process/__init__.py` | ProcessRegistry エクスポート追加 |
| `xkep_cae/process/tests/test_registry.py` | **新規**: 13テスト |
| `xkep_cae/process/tests/test_contracts.py` | ProcessRegistry 経由に変更 |
| `scripts/validate_process_contracts.py` | ProcessRegistry 経由に変更 |

## テスト結果

- process テスト: 96 パス（test_contracts.py の 22 件は既存の C3 テスト紐付け問題で変更なし）
- 契約検証スクリプト: 0 件違反
- lint: 0 件

## TODO（次セッション）

- [ ] S3 凍結解除判断（9-C）
- [ ] Phase 9-D: BatchProcess パイプライン改善（S3 再開しない場合）
