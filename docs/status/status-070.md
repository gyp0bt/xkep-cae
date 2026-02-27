# status-070: GitHub Actions CI 構成 + slow テストマーカー導入

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-27
**作業者**: Claude Code
**テスト数**: 1629（変更なし、構成変更のみ）

## 概要

全テスト実行時間の増大に対応するため、GitHub Actions CI を構成し、
重いテストを `@pytest.mark.slow` マーカーで分離。
ローカルでは fast テストのみ実行し、full テストは CI に外出しする運用に移行。

## 1. `@pytest.mark.slow` マーカー導入

### 対象ファイル・テスト数

| カテゴリ | ファイル | slow テスト数 |
|----------|---------|-------------|
| PINN大規模メッシュ | `test_pinn_extended.py` | 9 |
| PINN学習 | `test_pinn.py` | 6 |
| GNN比較 | `test_gnn_fc.py` | 4 |
| 不規則メッシュ比較 | `test_irregular_mesh.py` | 2 |
| サロゲート検証 | `test_surrogate_validation.py` | 4 |
| 撚線接触 | `test_twisted_wire_contact.py` | 45（全テスト） |
| 実梁接触 | `test_real_beam_contact.py` | 21（全テスト） |
| 被膜撚線 | `test_coated_wire_integration.py` | 20（全テスト） |
| 貫入テスト | `test_beam_contact_penetration.py` | 20（全テスト） |
| 大規模接触 | `test_large_scale_contact.py` | 11（全テスト） |
| 弾塑性バリデーション | `test_abaqus_validation_elastoplastic.py` | 5（全テスト） |
| 過渡応答解析 | `test_dynamics.py` | 66（全テスト） |
| 数値試験 | `test_numerical_tests.py` | 70（全テスト） |
| **合計** | **13ファイル** | **283テスト** |

### テスト分布

- **fast テスト**: 1344（`-m "not slow"`）— ローカル実行用
- **slow テスト**: 283（`-m "slow"`）— CI 外出し用
- **external テスト**: 2（`-m "external"`）— 外部依存、通常スキップ

## 2. GitHub Actions CI ワークフロー

`.github/workflows/ci.yml` を3ジョブ構成に拡張:

### ジョブ構成

| ジョブ | トリガー | 内容 | 所要時間目安 |
|--------|---------|------|-------------|
| `lint` | 全push/PR | ruff check + format | ~30秒 |
| `test-fast` | 全push/PR | 1344テスト（Python 3.10/3.11/3.12 × 3） | ~3分 |
| `test-slow` | PR/master のみ | 283テスト（Python 3.11） | ~20-30分 |

### トリガー設定

- `master` ブランチへの push/PR: 全ジョブ実行
- `claude/**` ブランチへの push: lint + test-fast のみ
- PR 作成時: 全ジョブ実行（slow 含む）

### 依存関係

```
lint → test-fast → test-slow (PR/masterのみ)
```

## 3. pyproject.toml 変更

### `[ml]` optional dependencies 追加

```toml
[project.optional-dependencies]
ml = [
    "torch>=2.0",
    "torch-geometric>=2.4",
]
```

### `slow` マーカー登録

```toml
[tool.pytest.ini_options]
markers = [
    ...
    "slow: PINN学習・大規模メッシュ・複数モデル訓練など時間がかかるテスト",
]
```

### `pytest-timeout` を dev に追加

```toml
dev = [
    "pytest>=7.0",
    "pytest-timeout>=2.0",
    "ruff>=0.4",
]
```

## 4. 運用手順

### ローカル開発時

```bash
# fast テストのみ（~3分）
python -m pytest tests/ -m "not slow and not external" -x

# thermal テストのみ（fast）
python -m pytest tests/thermal/ -m "not slow" -x

# slow テスト含む全量（~30分）
python -m pytest tests/ -m "not external" -x
```

### 2交代制運用（Codex / Claude Code）

1. 実装後、`claude/**` ブランチに push
2. CI が lint + fast テストを自動実行（~3分）
3. PR 作成時に slow テストも実行（~30分）
4. 次セッション冒頭で GHA 結果を確認 → 失敗あれば修正
5. GHA の URL は status ファイルに記録

## ファイル変更

### 変更
- `.github/workflows/ci.yml` — 3ジョブ構成に拡張（lint + test-fast + test-slow）
- `pyproject.toml` — `[ml]` extras追加 + `slow` マーカー + `pytest-timeout`
- `tests/thermal/test_pinn_extended.py` — 9テストに `@pytest.mark.slow`
- `tests/thermal/test_pinn.py` — 6テストに `@pytest.mark.slow`
- `tests/thermal/test_gnn_fc.py` — 4テストに `@pytest.mark.slow`
- `tests/thermal/test_irregular_mesh.py` — 2テストに `@pytest.mark.slow`
- `tests/thermal/test_surrogate_validation.py` — 4テストに `@pytest.mark.slow` + `import pytest` 追加
- `tests/contact/test_twisted_wire_contact.py` — `pytestmark = pytest.mark.slow` + `import pytest` 追加
- `tests/contact/test_real_beam_contact.py` — `pytestmark = pytest.mark.slow` + `import pytest` 追加
- `tests/contact/test_coated_wire_integration.py` — `pytestmark = pytest.mark.slow`
- `tests/contact/test_beam_contact_penetration.py` — `pytestmark = pytest.mark.slow` + `import pytest` 追加
- `tests/contact/test_large_scale_contact.py` — `pytestmark = pytest.mark.slow` + `import pytest` 追加
- `tests/test_abaqus_validation_elastoplastic.py` — `pytestmark = pytest.mark.slow`
- `tests/test_dynamics.py` — `pytestmark = pytest.mark.slow`
- `tests/test_numerical_tests.py` — `pytestmark = pytest.mark.slow`

## TODO

- [ ] CI 実行結果の確認（初回push後）
- [ ] slow テスト個別の timeout 調整が必要な場合の対応
- [ ] CI キャッシュ導入（pip install 高速化: actions/cache）
- [ ] CI バッジを README に追加

## 確認事項・懸念

- `torch` / `torch-geometric` の GHA 上でのインストール時間（pip install で5-10分かかる可能性）。将来的に pip キャッシュ導入を検討。
- slow テストの timeout は300秒（5分/テスト）に設定。PINN大規模メッシュ（~13分）はテスト単位では1テストあたり数分で収まる想定。全ジョブの timeout は45分。
- `test_dynamics.py`（66テスト）と `test_numerical_tests.py`（70テスト）はファイル全体を slow にした。個別テストの分類が必要になった場合は細分化を検討。

---
