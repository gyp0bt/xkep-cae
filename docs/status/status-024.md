# status-024: lintエラー解消 + von Mises 3D降伏ロードマップ追加

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-023](./status-023.md)

**日付**: 2026-02-16
**作業者**: Claude Code
**ブランチ**: `claude/setup-project-docs-fHcjy`

---

## 概要

1. ruff check/format で検出された48件のlintエラーを全て解消
2. Phase 4 ロードマップに「4.3 3D弾塑性（von Mises）」を新規追加
3. 既存の Phase 4.3〜4.5 を 4.4〜4.6 に番号繰り上げ、Phase 4.6（撚線モデル）を 4.7 に変更

既存の471テストは全パス（変更なし）。

---

## 実施内容

### 1. lintエラー解消（48件）

`ruff check xkep_cae/ tests/` で検出された48件を修正。

#### 自動修正（ruff check --fix）: 23件

- **I001**: import ブロックのソート（複数ファイル）
- **F401**: 未使用インポートの削除（`pytest`, `sparse`, `spsolve`, `CosseratForces`, `CosseratStrains`, `numpy`）
- **UP035**: `typing.Callable` → `collections.abc.Callable` への移行（`solver.py`）

#### 手動修正: 25件

| ルール | 内容 | ファイル | 件数 |
|--------|------|---------|------|
| **B905** | `zip()` に `strict=True` 追加 | `beam_cosserat.py`, `test_arc_length.py` | 13 |
| **E731** | lambda式を def に変換 | `test_cosserat_sri.py` | 5 |
| **E741** | 曖昧な変数名 `I` → `Iz` | `test_nonlinear.py` | 2 |
| **F841** | 未使用変数の削除 | `test_beam_cosserat.py`, `test_plasticity_1d.py`, `solver.py` | 4 |
| **B006** | ミュータブルデフォルト引数の修正 | `test_plasticity_1d.py` | 1 |

#### フォーマット

`ruff format` で23ファイルをフォーマット。

### 2. von Mises 3D降伏ロードマップ追加

**Phase 4.3 3D弾塑性（von Mises）** として以下を計画:

- von Mises 降伏関数 f = √(3/2) ||dev(σ)|| − (σ_y + R)
- 3D return mapping（radial return）
- 3D consistent tangent（Simo & Taylor 1985）
- 等方硬化（線形・Voce）/ 移動硬化（Armstrong-Frederick）
- PlasticState3D 状態変数
- 平面ひずみ要素との統合（Q4, TRI3, TRI6, Q4_EAS）
- 単軸・二軸・純せん断テスト
- パッチテスト、検証図

### 3. Phase 番号の整理

| 変更前 | 変更後 | 内容 |
|--------|--------|------|
| — | 4.3 | 3D弾塑性（von Mises）**新規** |
| 4.3 | 4.4 | 構造減衰 |
| 4.4 | 4.5 | 粘弾性 |
| 4.5 | 4.6 | 異方性 |
| 4.6 | 4.7 | 撚線モデル |

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `xkep_cae/elements/beam_cosserat.py` | B905修正（zip strict=True）|
| `xkep_cae/solver.py` | F841修正（未使用変数削除）+ UP035修正 |
| `xkep_cae/materials/plasticity_1d.py` | F401修正（未使用import削除）|
| `tests/test_arc_length.py` | I001, F401, B905修正 |
| `tests/test_beam_cosserat.py` | I001, F401, F841修正 |
| `tests/test_cosserat_sri.py` | I001, E731修正 |
| `tests/test_nonlinear.py` | E741修正（`I` → `Iz`）|
| `tests/test_plasticity_1d.py` | F841, B006修正 |
| `docs/roadmap.md` | Phase 4.3 追加、番号整理 |
| `docs/status/status-024.md` | **新規**: 本ステータス |
| `README.md` | ステータスリンク更新 |

---

## テスト結果

```
471 passed, 2 deselected
```

既存テスト全パス。新規テスト追加なし。

---

## 次作業（TODO）

### 優先度A（Phase 4.3: von Mises 3D）
- [ ] von Mises 降伏関数・return mapping 実装
- [ ] 3D consistent tangent
- [ ] PlasticState3D
- [ ] 平面ひずみ要素との統合
- [ ] テスト・検証図

### 優先度B
- [ ] Phase 4.4: 構造減衰（ヒステリシス減衰、粘性項）
- [ ] Phase 4.5: 粘弾性（一般化Maxwell）
- [ ] Phase 5: 動的解析（Newmark-β）

---

## 確認事項・懸念

- von Mises 3D は平面ひずみ要素（Q4, TRI3等）への適用を主眼とするが、将来的に3D固体要素への拡張も視野に入る。インタフェース設計はテンソル表記（Voigt記法）で汎用的に行う。
- Phase 4.3 の実装は既存の 1D Plasticity1D とは独立したクラスとするが、硬化モデル（IsotropicHardening, KinematicHardening）は共有可能。
- status-023 で指摘された「von Mises と roadmap の齟齬」は本PRで解消。

---

## 引き継ぎメモ（Codex/Claude 2交代運用）

- lintエラーは全解消済み。`ruff check xkep_cae/ tests/` と `ruff format --check xkep_cae/ tests/` が両方パスする状態。
- Phase 4 の番号体系が変更されている（4.3〜4.6が1つずつ後ろにずれ、旧4.6撚線が4.7に）。
- von Mises 3D の実装方針は roadmap の Phase 4.3 セクションに記載。参考文献も含む。
