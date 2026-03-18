# status-202: C17 dataclass 衛生チェック + 命名規約修正

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-17
**テスト数**: ~2260 + 374 新パッケージテスト
**契約違反**: C16: 0件, C17: 0件（既知例外3件あり）

---

## 概要

プライベートモジュール（`_xxx.py`）内の dataclass 衛生チェック（C17）を新設。
non-frozen dataclass および Input/Output で終わらないクラス名を違反として検出する。

`contact._contact_pair` の4つの dataclass が「めちゃくちゃ違法」状態であった問題に対応。

## 変更内容

### 1. C17 検知スクリプト追加（`scripts/validate_process_contracts.py`）

新ルール C17: プライベートモジュール dataclass 衛生

- **frozen チェック**: `_xxx.py` 内の dataclass に `frozen=True` 必須
- **命名チェック**: dataclass 名は `Input` または `Output` で終わる必要あり
- 既知の non-frozen 例外リスト（3件）あり — 段階的に解消予定

### 2. `_ContactConfig` → `_ContactConfigInput`（frozen=True）

- 69フィールドの純データコンテナ → `frozen=True` に変更
- 書き込み箇所ゼロのため安全に凍結
- クラス名を `_ContactConfigInput` にリネーム

### 3. 命名規約リネーム（82ファイル修正）

| 旧名 | 新名 | 理由 |
|------|------|------|
| `*Config`（3クラス） | `*Input` | 設定データ＝入力 |
| `*Result`（3クラス） | `*Output` | C17規約はInput/Output |
| `_ContactState` | `_ContactStateOutput` | 状態出力 |
| `_ContactPair` | `_ContactPairOutput` | ペア出力 |
| `_ContactManager` | `_ContactManagerInput` | 管理入力 |
| `BeamForces3D` | `BeamForces3DOutput` | 断面力出力 |
| `BeamSection`/`2D` | `BeamSectionInput`/`2DInput` | 断面入力 |
| `StrandInfo` | `StrandInfoOutput` | 撚線情報出力 |
| `TwistedWireMesh` | `TwistedWireMeshOutput` | メッシュ出力 |
| `SolverState` | `SolverStateOutput` | ソルバー状態出力 |
| `ConvergenceDiagnostics` | `ConvergenceDiagnosticsOutput` | 診断出力 |
| `_ContactEdge`/`_ContactGraph` | `*Output` | グラフ出力 |
| その他 `*Output`（16クラス） | 変更なし | 既に準拠 |

### 4. 既知の non-frozen 例外（3件）

| クラス | ファイル | 理由 |
|--------|---------|------|
| `_ContactStateOutput` | `_contact_pair.py` | 50+箇所で直接変異。frozen化にはProcess分割が必要 |
| `_ContactPairOutput` | `_contact_pair.py` | state経由で変異 |
| `_ContactManagerInput` | `_contact_pair.py` | メソッドで状態変更。Process分割が必要 |

## 検証結果

- `python scripts/validate_process_contracts.py` → 契約違反 0件
- `ruff check xkep_cae/ tests/` → 0件
- `ruff format --check xkep_cae/ tests/` → 全ファイル整形済み
- 関連テスト 141件 全パス

## TODO

- [ ] `_ContactStateOutput`/`_ContactPairOutput` → frozen=True 化
  - **`dataclasses.replace()` は禁止**
  - Process または Strategy の戻り値として新インスタンスを再作成する方式
  - 例: `pair.state.gap = v` → Strategy/Process が新しい `_ContactStateOutput` を返す
  - 50+箇所の変異を全てProcess/Strategy出力に変換
- [ ] `_ContactManagerInput` → Process 分割（Input共有 + 各操作をProcess APIで管理）
  - ContactManagerのmutableオブジェクト渡しでは改変追跡が不可能
  - detect_candidates / update_geometry / initialize_penalty は既にProcess化済み
  - 残りの直接変異（pair.state.xxx = ...）をProcess出力として管理すべき
  - 各ProcessがInput（旧state）を受け取りOutput（新state）を返すパターンに統一
- [ ] Phase 16: BackendRegistry 完全廃止（O2/O3 条例違反5件解消）
- [ ] 被膜モデル物理検証テスト

## 互換ヒストリー

| 旧名 | 新名 | 備考 |
|------|------|------|
| `AdaptiveSteppingConfig` | `AdaptiveSteppingInput` | C17 命名規約 |
| `NewtonUzawaDynamicConfig` | `NewtonUzawaDynamicInput` | C17 命名規約 |
| `NewtonUzawaStaticConfig` | `NewtonUzawaStaticInput` | C17 命名規約 |
| `_ContactConfig` | `_ContactConfigInput` | C17 + frozen=True |
| `_ContactState` | `_ContactStateOutput` | C17 命名規約 |
| `_ContactPair` | `_ContactPairOutput` | C17 命名規約 |
| `_ContactManager` | `_ContactManagerInput` | C17 命名規約 |
| `BeamForces3D` | `BeamForces3DOutput` | C17 命名規約 |
| `BeamSection` | `BeamSectionInput` | C17 命名規約 |
| `BeamSection2D` | `BeamSection2DInput` | C17 命名規約 |
| `StrandInfo` | `StrandInfoOutput` | C17 命名規約 |
| `TwistedWireMesh` | `TwistedWireMeshOutput` | C17 命名規約 |
| `SolverState` | `SolverStateOutput` | C17 命名規約 |
| `ConvergenceDiagnostics` | `ConvergenceDiagnosticsOutput` | C17 命名規約 |
| `_ContactEdge` | `_ContactEdgeOutput` | C17 命名規約 |
| `_ContactGraph` | `_ContactGraphOutput` | C17 命名規約 |
| `_ClosestPointResult` | `_ClosestPointOutput` | C17 命名規約 |
| `DynamicStepResult` | `DynamicStepOutput` | C17 命名規約 |
| `StaticStepResult` | `StaticStepOutput` | C17 命名規約 |

## 開発運用メモ

- C17 の non-frozen 例外リストは新規追加禁止。既存3件の段階的解消のみ。
- Input/Output 命名規約により、dataclass の役割（入力 or 出力）が型名から即座に判断可能になった。
- ContactManager の Process 分割は次PRで対応すべき最重要課題。mutableオブジェクトの直接渡しは改変追跡が不可能。
