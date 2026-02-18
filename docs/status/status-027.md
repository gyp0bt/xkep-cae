# status-027: 戻り値型リファクタリング + ロードマップ優先順位更新

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-18

## 概要

メソッド戻り値に NamedTuple 型クラスを導入し、ロードマップの次の優先順位を設定した。

## 実装内容

### 戻り値型リファクタリング

**新規ファイル**: `xkep_cae/core/results.py`

以下の NamedTuple 型クラスを定義し、各メソッドの戻り値を更新:

| 型クラス | 対象メソッド | フィールド |
|---------|-------------|----------|
| `LinearSolveResult` | `solve_displacement()` | u, info |
| `DirichletResult` | `apply_dirichlet()`, `apply_dirichlet_penalty()` | K, f |
| `AssemblyResult` | `assemble_cosserat_beam()` | K_T, f_int |
| `PlasticAssemblyResult` | `assemble_cosserat_beam_plastic()` | K_T, f_int, states |
| `FiberAssemblyResult` | `assemble_cosserat_beam_fiber()` | K_T, f_int, states |

**設計判断**:
- `NamedTuple` を採用（`dataclass` ではなく）
  - 既存のタプルアンパッキング（`K, f = apply_dirichlet(...)`）との後方互換性維持
  - 名前付きアクセス（`result.K`, `result.f`）も可能
  - 既存の `ReturnMappingResult` / `ReturnMappingResult3D` と一貫性がある
- 既存テストへの変更なし（後方互換性により全498テストそのまま合格）

**更新ファイル**:
- `xkep_cae/solver.py` — `solve_displacement()` の戻り値型
- `xkep_cae/bc.py` — `apply_dirichlet()`, `apply_dirichlet_penalty()` の戻り値型
- `xkep_cae/elements/beam_cosserat.py` — 3つのアセンブリ関数の戻り値型
- `xkep_cae/core/__init__.py` — 新型のエクスポート追加

### ロードマップ優先順位更新

次の優先順位を設定:

1. **幾何学非線形 Updated Lagrangian (UL)** — Phase 3.4
2. **非線形動解析** — Phase 5.4（新設）: Newton-Raphson + Newmark-β
3. **数値三点曲げ試験の非線形動解析対応** — numerical_tests フレームワーク拡張

## テスト

**全体テスト結果**: 498 passed, 2 skipped（変更なし）

## コミット履歴

1. `refactor: メソッド戻り値に NamedTuple 型クラスを導入` — core/results.py 新規, solver/bc/beam_cosserat 更新
2. `docs: ロードマップ優先順位更新・status-027 作成` — roadmap/README/status 更新

## TODO（残タスク）

- [ ] Phase 3.4: Updated Lagrangian (UL) 定式化
- [ ] Phase 5.4: 非線形動解析（Newton-Raphson + Newmark-β）
- [ ] 数値三点曲げ試験の非線形動解析対応
- [ ] Phase 4.3: von Mises 3D 弾塑性テスト実行（45テスト計画済み）
- [ ] Phase 5.2: ElementProtocol への `mass_matrix()` 統合

## 確認事項・懸念

- NamedTuple はイミュータブルなので、呼び出し元で結果を変更する必要がある場合は注意。現時点で問題になる箇所はない。
- `_compute_generalized_stress_plastic` と `_compute_generalized_stress_fiber` はプライベート関数のため型クラス化はスキップした。将来必要に応じて対応可。

---

## 追記
- assetsにtest用のAbaqus3点曲げ計算結果データを添付。validation時に活用する。今後もvalidation用途でデータは拡充予定。
