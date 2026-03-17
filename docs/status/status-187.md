# status-187: Phase 7 開始 — mesh/output C14 除去 + C16 クリア

[<- README](../../README.md) | [<- status-index](status-index.md)

## 日付

2026-03-16

## 作業者

Claude Code

## ブランチ

claude/execute-status-todos-OxkZv

## 概要

status-186 の TODO「Phase 7: deprecated 依存除去」を開始。mesh/process.py と output/__init__.py の C14 違反を除去し、契約違反を **4件 → 2件** に削減。C16 違反もゼロを維持。

## 変更内容

### 1. mesh/_twisted_wire.py 新設 — twisted_wire メッシュ生成関数移植

deprecated `__xkep_cae_deprecated/mesh/twisted_wire.py` から `make_twisted_wire_mesh()` と依存関数を新パッケージに移植:

- `_make_twisted_wire_mesh()`: メインファクトリ関数
- `TwistedWireMesh`, `StrandInfo`: frozen dataclass（C16 準拠）
- `_radii()`, `_n_nodes()`, `_n_elems()`: standalone 関数（旧プロパティ → 関数化）
- 全内部ヘルパー: `_helix_points`, `_straight_points`, `_compute_layer_structure`, `_minimum_strand_diameter`, `_validate_strand_geometry`, `_compute_min_safe_gap`, `_make_strand_layout`

移植方針:
- プライベートモジュール（`_twisted_wire.py`）+ 全関数 `_` prefix → C16 滅菌チェッククリア
- deprecated 版（1273行: 被膜/シース/コンプライアンス行列含む）から StrandMeshProcess が使う最小限（~370行）のみ移植
- scripts は deprecated 版を直接参照しているため影響なし

### 2. mesh/process.py 更新

- `importlib.import_module("__xkep_cae_deprecated.mesh.twisted_wire")` → `from xkep_cae.mesh._twisted_wire import` に変更
- `mesh.radii`（プロパティ）→ `_radii(mesh)`（standalone 関数）に変更

### 3. output/__init__.py — deprecated lazy-load 削除

- `__getattr__` による `__xkep_cae_deprecated.output` の遅延ロードを完全削除
- 調査の結果、新パッケージ内で `__getattr__` 経由でアクセスしているコードは **ゼロ件**
- scripts は `xkep_cae.output.render_beam_3d` 等のサブモジュールを直接指定しており、`__getattr__` は無関係

## テスト結果

- 全テスト: **284テスト合格**（変更なし）
- ruff check: 0 error
- ruff format: 0 issue
- 契約違反: **2件**（4件から2件削減）

## C14 残違反（2件）

| ファイル | 理由 | 対応予定 |
|---------|------|---------|
| `contact/setup/process.py:48` | ContactManager/ContactConfig 依存 | Phase 7 後半: ContactManager 移植 |
| `contact/solver/process.py:66` | deprecated default_strategies() 依存 | Phase 7 後半: friction/geometry stub 解消 |

## Strategy stub 状態の詳細調査

今回の作業で、新パッケージの Strategy 実装状況を精査:

| Strategy | 状態 | stub 箇所 |
|----------|------|---------|
| penalty | ✅ 完全実装 | — |
| time_integration | ✅ 完全実装 | — |
| contact_force | ✅ 完全実装 | — |
| coating | ✅ 完全実装 | — |
| **friction** | ⚠️ 部分stub | CoulombReturnMapping/SmoothPenaltyFriction の evaluate() がゼロ返却 |
| **geometry** | ⚠️ 部分stub | detect() が空リスト返却（narrowphase は実装済み） |

**根本原因**: friction.evaluate() と geometry.detect() は `ContactPair` オブジェクト（deprecated ContactManager が管理）に依存。ContactManager の新パッケージ移植が全 stub 解消の前提条件。

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `__xkep_cae_deprecated.mesh.twisted_wire.make_twisted_wire_mesh` | `xkep_cae.mesh._twisted_wire._make_twisted_wire_mesh` | status-187 |
| `output/__init__.py __getattr__` deprecated lazy-load | 完全削除（使用箇所ゼロ） | status-187 |

## TODO

- [ ] Phase 7 後半: ContactManager の新パッケージ移植（contact.pair → contact/setup/_contact_manager.py）
- [ ] Phase 7 後半: friction evaluate() stub 解消（ContactPair アクセスの実装）
- [ ] Phase 7 後半: geometry detect() stub 解消（broadphase の実装）
- [ ] Phase 7 後半: contact/solver/process.py の deprecated default_strategies() → 新版に切替
- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動の Phase 2 xfail 解消

## 懸念事項・メモ

- **ContactManager 移植が Phase 7 の最大課題**: ContactManager は ContactPair, ContactState, ContactStatus, broadphase_aabb 等の複数クラス/関数に依存。単純なコピーでは不十分で、Process Architecture に沿った再設計が必要。
- **deprecated twisted_wire.py の被膜/シース関数**: 移植対象外としたが、scripts から使われているため deprecated 側に残存。新パッケージへの移植は将来的に必要だが、ContactManager 移植より優先度低。
- **C14 残 2件は相互依存**: contact/setup（ContactManager生成） → contact/solver（strategies生成に依存）の流れで、両方とも ContactManager 移植で同時に解消可能。

---
