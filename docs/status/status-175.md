# status-175: 脱出ポット計画 Phase 1 — xkep_cae リネーム + PenaltyStrategy 完全書き直し

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-15
**作業者**: Claude Code
**ブランチ**: claude/migrate-contact-to-process-ZOtJR

## 概要

後方互換を完全に捨てる「脱出ポット計画」の Phase 1。
旧 `xkep_cae/` を `xkep_cae_deprecated/` にリネームし、
新 `xkep_cae/` を Process Architecture でゼロから構築開始。

**設計動機**: 旧パッケージに新コードを継ぎ足す方式では、deprecated インポートの
連鎖と契約違反の温床を根絶できない。クリーンな新パッケージから1 strategy ずつ
丁寧に書き直し、旧コードは参照用のみとする。

## 変更内容

### コミット 1: xkep_cae リネーム + 新パッケージ骨格

- `git mv xkep_cae xkep_cae_deprecated`
- 旧パッケージ内の全118箇所の内部インポートを `xkep_cae_deprecated.` に更新
- 新 `xkep_cae/` を作成（process 基盤ファイルは deprecated からコピー、インポート不可）
- 新パッケージ構成:

```
xkep_cae/
├── __init__.py
└── process/
    ├── __init__.py
    ├── base.py           ← 自己完結（deprecated からインポートしない）
    ├── categories.py
    ├── data.py
    ├── registry.py
    ├── runner.py
    ├── slots.py
    ├── testing.py
    ├── tree.py
    ├── presets.py
    └── strategies/
        ├── __init__.py
        └── protocols.py  ← 7 Protocol 定義
```

### コミット 2: PenaltyStrategy 完全書き直し

参照元（旧コード）:
- `xkep_cae_deprecated/process/strategies/penalty.py`
- `xkep_cae_deprecated/contact/law_normal.py`

新規ファイル:
```
xkep_cae/process/strategies/penalty/
├── __init__.py           ← 公開API
├── strategy.py           ← PenaltyStrategy 具象実装（3クラス）
├── law_normal.py         ← 法線力 Process 実装（純関数 + Process ラッパー）
├── docs/
│   └── penalty.md        ← 設計ドキュメント（ProcessMeta.document_path）
└── tests/
    ├── __init__.py
    ├── test_strategy.py  ← @binds_to + 意味論テスト（15テスト）
    └── test_law_normal.py ← 物理検証テスト（19テスト）
```

| クラス | 概要 |
|--------|------|
| `AutoBeamEIPenalty` | 梁曲げ剛性 EI/L³ ベースのペナルティ剛性自動推定 |
| `AutoEALPenalty` | 軸剛性 EA/L ベースのペナルティ剛性自動推定 |
| `ContinuationPenalty` | geometric/linear ランプによる段階的増加 |
| `ALNormalForceProcess` | AL 法線力（gap-based penalty + Lagrange multiplier） |
| `SmoothNormalForceProcess` | softplus 正則化 smooth penalty |

### コミット 3: PenaltyStrategy テスト（34テスト）

- `test_strategy.py`: Protocol適合、計算精度、frozen出力、エッジケース
- `test_law_normal.py`: softplus数学性質、AL物理検証、Smooth連続性、ベクトル版一致

### コミット 4: C14/C15 契約ルール追加

- **C14**: `xkep_cae/` 内から `xkep_cae_deprecated` をインポートしていないかAST検出
- **C15**: `ProcessMeta.document_path` で指定されたドキュメントが実在するか検証

## テスト結果

- ruff check: 0 error
- ruff format: 0 issue
- penalty テスト: 34 passed
- 契約違反検出: C14 OK, C15 OK（既存5件は未移行 Strategy による想定内エラー）

## 脱出ポット計画 全体ロードマップ

| status | 対象 | 状態 |
|--------|------|------|
| **175** | **基盤 + PenaltyStrategy** | **本status** |
| 176 | FrictionStrategy + law_friction | 未実施 |
| 177 | ContactForceStrategy + assembly + line_contact | 未実施 |
| 178 | ContactGeometryStrategy + geometry + broadphase | 未実施 |
| 179 | CoatingStrategy + sheath + staged_activation | 未実施 |
| 180 | solver_ncp Process ラップ | 未実施 |
| 181 | pair（データ型）+ diagnostics + graph | 未実施 |
| 182+ | 他モジュール (io, output, mesh, ...) | 未実施 |

## 制約

- `solver_ncp.py` 収束ロジック変更禁止
- 数学的出力は旧実装と一致（テストで保証）
- 新 `xkep_cae/` から旧 `xkep_cae_deprecated/` のインポート禁止（C14 で機械検出）

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `xkep_cae/` | `xkep_cae_deprecated/`（参照用のみ） | status-175 |
| `xkep_cae_deprecated/process/strategies/penalty.py` | `xkep_cae/process/strategies/penalty/strategy.py`（完全書き直し） | status-175 |
| `xkep_cae_deprecated/contact/law_normal.py` | `xkep_cae/process/strategies/penalty/law_normal.py`（純関数化） | status-175 |

## 今後の TODO

- [ ] FrictionStrategy 完全書き直し（status-176）
- [ ] ContactForceStrategy 完全書き直し（status-177）
- [ ] 残 Strategy + solver_ncp Process ラップ（status-178-180）
- [ ] 旧 CI テストの新パッケージ対応移行

---
