# status-152: ドキュメント大整理 — コロケーション配置移行 + 索引整備

[← status-index](status-index.md) | [← README](../../README.md)

**日付**: 2026-03-10
**テスト数**: 2310（変更なし）

## 概要

設計仕様書を新設計ルール（コロケーション方式: テスト・仕様書は実装コードのそばに配置）に基づき再配置。
散在していたドキュメントをタイプ別フォルダに整理し、全リンクを更新。

## 変更内容

### 1. 設計文書のコロケーション移行

設計仕様書を `docs/` から実装コードのそばに移動:

| 旧パス | 新パス | 理由 |
|--------|--------|------|
| `docs/cosserat-design.md` | `xkep_cae/elements/cosserat-design.md` | 要素モジュールの設計仕様 |
| `docs/transient-output-design.md` | `xkep_cae/output/transient-output-design.md` | 出力モジュールの設計仕様 |
| `docs/design/process-architecture.md` | `xkep_cae/process/process-architecture.md` | プロセスモジュールの設計仕様 |
| `docs/contact/*.md`（6ファイル） | `xkep_cae/contact/*.md` | 接触モジュールの設計仕様 |

### 2. 参考資料フォルダの新設

| 旧パス | 新パス |
|--------|--------|
| `docs/abaqus-differences.md` | `docs/reference/abaqus-differences.md` |
| `docs/examples.md` | `docs/reference/examples.md` |

### 3. 索引ファイルの作成・更新

| ファイル | 内容 |
|---------|------|
| `docs/design/README.md` | 設計文書索引（全ファイルへのリンク集） |
| `docs/reference/README.md` | 参考資料索引 |
| `xkep_cae/contact/design-index.md` | 接触設計文書の一覧・実装状況 |

### 4. リンク更新

以下のファイルで旧パスへの参照を新パスに修正:

| ファイル | 修正内容 |
|---------|---------|
| `README.md` | 設計文書一覧・使用例・Abaqus差異のリンク |
| `docs/roadmap.md` | process-architecture.mdリンク |
| `docs/status/status-150.md` | 設計仕様書リンク |
| `docs/status/status-151.md` | 設計仕様書テキスト参照 |
| `xkep_cae/contact/README.md` | 設計仕様書リンク（相対パスに変更） |
| `xkep_cae/elements/README.md` | cosserat-design/abaqus-differencesリンク |
| `xkep_cae/process/**/*.py`（9ファイル） | docstring内のパス参照 |

### 5. ドキュメント構造（移行後）

```
docs/
├── design/
│   └── README.md          ← 設計文書索引（コロケーション先へのリンク集）
├── reference/
│   ├── README.md          ← 参考資料索引
│   ├── abaqus-differences.md
│   └── examples.md
├── verification/          ← 検証画像・文書（変更なし）
├── status/                ← ステータスファイル（変更なし）
├── archive/               ← 完了フェーズ（変更なし）
└── roadmap.md             ← ロードマップ

xkep_cae/
├── contact/
│   ├── README.md
│   ├── design-index.md                      ← NEW
│   ├── beam_beam_contact_spec_v0.1.md       ← moved
│   ├── arc_length_contact_design.md         ← moved
│   ├── contact-algorithm-overhaul-c6.md     ← moved
│   ├── twisted_wire_contact_improvement.md  ← moved
│   ├── contact-prescreening-gnn-design.md   ← moved
│   └── kpen-estimation-ml-design.md         ← moved
├── elements/
│   ├── README.md
│   └── cosserat-design.md                   ← moved
├── output/
│   └── transient-output-design.md           ← moved
└── process/
    └── process-architecture.md              ← moved
```

## 設計判断

| 判断 | 選択 | 理由 |
|------|------|------|
| 設計文書配置 | コロケーション（実装コードのそば） | status-150設計ルール準拠。発見性・保守性向上 |
| 参考資料 | docs/reference/ に分離 | 特定モジュールに属さない横断的文書 |
| docs/design/ | 索引のみ残す | 全設計文書への入口として維持 |
| アーカイブ status | 旧パスのまま | 歴史的記録として変更不要 |

## TODO（次セッション）

- [ ] Phase 2 開始: Strategy 具象実装（status-151 TODO継承）
- [ ] 全テストのmm-ton-MPa移行（status-149 TODO継承）
- [ ] 被膜摩擦μ=0.25の収束達成（status-149 TODO継承）
- [ ] 19本→37本のスケールアップ（status-149 TODO継承）

## 懸念事項

- status-101で「docs/contact/の移動は既存リンクを壊すリスクがあるため見送り」としていたが、今回コロケーション方針に基づき移動を実施。アーカイブ内の旧パス参照は歴史的記録として残す。

---
