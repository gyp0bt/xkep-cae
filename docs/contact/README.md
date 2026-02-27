# 接触モジュール設計文書

[← README](../../README.md) | [← roadmap](../roadmap.md)

梁–梁接触モジュール（Phase C）および撚線接触に関する設計文書群。

## 文書一覧

| 文書 | 内容 | 対応Phase |
|------|------|----------|
| [beam_beam_contact_spec_v0.1.md](beam_beam_contact_spec_v0.1.md) | 梁–梁接触アルゴリズム全体設計（AL法, Active-set, return mapping, Outer/Inner分離） | Phase C0-C5 |
| [arc_length_contact_design.md](arc_length_contact_design.md) | 接触問題でのリミットポイント追跡（弧長法+接触）の設計方針 | Phase C 拡張 |
| [twisted_wire_contact_improvement.md](twisted_wire_contact_improvement.md) | 撚線接触の改善レビュー（7本撚り収束困難の原因分析と対策） | Phase 4.7 L0 |
| [contact-prescreening-gnn-design.md](contact-prescreening-gnn-design.md) | 接触プリスクリーニング用GNN設計仕様（撚線フェーズ次までペンディング） | Phase 6 応用 |
| [kpen-estimation-ml-design.md](kpen-estimation-ml-design.md) | k_pen最適推定MLモデル設計仕様（撚線フェーズ次までペンディング） | Phase 6 応用 |

## 実装状況

- **Phase C0-C5**: 全て実装完了（190テスト）
- **撚線接触改善**: 7本撚りNR収束達成（status-065）
- **ML設計仕様**: 設計文書のみ、実装はペンディング
