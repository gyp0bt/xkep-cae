# 接触モジュール設計文書

[← README](../../../README.md) | [← contact](README.md) | [← roadmap](../../../docs/roadmap.md)

梁–梁接触モジュール（Phase C）および撚線接触に関する設計文書群。

## 文書一覧

| 文書 | 内容 | 対応Phase | 状態 |
|------|------|----------|------|
| [beam_beam_contact_spec_v0.1.md](beam_beam_contact_spec_v0.1.md) | 梁–梁接触アルゴリズム全体設計（AL法, Active-set, return mapping） | Phase C0-C5 | 完了 |
| [arc_length_contact_design.md](arc_length_contact_design.md) | 接触問題でのリミットポイント追跡（弧長法+接触） | Phase C 拡張 | 凍結 |
| [contact-algorithm-overhaul-c6.md](contact-algorithm-overhaul-c6.md) | Phase C6 接触アルゴリズム大改修（NCP+Mortar+Line contact） | Phase C6 | 完了 |
| [twisted_wire_contact_improvement.md](twisted_wire_contact_improvement.md) | 撚線接触の改善レビュー（7本撚り収束困難の原因分析と対策） | Phase 4.7 L0 | 完了 |
| [contact-prescreening-gnn-design.md](contact-prescreening-gnn-design.md) | 接触プリスクリーニング用GNN設計仕様 | Phase 6 応用 | ペンディング |
| [kpen-estimation-ml-design.md](kpen-estimation-ml-design.md) | k_pen最適推定MLモデル設計仕様 | Phase 6 応用 | ペンディング |

## 実装状況

- **Phase C0-C6**: 全て実装完了（NCP+Mortar+Line contact+摩擦）
- **推奨構成**: `contact_mode="smooth_penalty"`（NCP鞍点系は摩擦接線剛性符号問題あり、status-147）
- **撚線接触改善**: 19本曲げ揺動収束達成（status-135）
- **ML設計仕様**: 設計文書のみ、実装はペンディング

---
