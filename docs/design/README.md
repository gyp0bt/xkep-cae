# 設計文書索引

[← README](../../README.md) | [← roadmap](../roadmap.md)

> 設計仕様書は実装コードのそばに配置（コロケーション方式）。
> 本ファイルは全設計文書へのリンク集。

## 設計文書一覧

| 文書 | 配置先 | 内容 | 対応Phase | 状態 |
|------|--------|------|----------|------|
| [process-architecture.md](../../xkep_cae/process/process-architecture.md) | `xkep_cae/process/` | プロセスアーキテクチャ設計（AbstractProcess + Strategy） | R1 | Phase 1完了 |
| [cosserat-design.md](../../xkep_cae/elements/cosserat-design.md) | `xkep_cae/elements/` | Cosserat Rod 設計仕様（幾何学的厳密梁） | Phase 2 | 完了 |
| [transient-output-design.md](../../xkep_cae/output/transient-output-design.md) | `xkep_cae/output/` | 過渡応答出力インターフェース設計 | Phase 5 | 完了 |

### 接触モジュール設計

| 文書 | 配置先 | 内容 | 対応Phase | 状態 |
|------|--------|------|----------|------|
| [beam_beam_contact_spec_v0.1.md](../../xkep_cae/contact/beam_beam_contact_spec_v0.1.md) | `xkep_cae/contact/` | 梁–梁接触アルゴリズム全体設計 | Phase C0-C5 | 完了 |
| [arc_length_contact_design.md](../../xkep_cae/contact/arc_length_contact_design.md) | `xkep_cae/contact/` | 弧長法+接触 | Phase C 拡張 | 凍結 |
| [contact-algorithm-overhaul-c6.md](../../xkep_cae/contact/contact-algorithm-overhaul-c6.md) | `xkep_cae/contact/` | Phase C6 大改修 | Phase C6 | 完了 |
| [twisted_wire_contact_improvement.md](../../xkep_cae/contact/twisted_wire_contact_improvement.md) | `xkep_cae/contact/` | 撚線接触改善レビュー | Phase 4.7 L0 | 完了 |
| [contact-prescreening-gnn-design.md](../../xkep_cae/contact/contact-prescreening-gnn-design.md) | `xkep_cae/contact/` | 接触プリスクリーニング用GNN設計 | Phase 6 応用 | ペンディング |
| [kpen-estimation-ml-design.md](../../xkep_cae/contact/kpen-estimation-ml-design.md) | `xkep_cae/contact/` | k_pen最適推定MLモデル設計 | Phase 6 応用 | ペンディング |

### 参考資料

| 文書 | 配置先 | 内容 |
|------|--------|------|
| [abaqus-differences.md](../reference/abaqus-differences.md) | `docs/reference/` | xkep-caeとAbaqusの差異 |
| [examples.md](../reference/examples.md) | `docs/reference/` | 使用例（高レベルAPI） |

---
