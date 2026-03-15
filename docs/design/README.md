# 設計文書索引

[← README](../../README.md) | [← roadmap](../roadmap.md)

> 設計仕様書は実装コードのそばに配置（コロケーション方式）。
> 本ファイルは全設計文書へのリンク集。

## 新 xkep_cae（Process Architecture）

| 文書 | 配置先 | 内容 | 状態 |
|------|--------|------|------|
| [penalty.md](../../xkep_cae/contact/penalty/docs/penalty.md) | `xkep_cae/contact/penalty/docs/` | Penalty Strategy（k_pen推定 + 法線力） | 完了 |
| [friction.md](../../xkep_cae/contact/friction/docs/friction.md) | `xkep_cae/contact/friction/docs/` | Friction Strategy（Coulomb return mapping） | 完了 |

> 新 xkep_cae のモジュールが増えるに伴い、ここにドキュメントリンクを追加していく。

## アーカイブ（旧 xkep_cae_deprecated）

以下は `xkep_cae_deprecated/` に移行済みの旧モジュール設計文書。
脱出ポット計画で新 xkep_cae へ移行される際に新しいドキュメントが作成される。

### プロセスアーキテクチャ設計

| 文書 | 配置先 | 内容 | 対応Phase |
|------|--------|------|----------|
| [process-architecture.md](../../xkep_cae_deprecated/process/docs/process-architecture.md) | `xkep_cae_deprecated/process/docs/` | プロセスアーキテクチャ設計（AbstractProcess + Strategy） | R1 |
| [phase8-design.md](../../xkep_cae_deprecated/process/docs/phase8-design.md) | `xkep_cae_deprecated/process/docs/` | Phase 8 設計（ProcessRunner / StrategySlot / Preset） | R1 Phase 8 |
| [solve-smooth-penalty-friction.md](../../xkep_cae_deprecated/process/docs/solve-smooth-penalty-friction.md) | `xkep_cae_deprecated/process/docs/` | Smooth Penalty 摩擦ソルバー設計 | R1 |

### Strategy 設計文書

| 文書 | 配置先 | 内容 | 対応Phase |
|------|--------|------|----------|
| [penalty.md](../../xkep_cae_deprecated/process/strategies/docs/penalty.md) | `xkep_cae_deprecated/process/strategies/docs/` | Penalty Strategy（旧版） | R1 Phase 3 |
| [contact_force.md](../../xkep_cae_deprecated/process/strategies/docs/contact_force.md) | `xkep_cae_deprecated/process/strategies/docs/` | ContactForce Strategy | R1 Phase 3 |
| [friction.md](../../xkep_cae_deprecated/process/strategies/docs/friction.md) | `xkep_cae_deprecated/process/strategies/docs/` | Friction Strategy | R1 Phase 3 |
| [time_integration.md](../../xkep_cae_deprecated/process/strategies/docs/time_integration.md) | `xkep_cae_deprecated/process/strategies/docs/` | TimeIntegration Strategy | R1 Phase 3 |
| [contact_geometry.md](../../xkep_cae_deprecated/process/strategies/docs/contact_geometry.md) | `xkep_cae_deprecated/process/strategies/docs/` | ContactGeometry Strategy | R1 Phase 4 |
| [coating.md](../../xkep_cae_deprecated/process/strategies/docs/coating.md) | `xkep_cae_deprecated/process/strategies/docs/` | CoatingStrategy | status-169 |

### モジュール設計文書

| 文書 | 配置先 | 内容 | 対応Phase |
|------|--------|------|----------|
| [cosserat-design.md](../../xkep_cae_deprecated/elements/docs/cosserat-design.md) | `xkep_cae_deprecated/elements/docs/` | Cosserat Rod 設計仕様 | Phase 2 |
| [transient-output-design.md](../../xkep_cae_deprecated/output/docs/transient-output-design.md) | `xkep_cae_deprecated/output/docs/` | 過渡応答出力設計 | Phase 5 |

### 接触モジュール設計

| 文書 | 配置先 | 内容 | 対応Phase |
|------|--------|------|----------|
| [beam_beam_contact_spec_v0.1.md](../../xkep_cae_deprecated/contact/docs/beam_beam_contact_spec_v0.1.md) | `xkep_cae_deprecated/contact/docs/` | 梁–梁接触アルゴリズム全体設計 | Phase C0-C5 |
| [arc_length_contact_design.md](../../xkep_cae_deprecated/contact/docs/arc_length_contact_design.md) | `xkep_cae_deprecated/contact/docs/` | 弧長法+接触 | 凍結 |
| [contact-algorithm-overhaul-c6.md](../../xkep_cae_deprecated/contact/docs/contact-algorithm-overhaul-c6.md) | `xkep_cae_deprecated/contact/docs/` | Phase C6 大改修 | Phase C6 |
| [twisted_wire_contact_improvement.md](../../xkep_cae_deprecated/contact/docs/twisted_wire_contact_improvement.md) | `xkep_cae_deprecated/contact/docs/` | 撚線接触改善レビュー | Phase 4.7 |
| [contact-prescreening-gnn-design.md](../../xkep_cae_deprecated/contact/docs/contact-prescreening-gnn-design.md) | `xkep_cae_deprecated/contact/docs/` | 接触プリスクリーニング用GNN設計 | ペンディング |
| [kpen-estimation-ml-design.md](../../xkep_cae_deprecated/contact/docs/kpen-estimation-ml-design.md) | `xkep_cae_deprecated/contact/docs/` | k_pen最適推定MLモデル設計 | ペンディング |

### 参考資料

| 文書 | 配置先 | 内容 |
|------|--------|------|
| [abaqus-differences.md](../reference/abaqus-differences.md) | `docs/reference/` | xkep-caeとAbaqusの差異 |
| [examples.md](../reference/examples.md) | `docs/reference/` | 使用例（旧 xkep_cae API） |

---
