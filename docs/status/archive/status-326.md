# status-326: ファイバー梁 Phase F1 実装 + culling/cache 効果定量計測

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-12
- **ブランチ**: `claude/execute-status-todos-YvRwF`
- **テスト数**: 459+13+22+5+8+12+12（Fiber1D API 6 + Physics 6 テスト追加）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

## TL;DR

status-325 TODO の直近3項目を実行:

1. **ファイバー梁 Phase F1 完了** — `Elastic1D` / `BilinearKinematicHardening1D` 実装 + 12 テスト全合格
2. **n=7/19/37 掃引で culling + cache 効果を定量計測** — status-319 ベースラインと比較し、ContactForceStStiffness **96-99% 高速化**、FrictionStStiffness **96-98% 高速化**を確認
3. 被膜 ON プロファイルは pypardiso 未インストール環境のため保留

---

## 1. ファイバー梁 Phase F1 実装

### 概要

status-313 設計仕様 Phase F1 の完了判定:
> `fiber/state.py` + `fiber/materials.py` に `Elastic1D`, `BilinearKinematicHardening1D` 実装、Physics テスト 6 件合格

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/elements/fiber/__init__.py` | 新規：モジュール公開 API |
| `xkep_cae/elements/fiber/state.py` | 新規：`Fiber1DState`, `SectionState` frozen dataclass |
| `xkep_cae/elements/fiber/materials.py` | 新規：`Elastic1D`, `BilinearKinematicHardening1D` |
| `xkep_cae/core/strategies/protocols.py` | `Fiber1DMaterialStrategy` Protocol 追加 |
| `xkep_cae/elements/fiber/tests/__init__.py` | 新規 |
| `xkep_cae/elements/fiber/tests/test_materials_api.py` | 新規：API テスト 6 件 |
| `xkep_cae/elements/fiber/tests/test_materials_physics.py` | 新規：Physics テスト 6 件 |

### 設計の要点

- **Fiber1DState**: frozen dataclass。`eps_p`（塑性ひずみ）+ `alpha`（背応力）+ `slip`/`slipped`（Phase F2 用予約）
- **Elastic1D**: σ = E·ε、状態不変。参照用。
- **BilinearKinematicHardening1D**: Prager 移動硬化。return mapping で `(eps_p, alpha)` を更新。一貫接線 `E_t = E·H/(E+H)`。
- **Fiber1DMaterialStrategy Protocol**: `evaluate(eps, state) → (sigma, dsigma_deps, new_state)` — 既存の ContactForceStrategy / FrictionStrategy と同じ作法で `@runtime_checkable`。
- **C17 準拠**: evaluate() は新しい frozen state を返す。入力 state は mutation しない。

### テスト（12 件）

| テスト | 分類 | 内容 |
|--------|------|------|
| `test_elastic1d_protocol_compliance` | API | Protocol `isinstance` 検査 |
| `test_bilinear_kh_protocol_compliance` | API | Protocol `isinstance` 検査 |
| `test_evaluate_return_types` | API | 戻り値型検査 |
| `test_state_frozen` | API | Fiber1DState mutation 不可 |
| `test_section_state_frozen` | API | SectionState mutation 不可 |
| `test_elastic_monotonic_sigma_eq_E_eps` | API | σ = E·ε 再現 |
| `test_uniaxial_cycle_residual_strain` | Physics | 閉ループ + 残留ひずみ |
| `test_perfect_elastoplastic_yield` | Physics | H=0 で σ_y 保持 |
| `test_energy_balance_loop_area` | Physics | 塑性仕事 = ループ面積 |
| `test_consistent_tangent_fd` | Physics | FD 接線検証 atol=1e-3 |
| `test_elastic_state_unchanged` | Physics | 弾性域で状態不変 |
| `test_kh_strand_friction_equivalence` | Physics | KH ≡ 撚線摩擦（Bauschinger 効果） |

---

## 2. n=7/19/37 掃引 — culling + cache 効果定量計測

### 概要

status-319 と同一条件（gap=0.07 固定、κ=0.005、n_inc=10）で n=7/19/37 の 3 ケース掃引を実施。
status-324（K_st distance culling）+ status-325（symbolic factorization reuse）+ status-322（_find_caller 高速化）+ status-321（K_st CSR/COO 最適化）等の累積効果を計測。

### 実測テーブル

| n_strands | ndof | n_inc | 総時間 [s] | dominant_leaf |
|-----------|------|-------|-----------|--------------|
| 7 | 222 | 36 | 8.04 | TangentAssemblyProcess |
| 19 | 582 | 23 | 14.04 | TangentAssemblyProcess |
| 37 | 1122 | 3 | 17.45 | TangentAssemblyProcess |

### status-319 ベースラインとの per-call 比較

| n_strands | Process | s319 [ms] | s326 [ms] | ratio | 改善 |
|-----------|---------|-----------|-----------|-------|------|
| 7 | TangentAssembly | 110.5 | 8.0 | 0.07x | **+92.7%** |
| 7 | ContactForceStStiffness | 29.4 | 1.0 | 0.04x | **+96.4%** |
| 7 | FrictionStStiffness | 31.3 | 1.3 | 0.04x | **+96.0%** |
| 19 | TangentAssembly | 170.9 | 16.9 | 0.10x | **+90.1%** |
| 19 | ContactForceStStiffness | 51.7 | 1.4 | 0.03x | **+97.2%** |
| 19 | FrictionStStiffness | 51.3 | 1.7 | 0.03x | **+96.7%** |
| 37 | TangentAssembly | 512.7 | 41.9 | 0.08x | **+91.8%** |
| 37 | ContactForceStStiffness | 204.9 | 3.2 | 0.02x | **+98.5%** |
| 37 | FrictionStStiffness | 199.9 | 4.1 | 0.02x | **+97.9%** |

### scaling 分析

**per-call avg の n_strands スケーリング指数 α（avg ∝ n^α）**:

| 区間 | TangentAssembly | ContactForceStStiffness | FrictionStStiffness |
|------|----------------|------------------------|---------------------|
| **status-319 (19→37)** | **1.65** | **2.07** | **2.04** |
| **status-326 (19→37)** | **1.36** | **1.24** | **1.32** |

**ContactForceStStiffness の n² (α≈2.07) が α≈1.24 に低減**。distance culling により、gap 閾値を超えるペアのアセンブリスキップが効いている。

### 注意事項

- 比較はクロスマシン（status-319 と status-326 は異なる環境）。絶対値の比較は参考。
- **scaling 指数の変化が本質的な改善指標**。α=2.07→1.24 は culling の構造的効果を示す。
- pypardiso 未インストールのため、symbolic factorization reuse（status-325）の効果は計測不可。scipy fallback での結果。
- n=19, n=37 は Type D stall で未収束（status-319 と同様）。per-call 時間は有効なインクリメント分のプロファイルから取得。

### 再現手順

```bash
# 掃引実行
PYTHONPATH=. python work/strand_profiling/status326_culling_cache_sweep.py \
    2>&1 | tee /tmp/log-status326-$(date +%s).log

# Phase F1 テスト
pytest xkep_cae/elements/fiber/tests/ -v

# 契約検証
python contracts/validate_process_contracts.py
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## 3. 被膜 ON プロファイル + pypardiso 環境再ベンチ

pypardiso 未インストール環境のため**保留**。次担当者の pypardiso 環境で実施。

---

## TODO（次担当者向け）

### 直近

- [ ] **被膜 ON プロファイル + pypardiso 環境再ベンチ** — status-320 TODO 継続。pypardiso 環境で _SolverCache の symbolic reuse 効果を定量計測
- [ ] **ファイバー梁 Phase F2 着手** — `MultiLayerFrictionDegrading1D` 実装（frozen 化込み）、`05_smooth_teardrop.py` 再現 rtol 1%
- [ ] **n=61+ Type D stall 対策** — 大 n_strands での接触活性化時に Type D stall が発生。K_st adj 項の再検討が必要

### 中期

- [ ] **リスタート解析方式への移行**: ContactFrictionProcess の I/O を `(u, v, a, 接触ペア)` 入出力に整理
- [ ] **ProcessMetaclass._profile_data と ProcessExecutionLog の統合** — status-322 TODO 継続
- [ ] **空間ブロック分離 or ペアクラスタリング**: 物理的接触ペア数の n² 成長を抑制する構造的対策

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: pytest -v 出力 + 掃引ログで確認
- [x] **再現手順記載**: 上記
- [x] **テスト数記載**: 459+13+22+5+8+12+12
- [x] **契約違反 0 件維持**: validate_process_contracts.py 実行済み
- [x] **lint/format 検証**: ruff check + ruff format --check OK
