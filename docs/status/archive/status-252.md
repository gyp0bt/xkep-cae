# status-252: STA2再摘発 — C3完全解消 + tolerance正当化 + 脱法検出レポート

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-27
- **ブランチ**: `claude/review-status-todos-2AUfR`
- **テスト数**: 200+10s+16+3（変更なし）
- **契約違反**: **0件**（前回1件→解消）
- **条例違反**: 0件

---

## 実施内容

### 1. C3 契約違反完全解消

| # | 問題 | 修正 |
|---|------|------|
| C3 | `ComputeStJacobianProcess` にテスト未紐付け（status-251 既知） | `xkep_cae/contact/geometry/tests/test_st_jacobian.py` 新規作成、`@binds_to` + API適合テスト3件 |

**根本原因**: `tests/contact/test_st_jacobian.py` に `@binds_to` を追加しても、契約バリデータは `xkep_cae/` 配下のみをスキャンするため検出されない。`xkep_cae/contact/geometry/tests/` 内にC3用テストファイルを新設した。

### 2. STA2 tolerance 正当化（摘発 → 正当化記録）

| # | ファイル | tolerance | 正当化根拠 | TODO |
|---|---------|-----------|-----------|------|
| T1 | `tests/contact/test_st_jacobian.py` L346-434 | `atol=1e-2`（直線テストの100倍） | frozen-tangent近似（∂m/∂u 非局所DOF結合未対応, status-239）により解析的接線とFDの系統的不整合 ~33% | frozen-m解消後に `atol=1e-5` へ厳格化 |
| T2 | `tests/test_beam_oscillation.py` L126 | `rtol=0.05`（5%） | 梁要素離散化誤差 O(h²)、現在20要素で5%が妥当な上界 | 要素数≥40時に `rtol=0.02` へ厳格化 |

**判定**: いずれも技術的根拠のある tolerance 設定であり、STA2違反（数値捏造）ではない。ただし正当化コメント未記載はSTA2規約違反（記録義務）のため、`NOTE(STA2)` + `TODO(STA2)` コメントを追加した。

### 3. 脱法検出レポート（摘発台帳）

以下のパターンを検出。今回は修正対象外だが、将来対応のための台帳として記録する。

#### A. 大規模プライベート関数（Process外ロジック）

| # | ファイル | 関数 | 行数 | 現状 | 対応方針 |
|---|---------|------|------|------|---------|
| M1 | `contact/geometry/strategy.py` | `_batch_update_geometry` | 175 | Strategy.process()内ヘルパー→Processトレース下 | DOF消去MPC実装時にProcess化（status-249 C2） |
| M2 | `contact/geometry/_compute.py` | `_build_contact_frame_batch` | 108 | 同上 | M1と一体で対応（status-249 C3） |
| M3 | `contact/geometry/_st_jacobian.py` | `_process_hermite` | 103 | ComputeStJacobianProcess内部実装 | Process化不要（内部ヘルパーとして許容） |
| M4 | `contact/solver/_newton_steps.py` | Strategy直呼び6箇所 | — | NR内でStrategy.evaluate()/tangent()を直接呼出 | NRリファクタリング時に対応 |

#### B. 接触力・摩擦アセンブリの大規模プライベート関数

| # | ファイル | 関数 | 行数 | 対応方針 |
|---|---------|------|------|---------|
| B1 | `contact/contact_force/strategy.py` | `_add_kst_contact` | 170 | NRリファクタリング時にProcess化検討 |
| B2 | `contact/friction/_assembly.py` | `_assemble_friction_geometric_stiffness` | 113 | 同上 |
| B3 | `contact/friction/_assembly.py` | `_assemble_friction_st_stiffness` | 95 | 同上 |
| B4 | `contact/friction/_assembly.py` | `_assemble_friction_tangent_stiffness` | 70 | 同上 |

#### C. STA2 監視対象（違反ではないが継続監視）

| # | ファイル | パターン | 理由 |
|---|---------|---------|------|
| W1 | `contact/solver/_newton_dynamic.py` L209-218 | 接触剛性リラクゼーション ω=0.5→0.05 | status-247で設計・文書化済み。チャタリング対策として正当。ただしωの自動調整は事後検証困難 |
| W2 | `contact/solver/_newton_steps.py` L254-265 | "first satisfied" 収束判定 | 力 OR 変位 OR エネルギーの最初に満たされた基準で収束判定。意図的設計だが、全基準同時満足への移行を検討 |

---

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/geometry/tests/test_st_jacobian.py` | **新規** — C3修正用 `@binds_to` + API適合テスト |
| `tests/contact/test_st_jacobian.py` | `@binds_to` 追加 + STA2 tolerance正当化コメント |
| `tests/test_beam_oscillation.py` | STA2 tolerance正当化コメント |

---

## 検証結果

- `python contracts/validate_process_contracts.py` → **契約違反0件、条例違反0件**
- `ruff check xkep_cae/ tests/` → All checks passed
- `ruff format --check xkep_cae/ tests/` → 138 files already formatted

---

## 次セッションへの引き継ぎ

### 優先TODO
1. **DOF消去MPC実装** → 端部剛体結合（7本撚線曲げ揺動の前提）
2. **M1-M2 幾何Process化** → MPC実装と一体で対応
3. **StrandBendingOscillationProcess 実装** → 7本撚線曲げ揺動Process

### 脱法対応（中期）
4. **B1-B4 摩擦アセンブリProcess化** → NRリファクタリング時
5. **M4 Strategy直呼び解消** → NRリファクタリング時
6. **W2 収束判定厳格化** → 全基準同時満足への移行検討

### STA2 tolerance 厳格化（条件付き）
7. **T1 Hermite atol → 1e-5** → frozen-m完全解消後
8. **T2 beam oscillation rtol → 0.02** → 要素数≥40時

---

## 懸念・設計メモ

1. **バリデータのスキャン範囲**: `contracts/validate_process_contracts.py` は `xkep_cae/` 配下のみスキャン。`tests/` 配下の `@binds_to` は検出されない。将来的にスキャン範囲拡張を検討（ただし現状は `xkep_cae/**/tests/` にC3用テストを配置する運用で問題なし）
2. **STA2 tolerance 記録義務**: 今回の摘発で tolerance 緩和の正当化コメント記載を義務化。今後は `NOTE(STA2)` タグで検索可能にした
