# status-253: DOF消去MPC実装 + 7本撚線曲げ揺動Process

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-27
- **ブランチ**: `claude/execute-status-todos-7J5f7`
- **テスト数**: 200+10s+16+3+23（新規23件）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. DOF消去MPC実装（Phase B完了）

MPCEliminationProcess: 梁端部slave節点群をmaster参照点に剛体結合する変換行列Tを構築するPreProcess。

**制約式（6DOF/node）**:
- 並進: `u_slave = u_master + [r]× θ_master`
- 回転: `θ_slave = θ_master`
- 変換: `K_red = T^T K T`, `f_red = T^T f`, `du_full = T @ du_red`

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/constraints/mpc_elimination.py` | **新規** — MPCEliminationProcess + MPCGroup + frozen I/O |
| `xkep_cae/constraints/__init__.py` | re-export追加 |
| `xkep_cae/constraints/tests/test_mpc_elimination.py` | **新規** — C3テスト14件 |
| `docs/constraints.md` | MPC仕様追記 |

### 2. ソルバーMPC統合

LinearSolveProcessにMPC DOF消去を統合。T^T K T の縮退系でソルブし、T @ du_red で全体系復元。

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/core/data.py` | BoundaryData に `mpc_transform` フィールド追加 |
| `xkep_cae/contact/solver/_newton_steps.py` | LinearSolveProcess に `_solve_with_mpc()` 追加、LinearSolveInput に `mpc_transform` |
| `xkep_cae/contact/solver/_newton_dynamic.py` | NewtonDynamicStepInput に `mpc_transform` パスルー |
| `xkep_cae/contact/solver/process.py` | ContactFrictionProcess → NewtonDynamic 伝搬 |
| `tests/constraints/test_mpc_solver_integration.py` | **新規** — ソルバー統合テスト4件 |

### 3. StrandBendingOscillationProcess（Phase C完了）

7本撚線の曲げ揺動解析を実行するBatchProcess。

**パイプライン**:
1. StrandMeshProcess でメッシュ生成
2. 端部参照点ノード追加（左端/右端の重心位置）
3. MPCEliminationProcess で端部剛体結合
4. ULCRBeamAssemblerProcess でアセンブラ構築（拡張系ラッパー付き）
5. 曲げ処方変位（右端θ_z）+ 揺動サイクル設定
6. ContactFrictionProcess で求解

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | **新規** — StrandBendingOscillationProcess |
| `xkep_cae/numerical_tests/__init__.py` | re-export追加 |
| `xkep_cae/numerical_tests/docs/strand_bending_oscillation.md` | **新規** — ドキュメント |
| `xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py` | **新規** — C3テスト5件 |

---

## テスト結果

- 新規テスト: 23件（MPC 14件 + ソルバー統合 4件 + StrandBending 5件）
- 既存テスト: 528 passed, 20 skipped, 1 xfailed（全合格）
- 契約違反: 0件
- 条例違反: 0件
- lint: 全合格
- 既知の無関係失敗: `test_stress_contour.py`（matplotlib依存）、`tests/contact/test_st_jacobian.py`（C3重複@binds_to既知）

---

## 再現手順

```bash
git checkout claude/execute-status-todos-7J5f7
pip install -e .
# MPC単体テスト
python -m pytest xkep_cae/constraints/tests/test_mpc_elimination.py -x -q
# ソルバー統合テスト
python -m pytest tests/constraints/test_mpc_solver_integration.py -x -q
# StrandBendingOscillation テスト
python -m pytest xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py -x -q
# 契約検証
python contracts/validate_process_contracts.py
# lint
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## 次セッションへの引き継ぎ

### 優先TODO
1. **7本撚線曲げ揺動の収束実行テスト** — StrandBendingOscillationProcess を実際に動かし、収束を確認
   - tee でログ出力必須
   - メッシュ: E=130MPa, ρ=8.96e-9, R=0.5mm, pitch=100mm
   - 曲げ曲率 κ=0.001 1/mm, n_cycles=1
2. **MPC + 動的ソルバーの力残差整合性確認** — MPC使用時のR_u[slave_dofs]の扱い
   - 現状: T^T R_u で自動集約されるため明示的ゼロ化不要の設計
   - 要検証: ContactForceAssemblyProcess の `R_u[fixed_dofs] = 0` がslave DOFと競合しないか

### 脱法対応（中期、status-252引き継ぎ）
3. **M1-M2 幾何Process化** → MPC収束確認後
4. **B1-B4 摩擦アセンブリProcess化** → NRリファクタリング時
5. **W2 収束判定厳格化** → 全基準同時満足への移行検討

### STA2 tolerance 厳格化（条件付き、status-252引き継ぎ）
6. **T1 Hermite atol → 1e-5** → frozen-m完全解消後
7. **T2 beam oscillation rtol → 0.02** → 要素数≥40時

---

## 懸念・設計メモ

1. **MPC + fixed_dofs の相互作用**: 現在の実装ではslave DOFをfixed_dofsに含めない前提。slave DOFはT変換で消去されるため、fixed_dofsに含まれるとBC二重適用になる。StrandBendingOscillationProcess では端部参照点のみをfixed/prescribedに設定し、slave DOFは含めていない。
2. **拡張系ラッパーのオーバーヘッド**: _assemble_tangent_extended は撚線部分の剛性行列を拡張系にゼロパディング。大規模モデルでは疎行列操作のコストが問題になる可能性あり。
3. **揺動周期の設定**: 現在は固有周期の10倍を揺動周期としているが、準静的挙動を保証するには検証が必要。
