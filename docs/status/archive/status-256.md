# status-256: B1-B4 摩擦アセンブリProcess化 + MPC接線FD診断

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-28
- **ブランチ**: `claude/execute-status-todos-OCsu3`
- **テスト数**: 200+10s+16+3+23+1+6+18（新規18件: B1×4 + B2×3 + B3×3 + B4×3 + FD診断×5）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. B1-B4 摩擦アセンブリProcess化（Phase D後半）

status-252で「脱法カテゴリB」として列挙された4つのプライベート関数をProcess管理下に移行。

| ID | 変換前 | 変換後 | テスト | ファイル |
|----|--------|--------|--------|---------|
| B4 | `_assemble_friction_tangent_stiffness` (70行) | `FrictionTangentStiffnessProcess` | 3件 | `friction/strategy.py` |
| B2 | `_assemble_friction_geometric_stiffness` (113行) | `FrictionGeometricStiffnessProcess` | 3件 | `friction/strategy.py` |
| B3 | `_assemble_friction_st_stiffness` (95行) | `FrictionStStiffnessProcess` | 3件 | `friction/strategy.py` |
| B1 | `_add_kst_contact` (170行) | `ContactForceStStiffnessProcess` | 4件 | `contact_force/strategy.py` |

**実装パターン**: C2-C3（status-255）と同一のProcess化パターン。
- frozen dataclass Input/Output
- ProcessMeta + uses宣言
- 内部ヘルパー関数（_assembly.py / module-level）を呼び出し

**主な変更点**:
- `CoulombReturnMappingProcess.tangent()` — 3つのProcess呼び出しに変更
- `HuberContactForceProcess.tangent()` — K_st計算をProcess経由に変更
- `_add_kst_contact` — メソッドからモジュールレベル関数`_add_kst_contact_to_coo`に移動
- `_huber_deriv` — 静的メソッドのモジュールレベル版`_huber_deriv_scalar`を追加
- `__init__.py` — 12シンボル追加export（B1: 3, B2-B4: 9）

### 2. TangentFDDiagnosticProcess（MPC+接触の接線剛性FD方向診断）

status-255で提案されていたFD診断をProcess化して実装。

**目的**: NRストール時にK_Tの方向有効性をFDで検証し、接線剛性不整合の箇所を特定する。

**診断項目**:
1. **cos(R_red, K_red@du)**: MPC縮退系での方向整合性（期待: ≈-1.0）
2. **||R(u+eps*du)||/||R(u)||**: du方向の残差減少率（<1なら方向有効）
3. **MPC縮退系 K_red@du のFD vs 解析比較**: 相対誤差+DOF別エラーランキング

**使用方法**:
```python
# NewtonDynamicInput に tangent_fd_diagnostic=True を追加
config = NewtonDynamicInput(
    ...,
    tangent_fd_diagnostic=True,  # ストール検知時にFD診断実行
)
```

**NRループ統合**: `_newton_dynamic.py`にストール検知後のフックを追加。
`_relax_active and _relax_iter == 0`の条件で初回のみ実行。

| ファイル | 変更内容 |
|---------|---------|
| `solver/_newton_steps.py` | TangentFDDiagnosticProcess + Input/Output 追加 |
| `solver/_newton_dynamic.py` | ストール検知時のFD診断フック + uses宣言 |
| `solver/__init__.py` | 3シンボル追加export |

---

## テスト結果

- 新規テスト: 18件（B1×4 + B2×3 + B3×3 + B4×3 + FD診断×5）
- 既存テスト: 552 passed, 20 skipped, 1 xfailed（回帰なし）
- 契約違反: 0件
- 条例違反: 0件
- lint: 全合格

---

## 再現手順

```bash
git checkout claude/execute-status-todos-OCsu3
pip install -e .
# 高速テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"
# B1-B4テスト
python -m pytest xkep_cae/contact/friction/tests/test_assembly_process.py -v
python -m pytest xkep_cae/contact/contact_force/tests/test_st_stiffness_process.py -v
# FD診断テスト
python -m pytest xkep_cae/contact/solver/tests/test_tangent_fd_diagnostic.py -v
# 契約検証
python contracts/validate_process_contracts.py
# lint
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## 次セッションへの引き継ぎ

### 優先TODO
1. **FD診断を実際の収束テストで実行** — `tangent_fd_diagnostic=True`でstrand bending testを実行し、不整合箇所を特定
   - `python -m pytest tests/numerical_tests/test_strand_bending_convergence.py -x -s -v --timeout=600 2>&1 | tee /tmp/log-fd-diag.log`
   - ただし`compute_residual` callableの実装が必要（現状はNRループ内での解析的チェックのみ）
2. **FD診断にcompute_residual callableを追加** — NRループ内のf_int/f_c/f_ext計算をラムダとして渡す
3. **不整合箇所の修正** — FD診断結果に基づき、K_cのslave DOF成分の事前集約 or MPC変換順序の修正

### STA2 tolerance 厳格化（status-252引き継ぎ）
4. **T1 Hermite atol → 1e-5** → frozen-m完全解消後
5. **T2 beam oscillation rtol → 0.02** → 要素数≥40時

---

## 懸念・設計メモ

1. **FD診断のcompute_residual未実装**: 現在NRループ内ではcompute_residualなしで呼ばれるため、解析的チェック（cos角度、K@du整合性）のみ。完全なFD残差チェックにはNRループ外から残差関数を渡す必要がある。
2. **B1のComputeStJacobianProcessバッチ化**: B1とB3はペアごとにComputeStJacobianProcessを呼び出す。大規模モデルでのバッチ化は将来課題。
3. **Process化の効果**: 4つの関数がProcess管理下に入り、プロファイリング・トレーサビリティ・uses依存グラフが完備。機能変更なし。
