# status-257: FD診断compute_residual実装 + 接線剛性K_c不整合特定

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-28
- **ブランチ**: `claude/execute-status-todos-Fc66E`
- **テスト数**: 200+10s+16+3+23+1+6+18+2（新規2件: MPC+compute_residual, 不整合検出）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. compute_residual callable 実装（status-256 TODO 2）

NRループ内にクロージャ `_compute_residual_at` を実装し、TangentFDDiagnosticProcess に渡すように修正。

**変更ファイル**:

| ファイル | 変更内容 |
|---------|---------|
| `solver/_newton_dynamic.py` | `_compute_residual_at` クロージャ追加（ContactForceAssemblyProcess + 動的項） |
| `solver/_newton_steps.py` | FD比較をR(u)もcompute_residualで再計算する方式に修正（リラクゼーション差異排除）|
| `solver/_newton_steps.py` | 全体系FD vs 解析比較を追加（MPC変換 vs K_c自体の切り分け）|
| `solver/_newton_steps.py` | R_u vs compute_residual(u) の乖離報告を追加 |
| `core/data.py` | `ContactFrictionInputData` に `tangent_fd_diagnostic` フィールド追加 |
| `contact/solver/process.py` | `tangent_fd_diagnostic` の伝搬追加 |
| `numerical_tests/strand_bending_oscillation.py` | `StrandBendingOscillationConfig` に `tangent_fd_diagnostic` 追加 |

### 2. FD診断テスト追加

| テスト | 内容 |
|--------|------|
| `test_mpc_with_compute_residual` | MPC + compute_residual 組み合わせで線形系FD/解析一致 |
| `test_inconsistent_tangent_detected` | 意図的不整合K_Tの検出確認 |

### 3. FD診断実行結果 — K_c不整合の決定的特定

7本撚線曲げ揺動テスト(`tangent_fd_diagnostic=True`)でFD診断を3回実行し、以下を確認。

#### 主要な発見

1. **全体系での相対誤差: 94〜100%** → **K_c自体が不正確**（MPC変換は原因ではない）
2. **cos(R_red, K_red@du) ≈ -0.08〜0.19** (期待: -1.0) → Newton方向が実質的に無効
3. **方向有効性 = 1.0** → du方向で残差が全く減少しない

#### 不整合DOFの物理的同定

| 全体系DOF | node | strand | comp | FD値 | 解析値 | 倍率 |
|-----------|------|--------|------|------|--------|------|
| 343 | 57 | 3 | y | -0.75 | -0.015 | 50x |
| 649 | 108 | 6 | y | +0.75 | +0.015 | 50x |
| 138 | 23 | 1 | x | -0.68 | -0.009 | 76x |
| 444 | 74 | 4 | x | +0.68 | +0.009 | 76x |
| 546 | 91 | 5 | x | +0.61 | +0.017 | 36x |
| 240 | 40 | 2 | x | -0.61 | -0.017 | 36x |

**パターン**: 対称ストランドペア (1/4, 2/5, 3/6) の中間スパン部の並進DOF。接触力の変化がK_cに反映されていない。

#### R_u vs compute_residual(u) の乖離

1e-6 〜 1e-3 と微小 → リラクゼーションによる乖離は主因ではない。

#### 診断ログ

```
/tmp/log-fd-diag3-*.log
```

---

## 診断の結論

**接触接線剛性 K_c は、接触ペアの相手側ノードへの力の感度を適切に表現していない。**

- K_c at contact DOFs ≈ 0（解析値 1e-2 〜 1e-4）
- 実際の感度（FD）は 50〜1000 倍大きい
- 結果: Newton方向が接触力の変化を無視 → 収束不能

### 推定原因

HuberContactForceProcess.tangent() の接線剛性が:
1. ペナルティ剛性 k_pen × (normal ⊗ normal) のみで、**幾何学的接線項**（法線方向変化、ギャップの変形依存）を欠いている
2. **クロスエレメント結合**（接触ペアの相手要素DOFへの微分）が不足
3. Line-to-line Gauss積分のHermite補間による4ノード結合が接線に反映されていない

---

## テスト結果

- 新規テスト: 2件（MPC+compute_residual, 不整合検出）
- 既存テスト: 554 passed, 20 skipped, 1 xfailed（回帰なし）
- 契約違反: 0件
- lint: 全合格

---

## 再現手順

```bash
git checkout claude/execute-status-todos-Fc66E
pip install -e .
# FD診断テスト
python -m pytest xkep_cae/contact/solver/tests/test_tangent_fd_diagnostic.py -v
# FD診断付き収束テスト
python -c "
from xkep_cae.numerical_tests.strand_bending_oscillation import *
cfg = StrandBendingOscillationConfig(
    n_strands=7, wire_radius=0.5, pitch_length=100.0,
    n_elements_per_pitch=16, n_pitches=1.0, E=130e3, nu=0.3,
    rho=8.96e-9, bending_curvature=0.001, n_cycles=1,
    n_increments_per_cycle=40, mu=0.15, max_nr_attempts=50,
    tol_force=1e-8, exclude_same_strand=True,
    tangent_fd_diagnostic=True,
)
result = StrandBendingOscillationProcess().process(cfg)
" 2>&1 | tee /tmp/log-fd-diag.log
# 高速テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"
# 契約検証
python contracts/validate_process_contracts.py
# lint
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## 次セッションへの引き継ぎ

### 優先TODO（K_c修正）

1. **HuberContactForceProcess.tangent() の調査** — 現在の接線剛性が何を計算しているか確認
   - `xkep_cae/contact/contact_force/strategy.py` の tangent() メソッド
   - ContactForceStStiffnessProcess（B1）の実装
   - line-to-line Gauss積分の接線への反映
2. **FD対解析のコンポーネント分離** — K_struct, K_c, K_fric, K_dynamic を個別にFDと比較
   - TangentAssemblyProcess の出力を拡張し、各コンポーネントを返す
3. **K_c の幾何学的接線項追加** — ∂g/∂u（ギャップのDOF微分）、∂n/∂u（法線方向のDOF微分）
4. **クロスエレメント結合の追加** — 接触ペアの両要素のDOFへの微分

### STA2 tolerance 厳格化（status-252引き継ぎ）

5. **T1 Hermite atol → 1e-5** → frozen-m完全解消後
6. **T2 beam oscillation rtol → 0.02** → 要素数≥40時

---

## 懸念・設計メモ

1. **compute_residual の副作用**: `ContactForceAssemblyProcess` 内の `UpdateGeometryProcess` が `manager.pairs` を in-place 変更する。FD診断後にジオメトリが摂動状態のまま残る。ストール後は次のNR反復で再計算されるため実害は限定的だが、将来的にはmanager状態のスナップショット/リストアが必要。
2. **動的項のFD**: compute_residual は `_time_strategy.correct(u_eval)` を呼ばないため、FDは**静的接線剛性のみ**をチェックする。動的項（c0*M + c1*C）はキャンセルされる。これはK_cの検証には十分だが、動的接線の整合性は別途検証が必要。
3. **eps = 1e-7 の妥当性**: dt cutback後の小さなduでFD値が1e3に達するケースあり。epsを||du||に適応させるスキームの検討余地あり。
