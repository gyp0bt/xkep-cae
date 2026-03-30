# status-271: frozen=False検証 + Hermite非局所∂g/∂u Step1実装

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-30
- **ブランチ**: `claude/check-status-todos-wIZE5`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2（新規2件）→ **合計594 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. frozen_hermite_tangent=False + n_elems=20 検証

status-270のTODO「frozen=False + n_elems=20の組み合わせ効果未検証」を実施。

#### 短縮テスト（50incr）

| 条件 | frac/50incr | incr | cutback |
|------|-------------|------|---------|
| frozen=True, n_elems=20 | 0.0590 | 50 | 39 |
| frozen=False, n_elems=20 | 0.0585 | 50 | 38 |

→ 50incr時点では**差異なし**。接触確立フェーズではfrozen=True/Falseの差は現れない。

#### フルテスト（2000incr）

| 条件 | frac | incr | cutback | 備考 |
|------|------|------|---------|------|
| status-270 Hermite OFF | 1.0000 | 919 | 727 | ベースライン |
| status-270 Hermite ON (frozen=True) | 1.0000 | 929 | 682 | 凍結接線 |
| **status-271 Hermite ON (frozen=False)** | **1.0000** | **607** | **389** | **35%高速（incr）、43%カットバック減** |

frozen=Falseは frozen=True 比で **incr 35%減、cutback 43%減**。Hermite接線の∂m/∂u補正（evaluate()時）が収束を大幅に改善。n_elems=20復元との相乗効果。

### 2. NR力収束の現状分析

50incrテストログから力収束パターンを定量分析。

| 指標 | 値 |
|------|------|
| 力収束達成数 | **0件 / 100incr** |
| 変位収束達成数 | 79件 |
| エネルギー収束達成数 | 2件 |
| 最小力残差比 ||R_t||/||f|| | ~1.3e-4 |
| 目標力残差比 | 1e-8 |
| ギャップ | **4桁**（1e-4 → 1e-8） |

**根本原因**: 接触活性集合変化がNR反復ごとに残差を不連続に変動させ、力残差が~1e-4で平坦化。変位収束（tol_disp=1e-8）で先に脱出する。

**改善の方向性**:
1. Hermite非局所∂g/∂u対応（接線剛性精度向上）← 本statusで Step1 実装
2. NR更新方向のスケーリング（回転/並進DOF分離）
3. char_length有効化による重み付きノルム

### 3. Hermite非局所∂g/∂u Step1実装

**背景**: Hermite補間使用時、gap関数gは4ノードペア外のDOFにも依存する。
接線ベクトルm_iは隣接ノードの位置に依存するため:
- `∂m_A0/∂x_{A-1} = -1/count_A0`（内部ノード時）
- `∂m_A1/∂x_{A+2} = +1/count_A1`（内部ノード時）

これにより ∂(s,t)/∂u が4ノード以外にも非ゼロとなる。

#### 実装内容

| ファイル | 変更内容 |
|----------|----------|
| `xkep_cae/contact/geometry/_compute.py` | `_compute_dm_ext_coeffs()` 追加 — 隣接ノードの∂m/∂x係数計算 |
| `xkep_cae/contact/geometry/_st_jacobian.py` | `StJacobianInput`: `dm_ext_A`, `dm_ext_B` フィールド追加 |
| 同上 | `StJacobianOutput`: `ds_du_adj`, `dt_du_adj` フィールド追加 |
| 同上 | `_compute_rhs_hermite_neighbor()` メソッド追加 |
| 同上 | `_process_hermite()`: 非局所微分計算を正常パスに追加 |
| `tests/contact/test_st_jacobian.py` | `TestComputeStJacobianHermiteNonlocal` 追加（FD検証2テスト） |

#### 数式

隣接ノードx_adjがm_{side}[m_slot]にのみ影響する場合:

```
∂pA/∂x_{A-1} = H10(s) · (∂m_A0/∂x_{A-1}) = H10(s) · dm_ext
∂(dpA/ds)/∂x_{A-1} = H10'(s) · dm_ext

→ RHS[0] = h_eff · dpA + dh_eff · δ  (A側)
  RHS[1] = -h_eff · dpB

→ [ds/du_{adj}, dt/du_{adj}]^T = -J^{-1} · RHS
```

#### テスト結果

- `test_nonlocal_fd_inner_nodes`: FD一致（atol=1e-5）✓
- `test_nonlocal_endpoint_zero`: 端点ノードはゼロ微分 ✓
- 既存テスト592 passed（回帰なし）

---

## テスト結果

- 新規テスト: 2件（`TestComputeStJacobianHermiteNonlocal`）
- 既存テスト: 592 passed, 20 skipped, 1 xfailed（回帰なし）
- 契約違反: 0件
- lint/format: 全合格

---

## 再現手順

```bash
git checkout claude/check-status-todos-wIZE5
pip install -e .

# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"

# StJacobian非局所FDテスト
python -m pytest tests/contact/test_st_jacobian.py -v -k "Nonlocal"

# 契約検証
python contracts/validate_process_contracts.py

# frozen=False 短縮テスト（比較用、~60秒）
python3 -c "
import warnings; warnings.filterwarnings('ignore')
from xkep_cae.numerical_tests.three_point_bend_jig import *
for frozen in [True, False]:
    cfg = DynamicThreePointBendContactJigConfig(
        E=25.0, n_periods=30.0, jig_push=30.0,
        n_elems_wire=20, max_increments=50,
        use_rigid_surface=True, frozen_hermite_tangent=frozen,
    )
    r = DynamicThreePointBendContactJigProcess().process(cfg)
    sr = r.solver_result
    print(f'[frozen={frozen}] frac={sr.load_history[-1]:.4f} incr={sr.n_increments} cutback={sr.n_cutbacks}')
" 2>&1 | tee /tmp/log-frozen-compare-271.log
```

---

## 次セッションへの引き継ぎ

### 残課題（優先度順）

1. **Hermite非局所∂g/∂u Step2: K_st拡張**
   - ds_du_adj/dt_du_adjをK_stアセンブリに結合
   - 隣接ノードDOFへのK_stエントリ追加
   - ContactPairに隣接ノード情報を格納
   - パイプライン貫通（_manager_process → contact_force/strategy）

2. **Hermite非局所∂g/∂u Step3: K_c拡張**
   - 力係数（Hermite shape function）の隣接ノード依存性
   - ∂f_c/∂x_adj |_{s,t=const} の計算

3. **NR力収束改善**
   - Step2-3完了後にFD検証で接線精度を定量評価
   - 力収束達成率の変化を計測

4. **既存Hermite FDテストのatol厳格化**
   - status-239のTODO: curved/skew/asymmetric テストの atol=1e-2 → 1e-5
   - Step2-3完了後に実施（非局所DOF結合が解消されるため）

### 設計メモ

1. **K_st拡張のアーキテクチャ**: K_st = -(df_ds ⊗ ds_du + df_dt ⊗ dt_du) の ds_du を拡張すると、K_stの列方向に隣接ノードDOFが追加される。行方向（力のDOF）は4ノードのまま。→ (12 × N_ext) 行列。
2. **K_c拡張の必要性**: f_c[k] = p_n · c_k · n の c_k がHermite形状関数に依存し、m経由で隣接ノードに依存。s,t固定でもf_cが隣接ノード位置に依存する。
3. **特異フォールバックでの非局所微分**: 1×1系フォールバック時はJ_invが2×2でないため、隣接ノード微分はNone（安全設計）。実運用ではフォールバック自体が稀。

### 開発運用メモ

- **step-by-step分割の有効性**: 非局所∂g/∂uを3ステップに分割し、各ステップでFD検証。Step1単独でもds_du_adjの正確性が確認でき、次セッションのStep2に直結。
- **FDテストの重要性**: `_finite_diff_st_jacobian_hermite_nonlocal()` は接線ベクトルを再計算する完全なFD。frozen-m近似との差分をatol=1e-5で検出可能。

---

## STA2 準拠チェック

### ベンチマーク条件記録

| 項目 | 値 |
|------|------|
| テスト名 | DynamicThreePointBendContactJigProcess (E=25, n_periods=30, jig_push=30) |
| ブランチ | `claude/check-status-todos-wIZE5` |
| ベースライン | status-270 commit `9866976` (frozen=True frac=1.0, frozen=False 未検証) |

### tee ログファイル

| ログ | パス | 内容 |
|------|------|------|
| 短縮テスト比較 | `/tmp/log-frozen-compare-50incr.log` | frozen=True/False 50incr比較 |
| frozen=Falseフル | `/tmp/log-frozen-false-2000incr.log` | 2000incr完走テスト |

### STA2 チェック項目

- [x] **tee ログ保存**: 全ベンチマーク実行を tee でファイル出力
- [x] **ベースライン先行取得**: frozen=True 50incrベースラインを先に取得
- [x] **再現手順記載**: 全コマンドを status に記載
- [x] **数値の捏造なし**: 力収束0件を正直に報告
- [x] **コミットハッシュ記録**: フルテスト完了後に記録

---
