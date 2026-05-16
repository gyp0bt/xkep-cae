# status-365: 候補 (e) 接触減衰 escape hatch — Phase 1 (Process 単体 + ユニットテスト)

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-23
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12 passed（contact +12: damping tests）

## 概要

status-363/364 引継ぎ 1. の **候補 (e) 接触減衰 escape hatch**
（status-363 §4 計画）の **Phase 1 インフラ**を実装。

status-363 で (c) BT line search パラメータ感度掃引の 4 ケース全却下により
「line search では active 集合振動を根本抑制できない」と確定したため、
次候補 (e) は Type D stall の震源である **active×mixed 領域**に対し
微小な粘性を **escape hatch** として導入する手法。

本 status では **単体 Process + 12 ユニットテスト** の Phase 1 インフラ
のみを完成させ、solver への配線と 7/19 本撚線 validation は
**Phase 2（status-366 予定）** に分離する（下記 §4 参照）。

Phase 分割の正当性:

- 本タスクは status-363 §4 見積りで「7 本 × 5 減衰値 + 19 本再評価 =
  ~30 分計算時間 + NR ループ配線（`_newton_dynamic.py` +
  `_newton_steps.py` で 2700 行触る）」の規模で 1 status に収めると
  骨格だけの実装か validation スキップの妥協実装になる。
- MCDD 脱法 pattern 6（「Phase 分割で困難を先送り」）を回避するため、
  Phase 1 の成功基準を **「動作する Process + 解析解との機械精度一致
  を示すテスト」** と明確化し、skeleton ではなく検証済み実装として
  Phase 2 に引き継ぐ。

## 1. 実装

### 1.1 新規モジュール: `xkep_cae/contact/damping/`

```
xkep_cae/contact/damping/
├── __init__.py                    # 公開 API（Input/Output/Process）
├── strategy.py                    # ContactNormalDampingProcess 本体（219行）
├── docs/
│   └── contact_damping.md         # 設計仕様（README バックリンク付き）
└── tests/
    ├── __init__.py
    └── test_strategy.py           # 12 ユニットテスト（@binds_to 紐付け済み）
```

### 1.2 ContactNormalDampingProcess — 核心実装

**入力** (`ContactNormalDampingInput`、frozen):

| フィールド | 型 | 意味 |
|---|---|---|
| `pairs` | `list` | 接触ペア列（`_ContactPairOutput` 互換、INACTIVE ペアは skip） |
| `velocity` | `np.ndarray (ndof,)` | 全体速度（Generalized-α `vel` 属性をそのまま渡す） |
| `c_n` | `float` | 法線減衰係数（0 なら no-op） |
| `stiffness_factor` | `float` | c1 = γ/(β·dt) — 時間積分スキーム依存の速度-変位感度 |
| `ndof_total` | `int` | 全体 DOF 数 |
| `ndof_per_node` | `int = 6` | ノードあたり DOF 数（撚線梁は 6、変位先頭 3） |

**出力** (`ContactNormalDampingOutput`、frozen):

| フィールド | 型 | 意味 |
|---|---|---|
| `f_damp` | `np.ndarray (ndof,)` | 残差加算向き `R_eff += f_damp`、物理符号 `-c_n v_n n̂` |
| `K_damp` | `sp.csr_matrix (ndof, ndof)` | 接線剛性加算向き `K_eff += K_damp`、常に対称半正定値 |
| `energy_rate` | `float` | 瞬時消散率 Σ c_n v_n² ≥ 0（dt 乗算は呼び出し側） |
| `n_active_pairs` | `int` | 組み立て対象 active ペア数（診断用） |

**数理** (`docs/contact_damping.md` §2):

```
線形形状係数:  coeff = [(1-s), s, -(1-t), -t]        （HuberContactForce と同一）
g_shape (12,) = [coeff_0·n̂, coeff_1·n̂, coeff_2·n̂, coeff_3·n̂]
v_local (12,) = [v(A0), v(A1), v(B0), v(B1)]         （各ノード先頭 3 DOF）
v_n = g_shape · v_local                              （法線相対速度、正=closing）

f_damp_local = -c_n * v_n * g_shape                  (12,)
K_damp_local = c_n * c1 * (g_shape ⊗ g_shape)        (12, 12)、c1 = γ/(β·dt)
E_damp_rate  = Σ_active c_n * v_n²                   ≥ 0（散逸性）
```

c_n ≥ 0 で K_damp_local は rank-1 な対称半正定値で NR 収束に対し安定化側。

### 1.3 StrandBendingOscillationConfig 拡張

`xkep_cae/numerical_tests/strand_bending_oscillation.py` に 2 フィールド追加
（**Phase 1 時点では保有のみ、Phase 2 で solver に連結**）:

```python
contact_damping_coefficient: float = 0.0              # c_n >= 0、0=無効（default）
contact_damping_energy_budget_ratio: float = 0.0      # E_damp/E_strain 許容上限
```

推奨値（Phase 2 validation で決定予定）:

- `contact_damping_coefficient`: 物理単位 [N·s/mm]、典型値は k_pen·dt の
  1-10% を目安（例: k_pen=1e6、dt=1e-4 なら c_n=1e-1 〜 1e0）
- `contact_damping_energy_budget_ratio`: 0.05〜0.20（5〜20% 散逸許容）

### 1.4 ユニットテスト（12 件、全合格）

`xkep_cae/contact/damping/tests/test_strategy.py`:

| クラス | テスト | 検証内容 |
|---|---|---|
| `TestContactNormalDampingProcessAPI` | 3 | c_n=0 / 空 pairs / INACTIVE ペア → 全ゼロ出力 |
| `TestContactNormalDampingProcessPhysics` | 6 | 単一ペア解析解（closing/separating） + K_damp 対称性 + 多ペア重畳 + E_rate≥0 + 接線方向速度で no-op |
| `TestContactNormalDampingProcessTangent` | 1 | v = c1·u 仮定下で K_damp と ∂f_damp/∂u の有限差分一致（rel/abs 1e-5） |
| `TestContactNormalDampingOutputTypes` | 1 | 出力型（np.ndarray / sparse / float / int）の厳密チェック |
| + `@binds_to(ContactNormalDampingProcess)` | 1 | C3 契約検査紐付け（meta.name 検証） |

解析解検証の核となる `test_single_pair_closing_velocity_analytic`:

```
s=t=0.5、n̂=(1,0,0)、B 側ノード 2,3 に v=(-2,0,0) を与えると
v_n = +2 （closing）、c_n=3 で:
  f_damp[A0.x] = f_damp[A1.x] = -0.5·3·2 = -3  （A 側は -x 方向）
  f_damp[B0.x] = f_damp[B1.x] = +0.5·3·2 = +3  （B 側は +x 方向）
  E_rate = 3·2² = 12
```

全成分で `pytest.approx` 等値を確認。

## 2. Gate

- `uv run python -m pytest xkep_cae/contact/damping/tests/ -q` → **12 passed in 0.90s**
- `uv run python -m pytest xkep_cae/contact/ -q` → **439 passed, 5 skipped in 50.68s**（回帰なし）
- `uv run python -m pytest tests/ -q` → **314 passed, 11 skipped in 212.79s**（回帰なし）
- `uv run python contracts/validate_process_contracts.py` → **契約違反 0 件 / 条例違反 0 件（全 24 検査 OK、C3 含む）**
- `uv run ruff check xkep_cae/ tests/ contracts/` → All checks passed
- `uv run ruff format --check xkep_cae/ tests/ contracts/` → 196 files already formatted

## 3. 設計判断

### 3.1 なぜ Generalized-α の C 行列を直接拡張しない？

既存 `GeneralizedAlphaProcess` (`xkep_cae/time_integration/strategy.py`) は
`self.C` を **コンストラクタ固定** で受け取り、`effective_stiffness` /
`effective_residual` で `(1-α_f)·c1·C` として組み込む構造。

接触減衰は **ペア依存（active 集合が NR 反復で変動）** のため、C に組み込むと
NR 反復ごとに C を再構築して `effective_*` を呼び直す必要がある。時間積分
モジュールをペアマネージャに依存させるのは責務分離違反。

本 Process は **接触力側（ペア依存処理層）** で f_damp + K_damp を組み立て、
NR の `R_eff += f_damp`、`K_eff += K_damp` として加算する設計。時間積分
モジュールは無変更、c1 だけ呼び出し側が計算して渡す。

### 3.2 なぜ @verified_by を付けない？

`TermExpansionContract("K_c_term_expansion")` の 5 項分解（material/geo/st/
closest/hermite_adj）は K_c = -∂f_c/∂u の解析的接線拡張。接触減衰項は
**Generalized-α の C 行列経路の代替**であり、K_c の項ではない。

Phase 2 で solver 配線時に、`K_damp` の FD 検証用 VerifyProcess
（`ContactNormalDampingFDDiagnosticProcess` 仮称）を新設するかは要検討
（K_damp は rank-1 outer product で FD 検証の必要性は低い — 本 status の
`test_tangent_matches_fd_under_v_is_c1_u` で既に機械精度の整合性を確認）。

### 3.3 なぜ Hermite 形状関数を使わない？

Phase 1 は線形形状 `(1-s), s, -(1-t), -t` で十分。Hermite 高次基底は
K_damp の `(g_shape ⊗ g_shape)` 構造に依らず、形状係数の選択だけで
拡張可能（Phase 2 以降）。

Phase 2 で `HuberContactForceProcess` の Hermite 経路と一致させる場合、
`use_hermite` フラグと `dm_A/dm_B` を入力に追加し、`_hermite_corrected_coeffs`
ヘルパ（`contact/contact_force/strategy.py`）を共有化する。

## 4. Phase 2 配線計画（status-366 へ引継ぎ）

### 4.1 ContactFrictionProcess への StrategySlot 追加

`xkep_cae/contact/solver/process.py`:

```python
damping_slot = StrategySlot(
    object,
    required=False,
    default_types=(),   # default OFF（c_n=0）
)
```

`default_strategies()` (`xkep_cae/core/data.py`) に `damping_slot` 用の
`None` / `ContactNormalDampingProcess()` インスタンスを追加（cfg 依存）。

### 4.2 NR ループへの加算

`xkep_cae/contact/solver/_newton_dynamic.py` の NR 反復内で:

```python
if cfg.contact_damping_coefficient > 0.0:
    damping_inp = ContactNormalDampingInput(
        pairs=manager.pairs,
        velocity=time_integ.vel,
        c_n=cfg.contact_damping_coefficient,
        stiffness_factor=time_integ.gamma / (time_integ.beta * dt),
        ndof_total=len(u),
    )
    damping_out = damping_proc.process(damping_inp)
    R_eff += damping_out.f_damp       # 残差加算
    K_eff += damping_out.K_damp       # 接線剛性加算
    E_damp_step += damping_out.energy_rate * dt  # energy monitor 用
```

配線点は `_newton_dynamic.py` の `tangent_components()` 呼び出しと
並行（contact_force の f_c / K_c 組み立て直後）。4 層 plumb-through:
`cfg → ContactFrictionProcess → NewtonDynamicProcess → NewtonStepProcess`。

### 4.3 ContactDampingEnergyMonitorProcess 新設

`xkep_cae/contact/damping/monitor.py`（Phase 2 新規）:

```python
class ContactDampingEnergyMonitorProcess(PostProcess):
    """10 step 毎に E_damp_total / E_strain を出力、budget 超過で警告."""
```

入力: E_damp 累積、E_strain 現在値、step index、budget_ratio
出力: 警告/エラー（budget 超過時）、診断ログ

### 4.4 Validation スクリプト

`work/beam_hysteresis/23_contact_damping_7strand_sweep.py`（Phase 2 新規）:

- 7本撚線 90° 曲げで c_n ∈ {0, 0.01, 0.02, 0.05, 0.10, 0.20} × k_pen·dt を実測
- Papailiou 解析解（`PapailiouSolution` Process）と比較、散逸エネルギー
  比（E_damp/W_load）を算出
- budget 許容線（E_damp/E_strain < 0.05〜0.20）を特定

`work/beam_hysteresis/24_contact_damping_19strand.py`（Phase 2 新規）:

- 7本で特定した最小 c_n で 19本撚線 Type D stall 解消を検証
- **MCDD 凍結解除条件**: frac=1.0 完走 + E_damp/E_strain < budget

### 4.5 MCDD Phase E C25 候補

`@verified_by` の challenge-test fixture 紐付け義務化（status-364 §5 提示の
偽陰性パターン対策）は Phase 2 の solver 配線後に検討。Phase 1 時点では
damping 自体が K_c の TermExpansionContract に属さないため該当しない。

## 5. ファイル変更

| ファイル | 変更 |
|---------|------|
| `xkep_cae/contact/damping/__init__.py` | **新規**: モジュール公開 API + Phase 1/2 説明 |
| `xkep_cae/contact/damping/strategy.py` | **新規**: `ContactNormalDampingProcess` + Input/Output（219 行） |
| `xkep_cae/contact/damping/docs/contact_damping.md` | **新規**: 設計仕様 + Phase 2 計画（README バックリンク付き） |
| `xkep_cae/contact/damping/tests/__init__.py` | **新規** |
| `xkep_cae/contact/damping/tests/test_strategy.py` | **新規**: 12 ユニットテスト + `@binds_to` |
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `contact_damping_coefficient` + `contact_damping_energy_budget_ratio` フィールド追加（保有のみ） |
| `docs/status/status-365.md` | **新規**: 本ファイル |
| `docs/status/status-index.md` | status-365 行追加 |
| `README.md` | 現在状況に候補 (e) Phase 1 追記 |
| `docs/roadmap.md` | 候補 (e) Phase 1 完了行追記 |

## 6. 引継ぎ（status-366 へ）

1. **最優先**: Phase 2 配線（§4.1〜4.3）
   - `ContactFrictionProcess.damping_slot` 追加
   - `_newton_dynamic.py` での NR 加算
   - `ContactDampingEnergyMonitorProcess` 新設
2. **validation**: 7本撚線で c_n budget 許容線確定（§4.4）
3. **MCDD 凍結解除判定**: 19本撚線で frac=1.0 完走 + E_damp/E_strain < budget
4. **副次（(e) 不十分時）**: 候補 (d) 接触凍結モード 19 本適用（status-284
   の 7 本 frac 0.40→0.70 手法の 19 本再評価）
5. **最終手段**: 候補 (f) Phase C-3' s-tracking 経路の 19 本再評価
6. **Phase E C25 候補**: VerifyProcess challenge-test fixture 義務化
   （status-364 §5 提示の偽陰性パターン対策、Phase 2 配線後に検討）
