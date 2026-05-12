"""work/beam_hysteresis/41_epsilon1_3strand_helical_no_contact.py — ε-1: 3 strand helical + 接触なし + explicit-TL 固定.

[← README](README.md) | [← project README](../../README.md)

status-397 ε-1 検証スクリプト（status-395 §6.3 / status-396 §7 で確定）.

**目的**:
status-396 で API 化された `explicit_ul_disable_update=True`（explicit-TL 固定モード、
UL `update_reference()` を一切呼ばない）を **3 strand helical + 接触なし** の実機系で
初検証する。直線 chain γ-3（status-395）で機械精度実証済の explicit-TL を、新たな 3 要素
（初期 curvature 上の CR / 多 strand global assembler / 端部 BC + free_end_mode）に
拡張して foundation 健全性を確認する。

**新たに検証される要素**（CLAUDE.md 「次セッション最優先（status-397）」より）:

1. **初期 curvature 上の CR**: 直線 reference (γ-3) vs 曲線 reference（helical strand）で
   `R_0`（局所軸）の構築 + initial gap 等が変わる。
2. **多 strand 並列 (no contact)**: 単独 strand と複数 strand の global assembler 振る舞い。
3. **端部 BC**: MPC + free_end_mode が explicit + TL モードで成立するか。

**比較対象**:
implicit baseline（同 BC + 同 mesh、`solver_mode="implicit"` + `explicit_ul_disable_update=False`）.

**3 指標 AND gate**（status-388 透明性ルール、CLAUDE.md「妥当性テストの透明性ルール」遵守）:

1. **u_x_tip 相対誤差 < 10%** (kinematics 1: 軸方向短縮)
2. **u_z_tip 相対誤差 < 10%** (kinematics 2: 横方向たわみ)
3. **E_strain 相対誤差 < 10%** (kinematics 独立: 歪エネルギー積分量)

加えて物理的妥当性 gate (status-380 gate (3)):

- `max |u_trans| < L_strand × 10` (撚線長 100mm に対し最大変位 1m 以内)

**判定（CLAUDE.md ロードマップより）**:

- **ε-1 PASS** → ε-2（接触あり 3 strand、status-398）へ進行
- **ε-1 FAIL** → Phase δ（接触あり 2 strand、`48_delta_2strand_contact.py`）に retreat
  して原因局在化（ヘリカル初期 κ / 多 strand global assembler / BC 経路）

**実行**:

    uv run --extra dev python work/beam_hysteresis/41_epsilon1_3strand_helical_no_contact.py \\
        2>&1 | tee /tmp/epsilon1_$(date +%s).log

**設計参照**:

- `docs/status/status-396.md` §7 引き継ぎ（z3 API 化完結）
- `docs/status/status-395.md` §6.3 ε-1 設計
- `docs/status/verification_matrix.md` §3 上位層改修対象
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
    _collect_end_nodes,
)


@dataclass
class RunSummary:
    """1 つの solver_mode 実行のサマリ."""

    label: str
    frac: float
    converged: bool
    n_increments: int
    n_cutbacks: int
    elapsed: float
    u: np.ndarray
    max_u_trans: float
    tip_disp: np.ndarray  # (u_x, u_y, u_z) 右端撚線ノード重心の並進変位
    e_strain: float
    e_kin: float
    e_ratio: float
    n_strand_nodes: int


def _extract_tip_disp(
    u: np.ndarray,
    mesh: object,
    *,
    ndof_per_node: int = 6,
) -> np.ndarray:
    """右端撚線ノード群の重心の並進変位 (u_x, u_y, u_z) を返す."""
    _, right_nodes = _collect_end_nodes(
        mesh.connectivity, int(mesh.n_strands), np.asarray(mesh.strand_ids)
    )
    if not right_nodes:
        return np.zeros(3)
    u_mat = u.reshape(-1, ndof_per_node)
    tip = np.zeros(3)
    for n in right_nodes:
        tip += u_mat[n, :3]
    tip /= len(right_nodes)
    return tip


def _make_base_cfg() -> dict:
    """両 solver_mode で共通の物理 / メッシュ / BC パラメータ.

    3 strand helical + 接触なし + free_end_mode + 中規模曲げ（κ·L=0.5 rad ≈ 28.6°）.
    γ-3 (θ=0.15 rad) より大きいが extreme 領域は避ける。
    """
    return dict(
        n_strands=3,  # ε-1 本命: 3 strand
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.001,  # 0.1 rad ≈ 5.7° 小曲げ（foundation 検証、γ-3 と同レンジ）
        n_cycles=1,
        n_increments_per_cycle=20,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=True,
        contact_enabled=False,  # ε-1 本命: 接触なし
        penalty_exponent=1.5,
    )


def _run_one(label: str, **overrides) -> RunSummary:
    base = _make_base_cfg()
    base.update(overrides)
    cfg = StrandBendingOscillationConfig(**base)

    print()
    print("─" * 72)
    print(
        f"[run] label={label}, solver_mode={base.get('solver_mode', 'implicit')}, "
        f"explicit_ul_disable_update={base.get('explicit_ul_disable_update', False)}"
    )
    print("─" * 72)

    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result

    frac = float(sr.load_history[-1]) if sr.load_history else 0.0
    e_hist = sr.energy_history
    e_kin = float(e_hist.entries[-1].kinetic_energy) if e_hist and e_hist.entries else 0.0
    e_str = float(e_hist.entries[-1].strain_energy) if e_hist and e_hist.entries else 0.0
    e_ratio = e_kin / max(abs(e_str), 1e-30)

    u = sr.u
    n_total_nodes = u.shape[0] // 6
    u_trans = u.reshape(n_total_nodes, 6)[:, :3]
    max_u_trans = float(np.max(np.linalg.norm(u_trans, axis=1)))

    tip = _extract_tip_disp(u, result.mesh)

    print(
        f"  frac={frac:.4f}, converged={sr.converged}, "
        f"incr={sr.n_increments}, cb={sr.n_cutbacks}, elapsed={elapsed:.2f}s"
    )
    print(f"  max |u_trans| = {max_u_trans:.4e} [mm]")
    print(f"  tip disp     = ({tip[0]:.4e}, {tip[1]:.4e}, {tip[2]:.4e}) [mm]")
    print(f"  E_strain     = {e_str:.4e} [N·mm]")
    print(f"  E_kin        = {e_kin:.4e} [N·mm]")
    print(f"  E_kin/|E_str|= {e_ratio:.3e}")

    return RunSummary(
        label=label,
        frac=frac,
        converged=bool(sr.converged),
        n_increments=int(sr.n_increments),
        n_cutbacks=int(sr.n_cutbacks),
        elapsed=elapsed,
        u=u.copy(),
        max_u_trans=max_u_trans,
        tip_disp=tip,
        e_strain=e_str,
        e_kin=e_kin,
        e_ratio=e_ratio,
        n_strand_nodes=int(result.n_strand_nodes),
    )


def _rel_err(a: float, b: float, *, eps: float = 1e-30) -> float:
    """|a-b| / max(|b|, eps)."""
    return abs(a - b) / max(abs(b), eps)


def main() -> int:
    print("=" * 72)
    print("status-397 ε-1: 3 strand helical + 接触なし + explicit-TL 固定")
    print("=" * 72)
    print("Reference: implicit (solver_mode='implicit', explicit_ul_disable_update=False)")
    print("Target   : explicit-TL (solver_mode='explicit', explicit_ul_disable_update=True)")
    print()
    print("3 指標 AND gate（10%）:")
    print("  (1) u_x_tip 相対誤差 vs implicit")
    print("  (2) u_z_tip 相対誤差 vs implicit")
    print("  (3) E_strain 相対誤差 vs implicit  [kinematics 独立]")
    print("物理的妥当性 gate: max |u_trans| < L_strand × 10 = 1000 mm")
    print()
    print("Config: bending_curvature=0.001 (κ·L=0.1 rad ≈ 5.7°), n_pitches=1.0,")
    print("        n_strands=3, contact_enabled=False, free_end_mode=True,")
    print("        n_increments_per_cycle=20")

    # implicit baseline
    imp = _run_one(
        "implicit",
        solver_mode="implicit",
    )

    # explicit-TL（status-396 で API 化された disable=True 適用）
    # mass scaling auto-tune で Courant 制約緩和（β cap 1e5）
    explicit_overrides = dict(
        solver_mode="explicit",
        explicit_ul_disable_update=True,  # ★ status-397 の本命 flag
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e5,  # 大 β cap で Courant 制約を緩和
        explicit_kinetic_energy_budget_ratio=0.05,
    )
    exp = _run_one("explicit_tl_disable", **explicit_overrides)

    # ── 3 指標 AND gate ──
    print()
    print("=" * 72)
    print("3 指標 AND gate（implicit を reference として）")
    print("=" * 72)
    err_ux = _rel_err(exp.tip_disp[0], imp.tip_disp[0])
    err_uz = _rel_err(exp.tip_disp[2], imp.tip_disp[2])
    err_es = _rel_err(exp.e_strain, imp.e_strain)

    print(
        f"  (1) u_x_tip:  imp={imp.tip_disp[0]:+.4e}, exp={exp.tip_disp[0]:+.4e}, "
        f"rel_err={err_ux:.3e}"
    )
    print(
        f"  (2) u_z_tip:  imp={imp.tip_disp[2]:+.4e}, exp={exp.tip_disp[2]:+.4e}, "
        f"rel_err={err_uz:.3e}"
    )
    print(f"  (3) E_strain: imp={imp.e_strain:+.4e}, exp={exp.e_strain:+.4e}, rel_err={err_es:.3e}")

    L_strand = 100.0
    g_ux = err_ux < 0.10
    g_uz = err_uz < 0.10
    g_es = err_es < 0.10
    g_phys_imp = imp.max_u_trans < L_strand * 10
    g_phys_exp = exp.max_u_trans < L_strand * 10
    g_frac_imp = imp.frac >= 0.999
    g_frac_exp = exp.frac >= 0.999

    print()
    print("─" * 72)
    print("Gate 判定")
    print("─" * 72)
    print(
        f"  imp frac=1.0 完走             : {'PASS' if g_frac_imp else 'FAIL'} (frac={imp.frac:.4f})"
    )
    print(
        f"  exp frac=1.0 完走             : {'PASS' if g_frac_exp else 'FAIL'} (frac={exp.frac:.4f})"
    )
    print(
        f"  imp 物理的妥当性 (max|u|<1m)  : {'PASS' if g_phys_imp else 'FAIL'} "
        f"(max|u|={imp.max_u_trans:.3e})"
    )
    print(
        f"  exp 物理的妥当性 (max|u|<1m)  : {'PASS' if g_phys_exp else 'FAIL'} "
        f"(max|u|={exp.max_u_trans:.3e})"
    )
    print(f"  (1) u_x_tip 誤差 < 10%         : {'PASS' if g_ux else 'FAIL'} ({err_ux:.2%})")
    print(f"  (2) u_z_tip 誤差 < 10%         : {'PASS' if g_uz else 'FAIL'} ({err_uz:.2%})")
    print(f"  (3) E_strain 誤差 < 10%        : {'PASS' if g_es else 'FAIL'} ({err_es:.2%})")
    print()
    gate_3metric = g_ux and g_uz and g_es
    overall = g_frac_imp and g_frac_exp and g_phys_imp and g_phys_exp and gate_3metric
    print(f"  3 指標 AND gate              : {'PASS' if gate_3metric else 'FAIL'}")
    print(f"  Total (foundation gate AND 3指標): {'PASS' if overall else 'FAIL'}")
    print("=" * 72)
    print()
    print("判定:")
    if overall:
        print("  ε-1 PASS → ε-2（接触あり 3 strand、status-398）へ進行可")
    else:
        print("  ε-1 FAIL → Phase δ（接触あり 2 strand）に retreat して局在化")
        print("  要因候補: 初期 curvature 上の CR / 多 strand assembler / BC 経路")
    print()

    # 参考: 処方変位 vs 観測（解析的 cantilever 曲げ近似）
    base_cfg = _make_base_cfg()
    strand_length = base_cfg["pitch_length"] * base_cfg["n_pitches"]
    bending_angle = base_cfg["bending_curvature"] * strand_length
    R_curv = strand_length / bending_angle if bending_angle > 1e-12 else float("inf")
    delta_lateral = R_curv * (1.0 - math.cos(bending_angle))
    delta_axial = R_curv * math.sin(bending_angle) - strand_length
    print(
        f"参考: bending_angle (κ·L) = {bending_angle:.4f} rad = {math.degrees(bending_angle):.1f}°"
    )
    print(f"     R = L/θ = {R_curv:.4e} [mm]")
    print(f"     解析 cantilever 先端横変位 δ_x ≈ R(1-cos θ) = {delta_lateral:.4e} [mm]")
    print(f"     解析 cantilever 軸短縮 δ_z ≈ R sin θ - L  = {delta_axial:.4e} [mm]")

    # 注意: E_strain は UL (implicit) vs TL (explicit) で意味が異なる:
    #   - UL with update: 最終 increment 内の incremental SE のみ
    #   - TL (disable_update=True): 初期 reference から累積した total SE
    # そのため E_strain 直接比較は formulation-dependent。3 指標 AND gate では
    # 「kinematics 独立」指標として使う場合は注意（status-397 §X 観察）。
    # 主判定は u_x_tip / u_z_tip（kinematic、formulation-invariant）。

    # ── 原因局在化 sub-experiment（ε-1 FAIL の 3 候補識別）──
    if not overall:
        print()
        print("=" * 72)
        print("原因局在化 sub-experiment（ε-1 FAIL 時のみ）")
        print("=" * 72)
        print("3 候補（CLAUDE.md「次セッション最優先」より）:")
        print("  (a) ヘリカル初期 κ — 曲線 reference 上の CR 構築")
        print("  (b) 多 strand global assembler — 単 strand と多 strand の差分")
        print("  (c) 端部 BC + free_end_mode — explicit + TL モードでの BC 経路")
        print()
        print("Sub-experiment: n_strands=1 (straight, not helical) で explicit-TL を試行.")
        print("  n_strands=1 PASS → (a) ヘリカル初期 κ が主因")
        print("  n_strands=1 FAIL → (b)/(c) （多 strand assembler / BC 経路）")
        print()
        try:
            imp_n1 = _run_one("implicit_n1", n_strands=1, solver_mode="implicit")
            exp_n1 = _run_one("explicit_tl_disable_n1", n_strands=1, **explicit_overrides)
            err_ux_n1 = _rel_err(exp_n1.tip_disp[0], imp_n1.tip_disp[0])
            err_uz_n1 = _rel_err(exp_n1.tip_disp[2], imp_n1.tip_disp[2])
            print()
            print("─" * 72)
            print("n_strands=1 (straight) sub-experiment 結果")
            print("─" * 72)
            print(
                f"  imp_n1 tip = ({imp_n1.tip_disp[0]:+.4e}, "
                f"{imp_n1.tip_disp[1]:+.4e}, {imp_n1.tip_disp[2]:+.4e})"
            )
            print(
                f"  exp_n1 tip = ({exp_n1.tip_disp[0]:+.4e}, "
                f"{exp_n1.tip_disp[1]:+.4e}, {exp_n1.tip_disp[2]:+.4e})"
            )
            print(f"  err_ux_n1 = {err_ux_n1:.2%} / err_uz_n1 = {err_uz_n1:.2%}")
            n1_pass = err_ux_n1 < 0.10 and err_uz_n1 < 0.10
            print(f"  n_strands=1 explicit-TL gate: {'PASS' if n1_pass else 'FAIL'}")
            if n1_pass:
                print("  → 主因候補 (a) ヘリカル初期 κ が支配的")
            else:
                print("  → 主因候補 (b) 多 strand assembler / (c) BC 経路")
                print("     （n_strands=1 でも FAIL のため helical 以前の問題）")
        except Exception as ex:
            print(f"  sub-experiment 失敗: {ex!r}")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
