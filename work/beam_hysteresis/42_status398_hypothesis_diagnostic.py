"""work/beam_hysteresis/42_status398_hypothesis_diagnostic.py — status-398: `_process_free_end` × explicit-TL 3 仮説切り分け診断.

[← README](README.md) | [← project README](../../README.md)

status-398 診断スクリプト。status-397 ε-1 で `_process_free_end` 駆動経路 + explicit-TL
が under-deformation を起こすことを 1 strand 規模で再現確定したが、CLAUDE.md「次セッション
最優先（status-398）」で 3 仮説の切り分けを要求された。

**3 仮説**（CLAUDE.md より）:

1. **prescribed BC の TL 増分処理**: process.py L653 `state.u[prescribed_dofs] =
   load_frac * prescribed_values` の **stepwise 適用**が問題（毎 increment で θ_y が
   jumps）。
2. **explicit driver の reaction force 累積**: ExplicitDynamicProcess は prescribed
   DOF を `fixed_dofs` に含めて acceleration=0 で扱う。reaction force は陽に
   accumulate しない。
3. **`_ExtendedULAssemblerWrapper` の TL モード対応**: free_end_mode では使われない
   が、隠れた update_reference 経路が他にある可能性。

**診断アプローチ**:

3 つの diagnostic case を ε-1 と同 BC (n_strands=1, contact_enabled=False,
free_end_mode=True, bending_curvature=0.001, explicit_ul_disable_update=True) で実行:

- **Case A**: `n_increments_per_cycle=200`（10× 細分化）。stepwise jump 幅を 1/10 に。
- **Case B**: `explicit_mass_scaling_max_beta=100`（β cap を 1/1000）。dt_sub を縮小し
  cutback で sub-step を増やす。
- **Case C**: `t_cycle_min_seconds=100.0`（100× 物理時間）。mass-scaled dynamics に
  十分な settling time を与える。

**判定**:

- いずれかが implicit baseline (u_x=4.996 mm) に近づけば、その軸が支配要因。
- いずれも改善しなければ、別の要因（hypothesis 2, 3 等）が支配。
- 複数で改善すれば、組合せ要因。

**実行**:

    uv run --extra dev python work/beam_hysteresis/42_status398_hypothesis_diagnostic.py \\
        2>&1 | tee /tmp/status398_diag_$(date +%s).log

**設計参照**:

- `docs/status/status-397.md` §5 仮説リスト
- `xkep_cae/contact/solver/process.py` L653-655 prescribed BC 適用
- `xkep_cae/contact/solver/_explicit_dynamic.py` L380-390 mass scaling auto-tune
"""

from __future__ import annotations

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
    label: str
    frac: float
    n_increments: int
    n_cutbacks: int
    elapsed: float
    max_u_trans: float
    tip_disp: np.ndarray
    e_strain: float
    e_kin: float


def _extract_tip(u: np.ndarray, mesh: object) -> np.ndarray:
    _, right = _collect_end_nodes(
        mesh.connectivity, int(mesh.n_strands), np.asarray(mesh.strand_ids)
    )
    if not right:
        return np.zeros(3)
    um = u.reshape(-1, 6)
    return np.mean(um[right, :3], axis=0)


def _base_cfg() -> dict:
    """ε-1 sub-experiment (n_strands=1 straight) の基本 config."""
    return dict(
        n_strands=1,  # 直線 strand（helical でない）に固定して driver 層単独評価
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.001,  # 0.1 rad ≈ 5.7°
        n_cycles=1,
        n_increments_per_cycle=20,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=True,
        contact_enabled=False,
        penalty_exponent=1.5,
    )


def _explicit_overrides() -> dict:
    """共通 explicit-TL overrides."""
    return dict(
        solver_mode="explicit",
        explicit_ul_disable_update=True,
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e5,
        explicit_kinetic_energy_budget_ratio=0.05,
    )


def _run(label: str, **overrides) -> RunSummary:
    base = _base_cfg()
    base.update(overrides)
    cfg = StrandBendingOscillationConfig(**base)

    print()
    print("─" * 72)
    print(f"[run] {label}")
    print(
        f"  n_inc={base['n_increments_per_cycle']}, "
        f"t_cycle_min={base.get('t_cycle_min_seconds', 1.0)}, "
        f"β_max={base.get('explicit_mass_scaling_max_beta', 'N/A')}"
    )
    print("─" * 72)

    t0 = time.perf_counter()
    try:
        result = StrandBendingOscillationProcess().process(cfg)
        sr = result.solver_result
        frac = float(sr.load_history[-1]) if sr.load_history else 0.0
        e_hist = sr.energy_history
        e_kin = float(e_hist.entries[-1].kinetic_energy) if e_hist and e_hist.entries else 0.0
        e_str = float(e_hist.entries[-1].strain_energy) if e_hist and e_hist.entries else 0.0
        u = sr.u
        u_trans = u.reshape(-1, 6)[:, :3]
        max_ut = float(np.max(np.linalg.norm(u_trans, axis=1)))
        tip = _extract_tip(u, result.mesh)
        n_incr = int(sr.n_increments)
        n_cb = int(sr.n_cutbacks)
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        return RunSummary(
            label=label,
            frac=0.0,
            n_increments=0,
            n_cutbacks=0,
            elapsed=time.perf_counter() - t0,
            max_u_trans=float("nan"),
            tip_disp=np.full(3, float("nan")),
            e_strain=float("nan"),
            e_kin=float("nan"),
        )

    elapsed = time.perf_counter() - t0
    print(f"  frac={frac:.4f}, incr={n_incr}, cb={n_cb}, elapsed={elapsed:.2f}s")
    print(f"  max|u| = {max_ut:.4e} mm, tip = ({tip[0]:+.4e}, {tip[1]:+.4e}, {tip[2]:+.4e})")
    print(f"  E_strain={e_str:.4e}, E_kin={e_kin:.4e}")

    return RunSummary(
        label=label,
        frac=frac,
        n_increments=n_incr,
        n_cutbacks=n_cb,
        elapsed=elapsed,
        max_u_trans=max_ut,
        tip_disp=tip,
        e_strain=e_str,
        e_kin=e_kin,
    )


def main() -> int:
    print("=" * 72)
    print("status-398: `_process_free_end` × explicit-TL 3 仮説切り分け診断")
    print("=" * 72)
    print("基準: n_strands=1 straight, free_end_mode=True, contact_enabled=False,")
    print("      bending_curvature=0.001 (0.1 rad), explicit_ul_disable_update=True")
    print()
    print("ε-1 sub-experiment (status-397) baseline:")
    print("  implicit: u_x=4.996e+00 mm（解析 cantilever 解と機械精度級一致）")
    print("  explicit-TL (default): u_x=1.855e-01 mm（96.29% under-deformation）")
    print()

    # implicit reference
    imp = _run("implicit_baseline", solver_mode="implicit")

    # baseline explicit-TL（ε-1 sub と同設定）
    exp_base = _run("explicit_TL_baseline", **_explicit_overrides())

    # Case A1, A2, A3: 細分化 n_increments_per_cycle 掃引（asymptotic 確認）
    exp_a1 = _run("A1_n_inc=200", n_increments_per_cycle=200, **_explicit_overrides())
    exp_a2 = _run("A2_n_inc=2000", n_increments_per_cycle=2000, **_explicit_overrides())

    # Case B: β cap を 1/1000 = 100
    exp_b_overrides = _explicit_overrides()
    exp_b_overrides["explicit_mass_scaling_max_beta"] = 100.0
    exp_b = _run("B_beta_max_100", **exp_b_overrides)

    # Case C: t_cycle_min_seconds=100.0（100× 物理時間）
    exp_c = _run(
        "C_t_cycle_100s",
        t_cycle_min_seconds=100.0,
        **_explicit_overrides(),
    )

    # Case D: Hybrid（n_inc=2000 + β cap=10）— 限界実験で asymptotic 一致を確認
    exp_d_overrides = _explicit_overrides()
    exp_d_overrides["explicit_mass_scaling_max_beta"] = 10.0
    exp_d = _run("D_n_inc=2000_β_max=10", n_increments_per_cycle=2000, **exp_d_overrides)

    # 結果サマリ
    print()
    print("=" * 72)
    print("結果サマリ（u_x_tip vs implicit reference）")
    print("=" * 72)
    cases = [imp, exp_base, exp_a1, exp_a2, exp_b, exp_c, exp_d]
    print(
        f"  {'label':<26} {'frac':>6} {'incr':>5} {'cb':>4} {'u_x [mm]':>12} {'rel_err vs imp':>16}"
    )
    for c in cases:
        rel = abs(c.tip_disp[0] - imp.tip_disp[0]) / max(abs(imp.tip_disp[0]), 1e-30)
        print(
            f"  {c.label:<26} {c.frac:>6.4f} {c.n_increments:>5d} "
            f"{c.n_cutbacks:>4d} {c.tip_disp[0]:>+12.4e} {rel:>15.2%}"
        )

    print()
    print("仮説判定:")
    err_base = abs(exp_base.tip_disp[0] - imp.tip_disp[0]) / max(abs(imp.tip_disp[0]), 1e-30)
    print(f"  baseline rel_err: {err_base:.2%}")
    for c in [exp_a1, exp_a2, exp_b, exp_c, exp_d]:
        rel = abs(c.tip_disp[0] - imp.tip_disp[0]) / max(abs(imp.tip_disp[0]), 1e-30)
        improved = rel < err_base * 0.5
        verdict = "PASS（< 0.5× baseline 誤差）" if improved else "FAIL"
        print(f"  {c.label}: rel_err={rel:.2%} → {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
