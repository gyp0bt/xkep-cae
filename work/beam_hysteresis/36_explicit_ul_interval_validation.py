"""work/beam_hysteresis/36_explicit_ul_interval_validation.py — q1 効果検証.

[← README](README.md) | [← project README](../../README.md)

status-383 候補 (q1)「explicit 中の UL update_reference 周期化」の効果を、
単梁 90° カンチレバー曲げで実測する。`explicit_ul_update_interval` を 1（既存
挙動）、5、10、20 と掃引し、(p3) damping + (p1) relax との組合せで MCDD
凍結解除条件 (5)「精度 gate < 10%」達成可否を判定する。

status-382 §3 の「真の根本原因 — UL update_reference 凍結」: 各増分で
update_reference を呼ぶと u_incr ≈ 0 となり relax phase の f_int(u_incr) が
ゼロ近傍で構造を平衡駆動できない。N 増分ごとに更新することで u_incr が
累積し f_int が非ゼロとなる。

Gate (MCDD 凍結解除条件 (5)、status-381 §5):
  - `|max|u_explicit| − max|u_analytical|| / max|u_analytical| < 0.10`
  - 単梁 90° カンチレバー曲げ: 解析解 max|u| = 73.30 mm
    （quarter circle、L=100mm、R = 2L/π ≈ 63.66 mm、|u| = √((L−R)² + R²)）

実行:
    uv run --extra dev python work/beam_hysteresis/36_explicit_ul_interval_validation.py \\
        2>&1 | tee /tmp/q1_validation_$(date +%s).log
"""

from __future__ import annotations

import time

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def _make_base_cfg() -> dict:
    """単梁 90° 曲げ baseline（接触なし、L=100 mm、E=130 GPa、撚線径=1mm）."""
    return dict(
        n_strands=1,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.015,
        n_cycles=1,
        n_increments_per_cycle=20,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=20000,
        exclude_same_strand=True,
        free_end_mode=True,
        contact_enabled=False,
    )


def _run_one(label: str, **overrides) -> dict:
    base = _make_base_cfg()
    base.update(overrides)
    cfg = StrandBendingOscillationConfig(**base)

    print()
    print("─" * 72)
    print(f"[run] {label}")
    print("─" * 72)

    t0 = time.perf_counter()
    try:
        result = StrandBendingOscillationProcess().process(cfg)
        elapsed = time.perf_counter() - t0
        sr = result.solver_result
        frac = float(sr.load_history[-1]) if sr.load_history else 0.0
        u = sr.u
        n_total_nodes = u.shape[0] // 6
        u_trans = u.reshape(n_total_nodes, 6)[:, :3]
        max_u_trans = float(np.max(np.linalg.norm(u_trans, axis=1)))
        diverged = False
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        frac = float("nan")
        max_u_trans = float("nan")
        sr = None
        diverged = True
        print(f"  [DIVERGED] {type(exc).__name__}: {exc}")

    if sr is not None:
        print(
            f"  frac={frac:.4f}, conv={sr.converged}, "
            f"incr={sr.n_increments}, cb={sr.n_cutbacks}, t={elapsed:.2f}s, "
            f"max|u|={max_u_trans:.3e} mm"
        )
    return dict(
        label=label,
        frac=frac,
        converged=bool(sr.converged) if sr is not None else False,
        n_increments=int(sr.n_increments) if sr is not None else 0,
        n_cutbacks=int(sr.n_cutbacks) if sr is not None else 0,
        elapsed=elapsed,
        max_u_trans=max_u_trans,
        diverged=diverged,
    )


def main() -> int:
    print("=" * 72)
    print("status-383 候補 (q1): explicit 中の UL update_reference 周期化")
    print("単梁 90° カンチレバー曲げ（L=100mm）")
    print("=" * 72)

    L_strand = 100.0
    R_analytical = 2.0 * L_strand / np.pi  # ≈ 63.66 mm
    u_analytical = float(np.sqrt((L_strand - R_analytical) ** 2 + R_analytical**2))
    print(f"\n解析解: R = 2L/π = {R_analytical:.3f} mm, max|u| = {u_analytical:.3f} mm")

    runs: list[dict] = []
    runs.append(_run_one("implicit_baseline", solver_mode="implicit"))

    # status-382 baseline — interval=1 + (p3)+(p1) (=既存最良構成)
    runs.append(_run_one(
        "q1_interval1_baseline",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e3,
        explicit_mass_scaling_max_growth_per_update=4.0,
        explicit_mass_proportional_damping_alpha=0.5,
        explicit_relax_steps=500,
        explicit_relax_tol=1.0e-4,
        explicit_ul_update_interval=1,
    ))

    # 候補 (q1): interval=5
    runs.append(_run_one(
        "q1_interval5",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e3,
        explicit_mass_scaling_max_growth_per_update=4.0,
        explicit_mass_proportional_damping_alpha=0.5,
        explicit_relax_steps=500,
        explicit_relax_tol=1.0e-4,
        explicit_ul_update_interval=5,
    ))

    # 候補 (q1): interval=10
    runs.append(_run_one(
        "q1_interval10",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e3,
        explicit_mass_scaling_max_growth_per_update=4.0,
        explicit_mass_proportional_damping_alpha=0.5,
        explicit_relax_steps=500,
        explicit_relax_tol=1.0e-4,
        explicit_ul_update_interval=10,
    ))

    # 候補 (q1): interval=20（n_increments_per_cycle=20 と同値 → main loop で 1 回のみ）
    runs.append(_run_one(
        "q1_interval20",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e3,
        explicit_mass_scaling_max_growth_per_update=4.0,
        explicit_mass_proportional_damping_alpha=0.5,
        explicit_relax_steps=500,
        explicit_relax_tol=1.0e-4,
        explicit_ul_update_interval=20,
    ))

    print()
    print("=" * 80)
    print(f"{'label':28s} | frac  |   max|u|     | err_anal | err_imp | gate")
    print("─" * 80)
    u_implicit = next(r["max_u_trans"] for r in runs if r["label"] == "implicit_baseline")
    for r in runs:
        if r["diverged"] or not np.isfinite(r["max_u_trans"]):
            print(
                f"  {r['label']:26s} | {'-':>5s} | {'DIVERGED':>11s} | "
                f"   ---  |   ---  | FAIL"
            )
            continue
        err_anal = abs(r["max_u_trans"] - u_analytical) / u_analytical
        err_imp = abs(r["max_u_trans"] - u_implicit) / u_implicit if u_implicit > 0 else 1.0
        gate_disp = r["max_u_trans"] < L_strand * 10.0
        gate_acc = (err_imp < 0.10) or (err_anal < 0.10)
        gate = "PASS" if (r["frac"] >= 0.999 and gate_disp and gate_acc) else "FAIL"
        print(
            f"  {r['label']:26s} | {r['frac']:5.3f} | {r['max_u_trans']:.3e} | "
            f"{err_anal * 100:6.2f}% | {err_imp * 100:6.2f}% | {gate}"
        )

    print()
    print("Gate (MCDD 凍結解除条件 (5)):")
    print("  - frac=1.0 完走 + max|u| < L_strand × 10 + 精度 < 10%")
    print(f"  - 解析解 max|u| = {u_analytical:.3f} mm（quarter circle）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
