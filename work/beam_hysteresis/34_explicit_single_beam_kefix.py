"""work/beam_hysteresis/34_explicit_single_beam_kefix.py — h-bug-1 KE 保存 fix 検証.

[← README](README.md) | [← project README](../../README.md)

status-381 修正後の接触なし単梁での妥当性確認:

修正内容:
- (1) `set_mass_scaling_beta()` で v/a を KE 保存リスケール（h-bug-1 fix）
- (2) auto-tune の β 増加レートを 4× / update に cap（h-bug-3 緩和）

status-380 で β=1000 固定では妥当 (max|u|=181mm) だったため、
`max_beta` を target に届く十分な大きさに設定すれば auto-tune も妥当解を返すはず。

Gate: max |u_trans| < L_strand × 10
"""

from __future__ import annotations

import time

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def _make_base_cfg() -> dict:
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
        max_increments=10000,
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
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result

    frac = float(sr.load_history[-1]) if sr.load_history else 0.0
    u = sr.u
    n_total_nodes = u.shape[0] // 6
    u_trans = u.reshape(n_total_nodes, 6)[:, :3]
    max_u_trans = float(np.max(np.linalg.norm(u_trans, axis=1)))

    print(
        f"  frac={frac:.4f}, conv={sr.converged}, "
        f"incr={sr.n_increments}, cb={sr.n_cutbacks}, t={elapsed:.2f}s, "
        f"max|u|={max_u_trans:.3e} mm"
    )
    return dict(label=label, frac=frac, converged=bool(sr.converged),
                n_increments=int(sr.n_increments), n_cutbacks=int(sr.n_cutbacks),
                elapsed=elapsed, max_u_trans=max_u_trans, u=u.copy())


def main() -> int:
    print("=" * 72)
    print("status-381 修正検証: 接触なし単梁 90° 曲げ")
    print("(h-bug-1 KE 保存 + h-bug-3 4× growth cap)")
    print("=" * 72)
    runs: list[dict] = []
    runs.append(_run_one("implicit_baseline", solver_mode="implicit"))
    # max_beta = 1e3 (status-380 baseline)
    runs.append(_run_one(
        "exp_auto_max1e3_cap4x",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e3,
        explicit_mass_scaling_max_growth_per_update=4.0,
    ))
    # max_beta = 1e5 (target ~4e3 で十分)
    runs.append(_run_one(
        "exp_auto_max1e5_cap4x",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e5,
        explicit_mass_scaling_max_growth_per_update=4.0,
    ))
    # max_beta = 1e5, smoother growth (2x)
    runs.append(_run_one(
        "exp_auto_max1e5_cap2x",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e5,
        explicit_mass_scaling_max_growth_per_update=2.0,
    ))
    # initial β=10 (warm start)
    runs.append(_run_one(
        "exp_auto_init10_max1e5_cap4x",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=10.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e5,
        explicit_mass_scaling_max_growth_per_update=4.0,
    ))
    # initial β=100 (warm start)
    runs.append(_run_one(
        "exp_auto_init100_max1e5_cap4x",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=100.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e5,
        explicit_mass_scaling_max_growth_per_update=4.0,
    ))
    # initial β=1000 fixed (status-380 で PASS した条件を再確認)
    runs.append(_run_one(
        "exp_init1000_noauto",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0e3,
        explicit_mass_scaling_auto=False,
    ))
    # initial β=5000 + auto (target 4346 を満たす warm start)
    runs.append(_run_one(
        "exp_init5000_auto_max1e5",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=5.0e3,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e5,
        explicit_mass_scaling_max_growth_per_update=4.0,
    ))

    L_strand = 100.0
    print()
    print("=" * 72)
    print(f"{'label':40s} | frac  |   max|u|     | gate")
    print("─" * 72)
    for r in runs:
        gate = "PASS" if (r["frac"] >= 0.999 and r["max_u_trans"] < L_strand * 10) else "FAIL"
        print(f"  {r['label']:38s} | {r['frac']:5.3f} | {r['max_u_trans']:.3e} | {gate}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
