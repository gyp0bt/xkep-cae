"""work/beam_hysteresis/33_explicit_single_beam_beta_fixed.py — β 固定 vs auto-tune 切り分け.

[← README](README.md) | [← project README](../../README.md)

status-381 の h-bug 切り分けスクリプト。

status-380 §2.0 で `32_explicit_single_beam_no_contact.py` により接触なし単梁でも
explicit + auto-tune が max |u|=1.58e8 mm に発散することが確定。

本スクリプトは **β 固定（auto-tune 無効）** で同じ単梁問題を解き、
発散原因の切り分けを行う:

- β 固定で妥当 → auto-tune 動的更新時の状態リスケール欠落（h-bug-1）が主因
- β 固定でも発散 → mass scaling 実装そのものが broken（h-bug-2 / 3）

Gate: max |u_trans| < L_strand × 10
"""

from __future__ import annotations

import sys
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
    print(f"[run] label={label}")
    for key in ("solver_mode", "explicit_mass_scaling_beta",
                "explicit_mass_scaling_auto", "explicit_mass_scaling_max_beta"):
        print(f"        {key}={base.get(key)}")
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
        f"  frac={frac:.4f}, converged={sr.converged}, "
        f"incr={sr.n_increments}, cb={sr.n_cutbacks}, elapsed={elapsed:.2f}s"
    )
    print(f"  max |u_trans| = {max_u_trans:.4e} mm")
    return dict(label=label, frac=frac, converged=bool(sr.converged),
                max_u_trans=max_u_trans, u=u.copy())


def main() -> int:
    print("=" * 72)
    print("status-381: β 固定 vs auto-tune 切り分け（接触なし単梁）")
    print("=" * 72)

    runs: list[dict] = []

    # implicit reference
    runs.append(_run_one("implicit", solver_mode="implicit"))

    # explicit β=1 (no scaling)
    runs.append(_run_one(
        "explicit_beta1_noauto",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=False,
    ))

    # explicit β=1e2 fixed
    runs.append(_run_one(
        "explicit_beta100_fixed",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0e2,
        explicit_mass_scaling_auto=False,
    ))

    # explicit β=1e3 fixed
    runs.append(_run_one(
        "explicit_beta1000_fixed",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0e3,
        explicit_mass_scaling_auto=False,
    ))

    # explicit auto-tune (status-380 baseline)
    runs.append(_run_one(
        "explicit_auto_max1e3",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e3,
    ))

    L_strand = 100.0
    print()
    print("=" * 72)
    print("Gate 判定（max |u_trans| < L_strand × 10 = 1m）")
    print("=" * 72)
    print(f"{'label':40s} | {'frac':6s} | {'max|u| [mm]':14s} | gate")
    for r in runs:
        gate = "PASS" if r["max_u_trans"] < L_strand * 10 else "FAIL"
        print(f"  {r['label']:38s} | {r['frac']:6.3f} | {r['max_u_trans']:.4e} | {gate}")

    print()
    # 切り分け
    fixed_betas = [r for r in runs if "fixed" in r["label"]]
    auto = [r for r in runs if "auto" in r["label"]]
    fixed_pass = all(r["max_u_trans"] < L_strand * 10 for r in fixed_betas)
    auto_pass = all(r["max_u_trans"] < L_strand * 10 for r in auto)

    print("=" * 72)
    print("切り分け")
    print("=" * 72)
    if fixed_pass and not auto_pass:
        print("  → β 固定で妥当・auto-tune で発散")
        print("  → h-bug-1（set_mass_scaling_beta() 動的更新時の v/a リスケール欠落）が主因")
    elif fixed_pass and auto_pass:
        print("  → 全 explicit case で妥当（baseline と異なる）")
    elif not fixed_pass and not auto_pass:
        print("  → β 固定でも発散")
        print("  → h-bug-2/3（mass scaling 実装そのものに問題）")
    else:
        print("  → 想定外: auto-tune は妥当だが β 固定で発散")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
