"""work/beam_hysteresis/24_contact_damping_19strand.py — 候補 (e) 接触減衰 19本撚線適用.

[← README](README.md) | [← project README](../../README.md)

status-367 で 7 本撚線にて特定した c_n を 19 本撚線 Type D stall 本体に適用し、
MCDD 凍結解除条件「frac=1.0 完走 + E_damp/E_strain < budget」を判定する。

**ベースライン（status-339 / status-360）**:

- baseline (c_n=0, smoothing_delta=1000 or 2000): frac≈0.484 で Type D stall
- line search (status-362): frac=0.5153 (+6.5% 改善、未完走)

**本スクリプト**:

- `c_n` を引数で受け取り（default: 7本で推奨された値）
- frac / incr / cb / elapsed + E_damp/E_strain ratio を出力
- MCDD 凍結解除条件（frac=1.0 & ratio < budget）判定

実行:
    # 7本で特定した c_n（例: 100.0）を 19本に適用
    uv run python work/beam_hysteresis/24_contact_damping_19strand.py 100.0 2>&1 \\
        | tee /tmp/contact_damping_19strand_$(date +%s).log
"""

from __future__ import annotations

import sys
import time

from xkep_cae.contact.damping import (
    ContactDampingEnergyMonitorInput,
    ContactDampingEnergyMonitorProcess,
)
from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def main() -> int:
    """19本撚線 90° 曲げ + 接触減衰."""
    c_n = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0

    print("=" * 72)
    print(f"候補 (e) 接触減衰 19本撚線 90° 曲げ — c_n={c_n:.3e}")
    print("=" * 72)
    print("ベースライン（status-339、c_n=0）: frac=0.4839, incr=271, cb=39, 534.68s")
    print("line search (status-362): frac=0.5153 (+6.5%, 未完走)")

    cfg = StrandBendingOscillationConfig(
        n_strands=19,
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
        penalty_exponent=1.5,
        smoothing_delta=1000.0,
        track_contact_mk=False,
        track_contact_pairs=False,
        # ★ 候補 (e) 接触減衰
        contact_damping_coefficient=c_n,
        contact_damping_energy_budget_ratio=0.20,
    )

    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result

    frac = sr.load_history[-1] if sr.load_history else 0.0
    damp_hist = sr.damping_energy_history
    e_damp_final = damp_hist[-1][1] if damp_hist else 0.0
    e_strain_final = 0.0
    if sr.energy_history and getattr(sr.energy_history, "entries", None):
        e_strain_final = float(sr.energy_history.entries[-1].strain_energy)

    mon_out = ContactDampingEnergyMonitorProcess().process(
        ContactDampingEnergyMonitorInput(
            damping_energy_history=damp_hist,
            energy_history=sr.energy_history,
            budget_ratio=0.20,
            log_every_n_steps=50,
        )
    )
    print()
    print(mon_out.report)

    print()
    print("=" * 72)
    print("結果")
    print("=" * 72)
    print(f"  c_n:            {c_n:.3e}")
    print(f"  frac_completed: {frac:.4f}")
    print(f"  converged:      {sr.converged}")
    print(f"  n_increments:   {sr.n_increments}")
    print(f"  n_cutbacks:     {sr.n_cutbacks}")
    print(f"  elapsed:        {elapsed:.2f} s")
    print(f"  E_damp_final:   {e_damp_final:.6e}")
    print(f"  E_strain_final: {e_strain_final:.6e}")
    print(f"  max_ratio:      {mon_out.max_ratio:.6e}")
    print(f"  final_ratio:    {mon_out.final_ratio:.6e}")

    # MCDD 凍結解除条件判定
    print()
    print("=" * 72)
    print("MCDD 凍結解除条件判定")
    print("=" * 72)
    cond_frac = frac >= 0.9999
    cond_budget_20 = mon_out.max_ratio < 0.20
    cond_budget_10 = mon_out.max_ratio < 0.10
    print(f"  frac=1.0 完走:            {'YES' if cond_frac else 'NO'} (frac={frac:.4f})")
    print(
        f"  E_damp/E_strain < 20%:   {'YES' if cond_budget_20 else 'NO'} (max={mon_out.max_ratio:.3e})"
    )
    print(f"  E_damp/E_strain < 10%:   {'YES' if cond_budget_10 else 'NO'}")
    if cond_frac and cond_budget_10:
        print("  → MCDD 凍結解除条件 **達成**（frac=1.0 + budget<10%）")
    elif cond_frac and cond_budget_20:
        print("  → 条件 **部分達成**（frac=1.0 + budget<20%、margin 薄い）")
    elif cond_frac:
        print("  → frac=1.0 完走したが budget 超過。c_n 縮小要")
    else:
        print("  → frac=1.0 未達。c_n 増量 or 他候補検討")

    return 0 if sr.converged and cond_frac else 1


if __name__ == "__main__":
    sys.exit(main())
