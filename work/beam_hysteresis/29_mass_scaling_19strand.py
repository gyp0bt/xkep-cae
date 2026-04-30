"""work/beam_hysteresis/29_mass_scaling_19strand.py — 候補 (h1) 質量スケーリング 19 本実機検証.

[← README](README.md) | [← project README](../../README.md)

status-379 候補 (h1) 質量スケーリング（Belytschko §6.4.2）の 19 本撚線 90° 曲げ
実機検証。`explicit_mass_scaling_beta` を手動掃引、または `mass_scaling_auto=True` で
Courant 違反を自動回避することで、status-378 で実測した Courant 比 3×10⁵ を縮める。

**ベースライン**: status-378 7 本 explicit smoke で frac=0.0052（max_increments=3 早期打切）。
status-376 implicit + AL n=2 で 19 本 frac=0.5746（gate 0.6 0.026 不足）。

**Gate**:
- frac=1.0 完走
- mass scaling 影響を E_kin / E_strain 比 < 5% に抑制（準静的近似破綻チェック）

実行例:
    # 手動 β 指定（mass_scaling_auto=False）
    python work/beam_hysteresis/29_mass_scaling_19strand.py 100 2>&1 | tee /tmp/ms_19strand_$(date +%s).log

    # auto-tune（max β 1000）
    python work/beam_hysteresis/29_mass_scaling_19strand.py auto 2>&1 | tee /tmp/ms_19strand_auto_$(date +%s).log

引数:
    β（数値）: explicit_mass_scaling_beta、explicit_mass_scaling_auto=False
    "auto":   explicit_mass_scaling_auto=True、max_beta=1000

設計参照:
    docs/status/status-378.md §5.2（Phase 3 候補）
    xkep_cae/time_integration/docs/time_integration_explicit.md §19 本撚線 Type D stall への意図
    xkep_cae/contact/solver/docs/explicit_dynamic.md §既知のスケーリング障壁
"""

from __future__ import annotations

import sys
import time

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else "1.0"

    if arg == "auto":
        beta = 1.0
        auto = True
        max_beta = 1000.0
    else:
        beta = float(arg)
        auto = False
        max_beta = 100.0

    print("=" * 72)
    print(f"候補 (h1) 質量スケーリング 19 本撚線 90° 曲げ — β={beta} auto={auto}")
    print("=" * 72)
    print("ベースライン: 19 本 implicit + AL n=2 frac=0.5746 (status-376)")
    print("Gate: frac=1.0 完走 + E_kin/E_strain < 5%")
    print()

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
        # ★ 候補 (h1) 質量スケーリング + 陽解法
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=beta,
        explicit_mass_scaling_auto=auto,
        explicit_mass_scaling_max_beta=max_beta,
        explicit_kinetic_energy_budget_ratio=0.05,
    )

    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result

    frac = sr.load_history[-1] if sr.load_history else 0.0
    n_incr = int(sr.n_increments)
    n_cb = int(sr.n_cutbacks)
    converged = bool(sr.converged)

    e_hist = sr.energy_history
    e_kin = e_hist.entries[-1].kinetic_energy if e_hist and e_hist.entries else 0.0
    e_str = e_hist.entries[-1].strain_energy if e_hist and e_hist.entries else 0.0
    e_ratio = e_kin / max(e_str, 1e-30)

    print()
    print("─" * 72)
    print(
        f"β={beta} auto={auto}: frac={frac:.4f}, "
        f"incr={n_incr}, cb={n_cb}, elapsed={elapsed:.2f}s, converged={converged}"
    )
    print(f"  E_kin (final) = {e_kin:.3e}")
    print(f"  E_strain (final) = {e_str:.3e}")
    print(f"  E_kin / E_strain = {e_ratio:.3e}")

    gate_frac = frac >= 0.999
    gate_energy = e_ratio < 0.05
    gate_passed = gate_frac and gate_energy
    print(f"  Gate frac=1.0: {'PASS' if gate_frac else 'FAIL'}")
    print(f"  Gate E_kin/E_strain<5%: {'PASS' if gate_energy else 'FAIL'}")
    print(f"  Total: {'PASS' if gate_passed else 'FAIL'}")
    print("=" * 72)

    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
