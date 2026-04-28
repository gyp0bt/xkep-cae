"""work/beam_hysteresis/28_al_outer_loop_19strand.py — 候補 (g2) AL 外側ループ 19 本実機検証.

[← README](README.md) | [← project README](../../README.md)

status-376 候補 (g2) Augmented Lagrangian 限定再導入（status-221 凍結解除）の
19 本撚線 90° 曲げ実機検証。`al_n_uzawa_max ∈ {1, 2, 3}` を掃引し、Type D stall
（active 集合振動 + K_c x/z カップリング不整合）が AL 外側 Uzawa 更新で escape できるか
評価する。

**ベースライン（status-357 19 本 K_c FD 再計測時）**: frac=0.3739。
**Gate 基準**（status-375 §引継ぎ）: 19 本 `frac ≥ 0.6`（baseline 比 +60%）。

数理: docs/math/03_huber_contact_penalty.md §9 "Augmented Lagrangian motivation"
（status-376 で追記）。

実装: `xkep_cae/contact/contact_force/strategy.py::HuberContactForceProcess`
の `set_al_lambda_offset()` で per-pair λ 注入、`p_n_eff = max(0, p_n_huber + λ)` を
f_c に内包。`xkep_cae/contact/solver/_newton_dynamic.py` の AL 外側 for ループで
内側 NR 完了後に Uzawa 更新 `λ_new = max(0, p_n_eff_converged)` を実施。

実行:
    python work/beam_hysteresis/28_al_outer_loop_19strand.py 2 2>&1 \\
        | tee /tmp/al_outer_19strand_$(date +%s).log

引数: al_n_uzawa_max（既定 2）。1 で AL 外側無効化（純粋ペナルティ baseline 再現）、
2〜3 で AL Uzawa 更新を 1〜2 回実施。
"""

from __future__ import annotations

import sys
import time

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def main() -> int:
    al_n_uzawa_max = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    print("=" * 72)
    print(f"候補 (g2) AL 外側ループ 19 本撚線 90° 曲げ — al_n_uzawa_max={al_n_uzawa_max}")
    print("=" * 72)
    print("ベースライン (status-357): frac=0.3739, Type D+E:67%, E:28%")
    print("Gate: frac >= 0.6 （baseline 0.3739 比 +60%）")
    print()

    al_enabled = al_n_uzawa_max >= 2
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
        # ★ 候補 (g2) AL 外側ループ
        al_outer_enabled=al_enabled,
        al_n_uzawa_max=al_n_uzawa_max,
    )

    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result

    frac = sr.load_history[-1] if sr.load_history else 0.0
    n_incr = int(sr.n_increments)
    n_cb = int(sr.n_cutbacks)
    converged = bool(sr.converged)

    print()
    print("─" * 72)
    print(
        f"al_n_uzawa_max={al_n_uzawa_max}: frac={frac:.4f}, "
        f"incr={n_incr}, cb={n_cb}, elapsed={elapsed:.2f}s, converged={converged}"
    )
    base_frac = 0.3739
    delta_pct = (frac - base_frac) / base_frac * 100.0
    print(f"  ベースライン比: {delta_pct:+.1f}% (baseline frac=0.3739)")
    gate_passed = frac >= 0.6
    print(f"  Gate frac >= 0.6: {'PASS' if gate_passed else 'FAIL'}")
    print("=" * 72)

    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
