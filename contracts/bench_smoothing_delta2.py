"""smoothing_delta 効果検証ベンチマーク（微調整版）.

delta=5000 が最良だったので、さらに小さい値を探索。
delta_h = k_pen / smoothing_delta なので、小さい smoothing_delta → 広い平滑化幅。

[← README](../README.md)
"""

from __future__ import annotations

import time

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def _run_case(label: str, cfg: StrandBendingOscillationConfig) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {label} (smoothing_delta={cfg.smoothing_delta})")
    print(f"{'=' * 60}")

    proc = StrandBendingOscillationProcess()
    t0 = time.perf_counter()
    result = proc.process(cfg)
    elapsed = time.perf_counter() - t0

    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0
    nr_attempts = [d.n_attempts for d in sr.increment_diagnostics]
    avg_nr = sum(nr_attempts) / len(nr_attempts) if nr_attempts else 0
    max_nr = max(nr_attempts) if nr_attempts else 0

    info = {
        "label": label,
        "smoothing_delta": cfg.smoothing_delta,
        "frac": frac,
        "converged": sr.converged,
        "n_increments": sr.n_increments,
        "n_cutbacks": sr.n_cutbacks,
        "total_attempts": sr.total_attempts,
        "avg_nr": avg_nr,
        "max_nr": max_nr,
        "elapsed": elapsed,
    }

    print(f"  frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}, "
          f"NR_avg={avg_nr:.1f}, NR_max={max_nr}, time={elapsed:.1f}s")

    return info


def main() -> None:
    base_cfg = dict(
        n_strands=7,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.001,
        n_cycles=1,
        n_increments_per_cycle=40,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=50,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
    )

    cases = [
        ("delta=1000", StrandBendingOscillationConfig(**base_cfg, smoothing_delta=1000.0)),
        ("delta=2000", StrandBendingOscillationConfig(**base_cfg, smoothing_delta=2000.0)),
        ("delta=3000", StrandBendingOscillationConfig(**base_cfg, smoothing_delta=3000.0)),
        ("delta=5000 (前回最良)", StrandBendingOscillationConfig(**base_cfg, smoothing_delta=5000.0)),
        ("delta=7500", StrandBendingOscillationConfig(**base_cfg, smoothing_delta=7500.0)),
    ]

    results = []
    for label, cfg in cases:
        try:
            info = _run_case(label, cfg)
            results.append(info)
        except Exception as e:
            print(f"\n!!! {label} failed: {e}")
            results.append({"label": label, "error": str(e)})

    print("\n\n" + "=" * 80)
    print("  smoothing_delta チューニング結果")
    print("=" * 80)
    header = f"{'label':30s} {'delta':>10s} {'frac':>8s} {'incr':>6s} {'cutback':>8s} {'NR_avg':>8s} {'NR_max':>8s} {'time':>8s}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['label']:30s}  ERROR: {r['error']}")
            continue
        print(
            f"{r['label']:30s} {r['smoothing_delta']:10.0f} {r['frac']:8.4f} "
            f"{r['n_increments']:6d} {r['n_cutbacks']:8d} "
            f"{r['avg_nr']:8.1f} {r['max_nr']:8d} {r['elapsed']:8.1f}s"
        )


if __name__ == "__main__":
    main()
