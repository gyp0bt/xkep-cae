"""huber_delta_h 直接指定スイープ（問題非依存最適値探索）.

status-261 TODO #2: delta_h 最適値の問題非依存探索。
梁-梁で delta_h=0.01-0.03 が有効範囲（δ=1000 → delta_h=k_pen/δ≈0.031）。
huber_delta_h 直接指定APIを使い、k_penスケール非依存の最適値を探索する。

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
    print(f"  {label} (huber_delta_h={cfg.huber_delta_h}, smoothing_delta={cfg.smoothing_delta})")
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
        "huber_delta_h": cfg.huber_delta_h,
        "frac": frac,
        "converged": sr.converged,
        "n_increments": sr.n_increments,
        "n_cutbacks": sr.n_cutbacks,
        "total_attempts": sr.total_attempts,
        "avg_nr": avg_nr,
        "max_nr": max_nr,
        "elapsed": elapsed,
    }

    print(
        f"  frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}, "
        f"NR_avg={avg_nr:.1f}, NR_max={max_nr}, time={elapsed:.1f}s"
    )

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
        smoothing_delta=0.0,  # huber_delta_h 直接指定時は smoothing_delta 無効化
    )

    # delta_h スイープ: 0.005 〜 0.05（status-260 分析: k_pen~31, δ=1000 → delta_h≈0.031）
    cases = [
        ("delta_h=0 (no smooth)", StrandBendingOscillationConfig(**base_cfg, huber_delta_h=0.0)),
        ("delta_h=0.005", StrandBendingOscillationConfig(**base_cfg, huber_delta_h=0.005)),
        ("delta_h=0.010", StrandBendingOscillationConfig(**base_cfg, huber_delta_h=0.010)),
        ("delta_h=0.015", StrandBendingOscillationConfig(**base_cfg, huber_delta_h=0.015)),
        ("delta_h=0.020", StrandBendingOscillationConfig(**base_cfg, huber_delta_h=0.020)),
        ("delta_h=0.025", StrandBendingOscillationConfig(**base_cfg, huber_delta_h=0.025)),
        ("delta_h=0.030", StrandBendingOscillationConfig(**base_cfg, huber_delta_h=0.030)),
        ("delta_h=0.040", StrandBendingOscillationConfig(**base_cfg, huber_delta_h=0.040)),
        ("delta_h=0.050", StrandBendingOscillationConfig(**base_cfg, huber_delta_h=0.050)),
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
    print("  huber_delta_h スイープ結果（strand_bending, 7本撚線）")
    print("=" * 80)
    header = (
        f"{'label':25s} {'delta_h':>10s} {'frac':>8s} {'incr':>6s} "
        f"{'cutback':>8s} {'NR_avg':>8s} {'NR_max':>8s} {'time':>8s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['label']:25s}  ERROR: {r['error']}")
            continue
        print(
            f"{r['label']:25s} {r['huber_delta_h']:10.3f} {r['frac']:8.4f} "
            f"{r['n_increments']:6d} {r['n_cutbacks']:8d} "
            f"{r['avg_nr']:8.1f} {r['max_nr']:8d} {r['elapsed']:8.1f}s"
        )


if __name__ == "__main__":
    main()
