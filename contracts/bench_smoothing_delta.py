"""smoothing_delta 効果検証ベンチマーク.

status-259 TODO: smoothing_delta 有効/無効でNR反復数・カットバック回数を比較。

[← README](../README.md)
"""

from __future__ import annotations

import sys
import time

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def _run_case(label: str, cfg: StrandBendingOscillationConfig) -> dict:
    """1ケース実行して結果を辞書で返す."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  smoothing_delta = {cfg.smoothing_delta}")
    print(f"{'=' * 60}")

    proc = StrandBendingOscillationProcess()
    t0 = time.perf_counter()
    result = proc.process(cfg)
    elapsed = time.perf_counter() - t0

    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0

    # NR反復統計
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

    print(f"\n--- 結果: {label} ---")
    for k, v in info.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # インクリメント詳細
    if sr.increment_diagnostics:
        print("\n  インクリメント診断:")
        for d in sr.increment_diagnostics:
            print(
                f"    step={d.step:3d}, frac={d.load_frac:.4f}, "
                f"nr={d.n_attempts:2d}, active={d.n_active}"
            )

    return info


def main() -> None:
    """ベースラインと smoothing_delta 有効を比較."""
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
        # smoothing_delta=1e30 → delta_h ≈ 0 → max(0,x) 相当（ベースライン）
        ("baseline (max(0,x))", StrandBendingOscillationConfig(**base_cfg, smoothing_delta=1e30)),
        # smoothing_delta=0 → 自動推定 5000/0.5 = 10000
        ("auto (5000/r=10000)", StrandBendingOscillationConfig(**base_cfg, smoothing_delta=0.0)),
        # smoothing_delta=2500/r = 5000
        ("half (2500/r=5000)", StrandBendingOscillationConfig(**base_cfg, smoothing_delta=5000.0)),
        # smoothing_delta=10000/r = 20000
        ("double (10000/r=20000)", StrandBendingOscillationConfig(**base_cfg, smoothing_delta=20000.0)),
    ]

    results = []
    for label, cfg in cases:
        try:
            info = _run_case(label, cfg)
            results.append(info)
        except Exception as e:
            print(f"\n!!! {label} failed: {e}")
            results.append({"label": label, "error": str(e)})

    # 比較表
    print("\n\n" + "=" * 80)
    print("  smoothing_delta 効果比較")
    print("=" * 80)
    header = f"{'label':30s} {'delta':>10s} {'frac':>8s} {'incr':>6s} {'cutback':>8s} {'NR_tot':>8s} {'NR_avg':>8s} {'NR_max':>8s} {'time':>8s}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['label']:30s}  ERROR: {r['error']}")
            continue
        sd = r["smoothing_delta"]
        sd_str = f"{sd:.0e}" if sd > 1e10 else f"{sd:.0f}"
        print(
            f"{r['label']:30s} {sd_str:>10s} {r['frac']:8.4f} "
            f"{r['n_increments']:6d} {r['n_cutbacks']:8d} "
            f"{r['total_attempts']:8d} {r['avg_nr']:8.1f} "
            f"{r['max_nr']:8d} {r['elapsed']:8.1f}s"
        )


if __name__ == "__main__":
    main()
