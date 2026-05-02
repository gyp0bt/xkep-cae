"""work/beam_hysteresis/40_explicit_n_inc_sweep.py — `n_increments` 大化掃引.

[← README](README.md) | [← project README](../../README.md)

status-386 §5.4 副次「t_cycle 据え置き + n_increments 大」探索。

status-386 #11 で `t_cycle=1.0s` 据え置き + `n_inc=200`（dt_sub=5e-3）+
`selective=False`（uniform β² 一律スケール）で max|u|=6.57mm（z1d 方向の **10x
改善**、解析解 73.30mm の 9% / err 91%）が得られた。本 status-387 では n_inc
をさらに段階的に拡大し、精度 gate (5)（err < 10%）に向かうのか、
あるいは UL 凍結の本質欠陥でプラトー化するのかを定量化する。

**理論予測**:

- target β = dt_sub / (0.9·dt_c_orig)、dt_c_orig ≈ 1.6e-6 s（単梁 L=100mm 推定）
- 弾性波伝播時間 = β · L / c、c = √(E/ρ) ≈ 3.81e6 mm/s
- 横断回数 n = t_cycle / wave_traverse_time = t_cycle · c / (β · L)

| n_inc | dt_sub [s] | target β | wave_traverse [s] | 横断回数 |
|------:|-----------:|---------:|------------------:|---------:|
|   200 |   5.00e-3  |     3500 |          92e-3   |     11   |
|   500 |   2.00e-3  |     1400 |          37e-3   |     27   |
|  1000 |   1.00e-3  |      700 |          18e-3   |     54   |
|  2000 |   5.00e-4  |      350 |          9.2e-3  |    109   |
|  4000 |   2.50e-4  |      175 |          4.6e-3  |    217   |
|  8000 |   1.25e-4  |       88 |          2.3e-3  |    435   |
| 16000 |   6.25e-5  |       44 |          1.1e-3  |    870   |

横断回数が十分大きい（>100）と過渡応答が定常解析解に収束するはず。

**Gate** (MCDD 凍結解除条件):
1. frac = 1.0
2. max|u_trans| < L_strand × 10 = 1000 mm
3. **|max|u_explicit| − u_analytical| / u_analytical < 0.10**（単梁解析解 73.30mm）

実行:
    uv run --extra dev python work/beam_hysteresis/40_explicit_n_inc_sweep.py \\
        2>&1 | tee /tmp/n_inc_sweep_$(date +%s).log
"""

from __future__ import annotations

import time

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def _make_single_beam_cfg() -> dict:
    """単梁 90° 曲げ baseline（接触なし、L=100 mm、E=130 GPa、`39` と同条件）."""
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


def _run_one(label: str, cfg_factory, **overrides) -> dict:
    base = cfg_factory()
    base.update(overrides)
    cfg = StrandBendingOscillationConfig(**base)

    print()
    print("─" * 72)
    print(f"[run] {label}")
    print("─" * 72)
    print(f"  solver_mode = {cfg.solver_mode}")
    print(f"  n_increments_per_cycle = {cfg.n_increments_per_cycle}")
    if cfg.solver_mode == "explicit":
        print(f"  explicit_mass_scaling_max_beta = {cfg.explicit_mass_scaling_max_beta:.2e}")
        print(f"  explicit_mass_scaling_selective = {cfg.explicit_mass_scaling_selective}")

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


def _analytical_max_u(L_strand: float) -> float:
    """90° カンチレバー曲げの解析解 max|u_trans|（quarter circle）."""
    R = 2.0 * L_strand / np.pi
    return float(np.sqrt((L_strand - R) ** 2 + R**2))


def _summarize(runs: list[dict], u_analytical: float, L_strand: float) -> None:
    print()
    print("=" * 96)
    print(f"{'label':54s} | frac  |   max|u|     | err_anal | t [s] | gate")
    print("─" * 96)
    for r in runs:
        if r["diverged"] or not np.isfinite(r["max_u_trans"]):
            print(
                f"  {r['label']:52s} | {'-':>5s} | {'DIVERGED':>11s} | "
                f"   ---  | {r['elapsed']:5.1f} | FAIL"
            )
            continue
        err_anal = abs(r["max_u_trans"] - u_analytical) / u_analytical
        gate_disp = r["max_u_trans"] < L_strand * 10.0
        gate_acc = err_anal < 0.10
        gate = "PASS" if (r["frac"] >= 0.999 and gate_disp and gate_acc) else "FAIL"
        print(
            f"  {r['label']:52s} | {r['frac']:5.3f} | {r['max_u_trans']:.3e} | "
            f"{err_anal * 100:6.2f}% | {r['elapsed']:5.1f} | {gate}"
        )


def main() -> int:
    print("=" * 72)
    print("status-387 副次 — `n_increments` 大化掃引（t_cycle=1.0s 据え置き）")
    print("=" * 72)

    L_strand = 100.0
    u_analytical = _analytical_max_u(L_strand)
    print(f"\n単梁 90° 曲げ解析解: L={L_strand}mm → max|u| = {u_analytical:.3f} mm")
    print("(MCDD 条件 (5) gate: |max|u_explicit| − u_anal| / u_anal < 0.10)")

    print("\n" + "=" * 72)
    print("単梁 90° 曲げ — explicit uniform β² (selective=False) n_inc 掃引")
    print("=" * 72)

    runs: list[dict] = []

    # ── reference ──
    runs.append(
        _run_one(
            "implicit_baseline (reference, n_inc=20)",
            _make_single_beam_cfg,
            solver_mode="implicit",
        )
    )

    # status-386 #11 を再掲（n_inc=200）
    runs.append(
        _run_one(
            "exp_n_inc=200 (status-386 #11 再掲)",
            _make_single_beam_cfg,
            solver_mode="explicit",
            n_increments_per_cycle=200,
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            explicit_mass_scaling_auto=True,
            explicit_mass_scaling_max_beta=1.0e4,
            explicit_mass_scaling_max_growth_per_update=4.0,
            explicit_mass_scaling_selective=False,
        )
    )

    # n_inc 段階的拡大
    for n_inc in (500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000):
        runs.append(
            _run_one(
                f"exp_n_inc={n_inc}",
                _make_single_beam_cfg,
                solver_mode="explicit",
                n_increments_per_cycle=n_inc,
                explicit_courant_safety=0.9,
                explicit_courant_check_interval=10,
                explicit_mass_scaling_auto=True,
                explicit_mass_scaling_max_beta=1.0e4,
                explicit_mass_scaling_max_growth_per_update=4.0,
                explicit_mass_scaling_selective=False,
            )
        )

    # ── damping + relax で overshoot 抑制を試す ──
    # n_inc=16000 で overshoot (max|u|=106mm, err=44.76%) が観測された場合、
    # 質量比例 damping + BC 完了後 relax phase で静的解析解 (73.30mm) に
    # 収束させられるか検証する。
    print("\n" + "=" * 72)
    print("damping + relax 併用 — sweet spot 周辺 + overshoot 抑制")
    print("=" * 72)
    for n_inc, alpha, relax in (
        (8000, 5.0, 200),
        (12000, 5.0, 200),
        (16000, 5.0, 500),
    ):
        runs.append(
            _run_one(
                f"exp_n_inc={n_inc}_damp{alpha}_relax{relax}",
                _make_single_beam_cfg,
                solver_mode="explicit",
                n_increments_per_cycle=n_inc,
                explicit_courant_safety=0.9,
                explicit_courant_check_interval=10,
                explicit_mass_scaling_auto=True,
                explicit_mass_scaling_max_beta=1.0e4,
                explicit_mass_scaling_max_growth_per_update=4.0,
                explicit_mass_scaling_selective=False,
                explicit_mass_proportional_damping_alpha=alpha,
                explicit_relax_steps=relax,
                explicit_relax_tol=1e-3,
            )
        )

    _summarize(runs, u_analytical, L_strand)

    print()
    print("Gate (MCDD 凍結解除条件):")
    print("  - (1) frac=1.0 完走")
    print(f"  - (2) max|u| < L_strand × 10 = {L_strand * 10.0:.0f} mm")
    print(f"  - (5) 解析解誤差 < 10% （単梁 90° 解析解 max|u| = {u_analytical:.3f} mm）")
    print()
    print("評価ポイント:")
    print("  - max|u| が n_inc 増加で単調改善 → gate 達成可能性あり")
    print("  - max|u| がプラトー化 → UL 凍結 (status-382/383) の本質欠陥が支配")
    print("  - elapsed が n_inc にほぼ比例 → 1 substep/increment（β が target を吸収）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
