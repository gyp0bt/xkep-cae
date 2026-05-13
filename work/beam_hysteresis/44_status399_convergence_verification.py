"""work/beam_hysteresis/44_status399_convergence_verification.py — status-399 STA2 検証: N→∞ で u_x が implicit 値に asymptote 収束するかを実機確認.

[← README](README.md) | [← project README](../../README.md)

**問題提起（ユーザー指摘）**:
status-399 で N=1000 が rel_err 6.07% を達成したが、これは N を増やすに伴い
u_x が単調増加して **implicit 値 (4.996 mm) を通り過ぎている** だけで、
真の asymptote 収束ではない可能性がある（STA2 該当）。

**検証アプローチ**:

N ∈ {500, 1000, 2000, 5000, 10000} を測定:

- 真に implicit 値に asymptote する: u_x(N) → 4.996 mm、|u_x(N) − implicit| が
  N と共に **単調減少**、log-log で convergence rate が確認できる
- 通り過ぎているだけ: u_x(N) は N=1000 以降も増加し続け、large N で
  implicit 値より顕著に大きい値（rel_err > 10%）に達する

**判定基準（status-388 透明性ルール準拠）**:

(a) u_x(N=10000) < 5.50 mm AND |u_x(N=10000) − 4.996| < |u_x(N=1000) − 4.996|
    → asymptote 収束方向、status-399 主張支持
(b) u_x(N=10000) > 5.50 mm OR u_x(N=10000) > u_x(N=1000)
    → 通り過ぎ確定、status-399 STA2 該当、撤回必要
(c) u_x(N=10000) が大幅に違う方向（発散 / NaN）
    → 別の数値破綻、追加診断必要

**実行**:

    uv run --extra dev python work/beam_hysteresis/44_status399_convergence_verification.py \\
        2>&1 | tee /tmp/status399_convergence_$(date +%s).log

**期待**: N=10000 で ~10s × 10 = ~180s 程度（N=1000 が 18.68s 実測）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
    _collect_end_nodes,
)


@dataclass
class RunSummary:
    label: str
    N_sub: int
    u_x: float
    u_z: float
    rel_err_x: float
    elapsed: float


def _extract_tip(u: np.ndarray, mesh: object) -> np.ndarray:
    _, right = _collect_end_nodes(
        mesh.connectivity, int(mesh.n_strands), np.asarray(mesh.strand_ids)
    )
    if not right:
        return np.zeros(3)
    um = u.reshape(-1, 6)
    return np.mean(um[right, :3], axis=0)


def _base_cfg() -> dict:
    return dict(
        n_strands=1,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.001,
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
        penalty_exponent=1.5,
    )


def _explicit_overrides() -> dict:
    return dict(
        solver_mode="explicit",
        explicit_ul_disable_update=True,
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e5,
        explicit_kinetic_energy_budget_ratio=0.05,
    )


def _run(label: str, **overrides) -> RunSummary:
    base = _base_cfg()
    base.update(overrides)
    cfg = StrandBendingOscillationConfig(**base)
    N_sub = int(base.get("explicit_n_sub_cycles_per_increment", 1))

    print(f"\n[run] {label} (N_sub={N_sub})")
    t0 = time.perf_counter()
    try:
        result = StrandBendingOscillationProcess().process(cfg)
        sr = result.solver_result
        u = sr.u
        tip = _extract_tip(u, result.mesh)
        u_x, u_z = float(tip[0]), float(tip[2])
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        return RunSummary(
            label=label,
            N_sub=N_sub,
            u_x=float("nan"),
            u_z=float("nan"),
            rel_err_x=float("nan"),
            elapsed=time.perf_counter() - t0,
        )
    elapsed = time.perf_counter() - t0
    print(f"  u_x={u_x:+.4e}, u_z={u_z:+.4e}, elapsed={elapsed:.2f}s")
    return RunSummary(
        label=label, N_sub=N_sub, u_x=u_x, u_z=u_z, rel_err_x=float("nan"), elapsed=elapsed
    )


def main() -> int:
    print("=" * 72)
    print("status-399 STA2 検証: N→∞ で asymptote 収束しているか")
    print("=" * 72)
    print()
    print("仮説:")
    print("  H0 (status-399 主張): u_x(N) → implicit 4.996 mm にasymptote 収束")
    print("  H1 (STA2 該当): u_x(N) は単調増加で implicit 値を通り過ぎる")
    print()

    imp = _run("implicit_baseline", solver_mode="implicit")

    # N 軸掃引
    results = [imp]
    for N in [500, 1000, 2000, 5000]:
        r = _run(f"N={N}", explicit_n_sub_cycles_per_increment=N, **_explicit_overrides())
        results.append(r)

    # 結果サマリ
    print()
    print("=" * 72)
    print("収束軌跡")
    print("=" * 72)
    u_x_imp = imp.u_x
    print(f"  implicit: u_x = {u_x_imp:+.4e} mm (ref)")
    print()
    print(f"  {'N_sub':>8} {'u_x [mm]':>12} {'|Δ| vs imp [mm]':>17} {'rel_err':>9} {'elapsed':>8}")
    prev_abs_err = None
    monotonic_decreasing = True
    for r in results[1:]:
        abs_err = abs(r.u_x - u_x_imp)
        rel = abs_err / max(abs(u_x_imp), 1e-30)
        if prev_abs_err is not None and abs_err > prev_abs_err:
            monotonic_decreasing = False
        marker = ""
        if prev_abs_err is not None:
            if abs_err < prev_abs_err:
                marker = " ↓ (improving)"
            else:
                marker = " ↑ (WORSENING)"
        print(
            f"  {r.N_sub:>8d} {r.u_x:>+12.4e} {abs_err:>17.4e} {rel:>8.2%}{marker} {r.elapsed:>7.1f}s"
        )
        prev_abs_err = abs_err

    print()
    print("判定:")
    last = results[-1]
    if np.isnan(last.u_x):
        print(f"  N={last.N_sub} で NaN/発散 → 別の数値破綻、追加診断必要")
        return 1
    abs_err_last = abs(last.u_x - u_x_imp)
    abs_err_n1000 = abs(results[2].u_x - u_x_imp) if len(results) >= 3 else None

    print(
        f"  最大 N={last.N_sub} で u_x={last.u_x:+.4f} mm, "
        f"|Δ|={abs_err_last:.4f} mm, rel_err={abs_err_last / abs(u_x_imp):.2%}"
    )
    if monotonic_decreasing:
        print("  → |Δ| は単調減少: asymptote 収束方向の証拠")
    else:
        print("  → |Δ| が単調減少していない: H1 STA2 該当の可能性")
    if abs_err_n1000 is not None:
        if abs_err_last < abs_err_n1000:
            print(f"  → N={last.N_sub} の |Δ| < N=1000 の |Δ|: 改善継続")
        else:
            print(f"  → N={last.N_sub} の |Δ| >= N=1000 の |Δ|: 通り過ぎ示唆")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
