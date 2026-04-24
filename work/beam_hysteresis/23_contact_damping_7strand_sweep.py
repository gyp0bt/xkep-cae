"""work/beam_hysteresis/23_contact_damping_7strand_sweep.py — 候補 (e) 接触減衰 7本撚線 c_n 掃引.

[← README](README.md) | [← project README](../../README.md)

status-366 Phase 2 で配線完了した `ContactNormalDampingProcess` に対する
validation スクリプト。7本撚線 90° 曲げで c_n を段階的に変化させ、
完走性能（frac / incr / cutback / elapsed）と減衰エネルギー消費
（E_damp_cumulative / E_strain）を測定して budget 許容線を確立する。

**ベースライン（status-359 実測）**:

- `smoothing_delta=1000`（2x 拡大）、c_n=0（damping OFF）
- frac=1.0000, incr=475, cutback=53, 259.92s

**本掃引の c_n 選定**:

- k_pen（動的）≈ c0·M_ii（GeneralizedAlpha ベース）、c1=γ/(β·dt)
- c_n 候補: 0（baseline）/ 10 / 100 / 1000 / 10000 を実測
- E_damp budget 目標: 5% / 10% / 20% を許容線として判定

**判定基準**:

1. frac=1.0 完走性維持（必須）
2. elapsed 10% 以上の悪化なし（active flip 抑制で改善期待）
3. E_damp_cum / E_strain_final < 0.20（散逸過剰でない）

実行:
    uv run python work/beam_hysteresis/23_contact_damping_7strand_sweep.py 2>&1 \\
        | tee /tmp/contact_damping_7strand_$(date +%s).log
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from xkep_cae.contact.damping import (
    ContactDampingEnergyMonitorInput,
    ContactDampingEnergyMonitorProcess,
)
from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


@dataclass
class SweepResult:
    """1 ケースの結果サマリ."""

    c_n: float
    frac: float
    converged: bool
    n_incr: int
    n_cutbacks: int
    elapsed: float
    e_damp_final: float
    e_strain_final: float
    ratio_max: float
    ratio_final: float


def _make_cfg(c_n: float) -> StrandBendingOscillationConfig:
    """7本撚線 90° 曲げ base config（status-359 設定、damping 追加のみ変更）."""
    return StrandBendingOscillationConfig(
        n_strands=7,
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
        # ★ 本掃引のパラメータ
        contact_damping_coefficient=c_n,
        contact_damping_energy_budget_ratio=0.20,  # 20% budget（超過時は report で ! 表示）
    )


def _run_one(c_n: float) -> SweepResult:
    print()
    print("=" * 72)
    print(f"[CASE c_n={c_n:.3e}] 7本撚線 90° 曲げ")
    print("=" * 72)
    cfg = _make_cfg(c_n)
    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0

    # 減衰エネルギー監査
    damp_hist = sr.damping_energy_history
    e_damp_final = damp_hist[-1][1] if damp_hist else 0.0
    # EnergyHistory.entries[-1].strain_energy
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
    print(mon_out.report)

    print(
        f"  RESULT: frac={frac:.4f} converged={sr.converged} "
        f"incr={sr.n_increments} cb={sr.n_cutbacks} elapsed={elapsed:.2f}s "
        f"E_damp={e_damp_final:.3e} E_strain={e_strain_final:.3e} "
        f"max_ratio={mon_out.max_ratio:.3e}"
    )
    return SweepResult(
        c_n=c_n,
        frac=frac,
        converged=bool(sr.converged),
        n_incr=int(sr.n_increments),
        n_cutbacks=int(sr.n_cutbacks),
        elapsed=elapsed,
        e_damp_final=e_damp_final,
        e_strain_final=e_strain_final,
        ratio_max=mon_out.max_ratio,
        ratio_final=mon_out.final_ratio,
    )


def main() -> int:
    """7本撚線 c_n ∈ {0, 10, 100, 1000, 10000} 掃引."""
    print("=" * 72)
    print("候補 (e) 接触減衰 — 7本撚線 c_n 掃引（status-367 validation）")
    print("=" * 72)
    print("ベースライン（status-359、c_n=0）: frac=1.0000, incr=475, cb=53, 259.92s")

    c_n_values = [0.0, 10.0, 100.0, 1000.0, 10000.0]
    results: list[SweepResult] = []
    for c_n in c_n_values:
        results.append(_run_one(c_n))

    print()
    print("=" * 72)
    print("掃引サマリ")
    print("=" * 72)
    print(
        f"{'c_n':>10} {'frac':>8} {'incr':>6} {'cb':>5} {'elapsed':>10} "
        f"{'E_damp':>12} {'E_strain':>12} {'max_ratio':>10}"
    )
    for r in results:
        print(
            f"{r.c_n:>10.1e} {r.frac:>8.4f} {r.n_incr:>6d} {r.n_cutbacks:>5d} "
            f"{r.elapsed:>9.2f}s {r.e_damp_final:>12.3e} {r.e_strain_final:>12.3e} "
            f"{r.ratio_max:>10.3e}"
        )

    # 判定: frac=1.0 完走 + elapsed 10% 悪化以内 + budget 内最大 c_n
    print()
    print("=" * 72)
    print("判定: frac=1.0 完走 + elapsed ≤ 110% baseline + E_damp/E_strain < budget")
    print("=" * 72)
    base = next((r for r in results if r.c_n == 0.0), None)
    if base is None:
        print("  ERROR: baseline (c_n=0) が取得できず")
        return 1
    for r in results:
        completed = r.frac >= 0.9999
        elapsed_delta = (r.elapsed - base.elapsed) / base.elapsed * 100.0
        budget_ok_05 = r.ratio_max < 0.05
        budget_ok_10 = r.ratio_max < 0.10
        budget_ok_20 = r.ratio_max < 0.20
        tag = "OK" if (completed and elapsed_delta <= 10.0 and budget_ok_20) else "--"
        print(
            f"  [{tag}] c_n={r.c_n:.1e}: frac={r.frac:.4f} "
            f"Δelapsed={elapsed_delta:+.1f}% "
            f"budget<5%:{'Y' if budget_ok_05 else 'N'} "
            f"<10%:{'Y' if budget_ok_10 else 'N'} "
            f"<20%:{'Y' if budget_ok_20 else 'N'}"
        )

    # 採用候補: frac=1.0 + budget<10% + 最大 c_n
    good = [r for r in results if r.frac >= 0.9999 and r.ratio_max < 0.10]
    if good:
        best = max(good, key=lambda r: r.c_n)
        print()
        print(
            f"→ 19 本適用候補: c_n = {best.c_n:.1e} "
            f"(max_ratio={best.ratio_max:.3e} < 0.10, frac=1.0)"
        )
    else:
        print()
        print("→ budget<10% を満たす完走ケースなし。19本適用は budget<20% 帯で再検討")

    return 0


if __name__ == "__main__":
    sys.exit(main())
