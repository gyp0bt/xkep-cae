"""三点曲げ接触収束の軽量診断スクリプト.

pytest ではなく直接実行して収束状況を確認する:
  python contracts/diagnose_three_point_bend.py 2>&1 | tee /tmp/log-diag-$(date +%s).log

[← README](../README.md)
"""

from __future__ import annotations

import math
import time

from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
    _beam_fundamental_frequency,
    _circle_section,
)


def _convergence_rate_label(res_history: list[float]) -> str:
    """残差履歴から収束率を判定."""
    if len(res_history) < 3:
        return "insufficient_data"
    # 最後の3点から収束次数を推定
    r = res_history
    ratios = []
    for i in range(1, len(r)):
        if r[i - 1] > 1e-30:
            ratios.append(r[i] / r[i - 1])
    if not ratios:
        return "stalled"
    avg_ratio = sum(ratios[-3:]) / min(len(ratios), 3)
    if avg_ratio < 0.01:
        return "quadratic"
    if avg_ratio < 0.3:
        return "superlinear"
    if avg_ratio < 0.9:
        return "linear"
    return "stalled"


def main() -> None:
    """軽量診断を実行."""
    print("=" * 70)
    print("  三点曲げ接触ジグ — 収束診断")
    print("=" * 70)

    # 軽量構成: 粗メッシュ、短時間、小変位
    cfg = DynamicThreePointBendContactJigConfig(
        n_elems_wire=16,
        n_periods=1.0,
        jig_push=0.05,
        initial_gap=0.0,
        mu=0.15,
        max_increments=200,
    )

    sec = _circle_section(cfg.wire_diameter, cfg.nu)
    f1 = _beam_fundamental_frequency(cfg.wire_length, cfg.E, sec["Iy"], cfg.rho, sec["A"])
    T1 = 1.0 / f1
    t_total = cfg.n_periods * T1
    k_eb = 48.0 * cfg.E * sec["Iy"] / cfg.wire_length**3

    print("\n  構成:")
    print(f"    n_elems={cfg.n_elems_wire}, jig_push={cfg.jig_push} mm")
    print(f"    n_periods={cfg.n_periods}, T1={T1:.6f} s, t_total={t_total:.6f} s")
    print(f"    k_pen={cfg.k_pen} (0=auto), smoothing_delta={cfg.smoothing_delta}")
    print(f"    n_uzawa_max={cfg.n_uzawa_max}, mu={cfg.mu}")
    print(f"    k_EB={k_eb:.4f} N/mm")
    print(f"    max_increments={cfg.max_increments}")

    # 実行
    print(f"\n{'=' * 70}")
    print("  ソルバー実行開始")
    print("=" * 70)

    t0 = time.perf_counter()
    proc = DynamicThreePointBendContactJigProcess()
    result = proc.process(cfg)
    elapsed = time.perf_counter() - t0

    sr = result.solver_result

    print(f"\n{'=' * 70}")
    print("  結果サマリ")
    print("=" * 70)
    print(f"  converged: {sr.converged}")
    print(f"  increments: {sr.n_increments}")
    print(f"  total NR attempts: {sr.total_attempts}")
    print(f"  cutbacks: {sr.n_cutbacks}")
    print(f"  elapsed: {elapsed:.2f} s")
    print(f"  wire midpoint deflection: {result.wire_midpoint_deflection:.6f} mm")
    print(f"  contact force norm: {result.contact_force_norm:.6f} N")

    # 最終ステップ診断
    diag = sr.diagnostics
    if diag is not None:
        print(f"\n  最終ステップ診断 (step={diag.step}, load_frac={diag.load_frac:.6f}):")
        rh = diag.res_history
        if rh:
            arrows = " → ".join(f"{r:.2e}" for r in rh)
            rate = _convergence_rate_label(rh)
            print(f"    NR残差: {arrows}")
            print(f"    収束率: {rate} ({len(rh)} iter)")
        if diag.n_active_history:
            print(f"    active pairs: {diag.n_active_history[-1]}")
        if diag.condition_number is not None:
            print(f"    condition number: {diag.condition_number:.2e}")

        # ペア別スナップショット
        if diag.pair_snapshots:
            last_snap = diag.pair_snapshots[-1]
            active = [p for p in last_snap if p.status != "inactive"]
            if active:
                print(f"    接触ペア (active={len(active)}):")
                for p in active[:5]:
                    print(f"      pair {p.pair_id}: gap={p.gap:.4e} p_n={p.p_n:.4e} [{p.status}]")

    # エネルギー診断
    if sr.energy_history is not None:
        eh = sr.energy_history
        if hasattr(eh, "kinetic") and hasattr(eh, "strain"):
            ke_list = eh.kinetic
            se_list = eh.strain
            if ke_list and se_list:
                print("\n  エネルギー診断 (最終ステップ):")
                print(f"    KE={ke_list[-1]:.6e}, SE={se_list[-1]:.6e}")

    # 収束率の定量評価（最終ステップ）
    if diag is not None and diag.res_history:
        rh = diag.res_history
        print("\n  収束率定量分析（最終ステップ）:")
        for i in range(1, len(rh)):
            if rh[i - 1] > 1e-30:
                ratio = rh[i] / rh[i - 1]
                log_ratio = math.log10(ratio) if ratio > 1e-30 else float("-inf")
                print(
                    f"    iter {i}: {rh[i]:.2e} / {rh[i - 1]:.2e} = {ratio:.4f} (log10={log_ratio:.2f})"
                )

    # ── Per-increment 診断（全ステップの NR 残差減衰率）──
    if sr.diagnostics_history:
        print(f"\n{'=' * 70}")
        print("  Per-increment NR 収束診断")
        print("=" * 70)
        contact_increments = 0
        total_nr = 0
        rate_sum = 0.0
        rate_count = 0
        for d in sr.diagnostics_history:
            rh = d.res_history
            n_active = d.n_active_history[-1] if d.n_active_history else 0
            if n_active == 0 and len(rh) <= 2:
                continue
            contact_increments += 1
            total_nr += len(rh)
            rate = _convergence_rate_label(rh)
            # 最後の減衰率
            last_ratio = ""
            if len(rh) >= 2 and rh[-2] > 1e-30:
                lr = rh[-1] / rh[-2]
                last_ratio = f"  last_rate={lr:.4f}"
                rate_sum += lr
                rate_count += 1
            print(
                f"  step {d.step:3d} frac={d.load_frac:.4f} "
                f"NR={len(rh):2d} active={n_active} "
                f"res={rh[-1]:.2e} [{rate}]{last_ratio}"
            )
        if rate_count > 0:
            print(f"\n  平均最終減衰率: {rate_sum / rate_count:.4f}")
        print(f"  接触あり増分: {contact_increments}, 総NR反復: {total_nr}")

    print(f"\n{'=' * 70}")
    status = "OK" if sr.converged else "FAILED"
    print(f"  診断完了: {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
