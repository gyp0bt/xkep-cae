"""work/beam_hysteresis/16_hypothesis_c_aprime_19strand.py — 仮説 C 候補 (a') 19本撚線検証.

[← README](README.md) | [← project README](../../README.md)

status-359 で 7 本撚線 90° 曲げにおいて `smoothing_delta=1000`（default 2000 の
1/2、δ_h 2x 拡大）が frac=1.0000 完走 + elapsed -42.5% の大幅改善を示したため
（採択方向）、本スクリプトは同設定を **19 本撚線（Type D stall 本体）**へ
スケールアップして検証する。MCDD 凍結解除条件「19 本 frac=1.0」達成可否の
判定材料となる。

**ベースライン（status-339 実測、10_kcr_measurement_19strand.py）**:

- smoothing_delta 自動 = 1000/r = 2000（default）
- frac=0.4839, incr=271, cb=39, 534.68s
- Type D stall で frac=0.484 付近で未完走

**本スクリプト（候補 (a'）、status-360 計測結果）**:

- smoothing_delta=1000（手動指定、2x 拡大）→ δ_h = k_pen / 1000 = 2x 広い
- **frac=0.3723, incr=164, cb=23, 365.29s（Type D.stall+E:72% で frac=0.37 早期停滞）**
- 比較: baseline 対比 **frac -23.1%（退化）**、incr -39%、cb -41%、elapsed -32% の
  見かけ短縮は **解析の早期打切り**（candidate (a) / status-358 と同パターン）
- 判定: **却下**。7 本で有効な 2x δ_h 拡大は 19 本 Type D stall 領域で逆効果。
  **`StrandBendingOscillationConfig.smoothing_delta` の default 変更は
  status-360 で実施せず**（7 本のみ最適値として記録、実装本体無変更）。
- 次候補: **(c) line search 強化**（NR 反復途中の過剰 active flip を
  backtracking line search で rejection、`_newton_dynamic.py` 拡張）。
  本スクリプトは **19 本撚線失敗実験の記録**として残置（status-359 の
  7 本成功実験 `15_hypothesis_c_7strand.py` と対称）。

実行:
    uv run python work/beam_hysteresis/16_hypothesis_c_aprime_19strand.py 2>&1 \\
        | tee /tmp/hypothesis_c_aprime_19strand_$(date +%s).log
"""

from __future__ import annotations

import sys
import time

import numpy as np

from xkep_cae.mesh.process import StrandMeshConfig, StrandMeshProcess
from xkep_cae.numerical_tests.contact_pair_analysis import (
    ContactPairAnalysisInput,
    ContactPairAnalysisProcess,
)
from xkep_cae.numerical_tests.contact_pair_layer_classifier import (
    ContactPairLayerClassifierInput,
    ContactPairLayerClassifierProcess,
)
from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def main() -> int:
    """19本撚線曲げ + 接触ペア解析（仮説 C (a') smoothing_delta=1000 適用）."""
    print("=" * 70)
    print("仮説 C 候補 (a'): smoothing_delta=1000 (2x 拡大、19本撚線 90°曲げ)")
    print("=" * 70)

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
        # ★ 仮説 C 候補 (a'): smoothing_delta = 1000（default 2000 の 1/2、δ_h 2x 拡大）
        smoothing_delta=1000.0,
        track_contact_mk=True,
        track_contact_pairs=True,
    )
    print(
        f"cfg: n_strands={cfg.n_strands}, n_pitches={cfg.n_pitches}, "
        f"κ_max={cfg.bending_curvature}, mu={cfg.mu}, "
        f"smoothing_delta={cfg.smoothing_delta} (baseline=2000), "
        f"n_incr_per_cycle={cfg.n_increments_per_cycle}"
    )
    print()

    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result

    frac = sr.load_history[-1] if sr.load_history else 0.0
    print()
    print("=" * 70)
    print("ソルバー結果（仮説 C (a') 19本撚線）")
    print("=" * 70)
    print(f"  frac_completed: {frac:.4f}")
    print(f"  converged:      {sr.converged}")
    print(f"  n_increments:   {sr.n_increments}")
    print(f"  n_cutbacks:     {sr.n_cutbacks}")
    print(f"  elapsed:        {elapsed:.2f} s")
    print(f"  max |u|:        {float(np.max(np.abs(sr.u))):.6e}")
    mk_len = len(sr.moment_curvature_history)
    pair_len = len(sr.contact_pair_history)
    print(f"  mk_history:     {mk_len} entries")
    print(f"  pair_history:   {pair_len} entries")

    # ── ベースライン比較（status-339） ──
    print()
    print("=" * 70)
    print("status-339 比較（default smoothing_delta=2000 ベースライン vs 候補 (a'))")
    print("=" * 70)
    print(f"  {'項目':<20s} {'baseline=2000':>18s} {'candidate=1000':>18s}")
    print(f"  {'frac_completed':<20s} {'0.4839':>18s} {frac:>18.4f}")
    print(f"  {'n_increments':<20s} {'271':>18s} {sr.n_increments:>18d}")
    print(f"  {'n_cutbacks':<20s} {'39':>18s} {sr.n_cutbacks:>18d}")
    print(f"  {'elapsed [s]':<20s} {'534.68':>18s} {elapsed:>18.2f}")
    base_frac = 0.4839
    frac_delta = (frac - base_frac) / base_frac * 100.0
    print(f"  frac 改善: {frac_delta:+.1f}%  (frac=1.0 完走で MCDD 凍結解除条件達成)")

    # ── ContactPairAnalysisProcess 実行 ──
    analysis = ContactPairAnalysisProcess().process(
        ContactPairAnalysisInput(
            contact_pair_history=sr.contact_pair_history,
            moment_curvature_history=sr.moment_curvature_history,
        )
    )

    print()
    print("=" * 70)
    print("ContactPairAnalysisProcess 結果")
    print("=" * 70)
    print(f"  n_steps:            {analysis.n_steps}")
    print(f"  n_unique_pairs:     {analysis.n_unique_pairs}")
    print(f"  n_slipped_pairs:    {analysis.n_slipped_pairs}")
    print(f"  total_dissipation:  {analysis.total_dissipation:.6e}")

    if analysis.n_slipped_pairs > 0:
        print(f"  κ_cr mean:          {analysis.kappa_cr_mean:.6e}")
        print(f"  κ_cr std:           {analysis.kappa_cr_std:.6e}")
        print(f"  κ_cr min:           {analysis.kappa_cr_min:.6e}")
        print(f"  κ_cr max:           {analysis.kappa_cr_max:.6e}")
        cv = analysis.kappa_cr_std / analysis.kappa_cr_mean if analysis.kappa_cr_mean > 0 else 0.0
        print(f"  κ_cr CV (std/mean): {cv:.4f}")

    # κ_cr ヒストグラム
    if analysis.n_slipped_pairs >= 5:
        kcrs = np.array(list(analysis.kappa_cr_per_pair.values()))
        kmin, kmax = kcrs.min(), kcrs.max()
        if kmax > kmin:
            n_bins = 15
            hist, edges = np.histogram(kcrs, bins=n_bins, range=(kmin, kmax))
            print()
            print(f"  κ_cr ヒストグラム（{n_bins} bin）:")
            for i in range(n_bins):
                bar = "#" * int(hist[i])
                print(f"    [{edges[i]:.3e}, {edges[i + 1]:.3e}): {hist[i]:3d} {bar}")

    # ── ContactPairLayerClassifierProcess 実行（層別 κ_cr 統計）──
    mesh_result = StrandMeshProcess().process(
        StrandMeshConfig(
            n_strands=cfg.n_strands,
            wire_radius=cfg.wire_radius,
            pitch_length=cfg.pitch_length,
            gap=cfg.gap,
            n_elements_per_pitch=cfg.n_elements_per_pitch,
            n_pitches=cfg.n_pitches,
        )
    )
    classifier = ContactPairLayerClassifierProcess().process(
        ContactPairLayerClassifierInput(
            kappa_cr_per_pair=analysis.kappa_cr_per_pair,
            per_pair_dissipation=analysis.per_pair_dissipation,
            strand_ids=mesh_result.mesh.strand_ids.tolist(),
            strand_layers=mesh_result.strand_layers,
        )
    )

    print()
    print("=" * 70)
    print("ContactPairLayerClassifier 結果（層別 κ_cr）")
    print("=" * 70)
    print(f"  n_unique_layer_pairs: {classifier.n_unique_layer_pairs}")
    print()
    print(
        f"  {'(l_min,l_max)':<14s} {'n_pairs':>8s} {'n_slip':>7s} "
        f"{'κ_cr mean':>11s} {'κ_cr std':>11s} {'κ_cr CV':>8s} {'diss_sum':>11s}"
    )
    for lp in sorted(classifier.per_layer_pair_stats.keys()):
        s = classifier.per_layer_pair_stats[lp]
        cv = s.kappa_cr_std / s.kappa_cr_mean if s.kappa_cr_mean > 0 else 0.0
        print(
            f"  {str(lp):<14s} {s.n_pairs:>8d} {s.n_slipped:>7d} "
            f"{s.kappa_cr_mean:>11.3e} {s.kappa_cr_std:>11.3e} "
            f"{cv:>8.3f} {s.dissipation_sum:>11.3e}"
        )

    print()
    print("=" * 70)
    print("完了（仮説 C (a') 19本撚線検証）")
    print("=" * 70)
    return 0 if sr.converged else 1


if __name__ == "__main__":
    sys.exit(main())
