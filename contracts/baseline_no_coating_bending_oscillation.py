"""被膜なし7本撚線 曲げ揺動 ベースライン計測.

status-313: 高速化(status-308〜312)後のベースライン確認。
比較対象: status-301 被膜なし incr=1900, cutback=72, 1527s。

構成:
  - 7本撚線, R=0.5mm, pitch=100mm, 16要素/ピッチ, 1ピッチ
  - 90度曲げ (κ=π/200) + ±48mm揺動 2サイクル
  - Hertz型(α=1.5), free_end_mode, 接触あり, 摩擦あり(μ=0.15)
  - 被膜なし (coating_stiffness=0, coating_thickness=0)

実行方法:
  python contracts/baseline_no_coating_bending_oscillation.py 2>&1 | tee /tmp/log-baseline-$(date +%s).log

[← README](../README.md)
"""

from __future__ import annotations

import math
import time

import numpy as np

from xkep_cae.core.base import ProcessMetaclass
from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def main() -> None:
    """被膜なし7本撚線 曲げ+揺動 ベースライン計測."""
    print("=" * 70)
    print("被膜なし7本撚線 曲げ揺動 ベースライン")
    print("  status-301比較: incr=1900, cutback=72, 1527s")
    print("=" * 70)

    cfg = StrandBendingOscillationConfig(
        n_strands=7,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=math.pi / 200.0,  # 90度曲げ: κ=π/(2*L)
        n_cycles=1,
        n_increments_per_cycle=40,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=True,
        contact_enabled=True,
        loading_mode="rotation",
        penalty_exponent=1.5,  # Hertz型
        # 揺動: ±48mm, 2サイクル
        n_oscillation_cycles=2,
        oscillation_amplitude=48.0,
        # 被膜なし（デフォルト値、明示）
        coating_stiffness=0.0,
        coating_thickness=0.0,
        coating_damping=0.0,
        coating_mu=0.0,
    )

    print("\n構成:")
    print(f"  n_strands:              {cfg.n_strands}")
    print(f"  wire_radius:            {cfg.wire_radius} mm")
    print(f"  pitch_length:           {cfg.pitch_length} mm")
    print(f"  n_elements_per_pitch:   {cfg.n_elements_per_pitch}")
    print(f"  bending_curvature:      {cfg.bending_curvature:.6f} 1/mm")
    print(
        f"  bending_angle:          {math.degrees(cfg.bending_curvature * cfg.pitch_length * cfg.n_pitches):.1f}°"
    )
    print(f"  penalty_exponent:       {cfg.penalty_exponent} (Hertz)")
    print(f"  free_end_mode:          {cfg.free_end_mode}")
    print(f"  n_oscillation_cycles:   {cfg.n_oscillation_cycles}")
    print(f"  oscillation_amplitude:  {cfg.oscillation_amplitude} mm")
    print(f"  coating_stiffness:      {cfg.coating_stiffness} (被膜なし)")
    print(f"  coating_thickness:      {cfg.coating_thickness} (被膜なし)")

    # プロファイルデータクリア
    ProcessMetaclass._profile_data.clear()

    print("\n解析開始...")
    t_start = time.perf_counter()
    proc = StrandBendingOscillationProcess()
    result = proc.process(cfg)
    t_elapsed_wall = time.perf_counter() - t_start

    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0

    print("\n" + "=" * 70)
    print("結果サマリ")
    print("=" * 70)
    print(f"  frac_completed:   {frac:.4f}")
    print(f"  converged:        {sr.converged}")
    print(f"  n_increments:     {sr.n_increments}")
    print(f"  n_cutbacks:       {sr.n_cutbacks}")
    print(f"  elapsed(solver):  {sr.elapsed_seconds:.2f} s")
    print(f"  elapsed(wall):    {t_elapsed_wall:.2f} s")
    print(f"  total_ndof:       {result.total_ndof}")
    print(f"  n_strand_nodes:   {result.n_strand_nodes}")
    print(
        f"  bending_angle:    {result.bending_angle:.6f} rad ({math.degrees(result.bending_angle):.1f}°)"
    )

    u_max = float(np.max(np.abs(sr.u)))
    print(f"  max |u|:          {u_max:.6e}")

    if sr.contact_force_history:
        print(f"  contact forces:   {len(sr.contact_force_history)} entries")
        print(f"  max contact F:    {max(sr.contact_force_history):.6e}")

    # status-301比較
    print("\n" + "-" * 70)
    print("status-301比較（高速化前ベースライン）")
    print("-" * 70)
    prev_incr = 1900
    prev_cb = 72
    prev_elapsed = 1527.0
    print(
        f"  increments:  {prev_incr} → {sr.n_increments}  ({(sr.n_increments - prev_incr) / prev_incr * 100:+.1f}%)"
    )
    print(
        f"  cutbacks:    {prev_cb} → {sr.n_cutbacks}  ({(sr.n_cutbacks - prev_cb) / prev_cb * 100:+.1f}%)"
    )
    print(
        f"  elapsed:     {prev_elapsed:.0f}s → {sr.elapsed_seconds:.1f}s  ({(sr.elapsed_seconds - prev_elapsed) / prev_elapsed * 100:+.1f}%)"
    )

    # プロセス別プロファイル（上位15件）
    print("\n" + "-" * 70)
    print("プロセス別プロファイル（上位15件）")
    print("-" * 70)
    profile_summary: list[tuple[str, int, float]] = []
    for name, times in ProcessMetaclass._profile_data.items():
        profile_summary.append((name, len(times), sum(times)))
    profile_summary.sort(key=lambda x: x[2], reverse=True)
    total_profile = sum(t for _, _, t in profile_summary)
    print(f"{'プロセス':<45} {'呼出数':>8} {'合計[s]':>10} {'割合':>8}")
    for name, count, total in profile_summary[:15]:
        pct = total / total_profile * 100 if total_profile > 0 else 0.0
        print(f"  {name:<43} {count:>8,} {total:>10.2f} {pct:>7.1f}%")

    # アサーション
    assert frac >= 0.95, f"frac={frac} < 0.95 — 完走できていない"
    assert u_max > 0.0, "変位がゼロ"

    print("\n✓ ベースライン計測完了")
    return


if __name__ == "__main__":
    main()
