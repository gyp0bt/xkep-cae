"""ベースライン検証: 動的三点曲げ frac>0.9 (ε=1e-6).

CLAUDE.md フォーカスガード条件:
  L=100, φ17, push=30, n_periods=30, E=25 MPa

status-228 実績: frac=0.9594, fc=189.5N

使い方:
  python contracts/baseline_dynamic_3pb.py 2>&1 | tee /tmp/log-baseline-3pb.log
"""

from __future__ import annotations

import time

from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
)


def main() -> None:
    cfg = DynamicThreePointBendContactJigConfig(
        wire_length=100.0,
        wire_diameter=17.0,
        n_elems_wire=20,
        E=25.0,  # MPa（軟質、フォーカスガード条件）
        nu=0.3,
        jig_push=30.0,
        n_periods=30.0,
        max_increments=500,
    )

    print("=" * 70)
    print("ベースライン検証: 動的三点曲げ (ε=1e-6)")
    print("=" * 70)
    print(f"  L={cfg.wire_length}, φ={cfg.wire_diameter}, E={cfg.E} MPa")
    print(f"  push={cfg.jig_push}, n_periods={cfg.n_periods}")
    print(f"  n_elems={cfg.n_elems_wire}, max_incr={cfg.max_increments}")
    print(f"  use_hermite={cfg.use_hermite_centerline}")
    print()

    proc = DynamicThreePointBendContactJigProcess()
    t0 = time.perf_counter()
    result = proc.process(cfg)
    elapsed = time.perf_counter() - t0

    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0
    fc = result.contact_force_norm

    print()
    print("=" * 70)
    print("結果")
    print("=" * 70)
    print(f"  frac          = {frac:.4f}")
    print(f"  接触力        = {fc:.1f} N")
    print(f"  中央変位      = {result.wire_midpoint_deflection:.4f} mm")
    print(f"  インクリメント = {sr.n_increments}")
    print(f"  カットバック   = {sr.n_cutbacks}")
    print(f"  NR 総反復     = {sr.total_attempts}")
    print(f"  収束          = {sr.converged}")
    print(f"  実行時間      = {elapsed:.1f} s")
    print()

    # 判定
    ok = True
    if frac <= 0.9:
        print(f"[FAIL] frac={frac:.4f} <= 0.9 — ベースライン未達")
        ok = False
    else:
        print(f"[PASS] frac={frac:.4f} > 0.9")

    if fc <= 100.0:
        print(f"[FAIL] fc={fc:.1f}N <= 100N — 荷重レベル不足")
        ok = False
    else:
        print(f"[PASS] fc={fc:.1f}N > 100N")

    print()
    if ok:
        print(">>> ベースライン PASS <<<")
    else:
        print(">>> ベースライン FAIL <<<")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
