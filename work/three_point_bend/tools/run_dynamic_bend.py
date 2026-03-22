"""動的接触三点曲げの実行スクリプト.

Usage:
    python work/three_point_bend/tools/run_dynamic_bend.py [--E 25] [--push 30] [--max-incr 500]

YAML インプットファイルは assets/ に自動保存される。
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
)

WORK_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="動的接触三点曲げ")
    parser.add_argument("--E", type=float, default=25.0, help="ヤング率 [MPa]")
    parser.add_argument("--push", type=float, default=30.0, help="押し込み量 [mm]")
    parser.add_argument("--max-incr", type=int, default=500, help="最大増分数")
    parser.add_argument("--k-pen", type=float, default=0.0, help="ペナルティ剛性 (0=自動)")
    parser.add_argument("--n-periods", type=float, default=30.0, help="周期数")
    parser.add_argument("--tag", type=str, default="", help="実行タグ（ファイル名用）")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)

    cfg = DynamicThreePointBendContactJigConfig(
        E=args.E,
        jig_push=args.push,
        max_increments=args.max_incr,
        k_pen=args.k_pen,
        n_periods=args.n_periods,
    )

    # 解析解
    I = np.pi * (cfg.wire_diameter / 2) ** 4 / 4
    k_eb = 48.0 * cfg.E * I / cfg.wire_length**3
    P_eb = k_eb * cfg.jig_push

    tag = args.tag or f"E{int(args.E)}"
    print(f"=== {tag}: E={cfg.E}, push={cfg.jig_push}, k_pen={cfg.k_pen}, max_incr={cfg.max_increments} ===")
    print(f"解析剛性(EB): k={k_eb:.2f} N/mm, P({cfg.jig_push}mm)={P_eb:.1f} N")

    t0 = time.perf_counter()
    result = DynamicThreePointBendContactJigProcess().process(cfg)
    elapsed = time.perf_counter() - t0

    sr = result.solver_result
    print(f"\n--- 結果 ---")
    print(f"収束={sr.converged}, incr={sr.n_increments}")
    print(f"計算時間={elapsed:.1f} s")
    print(f"たわみ={result.wire_midpoint_deflection:.4f} mm")
    print(f"接触力={result.contact_force_norm:.1f} N")
    print(f"実効剛性={result.effective_stiffness:.2f} N/mm")
    print(f"剛性誤差(EB)={result.stiffness_error_eb:.4f}")

    last_frac = sr.load_history[-1] if sr.load_history else 0.0
    push_reached = last_frac * cfg.jig_push
    print(f"最終frac={last_frac:.4f}, 到達push={push_reached:.2f} mm")

    # 接触力最大
    fc_hist = sr.contact_force_history
    if fc_hist:
        fc_max = max(np.linalg.norm(f) for f in fc_hist)
        print(f"接触力最大={fc_max:.1f} N")

    # 結果を results/ に保存
    results_dir = WORK_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    result_data = {
        "tag": tag,
        "converged": bool(sr.converged),
        "n_increments": int(sr.n_increments),
        "elapsed_seconds": round(elapsed, 1),
        "deflection_mm": round(result.wire_midpoint_deflection, 4),
        "contact_force_N": round(result.contact_force_norm, 1),
        "effective_stiffness": round(result.effective_stiffness, 2),
        "stiffness_error_eb": round(result.stiffness_error_eb, 4),
        "final_frac": round(last_frac, 4),
        "push_reached_mm": round(push_reached, 2),
    }
    result_file = results_dir / f"result_{tag}.yaml"
    with open(result_file, "w") as f:
        yaml.dump(result_data, f, default_flow_style=False, allow_unicode=True)
    print(f"\n結果保存: {result_file}")


if __name__ == "__main__":
    main()
