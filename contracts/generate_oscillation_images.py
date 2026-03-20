"""梁揺動解析の画像を tmp/oscillation/ に出力するスクリプト.

大振幅（5mm）で100要素、3周期を計算し、
S11（応力）、LE11（ひずみ）、SK1（曲率）の3Dコンター画像を生成する。

使い方:
    python contracts/generate_oscillation_images.py 2>&1 | tee /tmp/log-$(date +%s).log
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from xkep_cae.numerical_tests.beam_oscillation import (
    BeamOscillationConfig,
    BeamOscillationProcess,
)
from xkep_cae.output.stress_contour import (
    ContourFieldInput,
    StressContour3DConfig,
    StressContour3DProcess,
)


def main():
    t0 = time.time()

    # 大振幅揺動解析（銅: E=100e3, ν=0.3, ρ=8.96e-9）
    cfg = BeamOscillationConfig(
        wire_length=100.0,
        wire_diameter=2.0,
        n_elems_wire=100,
        amplitude=5.0,
        n_periods=3.0,
    )
    print(f"=== 梁揺動解析開始 (E={cfg.E}, ρ={cfg.rho}, amp={cfg.amplitude}mm) ===")

    proc = BeamOscillationProcess()
    result = proc.process(cfg)

    t_solve = time.time() - t0
    print(f"ソルバー完了: {t_solve:.1f}s")
    print(f"  収束: {result.solver_result.converged}")
    print(f"  インクリメント数: {result.solver_result.n_increments}")
    print(f"  最大変位: {result.max_deflection:.4f} mm")
    print(f"  固有振動数: {result.analytical_frequency_hz:.1f} Hz")

    # コンターフィールド確認
    for name, snaps in result.contour_fields.items():
        import numpy as np

        if snaps:
            max_val = max(float(np.max(s)) for s in snaps if len(s) > 0)
            print(f"  {name} max: {max_val:.4e}")

    # 画像レンダリング
    print("\n=== コンターレンダリング (S11, LE11, SK1) ===")
    contour_fields = [
        ContourFieldInput(name=name, snapshots=snaps)
        for name, snaps in result.contour_fields.items()
    ]
    render_cfg = StressContour3DConfig(
        mesh=result.mesh,
        node_coords_initial=result.mesh.node_coords,
        displacement_snapshots=result.solver_result.displacement_history,
        contour_fields=contour_fields,
        time_values=result.time_history,
        wire_radius=cfg.wire_diameter / 2.0,
        output_dir="tmp/oscillation",
        prefix="beam_osc",
        n_render_frames=8,
    )
    render_proc = StressContour3DProcess()
    render_result = render_proc.process(render_cfg)

    t_total = time.time() - t0
    print(f"\n出力画像: {len(render_result.image_paths)} 枚")
    for p in render_result.image_paths:
        print(f"  {p}")
    print(f"\nフィールド最大値:")
    for name, val in render_result.field_max_values.items():
        print(f"  {name}: {val:.4e}")
    print(f"\n合計時間: {t_total:.1f}s")


if __name__ == "__main__":
    main()
