"""work/visualization/01_strand_bending_vtk_demo.py — VtkExportProcess の実機デモ.

[← README](../../README.md) | [← status-400](../../docs/status/status-400.md)

`StrandBendingOscillationProcess` で n_strands 本撚線の 90° 曲げを implicit で
解いた後、`VtkExportProcess` で `.pvd` + `.vtu` 時系列を出力する。

ParaView で `output_dir/strand_<n>.pvd` を開くと、各 timestep が
`load_history` の値（0.0 → 1.0）で時系列再生される。

## 実行

```bash
# 既定: 3 本撚線（軽量、~30s 想定）
python work/visualization/01_strand_bending_vtk_demo.py 2>&1 | tee /tmp/vtk_demo.log

# 7 本撚線（status-301 baseline、数分想定）
python work/visualization/01_strand_bending_vtk_demo.py --n-strands 7

# 1 本ε-1（最軽量、~1s）
python work/visualization/01_strand_bending_vtk_demo.py --n-strands 1
```

## 出力

- `docs/verification/strand_bending_vtk/strand_<n>.pvd` — ParaView エントリポイント
- `docs/verification/strand_bending_vtk/strand_<n>_*.vtu` — 各 increment の VTK
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)
from xkep_cae.output import VtkExportConfig, VtkExportProcess


def _build_config(n_strands: int) -> StrandBendingOscillationConfig:
    """規模に応じた config 生成. 7 本は status-301 baseline 同等."""
    base = dict(
        n_strands=n_strands,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.015,  # 90° 曲げ
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
        smoothing_delta=1000.0,  # status-359 採択（7 本系で frac=1.0 安定）
        track_contact_pairs=True,  # contact_force フィールドを ParaView に流すため必須
    )
    if n_strands == 1:
        # ε-1 sub-experiment: 接触なし straight beam
        base["contact_enabled"] = False
        base["bending_curvature"] = 0.001  # 小さめ
    return StrandBendingOscillationConfig(**base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-strands",
        type=int,
        default=3,
        help="撚線本数 (1/3/7/19)。1 は ε-1 接触なし straight、それ以外は helical 90° bend",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/verification/strand_bending_vtk"),
        help="VTK 出力ディレクトリ",
    )
    parser.add_argument(
        "--tube-segments",
        type=int,
        default=8,
        help="円筒パイプメッシュの角度分割数 (0 で line のみ、8 推奨)",
    )
    args = parser.parse_args()

    print(f"[demo] n_strands={args.n_strands}")
    print(f"[demo] output_dir={args.output_dir}")

    cfg = _build_config(args.n_strands)

    t0 = time.perf_counter()
    print("[demo] 解析開始...")
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result
    frac = float(sr.load_history[-1]) if sr.load_history else 0.0
    n_history = len(sr.displacement_history)
    print(
        f"[demo] 解析完了 — frac={frac:.4f}, incr={sr.n_increments}, "
        f"cb={sr.n_cutbacks}, history_len={n_history}, elapsed={elapsed:.2f}s"
    )

    u_trans = sr.u.reshape(-1, 6)[:, :3]
    print(f"[demo] max|u_trans| = {np.max(np.linalg.norm(u_trans, axis=1)):.4e} mm")

    # VTK 出力
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vtk_cfg = VtkExportConfig(
        solver_result=sr,
        mesh=result.mesh,
        output_dir=str(args.output_dir),
        prefix=f"strand_{args.n_strands}",
        young_modulus=130.0e3,
        tube_n_segments=args.tube_segments,
    )
    print("[demo] VTK 書き出し...")
    out = VtkExportProcess().process(vtk_cfg)
    print(
        f"[demo] VTK 完了 — n_timesteps={out.n_timesteps}, "
        f"n_points={out.n_points}, n_cells={out.n_cells}"
    )
    if out.pipe_pvd_path:
        print(
            f"[demo] pipe mesh — n_points={out.pipe_n_points}, n_cells={out.pipe_n_cells}"
        )
    print()
    if out.pipe_pvd_path:
        print(f"[demo] ✅ ParaView (パイプ): {out.pipe_pvd_path}")
    if out.pvd_path:
        print(f"[demo] ✅ ParaView (ライン): {out.pvd_path}")
    elif out.vtu_paths:
        print(f"[demo] ✅ ParaView: {out.vtu_paths[0]}")


if __name__ == "__main__":
    main()
