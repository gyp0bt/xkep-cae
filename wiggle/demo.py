"""1+6 古典撚線（中心 1 本 + 外周 6 本）の撚り上がりを MP4 で書き出す。"""

from __future__ import annotations

from pathlib import Path

from wiggle.kinematics import StranderConfig
from wiggle.render import render_animation, render_still


def main() -> None:
    cfg = StranderConfig(
        n_outer=6,
        R_layer=1.1,
        R_bobbin=4.5,
        L_pitch=9.0,
        z_lay=5.5,
        L_tail_in=5.5,
        L_tail_out=22.0,
        n_segments=320,
        core_radius=0.22,
        outer_radius=0.20,
    )
    print(f"helix angle = {cfg.helix_angle_deg:.2f} deg")

    out_dir = Path("results/wiggle")
    out_dir.mkdir(parents=True, exist_ok=True)

    still = render_still(cfg, out_dir / "strander_1plus6_t0.png", t=0.0)
    print(f"still: {still}")

    mp4 = render_animation(
        cfg,
        out_dir / "strander_1plus6.mp4",
        duration=6.0,
        fps=24,
    )
    print(f"mp4: {mp4}")


if __name__ == "__main__":
    main()
