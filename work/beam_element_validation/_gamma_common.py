"""Phase γ 共通ヘルパ — multi-element CR Timoshenko 梁の implicit static アセンブラ.

[← README](README.md) | [← project README](../../README.md)

status-389 §2 Phase γ 計画 + status-391 §6.1 に沿い、CR Timoshenko 3D 梁要素を
**直線チェーン状 n_elements 個** に並べた系を ``timo_beam3d_cr_internal_force`` /
``timo_beam3d_cr_tangent_analytical`` の **直接アセンブル** で駆動する。
xkep_cae 本体の assembler を経由しない最小ドライバ。

提供物:

1. ``ChainedBeamSection``: ``BeamSection`` の multi-element 拡張。
   要素長 ``L_total / n_elements`` の単純チェーンを x 軸沿いに配置。
2. ``assemble_internal_force`` / ``assemble_tangent``: 全 DOF の f_int / K_T を
   要素ループでアセンブル。
3. ``solve_static_nr_chain``: 多要素 NR static（load stepping + prescribed disp 対応）。
4. ``compute_chord_total``: 始点〜終点直線距離（diagnostic 用）。

設計意図: Phase α/β と同様に foundation を最小単位で検証する（status-389 §1.1）。
``ChainedBeamSection`` は既存 ``BeamSection``（1 要素用）と独立で、共有は材料パラメータ
（E, nu, r, kappa_s）と ``timo_beam3d_cr_*`` 関数のみ。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from xkep_cae.elements._beam_cr import (
    timo_beam3d_cr_internal_force,
    timo_beam3d_cr_tangent_analytical,
)


@dataclass(frozen=True)
class ChainedBeamSection:
    """n_elements 要素を x 軸沿いに直列配置したチェーン梁.

    Args:
        L_total: チェーン全長 [mm]
        n_elements: 要素数 (≥ 1)
        r: 円形断面半径 [mm]
        E: 弾性係数 [MPa]
        nu: ポアソン比
        kappa_s: せん断補正係数（円形断面で 6/7）
    """

    L_total: float
    n_elements: int
    r: float
    E: float
    nu: float = 0.3
    kappa_s: float = 6.0 / 7.0

    def __post_init__(self) -> None:
        if self.n_elements < 1:
            raise ValueError(f"n_elements must be ≥ 1 (got {self.n_elements})")
        if self.L_total <= 0.0:
            raise ValueError(f"L_total must be > 0 (got {self.L_total})")

    @property
    def L_elem(self) -> float:
        return self.L_total / self.n_elements

    @property
    def n_nodes(self) -> int:
        return self.n_elements + 1

    @property
    def n_dofs(self) -> int:
        return 6 * self.n_nodes

    @property
    def A(self) -> float:
        return math.pi * self.r**2

    @property
    def I(self) -> float:  # noqa: E743 — beam mechanics standard symbol
        return math.pi * self.r**4 / 4.0

    @property
    def J(self) -> float:
        return math.pi * self.r**4 / 2.0

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def EI(self) -> float:
        return self.E * self.I

    @property
    def EA(self) -> float:
        return self.E * self.A

    def coords_init(self) -> np.ndarray:
        """(n_nodes, 3) initial node coordinates — x 軸 0..L_total に等間隔."""
        coords = np.zeros((self.n_nodes, 3), dtype=float)
        coords[:, 0] = np.linspace(0.0, self.L_total, self.n_nodes)
        return coords

    def beam_kwargs(self) -> dict:
        return dict(
            E=self.E,
            G=self.G,
            A=self.A,
            Iy=self.I,
            Iz=self.I,
            J=self.J,
            kappa_y=self.kappa_s,
            kappa_z=self.kappa_s,
        )

    def element_dofs(self, e: int) -> np.ndarray:
        """要素 e (0-indexed) の global DOF indices (12,)."""
        n0 = e
        n1 = e + 1
        return np.concatenate([np.arange(6 * n0, 6 * n0 + 6), np.arange(6 * n1, 6 * n1 + 6)])

    def element_coords(self, e: int) -> np.ndarray:
        """要素 e (0-indexed) の (2, 3) initial coords — coords_init から切り出し."""
        coords = self.coords_init()
        return coords[[e, e + 1]]


def assemble_internal_force(section: ChainedBeamSection, u_global: np.ndarray) -> np.ndarray:
    """n_dofs の内力ベクトルを要素ループでアセンブル."""
    if u_global.shape[0] != section.n_dofs:
        raise ValueError(f"u_global shape {u_global.shape} != n_dofs {section.n_dofs}")
    f_global = np.zeros(section.n_dofs, dtype=float)
    bk = section.beam_kwargs()
    for e in range(section.n_elements):
        coords_e = section.element_coords(e)
        dof_e = section.element_dofs(e)
        u_e = u_global[dof_e]
        f_e = timo_beam3d_cr_internal_force(coords_e, u_e, **bk)
        f_global[dof_e] += f_e
    return f_global


def assemble_tangent(section: ChainedBeamSection, u_global: np.ndarray) -> np.ndarray:
    """(n_dofs, n_dofs) 接線剛性をアセンブル（解析的接線使用）."""
    if u_global.shape[0] != section.n_dofs:
        raise ValueError(f"u_global shape {u_global.shape} != n_dofs {section.n_dofs}")
    K_global = np.zeros((section.n_dofs, section.n_dofs), dtype=float)
    bk = section.beam_kwargs()
    for e in range(section.n_elements):
        coords_e = section.element_coords(e)
        dof_e = section.element_dofs(e)
        u_e = u_global[dof_e]
        K_e = timo_beam3d_cr_tangent_analytical(coords_e, u_e, **bk)
        K_global[np.ix_(dof_e, dof_e)] += K_e
    return K_global


@dataclass
class ChainNRResult:
    """multi-element NR static の結果."""

    u: np.ndarray  # (n_dofs,)
    converged: bool
    n_steps: int
    n_iters_total: int
    final_residual: float
    f_int_final: np.ndarray  # (n_dofs,)


def solve_static_nr_chain(
    section: ChainedBeamSection,
    fixed_dofs: list[int],
    F_ext: np.ndarray,
    prescribed_disp: dict[int, float] | None = None,
    n_load_steps: int = 1,
    tol_rel: float = 1.0e-9,
    tol_abs: float = 1.0e-14,
    max_iter: int = 100,
    verbose: bool = False,
) -> ChainNRResult:
    """multi-element CR 梁の implicit static Newton-Raphson 解析.

    Args:
        section: ``ChainedBeamSection``.
        fixed_dofs: u=0 で固定する global DOF index list.
        F_ext: (n_dofs,) 外力ベクトル (load stepping で線形 ramp).
        prescribed_disp: {global DOF: 最終値} (load stepping で線形 ramp).
        n_load_steps: 荷重ステップ数 (large rotation で必要).
        tol_rel: 相対残差収束閾値.
        tol_abs: 絶対残差収束閾値.
        max_iter: 1 ステップあたり最大 NR 反復数.
        verbose: True なら各 NR 反復で残差をログ出力.
    """
    if F_ext.shape[0] != section.n_dofs:
        raise ValueError(f"F_ext shape {F_ext.shape} != n_dofs {section.n_dofs}")

    if prescribed_disp is None:
        prescribed_disp = {}

    fixed_set = set(fixed_dofs)
    prescribed_set = set(prescribed_disp.keys())
    if fixed_set & prescribed_set:
        raise ValueError(f"DOF {fixed_set & prescribed_set} は fixed と prescribed 両方に指定")
    active_dofs = sorted(set(range(section.n_dofs)) - fixed_set - prescribed_set)
    a_idx = np.array(active_dofs, dtype=int)

    u = np.zeros(section.n_dofs, dtype=float)
    n_iters_total = 0
    last_residual = 0.0

    for step in range(1, n_load_steps + 1):
        lam = step / n_load_steps
        F_step = lam * F_ext

        for dof, val in prescribed_disp.items():
            u[dof] = lam * val

        for it in range(max_iter):
            f_int = assemble_internal_force(section, u)
            r_full = F_step - f_int
            r_a = r_full[a_idx]

            res = float(np.linalg.norm(r_a))
            ref_F = float(np.linalg.norm(F_step[a_idx]))
            ref_fint = float(np.linalg.norm(f_int[a_idx]))
            ref = max(ref_F, ref_fint, 1.0)

            last_residual = res
            n_iters_total += 1

            if verbose:
                print(
                    f"  [step {step:3d}/{n_load_steps}] iter {it:3d}: "
                    f"||r||={res:.3e}, ref={ref:.3e}, ||r||/ref={res / ref:.3e}"
                )

            if res < tol_abs or res / ref < tol_rel:
                break

            K = assemble_tangent(section, u)
            K_aa = K[np.ix_(a_idx, a_idx)]
            try:
                du_a = np.linalg.solve(K_aa, r_a)
            except np.linalg.LinAlgError:
                return ChainNRResult(
                    u=u.copy(),
                    converged=False,
                    n_steps=step,
                    n_iters_total=n_iters_total,
                    final_residual=res,
                    f_int_final=f_int.copy(),
                )
            u[a_idx] += du_a
        else:
            f_int_final = assemble_internal_force(section, u)
            return ChainNRResult(
                u=u.copy(),
                converged=False,
                n_steps=step,
                n_iters_total=n_iters_total,
                final_residual=last_residual,
                f_int_final=f_int_final,
            )

    f_int_final = assemble_internal_force(section, u)
    return ChainNRResult(
        u=u.copy(),
        converged=True,
        n_steps=n_load_steps,
        n_iters_total=n_iters_total,
        final_residual=last_residual,
        f_int_final=f_int_final,
    )


def compute_chord_total(section: ChainedBeamSection, u_global: np.ndarray) -> float:
    """始点〜終点の直線距離 [mm]（変形後）.

    1 要素 CR は chord = L_elem を保存するが、multi-element では各要素の
    chord 和（polyline 長）と、両端ノード間の直線距離（end-to-end chord）は
    異なる。本ヘルパは **end-to-end chord**（後者）を返す。
    circular arc 解では `2R sin(θ/2)` に対応。
    """
    coords = section.coords_init()
    p_start = coords[0] + u_global[0:3]
    p_end = coords[-1] + u_global[6 * (section.n_nodes - 1) : 6 * (section.n_nodes - 1) + 3]
    return float(np.linalg.norm(p_end - p_start))


def compute_polyline_length(section: ChainedBeamSection, u_global: np.ndarray) -> float:
    """変形後 polyline 長 = Σ_e |p_{e+1} − p_e| [mm].

    1 要素 CR は L_elem を厳密保存するため、polyline 長は理論上 L_total に等しい
    （multi-element でも各要素の局所 chord は CR が保存）。
    """
    coords = section.coords_init()
    L = 0.0
    for e in range(section.n_elements):
        p0 = coords[e] + u_global[6 * e : 6 * e + 3]
        p1 = coords[e + 1] + u_global[6 * (e + 1) : 6 * (e + 1) + 3]
        L += float(np.linalg.norm(p1 - p0))
    return L
