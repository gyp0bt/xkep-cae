"""Phase α 共通ヘルパ — 1 要素 implicit static Newton-Raphson + 3 指標 AND gate.

[← README](README.md) | [← project README](../../README.md)

status-389 §2 Phase α 計画に沿い、CR Timoshenko 3D 梁要素を 1 要素の単純荷重ケース
で implicit static 検証する。本モジュールは:

1. `BeamSection`: L / r / E / ν 等から A, Iy, Iz, J, G, kappa_s を計算
2. `solve_static_nr`: 12 DOF 単要素の Newton-Raphson 静解析（load stepping 対応、
   prescribed displacement 対応）
3. `evaluate_three_metric_gate`: 3 指標 AND gate の判定 + 結果出力ヘルパ

`xkep_cae/elements/_beam_cr.py` の `timo_beam3d_cr_internal_force` /
`timo_beam3d_cr_tangent_analytical` を直接呼び出し、assembler 経由を避けて
foundation を最小単位で検証する（status-389 §1.1）。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from xkep_cae.elements._beam_cr import (
    timo_beam3d_cr_internal_force,
    timo_beam3d_cr_tangent_analytical,
)


@dataclass(frozen=True)
class BeamSection:
    """1 要素 CR Timoshenko 梁の断面 + 材料パラメータ.

    Args:
        L: 要素長 [mm]
        r: 円形断面半径 [mm]
        E: 弾性係数 [MPa = N/mm²]
        nu: ポアソン比
        kappa_s: せん断補正係数（Timoshenko、円形断面で 6/7）
    """

    L: float
    r: float
    E: float
    nu: float = 0.3
    kappa_s: float = 6.0 / 7.0

    @property
    def A(self) -> float:
        return math.pi * self.r**2

    @property
    def I(self) -> float:  # noqa: E743 — beam mechanics 標準記号 I を使う（曖昧性なし）
        """対称断面の慣性モーメント Iy = Iz = πr⁴/4."""
        return math.pi * self.r**4 / 4.0

    @property
    def J(self) -> float:
        """ねじり慣性モーメント（円形断面）J = π r⁴/2 = Iy + Iz."""
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

    def coords_2node(self) -> np.ndarray:
        """(2, 3) 節点座標 — x 軸沿いに 1 要素を配置."""
        return np.array([[0.0, 0.0, 0.0], [self.L, 0.0, 0.0]])

    def beam_kwargs(self) -> dict:
        """`timo_beam3d_cr_*` 関数群への共通 kwargs."""
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


@dataclass
class NRResult:
    """Newton-Raphson 解析結果."""

    u: np.ndarray  # (12,)
    converged: bool
    n_steps: int
    n_iters_total: int
    final_residual: float
    f_int_final: np.ndarray  # (12,)


def solve_static_nr(
    section: BeamSection,
    fixed_dofs: list[int],
    F_ext: np.ndarray,
    prescribed_disp: dict[int, float] | None = None,
    n_load_steps: int = 1,
    tol_rel: float = 1.0e-9,
    tol_abs: float = 1.0e-14,
    max_iter: int = 100,
    verbose: bool = False,
) -> NRResult:
    """1 要素 12 DOF の implicit static Newton-Raphson 解析.

    Args:
        section: 梁断面パラメータ
        fixed_dofs: u=0 で固定する DOF index list
        F_ext: (12,) 外力ベクトル — load stepping で `step_factor * F_ext` を適用
        prescribed_disp: {DOF index: 最終値} — load stepping で線形ランプ
        n_load_steps: 荷重ステップ数（large rotation で必要）
        tol_rel: 相対残差収束閾値（||r||/||F_ref||）
        tol_abs: 絶対残差収束閾値
        max_iter: 1 ステップあたり最大 NR 反復数
    """
    coords = section.coords_2node()
    bk = section.beam_kwargs()

    if prescribed_disp is None:
        prescribed_disp = {}

    fixed_set = set(fixed_dofs)
    prescribed_set = set(prescribed_disp.keys())
    if fixed_set & prescribed_set:
        raise ValueError(f"DOF {fixed_set & prescribed_set} は fixed と prescribed 両方に指定")
    active_dofs = sorted(set(range(12)) - fixed_set - prescribed_set)
    a_idx = np.array(active_dofs, dtype=int)

    u = np.zeros(12)
    n_iters_total = 0
    last_residual = 0.0

    for step in range(1, n_load_steps + 1):
        lam = step / n_load_steps
        F_step = lam * F_ext

        # 処方変位を当該ステップ目標値にセット（前ステップから一気にジャンプ → NR で f_int 整合）
        for dof, val in prescribed_disp.items():
            u[dof] = lam * val

        for it in range(max_iter):
            f_int = timo_beam3d_cr_internal_force(coords, u, **bk)
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

            K = timo_beam3d_cr_tangent_analytical(coords, u, **bk)
            K_aa = K[np.ix_(a_idx, a_idx)]
            try:
                du_a = np.linalg.solve(K_aa, r_a)
            except np.linalg.LinAlgError:
                return NRResult(
                    u=u.copy(),
                    converged=False,
                    n_steps=step,
                    n_iters_total=n_iters_total,
                    final_residual=res,
                    f_int_final=f_int.copy(),
                )
            u[a_idx] += du_a
        else:
            f_int_final = timo_beam3d_cr_internal_force(coords, u, **bk)
            return NRResult(
                u=u.copy(),
                converged=False,
                n_steps=step,
                n_iters_total=n_iters_total,
                final_residual=last_residual,
                f_int_final=f_int_final,
            )

    f_int_final = timo_beam3d_cr_internal_force(coords, u, **bk)
    return NRResult(
        u=u.copy(),
        converged=True,
        n_steps=n_load_steps,
        n_iters_total=n_iters_total,
        final_residual=last_residual,
        f_int_final=f_int_final,
    )


def compute_chord_arc_length(section: BeamSection, u: np.ndarray) -> float:
    """変形後 chord 長 — 1 要素なので 2 節点間距離."""
    coords = section.coords_2node()
    p1 = coords[0] + u[0:3]
    p2 = coords[1] + u[6:9]
    return float(np.linalg.norm(p2 - p1))


def numerical_strain_energy(u: np.ndarray, f_int: np.ndarray) -> float:
    """線形領域の歪エネルギー近似 SE ≈ (1/2) u^T f_int.

    注: 大変形 corotational 定式化では正確には
        SE_local = 0.5 d_cr^T Ke_local d_cr
    となるが、本ヘルパは（小変形領域での）診断値として 0.5 u^T f_int を返す。
    α-3 のような large κ では別途解析解との比較で 10〜30% 程度の誤差が出る
    可能性があり、SE は **診断列のみ** で gate に組み込まないことを推奨
    （status-388 で MPC 拘束 DOF 消去のため SE が桁違いに出た事例あり）。
    """
    return 0.5 * float(np.dot(u, f_int))


@dataclass
class MetricRow:
    """1 指標の比較行.

    Args:
        compare_abs: True なら `|analytical|` vs `|numerical|` を比較（符号規約を吸収）.
            梁要素の局所剛性符号は XZ 平面 M_y / θ_y 規約等で実装ごとに異なるため、
            kinematic 量は絶対値比較で透明性ルールを保つ（status-388 同様）.
        gate_threshold: 0 を超える相対誤差閾値. analytical=0 のときは絶対誤差を
            参照スケール `abs_zero_ref` で正規化して比較.
        abs_zero_ref: analytical ≈ 0 のときの絶対誤差正規化スケール.
    """

    name: str
    analytical: float
    numerical: float
    gate_threshold: float = 0.10  # 相対誤差 10%
    in_gate: bool = True
    compare_abs: bool = False
    abs_zero_ref: float = 1.0

    @property
    def rel_err(self) -> float:
        a = abs(self.analytical) if self.compare_abs else self.analytical
        n = abs(self.numerical) if self.compare_abs else self.numerical
        if abs(a) > 1e-30:
            return abs(n - a) / abs(a)
        # analytical ≈ 0: 絶対誤差を abs_zero_ref で正規化
        return abs(n - a) / max(abs(self.abs_zero_ref), 1e-30)

    @property
    def passed(self) -> bool:
        return self.rel_err < self.gate_threshold


def evaluate_three_metric_gate(
    case_name: str,
    rows: list[MetricRow],
    converged: bool,
    n_iters: int,
    n_steps: int,
) -> bool:
    """status-388 透明性ルール: 3 指標 AND gate 判定 + 表形式出力.

    Returns:
        bool: gate 全 PASS なら True。in_gate=True の行が 3 個未満なら FAIL.
    """
    print()
    print("=" * 80)
    print(f"[{case_name}] 3 指標 AND gate 判定")
    print("=" * 80)
    print(f"  Newton 収束: {'✓' if converged else '✗'}, steps={n_steps}, iters={n_iters}")

    print(f"\n  {'指標':30s} {'解析解':>15s} {'数値解':>15s} {'相対誤差':>10s} {'gate':>6s}")
    print("  " + "-" * 78)

    n_in_gate = 0
    n_passed = 0
    for row in rows:
        flag = "[診断]" if not row.in_gate else ("✓" if row.passed else "✗")
        print(
            f"  {row.name:30s} {row.analytical:>15.6e} "
            f"{row.numerical:>15.6e} {row.rel_err * 100:>9.3f}% {flag:>6s}"
        )
        if row.in_gate:
            n_in_gate += 1
            if row.passed:
                n_passed += 1

    print("  " + "-" * 78)
    if n_in_gate < 3:
        print(f"  [FAIL] gate 対象指標 {n_in_gate} < 3 — 透明性ルール違反")
        return False
    if not converged:
        print("  [FAIL] Newton 未収束")
        return False
    if n_passed < n_in_gate:
        print(f"  [FAIL] {n_passed}/{n_in_gate} 指標 PASS — 3 指標 AND 不成立")
        return False
    print(f"  [PASS] {n_passed}/{n_in_gate} 指標 全 PASS — 解析解との一致確定")
    return True


def run_case(
    case_name: str,
    section: BeamSection,
    fixed_dofs: list[int],
    F_ext: np.ndarray,
    analytical_metrics: Callable[[BeamSection, NRResult], list[MetricRow]],
    prescribed_disp: dict[int, float] | None = None,
    n_load_steps: int = 1,
    tol_rel: float = 1.0e-9,
    max_iter: int = 100,
    verbose: bool = False,
) -> bool:
    """1 ケース 1 要素 implicit static 検証の標準ドライバ.

    Args:
        analytical_metrics: 解析解と数値解から 3+ 指標 MetricRow リストを返す callback

    Returns:
        bool: 3 指標 AND gate PASS なら True
    """
    print(f"\n{'=' * 80}")
    print(f"  Phase α: {case_name}")
    print(f"{'=' * 80}")
    print(
        f"  L={section.L} mm, r={section.r} mm, E={section.E} MPa, "
        f"ν={section.nu}, A={section.A:.4f} mm², I={section.I:.4e} mm⁴"
    )
    print(f"  EA={section.EA:.4e} N, EI={section.EI:.4e} N·mm², G={section.G:.4e} MPa")

    res = solve_static_nr(
        section=section,
        fixed_dofs=fixed_dofs,
        F_ext=F_ext,
        prescribed_disp=prescribed_disp,
        n_load_steps=n_load_steps,
        tol_rel=tol_rel,
        max_iter=max_iter,
        verbose=verbose,
    )

    rows = analytical_metrics(section, res)
    return evaluate_three_metric_gate(
        case_name=case_name,
        rows=rows,
        converged=res.converged,
        n_iters=res.n_iters_total,
        n_steps=res.n_steps,
    )
