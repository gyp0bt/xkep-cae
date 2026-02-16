"""Euler elastica ベンチマーク — 幾何学的非線形梁の検証.

片持ち梁の大変形解析を解析解と比較し、非線形 Cosserat rod 要素の妥当性を検証する。

テスト 1: 端モーメント (pure bending)
  片持ち梁に先端モーメント M を載荷 → 一様曲率 κ = M/EI で円弧に変形。
  解析解:
    x_tip = (EI/M) · sin(ML/EI)
    y_tip = (EI/M) · (1 - cos(ML/EI))
  θ = ML/EI = π/4, π/2, π, 3π/2, 2π (完全円) でパラメトリックテスト。

テスト 2: 先端集中荷重 (tip load)
  片持ち梁に先端鉛直荷重 P を載荷。
  Elastica の厳密 ODE (EI·θ'' = -P·cos θ) の数値解（shooting method）と比較。
  PL²/EI = 1, 2, 5, 10
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.elements.beam_cosserat import (
    CosseratRod,
    assemble_cosserat_beam,
)
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection
from xkep_cae.solver import newton_raphson

# --- 共通パラメータ ---
# Euler-Bernoulli 極限に近い細長い梁（せん断変形無視）
E = 1.0e6  # ヤング率
NU = 0.0  # ポアソン比 0 で G = E/2
L = 10.0  # 梁長さ

# 正方形断面 (薄い)
_b = 0.1  # 幅
_h = 0.1  # 高さ
SEC = BeamSection.rectangle(_b, _h)
EI = E * SEC.Iz  # = E * b*h³/12


def _make_material() -> BeamElastic1D:
    return BeamElastic1D(E=E, nu=NU)


def _solve_cantilever_moment(
    M: float,
    n_elems: int = 20,
    n_load_steps: int = 10,
    max_iter: int = 50,
) -> np.ndarray:
    """片持ち梁に先端モーメント M を載荷して解く.

    Returns:
        u: (total_dof,) 最終変位ベクトル
    """
    mat = _make_material()
    # nonlinear=True で幾何学的非線形定式化を使用
    rod = CosseratRod(section=SEC, nonlinear=True)
    n_nodes = n_elems + 1
    total_dof = n_nodes * 6

    def _assemble_tangent(u):
        K, _ = assemble_cosserat_beam(
            n_elems,
            L,
            rod,
            mat,
            u,
            stiffness=True,
            internal_force=False,
        )
        return sp.csr_matrix(K)

    def _assemble_fint(u):
        _, f_int = assemble_cosserat_beam(
            n_elems,
            L,
            rod,
            mat,
            u,
            stiffness=False,
            internal_force=True,
        )
        return f_int

    # 先端モーメント: z軸まわり（xy面内曲げ）
    f_ext = np.zeros(total_dof)
    tip_mz_dof = 6 * n_elems + 5  # 先端ノードの θz
    f_ext[tip_mz_dof] = M

    # 固定端: 節点0の全6DOF
    fixed_dofs = np.arange(6)

    result = newton_raphson(
        f_ext,
        fixed_dofs,
        _assemble_tangent,
        _assemble_fint,
        n_load_steps=n_load_steps,
        max_iter=max_iter,
        show_progress=False,
    )
    assert result.converged, f"M={M:.4e} で NR が収束しない"
    return result.u


def _solve_cantilever_tip_load(
    P: float,
    n_elems: int = 20,
    n_load_steps: int = 20,
    max_iter: int = 50,
) -> np.ndarray:
    """片持ち梁に先端鉛直荷重 P を載荷して解く.

    Returns:
        u: (total_dof,) 最終変位ベクトル
    """
    mat = _make_material()
    rod = CosseratRod(section=SEC, nonlinear=True)
    n_nodes = n_elems + 1
    total_dof = n_nodes * 6

    def _assemble_tangent(u):
        K, _ = assemble_cosserat_beam(
            n_elems,
            L,
            rod,
            mat,
            u,
            stiffness=True,
            internal_force=False,
        )
        return sp.csr_matrix(K)

    def _assemble_fint(u):
        _, f_int = assemble_cosserat_beam(
            n_elems,
            L,
            rod,
            mat,
            u,
            stiffness=False,
            internal_force=True,
        )
        return f_int

    # 先端 y方向荷重
    f_ext = np.zeros(total_dof)
    tip_uy_dof = 6 * n_elems + 1  # 先端ノードの uy
    f_ext[tip_uy_dof] = P

    fixed_dofs = np.arange(6)

    result = newton_raphson(
        f_ext,
        fixed_dofs,
        _assemble_tangent,
        _assemble_fint,
        n_load_steps=n_load_steps,
        max_iter=max_iter,
        show_progress=False,
    )
    assert result.converged, f"P={P:.4e} で NR が収束しない"
    return result.u


class TestEndMoment:
    """端モーメントによる純曲げ — 解析解との比較.

    一様な端モーメント M による片持ち梁の変形。
    曲率 κ = M/EI が一様なので、梁は円弧状に変形する。

    解析解:
      θ_total = ML/EI
      x_tip = (EI/M) · sin(θ_total)
      y_tip = (EI/M) · (1 - cos(θ_total))
    """

    @pytest.mark.parametrize(
        "theta_total, n_elems, n_steps, tol_pct",
        [
            (np.pi / 4, 20, 5, 2.0),  # 45°: 穏やかな非線形
            (np.pi / 2, 20, 10, 2.0),  # 90°: 中程度
            (np.pi, 20, 20, 3.0),  # 180°: 半円
            (3 * np.pi / 2, 30, 30, 5.0),  # 270°: 大変形
            (2 * np.pi, 40, 40, 5.0),  # 360°: 完全円
        ],
        ids=["pi/4", "pi/2", "pi", "3pi/2", "2pi"],
    )
    def test_tip_position(self, theta_total, n_elems, n_steps, tol_pct):
        """先端位置が解析解と一致する."""
        M = theta_total * EI / L

        u = _solve_cantilever_moment(
            M,
            n_elems=n_elems,
            n_load_steps=n_steps,
            max_iter=100,
        )

        # 先端の変形後位置
        tip_node = n_elems
        x_tip = L + u[6 * tip_node + 0]  # x座標: 初期 + 変位
        y_tip = 0.0 + u[6 * tip_node + 1]  # y座標: 初期 + 変位

        # 解析解
        R_curv = EI / M  # 曲率半径
        x_exact = R_curv * np.sin(theta_total)
        y_exact = R_curv * (1.0 - np.cos(theta_total))

        # 誤差 (先端位置のユークリッド距離)
        pos_error = np.sqrt((x_tip - x_exact) ** 2 + (y_tip - y_exact) ** 2)
        tol = tol_pct / 100.0 * L

        assert pos_error < tol, (
            f"θ={theta_total / np.pi:.2f}π: "
            f"数値解 ({x_tip:.4f}, {y_tip:.4f}), "
            f"解析解 ({x_exact:.4f}, {y_exact:.4f}), "
            f"誤差 {pos_error:.4f} > 許容 {tol:.4f} ({tol_pct}%L)"
        )


class TestTipLoad:
    """先端集中荷重 — elastica 厳密解との比較.

    片持ち梁に先端鉛直荷重 P を載荷。
    無次元パラメータ α = PL²/EI に対する先端変位を比較。

    参照値は elastica の厳密 ODE (EI·θ'' = -P·cos θ) を
    shooting method (scipy.optimize.brentq + solve_ivp) で
    rtol=1e-12 の精度で解いた値。

    参照値:
      α=1:  δx/L = 0.05634, δy/L = 0.30174
      α=2:  δx/L = 0.16056, δy/L = 0.49349
      α=5:  δx/L = 0.38756, δy/L = 0.71384
      α=10: δx/L = 0.55494, δy/L = 0.81066
    """

    # Elastica 厳密解: (alpha, shortening/L, deflection/L)
    REFERENCE = {
        1: (0.05634, 0.30174),
        2: (0.16056, 0.49349),
        5: (0.38756, 0.71384),
        10: (0.55494, 0.81066),
    }

    @pytest.mark.parametrize(
        "alpha, n_elems, n_steps",
        [
            (1, 20, 10),
            (2, 20, 20),
            (5, 30, 30),
            (10, 40, 40),
        ],
        ids=["alpha=1", "alpha=2", "alpha=5", "alpha=10"],
    )
    def test_tip_displacement(self, alpha, n_elems, n_steps):
        """先端変位が Mattiasson 参照値と一致する."""
        P = alpha * EI / L**2

        u = _solve_cantilever_tip_load(
            P,
            n_elems=n_elems,
            n_load_steps=n_steps,
            max_iter=100,
        )

        tip_node = n_elems
        dx = u[6 * tip_node + 0]  # x方向変位（短縮: 負）
        dy = u[6 * tip_node + 1]  # y方向変位（正）

        dx_ref, dy_ref = self.REFERENCE[alpha]
        # Mattiasson の dx は短縮量の絶対値
        dx_num = -dx / L  # 数値解の短縮量（正値に変換）
        dy_num = dy / L

        tol = 0.05  # 5% 相対許容誤差
        err_dx = abs(dx_num - dx_ref) / max(dx_ref, 1e-10)
        err_dy = abs(dy_num - dy_ref) / dy_ref

        assert err_dx < tol, (
            f"α={alpha}: δx/L 数値={dx_num:.4f}, 参照={dx_ref:.4f}, 相対誤差={err_dx:.3f} > {tol}"
        )
        assert err_dy < tol, (
            f"α={alpha}: δy/L 数値={dy_num:.4f}, 参照={dy_ref:.4f}, 相対誤差={err_dy:.3f} > {tol}"
        )
