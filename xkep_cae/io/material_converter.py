"""Abaqus 材料定義 → xkep-cae 構成則オブジェクトの変換.

AbaqusMaterial（.inp パーサーの出力）から Plasticity1D / PlaneStrainPlasticity
などの構成則オブジェクトを生成するコンバータ関数群。

対応範囲:
  - *ELASTIC → ヤング率 / ポアソン比
  - *PLASTIC (HARDENING=ISOTROPIC) → TabularIsotropicHardening
  - *PLASTIC (HARDENING=KINEMATIC) → Armstrong-Frederick 移動硬化

制限事項:
  - HARDENING=COMBINED は未対応（ValueError）
  - *DENSITY は構成則オブジェクトには含まれない（別途参照）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from xkep_cae.materials.plasticity_1d import (
    IsotropicHardening,
    KinematicHardening,
    Plasticity1D,
    TabularIsotropicHardening,
)

if TYPE_CHECKING:
    from xkep_cae.io.abaqus_inp import AbaqusMaterial
    from xkep_cae.materials.plasticity_3d import PlaneStrainPlasticity


def kinematic_table_to_armstrong_frederick(
    table: list[tuple[float, float]],
) -> tuple[float, float, float]:
    """KINEMATIC テーブル → (sigma_y0, C_kin, gamma_kin) 変換.

    Abaqus *PLASTIC, HARDENING=KINEMATIC のテーブルデータを
    Armstrong-Frederick 移動硬化パラメータに変換する。

    テーブルの各点 (sigma_i, eps_p_i) における背応力を
    beta_i = sigma_i - sigma_y0 として算出し:

      - 1点テーブル: 完全弾塑性（C_kin=0, gamma_kin=0）
      - 2点テーブル: 線形移動硬化（C_kin=勾配, gamma_kin=0）
      - 3点以上: Armstrong-Frederick 非線形フィッティング
        beta(eps_p) = (C_kin / gamma_kin)(1 - exp(-gamma_kin * eps_p))

    3点以上の場合、Abaqus の多直線移動硬化を AF 指数関数で近似する。
    近似精度はテーブルデータの形状に依存するため、フィッティング残差を
    確認することを推奨する。

    Args:
        table: [(sigma_y, eps_p), ...] テーブルデータ。
            eps_p は単調増加。最初の点は eps_p=0 を想定。

    Returns:
        (sigma_y0, C_kin, gamma_kin) タプル

    Raises:
        ValueError: テーブルが空の場合
    """
    if len(table) < 1:
        raise ValueError("テーブルは最低1点必要")

    sigma_y0 = table[0][0]

    # 背応力データ: beta_i = sigma_i - sigma_y0
    eps_p_data: list[float] = []
    beta_data: list[float] = []
    for sigma_i, eps_p_i in table:
        if eps_p_i > 0:
            eps_p_data.append(eps_p_i)
            beta_data.append(sigma_i - sigma_y0)

    # 有効データ点がない場合: 完全弾塑性
    if len(eps_p_data) == 0:
        return sigma_y0, 0.0, 0.0

    # 2点テーブル（1有効データ点）: 線形移動硬化
    if len(eps_p_data) == 1:
        C_kin = beta_data[0] / eps_p_data[0]
        return sigma_y0, C_kin, 0.0

    # 3点以上: Armstrong-Frederick 非線形フィッティング
    from scipy.optimize import least_squares

    eps_p_arr = np.asarray(eps_p_data, dtype=float)
    beta_arr = np.asarray(beta_data, dtype=float)

    # 初期推定
    C_init = beta_data[0] / eps_p_data[0]  # 初期勾配
    beta_sat = beta_data[-1]
    gamma_init = C_init / beta_sat if beta_sat > 0 else 1.0

    def _residuals(params: np.ndarray) -> np.ndarray:
        C, gamma = params
        if gamma < 1e-12:
            beta_model = C * eps_p_arr
        else:
            beta_model = (C / gamma) * (1.0 - np.exp(-gamma * eps_p_arr))
        return beta_model - beta_arr

    result = least_squares(
        _residuals,
        x0=[C_init, max(gamma_init, 1e-6)],
        bounds=([0.0, 0.0], [np.inf, np.inf]),
        method="trf",
    )

    C_kin_fit, gamma_kin_fit = result.x

    # gamma_kin が非常に小さい場合は線形とみなす
    # 最適化の数値精度を考慮し 1e-4 を閾値とする
    if gamma_kin_fit < 1e-4:
        # 線形最小二乗: beta = C * eps_p
        C_kin_lin = float(np.sum(eps_p_arr * beta_arr) / np.sum(eps_p_arr**2))
        return sigma_y0, C_kin_lin, 0.0

    return sigma_y0, float(C_kin_fit), float(gamma_kin_fit)


def abaqus_material_to_plasticity_1d(
    mat: AbaqusMaterial,
) -> Plasticity1D:
    """AbaqusMaterial から Plasticity1D を生成する.

    Args:
        mat: AbaqusMaterial オブジェクト（*ELASTIC 必須）

    Returns:
        Plasticity1D オブジェクト

    Raises:
        ValueError: *ELASTIC が未定義、またはHARDENING=COMBINED の場合
    """
    if mat.elastic is None:
        raise ValueError(f"材料 '{mat.name}' に *ELASTIC が未定義")

    E, _nu = mat.elastic

    if mat.plastic is not None:
        if mat.plastic_hardening == "ISOTROPIC":
            iso: IsotropicHardening | TabularIsotropicHardening = TabularIsotropicHardening(
                table=mat.plastic
            )
            return Plasticity1D(E=E, iso=iso)

        if mat.plastic_hardening == "KINEMATIC":
            sigma_y0, C_kin, gamma_kin = kinematic_table_to_armstrong_frederick(
                mat.plastic,
            )
            iso_kin = IsotropicHardening(sigma_y0=sigma_y0)
            kin = KinematicHardening(C_kin=C_kin, gamma_kin=gamma_kin)
            return Plasticity1D(E=E, iso=iso_kin, kin=kin)

        raise ValueError(
            f"材料 '{mat.name}': HARDENING={mat.plastic_hardening} は "
            "Plasticity1D では未対応（ISOTROPIC / KINEMATIC のみ）"
        )

    iso_elastic = IsotropicHardening(sigma_y0=E * 1e10)  # 弾性のみ
    return Plasticity1D(E=E, iso=iso_elastic)


def abaqus_material_to_plane_strain_plasticity(
    mat: AbaqusMaterial,
) -> PlaneStrainPlasticity:
    """AbaqusMaterial から PlaneStrainPlasticity を生成する.

    Args:
        mat: AbaqusMaterial オブジェクト（*ELASTIC 必須、nu > 0）

    Returns:
        PlaneStrainPlasticity オブジェクト

    Raises:
        ValueError: *ELASTIC 未定義、nu 未指定、または COMBINED の場合
    """
    from xkep_cae.materials.plasticity_3d import (
        IsotropicHardening3D,
        KinematicHardening3D,
        PlaneStrainPlasticity,
    )

    if mat.elastic is None:
        raise ValueError(f"材料 '{mat.name}' に *ELASTIC が未定義")

    E, nu = mat.elastic

    if mat.plastic is not None:
        if mat.plastic_hardening == "ISOTROPIC":
            iso: IsotropicHardening3D | TabularIsotropicHardening = TabularIsotropicHardening(
                table=mat.plastic
            )
            return PlaneStrainPlasticity(E=E, nu=nu, iso=iso)

        if mat.plastic_hardening == "KINEMATIC":
            sigma_y0, C_kin, gamma_kin = kinematic_table_to_armstrong_frederick(
                mat.plastic,
            )
            iso_kin = IsotropicHardening3D(sigma_y0=sigma_y0)
            kin = KinematicHardening3D(C_kin=C_kin, gamma_kin=gamma_kin)
            return PlaneStrainPlasticity(E=E, nu=nu, iso=iso_kin, kin=kin)

        raise ValueError(
            f"材料 '{mat.name}': HARDENING={mat.plastic_hardening} は "
            "PlaneStrainPlasticity では未対応（ISOTROPIC / KINEMATIC のみ）"
        )

    iso_elastic = IsotropicHardening3D(sigma_y0=E * 1e10)
    return PlaneStrainPlasticity(E=E, nu=nu, iso=iso_elastic)


__all__ = [
    "abaqus_material_to_plane_strain_plasticity",
    "abaqus_material_to_plasticity_1d",
    "kinematic_table_to_armstrong_frederick",
]
