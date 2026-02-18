"""Abaqus 材料定義 → xkep-cae 構成則オブジェクトの変換.

AbaqusMaterial（.inp パーサーの出力）から Plasticity1D / PlaneStrainPlasticity
などの構成則オブジェクトを生成するコンバータ関数群。

対応範囲:
  - *ELASTIC → ヤング率 / ポアソン比
  - *PLASTIC (HARDENING=ISOTROPIC) → TabularIsotropicHardening

制限事項:
  - HARDENING=KINEMATIC / COMBINED は未対応（ValueError）
  - *DENSITY は構成則オブジェクトには含まれない（別途参照）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from xkep_cae.materials.plasticity_1d import (
    IsotropicHardening,
    Plasticity1D,
    TabularIsotropicHardening,
)

if TYPE_CHECKING:
    from xkep_cae.io.abaqus_inp import AbaqusMaterial
    from xkep_cae.materials.plasticity_3d import PlaneStrainPlasticity


def abaqus_material_to_plasticity_1d(
    mat: AbaqusMaterial,
) -> Plasticity1D:
    """AbaqusMaterial から Plasticity1D を生成する.

    Args:
        mat: AbaqusMaterial オブジェクト（*ELASTIC 必須）

    Returns:
        Plasticity1D オブジェクト

    Raises:
        ValueError: *ELASTIC が未定義、またはHARDENING=KINEMATIC/COMBINED の場合
    """
    if mat.elastic is None:
        raise ValueError(f"材料 '{mat.name}' に *ELASTIC が未定義")

    E, _nu = mat.elastic

    if mat.plastic is not None:
        if mat.plastic_hardening != "ISOTROPIC":
            raise ValueError(
                f"材料 '{mat.name}': HARDENING={mat.plastic_hardening} は "
                "Plasticity1D では未対応（ISOTROPIC のみ）"
            )
        iso: IsotropicHardening | TabularIsotropicHardening = TabularIsotropicHardening(
            table=mat.plastic,
        )
    else:
        iso = IsotropicHardening(sigma_y0=E * 1e10)  # 弾性のみ（降伏しない）

    return Plasticity1D(E=E, iso=iso)


def abaqus_material_to_plane_strain_plasticity(
    mat: AbaqusMaterial,
) -> PlaneStrainPlasticity:
    """AbaqusMaterial から PlaneStrainPlasticity を生成する.

    Args:
        mat: AbaqusMaterial オブジェクト（*ELASTIC 必須、nu > 0）

    Returns:
        PlaneStrainPlasticity オブジェクト

    Raises:
        ValueError: *ELASTIC 未定義、nu 未指定、または KINEMATIC/COMBINED の場合
    """
    from xkep_cae.materials.plasticity_3d import (
        IsotropicHardening3D,
        PlaneStrainPlasticity,
    )

    if mat.elastic is None:
        raise ValueError(f"材料 '{mat.name}' に *ELASTIC が未定義")

    E, nu = mat.elastic

    if mat.plastic is not None:
        if mat.plastic_hardening != "ISOTROPIC":
            raise ValueError(
                f"材料 '{mat.name}': HARDENING={mat.plastic_hardening} は "
                "PlaneStrainPlasticity では未対応（ISOTROPIC のみ）"
            )
        iso: IsotropicHardening3D | TabularIsotropicHardening = TabularIsotropicHardening(
            table=mat.plastic,
        )
    else:
        iso = IsotropicHardening3D(sigma_y0=E * 1e10)

    return PlaneStrainPlasticity(E=E, nu=nu, iso=iso)


__all__ = [
    "abaqus_material_to_plane_strain_plasticity",
    "abaqus_material_to_plasticity_1d",
]
