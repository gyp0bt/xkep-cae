"""ファイバー断面 CR 梁要素 Process.

CR（Corotational）定式化の Timoshenko 梁要素で、
断面応答をファイバー積分で評価する。

既存 `_beam_cr.timo_beam3d_cr_internal_force` との差分:
  - EI / EA を定数ではなく、セクション積分結果から取り出す
  - 要素ごとに SectionState を保持
  - 弾性材料では線形梁と完全一致する

設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md Phase F4
[← README](../../../README.md)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from xkep_cae.core.base import AbstractProcess, ProcessMeta
from xkep_cae.elements._beam_cr import (
    _beam3d_length_and_direction,
    _build_local_axes,
    _rodrigues_rotation,
    _rotmat_to_rotvec,
    _rotvec_to_rotmat,
    _skew,
    _tangent_operator,
    _tangent_operator_inv,
    _transformation_matrix_3d,
)
from xkep_cae.elements.fiber.integrator import (
    FiberIntegratorConfig,
    FiberSectionIntegratorProcess,
)
from xkep_cae.elements.fiber.section import CircularFiberSection
from xkep_cae.elements.fiber.state import SectionState

# =====================================================================
# Process I/O
# =====================================================================


@dataclass(frozen=True)
class StrandFiberBeamConfig:
    """ファイバー梁要素の入力.

    Attributes:
        coords_init: 要素初期座標 (2, 3)
        u_elem: 要素変位ベクトル (12,)
        section: ファイバー断面定義
        material: 1D 材料則（Fiber1DMaterialStrategy 準拠）
        section_state: セクション状態（1 積分点）
        G: せん断弾性係数
        J: ねじり定数
        kappa_y: せん断補正係数 (y)
        kappa_z: せん断補正係数 (z)
        v_ref: 参照ベクトル
    """

    coords_init: np.ndarray
    u_elem: np.ndarray
    section: CircularFiberSection
    material: object  # Fiber1DMaterialStrategy
    section_state: SectionState
    G: float
    J: float
    kappa_y: float
    kappa_z: float = 0.0
    v_ref: np.ndarray | None = None


@dataclass(frozen=True)
class StrandFiberBeamResult:
    """ファイバー梁要素の出力.

    Attributes:
        f_int: 内力ベクトル (12,) グローバル座標
        K_elem: 接線剛性行列 (12, 12) グローバル座標
        section_state_new: 更新後のセクション状態
    """

    f_int: np.ndarray
    K_elem: np.ndarray
    section_state_new: SectionState


# =====================================================================
# 補助関数
# =====================================================================


def _timo_ke_from_section(
    EA: float,
    EI_y: float,
    EI_z: float,
    GJ: float,
    kGA_y: float,
    kGA_z: float,
    L: float,
) -> np.ndarray:
    """断面特性から Timoshenko 12x12 局所剛性行列を構築.

    標準の `timo_beam3d_ke_local` と同一構造だが、
    EA, EI_y, EI_z を独立パラメータとして受け取る。
    せん断剛性 kGA は別途指定し、bending-shear のカップリング（Φ項）を正確に構築。

    Args:
        EA: 軸剛性（断面積分から）
        EI_y: y 軸周り曲げ剛性（断面積分から）
        EI_z: z 軸周り曲げ剛性（断面積分から）
        GJ: ねじり剛性
        kGA_y: せん断剛性 y 方向（kappa_y * G * A）
        kGA_z: せん断剛性 z 方向（kappa_z * G * A）
        L: 要素長さ

    Returns:
        Ke: (12, 12) 局所剛性行列
    """
    Phi_y = 12.0 * EI_y / (kGA_y * L * L) if kGA_y > 0 else 0.0
    Phi_z = 12.0 * EI_z / (kGA_z * L * L) if kGA_z > 0 else 0.0

    denom_y = 1.0 + Phi_y
    denom_z = 1.0 + Phi_z

    Ke = np.zeros((12, 12), dtype=float)

    # 軸剛性
    EA_L = EA / L
    Ke[0, 0] = EA_L
    Ke[0, 6] = -EA_L
    Ke[6, 0] = -EA_L
    Ke[6, 6] = EA_L

    # ねじり剛性
    GJ_L = GJ / L
    Ke[3, 3] = GJ_L
    Ke[3, 9] = -GJ_L
    Ke[9, 3] = -GJ_L
    Ke[9, 9] = GJ_L

    # xy 面曲げ（EI_z を使用、DOFs: v1, θz1, v2, θz2 = 1,5,7,11）
    EIz_L3 = EI_z / (L**3)
    EIz_L2 = EI_z / (L**2)
    EIz_L = EI_z / L

    Ke[1, 1] = 12.0 * EIz_L3 / denom_z
    Ke[1, 5] = 6.0 * EIz_L2 / denom_z
    Ke[1, 7] = -12.0 * EIz_L3 / denom_z
    Ke[1, 11] = 6.0 * EIz_L2 / denom_z
    Ke[5, 1] = 6.0 * EIz_L2 / denom_z
    Ke[5, 5] = (4.0 + Phi_z) * EIz_L / denom_z
    Ke[5, 7] = -6.0 * EIz_L2 / denom_z
    Ke[5, 11] = (2.0 - Phi_z) * EIz_L / denom_z
    Ke[7, 1] = -12.0 * EIz_L3 / denom_z
    Ke[7, 5] = -6.0 * EIz_L2 / denom_z
    Ke[7, 7] = 12.0 * EIz_L3 / denom_z
    Ke[7, 11] = -6.0 * EIz_L2 / denom_z
    Ke[11, 1] = 6.0 * EIz_L2 / denom_z
    Ke[11, 5] = (2.0 - Phi_z) * EIz_L / denom_z
    Ke[11, 7] = -6.0 * EIz_L2 / denom_z
    Ke[11, 11] = (4.0 + Phi_z) * EIz_L / denom_z

    # xz 面曲げ（EI_y を使用、DOFs: w1, θy1, w2, θy2 = 2,4,8,10）
    EIy_L3 = EI_y / (L**3)
    EIy_L2 = EI_y / (L**2)
    EIy_L = EI_y / L

    Ke[2, 2] = 12.0 * EIy_L3 / denom_y
    Ke[2, 4] = -6.0 * EIy_L2 / denom_y
    Ke[2, 8] = -12.0 * EIy_L3 / denom_y
    Ke[2, 10] = -6.0 * EIy_L2 / denom_y
    Ke[4, 2] = -6.0 * EIy_L2 / denom_y
    Ke[4, 4] = (4.0 + Phi_y) * EIy_L / denom_y
    Ke[4, 8] = 6.0 * EIy_L2 / denom_y
    Ke[4, 10] = (2.0 - Phi_y) * EIy_L / denom_y
    Ke[8, 2] = -12.0 * EIy_L3 / denom_y
    Ke[8, 4] = 6.0 * EIy_L2 / denom_y
    Ke[8, 8] = 12.0 * EIy_L3 / denom_y
    Ke[8, 10] = 6.0 * EIy_L2 / denom_y
    Ke[10, 2] = -6.0 * EIy_L2 / denom_y
    Ke[10, 4] = (2.0 - Phi_y) * EIy_L / denom_y
    Ke[10, 8] = 6.0 * EIy_L2 / denom_y
    Ke[10, 10] = (4.0 + Phi_y) * EIy_L / denom_y

    return Ke


@dataclass
class _CRKinematics:
    """CR 運動学の中間結果."""

    d_cr: np.ndarray  # (12,) 局所変位
    R_cr: np.ndarray  # (3, 3) corotated frame
    T_cr: np.ndarray  # (12, 12) 変換行列
    L_0: float  # 初期要素長
    L_def: float  # 変形後要素長
    e_x_def: np.ndarray  # (3,) 変形後軸方向ベクトル
    theta_def1: np.ndarray  # (3,) ノード1の CR 局所回転
    theta_def2: np.ndarray  # (3,) ノード2の CR 局所回転
    u_elem: np.ndarray  # (12,) 元の変位ベクトル


def _extract_cr_kinematics(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
    v_ref: np.ndarray | None = None,
) -> _CRKinematics:
    """CR 運動学: 解析的接線に必要な全中間量を抽出.

    Args:
        coords_init: (2, 3) 要素初期座標
        u_elem: (12,) 要素変位ベクトル
        v_ref: 参照ベクトル

    Returns:
        _CRKinematics
    """
    L_0, e_x_0 = _beam3d_length_and_direction(coords_init)
    R_0 = _build_local_axes(e_x_0, v_ref)

    x1_def = coords_init[0] + u_elem[0:3]
    x2_def = coords_init[1] + u_elem[6:9]
    coords_def = np.array([x1_def, x2_def])
    L_def, e_x_def = _beam3d_length_and_direction(coords_def)

    R_rod = _rodrigues_rotation(e_x_0, e_x_def)
    R_cr = R_0 @ R_rod.T

    R_node1 = _rotvec_to_rotmat(u_elem[3:6])
    R_node2 = _rotvec_to_rotmat(u_elem[9:12])

    R_def1 = R_cr @ R_node1 @ R_0.T
    R_def2 = R_cr @ R_node2 @ R_0.T
    theta_def1 = _rotmat_to_rotvec(R_def1)
    theta_def2 = _rotmat_to_rotvec(R_def2)

    d_cr = np.zeros(12, dtype=float)
    d_cr[3:6] = theta_def1
    d_cr[6] = L_def - L_0
    d_cr[9:12] = theta_def2

    T_cr = _transformation_matrix_3d(R_cr)

    return _CRKinematics(
        d_cr=d_cr,
        R_cr=R_cr,
        T_cr=T_cr,
        L_0=L_0,
        L_def=L_def,
        e_x_def=e_x_def,
        theta_def1=theta_def1,
        theta_def2=theta_def2,
        u_elem=u_elem,
    )


def _compute_cr_tangent(
    kin: _CRKinematics,
    Ke_local: np.ndarray,
    f_cr: np.ndarray,
) -> np.ndarray:
    """CR 解析的接線剛性（Battini & Pacoste 2002）.

    材料剛性 K_mat と幾何剛性 K_geo を構築する。
    timo_beam3d_cr_tangent_analytical と同一アルゴリズムだが、
    Ke_local を外部から受け取る（ファイバー断面特性を使用）。

    Args:
        kin: CR 運動学の中間結果
        Ke_local: (12, 12) 局所材料剛性（ファイバー断面特性から構築）
        f_cr: (12,) 局所内力ベクトル

    Returns:
        K_T: (12, 12) 全体座標の接線剛性行列
    """
    # B マトリクス構築（dd_cr/du）
    B = np.zeros((12, 12), dtype=float)
    B[6, 0:3] = -kin.e_x_def
    B[6, 6:9] = kin.e_x_def

    S_ex = _skew(kin.e_x_def)
    R_cr_S_ex = kin.R_cr @ S_ex
    dpsi_du1 = (1.0 / kin.L_def) * R_cr_S_ex
    dpsi_du2 = -(1.0 / kin.L_def) * R_cr_S_ex

    T_inv1 = _tangent_operator_inv(kin.theta_def1)
    T_inv2 = _tangent_operator_inv(kin.theta_def2)
    T_s1 = _tangent_operator(kin.u_elem[3:6])
    T_s2 = _tangent_operator(kin.u_elem[9:12])

    B[3:6, 0:3] = T_inv1 @ dpsi_du1
    B[3:6, 6:9] = T_inv1 @ dpsi_du2
    B[3:6, 3:6] = T_inv1 @ kin.R_cr @ T_s1
    B[9:12, 0:3] = T_inv2 @ dpsi_du1
    B[9:12, 6:9] = T_inv2 @ dpsi_du2
    B[9:12, 9:12] = T_inv2 @ kin.R_cr @ T_s2

    # 材料接線: K_mat = T_cr^T @ Ke_local @ B
    K_mat = kin.T_cr.T @ Ke_local @ B

    # 幾何剛性: K_geo
    K_geo = np.zeros((12, 12), dtype=float)
    for blk in range(4):
        f_blk = f_cr[3 * blk : 3 * blk + 3]
        if np.linalg.norm(f_blk) < 1e-30:
            continue
        RtSf = kin.R_cr.T @ _skew(f_blk)
        K_geo[3 * blk : 3 * blk + 3, 0:3] += RtSf @ dpsi_du1
        K_geo[3 * blk : 3 * blk + 3, 6:9] += RtSf @ dpsi_du2

    K_T = K_mat + K_geo
    return 0.5 * (K_T + K_T.T)


# =====================================================================
# Process
# =====================================================================


class StrandFiberBeamProcess(
    AbstractProcess[StrandFiberBeamConfig, StrandFiberBeamResult],
):
    """ファイバー断面 CR 梁要素 Process.

    CR 定式化で大回転を処理し、材料非線形性はファイバー断面積分で評価。
    弾性材料では既存の線形 Timoshenko 梁と一致する。

    実装方針:
      1. CR 運動学で d_cr を抽出
      2. d_cr から断面ひずみ (ε0, κ_y, κ_z) を計算
      3. FiberSectionIntegratorProcess で断面応答を取得
      4. 有効断面特性 (EA, EI_y, EI_z) から Timoshenko 剛性を構築
      5. f_cr = Ke_local @ d_cr → T_cr^T @ f_cr で全体座標へ

    設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md Phase F4
    """

    meta = ProcessMeta(
        name="StrandFiberBeam",
        module="elements",
        version="1.0.0",
        document_path="../docs/fiber_beam_strand.md",
    )
    uses: list = [FiberSectionIntegratorProcess]

    def process(
        self,
        input_data: StrandFiberBeamConfig,
    ) -> StrandFiberBeamResult:
        """ファイバー梁要素の内力・接線剛性を計算."""
        cfg = input_data

        # --- 1. CR 運動学 ---
        kin = _extract_cr_kinematics(cfg.coords_init, cfg.u_elem, cfg.v_ref)

        # --- 2. 断面ひずみ抽出 ---
        eps0 = kin.d_cr[6] / kin.L_0
        kappa_y = (kin.d_cr[10] - kin.d_cr[4]) / kin.L_0
        kappa_z = (kin.d_cr[11] - kin.d_cr[5]) / kin.L_0

        # --- 3. ファイバー断面積分 ---
        integrator = FiberSectionIntegratorProcess()
        int_cfg = FiberIntegratorConfig(
            eps0=eps0,
            kappa_y=kappa_y,
            kappa_z=kappa_z,
            section=cfg.section,
            material=cfg.material,
            state=cfg.section_state,
        )
        int_result = integrator.process(int_cfg)

        # --- 4. 有効断面特性から Timoshenko 剛性構築 ---
        EA_eff = int_result.C_section[0, 0]
        EI_y_eff = int_result.C_section[1, 1]
        EI_z_eff = int_result.C_section[2, 2]
        GJ = cfg.G * cfg.J

        # 断面積は直径から計算（せん断剛性用）
        A = math.pi * (cfg.section.diameter / 2.0) ** 2
        ky = cfg.kappa_y if cfg.kappa_y > 0 else 1.0
        kz = cfg.kappa_z if cfg.kappa_z > 0 else ky
        kGA_y = ky * cfg.G * A
        kGA_z = kz * cfg.G * A

        Ke_local = _timo_ke_from_section(EA_eff, EI_y_eff, EI_z_eff, GJ, kGA_y, kGA_z, kin.L_0)

        # --- 5. 内力（f_cr = Ke_local @ d_cr → T_cr^T で全体座標へ）---
        f_cr = Ke_local @ kin.d_cr
        f_global = kin.T_cr.T @ f_cr

        # --- 6. 接線剛性（Battini & Pacoste 解析的接線: 材料 + 幾何）---
        K_global = _compute_cr_tangent(kin, Ke_local, f_cr)

        return StrandFiberBeamResult(
            f_int=f_global,
            K_elem=K_global,
            section_state_new=int_result.state_new,
        )
