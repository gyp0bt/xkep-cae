"""シース-素線/被膜 有限滑り接触モジュール (Stage S3).

撚線の最外層素線とシース内面の接触を表現する。
素線がシース内面を周方向・軸方向に滑る「有限滑り」を扱う。

== 接触モデル ==

各最外層素線 i に対して z 断面ごとに1接触点を定義:

  法線方向: 径方向（中心→素線方向）
  接線方向1: 周方向（θ増加方向）
  接線方向2: 軸方向（z方向）

ギャップ:
  g_i = r_inner(θ_i) - (|r_i| + r_eff)
  r_eff = wire_radius (+ coating.thickness)

  g < 0 → 貫入（接触力が発生）
  g >= 0 → 非接触

法線力:
  ペナルティ法: F_n = k_pen * max(0, -g)

  コンプライアンス行列 C を使う場合:
    δr = C @ F_contact (N×N 行列)
    F_contact = C^{-1} @ (-g)  （ただし g < 0 の成分のみ）

摩擦力:
  既存の friction_return_mapping のパターンを踏襲:
  - 弾性予測 → Coulomb 条件判定 → stick/slip → return mapping
  - 周方向 + 軸方向の2成分

θ再配置:
  変形後の素線中心座標からθを再計算: θ_i = atan2(y_i, x_i)
  θが大きく変化した場合はコンプライアンス行列を再構築。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xkep_cae.mesh.twisted_wire import CoatingModel, SheathModel, TwistedWireMesh


@dataclass
class SheathContactPoint:
    """シース-素線間の1接触点の状態.

    Attributes:
        strand_id: 対応する素線ID
        node_indices: この素線の節点インデックス配列（z断面代表点用）
        theta: 現在の接触角度 [rad]
        theta_ref: 参照配置での接触角度 [rad]
        gap: 径方向ギャップ（正=非接触, 負=貫入）[m]
        p_n: 法線（径方向）接触力 [N]（圧縮正）
        z_t: 摩擦履歴ベクトル (2,) [周方向, 軸方向]
        k_pen: 法線ペナルティ剛性
        k_t: 接線ペナルティ剛性
        stick: stick状態フラグ
        active: 接触活性フラグ
        dissipation: 散逸エネルギー増分
    """

    strand_id: int
    node_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    theta: float = 0.0
    theta_ref: float = 0.0
    gap: float = 0.0
    p_n: float = 0.0
    z_t: np.ndarray = field(default_factory=lambda: np.zeros(2))
    k_pen: float = 0.0
    k_t: float = 0.0
    stick: bool = True
    active: bool = False
    dissipation: float = 0.0

    def copy(self) -> SheathContactPoint:
        """深いコピーを返す."""
        return SheathContactPoint(
            strand_id=self.strand_id,
            node_indices=self.node_indices.copy(),
            theta=self.theta,
            theta_ref=self.theta_ref,
            gap=self.gap,
            p_n=self.p_n,
            z_t=self.z_t.copy(),
            k_pen=self.k_pen,
            k_t=self.k_t,
            stick=self.stick,
            active=self.active,
            dissipation=self.dissipation,
        )


@dataclass
class SheathContactConfig:
    """シース接触設定.

    Attributes:
        k_pen: 法線ペナルティ剛性 [N/m]
        k_t_ratio: 接線/法線ペナルティ比
        mu: 摩擦係数
        theta_rebuild_tol: θ変化がこの閾値を超えたらC行列を再構築 [rad]
        use_compliance_matrix: True でコンプライアンス行列ベース、False で純ペナルティ
    """

    k_pen: float = 1e6
    k_t_ratio: float = 0.5
    mu: float = 0.3
    theta_rebuild_tol: float = 0.05  # ~3度
    use_compliance_matrix: bool = False


@dataclass
class SheathContactManager:
    """シース-素線接触の管理.

    最外層素線とシース内面の接触を管理する。
    各素線に対して1つの SheathContactPoint を保持する。

    Attributes:
        points: 接触点リスト（最外層素線1本ごと）
        config: 接触設定
        compliance_matrix: コンプライアンス行列 C (N×N)。None ならペナルティのみ
        r_inner_func: θ → シース内面半径のマッピング関数
        r_eff: 素線の有効半径（素線半径 + 被膜厚さ）
        sheath_r_inner_base: シース基本内径
    """

    points: list[SheathContactPoint] = field(default_factory=list)
    config: SheathContactConfig = field(default_factory=SheathContactConfig)
    compliance_matrix: np.ndarray | None = None
    _r_inner_profile_theta: np.ndarray | None = None
    _r_inner_profile_r: np.ndarray | None = None
    r_eff: float = 0.0
    sheath_r_inner_base: float = 0.0

    @property
    def n_points(self) -> int:
        """接触点数."""
        return len(self.points)

    @property
    def n_active(self) -> int:
        """活性接触点数."""
        return sum(1 for p in self.points if p.active)

    def get_contact_angles(self) -> np.ndarray:
        """現在の全接触角度を返す."""
        return np.array([p.theta for p in self.points])

    def get_gaps(self) -> np.ndarray:
        """全ギャップを返す."""
        return np.array([p.gap for p in self.points])

    def get_normal_forces(self) -> np.ndarray:
        """全法線力を返す."""
        return np.array([p.p_n for p in self.points])


def build_sheath_contact_manager(
    mesh: TwistedWireMesh,
    sheath: SheathModel,
    *,
    coating: CoatingModel | None = None,
    config: SheathContactConfig | None = None,
) -> SheathContactManager:
    """撚線メッシュ + シースモデルからシース接触マネージャを構築する.

    最外層素線ごとに SheathContactPoint を1つ作成し、
    初期配置でのθと内面プロファイルを設定する。

    Args:
        mesh: 撚線メッシュ
        sheath: シースモデル
        coating: 被膜モデル（被膜がある場合）
        config: 接触設定（None でデフォルト）

    Returns:
        SheathContactManager インスタンス
    """
    from xkep_cae.mesh.twisted_wire import (
        compute_inner_surface_profile,
        outermost_layer,
        sheath_inner_radius,
    )

    if config is None:
        config = SheathContactConfig(mu=sheath.mu)

    # 有効素線半径
    r_eff = mesh.wire_radius
    if coating is not None:
        r_eff += coating.thickness

    # シース基本内径
    r_inner_base = sheath_inner_radius(mesh, sheath, coating=coating)

    # 内面プロファイル
    theta_profile, r_profile = compute_inner_surface_profile(mesh, n_theta=360, coating=coating)

    # 最外層素線
    outer = outermost_layer(mesh)
    outer_infos = [info for info in mesh.strand_infos if info.layer == outer]

    # 接触点を生成
    points = []
    for info in outer_infos:
        node_start, node_end = mesh.strand_node_ranges[info.strand_id]
        pt = SheathContactPoint(
            strand_id=info.strand_id,
            node_indices=np.arange(node_start, node_end),
            theta=info.angle_offset,
            theta_ref=info.angle_offset,
            k_pen=config.k_pen,
            k_t=config.k_pen * config.k_t_ratio,
        )
        points.append(pt)

    manager = SheathContactManager(
        points=points,
        config=config,
        r_eff=r_eff,
        sheath_r_inner_base=r_inner_base,
    )
    manager._r_inner_profile_theta = theta_profile
    manager._r_inner_profile_r = r_profile + sheath.clearance

    return manager


def compute_strand_theta(
    node_coords: np.ndarray,
    node_indices: np.ndarray,
) -> float:
    """素線節点の現在位置から代表接触角度θを計算する.

    z=0 端（最初のノード）の (x, y) 座標からθを推定する。
    ヘリカル配置では z 位置によってθが異なるため、
    初期位相角 angle_offset と整合するz=0端を使用する。

    Args:
        node_coords: (n_nodes, 3) 全節点座標
        node_indices: この素線の節点インデックス

    Returns:
        θ [rad] ∈ [0, 2π)
    """
    coords = node_coords[node_indices]  # (n_strand_nodes, 3)

    # z=0 端（最小z値のノード）を使用
    z_vals = coords[:, 2]
    z0_idx = int(np.argmin(z_vals))

    x, y = coords[z0_idx, 0], coords[z0_idx, 1]
    theta = math.atan2(y, x)
    if theta < 0:
        theta += 2.0 * math.pi
    return theta


def update_contact_angles(
    manager: SheathContactManager,
    node_coords: np.ndarray,
) -> np.ndarray:
    """全接触点のθを変形後座標から再計算する.

    Args:
        manager: シース接触マネージャ
        node_coords: (n_nodes, 3) 現在の節点座標

    Returns:
        delta_theta: (N,) 各点のθ変化量 [rad]
    """
    delta = np.zeros(manager.n_points)
    for i, pt in enumerate(manager.points):
        theta_new = compute_strand_theta(node_coords, pt.node_indices)
        delta[i] = _angle_diff(theta_new, pt.theta)
        pt.theta = theta_new
    return delta


def _angle_diff(a: float, b: float) -> float:
    """角度差 (a - b) を [-π, π) の範囲に正規化する."""
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def evaluate_sheath_inner_radius(
    manager: SheathContactManager,
    theta: float,
) -> float:
    """シース内面半径を角度θで評価する.

    内面プロファイル（Fourier近似）から線形補間で取得。

    Args:
        manager: シース接触マネージャ
        theta: 角度 [rad]

    Returns:
        r_inner [m]
    """
    tp = manager._r_inner_profile_theta
    rp = manager._r_inner_profile_r
    if tp is None or rp is None:
        return manager.sheath_r_inner_base

    # 周期的補間
    theta_mod = theta % (2.0 * math.pi)
    tp_ext = np.concatenate([tp - 2.0 * math.pi, tp, tp + 2.0 * math.pi])
    rp_ext = np.concatenate([rp, rp, rp])
    return float(np.interp(theta_mod, tp_ext, rp_ext))


def compute_sheath_gaps(
    manager: SheathContactManager,
    node_coords: np.ndarray,
) -> np.ndarray:
    """全接触点の径方向ギャップを計算する.

    g_i = r_inner(θ_i) - (|r_i| + r_eff)
    正=非接触、負=貫入

    Args:
        manager: シース接触マネージャ
        node_coords: (n_nodes, 3) 現在の節点座標

    Returns:
        gaps: (N,) ギャップ配列 [m]
    """
    gaps = np.zeros(manager.n_points)
    for i, pt in enumerate(manager.points):
        # 代表節点（z=0端）の径方向位置
        coords = node_coords[pt.node_indices]
        z_vals = coords[:, 2]
        z0_idx = int(np.argmin(z_vals))
        x, y = coords[z0_idx, 0], coords[z0_idx, 1]
        r_strand = math.sqrt(x**2 + y**2)

        # シース内面半径
        r_inner = evaluate_sheath_inner_radius(manager, pt.theta)

        gap = r_inner - (r_strand + manager.r_eff)
        pt.gap = gap
        gaps[i] = gap

    return gaps


def evaluate_normal_forces(
    manager: SheathContactManager,
) -> np.ndarray:
    """全接触点の法線（径方向）接触力を評価する.

    ペナルティ法: p_n = k_pen * max(0, -g)

    コンプライアンス行列モード（use_compliance_matrix=True）:
    貫入量ベクトル δ = max(0, -g) を構成し、
    K = C^{-1} として F = K @ δ

    Args:
        manager: シース接触マネージャ

    Returns:
        forces: (N,) 法線力配列 [N]（圧縮正）
    """
    N = manager.n_points
    forces = np.zeros(N)
    penetrations = np.zeros(N)

    for i, pt in enumerate(manager.points):
        penetrations[i] = max(0.0, -pt.gap)

    if manager.config.use_compliance_matrix and manager.compliance_matrix is not None:
        # コンプライアンス行列ベース
        C = manager.compliance_matrix
        # K = C^{-1}, F = K @ δ
        # ただし非貫入点は力ゼロを保証するため反復的に解く
        active_mask = penetrations > 0.0
        if np.any(active_mask):
            idx = np.where(active_mask)[0]
            C_sub = C[np.ix_(idx, idx)]
            delta_sub = penetrations[idx]
            try:
                forces_sub = np.linalg.solve(C_sub, delta_sub)
                # 引張力（負値）が出たら0にクリップ
                forces_sub = np.maximum(forces_sub, 0.0)
                forces[idx] = forces_sub
            except np.linalg.LinAlgError:
                # 特異行列の場合はペナルティにフォールバック
                for i, pt in enumerate(manager.points):
                    forces[i] = pt.k_pen * penetrations[i]
    else:
        # 純ペナルティ
        for i, pt in enumerate(manager.points):
            forces[i] = pt.k_pen * penetrations[i]

    # 状態更新
    for i, pt in enumerate(manager.points):
        pt.p_n = forces[i]
        pt.active = forces[i] > 0.0

    return forces


def build_contact_frame_sheath(
    theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """シース接触のローカルフレームを構築する.

    法線: 径方向外向き（中心→素線方向）
    接線1: 周方向（θ増加方向）
    接線2: 軸方向（z方向）

    Args:
        theta: 接触角度 [rad]

    Returns:
        (normal, tangent1, tangent2) 各(3,)ベクトル
    """
    normal = np.array([math.cos(theta), math.sin(theta), 0.0])
    tangent1 = np.array([-math.sin(theta), math.cos(theta), 0.0])  # 周方向
    tangent2 = np.array([0.0, 0.0, 1.0])  # 軸方向
    return normal, tangent1, tangent2


def compute_tangential_displacement_sheath(
    pt: SheathContactPoint,
    node_coords_cur: np.ndarray,
    node_coords_ref: np.ndarray,
) -> np.ndarray:
    """シース接触点での接線方向相対変位増分を計算する.

    素線代表点のΔu（周方向, 軸方向成分）をシース面上の変位として扱う。
    シースは剛体として固定されている想定。

    Args:
        pt: 接触点
        node_coords_cur: (n_nodes, 3) 現在の節点座標
        node_coords_ref: (n_nodes, 3) 参照節点座標

    Returns:
        delta_ut: (2,) [周方向変位, 軸方向変位]
    """
    # 代表節点（z=0端）
    coords_cur = node_coords_cur[pt.node_indices]
    coords_ref = node_coords_ref[pt.node_indices]
    z_vals = coords_ref[:, 2]
    z0_idx = int(np.argmin(z_vals))
    mid_idx = z0_idx

    du = coords_cur[mid_idx] - coords_ref[mid_idx]  # (3,)

    # 接触フレーム
    _, t1, t2 = build_contact_frame_sheath(pt.theta)

    delta_ut = np.array([float(du @ t1), float(du @ t2)])
    return delta_ut


def sheath_friction_return_mapping(
    pt: SheathContactPoint,
    delta_ut: np.ndarray,
    mu: float,
) -> np.ndarray:
    """シース接触での Coulomb 摩擦 return mapping.

    既存の friction_return_mapping と同じアルゴリズム:
      1. q_trial = z_t_old + k_t * Δu_t
      2. Coulomb: ||q_trial|| ≤ μ * p_n
      3. stick/slip 分岐 + radial return

    Args:
        pt: シース接触点
        delta_ut: (2,) 接線相対変位増分 [周方向, 軸方向]
        mu: 有効摩擦係数

    Returns:
        q: (2,) 接線摩擦力 [周方向, 軸方向]
    """
    if not pt.active or pt.p_n <= 0.0 or mu <= 0.0:
        pt.stick = True
        pt.dissipation = 0.0
        return np.zeros(2)

    z_t_old = pt.z_t.copy()
    k_t = pt.k_t
    p_n = pt.p_n

    # 弾性予測
    q_trial = z_t_old + k_t * delta_ut
    q_trial_norm = float(np.linalg.norm(q_trial))

    # Coulomb 条件
    f_yield = mu * p_n

    if q_trial_norm <= f_yield:
        # stick
        q = q_trial.copy()
        pt.z_t = q.copy()
        pt.stick = True
    else:
        # slip → radial return
        q = f_yield * q_trial / q_trial_norm
        pt.z_t = q.copy()
        pt.stick = False

    # 散逸増分
    pt.dissipation = float(q @ delta_ut)

    return q


def evaluate_sheath_contact(
    manager: SheathContactManager,
    node_coords: np.ndarray,
    node_coords_ref: np.ndarray | None = None,
) -> dict:
    """シース接触の一括評価（θ更新 + ギャップ + 法線力 + 摩擦力）.

    Args:
        manager: シース接触マネージャ
        node_coords: (n_nodes, 3) 現在の節点座標
        node_coords_ref: (n_nodes, 3) 参照座標（摩擦計算用、None なら摩擦なし）

    Returns:
        dict:
            delta_theta: (N,) θ変化量
            gaps: (N,) ギャップ
            normal_forces: (N,) 法線力
            friction_forces: (N, 2) 摩擦力 [[周方向, 軸方向], ...]
            n_active: 活性接触点数
    """
    # 1. θ再配置
    delta_theta = update_contact_angles(manager, node_coords)

    # 2. ギャップ計算
    gaps = compute_sheath_gaps(manager, node_coords)

    # 3. 法線力評価
    normal_forces = evaluate_normal_forces(manager)

    # 4. 摩擦力評価
    N = manager.n_points
    friction_forces = np.zeros((N, 2))
    mu = manager.config.mu

    if node_coords_ref is not None and mu > 0.0:
        for i, pt in enumerate(manager.points):
            if pt.active:
                delta_ut = compute_tangential_displacement_sheath(pt, node_coords, node_coords_ref)
                q = sheath_friction_return_mapping(pt, delta_ut, mu)
                friction_forces[i] = q

    return {
        "delta_theta": delta_theta,
        "gaps": gaps,
        "normal_forces": normal_forces,
        "friction_forces": friction_forces,
        "n_active": manager.n_active,
    }


def assemble_sheath_forces(
    manager: SheathContactManager,
    node_coords: np.ndarray,
    ndof_per_node: int = 6,
) -> np.ndarray:
    """シース接触力を全体力ベクトルに組み立てる.

    各接触点の法線力 + 摩擦力を、対応する素線代表節点の
    並進DOFに加算する。

    法線力: 径方向内向き（素線を中心方向に押す）
    摩擦力: 接触フレームの接線方向

    Args:
        manager: シース接触マネージャ
        node_coords: (n_nodes, 3) 現在の節点座標
        ndof_per_node: 1節点あたりのDOF数

    Returns:
        f_contact: (n_dof_total,) 接触力ベクトル
    """
    n_nodes = node_coords.shape[0]
    n_dof = n_nodes * ndof_per_node
    f = np.zeros(n_dof)

    for pt in manager.points:
        if not pt.active:
            continue

        # 代表節点（z=0端）
        coords = node_coords[pt.node_indices]
        z_vals = coords[:, 2]
        z0_idx = int(np.argmin(z_vals))
        node_id = pt.node_indices[z0_idx]

        # 接触フレーム
        normal, t1, t2 = build_contact_frame_sheath(pt.theta)

        # 法線力: 素線を内向きに押す（normal は外向きなので -normal 方向）
        f_normal = -pt.p_n * normal

        # 摩擦力: 接線方向
        f_friction = pt.z_t[0] * t1 + pt.z_t[1] * t2

        # 合力
        f_total = f_normal + f_friction

        # 並進DOFに加算
        dof_start = node_id * ndof_per_node
        f[dof_start : dof_start + 3] += f_total

    return f


def check_theta_rebuild_needed(
    manager: SheathContactManager,
    delta_theta: np.ndarray,
) -> bool:
    """θ変化がコンプライアンス行列の再構築閾値を超えたか判定する.

    Args:
        manager: シース接触マネージャ
        delta_theta: (N,) θ変化量

    Returns:
        True なら再構築が必要
    """
    return bool(np.max(np.abs(delta_theta)) > manager.config.theta_rebuild_tol)


def rebuild_compliance_matrix(
    manager: SheathContactManager,
    mesh: TwistedWireMesh,
    sheath: SheathModel,
    *,
    coating: CoatingModel | None = None,
    n_modes: int | None = None,
) -> np.ndarray:
    """現在のθ配置でコンプライアンス行列を再構築する.

    Args:
        manager: シース接触マネージャ
        mesh: 撚線メッシュ
        sheath: シースモデル
        coating: 被膜モデル
        n_modes: Fourier打ち切りモード数

    Returns:
        C: (N, N) 更新されたコンプライアンス行列
    """
    from xkep_cae.mesh.ring_compliance import build_variable_thickness_compliance_matrix

    N = manager.n_points
    contact_angles = manager.get_contact_angles()

    # 各接触点での内面半径
    r_inner_at_contacts = np.array(
        [evaluate_sheath_inner_radius(manager, th) for th in contact_angles]
    )

    # シース外径
    r_outer = np.max(r_inner_at_contacts) + sheath.thickness

    C = build_variable_thickness_compliance_matrix(
        N,
        contact_angles,
        r_inner_at_contacts,
        r_outer,
        sheath.E,
        sheath.nu,
        n_modes=n_modes,
    )

    manager.compliance_matrix = C
    return C
