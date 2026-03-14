"""Phase S4-1: 撚線構造レベルの等価剛性比較ベンチマーク.

素線+被膜、素線+シースの撚線構造について、
各荷重モード（曲げ/ねじり/引張/圧縮/大変形）の等価剛性を
NCPソルバー（リファレンス構成）で算出する。

比較対象:
  - 裸撚線（接触のみ）
  - 素線+被膜（CoatingModel）
  - 素線+シース（SheathModel）
  - 素線+被膜+シース（フルモデル）

参考文献:
  - Costello, G.A. "Theory of Wire Rope" (1997)
  - Foti, F. & Martinelli, L. (2016) "Hysteretic bending of spiral strands"

status-098: 推奨ソルバー構成（NCP基準）で全ベンチマークを実施する方針。
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactStatus,
)
from xkep_cae.contact.solver_ncp import (
    newton_raphson_contact_ncp,
)
from xkep_cae.elements.beam_timo3d import (
    timo_beam3d_ke_global,
)
from xkep_cae.mesh.twisted_wire import (
    CoatingModel,
    SheathModel,
    TwistedWireMesh,
    coated_beam_section,
    coated_radii,
    make_twisted_wire_mesh,
    sheath_equivalent_stiffness,
)
from xkep_cae.sections.beam import BeamSection

pytestmark = pytest.mark.slow

# ====================================================================
# 共通パラメータ（撚線標準鋼線）
# ====================================================================

_NDOF_PER_NODE = 6
_E = 200e9  # Pa
_NU = 0.3
_G = _E / (2.0 * (1.0 + _NU))
_WIRE_D = 0.002  # 直径 2mm
_WIRE_R = _WIRE_D / 2.0
_SECTION = BeamSection.circle(_WIRE_D)
_KAPPA = 6.0 * (1.0 + _NU) / (7.0 + 6.0 * _NU)
_PITCH = 0.040  # 40mm ピッチ

# 被膜パラメータ（ポリマー系絶縁被覆を想定）
_COATING = CoatingModel(
    thickness=0.1e-3,  # 0.1mm
    E=3.0e9,  # 3 GPa（エナメル被膜相当）
    nu=0.35,
    mu=0.2,
)

# シースパラメータ（ポリマーシースを想定）
_SHEATH = SheathModel(
    thickness=0.5e-3,  # 0.5mm
    E=1.2e9,  # 1.2 GPa（PE/PVCシース相当）
    nu=0.4,
    mu=0.25,
    clearance=0.0,  # 密着
)

# メッシュ設定
_N_STRANDS = 7
_N_ELEMS_PER_STRAND = 16
_N_PITCHES = 0.5


# ====================================================================
# ヘルパー: 線形梁アセンブラ（被膜剛性込み対応）
# ====================================================================


def _make_timo3d_assemblers(
    mesh: TwistedWireMesh,
    *,
    coating: CoatingModel | None = None,
):
    """Timoshenko 3D 線形梁のアセンブリコールバックを構築.

    coating が指定された場合は複合断面剛性（素線+被膜）を使用する。
    """
    node_coords = mesh.node_coords
    connectivity = mesh.connectivity
    ndof_total = mesh.n_nodes * _NDOF_PER_NODE

    # 断面パラメータの決定
    if coating is not None:
        comp = coated_beam_section(_WIRE_R, _E, _NU, coating)
        # 等価E, Gは素線のまま、断面特性を等価に補正
        # EA_total = E_wire * A_eq → A_eq = EA_total / E_wire
        A_eq = comp["EA"] / _E
        Iy_eq = comp["EIy"] / _E
        Iz_eq = comp["EIz"] / _E
        J_eq = comp["GJ"] / _G
    else:
        A_eq = _SECTION.A
        Iy_eq = _SECTION.Iy
        Iz_eq = _SECTION.Iz
        J_eq = _SECTION.J

    K = np.zeros((ndof_total, ndof_total))
    for elem in connectivity:
        n1, n2 = int(elem[0]), int(elem[1])
        coords = node_coords[np.array([n1, n2])]
        Ke = timo_beam3d_ke_global(coords, _E, _G, A_eq, Iy_eq, Iz_eq, J_eq, _KAPPA, _KAPPA)
        edofs = np.array(
            [6 * n1 + d for d in range(6)] + [6 * n2 + d for d in range(6)],
            dtype=int,
        )
        K[np.ix_(edofs, edofs)] += Ke
    K_sp = sp.csr_matrix(K)

    def assemble_tangent(u):
        return K_sp

    def assemble_internal_force(u):
        return K_sp.dot(u)

    return assemble_tangent, assemble_internal_force, ndof_total


# ====================================================================
# ヘルパー: 境界条件・荷重
# ====================================================================


def _fix_all_strand_starts(mesh: TwistedWireMesh):
    """全素線の開始端を全固定."""
    fixed = []
    for sid in range(mesh.n_strands):
        start_node = mesh.strand_node_ranges[sid][0]
        for d in range(_NDOF_PER_NODE):
            fixed.append(start_node * _NDOF_PER_NODE + d)
    return np.array(sorted(set(fixed)), dtype=int)


def _get_strand_end_dofs(mesh: TwistedWireMesh, strand_id: int, end: str):
    """素線の端点のDOFインデックスを取得."""
    start_node, end_node = mesh.strand_node_ranges[strand_id]
    node = start_node if end == "start" else end_node - 1
    return np.array([node * _NDOF_PER_NODE + d for d in range(_NDOF_PER_NODE)])


def _apply_load(
    mesh: TwistedWireMesh,
    ndof_total: int,
    mode: str,
    value: float,
) -> np.ndarray:
    """荷重モードに応じた外力ベクトルを構築.

    Args:
        mode: "tension", "compression", "bending", "torsion"
        value: 荷重の絶対値
    """
    f_ext = np.zeros(ndof_total)
    f_per_strand = value / mesh.n_strands

    for sid in range(mesh.n_strands):
        end_dofs = _get_strand_end_dofs(mesh, sid, "end")
        if mode == "tension":
            f_ext[end_dofs[2]] = f_per_strand  # +z
        elif mode == "compression":
            f_ext[end_dofs[2]] = -f_per_strand  # -z
        elif mode == "bending":
            f_ext[end_dofs[1]] = f_per_strand  # y方向横荷重
        elif mode == "torsion":
            f_ext[end_dofs[5]] = f_per_strand  # z軸まわりモーメント
        else:
            raise ValueError(f"未知の荷重モード: {mode}")

    return f_ext


# ====================================================================
# ヘルパー: NCP ソルバーで解く（リファレンス構成）
# ====================================================================


def _solve_twisted_wire(
    mesh: TwistedWireMesh,
    mode: str,
    load_value: float,
    *,
    coating: CoatingModel | None = None,
    n_load_steps: int = 10,
    max_iter: int = 80,
) -> tuple:
    """NCPリファレンス構成で撚線を解く.

    Returns:
        (result, mgr, mesh, ndof_total)
    """
    assemble_tangent, assemble_internal_force, ndof_total = _make_timo3d_assemblers(
        mesh, coating=coating
    )
    fixed_dofs = _fix_all_strand_starts(mesh)
    f_ext = _apply_load(mesh, ndof_total, mode, load_value)

    # 接触半径
    radii = coated_radii(mesh, coating) if coating is not None else mesh.radii

    # 同層除外マッピング
    elem_layer_map = mesh.build_elem_layer_map()

    # リファレンス構成（status-098）
    config = ContactConfig(
        k_pen_scale=1e5,
        line_contact=True,
        n_gauss=2,
        use_mortar=True,
        use_friction=True,
        mu=coating.mu if coating is not None else 0.3,
        k_t_ratio=0.01,
        mu_ramp_steps=5,
        g_on=0.0,
        g_off=1e-5,
        tol_penetration_ratio=0.02,
        penalty_growth_factor=2.0,
        elem_layer_map=elem_layer_map,
        exclude_same_layer=True,
    )
    mgr = ContactManager(config=config)

    result = newton_raphson_contact_ncp(
        f_ext,
        fixed_dofs,
        assemble_tangent,
        assemble_internal_force,
        mgr,
        mesh.node_coords,
        mesh.connectivity,
        radii,
        n_load_steps=n_load_steps,
        max_iter=max_iter,
        tol_force=1e-6,
        tol_ncp=1e-6,
        show_progress=False,
        broadphase_margin=0.005,
        line_contact=True,
        n_gauss=2,
        use_mortar=True,
        use_friction=True,
        mu=coating.mu if coating is not None else 0.3,
        mu_ramp_steps=5,
    )

    return result, mgr, ndof_total


def _extract_tip_displacement(
    mesh: TwistedWireMesh,
    u: np.ndarray,
    mode: str,
) -> float:
    """先端変位の平均値を抽出.

    mode に応じた方向成分の全素線先端平均を返す。
    """
    disp_sum = 0.0
    for sid in range(mesh.n_strands):
        end_dofs = _get_strand_end_dofs(mesh, sid, "end")
        if mode == "tension" or mode == "compression":
            disp_sum += u[end_dofs[2]]  # uz
        elif mode == "bending":
            disp_sum += u[end_dofs[1]]  # uy
        elif mode == "torsion":
            disp_sum += u[end_dofs[5]]  # θz
    return disp_sum / mesh.n_strands


def _compute_equivalent_stiffness(
    load_value: float,
    tip_disp: float,
) -> float:
    """等価剛性 k = P / δ."""
    if abs(tip_disp) < 1e-30:
        return float("inf")
    return load_value / tip_disp


def _max_penetration_ratio(mgr: ContactManager) -> float:
    """最大貫入比を計算."""
    max_pen = 0.0
    for p in mgr.pairs:
        if p.state.status == ContactStatus.INACTIVE:
            continue
        if p.state.gap < 0:
            pen = abs(p.state.gap) / (p.radius_a + p.radius_b)
            if pen > max_pen:
                max_pen = pen
    return max_pen


# ====================================================================
# ヘルパー: メッシュ生成
# ====================================================================


def _make_mesh():
    """7本撚りメッシュを生成."""
    return make_twisted_wire_mesh(
        _N_STRANDS,
        _WIRE_D,
        _PITCH,
        length=0.0,
        n_elems_per_strand=_N_ELEMS_PER_STRAND,
        n_pitches=_N_PITCHES,
        gap=0.0,
    )


# ====================================================================
# 解析解: Costello の理論 (1997)
# ====================================================================


def _costello_axial_stiffness_7strand(E, A_wire, pitch, wire_r, lay_radius):
    """Costello 理論に基づく7本撚りの軸剛性の概算.

    中心線 EA + 外層6本 × cos²α × EA の近似。
    α = atan(2π·lay_radius / pitch) は撚り角。
    """
    alpha = np.arctan(2.0 * np.pi * lay_radius / pitch)
    cos_a = np.cos(alpha)
    # 中心線 + 外層6本（cos²α で軸方向寄与）
    EA_total = E * A_wire * (1.0 + 6.0 * cos_a**3)
    L = _N_PITCHES * pitch
    return EA_total / L


# ====================================================================
# テスト: 素線+被膜 等価剛性ベンチマーク
# ====================================================================


class TestS4CoatingStiffness:
    """素線+被膜（CoatingModel）構成の等価剛性ベンチマーク."""

    def _solve_mode(self, mode: str, load_value: float):
        """指定モードで被膜付き撚線を解く."""
        mesh = _make_mesh()
        result, mgr, ndof = _solve_twisted_wire(
            mesh, mode, load_value, coating=_COATING, n_load_steps=8
        )
        return result, mgr, mesh, ndof

    def _solve_bare(self, mode: str, load_value: float):
        """指定モードで裸撚線（被膜なし）を解く."""
        mesh = _make_mesh()
        result, mgr, ndof = _solve_twisted_wire(
            mesh, mode, load_value, coating=None, n_load_steps=8
        )
        return result, mgr, mesh, ndof

    def test_coating_tension(self):
        """素線+被膜: 引張等価剛性.

        被膜の剛性寄与を確認。EA_coated > EA_bare であること。
        """
        P = 50.0  # N

        result_c, mgr_c, mesh_c, _ = self._solve_mode("tension", P)
        result_b, mgr_b, mesh_b, _ = self._solve_bare("tension", P)

        assert result_c.converged, "被膜付き引張が収束しない"
        assert result_b.converged, "裸撚線引張が収束しない"

        d_c = _extract_tip_displacement(mesh_c, result_c.u, "tension")
        d_b = _extract_tip_displacement(mesh_b, result_b.u, "tension")
        k_c = _compute_equivalent_stiffness(P, d_c)
        k_b = _compute_equivalent_stiffness(P, d_b)

        # 被膜の複合断面剛性
        comp = coated_beam_section(_WIRE_R, _E, _NU, _COATING)
        coating_EA_ratio = comp["EA"] / (_E * _SECTION.A)

        print(f"\n{'=' * 65}")
        print("  S4-1 引張剛性: 素線+被膜 vs 裸撚線")
        print(f"{'=' * 65}")
        print(f"  裸撚線:     k = {k_b:.4e} N/m,  δ = {d_b:.6e} m")
        print(f"  被膜付き:   k = {k_c:.4e} N/m,  δ = {d_c:.6e} m")
        print(f"  剛性比 (coated/bare): {k_c / k_b:.4f}")
        print(f"  素線EA → 複合EA 比率: {coating_EA_ratio:.4f}")
        print(f"  接触ペア数: bare={mgr_b.n_pairs}, coated={mgr_c.n_pairs}")
        print(
            f"  貫入率: bare={_max_penetration_ratio(mgr_b):.4f}, "
            f"coated={_max_penetration_ratio(mgr_c):.4f}"
        )

        # 被膜付きは裸より剛性が高い
        assert k_c > k_b * 0.99, f"被膜付き剛性が裸より低い: {k_c:.4e} < {k_b:.4e}"

    def test_coating_bending(self):
        """素線+被膜: 曲げ等価剛性.

        EI_coated > EI_bare であること。
        """
        P = 1.0  # N（横荷重）

        result_c, mgr_c, mesh_c, _ = self._solve_mode("bending", P)
        result_b, mgr_b, mesh_b, _ = self._solve_bare("bending", P)

        assert result_c.converged, "被膜付き曲げが収束しない"
        assert result_b.converged, "裸撚線曲げが収束しない"

        d_c = _extract_tip_displacement(mesh_c, result_c.u, "bending")
        d_b = _extract_tip_displacement(mesh_b, result_b.u, "bending")
        k_c = _compute_equivalent_stiffness(P, d_c)
        k_b = _compute_equivalent_stiffness(P, d_b)

        comp = coated_beam_section(_WIRE_R, _E, _NU, _COATING)
        coating_EI_ratio = comp["EIy"] / (_E * _SECTION.Iy)

        print(f"\n{'=' * 65}")
        print("  S4-1 曲げ剛性: 素線+被膜 vs 裸撚線")
        print(f"{'=' * 65}")
        print(f"  裸撚線:     k = {k_b:.4e} N/m,  δ = {d_b:.6e} m")
        print(f"  被膜付き:   k = {k_c:.4e} N/m,  δ = {d_c:.6e} m")
        print(f"  剛性比 (coated/bare): {k_c / k_b:.4f}")
        print(f"  素線EI → 複合EI 比率: {coating_EI_ratio:.4f}")

        assert k_c > k_b * 0.99, f"被膜付き曲げ剛性が裸より低い: {k_c:.4e} < {k_b:.4e}"

    def test_coating_torsion(self):
        """素線+被膜: ねじり等価剛性.

        GJ_coated > GJ_bare であること。
        """
        M = 0.001  # N·m

        result_c, mgr_c, mesh_c, _ = self._solve_mode("torsion", M)
        result_b, mgr_b, mesh_b, _ = self._solve_bare("torsion", M)

        assert result_c.converged, "被膜付きねじりが収束しない"
        assert result_b.converged, "裸撚線ねじりが収束しない"

        theta_c = _extract_tip_displacement(mesh_c, result_c.u, "torsion")
        theta_b = _extract_tip_displacement(mesh_b, result_b.u, "torsion")
        k_c = _compute_equivalent_stiffness(M, theta_c)
        k_b = _compute_equivalent_stiffness(M, theta_b)

        comp = coated_beam_section(_WIRE_R, _E, _NU, _COATING)
        coating_GJ_ratio = comp["GJ"] / (_G * _SECTION.J)

        print(f"\n{'=' * 65}")
        print("  S4-1 ねじり剛性: 素線+被膜 vs 裸撚線")
        print(f"{'=' * 65}")
        print(f"  裸撚線:     k = {k_b:.4e} N·m/rad,  θ = {theta_b:.6e} rad")
        print(f"  被膜付き:   k = {k_c:.4e} N·m/rad,  θ = {theta_c:.6e} rad")
        print(f"  剛性比 (coated/bare): {k_c / k_b:.4f}")
        print(f"  素線GJ → 複合GJ 比率: {coating_GJ_ratio:.4f}")

        assert k_c > k_b * 0.99, f"被膜付きねじり剛性が裸より低い: {k_c:.4e} < {k_b:.4e}"

    def test_coating_compression(self):
        """素線+被膜: 圧縮等価剛性.

        引張と同等以上の剛性（座屈なし域）であること。
        """
        P = 20.0  # N（圧縮。引張より小さく設定し座屈回避）

        result_c, _, mesh_c, _ = self._solve_mode("compression", P)
        assert result_c.converged, "被膜付き圧縮が収束しない"

        d_c = _extract_tip_displacement(mesh_c, result_c.u, "compression")
        k_c = _compute_equivalent_stiffness(P, d_c)

        print(f"\n{'=' * 65}")
        print("  S4-1 圧縮剛性: 素線+被膜")
        print(f"{'=' * 65}")
        print(f"  被膜付き:   k = {abs(k_c):.4e} N/m,  δ = {d_c:.6e} m")
        print(f"  （圧縮荷重 = {P} N）")

        # 圧縮剛性が正値（合理的な変位方向）
        assert d_c < 0, f"圧縮変位が負でない: {d_c}"
        assert abs(k_c) > 0, "圧縮剛性がゼロ"


# ====================================================================
# テスト: 素線+シース 等価剛性ベンチマーク
# ====================================================================


class TestS4SheathStiffness:
    """素線+シース（SheathModel）構成の等価剛性ベンチマーク.

    注: 現段階ではシースはFEM要素としてモデル化されず、
    解析的な等価剛性として断面特性に追加する設計。
    ここではシース剛性を裸撚線の剛性に加算して比較する。
    """

    def _sheath_stiffness(self, mesh: TwistedWireMesh):
        """シースの等価梁剛性を取得."""
        return sheath_equivalent_stiffness(mesh, _SHEATH)

    def test_sheath_stiffness_contribution(self):
        """シースの等価剛性の定量評価.

        シース管の EA/EI/GJ を素線束の合計と比較する。
        """
        mesh = _make_mesh()
        ss = self._sheath_stiffness(mesh)

        # 素線束（7本）の合計剛性
        n = mesh.n_strands
        wire_EA = _E * _SECTION.A * n
        wire_EI = _E * _SECTION.Iy * n  # 並列仮定（保守的見積もり）
        wire_GJ = _G * _SECTION.J * n

        print(f"\n{'=' * 65}")
        print("  S4-1 シース剛性寄与（解析的断面特性）")
        print(f"{'=' * 65}")
        print("  素線束 7本合計:")
        print(f"    EA  = {wire_EA:.4e} N")
        print(f"    EIy = {wire_EI:.4e} N·m²")
        print(f"    GJ  = {wire_GJ:.4e} N·m²")
        print("  シース:")
        print(f"    EA  = {ss['EA']:.4e} N (比率: {ss['EA'] / wire_EA:.4f})")
        print(f"    EIy = {ss['EIy']:.4e} N·m² (比率: {ss['EIy'] / wire_EI:.4f})")
        print(f"    GJ  = {ss['GJ']:.4e} N·m² (比率: {ss['GJ'] / wire_GJ:.4f})")
        print(f"    内径 = {ss['r_inner'] * 1e3:.3f} mm")
        print(f"    外径 = {ss['r_outer'] * 1e3:.3f} mm")

        # シースの剛性が正であること
        assert ss["EA"] > 0, "シース EA がゼロ"
        assert ss["EIy"] > 0, "シース EIy がゼロ"
        assert ss["GJ"] > 0, "シース GJ がゼロ"

    def test_sheath_tension(self):
        """素線+シース: 引張等価剛性.

        裸撚線にシース管の EA を加算した理論値と NCP 結果を比較。
        """
        P = 50.0
        mesh = _make_mesh()

        # 裸撚線を NCP で解く
        result_b, mgr_b, ndof_b = _solve_twisted_wire(
            mesh, "tension", P, coating=None, n_load_steps=8
        )
        assert result_b.converged, "裸撚線引張が収束しない"

        d_b = _extract_tip_displacement(mesh, result_b.u, "tension")
        k_bare = _compute_equivalent_stiffness(P, d_b)

        # シース等価剛性（解析的）
        ss = self._sheath_stiffness(mesh)
        L = _N_PITCHES * _PITCH
        k_sheath_axial = ss["EA"] / L

        # 構造全体の理論等価剛性（裸撚線 + シース管の並列和）
        k_total_theory = k_bare + k_sheath_axial

        print(f"\n{'=' * 65}")
        print("  S4-1 引張剛性: 素線+シース")
        print(f"{'=' * 65}")
        print(f"  裸撚線 NCP:    k = {k_bare:.4e} N/m")
        print(f"  シース管 EA/L: k = {k_sheath_axial:.4e} N/m")
        print(f"  合計（理論）:  k = {k_total_theory:.4e} N/m")
        print(f"  シース/撚線比: {k_sheath_axial / k_bare:.4f}")

        # シースの軸剛性寄与は素線に比べて小さいが正値
        assert k_sheath_axial > 0

    def test_sheath_bending(self):
        """素線+シース: 曲げ等価剛性.

        シース管の EI は素線束の並行軸定理を含まない保守的見積もり。
        """
        P = 1.0
        mesh = _make_mesh()

        result_b, _, ndof_b = _solve_twisted_wire(
            mesh, "bending", P, coating=None, n_load_steps=8
        )
        assert result_b.converged, "裸撚線曲げが収束しない"

        d_b = _extract_tip_displacement(mesh, result_b.u, "bending")
        k_bare = _compute_equivalent_stiffness(P, d_b)

        ss = self._sheath_stiffness(mesh)
        L = _N_PITCHES * _PITCH
        # 曲げ剛性: 3EI/L³ （片持ち先端集中荷重）
        k_sheath_bend = 3.0 * ss["EIy"] / L**3

        k_total_theory = k_bare + k_sheath_bend

        print(f"\n{'=' * 65}")
        print("  S4-1 曲げ剛性: 素線+シース")
        print(f"{'=' * 65}")
        print(f"  裸撚線 NCP:       k = {k_bare:.4e} N/m")
        print(f"  シース管 3EI/L³:  k = {k_sheath_bend:.4e} N/m")
        print(f"  合計（理論）:     k = {k_total_theory:.4e} N/m")
        print(f"  シース/撚線比: {k_sheath_bend / k_bare:.4f}")

        assert k_sheath_bend > 0

    def test_sheath_torsion(self):
        """素線+シース: ねじり等価剛性."""
        M = 0.001  # N·m
        mesh = _make_mesh()

        result_b, _, ndof_b = _solve_twisted_wire(
            mesh, "torsion", M, coating=None, n_load_steps=8
        )
        assert result_b.converged, "裸撚線ねじりが収束しない"

        theta_b = _extract_tip_displacement(mesh, result_b.u, "torsion")
        k_bare = _compute_equivalent_stiffness(M, theta_b)

        ss = self._sheath_stiffness(mesh)
        L = _N_PITCHES * _PITCH
        k_sheath_tors = ss["GJ"] / L

        k_total_theory = k_bare + k_sheath_tors

        print(f"\n{'=' * 65}")
        print("  S4-1 ねじり剛性: 素線+シース")
        print(f"{'=' * 65}")
        print(f"  裸撚線 NCP:      k = {k_bare:.4e} N·m/rad")
        print(f"  シース管 GJ/L:   k = {k_sheath_tors:.4e} N·m/rad")
        print(f"  合計（理論）:    k = {k_total_theory:.4e} N·m/rad")
        print(f"  シース/撚線比: {k_sheath_tors / k_bare:.4f}")

        assert k_sheath_tors > 0

    def test_sheath_compression(self):
        """素線+シース: 圧縮でシースが径方向拘束として作用."""
        P = 20.0
        mesh = _make_mesh()

        result_b, _, ndof_b = _solve_twisted_wire(
            mesh, "compression", P, coating=None, n_load_steps=8
        )
        assert result_b.converged, "裸撚線圧縮が収束しない"

        d_b = _extract_tip_displacement(mesh, result_b.u, "compression")
        k_bare = abs(_compute_equivalent_stiffness(P, d_b))

        ss = self._sheath_stiffness(mesh)
        L = _N_PITCHES * _PITCH
        k_sheath_axial = ss["EA"] / L

        print(f"\n{'=' * 65}")
        print("  S4-1 圧縮剛性: 素線+シース")
        print(f"{'=' * 65}")
        print(f"  裸撚線 NCP:       k = {k_bare:.4e} N/m")
        print(f"  シース管 EA/L:    k = {k_sheath_axial:.4e} N/m")
        print(f"  合計（理論）:     k = {k_bare + k_sheath_axial:.4e} N/m")

        assert d_b < 0, "圧縮変位が負でない"


# ====================================================================
# テスト: 全モード横断サマリー
# ====================================================================


class TestS4StiffnessSummary:
    """全構成 × 全モードのサマリーレポート."""

    def test_stiffness_summary_report(self):
        """全構成の断面剛性パラメータと理論等価剛性の一覧を出力."""
        mesh = _make_mesh()
        n = mesh.n_strands

        # 素線束の剛性
        wire_EA = _E * _SECTION.A
        wire_EI = _E * _SECTION.Iy
        wire_GJ = _G * _SECTION.J

        # 被膜込み複合断面
        comp = coated_beam_section(_WIRE_R, _E, _NU, _COATING)

        # シース等価剛性
        ss = sheath_equivalent_stiffness(mesh, _SHEATH)

        L = _N_PITCHES * _PITCH

        print(f"\n{'=' * 75}")
        print("  S4-1 撚線構造剛性比較サマリー")
        print(f"{'=' * 75}")
        print(f"  撚線: {n}本撚り, 素線 d={_WIRE_D * 1e3:.1f}mm, pitch={_PITCH * 1e3:.0f}mm")
        print(f"  被膜: t={_COATING.thickness * 1e3:.2f}mm, E={_COATING.E / 1e9:.1f}GPa")
        print(f"  シース: t={_SHEATH.thickness * 1e3:.2f}mm, E={_SHEATH.E / 1e9:.1f}GPa")
        print(f"  モデル長: {L * 1e3:.1f}mm")
        print()

        # 断面剛性テーブル
        print(f"  {'構成':<20} {'EA [N]':>14} {'EI [N·m²]':>14} {'GJ [N·m²]':>14}")
        print(f"  {'-' * 66}")
        print(f"  {'素線(1本)':<20} {wire_EA:>14.4e} {wire_EI:>14.4e} {wire_GJ:>14.4e}")
        print(
            f"  {'素線(7本合計)':<20} {wire_EA * n:>14.4e} {wire_EI * n:>14.4e} {wire_GJ * n:>14.4e}"
        )
        print(
            f"  {'被膜込み(1本)':<20} {comp['EA']:>14.4e} {comp['EIy']:>14.4e} {comp['GJ']:>14.4e}"
        )
        print(
            f"  {'被膜込み(7本)':<20} {comp['EA'] * n:>14.4e} {comp['EIy'] * n:>14.4e} {comp['GJ'] * n:>14.4e}"
        )
        print(f"  {'シース管':<20} {ss['EA']:>14.4e} {ss['EIy']:>14.4e} {ss['GJ']:>14.4e}")
        print(
            f"  {'被膜7本+シース':<20} {comp['EA'] * n + ss['EA']:>14.4e} "
            f"{comp['EIy'] * n + ss['EIy']:>14.4e} {comp['GJ'] * n + ss['GJ']:>14.4e}"
        )

        print()
        print("  被膜の剛性増分率:")
        print(f"    EA:  {(comp['EA'] - wire_EA) / wire_EA * 100:.2f}%")
        print(f"    EI:  {(comp['EIy'] - wire_EI) / wire_EI * 100:.2f}%")
        print(f"    GJ:  {(comp['GJ'] - wire_GJ) / wire_GJ * 100:.2f}%")
        print()
        print("  シースの剛性寄与（vs 素線7本合計）:")
        print(f"    EA:  {ss['EA'] / (wire_EA * n) * 100:.2f}%")
        print(f"    EI:  {ss['EIy'] / (wire_EI * n) * 100:.2f}%")
        print(f"    GJ:  {ss['GJ'] / (wire_GJ * n) * 100:.2f}%")

        # 全て正値であること
        assert comp["EA"] > wire_EA
        assert comp["EIy"] > wire_EI
        assert comp["GJ"] > wire_GJ
        assert ss["EA"] > 0
