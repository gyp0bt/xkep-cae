"""応力コンター・曲率コンターの隣接要素間連続性チェック（物理テスト）.

隣接要素間の応力差・曲率差が物理的に不自然な離散的ジャンプを
起こしていないことを自動検証する。

CLAUDE.md の物理テスト思想に基づく:
  「応力・曲率の連続性: 隣接要素間の応力差が極端に離散的でないか」

テスト方針:
  1. 片持ち梁の集中荷重 → 応力分布は線形 → 隣接要素間の応力差は一定
  2. 等分布荷重 → 曲率分布は放物線的 → 隣接要素間の曲率変化は滑らか
  3. 変化率の最大/平均比が閾値以内であることを検証
"""

import numpy as np

from xkep_cae.elements.beam_timo3d import (
    beam3d_section_forces,
    timo_beam3d_ke_global,
)
from xkep_cae.sections.beam import BeamSection


def _solve_cantilever_3d(
    n_elems: int,
    L: float,
    E: float,
    load_type: str = "tip",
    P: float = 1.0,
):
    """3D片持ち梁を解いて要素ごとの断面力と座標を返す.

    Args:
        n_elems: 要素数
        L: 梁の長さ
        E: ヤング率
        load_type: "tip" (先端集中荷重) or "distributed" (等分布荷重)
        P: 荷重値

    Returns:
        forces_list: 各要素の (BeamForces3D_node1, BeamForces3D_node2)
        elem_centers: 各要素の中心座標 (n_elems,)
        section: BeamSection
    """
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    d = 0.01  # 10mm 直径
    section = BeamSection.circle(d)
    kappa = 5.0 / 6.0

    n_nodes = n_elems + 1
    ndof = n_nodes * 6
    dx = L / n_elems

    # 節点座標
    coords_all = np.zeros((n_nodes, 3))
    coords_all[:, 2] = np.linspace(0, L, n_nodes)  # z方向に延伸

    # 要素接続
    connectivity = [(i, i + 1) for i in range(n_elems)]

    # 剛性行列
    K = np.zeros((ndof, ndof))
    for n1, n2 in connectivity:
        elem_coords = coords_all[np.array([n1, n2])]
        Ke = timo_beam3d_ke_global(
            elem_coords,
            E,
            G,
            section.A,
            section.Iy,
            section.Iz,
            section.J,
            kappa,
            kappa,
        )
        edofs = np.array([6 * n1 + d for d in range(6)] + [6 * n2 + d for d in range(6)])
        K[np.ix_(edofs, edofs)] += Ke

    # 外力
    f = np.zeros(ndof)
    if load_type == "tip":
        # 先端 y方向荷重
        f[6 * (n_nodes - 1) + 1] = P
    elif load_type == "distributed":
        # 等分布荷重（一貫節点力に変換）
        q = P / L  # 単位長さ当たり荷重
        for n1, n2 in connectivity:
            f[6 * n1 + 1] += q * dx / 2
            f[6 * n2 + 1] += q * dx / 2

    # 境界条件（固定端: node 0 の全6自由度を拘束）
    fixed = np.arange(6)
    free = np.setdiff1d(np.arange(ndof), fixed)

    # 解法
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])

    # 断面力抽出
    forces_list = []
    elem_centers = np.zeros(n_elems)
    for ei, (n1, n2) in enumerate(connectivity):
        elem_coords = coords_all[np.array([n1, n2])]
        edofs = np.array([6 * n1 + d for d in range(6)] + [6 * n2 + d for d in range(6)])
        u_elem = u[edofs]
        f1, f2 = beam3d_section_forces(
            elem_coords,
            u_elem,
            E,
            G,
            section.A,
            section.Iy,
            section.Iz,
            section.J,
            kappa,
            kappa,
        )
        forces_list.append((f1, f2))
        elem_centers[ei] = (coords_all[n1, 2] + coords_all[n2, 2]) / 2

    return forces_list, elem_centers, section


def _compute_element_stress(forces_list, section):
    """各要素の代表応力値（軸力/断面積 + 曲げ応力）を計算."""
    import math

    n_elems = len(forces_list)
    stress = np.zeros(n_elems)
    # 円形断面: A = π r² → r = sqrt(A/π)
    r = math.sqrt(section.A / math.pi)
    for i, (_f1, f2) in enumerate(forces_list):
        sigma_axial = f2.N / section.A
        sigma_bend = abs(f2.My) * r / section.Iy
        stress[i] = sigma_axial + sigma_bend
    return stress


def _compute_element_curvature(forces_list, E, section):
    """各要素の代表曲率値を計算（M/EI）."""
    n_elems = len(forces_list)
    curvature = np.zeros(n_elems)
    for i, (_f1, f2) in enumerate(forces_list):
        # 要素中央の曲率 ≈ node2 のモーメントから
        curvature[i] = abs(f2.My) / (E * section.Iy) if E * section.Iy > 0 else 0.0
    return curvature


def _adjacent_change_rate(values):
    """隣接要素間の変化率を計算.

    Returns:
        changes: 隣接差の絶対値 (n-1,)
        max_min_ratio: 変化率の最大/最小比（均一性の指標）
    """
    if len(values) < 2:
        return np.array([]), 1.0

    diffs = np.abs(np.diff(values))

    # 非ゼロの変化だけで比率を計算
    nonzero = diffs[diffs > 1e-15]
    if len(nonzero) < 2:
        return diffs, 1.0

    ratio = nonzero.max() / nonzero.min()
    return diffs, ratio


class TestStressContinuityPhysics:
    """応力コンターの隣接要素間連続性テスト."""

    def test_tip_load_shear_constant(self):
        """片持ち梁の先端荷重: せん断力は全要素で一定."""
        forces, centers, sec = _solve_cantilever_3d(
            n_elems=10,
            L=1.0,
            E=200e9,
            load_type="tip",
            P=100.0,
        )
        shear = np.array([f2.Vy for _, f2 in forces])

        # せん断力は一定（変動が平均の1%未満）
        mean_v = np.mean(np.abs(shear))
        if mean_v > 1e-10:
            variation = np.std(shear) / mean_v
            assert variation < 0.01, f"せん断力の変動が大きすぎる: {variation:.4f} > 0.01"

    def test_tip_load_moment_linear(self):
        """片持ち梁の先端荷重: モーメントは線形分布 → 隣接差は一定."""
        forces, centers, sec = _solve_cantilever_3d(
            n_elems=10,
            L=1.0,
            E=200e9,
            load_type="tip",
            P=100.0,
        )
        moments = np.array([f2.My for _, f2 in forces])
        diffs, ratio = _adjacent_change_rate(moments)

        # 線形分布なら隣接差は一定 → ratio ≈ 1.0
        assert ratio < 1.5, f"モーメント隣接差の最大/最小比が大きすぎる: {ratio:.2f} > 1.5"

    def test_tip_load_stress_smooth(self):
        """先端荷重: 応力分布が滑らか（隣接要素間ジャンプなし）."""
        forces, centers, sec = _solve_cantilever_3d(
            n_elems=20,
            L=1.0,
            E=200e9,
            load_type="tip",
            P=100.0,
        )
        stress = _compute_element_stress(forces, sec)
        diffs, ratio = _adjacent_change_rate(stress)

        # 滑らかな分布では隣接差の比率が小さい
        assert ratio < 2.0, f"応力の隣接要素間変化率が不連続: ratio={ratio:.2f} > 2.0"

    def test_distributed_load_moment_quadratic(self):
        """等分布荷重: モーメントは放物線分布 → 隣接差は線形変化."""
        forces, centers, sec = _solve_cantilever_3d(
            n_elems=10,
            L=1.0,
            E=200e9,
            load_type="distributed",
            P=100.0,
        )
        moments = np.array([f2.My for _, f2 in forces])
        diffs = np.abs(np.diff(moments))

        # 放物線のモーメント → 隣接差は線形減少
        # 差の二次差分がほぼゼロ（線形であるため）
        if len(diffs) >= 3:
            second_diffs = np.abs(np.diff(diffs))
            mean_first_diff = np.mean(diffs)
            if mean_first_diff > 1e-10:
                linearity = np.max(second_diffs) / mean_first_diff
                assert linearity < 0.3, f"モーメント差の非線形性が大きい: {linearity:.4f} > 0.3"


class TestCurvatureContinuityPhysics:
    """曲率コンターの隣接要素間連続性テスト."""

    def test_tip_load_curvature_linear(self):
        """先端荷重: 曲率（M/EI）は線形分布."""
        E = 200e9
        forces, centers, sec = _solve_cantilever_3d(
            n_elems=10,
            L=1.0,
            E=E,
            load_type="tip",
            P=100.0,
        )
        curvature = _compute_element_curvature(forces, E, sec)
        diffs, ratio = _adjacent_change_rate(curvature)

        # 線形分布なら隣接差は一定
        assert ratio < 1.5, f"曲率隣接差の最大/最小比が大きすぎる: {ratio:.2f} > 1.5"

    def test_distributed_load_curvature_smooth(self):
        """等分布荷重: 曲率は滑らかに変化（二次差分が小さい）."""
        E = 200e9
        forces, centers, sec = _solve_cantilever_3d(
            n_elems=20,
            L=1.0,
            E=E,
            load_type="distributed",
            P=100.0,
        )
        curvature = _compute_element_curvature(forces, E, sec)
        diffs = np.abs(np.diff(curvature))

        # 放物線分布の曲率差分は線形的に変化する
        # 二次差分（差分の差分）が一定であることを確認
        if len(diffs) >= 3:
            second_diffs = np.abs(np.diff(diffs))
            mean_second = np.mean(second_diffs)
            max_second = np.max(second_diffs)
            # 二次差分のばらつきが小さいことを確認
            if mean_second > 1e-15:
                uniformity = max_second / mean_second
                assert uniformity < 5.0, f"曲率の二次差分が不均一: {uniformity:.2f} > 5.0"

    def test_curvature_monotonic_tip_load(self):
        """先端荷重: 曲率は固定端から先端に向かって単調減少."""
        E = 200e9
        forces, centers, sec = _solve_cantilever_3d(
            n_elems=10,
            L=1.0,
            E=E,
            load_type="tip",
            P=100.0,
        )
        curvature = _compute_element_curvature(forces, E, sec)

        # 固定端（index 0）が最大、先端（index -1）が最小
        assert curvature[0] >= curvature[-1], (
            f"曲率が固定端→先端で減少していない: "
            f"κ(0)={curvature[0]:.6f}, κ(end)={curvature[-1]:.6f}"
        )

        # 単調減少チェック（許容誤差付き）
        for i in range(len(curvature) - 1):
            assert curvature[i] >= curvature[i + 1] - 1e-10, (
                f"曲率が単調減少でない: κ[{i}]={curvature[i]:.6f} < κ[{i + 1}]={curvature[i + 1]:.6f}"
            )


class TestContourJumpDetectionAPI:
    """隣接要素間変化率チェックのユーティリティテスト."""

    def test_constant_values(self):
        """一定値の場合: 変化率 = 1.0."""
        diffs, ratio = _adjacent_change_rate(np.ones(10))
        assert ratio == 1.0

    def test_linear_values(self):
        """線形値の場合: 隣接差は一定 → ratio ≈ 1.0."""
        vals = np.linspace(0, 10, 11)
        diffs, ratio = _adjacent_change_rate(vals)
        assert ratio < 1.01

    def test_discontinuous_values(self):
        """不連続値の場合: ratio が大きくなる."""
        vals = np.array([1.0, 1.1, 1.2, 5.0, 5.1, 5.2])
        diffs, ratio = _adjacent_change_rate(vals)
        assert ratio > 10.0  # 不連続ジャンプを検出

    def test_single_element(self):
        """要素1つの場合: 空の差分."""
        diffs, ratio = _adjacent_change_rate(np.array([1.0]))
        assert len(diffs) == 0
        assert ratio == 1.0
