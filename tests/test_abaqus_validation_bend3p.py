"""Abaqus 三点曲げバリデーションテスト.

assets/test_assets/Abaqus/1-bend3p/ のAbaqus結果と
xkep-cae の解析結果を比較する。

Abaqusモデル概要:
  - yz対称1/2モデル
  - 線長100mm (半モデル: x=0〜50mm)
  - 円形断面 直径1mm (r=0.5mm)
  - 曲げ治具R5 + 抑え治具R5
  - E=100 GPa, ν=0.3
  - idx1: 弾性解析, idx2: 弾塑性解析

xkep-caeモデル（簡略化）:
  - 同じワイヤメッシュ（100要素, x=0〜50mm）
  - 接触の代わりに点支持（x=25mm）
  - 対称BC（x=0: ux=0, θy=0, θz=0）
  - 変位制御（x=0: uy を規定変位）
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.assembly import assemble_global_stiffness
from xkep_cae.elements.beam_timo3d import TimoshenkoBeam3D
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection

ASSET_DIR = (
    Path(__file__).resolve().parent.parent / "assets" / "test_assets" / "Abaqus" / "1-bend3p"
)


# ---------------------------------------------------------------------------
# Abaqus データ読み込みユーティリティ
# ---------------------------------------------------------------------------


def load_abaqus_rf(csv_path: Path) -> dict[str, np.ndarray]:
    """Abaqus RF CSV を読み込む."""
    data: dict[str, list[float]] = {}
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                data.setdefault(key, []).append(float(val))
    return {k: np.array(v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# xkep-cae 三点曲げ半モデル構築
# ---------------------------------------------------------------------------


def build_bend3p_half_model(
    n_elems: int = 100,
    L_half: float = 50.0,
    diameter: float = 1.0,
    E: float = 100e3,
    nu: float = 0.3,
) -> dict:
    """Abaqus三点曲げ半モデルに対応するxkep-caeモデルを構築する."""
    n_nodes = n_elems + 1
    nodes = np.zeros((n_nodes, 3), dtype=float)
    nodes[:, 0] = np.linspace(0, L_half, n_nodes)

    conn = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])

    section = BeamSection.circle(diameter)
    material = BeamElastic1D(E=E, nu=nu)
    beam = TimoshenkoBeam3D(
        section=section,
        kappa_y="cowper",
        kappa_z="cowper",
        v_ref=np.array([0.0, 1.0, 0.0]),
    )

    ndof_per_node = 6
    ndof_total = ndof_per_node * n_nodes

    dx = L_half / n_elems
    support_idx = int(round(25.0 / dx))
    center_idx = 0

    # 拘束DOF（ゼロ変位のみ。中心 uy は変位制御で別途処理）
    fixed_dofs = set()
    # 対称BC (x=0): ux=0, uz=0, θx=0, θy=0, θz=0
    for dof in [0, 2, 3, 4, 5]:
        fixed_dofs.add(center_idx * ndof_per_node + dof)
    # 支持BC (x=25): uy=0
    fixed_dofs.add(support_idx * ndof_per_node + 1)

    fixed_dofs_arr = np.array(sorted(fixed_dofs), dtype=int)

    center_uy_dof = center_idx * ndof_per_node + 1
    support_uy_dof = support_idx * ndof_per_node + 1

    return {
        "nodes": nodes,
        "conn": conn,
        "section": section,
        "material": material,
        "beam": beam,
        "n_nodes": n_nodes,
        "ndof_per_node": ndof_per_node,
        "ndof_total": ndof_total,
        "fixed_dofs": fixed_dofs_arr,
        "center_idx": center_idx,
        "support_idx": support_idx,
        "center_uy_dof": center_uy_dof,
        "support_uy_dof": support_uy_dof,
    }


def solve_displacement_control(model: dict, delta_y: float) -> dict:
    """変位制御で半モデルを解く（密行列ベース）.

    スパース行列の apply_dirichlet では非ゼロ規定変位処理で
    数値的問題が生じうるため、密行列で BC を直接適用する。

    Args:
        model: build_bend3p_half_model() の戻り値
        delta_y: 中心ノードに規定する y方向変位 [mm]（負=下向き）

    Returns:
        {"u": ndarray, "reaction_support_y": float}
    """
    # アセンブリ
    K = assemble_global_stiffness(
        model["nodes"],
        [(model["beam"], model["conn"])],
        model["material"],
        show_progress=False,
    )
    K_dense = K.toarray() if sp.issparse(K) else np.array(K, dtype=float)

    ndof_total = model["ndof_total"]
    f = np.zeros(ndof_total)

    center_uy_dof = model["center_uy_dof"]

    # 全拘束DOF: ゼロ変位BC + 変位制御
    all_fixed = np.unique(np.concatenate([model["fixed_dofs"], [center_uy_dof]]))
    values = np.zeros(len(all_fixed))
    for i, dof in enumerate(all_fixed):
        if dof == center_uy_dof:
            values[i] = delta_y

    # 密行列で BC 適用（行列消去法）
    K_bc = K_dense.copy()
    f_bc = f.copy()
    for d, v in zip(all_fixed, values, strict=True):
        f_bc -= K_bc[:, d] * v
        K_bc[:, d] = 0.0
        K_bc[d, :] = 0.0
        K_bc[d, d] = 1.0
        f_bc[d] = v

    # 密行列ソルバー
    u = np.linalg.solve(K_bc, f_bc)

    # 反力: R = K_original * u - f_ext
    R = K_dense @ u - f
    reaction_support_y = R[model["support_uy_dof"]]

    return {
        "u": u,
        "reaction_support_y": reaction_support_y,
        "reaction_total": R,
    }


# ---------------------------------------------------------------------------
# テスト
# ---------------------------------------------------------------------------


class TestAbaqusBend3pElastic:
    """Abaqus三点曲げ弾性解析（idx1）との比較テスト."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """モデルとAbaqusデータを読み込む."""
        self.model = build_bend3p_half_model()
        rf_path = ASSET_DIR / "go_idx1_RF.csv"
        if rf_path.exists():
            self.abaqus_rf = load_abaqus_rf(rf_path)
        else:
            pytest.skip("Abaqus RF データが見つかりません")

    def test_linear_stiffness_linearity(self):
        """線形解析で荷重-変位が線形関係である."""
        delta1 = -0.1
        delta2 = -0.2
        result1 = solve_displacement_control(self.model, delta1)
        result2 = solve_displacement_control(self.model, delta2)

        K1 = abs(result1["reaction_support_y"] / delta1)
        K2 = abs(result2["reaction_support_y"] / delta2)

        error = abs(K1 - K2) / K1
        assert error < 1e-8, f"線形性誤差: {error:.6e}"

    def test_stiffness_vs_abaqus(self):
        """xkep-cae の線形剛性が Abaqus 結果と整合する.

        Abaqus のクワジスタティック動解析（弾性 idx1）の
        初期線形領域での反力-変位関係を比較する。

        許容誤差: 10%（接触分布・動的効果・NLGEOM の差異を考慮）
        """
        delta = -1.0
        result = solve_displacement_control(self.model, delta)
        R_xkep = abs(result["reaction_support_y"])

        t = self.abaqus_rf["t"]
        rf2 = self.abaqus_rf["RF2"]

        # 初期線形領域（t=5〜15s, δ=1.5〜4.5mm）
        mask = (t >= 5) & (t <= 15)
        if not np.any(mask):
            pytest.skip("Abaqus データの時間範囲が不十分")

        t_lin = t[mask]
        rf2_lin = rf2[mask]
        delta_lin = 0.3 * t_lin  # mm

        K_abaqus_arr = np.abs(rf2_lin) / delta_lin
        K_abaqus = np.mean(K_abaqus_arr)
        K_xkep = R_xkep / abs(delta)

        relative_diff = abs(K_xkep - K_abaqus) / K_abaqus

        print(f"\n  xkep-cae 線形剛性: {K_xkep:.4f} N/mm")
        print(f"  Abaqus 線形剛性:   {K_abaqus:.4f} N/mm")
        print(f"  相対差異: {relative_diff * 100:.2f}%")

        assert relative_diff < 0.10, (
            f"剛性差異 {relative_diff * 100:.1f}% > 10%: xkep={K_xkep:.4f}, abaqus={K_abaqus:.4f}"
        )

    def test_reaction_force_direction(self):
        """下向き荷重 → 支持点は上向き反力."""
        result = solve_displacement_control(self.model, -1.0)
        assert result["reaction_support_y"] > 0

    def test_displacement_profile(self):
        """x=0 で最大たわみ、x=25 でゼロ、x=50 で上向き."""
        result = solve_displacement_control(self.model, -1.0)
        u = result["u"]
        ndof = self.model["ndof_per_node"]

        uy = np.array([u[i * ndof + 1] for i in range(self.model["n_nodes"])])

        center_uy = uy[self.model["center_idx"]]
        support_uy = uy[self.model["support_idx"]]
        tip_uy = uy[-1]

        assert center_uy < 0, "対称面のたわみが負（下向き）であること"
        assert abs(support_uy) < 1e-12, f"支持点変位がゼロでない: {support_uy}"
        assert tip_uy > 0, "オーバーハング先端が上向きであること"

    def test_symmetry_bcs(self):
        """対称境界条件: x=0 で ux=0, uz=0, θx=0, θy=0, θz=0."""
        result = solve_displacement_control(self.model, -1.0)
        u = result["u"]
        ndof = self.model["ndof_per_node"]
        ci = self.model["center_idx"]

        assert abs(u[ci * ndof + 0]) < 1e-12  # ux
        assert abs(u[ci * ndof + 2]) < 1e-12  # uz
        assert abs(u[ci * ndof + 3]) < 1e-12  # θx
        assert abs(u[ci * ndof + 4]) < 1e-12  # θy
        assert abs(u[ci * ndof + 5]) < 1e-12  # θz


class TestAbaqusBend3pCurvature:
    """曲率検証."""

    def test_curvature_sign_at_center(self):
        """中心付近の曲率が非ゼロ."""
        model = build_bend3p_half_model()
        result = solve_displacement_control(model, -1.0)
        u = result["u"]
        ndof = model["ndof_per_node"]

        theta_z_0 = u[0 * ndof + 5]
        theta_z_1 = u[1 * ndof + 5]
        dx = model["nodes"][1, 0] - model["nodes"][0, 0]
        curvature_approx = (theta_z_1 - theta_z_0) / dx

        assert abs(curvature_approx) > 1e-6, f"中心付近の曲率がゼロに近い: {curvature_approx}"


class TestAbaqusBend3pNLGEOM:
    """Abaqus NLGEOM 大ストローク比較（CR梁 vs Abaqus B31）."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """モデルとAbaqusデータを読み込む."""
        self.model = build_bend3p_half_model()
        rf_path = ASSET_DIR / "go_idx1_RF.csv"
        if rf_path.exists():
            self.abaqus_rf = load_abaqus_rf(rf_path)
        else:
            pytest.skip("Abaqus RF データが見つかりません")

    @staticmethod
    def _solve_cr_displacement_control(
        model: dict,
        delta_y: float,
        n_steps: int = 20,
        tol: float = 1e-6,
        max_iter: int = 50,
    ) -> list[dict]:
        """CR梁による変位制御NR解析.

        Args:
            model: build_bend3p_half_model() の戻り値
            delta_y: 最終規定変位 [mm]（負=下向き）
            n_steps: 荷重ステップ数
            tol: NR収束トレランス
            max_iter: NR最大反復回数

        Returns:
            各ステップの {"delta": float, "reaction": float, "converged": bool} のリスト
        """
        from xkep_cae.elements.beam_timo3d import assemble_cr_beam3d

        nodes = model["nodes"]
        conn = model["conn"]
        E = model["material"].E
        nu = model["material"].nu
        G = E / (2.0 * (1.0 + nu))
        sec = model["section"]
        ky = sec.cowper_kappa_y(nu)
        kz = sec.cowper_kappa_z(nu)

        ndof = model["ndof_total"]
        u = np.zeros(ndof, dtype=float)

        center_uy_dof = model["center_uy_dof"]
        all_fixed = np.unique(np.concatenate([model["fixed_dofs"], [center_uy_dof]]))
        free = np.setdiff1d(np.arange(ndof), all_fixed)

        results = []
        for step in range(1, n_steps + 1):
            target = delta_y * step / n_steps
            u[center_uy_dof] = target

            converged = False
            for _k in range(max_iter):
                K_T, f_int = assemble_cr_beam3d(
                    nodes,
                    conn,
                    u,
                    E,
                    G,
                    sec.A,
                    sec.Iy,
                    sec.Iz,
                    sec.J,
                    ky,
                    kz,
                    stiffness=True,
                    internal_force=True,
                    sparse=False,
                )
                R = -f_int.copy()
                R[all_fixed] = 0.0
                res_norm = float(np.linalg.norm(R[free]))
                ref_norm = max(float(np.linalg.norm(f_int)), 1.0)
                if res_norm / ref_norm < tol:
                    converged = True
                    break
                K_ff = K_T[np.ix_(free, free)]
                du = np.linalg.solve(K_ff, R[free])
                u[free] += du

            _, f_int_final = assemble_cr_beam3d(
                nodes,
                conn,
                u,
                E,
                G,
                sec.A,
                sec.Iy,
                sec.Iz,
                sec.J,
                ky,
                kz,
                stiffness=False,
                internal_force=True,
            )
            reaction_y = f_int_final[model["support_uy_dof"]]
            results.append(
                {
                    "delta": target,
                    "reaction": reaction_y,
                    "converged": converged,
                }
            )

        return results

    def test_nlgeom_converges_all_steps(self):
        """CR梁NLGEOMが全ステップで収束する（δ=-15mm, 20ステップ）."""
        results = self._solve_cr_displacement_control(self.model, delta_y=-15.0, n_steps=20)
        for i, r in enumerate(results):
            assert r["converged"], f"Step {i + 1} (δ={r['delta']:.2f}) not converged"

    def test_nlgeom_initial_stiffness_matches_abaqus(self):
        """NLGEOM初期線形域の剛性がAbaqusと一致（< 10%）."""
        results = self._solve_cr_displacement_control(self.model, delta_y=-5.0, n_steps=10)
        K_cr = abs(results[0]["reaction"] / results[0]["delta"])

        t = self.abaqus_rf["t"]
        rf2 = self.abaqus_rf["RF2"]
        mask = (t >= 5) & (t <= 15)
        if not np.any(mask):
            pytest.skip("Abaqus データの時間範囲が不十分")
        K_abaqus = np.mean(np.abs(rf2[mask]) / (0.3 * t[mask]))

        rel_diff = abs(K_cr - K_abaqus) / K_abaqus
        assert rel_diff < 0.10, f"初期剛性差異 {rel_diff * 100:.1f}% > 10%"

    def test_nlgeom_large_stroke_reaction_trend(self):
        """大ストロークで反力が単調増加する."""
        results = self._solve_cr_displacement_control(self.model, delta_y=-15.0, n_steps=15)
        reactions = [abs(r["reaction"]) for r in results]
        for i in range(1, len(reactions)):
            assert reactions[i] >= reactions[i - 1] * 0.95, (
                f"反力が減少: step {i}: {reactions[i - 1]:.4f} → {reactions[i]:.4f}"
            )

    def test_nlgeom_rf_curve_within_envelope(self):
        """RF曲線がAbaqusと同オーダー（δ ≤ 10mm, 点支持vs接触の差異を考慮）.

        大ストローク（δ > 10mm）では接触分布と点支持の差が大きくなるため、
        比較範囲をδ ≤ 10mm に限定する。
        """
        cr_results = self._solve_cr_displacement_control(self.model, delta_y=-10.0, n_steps=10)
        t = self.abaqus_rf["t"]
        rf2 = self.abaqus_rf["RF2"]

        for r in cr_results:
            delta_mm = abs(r["delta"])
            t_abaqus = delta_mm / 0.3
            if t_abaqus < 1.0 or t_abaqus > 99.0:
                continue
            idx = np.argmin(np.abs(t - t_abaqus))
            rf_abaqus = abs(rf2[idx])
            rf_cr = abs(r["reaction"])
            if rf_abaqus > 0.01:
                ratio = rf_cr / rf_abaqus
                assert 0.3 < ratio < 3.0, (
                    f"δ={delta_mm:.1f}mm: CR={rf_cr:.4f}, Abaqus={rf_abaqus:.4f}, ratio={ratio:.2f}"
                )


class TestModelConstruction:
    """モデル構築の基本検証."""

    def test_node_count(self):
        """半モデルのノード数が正しい."""
        model = build_bend3p_half_model(n_elems=100, L_half=50.0)
        assert model["n_nodes"] == 101

    def test_support_at_x25(self):
        """支持ノードが x=25mm にある."""
        model = build_bend3p_half_model(n_elems=100, L_half=50.0)
        x_support = model["nodes"][model["support_idx"], 0]
        assert abs(x_support - 25.0) < 1e-10

    def test_section_properties(self):
        """断面特性が正しい（直径1mmの円形断面）."""
        model = build_bend3p_half_model(diameter=1.0)
        sec = model["section"]
        r = 0.5
        A_expected = np.pi * r**2
        I_expected = np.pi * (2 * r) ** 4 / 64.0

        assert abs(sec.A - A_expected) / A_expected < 1e-10
        assert abs(sec.Iy - I_expected) / I_expected < 1e-10

    def test_mesh_convergence(self):
        """メッシュ細分化で剛性が収束する（< 2%）."""
        model_coarse = build_bend3p_half_model(n_elems=20)
        model_fine = build_bend3p_half_model(n_elems=100)

        delta = -0.1
        result_coarse = solve_displacement_control(model_coarse, delta)
        result_fine = solve_displacement_control(model_fine, delta)

        K_coarse = abs(result_coarse["reaction_support_y"] / delta)
        K_fine = abs(result_fine["reaction_support_y"] / delta)

        assert K_fine > 0, f"Fine model reaction is zero: {K_fine}"
        error = abs(K_coarse - K_fine) / K_fine
        assert error < 0.02, f"メッシュ収束性: {error * 100:.2f}% > 2%"
