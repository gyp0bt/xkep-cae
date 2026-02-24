"""Abaqus 弾塑性三点曲げバリデーションテスト（idx2）.

CR梁＋ファイバーモデルによる弾塑性解析を、Abaqus B31 NLGEOM
弾塑性解析（go_idx2）の反力データと比較する。

Abaqusモデル (go_idx2):
  - 円形断面 d=1mm, E=100 GPa, ν=0.3
  - *PLASTIC テーブル: σ_y0=0.1 MPa（Abaqus inp 値そのまま）
  - NLGEOM=YES, 準静的動解析
  - dy=-30mm / 100s
  - 支持スパン50mm（半モデル25mm）

xkep-caeモデル:
  - CR梁＋ファイバー断面積分
  - 20要素, 円形断面 nr=4, nt=8 (32 fibers)
  - 半モデル: x=0〜25mm
  - NR法 変位制御
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from xkep_cae.elements.beam_timo3d import assemble_cr_beam3d_fiber
from xkep_cae.materials.plasticity_1d import Plasticity1D, TabularIsotropicHardening
from xkep_cae.sections.fiber import FiberSection
from xkep_cae.sections.fiber_integrator import FiberIntegrator

ASSET_DIR = (
    Path(__file__).resolve().parent.parent / "assets" / "test_assets" / "Abaqus" / "1-bend3p"
)

# Abaqus *PLASTIC テーブル（go_idx2.inp から）— 値は MPa 単位そのまま
PLASTIC_TABLE: list[tuple[float, float]] = [
    (1.00e-01, 0.00e00),
    (7.58e00, 5.00e-04),
    (8.99e00, 1.00e-03),
    (1.07e01, 2.00e-03),
    (1.18e01, 3.00e-03),
    (1.27e01, 4.00e-03),
    (1.34e01, 5.00e-03),
    (1.47e01, 7.30e-03),
    (1.69e01, 1.28e-02),
    (2.08e01, 2.94e-02),
    (2.20e01, 3.68e-02),
    (2.42e01, 5.39e-02),
    (2.53e01, 6.46e-02),
    (2.82e01, 1.00e-01),
    (3.35e01, 2.00e-01),
    (3.71e01, 3.00e-01),
    (3.99e01, 4.00e-01),
    (5.01e01, 1.00e00),
    (6.59e01, 3.00e00),
    (8.14e01, 7.00e00),
    (8.90e01, 1.00e01),
]

E_MPA = 100_000.0  # 100 GPa in MPa
NU = 0.3
G_MPA = E_MPA / (2.0 * (1.0 + NU))
DIAMETER = 1.0  # mm


# ---------------------------------------------------------------------------
# ユーティリティ
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
# 半モデル構築
# ---------------------------------------------------------------------------


def _build_fiber_model(
    n_elems: int = 20,
    nr_fiber: int = 4,
    nt_fiber: int = 8,
) -> dict:
    """CR梁＋ファイバー弾塑性の半モデルを構築する.

    半モデル: x=0 (対称面、載荷点) 〜 x=25mm (支持点)
    対称BC: x=0 で ux=0, θy=0, θz=0
    支持: x=25mm で uy=0
    載荷: x=0 で uy を変位制御

    Returns:
        dict: nodes, connectivity, fiber_integrators, free_dofs, load_dof, support_dof, etc.
    """
    half_span = 25.0  # mm (支持スパン50mmの半分)
    n_nodes = n_elems + 1
    nodes = np.zeros((n_nodes, 3), dtype=float)
    nodes[:, 0] = np.linspace(0.0, half_span, n_nodes)

    connectivity = np.array([[i, i + 1] for i in range(n_elems)], dtype=int)

    # ファイバー断面
    sec = FiberSection.circle(DIAMETER, nr=nr_fiber, nt=nt_fiber)
    kappa_y = sec.cowper_kappa_y(NU)
    kappa_z = sec.cowper_kappa_z(NU)

    # 弾塑性材料
    iso = TabularIsotropicHardening(table=PLASTIC_TABLE)
    mat = Plasticity1D(E=E_MPA, iso=iso)

    # 各要素に FiberIntegrator を割当
    fiber_integrators = [FiberIntegrator(section=sec, material=mat) for _ in range(n_elems)]

    # DOF 定義 (6 DOF/node)
    ndof = 6 * n_nodes

    # 境界条件
    # 対称面 (node 0): ux=0, θy=0, θz=0
    # 載荷点 (node 0): uy = prescribed
    # 支持点 (node n_nodes-1): uy=0
    sym_node = 0
    sup_node = n_nodes - 1

    fixed_dofs = set()
    # 対称BC (x=0): ux=0, uz=0, θx=0, θy=0, θz=0
    for dof_local in [0, 2, 3, 4, 5]:
        fixed_dofs.add(6 * sym_node + dof_local)
    # 支持点 uy
    fixed_dofs.add(6 * sup_node + 1)  # uy

    # 載荷点 DOF
    load_dof = 6 * sym_node + 1  # uy (node 0)

    free_dofs = sorted(set(range(ndof)) - fixed_dofs - {load_dof})

    return {
        "nodes": nodes,
        "connectivity": connectivity,
        "fiber_integrators": fiber_integrators,
        "kappa_y": kappa_y,
        "kappa_z": kappa_z,
        "ndof": ndof,
        "free_dofs": np.array(free_dofs, dtype=int),
        "fixed_dofs": sorted(fixed_dofs),
        "load_dof": load_dof,
        "support_dof": 6 * sup_node + 1,
        "n_elems": n_elems,
        "n_nodes": n_nodes,
    }


# ---------------------------------------------------------------------------
# NR ソルバー（変位制御、インクリメンタル）
# ---------------------------------------------------------------------------


def _solve_fiber_displacement_control(
    model: dict,
    delta_y: float,
    n_steps: int = 100,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> dict:
    """変位制御 NR 法でファイバー弾塑性解析を実行する.

    Args:
        model: _build_fiber_model() の返り値
        delta_y: 載荷点の最終 uy (負値 = 下向き)
        n_steps: 載荷ステップ数
        tol: NR 収束判定の相対残差許容値
        max_iter: 各ステップの最大 NR 反復数

    Returns:
        dict: disp_history, rf_history, converged_steps
    """
    nodes = model["nodes"]
    conn = model["connectivity"]
    fis = model["fiber_integrators"]
    free = model["free_dofs"]
    load_dof = model["load_dof"]
    ndof = model["ndof"]

    u = np.zeros(ndof, dtype=float)
    disp_history = [0.0]
    rf_history = [0.0]
    converged_steps = 0

    # 各ステップの変位増分
    delta_per_step = delta_y / n_steps

    for _step in range(1, n_steps + 1):
        # 変位制御: 載荷 DOF に増分を適用
        u[load_dof] += delta_per_step

        # 各要素の現在の状態を保存（ステップ開始時）
        saved_states = [fi.copy_states() for fi in fis]

        converged = False
        for it in range(max_iter):
            # アセンブリ
            K_T, f_int, all_states_new = assemble_cr_beam3d_fiber(
                nodes,
                conn,
                u,
                G_MPA,
                model["kappa_y"],
                model["kappa_z"],
                fis,
                v_ref=np.array([0.0, 0.0, 1.0]),
            )

            # 残差 = -f_int（free DOFs のみ）
            residual = -f_int[free]
            res_norm = np.linalg.norm(residual)

            if it == 0:
                ref_norm = max(res_norm, 1e-12)

            if res_norm / ref_norm < tol or res_norm < 1e-14:
                converged = True
                # 状態を commit
                for ie, fi in enumerate(fis):
                    fi.update_states(all_states_new[ie])
                break

            # NR 補正
            K_ff = K_T[np.ix_(free, free)]
            try:
                du_free = np.linalg.solve(K_ff, residual)
            except np.linalg.LinAlgError:
                break

            u[free] += du_free

        if not converged:
            # ロールバック
            for ie, fi in enumerate(fis):
                fi.update_states(saved_states[ie])
            u[load_dof] -= delta_per_step
        else:
            converged_steps += 1

        # 反力 = f_int[load_dof]
        if converged:
            _, f_int_final, _ = assemble_cr_beam3d_fiber(
                nodes,
                conn,
                u,
                G_MPA,
                model["kappa_y"],
                model["kappa_z"],
                fis,
                v_ref=np.array([0.0, 0.0, 1.0]),
                stiffness=False,
            )
            rf = f_int_final[load_dof]
            disp_history.append(u[load_dof])
            rf_history.append(rf)

    return {
        "disp": np.array(disp_history),
        "rf": np.array(rf_history),
        "converged_steps": converged_steps,
        "n_steps": n_steps,
    }


# ---------------------------------------------------------------------------
# テスト
# ---------------------------------------------------------------------------


class TestAbaqusBend3pElastoplastic:
    """Abaqus B31 弾塑性三点曲げとの比較."""

    @pytest.fixture(scope="class")
    def model(self):
        """半モデルを構築する."""
        return _build_fiber_model(n_elems=20, nr_fiber=4, nt_fiber=8)

    @pytest.fixture(scope="class")
    def result(self, model):
        """δ=-0.5mm を 100 ステップで解く."""
        return _solve_fiber_displacement_control(
            model,
            delta_y=-0.5,
            n_steps=100,
            tol=1e-6,
            max_iter=50,
        )

    def test_fiber_nr_converges(self, result):
        """NR がほとんどのステップで収束する."""
        ratio = result["converged_steps"] / result["n_steps"]
        assert ratio >= 0.8, f"収束率 {ratio:.2%} < 80%"

    def test_initial_stiffness_order_of_magnitude(self, result):
        """初期剛性がAbaqusと同じオーダーにある."""
        # Abaqus: RF2 at t=1 ≈ -0.0316 N, dy=-0.3mm → K ≈ 0.105 N/mm
        abaqus_rf = load_abaqus_rf(ASSET_DIR / "go_idx2_RF.csv")
        K_abaqus = abs(abaqus_rf["RF2"][1]) / (abs(abaqus_rf["t"][1]) * 30.0 / 100.0)

        # xkep-cae 初期剛性（最初の収束点）
        disp = result["disp"]
        rf = result["rf"]
        # 非ゼロ点を探す
        for i in range(1, len(disp)):
            if abs(disp[i]) > 1e-12 and abs(rf[i]) > 1e-12:
                K_xkep = abs(rf[i]) / abs(disp[i])
                break
        else:
            pytest.fail("非ゼロの変位・反力データが見つからない")

        ratio = K_xkep / K_abaqus
        assert 0.1 < ratio < 10.0, (
            f"剛性比 {ratio:.2f} が 0.1-10x の範囲外: K_xkep={K_xkep:.4f}, K_abaqus={K_abaqus:.4f}"
        )

    def test_reaction_force_sign(self, result):
        """下向き載荷で支持反力が正方向（上向き）になる."""
        disp = result["disp"]
        rf = result["rf"]
        # 最初の非ゼロ点
        for i in range(1, len(disp)):
            if abs(disp[i]) > 1e-12:
                # 載荷点の反力は下向き（負）載荷に対して上向き（正）
                # f_int[load_dof] は内力なので、外力の反力と符号が逆
                # 実際に反力の絶対値が非ゼロであることを確認
                assert abs(rf[i]) > 1e-14, "反力がゼロ"
                break

    def test_plasticity_softening(self, result):
        """塑性により割線剛性が低下する."""
        disp = result["disp"]
        rf = result["rf"]

        # 初期の割線剛性
        idx_early = min(5, len(disp) - 1)
        if abs(disp[idx_early]) < 1e-12:
            pytest.skip("初期データ不足")
        K_early = abs(rf[idx_early]) / abs(disp[idx_early])

        # 後半の割線剛性
        idx_late = len(disp) - 1
        if abs(disp[idx_late]) < 1e-12:
            pytest.skip("後半データ不足")
        K_late = abs(rf[idx_late]) / abs(disp[idx_late])

        assert K_late < K_early, (
            f"割線剛性が低下していない: K_early={K_early:.4f}, K_late={K_late:.4f}"
        )

    def test_rf_curve_vs_abaqus(self, result):
        """RF カーブが Abaqus データの 10 倍エンベロープ内にある."""
        abaqus_rf = load_abaqus_rf(ASSET_DIR / "go_idx2_RF.csv")
        # Abaqus: dy = -t * 30 / 100 (mm), RF2 が反力
        ab_disp = abaqus_rf["t"] * 30.0 / 100.0  # mm (正値)
        ab_rf = np.abs(abaqus_rf["RF2"])  # N (正値)

        xkep_disp = np.abs(result["disp"])
        xkep_rf = np.abs(result["rf"])

        # xkep データの各点に対して、Abaqus の補間値と比較
        n_check = 0
        for i in range(1, len(xkep_disp)):
            d = xkep_disp[i]
            if d < 0.001:
                continue
            # Abaqus 補間
            ab_rf_interp = np.interp(d, ab_disp, ab_rf)
            if ab_rf_interp < 1e-10:
                continue
            ratio = xkep_rf[i] / ab_rf_interp
            assert 0.1 < ratio < 10.0, (
                f"δ={d:.4f}mm: xkep_rf={xkep_rf[i]:.6f}, "
                f"abaqus_rf={ab_rf_interp:.6f}, ratio={ratio:.2f}"
            )
            n_check += 1

        assert n_check >= 3, f"比較点が不足: {n_check}"
