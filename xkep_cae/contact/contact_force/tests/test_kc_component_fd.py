"""K_c コンポーネント（K_mat, K_geo, K_st）の個別FD検証（status-291）.

status-290 で K_c がFDの約1/10しかないことが判明。
K_mat / K_geo / K_st のどれが過小かを特定するためのテスト。

K_c = K_mat - K_geo + K_st

FD検証方針:
1. 簡易接触シナリオ（2本平行セグメント + 貫入）を構築
2. f_c(u) を evaluate() で計算
3. 各DOFを摂動して f_c(u + eps*e_j) を計算（geometry再計算込み）
4. FD_K_c = (f_c_pert - f_c_base) / eps と解析的 K_c を比較
5. K_mat, K_geo, K_st のノルムを報告して不足成分を特定
"""

from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact._contact_pair import (
    _ContactConfigInput,
    _ContactManagerInput,
    _ContactPairOutput,
    _ContactStateOutput,
)
from xkep_cae.contact._manager_process import (
    UpdateGeometryInput,
    UpdateGeometryProcess,
)
from xkep_cae.contact.contact_force.strategy import HuberContactForceProcess


def _make_two_segment_scenario(
    *,
    gap_target: float = -0.01,
    use_hermite: bool = False,
    penalty_exponent: float = 1.0,
    k_pen: float = 1e4,
    n_elems: int = 1,
    helical_z: bool = False,
) -> dict:
    """2本の近接平行セグメント接触シナリオを構築.

    n_elems=1: node 0-1 (segA), node 2-3 (segB) → 4ノード
    n_elems=3: チェーン構造 0-1-2-3 (wireA), 4-5-6-7 (wireB) → 8ノード
    helical_z=True: 3Dヘリカル配置（z方向オフセット付き、status-292）
    """
    if n_elems == 1:
        # 2本の平行セグメント（y方向に近接）
        r = 0.05  # radius
        separation = 2 * r + gap_target  # gap = separation - 2r = gap_target
        node_coords = np.array(
            [
                [0.0, 0.0, 0.0],  # 0: segA start
                [1.0, 0.0, 0.0],  # 1: segA end
                [0.0, separation, 0.0],  # 2: segB start
                [1.0, separation, 0.0],  # 3: segB end
            ]
        )
        n_nodes = 4
        connectivity = np.array([[0, 1], [2, 3]])
        elem_pairs = [(0, 1)]  # (elem_a, elem_b)
    else:
        # 3要素チェーン: 端部ノード(node 2, 6)が最近接点になる配置
        # K_st の端部不整合を検証するシナリオ（status-290 TODO）
        r = 0.05
        separation = 2 * r + gap_target
        node_coords = np.array(
            [
                [0.0, 0.0, 0.0],  # 0
                [1.0, 0.0, 0.0],  # 1
                [2.0, 0.05, 0.0],  # 2 (わずかにオフセット)
                [3.0, 0.1, 0.0],  # 3
                [0.0, separation, 0.0],  # 4
                [1.0, separation + 0.02, 0.0],  # 5
                [2.0, separation + 0.03, 0.0],  # 6
                [3.0, separation + 0.05, 0.0],  # 7
            ]
        )
        n_nodes = 8
        connectivity = np.array([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7]])
        # 中央要素ペア
        elem_pairs = [(1, 4)]  # elem 1 (nodes 1,2), elem 4 (nodes 5,6)

    # 3Dヘリカル配置（status-292: z方向DOFカップリング検証用）
    # 4ノード（n_elems=1）で法線にz成分を持つ交差配置:
    # セグメントAはx方向、セグメントBもx方向だがz方向にオフセット
    # これにより法線ベクトルにy,z成分が生じる
    if helical_z and n_elems == 1:
        r = 0.05
        # スキュー配置: segAはx方向（長い）、segBは短く中央寄りに配置
        # segBにy,z方向オフセット → 法線にy,z成分
        # 最近接点: s≈0.3（内部）、t=0（端部）
        dy = 0.06  # y方向分離
        dz = 0.04  # z方向オフセット
        node_coords = np.array(
            [
                [0.0, 0.0, 0.0],  # 0: segA start
                [1.0, 0.0, 0.0],  # 1: segA end
                [0.3, dy, dz],  # 2: segB start
                [0.7, dy + 0.1, dz],  # 3: segB end（わずかに傾斜）
            ]
        )
        connectivity = np.array([[0, 1], [2, 3]])
        elem_pairs = [(0, 1)]
    elif helical_z and n_elems >= 3:
        # Hermite 3要素チェーン + z方向傾き
        half = n_nodes // 2
        for i in range(half):
            x = node_coords[i, 0]
            node_coords[i, 2] += x * 0.003
        for i in range(half, n_nodes):
            x = node_coords[i, 0]
            node_coords[i, 2] -= x * 0.003

    ndof_per_node = 6
    ndof = n_nodes * ndof_per_node
    u = np.zeros(ndof)

    config = _ContactConfigInput(
        use_hermite_centerline=use_hermite,
        consistent_st_tangent=True,
        use_geometric_stiffness=True,
    )

    # 初期ペア構築（geometryはUpdateGeometryProcessで再計算）
    pairs = []
    for ea, eb in elem_pairs:
        nodes_a = connectivity[ea]
        nodes_b = connectivity[eb]
        state = _ContactStateOutput(
            s=0.5,
            t=0.5,
            gap=gap_target,
            normal=np.array([0.0, 1.0, 0.0]),
            tangent1=np.array([1.0, 0.0, 0.0]),
            tangent2=np.array([0.0, 0.0, 1.0]),
            p_n=0.0,
        )
        pair = _ContactPairOutput(
            elem_a=ea,
            elem_b=eb,
            nodes_a=np.array(nodes_a),
            nodes_b=np.array(nodes_b),
            state=state,
            radius_a=r,
            radius_b=r,
        )
        pairs.append(pair)

    manager = _ContactManagerInput(
        pairs=pairs,
        config=config,
        connectivity=connectivity,
    )

    strategy = HuberContactForceProcess(
        ndof=ndof,
        ndof_per_node=ndof_per_node,
        penalty_exponent=penalty_exponent,
    )

    return {
        "node_coords": node_coords,
        "u": u,
        "manager": manager,
        "strategy": strategy,
        "k_pen": k_pen,
        "ndof": ndof,
        "ndof_per_node": ndof_per_node,
        "connectivity": connectivity,
        "n_nodes": n_nodes,
    }


def _update_geometry_and_evaluate(
    strategy: HuberContactForceProcess,
    manager: _ContactManagerInput,
    u: np.ndarray,
    node_coords_ref: np.ndarray,
    k_pen: float,
    ndof_per_node: int,
    connectivity: np.ndarray,
) -> np.ndarray:
    """u に対して geometry 更新 → f_c 計算を実行.

    Returns:
        f_c ベクトル（残差系符号: -f_c_raw。R = f_int + f_c - f_ext）
    """
    # 変形座標
    n_nodes = len(node_coords_ref)
    coords_def = node_coords_ref.copy()
    for i in range(n_nodes):
        coords_def[i, 0:3] += u[i * ndof_per_node : i * ndof_per_node + 3]

    # geometry 更新
    ug_out = UpdateGeometryProcess().process(
        UpdateGeometryInput(
            manager=manager,
            node_coords=coords_def,
            connectivity=connectivity,
            freeze_active_set=False,
            freeze_st=False,
        )
    )
    manager.pairs[:] = ug_out.manager.pairs

    # f_c 計算（符号反転: tangent()と整合させるため -f_c_raw を返す）
    f_c, _ = strategy.evaluate(u, manager, k_pen)
    return -f_c


def _compute_fd_kc(
    scenario: dict,
    eps: float = 1e-7,
    dof_indices: np.ndarray | None = None,
) -> sp.csr_matrix:
    """全DOF（または指定DOF）のFD K_c を列ごとに計算.

    Returns:
        FD_K_c: (ndof, len(dof_indices)) dense → sparse
    """
    s = scenario
    ndof = s["ndof"]

    if dof_indices is None:
        dof_indices = np.arange(ndof)

    # ベース f_c
    manager_base = copy.deepcopy(s["manager"])
    u_base = s["u"].copy()
    f_c_base = _update_geometry_and_evaluate(
        s["strategy"],
        manager_base,
        u_base,
        s["node_coords"],
        s["k_pen"],
        s["ndof_per_node"],
        s["connectivity"],
    )

    # FD 列の計算
    fd_cols = np.zeros((ndof, len(dof_indices)))
    for j_idx, j in enumerate(dof_indices):
        manager_pert = copy.deepcopy(s["manager"])
        u_pert = s["u"].copy()
        u_pert[j] += eps
        f_c_pert = _update_geometry_and_evaluate(
            s["strategy"],
            manager_pert,
            u_pert,
            s["node_coords"],
            s["k_pen"],
            s["ndof_per_node"],
            s["connectivity"],
        )
        fd_cols[:, j_idx] = (f_c_pert - f_c_base) / eps

    # sparse 全体行列として再構成
    rows, cols, vals = [], [], []
    for j_idx, j in enumerate(dof_indices):
        col_data = fd_cols[:, j_idx]
        nz = np.nonzero(np.abs(col_data) > 1e-30)[0]
        for r in nz:
            rows.append(r)
            cols.append(j)
            vals.append(col_data[r])

    if len(vals) > 0:
        return sp.coo_matrix(
            (vals, (rows, cols)),
            shape=(ndof, ndof),
        ).tocsr()
    return sp.csr_matrix((ndof, ndof))


def _get_analytical_kc(scenario: dict) -> sp.csr_matrix:
    """解析的 K_c を tangent() で取得."""
    s = scenario
    # まず geometry を更新して p_n を設定
    manager = copy.deepcopy(s["manager"])
    u = s["u"].copy()
    _update_geometry_and_evaluate(
        s["strategy"],
        manager,
        u,
        s["node_coords"],
        s["k_pen"],
        s["ndof_per_node"],
        s["connectivity"],
    )
    # tangent() で K_c を取得
    n_nodes = s["n_nodes"]
    ndpn = s["ndof_per_node"]
    coords_def = s["node_coords"].copy()
    for i in range(n_nodes):
        coords_def[i, 0:3] += u[i * ndpn : i * ndpn + 3]

    return s["strategy"].tangent(u, manager, s["k_pen"], node_coords=coords_def)


def _get_analytical_components(
    scenario: dict,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """K_mat, K_geo, K_st を tangent_components() で取得."""
    s = scenario
    manager = copy.deepcopy(s["manager"])
    u = s["u"].copy()
    _update_geometry_and_evaluate(
        s["strategy"],
        manager,
        u,
        s["node_coords"],
        s["k_pen"],
        s["ndof_per_node"],
        s["connectivity"],
    )

    n_nodes = s["n_nodes"]
    ndpn = s["ndof_per_node"]
    coords_def = s["node_coords"].copy()
    for i in range(n_nodes):
        coords_def[i, 0:3] += u[i * ndpn : i * ndpn + 3]

    return s["strategy"].tangent_components(u, manager, s["k_pen"], node_coords=coords_def)


class TestKcComponentFD:
    """K_c コンポーネント個別FD検証（status-291）."""

    def _run_fd_verification(
        self,
        *,
        use_hermite: bool = False,
        penalty_exponent: float = 1.0,
        n_elems: int = 1,
        eps: float = 1e-7,
        atol: float = 1e-4,
        rtol: float = 0.1,
        helical_z: bool = False,
    ) -> dict:
        """FD検証を実行して結果を返す."""
        scenario = _make_two_segment_scenario(
            use_hermite=use_hermite,
            penalty_exponent=penalty_exponent,
            n_elems=n_elems,
            helical_z=helical_z,
        )

        # 関連DOFのみ（並進DOFのみ、計算時間削減）
        ndpn = scenario["ndof_per_node"]
        n_nodes = scenario["n_nodes"]
        trans_dofs = []
        for node in range(n_nodes):
            for d in range(3):
                trans_dofs.append(node * ndpn + d)
        trans_dofs = np.array(trans_dofs, dtype=int)

        # FD K_c
        fd_kc = _compute_fd_kc(scenario, eps=eps, dof_indices=trans_dofs)

        # 解析的コンポーネント
        K_mat, K_geo, K_st = _get_analytical_components(scenario)
        K_c_analytical = K_mat - K_geo + K_st

        # コンポーネント別ノルム
        norm_kmat = sp.linalg.norm(K_mat)
        norm_kgeo = sp.linalg.norm(K_geo)
        norm_kst = sp.linalg.norm(K_st)
        norm_kc = sp.linalg.norm(K_c_analytical)
        norm_fd = sp.linalg.norm(fd_kc)

        # 差分
        diff = K_c_analytical - fd_kc
        norm_diff = sp.linalg.norm(diff)
        rel_err = norm_diff / max(norm_fd, 1e-30)

        result = {
            "K_mat_norm": norm_kmat,
            "K_geo_norm": norm_kgeo,
            "K_st_norm": norm_kst,
            "K_c_analytical_norm": norm_kc,
            "K_c_fd_norm": norm_fd,
            "diff_norm": norm_diff,
            "rel_err": rel_err,
            "K_mat": K_mat,
            "K_geo": K_geo,
            "K_st": K_st,
            "K_c_analytical": K_c_analytical,
            "fd_kc": fd_kc,
        }

        return result

    def test_linear_penalty_no_hermite(self):
        """線形ペナルティ + 線形形状関数での K_c FD検証."""
        result = self._run_fd_verification(
            use_hermite=False,
            penalty_exponent=1.0,
            n_elems=1,
        )
        print("\n=== 線形ペナルティ + 線形形状関数 ===")
        print(f"  ||K_mat|| = {result['K_mat_norm']:.4e}")
        print(f"  ||K_geo|| = {result['K_geo_norm']:.4e}")
        print(f"  ||K_st||  = {result['K_st_norm']:.4e}")
        print(f"  ||K_c||   = {result['K_c_analytical_norm']:.4e}")
        print(f"  ||FD_Kc|| = {result['K_c_fd_norm']:.4e}")
        print(f"  ||diff||  = {result['diff_norm']:.4e}")
        print(f"  rel_err   = {result['rel_err']:.4e}")

        # 解析的 K_c がFDと一致すること
        assert result["rel_err"] < 0.1, (
            f"K_c FD不整合: rel_err={result['rel_err']:.4e} "
            f"(K_mat={result['K_mat_norm']:.2e}, "
            f"K_geo={result['K_geo_norm']:.2e}, "
            f"K_st={result['K_st_norm']:.2e})"
        )

    def test_linear_penalty_hermite(self):
        """線形ペナルティ + Hermite形状関数での K_c FD検証."""
        result = self._run_fd_verification(
            use_hermite=True,
            penalty_exponent=1.0,
            n_elems=3,  # Hermiteはチェーン構造が必要
        )
        print("\n=== 線形ペナルティ + Hermite ===")
        print(f"  ||K_mat|| = {result['K_mat_norm']:.4e}")
        print(f"  ||K_geo|| = {result['K_geo_norm']:.4e}")
        print(f"  ||K_st||  = {result['K_st_norm']:.4e}")
        print(f"  ||K_c||   = {result['K_c_analytical_norm']:.4e}")
        print(f"  ||FD_Kc|| = {result['K_c_fd_norm']:.4e}")
        print(f"  ||diff||  = {result['diff_norm']:.4e}")
        print(f"  rel_err   = {result['rel_err']:.4e}")

        # 不整合がある場合はコンポーネント割合を出力
        if result["rel_err"] > 0.1:
            self._report_component_breakdown(result)

    def test_hertz_penalty_no_hermite(self):
        """Hertz型ペナルティ（p=1.5）+ 線形形状関数での K_c FD検証."""
        result = self._run_fd_verification(
            use_hermite=False,
            penalty_exponent=1.5,
            n_elems=1,
        )
        print("\n=== Hertz型ペナルティ + 線形形状関数 ===")
        print(f"  ||K_mat|| = {result['K_mat_norm']:.4e}")
        print(f"  ||K_geo|| = {result['K_geo_norm']:.4e}")
        print(f"  ||K_st||  = {result['K_st_norm']:.4e}")
        print(f"  ||K_c||   = {result['K_c_analytical_norm']:.4e}")
        print(f"  ||FD_Kc|| = {result['K_c_fd_norm']:.4e}")
        print(f"  ||diff||  = {result['diff_norm']:.4e}")
        print(f"  rel_err   = {result['rel_err']:.4e}")

        if result["rel_err"] > 0.1:
            self._report_component_breakdown(result)

    def test_hertz_penalty_hermite(self):
        """Hertz型ペナルティ + Hermiteでの K_c FD検証（status-290の再現構成）."""
        result = self._run_fd_verification(
            use_hermite=True,
            penalty_exponent=1.5,
            n_elems=3,
        )
        print("\n=== Hertz型ペナルティ + Hermite（90度曲げ相当構成）===")
        print(f"  ||K_mat|| = {result['K_mat_norm']:.4e}")
        print(f"  ||K_geo|| = {result['K_geo_norm']:.4e}")
        print(f"  ||K_st||  = {result['K_st_norm']:.4e}")
        print(f"  ||K_c||   = {result['K_c_analytical_norm']:.4e}")
        print(f"  ||FD_Kc|| = {result['K_c_fd_norm']:.4e}")
        print(f"  ||diff||  = {result['diff_norm']:.4e}")
        print(f"  rel_err   = {result['rel_err']:.4e}")

        if result["rel_err"] > 0.1:
            self._report_component_breakdown(result)

    def test_helical_3d_hermite(self):
        """3Dヘリカル配置 + Hermiteでの K_c FD検証（status-292: z方向DOFカップリング）.

        status-289でz方向不整合が発見された。ヘリカル配置で法線にz成分があるとき
        K_stがz方向DOFへのカップリングを正しく含んでいるかを検証する。
        """
        result = self._run_fd_verification(
            use_hermite=True,
            penalty_exponent=1.5,
            n_elems=3,
            helical_z=True,
        )
        print("\n=== 3Dヘリカル + Hermite + Hertz ===")
        print(f"  ||K_mat|| = {result['K_mat_norm']:.4e}")
        print(f"  ||K_geo|| = {result['K_geo_norm']:.4e}")
        print(f"  ||K_st||  = {result['K_st_norm']:.4e}")
        print(f"  ||K_c||   = {result['K_c_analytical_norm']:.4e}")
        print(f"  ||FD_Kc|| = {result['K_c_fd_norm']:.4e}")
        print(f"  ||diff||  = {result['diff_norm']:.4e}")
        print(f"  rel_err   = {result['rel_err']:.4e}")

        # z方向不整合の分析
        self._report_z_direction_analysis(result)

        if result["rel_err"] > 0.1:
            self._report_component_breakdown(result)

    def test_helical_3d_linear(self):
        """3Dヘリカル配置 + 線形での K_c FD検証（z方向ベースライン）."""
        result = self._run_fd_verification(
            use_hermite=False,
            penalty_exponent=1.5,
            n_elems=1,
            helical_z=True,
        )
        print("\n=== 3Dヘリカル + 線形 + Hertz ===")
        print(f"  ||K_mat|| = {result['K_mat_norm']:.4e}")
        print(f"  ||K_geo|| = {result['K_geo_norm']:.4e}")
        print(f"  ||K_st||  = {result['K_st_norm']:.4e}")
        print(f"  ||K_c||   = {result['K_c_analytical_norm']:.4e}")
        print(f"  ||FD_Kc|| = {result['K_c_fd_norm']:.4e}")
        print(f"  ||diff||  = {result['diff_norm']:.4e}")
        print(f"  rel_err   = {result['rel_err']:.4e}")

        if result["rel_err"] > 0.1:
            self._report_component_breakdown(result)

    def _report_z_direction_analysis(self, result: dict) -> None:
        """z方向DOFの不整合を特定分析（status-292）."""
        diff_dense = (result["K_c_analytical"] - result["fd_kc"]).toarray()
        ndpn = 6
        n_nodes = diff_dense.shape[0] // ndpn

        # comp別の不整合ノルム
        comp_names = ["x", "y", "z", "θx", "θy", "θz"]
        comp_errs = np.zeros(6)
        for c in range(6):
            dofs_c = [nd * ndpn + c for nd in range(n_nodes)]
            comp_errs[c] = np.linalg.norm(diff_dense[dofs_c, :])

        total_err = np.linalg.norm(diff_dense)
        print("  --- comp別不整合ノルム ---")
        for c in range(6):
            pct = comp_errs[c] / max(total_err, 1e-30) * 100
            print(f"    {comp_names[c]:>3}: {comp_errs[c]:.4e} ({pct:.1f}%)")

    def test_component_magnitudes(self):
        """全構成のコンポーネント割合を一覧出力（診断用）."""
        configs = [
            ("linear+linear", False, 1.0, 1),
            ("linear+hermite", True, 1.0, 3),
            ("hertz+linear", False, 1.5, 1),
            ("hertz+hermite", True, 1.5, 3),
        ]
        print(
            f"\n{'config':<20} {'K_mat':>10} {'K_geo':>10} {'K_st':>10} {'K_c':>10} {'FD':>10} {'err':>10}"
        )
        print("-" * 80)
        for name, hermite, exp, ne in configs:
            result = self._run_fd_verification(
                use_hermite=hermite,
                penalty_exponent=exp,
                n_elems=ne,
            )
            print(
                f"{name:<20} "
                f"{result['K_mat_norm']:>10.2e} "
                f"{result['K_geo_norm']:>10.2e} "
                f"{result['K_st_norm']:>10.2e} "
                f"{result['K_c_analytical_norm']:>10.2e} "
                f"{result['K_c_fd_norm']:>10.2e} "
                f"{result['rel_err']:>10.2e}"
            )

    def _report_component_breakdown(self, result: dict) -> None:
        """コンポーネント別不整合の分解レポート."""
        fd = result["fd_kc"]
        K_mat = result["K_mat"]
        K_geo = result["K_geo"]
        K_st = result["K_st"]

        # K_mat のみで FD とどれだけ近いか
        diff_mat_only = K_mat - fd
        diff_geo_only = -K_geo - fd
        diff_st_only = K_st - fd

        print("  --- コンポーネント分解 ---")
        print(
            f"  ||K_mat - FD|| / ||FD|| = {sp.linalg.norm(diff_mat_only) / max(sp.linalg.norm(fd), 1e-30):.4e}"
        )
        print(
            f"  ||-K_geo - FD|| / ||FD|| = {sp.linalg.norm(diff_geo_only) / max(sp.linalg.norm(fd), 1e-30):.4e}"
        )
        print(
            f"  ||K_st - FD|| / ||FD|| = {sp.linalg.norm(diff_st_only) / max(sp.linalg.norm(fd), 1e-30):.4e}"
        )
        print(
            f"  ||K_mat|| / ||FD|| = {sp.linalg.norm(K_mat) / max(sp.linalg.norm(fd), 1e-30):.4e}"
        )
        print(
            f"  ||K_geo|| / ||FD|| = {sp.linalg.norm(K_geo) / max(sp.linalg.norm(fd), 1e-30):.4e}"
        )
        print(f"  ||K_st|| / ||FD|| = {sp.linalg.norm(K_st) / max(sp.linalg.norm(fd), 1e-30):.4e}")

        # DOFレベルの不整合上位
        diff_dense = (result["K_c_analytical"] - fd).toarray()
        fd_dense = fd.toarray()
        # 列ごとの不整合
        col_errs = np.linalg.norm(diff_dense, axis=0)
        col_fds = np.linalg.norm(fd_dense, axis=0)
        top_cols = np.argsort(col_errs)[::-1][:10]
        ndpn = 6
        comp_names = ["x", "y", "z", "θx", "θy", "θz"]
        print("  --- 不整合上位10 DOF列 ---")
        for j in top_cols:
            if col_errs[j] > 1e-30:
                node = j // ndpn
                comp = j % ndpn
                cname = comp_names[comp] if comp < 6 else f"c{comp}"
                print(
                    f"    dof={j} (node={node}, {cname}): "
                    f"err={col_errs[j]:.4e}, fd={col_fds[j]:.4e}, "
                    f"ratio={col_errs[j] / max(col_fds[j], 1e-30):.2f}"
                )
