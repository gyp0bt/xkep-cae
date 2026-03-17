#!/usr/bin/env python3
"""7本撚線 曲げ揺動 収束診断スクリプト.

収束失敗の根本原因を特定するため、以下の観点で詳細ログを取得:
1. 貫入/形状モデリング: 初期貫入量、gap値の変動
2. アクティブセットチャタリング: 反復ごとのactive set変動
3. 時間増分管理: adaptive timesteppingの挙動
4. 要素定式化: CR梁の接線剛性の条件数
5. NCP定式化: Fischer-Burmeister残差の内訳

[← README](../README.md)
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np

sys.path.insert(0, ".")

from xkep_cae.contact.pair import ContactConfig, ContactManager, ContactStatus
from xkep_cae.contact.solver_ncp import newton_raphson_contact_ncp
from xkep_cae.elements.beam_timo3d import assemble_cr_beam3d
from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh
from xkep_cae.sections.beam import BeamSectionInput

# ====================================================================
# パラメータ
# ====================================================================
_WIRE_D = 0.002  # 2 mm
_E = 200e9
_NU = 0.3
_NDOF = 6
_PITCH = 0.040
_N_ELEMS_PER_PITCH = 16


def _G(E, nu):
    return E / (2.0 * (1.0 + nu))


def _kappa(nu):
    return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)


# ====================================================================
# 診断1: 初期メッシュ形状の解析
# ====================================================================
def diagnose_mesh_geometry(n_strands=7, n_pitches=0.5):
    """メッシュ生成後の初期形状を解析: 素線間距離、初期貫入を確認."""
    print("\n" + "=" * 70)
    print("  診断1: メッシュ形状解析")
    print("=" * 70)

    mesh = make_twisted_wire_mesh(
        n_strands,
        _WIRE_D,
        _PITCH,
        length=0.0,
        n_elems_per_strand=max(4, int(round(_N_ELEMS_PER_PITCH * n_pitches))),
        n_pitches=n_pitches,
        min_elems_per_pitch=16,
    )

    section = BeamSectionInput.circle(_WIRE_D)
    r = _WIRE_D / 2.0

    print(f"  素線数: {n_strands}")
    print(f"  要素数: {mesh.n_elems}, 節点数: {mesh.n_nodes}")
    print(f"  モデル長: {mesh.length * 1000:.2f} mm")
    print(f"  素線半径: {r * 1000:.3f} mm")

    # 各素線の端点座標
    print("\n  素線端点座標 (z=0):")
    for sid in range(n_strands):
        nodes = mesh.strand_nodes(sid)
        n0 = nodes[0]
        x, y, z = mesh.node_coords[n0]
        print(f"    strand {sid}: ({x * 1000:.4f}, {y * 1000:.4f}, {z * 1000:.4f}) mm")

    # 隣接素線間の最小距離
    print("\n  隣接素線間の距離 (全節点中の最小):")
    coords = mesh.node_coords
    min_dists = {}
    for si in range(n_strands):
        for sj in range(si + 1, n_strands):
            nodes_i = mesh.strand_nodes(si)
            nodes_j = mesh.strand_nodes(sj)
            min_d = float("inf")
            for ni in nodes_i:
                for nj in nodes_j:
                    d = np.linalg.norm(coords[ni] - coords[nj])
                    if d < min_d:
                        min_d = d
            gap = min_d - 2 * r
            min_dists[(si, sj)] = (min_d, gap)
            if gap < 0:
                print(
                    f"    *** 貫入検出 *** strand {si}-{sj}: "
                    f"dist={min_d * 1000:.4f} mm, gap={gap * 1000:.4f} mm, "
                    f"pen_ratio={abs(gap) / (2 * r):.4f}"
                )
            elif gap < r * 0.1:
                print(
                    f"    接近 strand {si}-{sj}: "
                    f"dist={min_d * 1000:.4f} mm, gap={gap * 1000:.4f} mm"
                )

    # ContactManager で初期貫入を検出
    elem_layer_map = mesh.build_elem_layer_map()
    mgr = ContactManager(
        config=ContactConfig(
            k_pen_mode="beam_ei",
            k_pen_scale=0.1,
            beam_E=_E,
            beam_I=section.Iy,
            beam_A=section.A,
            g_on=0.0005,
            g_off=0.001,
            elem_layer_map=elem_layer_map,
            exclude_same_layer=True,
            midpoint_prescreening=True,
            line_contact=True,
            use_mortar=False,
        ),
    )
    mgr.detect_candidates(mesh.node_coords, mesh.connectivity, mesh.radii, margin=0.01)
    n_pen = mgr.check_initial_penetration(mesh.node_coords)

    print(f"\n  ContactManager 初期貫入ペア数: {n_pen}/{len(mgr.pairs)}")
    if n_pen > 0:
        print(f"  初期貫入検出: {n_pen}ペア")
    print(f"  全候補ペア数: {len(mgr.pairs)}")

    # 法線角度の不連続性チェック（隣接セグメント間）
    print("\n  接触ペアの法線方向:")
    active_pairs = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
    if active_pairs:
        normals = [p.state.normal for p in active_pairs[:10]]
        for i, n in enumerate(normals):
            print(f"    pair {i}: normal=({n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f})")
    else:
        print("    初期状態でアクティブペアなし")

    return mesh, mgr


# ====================================================================
# 診断2: 段階的曲げ角度での収束テスト
# ====================================================================
def diagnose_incremental_bending(angles_deg=None):
    """異なる曲げ角度でNCP収束を試行し、限界角度を特定."""
    if angles_deg is None:
        angles_deg = [5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0]

    print("\n" + "=" * 70)
    print("  診断2: 段階的曲げ角度テスト")
    print("=" * 70)

    results = {}
    for angle in angles_deg:
        print(f"\n--- 曲げ角度: {angle}° ---")
        n_steps = max(5, int(angle / 5))  # 5°あたり1ステップ

        try:
            result = _run_ncp_bending(
                bend_angle_deg=angle,
                n_bending_steps=n_steps,
                max_iter=40,
                tol_force=1e-4,
                show_progress=True,
                adaptive_timestepping=True,
            )
            results[angle] = {
                "converged": result.converged,
                "n_steps": result.n_load_steps,
                "total_iters": result.total_newton_iterations,
                "n_active": result.n_active_final,
            }
            print(
                f"  結果: converged={result.converged}, "
                f"steps={result.n_load_steps}, iters={result.total_newton_iterations}, "
                f"active={result.n_active_final}"
            )
        except Exception as e:
            results[angle] = {"converged": False, "error": str(e)}
            print(f"  エラー: {e}")
            traceback.print_exc()

        # 失敗したらこれ以上の角度はスキップ
        if not results[angle].get("converged", False):
            print(f"\n  *** {angle}°で収束失敗 — 以降の角度をスキップ ***")
            break

    print("\n" + "=" * 70)
    print("  段階的曲げ角度テスト結果サマリー")
    print("=" * 70)
    for angle, r in sorted(results.items()):
        status = "OK" if r.get("converged") else "FAIL"
        print(f"  {angle:6.1f}° : {status}  {r}")

    return results


# ====================================================================
# 診断3: アクティブセットチャタリング詳細解析
# ====================================================================
def diagnose_active_set_chattering(angle_deg=30.0):
    """特定角度での反復ごとのアクティブセット変動を詳細に追跡."""
    print("\n" + "=" * 70)
    print(f"  診断3: アクティブセットチャタリング解析 ({angle_deg}°)")
    print("=" * 70)

    # ここではNCPソルバーの内部ログを直接取得するために、
    # 小さいステップで実行し、各ステップの詳細を観察する
    n_steps = max(5, int(angle_deg / 3))

    result = _run_ncp_bending(
        bend_angle_deg=angle_deg,
        n_bending_steps=n_steps,
        max_iter=50,
        tol_force=1e-4,
        show_progress=True,  # 全反復の詳細表示
        adaptive_timestepping=True,
    )

    print(f"\n  収束: {result.converged}")
    print(f"  総NR反復: {result.total_newton_iterations}")
    print(f"  ロードステップ: {result.n_load_steps}")
    print(f"  最終アクティブペア: {result.n_active_final}")

    # load_historyから各ステップの収束状況を分析
    if hasattr(result, "load_history") and result.load_history:
        print(f"\n  ロード履歴: {len(result.load_history)} ステップ")
        for i, lh in enumerate(result.load_history[:20]):
            print(f"    step {i}: load_frac={lh:.4f}")

    return result


# ====================================================================
# 診断4: k_pen感度解析
# ====================================================================
def diagnose_kpen_sensitivity(angle_deg=20.0):
    """異なるk_penでの収束挙動を比較."""
    print("\n" + "=" * 70)
    print(f"  診断4: k_pen感度解析 ({angle_deg}°)")
    print("=" * 70)

    # 自動推定値の確認
    mesh = make_twisted_wire_mesh(
        7,
        _WIRE_D,
        _PITCH,
        length=0.0,
        n_elems_per_strand=max(4, int(round(_N_ELEMS_PER_PITCH * 0.5))),
        n_pitches=0.5,
        min_elems_per_pitch=16,
    )
    section = BeamSectionInput.circle(_WIRE_D)
    L_elem = mesh.length / (mesh.n_elems / 7)

    # beam_ei mode: k_pen = scale * 12*E*I/L^3
    k_pen_beam_ei = 0.1 * 12 * _E * section.Iy / L_elem**3
    # ea_l mode: k_pen = scale * E*A/L
    k_pen_ea_l = 0.1 * _E * section.A / L_elem

    print(f"  推定 k_pen (beam_ei, scale=0.1): {k_pen_beam_ei:.3e}")
    print(f"  推定 k_pen (ea_l, scale=0.1): {k_pen_ea_l:.3e}")
    print(f"  要素長: {L_elem * 1000:.4f} mm")

    kpen_values = [0.0, k_pen_beam_ei * 0.1, k_pen_beam_ei, k_pen_beam_ei * 10]
    n_steps = max(5, int(angle_deg / 3))

    for kp in kpen_values:
        print(f"\n--- k_pen = {kp:.3e} ---")
        try:
            result = _run_ncp_bending(
                bend_angle_deg=angle_deg,
                n_bending_steps=n_steps,
                max_iter=40,
                tol_force=1e-4,
                show_progress=True,
                adaptive_timestepping=True,
                ncp_k_pen=kp,
            )
            print(f"  結果: converged={result.converged}, iters={result.total_newton_iterations}")
        except Exception as e:
            print(f"  エラー: {e}")


# ====================================================================
# 診断5: adaptive_timestepping ON/OFF比較
# ====================================================================
def diagnose_adaptive_timestepping(angle_deg=30.0):
    """adaptive timesteppingのON/OFF比較."""
    print("\n" + "=" * 70)
    print(f"  診断5: adaptive timestepping比較 ({angle_deg}°)")
    print("=" * 70)

    n_steps_list = [10, 20, 30, 50]

    for adaptive in [True, False]:
        for n_steps in n_steps_list:
            label = f"adaptive={'ON' if adaptive else 'OFF'}, n_steps={n_steps}"
            print(f"\n--- {label} ---")
            try:
                t0 = time.perf_counter()
                result = _run_ncp_bending(
                    bend_angle_deg=angle_deg,
                    n_bending_steps=n_steps,
                    max_iter=40,
                    tol_force=1e-4,
                    show_progress=False,
                    adaptive_timestepping=adaptive,
                )
                dt = time.perf_counter() - t0
                print(
                    f"  {label}: converged={result.converged}, "
                    f"iters={result.total_newton_iterations}, time={dt:.1f}s"
                )
            except Exception as e:
                print(f"  {label}: エラー={e}")

            # 収束したら次のn_stepsはスキップ（効率のため）
            if adaptive and result.converged:
                break


# ====================================================================
# 診断6: Mortar ON/OFF比較
# ====================================================================
def diagnose_mortar_effect(angle_deg=30.0):
    """Mortar有無での収束比較."""
    print("\n" + "=" * 70)
    print(f"  診断6: Mortar有無比較 ({angle_deg}°)")
    print("=" * 70)

    n_steps = max(10, int(angle_deg / 3))

    for mortar in [True, False]:
        label = f"mortar={'ON' if mortar else 'OFF'}"
        print(f"\n--- {label} ---")
        try:
            t0 = time.perf_counter()
            result = _run_ncp_bending(
                bend_angle_deg=angle_deg,
                n_bending_steps=n_steps,
                max_iter=40,
                tol_force=1e-4,
                show_progress=True,
                adaptive_timestepping=True,
                use_mortar=mortar,
            )
            dt = time.perf_counter() - t0
            print(
                f"  {label}: converged={result.converged}, "
                f"iters={result.total_newton_iterations}, time={dt:.1f}s, "
                f"active={result.n_active_final}"
            )
        except Exception as e:
            print(f"  {label}: エラー={e}")


# ====================================================================
# 診断7: 接線剛性の条件数チェック
# ====================================================================
def diagnose_tangent_condition(angle_deg=20.0):
    """曲げ変形中の接線剛性行列の条件数をモニタリング."""
    print("\n" + "=" * 70)
    print(f"  診断7: 接線剛性条件数チェック ({angle_deg}°)")
    print("=" * 70)

    mesh = make_twisted_wire_mesh(
        7,
        _WIRE_D,
        _PITCH,
        length=0.0,
        n_elems_per_strand=max(4, int(round(_N_ELEMS_PER_PITCH * 0.5))),
        n_pitches=0.5,
        min_elems_per_pitch=16,
    )
    section = BeamSectionInput.circle(_WIRE_D)
    G = _G(_E, _NU)
    kappa = _kappa(_NU)

    # 初期状態の接線剛性
    ndof = mesh.n_nodes * _NDOF
    u_zero = np.zeros(ndof)

    K_T, _ = assemble_cr_beam3d(
        mesh.node_coords,
        mesh.connectivity,
        u_zero,
        _E,
        G,
        section.A,
        section.Iy,
        section.Iz,
        section.J,
        kappa,
        kappa,
        stiffness=True,
        internal_force=False,
    )

    from scipy.sparse.linalg import eigsh

    # 境界条件適用前の最大・最小固有値
    try:
        K_dense = K_T.toarray()
        # 固定DOFを除外
        fixed = set()
        for sid in range(7):
            nodes = mesh.strand_nodes(sid)
            n0 = nodes[0]
            for d in range(_NDOF):
                fixed.add(_NDOF * n0 + d)
        free = sorted(set(range(ndof)) - fixed)
        K_free = K_dense[np.ix_(free, free)]

        eig_max = eigsh(K_free.astype(np.float64), k=1, which="LM", return_eigenvectors=False)[0]
        eig_min = eigsh(
            K_free.astype(np.float64),
            k=1,
            which="SM",
            return_eigenvectors=False,
            sigma=0,
            mode="normal",
        )[0]
        cond = abs(eig_max / eig_min) if abs(eig_min) > 1e-30 else float("inf")
        print(f"  初期状態:")
        print(f"    最大固有値: {eig_max:.3e}")
        print(f"    最小固有値: {eig_min:.3e}")
        print(f"    条件数: {cond:.3e}")
        print(f"    自由DOF数: {len(free)}")
    except Exception as e:
        print(f"  固有値計算エラー: {e}")


# ====================================================================
# 診断8: tol_force感度
# ====================================================================
def diagnose_tolerance_sensitivity(angle_deg=30.0):
    """異なる収束公差での挙動比較."""
    print("\n" + "=" * 70)
    print(f"  診断8: 収束公差感度 ({angle_deg}°)")
    print("=" * 70)

    n_steps = max(10, int(angle_deg / 3))
    tols = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]

    for tol in tols:
        print(f"\n--- tol_force = {tol:.0e} ---")
        try:
            t0 = time.perf_counter()
            result = _run_ncp_bending(
                bend_angle_deg=angle_deg,
                n_bending_steps=n_steps,
                max_iter=50,
                tol_force=tol,
                show_progress=False,
                adaptive_timestepping=True,
            )
            dt = time.perf_counter() - t0
            print(
                f"  tol={tol:.0e}: converged={result.converged}, "
                f"iters={result.total_newton_iterations}, time={dt:.1f}s"
            )
        except Exception as e:
            print(f"  tol={tol:.0e}: エラー={e}")


# ====================================================================
# ヘルパー: NCP曲げ実行
# ====================================================================
def _run_ncp_bending(
    bend_angle_deg=45.0,
    n_bending_steps=10,
    max_iter=30,
    tol_force=1e-4,
    show_progress=True,
    adaptive_timestepping=True,
    use_mortar=True,
    ncp_k_pen=0.0,
    n_pitches=0.5,
):
    """7本撚線NCP曲げを実行（Phase1のみ）."""
    mesh = make_twisted_wire_mesh(
        7,
        _WIRE_D,
        _PITCH,
        length=0.0,
        n_elems_per_strand=max(4, int(round(_N_ELEMS_PER_PITCH * n_pitches))),
        n_pitches=n_pitches,
        min_elems_per_pitch=16,
    )
    section = BeamSectionInput.circle(_WIRE_D)
    G = _G(_E, _NU)
    kappa = _kappa(_NU)
    ndof = mesh.n_nodes * _NDOF

    def assemble_tangent(u):
        K_T, _ = assemble_cr_beam3d(
            mesh.node_coords,
            mesh.connectivity,
            u,
            _E,
            G,
            section.A,
            section.Iy,
            section.Iz,
            section.J,
            kappa,
            kappa,
            stiffness=True,
            internal_force=False,
        )
        return K_T

    def assemble_internal_force(u):
        _, f_int = assemble_cr_beam3d(
            mesh.node_coords,
            mesh.connectivity,
            u,
            _E,
            G,
            section.A,
            section.Iy,
            section.Iz,
            section.J,
            kappa,
            kappa,
            stiffness=False,
            internal_force=True,
        )
        return f_int

    # 固定DOF
    fixed = set()
    for sid in range(7):
        nodes = mesh.strand_nodes(sid)
        n0 = nodes[0]
        for d in range(_NDOF):
            fixed.add(_NDOF * n0 + d)
    fixed_dofs = np.array(sorted(fixed), dtype=int)

    # 処方変位: rx DOF
    rx_dofs = []
    for sid in range(7):
        nodes = mesh.strand_nodes(sid)
        n_end = nodes[-1]
        rx_dofs.append(_NDOF * n_end + 3)
    rx_dofs_arr = np.array(rx_dofs, dtype=int)
    bend_angle_rad = np.deg2rad(bend_angle_deg)
    prescribed_vals = np.full(len(rx_dofs), bend_angle_rad)

    # ContactManager
    elem_layer_map = mesh.build_elem_layer_map()
    mgr = ContactManager(
        config=ContactConfig(
            k_pen_mode="beam_ei",
            k_pen_scale=0.1,
            beam_E=_E,
            beam_I=section.Iy,
            beam_A=section.A,
            g_on=0.0005,
            g_off=0.001,
            elem_layer_map=elem_layer_map,
            exclude_same_layer=True,
            midpoint_prescreening=True,
            line_contact=True,
            use_mortar=use_mortar,
            n_gauss=2,
            use_geometric_stiffness=True,
        ),
    )
    mgr.detect_candidates(mesh.node_coords, mesh.connectivity, mesh.radii, margin=0.01)
    mgr.check_initial_penetration(mesh.node_coords)

    result = newton_raphson_contact_ncp(
        np.zeros(ndof),
        fixed_dofs,
        assemble_tangent,
        assemble_internal_force,
        mgr,
        mesh.node_coords,
        mesh.connectivity,
        mesh.radii,
        n_load_steps=n_bending_steps,
        max_iter=max_iter,
        tol_force=tol_force,
        tol_ncp=tol_force,
        show_progress=show_progress,
        broadphase_margin=0.01,
        line_contact=True,
        use_mortar=use_mortar,
        n_gauss=2,
        k_pen=ncp_k_pen,
        adaptive_timestepping=adaptive_timestepping,
        modified_nr_threshold=5,
        prescribed_dofs=rx_dofs_arr,
        prescribed_values=prescribed_vals,
    )
    return result


# ====================================================================
# メイン
# ====================================================================
def main():
    print("=" * 70)
    print("  7本撚線 曲げ揺動 収束診断")
    print("  目的: 収束失敗の根本原因を特定する")
    print("=" * 70)

    t_start = time.perf_counter()

    # 1. メッシュ形状解析
    diagnose_mesh_geometry()

    # 2. 段階的曲げ角度テスト（最重要）
    angle_results = diagnose_incremental_bending(
        angles_deg=[5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0]
    )

    # 収束限界角度を特定
    max_converged_angle = 0.0
    for angle, r in sorted(angle_results.items()):
        if r.get("converged", False):
            max_converged_angle = angle
    first_fail_angle = min(
        (a for a, r in angle_results.items() if not r.get("converged", False)),
        default=None,
    )

    print(f"\n  *** 収束限界角度: {max_converged_angle}° ***")
    if first_fail_angle:
        print(f"  *** 最初の失敗角度: {first_fail_angle}° ***")

    # 3. 失敗角度付近でのチャタリング解析
    target_angle = first_fail_angle or 30.0
    diagnose_active_set_chattering(target_angle)

    # 4. k_pen感度
    diagnose_kpen_sensitivity(min(20.0, max_converged_angle or 20.0))

    # 5. adaptive timestepping ON/OFF
    diagnose_adaptive_timestepping(target_angle)

    # 6. Mortar ON/OFF
    diagnose_mortar_effect(target_angle)

    # 7. 接線剛性条件数
    diagnose_tangent_condition()

    # 8. 公差感度
    diagnose_tolerance_sensitivity(target_angle)

    t_total = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"  診断完了: 総時間 {t_total:.1f}s")
    print("=" * 70)

    # サマリー
    print("\n" + "=" * 70)
    print("  診断結果サマリー")
    print("=" * 70)
    print(f"  収束限界角度: {max_converged_angle}°")
    print(f"  最初の失敗: {first_fail_angle}°")
    print("  →原因特定のためには上記の各診断結果を確認してください")


if __name__ == "__main__":
    main()
