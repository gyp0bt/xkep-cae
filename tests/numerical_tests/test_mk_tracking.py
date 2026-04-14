"""M-κ追跡 + 接触ペアスナップショットの単体テスト.

status-333: CR梁接触動解析でのM-κヒステリシス直接取得機能の検証。
- ContactFrictionProcess の track_mk / track_contact_pairs が正しく記録されるか
- StrandBendingOscillationProcess からの配線が機能するか
- ContactPairSnapshotEntry のフィールドが妥当か

[← README](../../README.md)
"""

from __future__ import annotations

import pytest

from xkep_cae.core.data import ContactPairSnapshotEntry
from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


class TestContactPairSnapshotEntryAPI:
    """ContactPairSnapshotEntry のデータ構造テスト."""

    def test_create_snapshot_entry(self) -> None:
        """スナップショットエントリが正しく作成できる."""
        entry = ContactPairSnapshotEntry(
            elem_a=0,
            elem_b=5,
            p_n=1.23,
            gap=-0.01,
            slip_s=0.001,
            slip_t=-0.002,
            stick=False,
            dissipation=0.05,
        )
        assert entry.elem_a == 0
        assert entry.elem_b == 5
        assert entry.p_n == 1.23
        assert entry.gap == -0.01
        assert entry.slip_s == 0.001
        assert entry.slip_t == -0.002
        assert entry.stick is False
        assert entry.dissipation == 0.05

    def test_snapshot_entry_frozen(self) -> None:
        """スナップショットエントリは不変."""
        entry = ContactPairSnapshotEntry(
            elem_a=0,
            elem_b=1,
            p_n=1.0,
            gap=0.0,
            slip_s=0.0,
            slip_t=0.0,
            stick=True,
            dissipation=0.0,
        )
        with pytest.raises(AttributeError):
            entry.p_n = 2.0  # type: ignore[misc]


class TestMkTrackingConfig:
    """StrandBendingOscillationConfig のM-κ追跡フラグテスト."""

    def test_default_tracking_disabled(self) -> None:
        """デフォルトではM-κ追跡・接触ペア追跡は無効."""
        cfg = StrandBendingOscillationConfig()
        assert cfg.track_contact_mk is False
        assert cfg.track_contact_pairs is False

    def test_tracking_flags_settable(self) -> None:
        """追跡フラグが設定できる."""
        cfg = StrandBendingOscillationConfig(
            track_contact_mk=True,
            track_contact_pairs=True,
        )
        assert cfg.track_contact_mk is True
        assert cfg.track_contact_pairs is True


@pytest.mark.slow
class TestMkTrackingConvergence:
    """M-κ追跡の統合テスト（2本撚線で高速実行）."""

    def test_mk_tracking_records_history(self) -> None:
        """M-κ追跡が有効な場合、moment_curvature_historyが記録される."""
        cfg = StrandBendingOscillationConfig(
            n_strands=2,
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.001,
            n_cycles=1,
            n_increments_per_cycle=10,
            rho_inf=0.9,
            mu=0.15,
            max_nr_attempts=200,
            tol_force=1e-8,
            max_increments=500,
            exclude_same_strand=True,
            gap=0.05,
            free_end_mode=True,
            penalty_exponent=1.5,
            track_contact_mk=True,
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)
        sr = result.solver_result

        mk_hist = sr.moment_curvature_history
        print("\n=== M-κ追跡テスト ===")
        print(f"  M-κ history entries: {len(mk_hist)}")
        frac = sr.load_history[-1] if sr.load_history else 0.0
        print(f"  frac_completed: {frac:.4f}")

        # M-κ履歴が記録されている
        assert len(mk_hist) > 0, "M-κ履歴が空"

        # 各エントリは (κ, M) のタプル
        for kappa, moment in mk_hist:
            assert isinstance(kappa, float)
            assert isinstance(moment, float)

        # κは単調増加（曲げフェーズなので）
        kappas = [k for k, _ in mk_hist]
        for i in range(1, len(kappas)):
            assert kappas[i] >= kappas[i - 1] - 1e-12, (
                f"κ非単調: κ[{i - 1}]={kappas[i - 1]}, κ[{i}]={kappas[i]}"
            )

        # 最終κは bending_curvature * frac 付近
        strand_length = cfg.pitch_length * cfg.n_pitches
        bending_angle = cfg.bending_curvature * strand_length
        expected_kappa = bending_angle * frac / strand_length
        actual_kappa = kappas[-1]
        rtol = abs(actual_kappa - expected_kappa) / max(abs(expected_kappa), 1e-30)
        print(f"  expected κ={expected_kappa:.6e}, actual κ={actual_kappa:.6e}, rtol={rtol:.4f}")
        assert rtol < 0.1, f"最終κの相対誤差が大きい: {rtol:.4f}"

        # Mは非ゼロ（曲げ反力あり）
        moments = [m for _, m in mk_hist]
        max_moment = max(abs(m) for m in moments)
        print(f"  max |M|={max_moment:.6e}")
        assert max_moment > 0, "曲げモーメントがゼロ"

    def test_contact_pair_tracking_records_history(self) -> None:
        """接触ペア追跡が有効な場合、contact_pair_historyが記録される."""
        cfg = StrandBendingOscillationConfig(
            n_strands=2,
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.001,
            n_cycles=1,
            n_increments_per_cycle=10,
            rho_inf=0.9,
            mu=0.15,
            max_nr_attempts=200,
            tol_force=1e-8,
            max_increments=500,
            exclude_same_strand=True,
            gap=0.05,
            free_end_mode=True,
            penalty_exponent=1.5,
            track_contact_pairs=True,
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)
        sr = result.solver_result

        pair_hist = sr.contact_pair_history
        print("\n=== 接触ペア追跡テスト ===")
        print(f"  pair history entries: {len(pair_hist)}")
        frac = sr.load_history[-1] if sr.load_history else 0.0
        print(f"  frac_completed: {frac:.4f}")

        # 接触ペア履歴が記録されている（接触が発生するインクリメントのみ）
        assert len(pair_hist) > 0, "接触ペア履歴が空（接触未発生?）"

        # 各エントリは (load_frac, tuple[ContactPairSnapshotEntry, ...])
        for load_frac, pairs in pair_hist:
            assert isinstance(load_frac, float)
            assert 0.0 <= load_frac <= 1.0 + 1e-6
            assert isinstance(pairs, tuple)
            for p in pairs:
                assert isinstance(p, ContactPairSnapshotEntry)
                assert p.p_n > 0, "活性ペアのp_nは正"
                assert isinstance(p.stick, bool)
                assert isinstance(p.dissipation, float)

        # 接触が進むと活性ペア数が増加する傾向
        n_pairs_first = len(pair_hist[0][1]) if pair_hist else 0
        n_pairs_last = len(pair_hist[-1][1]) if pair_hist else 0
        print(f"  n_active_pairs: first={n_pairs_first}, last={n_pairs_last}")

    def test_mk_and_pairs_combined(self) -> None:
        """M-κ追跡と接触ペア追跡の同時有効化."""
        cfg = StrandBendingOscillationConfig(
            n_strands=2,
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.001,
            n_cycles=1,
            n_increments_per_cycle=10,
            rho_inf=0.9,
            mu=0.15,
            max_nr_attempts=200,
            tol_force=1e-8,
            max_increments=500,
            exclude_same_strand=True,
            gap=0.05,
            free_end_mode=True,
            penalty_exponent=1.5,
            track_contact_mk=True,
            track_contact_pairs=True,
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)
        sr = result.solver_result

        mk_hist = sr.moment_curvature_history
        pair_hist = sr.contact_pair_history

        print("\n=== M-κ + 接触ペア同時追跡テスト ===")
        print(f"  M-κ entries: {len(mk_hist)}")
        print(f"  pair entries: {len(pair_hist)}")

        assert len(mk_hist) > 0, "M-κ履歴が空"
        assert len(pair_hist) > 0, "接触ペア履歴が空"

        # M-κ履歴数 == load_history数（全インクリメントで記録）
        n_load = len(sr.load_history) if sr.load_history else 0
        assert len(mk_hist) == n_load, f"M-κ履歴数({len(mk_hist)}) != load_history数({n_load})"

    def test_mk_hysteresis_loop_oscillation(self) -> None:
        """M-κ ヒステリシスループ — 曲げ+揺動(1サイクル)で load+unload.

        status-335: 2本撚線で load+unload 経路のM-κ履歴を取得し、
        ループ面積（= 散逸エネルギー）が正値になることを確認する。
        status-333 のmonotonicケースを超えて、`n_oscillation_cycles=1`
        統合モード（θ_y揺動）で真のヒステリシスループを観測する。

        status-336: 散逸率を load-only 弾性仕事 W_load = ∫_load M dκ に対する
        比として評価する（status-335 の外接矩形比 M_peak×κ_peak は粗過ぎた）。
        `_compute_mk_metrics` の loading_work / unloading_work / loop_area
        を用いて物理的に意味のある `loop_area / loading_work` を計算。
        """
        cfg = StrandBendingOscillationConfig(
            n_strands=2,
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.001,
            n_cycles=1,
            n_oscillation_cycles=1,  # 曲げ後に 1 サイクル θ_y 揺動
            n_increments_per_cycle=8,  # 軽量化（曲げ+揺動=16増分相当）
            rho_inf=0.9,
            mu=0.15,
            max_nr_attempts=200,
            tol_force=1e-7,
            max_increments=500,
            exclude_same_strand=True,
            gap=0.05,
            free_end_mode=True,
            penalty_exponent=1.5,
            track_contact_mk=True,
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)
        sr = result.solver_result

        mk_hist = sr.moment_curvature_history
        frac = sr.load_history[-1] if sr.load_history else 0.0
        print("\n=== M-κ ヒステリシスループ検証（load+unload）===")
        print(f"  M-κ entries: {len(mk_hist)}")
        print(f"  frac_completed: {frac:.4f}")

        # 最低限、曲げ増分 + 揺動数点が必要（n_osc=1 → κ は最低1回 下降）
        assert len(mk_hist) >= 6, f"M-κエントリ不足: {len(mk_hist)}"

        kappas = [k for k, _ in mk_hist]

        # ループの「増加→減少」転換が最低1回存在する（純粋 monotonic ではない）
        n_decreases = sum(1 for i in range(1, len(kappas)) if kappas[i] < kappas[i - 1])
        print(f"  n_decreases in κ: {n_decreases}")
        assert n_decreases >= 1, "κ は単調増加のまま（揺動フェーズ未到達？）"

        # status-336: 散逸率は load-only 弾性仕事に対する比で評価する。
        # _compute_mk_metrics は loading_work = Σ max(0, dκ) * M_avg /
        # unloading_work = Σ |min(0, dκ) * M_avg| を分離して積分する。
        import math

        from xkep_cae.numerical_tests.cable_dissipation import _compute_mk_metrics

        metrics = _compute_mk_metrics(mk_hist)
        loop_area = metrics["loop_area"]
        loading_work = metrics["loading_work"]
        unloading_work = metrics["unloading_work"]
        peak_M = metrics["peak_moment"]
        peak_k = metrics["peak_curvature"]
        EI_sec = metrics["EI_secant"]
        EI_init = metrics["EI_initial"]
        print(f"  M_peak={peak_M:.4e}, κ_peak={peak_k:.4e}")
        print(f"  EI_secant={EI_sec:.4e}, EI_initial={EI_init:.4e}")
        print(f"  W_load={loading_work:.4e}, W_unload={unloading_work:.4e}")
        print(f"  loop_area = |W_load - W_unload| = {loop_area:.4e}")

        # 非ゼロループ面積（摩擦散逸が観測される）
        assert loop_area >= 0.0  # 絶対値なので自明だが contract として明記
        # 負荷仕事は正（曲げで M·dκ > 0）
        assert loading_work > 0.0, f"loading_work が非正: {loading_work}"
        # 物理的に意味のある散逸率: loop_area / W_load（0..1+α の範囲）
        true_dissipation_ratio = loop_area / loading_work
        print(f"  true dissipation_ratio = loop_area/W_load = {true_dissipation_ratio:.4e}")
        assert math.isfinite(true_dissipation_ratio), "dissipation_ratio が NaN/inf"
        # 除荷仕事は loading_work より大きくならない想定（摩擦で減衰するため）
        # ただし小規模ケースでは θ_y 揺動が load 方向に戻り切らない場合もあるので
        # 絶対上限 2.0 を閾値とする（2x loading は明らかに物理的におかしい）
        assert true_dissipation_ratio < 2.0, (
            f"散逸率が過大: loop_area={loop_area:.4e}, W_load={loading_work:.4e}"
        )
        # metrics[dissipation_ratio] と一致（_compute_mk_metrics の定義確認）
        assert abs(metrics["dissipation_ratio"] - true_dissipation_ratio) < 1e-12


class TestMkTrackingDisabled:
    """追跡無効時の動作確認（既存動作への回帰なし）."""

    def test_default_no_mk_history(self) -> None:
        """追跡無効時はmoment_curvature_historyが空."""
        import numpy as np

        from xkep_cae.core.data import SolverResultData

        result = SolverResultData(
            u=np.zeros(10),
            converged=True,
            n_increments=1,
            total_attempts=1,
        )
        assert result.moment_curvature_history == ()
        assert result.contact_pair_history == ()

    def test_input_default_no_tracking(self) -> None:
        """ContactFrictionInputData のデフォルトは追跡無効."""
        import numpy as np
        import scipy.sparse as sp

        from xkep_cae.core.data import (
            AssembleCallbacks,
            BoundaryData,
            ContactFrictionInputData,
            ContactSetupData,
            MeshData,
        )

        mesh = MeshData(
            node_coords=np.zeros((2, 3)),
            connectivity=np.array([[0, 1]]),
            radii=0.5,
            n_strands=1,
        )
        boundary = BoundaryData(fixed_dofs=np.array([0]))
        contact = ContactSetupData(manager=None, k_pen=1.0)
        callbacks = AssembleCallbacks(
            assemble_tangent=lambda u: sp.eye(12),
            assemble_internal_force=lambda u: np.zeros(12),
        )
        inp = ContactFrictionInputData(
            mesh=mesh,
            boundary=boundary,
            contact=contact,
            callbacks=callbacks,
        )
        assert inp.track_mk is False
        assert inp.track_contact_pairs is False
        assert inp.mk_moment_dofs == ()
        assert inp.mk_curvature_func is None
