"""ContactPairAnalysisProcess の単体テスト.

status-337: 接触ペア履歴後処理 Process の検証。
- 合成履歴（固定 ContactPairSnapshotEntry 列）での純粋関数的挙動
- StrandBendingOscillationProcess との統合（2本撚線で実履歴を取得）

[← README](../../README.md)
"""

from __future__ import annotations

import pytest

from xkep_cae.core import binds_to
from xkep_cae.core.data import ContactPairSnapshotEntry
from xkep_cae.numerical_tests.contact_pair_analysis import (
    ContactPairAnalysisInput,
    ContactPairAnalysisProcess,
)
from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)

# ====================================================================
# ヘルパー: 合成スナップショット生成
# ====================================================================


def _entry(
    elem_a: int,
    elem_b: int,
    p_n: float,
    *,
    slip_mag: float = 0.0,
    stick: bool = True,
    dissipation: float = 0.0,
) -> ContactPairSnapshotEntry:
    """テスト用スナップショットエントリを生成."""
    # slip_mag を s 成分に全振り（hypot(s, 0) = |s|）
    return ContactPairSnapshotEntry(
        elem_a=elem_a,
        elem_b=elem_b,
        p_n=p_n,
        gap=-0.01,
        slip_s=slip_mag,
        slip_t=0.0,
        stick=stick,
        dissipation=dissipation,
    )


# ====================================================================
# 純粋関数的テスト（合成履歴）
# ====================================================================


@binds_to(ContactPairAnalysisProcess)
class TestContactPairAnalysisAPI:
    """合成履歴による純粋関数的検証."""

    def test_empty_history(self) -> None:
        """空履歴は全ゼロ結果を返す."""
        result = ContactPairAnalysisProcess().process(
            ContactPairAnalysisInput(contact_pair_history=())
        )
        assert result.n_steps == 0
        assert result.n_unique_pairs == 0
        assert result.n_active_per_step == ()
        assert result.load_frac_per_step == ()
        assert result.kappa_cr_per_pair == {}
        assert result.per_pair_dissipation == {}
        assert result.total_dissipation == 0.0
        assert result.n_slipped_pairs == 0

    def test_single_step_stick_only(self) -> None:
        """全ペア stick のみ → κ_cr は空."""
        hist = (
            (
                0.5,
                (
                    _entry(0, 1, p_n=1.0, stick=True, slip_mag=0.0),
                    _entry(0, 2, p_n=0.5, stick=True, slip_mag=0.0),
                ),
            ),
        )
        result = ContactPairAnalysisProcess().process(
            ContactPairAnalysisInput(contact_pair_history=hist)
        )
        assert result.n_steps == 1
        assert result.n_unique_pairs == 2
        assert result.n_active_per_step == (2,)
        assert result.kappa_cr_per_pair == {}
        assert result.n_slipped_pairs == 0

    def test_kappa_cr_records_first_slip(self) -> None:
        """κ_cr は初回スリップ時の κ（ここでは load_frac）."""
        hist = (
            # step 0: stick only
            (0.1, (_entry(0, 1, p_n=1.0, stick=True, slip_mag=0.0),)),
            # step 1: slip detected（初回）
            (0.3, (_entry(0, 1, p_n=1.0, stick=False, slip_mag=1e-3),)),
            # step 2: slip 継続（κ_cr は上書きされない）
            (0.5, (_entry(0, 1, p_n=1.0, stick=False, slip_mag=2e-3),)),
        )
        result = ContactPairAnalysisProcess().process(
            ContactPairAnalysisInput(contact_pair_history=hist)
        )
        assert result.n_slipped_pairs == 1
        assert result.kappa_cr_per_pair == {(0, 1): pytest.approx(0.3)}

    def test_slip_threshold_filters_noise(self) -> None:
        """|slip| <= slip_threshold は数値雑音とみなしスリップ判定しない."""
        hist = (
            # stick=False だが |slip| = 1e-9 < threshold
            (0.2, (_entry(0, 1, p_n=1.0, stick=False, slip_mag=1e-9),)),
            # |slip| = 1e-5 > threshold（デフォルト 1e-6）
            (0.4, (_entry(0, 1, p_n=1.0, stick=False, slip_mag=1e-5),)),
        )
        result = ContactPairAnalysisProcess().process(
            ContactPairAnalysisInput(contact_pair_history=hist)
        )
        assert result.kappa_cr_per_pair == {(0, 1): pytest.approx(0.4)}

    def test_dissipation_tracks_last_value(self) -> None:
        """per_pair_dissipation は履歴最終値."""
        hist = (
            (0.1, (_entry(0, 1, p_n=1.0, dissipation=0.001),)),
            (0.2, (_entry(0, 1, p_n=1.0, dissipation=0.005),)),
            (0.3, (_entry(0, 1, p_n=1.0, dissipation=0.012),)),
        )
        result = ContactPairAnalysisProcess().process(
            ContactPairAnalysisInput(contact_pair_history=hist)
        )
        assert result.per_pair_dissipation == {(0, 1): pytest.approx(0.012)}
        assert result.total_dissipation == pytest.approx(0.012)

    def test_uses_moment_curvature_history_for_kappa(self) -> None:
        """moment_curvature_history 指定時は load_frac ではなく κ を記録."""
        hist = (
            (0.1, (_entry(0, 1, p_n=1.0, stick=True, slip_mag=0.0),)),
            (0.3, (_entry(0, 1, p_n=1.0, stick=False, slip_mag=1e-3),)),
        )
        mk = (
            (1.0e-4, 0.5),  # step 0 の κ
            (3.0e-4, 1.2),  # step 1 の κ
        )
        result = ContactPairAnalysisProcess().process(
            ContactPairAnalysisInput(
                contact_pair_history=hist,
                moment_curvature_history=mk,
            )
        )
        assert result.kappa_cr_per_pair == {(0, 1): pytest.approx(3.0e-4)}

    def test_kappa_cr_distribution_stats(self) -> None:
        """κ_cr 分布統計（mean/std/min/max/n）."""
        hist = (
            (
                0.2,
                (
                    _entry(0, 1, p_n=1.0, stick=False, slip_mag=1e-3),
                    _entry(0, 2, p_n=1.0, stick=True, slip_mag=0.0),
                ),
            ),
            (
                0.4,
                (
                    _entry(0, 1, p_n=1.0, stick=False, slip_mag=2e-3),
                    _entry(0, 2, p_n=1.0, stick=False, slip_mag=1e-3),
                ),
            ),
            (
                0.6,
                (_entry(0, 3, p_n=1.0, stick=False, slip_mag=1e-3),),
            ),
        )
        result = ContactPairAnalysisProcess().process(
            ContactPairAnalysisInput(contact_pair_history=hist)
        )
        # κ_cr: (0,1)→0.2, (0,2)→0.4, (0,3)→0.6
        assert result.n_slipped_pairs == 3
        assert result.kappa_cr_mean == pytest.approx(0.4)
        assert result.kappa_cr_min == pytest.approx(0.2)
        assert result.kappa_cr_max == pytest.approx(0.6)
        # std: sqrt(((0.2-0.4)^2 + 0 + (0.6-0.4)^2)/3) = sqrt(0.08/3)
        import math

        expected_std = math.sqrt(0.08 / 3.0)
        assert result.kappa_cr_std == pytest.approx(expected_std)

    def test_n_active_per_step_matches_entries(self) -> None:
        """n_active_per_step は各インクリメントのエントリ数と一致."""
        hist = (
            (0.1, (_entry(0, 1, p_n=1.0), _entry(0, 2, p_n=1.0))),
            (0.2, ()),
            (0.3, (_entry(0, 1, p_n=1.0),)),
        )
        result = ContactPairAnalysisProcess().process(
            ContactPairAnalysisInput(contact_pair_history=hist)
        )
        assert result.n_active_per_step == (2, 0, 1)
        assert result.load_frac_per_step == (
            pytest.approx(0.1),
            pytest.approx(0.2),
            pytest.approx(0.3),
        )


# ====================================================================
# 統合テスト（StrandBendingOscillation 実履歴）
# ====================================================================


@pytest.mark.slow
class TestContactPairAnalysisConvergence:
    """2本撚線で実履歴を取得して後処理が機能することを確認."""

    def test_end_to_end_2strand(self) -> None:
        """2本撚線曲げで contact_pair_history を取得し解析する."""
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
        result = StrandBendingOscillationProcess().process(cfg)
        sr = result.solver_result

        analysis = ContactPairAnalysisProcess().process(
            ContactPairAnalysisInput(
                contact_pair_history=sr.contact_pair_history,
                moment_curvature_history=sr.moment_curvature_history,
            )
        )

        print("\n=== 接触ペア解析（2本撚線 end-to-end）===")
        print(f"  n_steps={analysis.n_steps}")
        print(f"  n_unique_pairs={analysis.n_unique_pairs}")
        print(f"  n_slipped_pairs={analysis.n_slipped_pairs}")
        print(f"  total_dissipation={analysis.total_dissipation:.4e}")
        if analysis.n_slipped_pairs > 0:
            print(
                f"  κ_cr = {analysis.kappa_cr_mean:.3e} "
                f"± {analysis.kappa_cr_std:.3e} "
                f"(min={analysis.kappa_cr_min:.3e}, max={analysis.kappa_cr_max:.3e})"
            )

        # 履歴は記録されている
        assert analysis.n_steps > 0, "contact_pair_history が空"
        # n_active_per_step は n_steps と同じ長さ
        assert len(analysis.n_active_per_step) == analysis.n_steps
        assert len(analysis.load_frac_per_step) == analysis.n_steps
        # load_frac は非減少（曲げフェーズでは単調増加）
        fracs = analysis.load_frac_per_step
        for i in range(1, len(fracs)):
            assert fracs[i] >= fracs[i - 1] - 1e-9, "load_frac 非単調"
        # 構造: 活性ペアが存在する場合、κ_cr/dissipation の整合性
        # 2本撚線+小曲率では接触未発生（n_active=0）が通常のため ここでは
        # パイプラインが例外なく走ることだけ確認（物理的検証は 7本撚線後続）。
        if analysis.n_unique_pairs > 0:
            # n_slipped_pairs は n_unique_pairs の部分集合
            assert analysis.n_slipped_pairs <= analysis.n_unique_pairs
            # per_pair_dissipation は n_unique_pairs と同数
            assert len(analysis.per_pair_dissipation) == analysis.n_unique_pairs
