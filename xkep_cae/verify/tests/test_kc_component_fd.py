"""ContactKcComponentFDDiagnosticProcess の単体テスト.

status-343: K_c を K_mat / K_geo / K_st に分解した FD 診断 Process の
Process 契約・数値整合性・コンポーネント分解の妥当性を検証する。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae.core import SolverProcess
from xkep_cae.core.testing import binds_to
from xkep_cae.verify.kc_component_fd import (
    ContactKcComponentFDDiagnosticInput,
    ContactKcComponentFDDiagnosticOutput,
    ContactKcComponentFDDiagnosticProcess,
)


def _make_linear_compute(K: sp.spmatrix):
    """f_c(u) = K @ u となる線形 compute_contact_force を作る.

    このとき ∂f_c/∂u = K で一定なので FD と解析が完全一致するはず。
    """

    def _compute(u: np.ndarray) -> np.ndarray:
        return np.asarray(K @ u).ravel()

    return _compute


def _random_spmatrix(n: int, density: float = 0.2, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((n, n)) * (rng.random((n, n)) < density)
    return sp.csr_matrix(dense)


@binds_to(ContactKcComponentFDDiagnosticProcess)
class TestContactKcComponentFDDiagnosticProcess:
    """ContactKcComponentFDDiagnosticProcess の単体テスト."""

    def test_is_solver_process(self):
        proc = ContactKcComponentFDDiagnosticProcess()
        assert isinstance(proc, SolverProcess)

    def test_meta_name_and_module(self):
        meta = ContactKcComponentFDDiagnosticProcess.meta
        assert meta.name == "ContactKcComponentFDDiagnostic"
        assert meta.module == "verify"

    def test_linear_full_agrees_with_fd(self):
        """f_c = K_c @ u の線形系で full rel_err ≈ 0."""
        n = 12  # 2 ノード × 6DOF
        K_mat = _random_spmatrix(n, seed=1)
        K_geo = _random_spmatrix(n, seed=2)
        K_st = _random_spmatrix(n, seed=3)
        K_c = K_mat - K_geo + K_st

        rng = np.random.default_rng(42)
        u = rng.standard_normal(n)
        du = rng.standard_normal(n)

        proc = ContactKcComponentFDDiagnosticProcess()
        out = proc.process(
            ContactKcComponentFDDiagnosticInput(
                u=u,
                du=du,
                compute_contact_force=_make_linear_compute(K_c),
                K_mat=K_mat,
                K_geo=K_geo,
                K_st=K_st,
                eps=1e-7,
            )
        )
        # 線形系なので full_rel_err ≈ 0（浮動小数精度の範囲）
        assert out.full_rel_err >= 0.0
        assert out.full_rel_err < 1e-5

    def test_mat_only_detects_missing_geo_st(self):
        """f_c = K_c @ u のとき mat-only では誤差が出る（st/geo 除外の影響検出）."""
        n = 12
        K_mat = _random_spmatrix(n, seed=1)
        K_geo = _random_spmatrix(n, seed=2)
        K_st = _random_spmatrix(n, seed=3)
        K_c = K_mat - K_geo + K_st

        rng = np.random.default_rng(123)
        u = rng.standard_normal(n)
        du = rng.standard_normal(n)

        proc = ContactKcComponentFDDiagnosticProcess()
        out = proc.process(
            ContactKcComponentFDDiagnosticInput(
                u=u,
                du=du,
                compute_contact_force=_make_linear_compute(K_c),
                K_mat=K_mat,
                K_geo=K_geo,
                K_st=K_st,
            )
        )
        # full は一致、mat_only / mat_geo / mat_st はいずれも非ゼロ
        assert out.full_rel_err < 1e-5
        assert out.mat_only_rel_err > 0.01
        assert out.mat_geo_rel_err > 0.01
        assert out.mat_st_rel_err > 0.01

    def test_st_primary_driver_isolation(self):
        """FD が K_mat - K_geo のみに一致する系では mat_geo が最良になる."""
        n = 12
        K_mat = _random_spmatrix(n, seed=1)
        K_geo = _random_spmatrix(n, seed=2)
        K_st_true = sp.csr_matrix((n, n))  # 真の K_st はゼロ
        K_st_wrong = _random_spmatrix(n, seed=3)  # 観測として渡す（誤差源）
        K_c_truth = K_mat - K_geo + K_st_true

        rng = np.random.default_rng(7)
        u = rng.standard_normal(n)
        du = rng.standard_normal(n)

        proc = ContactKcComponentFDDiagnosticProcess()
        out = proc.process(
            ContactKcComponentFDDiagnosticInput(
                u=u,
                du=du,
                compute_contact_force=_make_linear_compute(K_c_truth),
                K_mat=K_mat,
                K_geo=K_geo,
                K_st=K_st_wrong,
            )
        )
        # 真の K_st=0 なので K_st_wrong を加えると誤差増加 → mat_geo が最良
        assert out.mat_geo_rel_err < 1e-5
        assert out.full_rel_err > out.mat_geo_rel_err
        assert out.mat_st_rel_err > out.mat_geo_rel_err

    def test_comp_shares_are_bounded(self):
        """comp_share_* は L2 ノルム比で定義され、各成分 [0, 100] % の範囲に入る."""
        n = 12
        K_mat = _random_spmatrix(n, seed=1)
        K_geo = _random_spmatrix(n, seed=2)
        K_st = _random_spmatrix(n, seed=3)

        rng = np.random.default_rng(99)
        u = rng.standard_normal(n)
        du = rng.standard_normal(n)

        # 意図的に f_c を K_mat のみに一致させる（K_geo/K_st を誤差源に）
        proc = ContactKcComponentFDDiagnosticProcess()
        out = proc.process(
            ContactKcComponentFDDiagnosticInput(
                u=u,
                du=du,
                compute_contact_force=_make_linear_compute(K_mat),
                K_mat=K_mat,
                K_geo=K_geo,
                K_st=K_st,
            )
        )
        # 各成分シェアは [0, 100]% の範囲（L2 ノルム部分シェアなので
        # 総和は sqrt(n_comp) 倍まで膨らみ得るが各成分単独は 100% 以下）
        for c, v in out.comp_share_full.items():
            assert 0.0 <= v <= 100.0 + 1e-6, f"comp {c} out of range: {v}"
        # mat_only は一致するので各成分シェアも [0, 100]%
        for _c, v in out.comp_share_mat_only.items():
            assert 0.0 <= v <= 100.0 + 1e-6
        # rel_err は mat_only が最良
        assert out.mat_only_rel_err < 1e-5
        assert out.full_rel_err > 0.01

    def test_zero_du_returns_default_output(self):
        n = 12
        K_mat = _random_spmatrix(n, seed=1)
        K_geo = _random_spmatrix(n, seed=2)
        K_st = _random_spmatrix(n, seed=3)
        u = np.zeros(n)
        du = np.zeros(n)

        proc = ContactKcComponentFDDiagnosticProcess()
        out = proc.process(
            ContactKcComponentFDDiagnosticInput(
                u=u,
                du=du,
                compute_contact_force=_make_linear_compute(K_mat),
                K_mat=K_mat,
                K_geo=K_geo,
                K_st=K_st,
            )
        )
        # du=0 は早期 return（rel_err は -1.0 デフォルト）
        assert out.full_rel_err == -1.0
        assert "du がゼロ" in out.report

    def test_report_contains_key_sections(self):
        n = 12
        K_mat = _random_spmatrix(n, seed=1)
        K_geo = _random_spmatrix(n, seed=2)
        K_st = _random_spmatrix(n, seed=3)
        K_c = K_mat - K_geo + K_st
        rng = np.random.default_rng(5)
        u = rng.standard_normal(n)
        du = rng.standard_normal(n)

        proc = ContactKcComponentFDDiagnosticProcess()
        out = proc.process(
            ContactKcComponentFDDiagnosticInput(
                u=u,
                du=du,
                compute_contact_force=_make_linear_compute(K_c),
                K_mat=K_mat,
                K_geo=K_geo,
                K_st=K_st,
                label="trig-42",
            )
        )
        assert "K_c 成分分解 FD 診断" in out.report
        assert "trig-42" in out.report
        assert "full" in out.report
        assert "mat_only" in out.report

    def test_share_ratios_non_negative(self):
        n = 12
        K_mat = _random_spmatrix(n, seed=1)
        K_geo = _random_spmatrix(n, seed=2)
        K_st = _random_spmatrix(n, seed=3)
        K_c = K_mat - K_geo + K_st
        rng = np.random.default_rng(11)
        u = rng.standard_normal(n)
        du = rng.standard_normal(n)

        proc = ContactKcComponentFDDiagnosticProcess()
        out = proc.process(
            ContactKcComponentFDDiagnosticInput(
                u=u,
                du=du,
                compute_contact_force=_make_linear_compute(K_c),
                K_mat=K_mat,
                K_geo=K_geo,
                K_st=K_st,
            )
        )
        assert out.share_mat >= 0.0
        assert out.share_geo >= 0.0
        assert out.share_st >= 0.0

    def test_output_type(self):
        n = 12
        K_mat = _random_spmatrix(n, seed=1)
        K_geo = _random_spmatrix(n, seed=2)
        K_st = _random_spmatrix(n, seed=3)
        K_c = K_mat - K_geo + K_st
        rng = np.random.default_rng(3)
        u = rng.standard_normal(n)
        du = rng.standard_normal(n)
        proc = ContactKcComponentFDDiagnosticProcess()
        out = proc.process(
            ContactKcComponentFDDiagnosticInput(
                u=u,
                du=du,
                compute_contact_force=_make_linear_compute(K_c),
                K_mat=K_mat,
                K_geo=K_geo,
                K_st=K_st,
            )
        )
        assert isinstance(out, ContactKcComponentFDDiagnosticOutput)

    def test_share_report_preserves_small_geo_precision(self):
        """status-345: K_geo が K_mat の 0.2% 相当でも share 値が report から復元可能.

        status-344 の 19本実測で ||K_geo@du|| ≈ 1e-9, ||K_mat@du|| ≈ 1e-6
        （比 ~0.002）が観測された。`{:5.2f}` フォーマットではこれが 0.00 に
        丸められて K_geo が "消失" したように見える不具合があった。
        科学表記（`{:.3e}`）では 2.0e-03 相当の値が保全される。
        """
        n = 12
        # K_mat 約 500x K_geo（実測比 ~500:1 を模擬）
        K_mat = _random_spmatrix(n, seed=1) * 500.0
        K_geo = _random_spmatrix(n, seed=2)  # 単位スケール
        K_st = _random_spmatrix(n, seed=3) * 200.0
        K_c = K_mat - K_geo + K_st

        rng = np.random.default_rng(317)
        u = rng.standard_normal(n)
        du = rng.standard_normal(n)

        proc = ContactKcComponentFDDiagnosticProcess()
        out = proc.process(
            ContactKcComponentFDDiagnosticInput(
                u=u,
                du=du,
                compute_contact_force=_make_linear_compute(K_c),
                K_mat=K_mat,
                K_geo=K_geo,
                K_st=K_st,
            )
        )
        # share_geo は微小だが非ゼロで dataclass から取得可能
        assert out.share_geo > 0.0
        assert out.share_geo < 0.05  # K_mat 支配下での小 share
        # report 文字列に科学表記で残っている（"geo=X.XXXe-0Y" 形式）
        import re

        m = re.search(r"geo=([0-9.eE+\-]+)", out.report)
        assert m is not None, f"geo share missing in report: {out.report}"
        parsed_geo = float(m.group(1))
        assert abs(parsed_geo - out.share_geo) / max(out.share_geo, 1e-30) < 1e-2
        # 絶対 L2 ノルムも dataclass に公開
        assert out.geo_du_norm > 0.0
        assert out.mat_du_norm > out.geo_du_norm

    def test_input_frozen(self):
        n = 12
        K_mat = _random_spmatrix(n, seed=1)
        K_geo = _random_spmatrix(n, seed=2)
        K_st = _random_spmatrix(n, seed=3)
        u = np.zeros(n)
        du = np.ones(n)
        inp = ContactKcComponentFDDiagnosticInput(
            u=u,
            du=du,
            compute_contact_force=_make_linear_compute(K_mat),
            K_mat=K_mat,
            K_geo=K_geo,
            K_st=K_st,
        )
        try:
            inp.eps = 1e-5  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow mutation")
        except AttributeError:
            pass
