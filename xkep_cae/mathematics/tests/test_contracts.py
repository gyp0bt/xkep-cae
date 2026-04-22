"""MathematicalContract 5 種の単体テスト.

status-346（Phase A-1）で新設。型レベルの検査のみを行い、Process への
契約宣言（Phase A-2）や実数値検証（Phase D）は対象外。
"""

from __future__ import annotations

import dataclasses

import pytest

from xkep_cae.mathematics import (
    FDConsistencyContract,
    IdentityContract,
    InequalityContract,
    MathematicalContract,
    SymmetryContract,
    TermExpansionContract,
)

# ======================================================================
# 共通: frozen 性と基本バリデーション
# ======================================================================


class TestMathematicalContractBase:
    """抽象基底の共通検査.

    MathematicalContract は dataclass の `__post_init__` で name/equation_ref/
    severity の必須性・有効性を担保する。
    """

    def test_identity_contract_is_frozen(self) -> None:
        """IdentityContract は frozen dataclass（書き換え不可）."""
        c = IdentityContract(
            name="test",
            equation_ref="math.md#eq",
            lhs="K",
            rhs="\\partial f / \\partial u",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.name = "changed"  # type: ignore[misc]

    def test_symmetry_contract_is_frozen(self) -> None:
        c = SymmetryContract(
            name="sym",
            equation_ref="math.md#eq",
            matrix_name="K_mat",
            kind="symmetric",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.tol = 1e-5  # type: ignore[misc]

    def test_empty_name_rejected(self) -> None:
        """name 空文字列は ValueError."""
        with pytest.raises(ValueError, match="name は必須"):
            IdentityContract(name="", equation_ref="math.md#eq", lhs="A", rhs="B")

    def test_empty_equation_ref_rejected(self) -> None:
        """equation_ref 空文字列は ValueError."""
        with pytest.raises(ValueError, match="equation_ref は必須"):
            IdentityContract(name="t", equation_ref="", lhs="A", rhs="B")

    def test_invalid_severity_rejected(self) -> None:
        """severity は hard/soft/nightly のみ."""
        with pytest.raises(ValueError, match="severity"):
            IdentityContract(
                name="t",
                equation_ref="math.md#eq",
                lhs="A",
                rhs="B",
                severity="critical",  # type: ignore[arg-type]
            )

    def test_all_contracts_are_mathematical_contract(self) -> None:
        """全 5 種が基底 MathematicalContract のサブクラス."""
        c1 = IdentityContract(name="t", equation_ref="m#e", lhs="A", rhs="B")
        c2 = InequalityContract(name="t", equation_ref="m#e", expr="x", kind="geq", bound="0")
        c3 = FDConsistencyContract(name="t", equation_ref="m#e", vector_name="f", jacobian_name="K")
        c4 = SymmetryContract(name="t", equation_ref="m#e", matrix_name="K")
        c5 = TermExpansionContract(
            name="t",
            equation_ref="m#e",
            total_name="K",
            term_names=("K1",),
            providers=("P1",),
        )
        for c in (c1, c2, c3, c4, c5):
            assert isinstance(c, MathematicalContract)


# ======================================================================
# IdentityContract
# ======================================================================


class TestIdentityContract:
    def test_valid(self) -> None:
        c = IdentityContract(
            name="K_c_is_jacobian",
            equation_ref="03_huber.md#eq-kc",
            lhs="K_c",
            rhs="\\partial f_c / \\partial u",
            severity="soft",
        )
        assert c.lhs == "K_c"
        assert c.rhs == "\\partial f_c / \\partial u"
        assert c.severity == "soft"

    def test_empty_lhs_rejected(self) -> None:
        with pytest.raises(ValueError, match="lhs/rhs 両方必須"):
            IdentityContract(name="t", equation_ref="m#e", lhs="", rhs="B")

    def test_empty_rhs_rejected(self) -> None:
        with pytest.raises(ValueError, match="lhs/rhs 両方必須"):
            IdentityContract(name="t", equation_ref="m#e", lhs="A", rhs="")


# ======================================================================
# InequalityContract
# ======================================================================


class TestInequalityContract:
    def test_valid(self) -> None:
        c = InequalityContract(
            name="p_n_nonneg",
            equation_ref="03_huber.md#eq-pn",
            expr="p_n",
            kind="geq",
            bound="0",
        )
        assert c.expr == "p_n"
        assert c.kind == "geq"
        assert c.bound == "0"

    def test_empty_expr_rejected(self) -> None:
        with pytest.raises(ValueError, match="expr 必須"):
            InequalityContract(name="t", equation_ref="m#e", expr="", kind="geq", bound="0")

    def test_invalid_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            InequalityContract(
                name="t",
                equation_ref="m#e",
                expr="x",
                kind="eq",  # type: ignore[arg-type]
                bound="0",
            )

    def test_all_kinds_accepted(self) -> None:
        for k in ("geq", "leq", "gt", "lt"):
            c = InequalityContract(
                name=f"t_{k}",
                equation_ref="m#e",
                expr="x",
                kind=k,  # type: ignore[arg-type]
                bound="0",
            )
            assert c.kind == k


# ======================================================================
# FDConsistencyContract
# ======================================================================


class TestFDConsistencyContract:
    def test_valid(self) -> None:
        c = FDConsistencyContract(
            name="K_c_fd",
            equation_ref="03_huber.md#eq-kc-fd",
            vector_name="f_c",
            jacobian_name="K_c",
            tol_rel=5e-3,
            perturbation=1e-7,
            scope=("x", "z"),
            severity="nightly",
        )
        assert c.vector_name == "f_c"
        assert c.jacobian_name == "K_c"
        assert c.tol_rel == pytest.approx(5e-3)
        assert c.scope == ("x", "z")

    def test_default_scope_is_empty(self) -> None:
        c = FDConsistencyContract(name="t", equation_ref="m#e", vector_name="f", jacobian_name="K")
        assert c.scope == ()

    def test_empty_vector_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="vector_name/jacobian_name 両方必須"):
            FDConsistencyContract(name="t", equation_ref="m#e", vector_name="", jacobian_name="K")

    def test_empty_jacobian_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="vector_name/jacobian_name 両方必須"):
            FDConsistencyContract(name="t", equation_ref="m#e", vector_name="f", jacobian_name="")

    def test_nonpositive_tol_rejected(self) -> None:
        with pytest.raises(ValueError, match="tol_rel は正の値必須"):
            FDConsistencyContract(
                name="t",
                equation_ref="m#e",
                vector_name="f",
                jacobian_name="K",
                tol_rel=0.0,
            )

    def test_nonpositive_perturbation_rejected(self) -> None:
        with pytest.raises(ValueError, match="perturbation は正の値必須"):
            FDConsistencyContract(
                name="t",
                equation_ref="m#e",
                vector_name="f",
                jacobian_name="K",
                perturbation=-1e-7,
            )


# ======================================================================
# SymmetryContract
# ======================================================================


class TestSymmetryContract:
    def test_valid_symmetric(self) -> None:
        c = SymmetryContract(
            name="K_mat_sym",
            equation_ref="03_huber.md#eq-kmat",
            matrix_name="K_mat",
            kind="symmetric",
            tol=1e-10,
        )
        assert c.kind == "symmetric"
        assert c.tol == pytest.approx(1e-10)

    def test_valid_skew(self) -> None:
        c = SymmetryContract(
            name="Omega_skew",
            equation_ref="06_cr.md#eq-omega",
            matrix_name="Omega",
            kind="skew",
        )
        assert c.kind == "skew"

    def test_empty_matrix_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="matrix_name 必須"):
            SymmetryContract(name="t", equation_ref="m#e", matrix_name="", kind="symmetric")

    def test_invalid_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            SymmetryContract(
                name="t",
                equation_ref="m#e",
                matrix_name="K",
                kind="anti",  # type: ignore[arg-type]
            )

    def test_negative_tol_rejected(self) -> None:
        with pytest.raises(ValueError, match="tol は非負値必須"):
            SymmetryContract(
                name="t",
                equation_ref="m#e",
                matrix_name="K",
                kind="symmetric",
                tol=-1e-10,
            )


# ======================================================================
# TermExpansionContract ★MCDD の核
# ======================================================================


class TestTermExpansionContract:
    def test_valid_full(self) -> None:
        """status-352 で宣言予定の完全形を模擬.

        K_c = K_mat_nn - K_geo + K_st + K_mat_ndir + K_closest + K_hermite_adj
        の 6 項 add_sub パターン。
        """
        c = TermExpansionContract(
            name="K_c_term_expansion",
            equation_ref="03_huber.md#eq-kc-full-decomposition",
            total_name="K_c",
            term_names=(
                "K_mat_nn",
                "K_mat_ndir",
                "K_closest",
                "K_hermite_adj",
                "K_geo",
                "K_st",
            ),
            providers=(
                "KcNormalStiffnessProcess",
                "KcNormalDirectionStiffnessProcess",
                "KcClosestPointStiffnessProcess",
                "KcHermiteNonlocalStiffnessProcess",
                "KcGeoStiffnessProcess",
                "KcStStiffnessProcess",
            ),
            combinator="add_sub",
            tol_rel=5e-3,
            severity="nightly",
        )
        assert c.total_name == "K_c"
        assert len(c.term_names) == 6
        assert len(c.providers) == 6
        assert c.combinator == "add_sub"

    def test_empty_total_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_name 必須"):
            TermExpansionContract(
                name="t",
                equation_ref="m#e",
                total_name="",
                term_names=("K1",),
                providers=("P1",),
            )

    def test_empty_term_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="term_names 必須"):
            TermExpansionContract(
                name="t",
                equation_ref="m#e",
                total_name="K",
                term_names=(),
                providers=(),
            )

    def test_mismatched_term_provider_length_rejected(self) -> None:
        with pytest.raises(ValueError, match="長さが一致"):
            TermExpansionContract(
                name="t",
                equation_ref="m#e",
                total_name="K",
                term_names=("K1", "K2"),
                providers=("P1",),
            )

    def test_invalid_combinator_rejected(self) -> None:
        with pytest.raises(ValueError, match="combinator"):
            TermExpansionContract(
                name="t",
                equation_ref="m#e",
                total_name="K",
                term_names=("K1",),
                providers=("P1",),
                combinator="multiply",  # type: ignore[arg-type]
            )

    def test_nonpositive_tol_rejected(self) -> None:
        with pytest.raises(ValueError, match="tol_rel は正の値必須"):
            TermExpansionContract(
                name="t",
                equation_ref="m#e",
                total_name="K",
                term_names=("K1",),
                providers=("P1",),
                tol_rel=0.0,
            )

    def test_duplicate_providers_rejected(self) -> None:
        """providers 重複は ValueError.

        脱法実装防止: 同じ Process を複数 term に重複登録して網羅性を水増し
        するのを防ぐ（プラン § 脱法実装防止 2 項の構造的ガード）。
        """
        with pytest.raises(ValueError, match="providers に重複"):
            TermExpansionContract(
                name="t",
                equation_ref="m#e",
                total_name="K",
                term_names=("K1", "K2"),
                providers=("P1", "P1"),
            )

    def test_duplicate_term_names_rejected(self) -> None:
        """term_names 重複は ValueError（status-360 C21）.

        項名の重複は合計検証 Σ K_k の参照同一性を崩すため構造的に排除する。
        """
        with pytest.raises(ValueError, match="term_names に重複"):
            TermExpansionContract(
                name="t",
                equation_ref="m#e",
                total_name="K",
                term_names=("K1", "K1"),
                providers=("P1", "P2"),
            )

    def test_sum_combinator(self) -> None:
        """sum パターン（全項加算）も受理."""
        c = TermExpansionContract(
            name="t",
            equation_ref="m#e",
            total_name="K",
            term_names=("K1", "K2"),
            providers=("P1", "P2"),
            combinator="sum",
        )
        assert c.combinator == "sum"


# ======================================================================
# 統合: パッケージ export 確認
# ======================================================================


class TestPackageExports:
    """xkep_cae.mathematics からの公開 API が期待通りであること."""

    def test_all_contracts_exported(self) -> None:
        import xkep_cae.mathematics as m

        expected = {
            "MathematicalContract",
            "IdentityContract",
            "InequalityContract",
            "FDConsistencyContract",
            "SymmetryContract",
            "TermExpansionContract",
        }
        assert expected.issubset(set(m.__all__))
        for name in expected:
            assert hasattr(m, name)
