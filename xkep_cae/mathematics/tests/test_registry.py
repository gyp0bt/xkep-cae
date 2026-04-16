"""`ProcessContractRegistry` / `@verified_by` の単体テスト.

status-347（MCDD Phase A-2）で新設。レジストリ API とデコレータの
紐付け検証、および dummy VerifyProcess 拒否の AST 検査を対象とする。

Phase A-1 の `test_contracts.py` と併せて MCDD 基盤の全型・全配線を覆う。
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from xkep_cae.core.base import AbstractProcess, ProcessMeta
from xkep_cae.core.categories import SolverProcess, VerifyProcess
from xkep_cae.mathematics import (
    DummyVerifyProcessError,
    FDConsistencyContract,
    IdentityContract,
    ProcessContractRegistry,
    SymmetryContract,
    verified_by,
)

# ======================================================================
# テスト用ダミー Process の最小フィクスチャ
# ======================================================================
#
# `_skip_registry = True` により `ProcessRegistry`（本体レジストリ）への
# 登録はスキップされるが、`ProcessContractRegistry` への自動登録は
# `AbstractProcess.__init_subclass__` 内で別ロジックが走る。
# したがってテストごとに `ProcessContractRegistry` を差し替えて
# 汚染を避ける（`setup_method` / `teardown_method`）。


_DOC_REL = "../docs/mathematics.md"  # 既存のドキュメントを流用（C15 を満たす）


class _TargetProcess(SolverProcess[object, object]):
    """契約を持たない最小 Process（登録側のホスト）."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="_TargetProcess",
        module="solve",
        version="0.1.0",
        document_path=_DOC_REL,
        stability="experimental",
        support_tier="dev-only",
    )
    _skip_registry = True

    def process(self, input_data: object) -> object:  # pragma: no cover - テスト用
        return input_data


class _ProperVerifyProcess(VerifyProcess[object, bool]):
    """実装の中身がある VerifyProcess（non-dummy）."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="_ProperVerifyProcess",
        module="verify",
        version="0.1.0",
        document_path=_DOC_REL,
        stability="experimental",
        support_tier="dev-only",
    )
    _skip_registry = True

    def process(self, input_data: object) -> bool:
        # 意味のある処理: 数値比較を模擬
        threshold = 1e-6
        measured = abs(float(len(repr(input_data)))) * 1e-8
        return measured < threshold


class _DummyPassVerifyProcess(VerifyProcess[object, bool]):
    """`pass` のみの dummy VerifyProcess（拒否対象）."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="_DummyPassVerifyProcess",
        module="verify",
        version="0.1.0",
        document_path=_DOC_REL,
        stability="experimental",
        support_tier="dev-only",
    )
    _skip_registry = True

    def process(self, input_data: object) -> bool:
        pass  # type: ignore[return-value]


class _DummyEllipsisVerifyProcess(VerifyProcess[object, bool]):
    """`...` のみの dummy VerifyProcess（拒否対象）."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="_DummyEllipsisVerifyProcess",
        module="verify",
        version="0.1.0",
        document_path=_DOC_REL,
        stability="experimental",
        support_tier="dev-only",
    )
    _skip_registry = True

    def process(self, input_data: object) -> bool: ...


class _DummyDocstringOnlyVerifyProcess(VerifyProcess[object, bool]):
    """docstring のみで実装のない dummy VerifyProcess（拒否対象）."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="_DummyDocstringOnlyVerifyProcess",
        module="verify",
        version="0.1.0",
        document_path=_DOC_REL,
        stability="experimental",
        support_tier="dev-only",
    )
    _skip_registry = True

    def process(self, input_data: object) -> bool:
        """This docstring pretends to do something but actually returns nothing."""
        return None  # type: ignore[return-value]


class _DummyNotImplementedVerifyProcess(VerifyProcess[object, bool]):
    """`raise NotImplementedError` のみの dummy VerifyProcess（拒否対象）."""

    meta: ClassVar[ProcessMeta] = ProcessMeta(
        name="_DummyNotImplementedVerifyProcess",
        module="verify",
        version="0.1.0",
        document_path=_DOC_REL,
        stability="experimental",
        support_tier="dev-only",
    )
    _skip_registry = True

    def process(self, input_data: object) -> bool:
        raise NotImplementedError("未実装")


# ======================================================================
# テスト
# ======================================================================


class TestProcessContractRegistryBasics:
    """レジストリの singleton / 基本操作."""

    def test_default_is_singleton(self) -> None:
        a = ProcessContractRegistry.default()
        b = ProcessContractRegistry.default()
        assert a is b

    def test_set_default_allows_override(self) -> None:
        original = ProcessContractRegistry.default()
        try:
            new = ProcessContractRegistry()
            ProcessContractRegistry._set_default(new)
            assert ProcessContractRegistry.default() is new
        finally:
            ProcessContractRegistry._set_default(original)

    def test_clear_resets_state(self) -> None:
        r = ProcessContractRegistry()
        r.register_contracts(
            _TargetProcess,
            (IdentityContract(name="t", equation_ref="m#e", lhs="A", rhs="B"),),
        )
        assert r.contracts_of(_TargetProcess) != ()
        r.clear()
        assert r.contracts_of(_TargetProcess) == ()
        assert r.all_contracts() == {}
        assert r.all_bindings() == {}

    def test_repr_contains_counts(self) -> None:
        r = ProcessContractRegistry()
        r.register_contracts(
            _TargetProcess,
            (IdentityContract(name="t", equation_ref="m#e", lhs="A", rhs="B"),),
        )
        text = repr(r)
        assert "1 processes" in text
        assert "0 bindings" in text


class TestContractRegistration:
    """`register_contracts` の正常系・異常系."""

    def setup_method(self) -> None:
        self.registry = ProcessContractRegistry()

    def test_register_and_query(self) -> None:
        c = IdentityContract(name="K_c_is_jacobian", equation_ref="m#e", lhs="K_c", rhs="df/du")
        self.registry.register_contracts(_TargetProcess, (c,))
        assert self.registry.contracts_of(_TargetProcess) == (c,)
        # 名前文字列でも取得できる
        assert self.registry.contracts_of("_TargetProcess") == (c,)

    def test_contract_by_name(self) -> None:
        c1 = IdentityContract(name="c1", equation_ref="m#e", lhs="A", rhs="B")
        c2 = SymmetryContract(name="c2", equation_ref="m#e", matrix_name="K")
        self.registry.register_contracts(_TargetProcess, (c1, c2))
        assert self.registry.contract_by_name(_TargetProcess, "c2") is c2
        assert self.registry.contract_by_name(_TargetProcess, "missing") is None

    def test_non_tuple_rejected(self) -> None:
        c = IdentityContract(name="c1", equation_ref="m#e", lhs="A", rhs="B")
        with pytest.raises(TypeError, match="tuple 必須"):
            self.registry.register_contracts(_TargetProcess, [c])  # type: ignore[arg-type]

    def test_non_contract_element_rejected(self) -> None:
        with pytest.raises(TypeError, match="MathematicalContract のサブクラスではありません"):
            self.registry.register_contracts(
                _TargetProcess,
                ("not_a_contract",),  # type: ignore[arg-type]
            )

    def test_duplicate_name_within_batch_rejected(self) -> None:
        c1 = IdentityContract(name="dup", equation_ref="m#e", lhs="A", rhs="B")
        c2 = SymmetryContract(name="dup", equation_ref="m#e", matrix_name="K")
        with pytest.raises(ValueError, match="重複"):
            self.registry.register_contracts(_TargetProcess, (c1, c2))

    def test_duplicate_name_across_calls_rejected(self) -> None:
        """追加登録で既存名と重複したら拒否."""
        c1 = IdentityContract(name="dup", equation_ref="m#e", lhs="A", rhs="B")
        c2 = SymmetryContract(name="dup", equation_ref="m#e", matrix_name="K")
        self.registry.register_contracts(_TargetProcess, (c1,))
        with pytest.raises(ValueError, match="重複"):
            self.registry.register_contracts(_TargetProcess, (c2,))

    def test_empty_tuple_is_noop(self) -> None:
        self.registry.register_contracts(_TargetProcess, ())
        # 登録なしの扱い（空 tuple が入っていても空 tuple を返す）
        assert self.registry.contracts_of(_TargetProcess) == ()

    def test_unregistered_process_returns_empty(self) -> None:
        assert self.registry.contracts_of("NeverRegistered") == ()
        assert self.registry.contract_by_name("NeverRegistered", "x") is None


class TestBindVerifier:
    """`bind_verifier` の正常系・異常系（dummy 検出を含む）."""

    def setup_method(self) -> None:
        self.registry = ProcessContractRegistry()
        self.contract = FDConsistencyContract(
            name="K_c_fd_consistency",
            equation_ref="m#e",
            vector_name="f_c",
            jacobian_name="K_c",
            tol_rel=5e-3,
        )
        self.registry.register_contracts(_TargetProcess, (self.contract,))

    def test_bind_success(self) -> None:
        self.registry.bind_verifier(_TargetProcess, "K_c_fd_consistency", _ProperVerifyProcess)
        assert (
            self.registry.verifier_of(_TargetProcess, "K_c_fd_consistency") is _ProperVerifyProcess
        )

    def test_bind_idempotent_same_class(self) -> None:
        """同じ verify_cls を二回紐付けても idempotent（エラーにしない）."""
        self.registry.bind_verifier(_TargetProcess, "K_c_fd_consistency", _ProperVerifyProcess)
        self.registry.bind_verifier(
            _TargetProcess, "K_c_fd_consistency", _ProperVerifyProcess
        )  # 例外なし

    def test_bind_conflict_different_class_rejected(self) -> None:
        class _AnotherVerify(VerifyProcess[object, bool]):
            meta: ClassVar[ProcessMeta] = ProcessMeta(
                name="_AnotherVerify",
                module="verify",
                version="0.1.0",
                document_path=_DOC_REL,
                stability="experimental",
                support_tier="dev-only",
            )
            _skip_registry = True

            def process(self, input_data: object) -> bool:
                flag = bool(input_data)
                return flag

        self.registry.bind_verifier(_TargetProcess, "K_c_fd_consistency", _ProperVerifyProcess)
        with pytest.raises(ValueError, match="既に .+ に紐付いています"):
            self.registry.bind_verifier(_TargetProcess, "K_c_fd_consistency", _AnotherVerify)

    def test_bind_non_process_rejected(self) -> None:
        class _NotAProcess:
            pass

        with pytest.raises(TypeError, match="AbstractProcess のサブクラスではありません"):
            self.registry.bind_verifier(
                _TargetProcess,
                "K_c_fd_consistency",
                _NotAProcess,  # type: ignore[arg-type]
            )

    def test_bind_abstract_process_rejected(self) -> None:
        # VerifyProcess 自体は abstract（process() 未実装）
        with pytest.raises(TypeError, match="抽象クラスです"):
            self.registry.bind_verifier(
                _TargetProcess,
                "K_c_fd_consistency",
                VerifyProcess,  # type: ignore[arg-type]
            )

    def test_bind_unknown_contract_rejected(self) -> None:
        with pytest.raises(ValueError, match="契約 'nope' が宣言されていません"):
            self.registry.bind_verifier(_TargetProcess, "nope", _ProperVerifyProcess)

    @pytest.mark.parametrize(
        "dummy_cls",
        [
            _DummyPassVerifyProcess,
            _DummyEllipsisVerifyProcess,
            _DummyDocstringOnlyVerifyProcess,
            _DummyNotImplementedVerifyProcess,
        ],
    )
    def test_bind_dummy_rejected(self, dummy_cls: type[AbstractProcess]) -> None:
        """dummy（pass/.../None 裸 return/NotImplementedError 単体）は AST 検査で拒否."""
        with pytest.raises(DummyVerifyProcessError, match="dummy"):
            self.registry.bind_verifier(_TargetProcess, "K_c_fd_consistency", dummy_cls)

    def test_unverified_contracts(self) -> None:
        """紐付け前は契約が `unverified`、紐付け後は空."""
        assert self.registry.unverified_contracts(_TargetProcess) == (self.contract,)
        self.registry.bind_verifier(_TargetProcess, "K_c_fd_consistency", _ProperVerifyProcess)
        assert self.registry.unverified_contracts(_TargetProcess) == ()

    def test_all_bindings_snapshot(self) -> None:
        self.registry.bind_verifier(_TargetProcess, "K_c_fd_consistency", _ProperVerifyProcess)
        bindings = self.registry.all_bindings()
        assert bindings == {("_TargetProcess", "K_c_fd_consistency"): _ProperVerifyProcess}


class TestVerifiedByDecorator:
    """`@verified_by` デコレータの動作（default registry を通る経路）."""

    def setup_method(self) -> None:
        self._saved = ProcessContractRegistry.default()
        self._fresh = ProcessContractRegistry()
        ProcessContractRegistry._set_default(self._fresh)

    def teardown_method(self) -> None:
        ProcessContractRegistry._set_default(self._saved)

    def test_decorator_binds_via_default_registry(self) -> None:
        contract = FDConsistencyContract(
            name="K_c_fd_consistency",
            equation_ref="m#e",
            vector_name="f_c",
            jacobian_name="K_c",
        )
        self._fresh.register_contracts(_TargetProcess, (contract,))

        # デコレータ適用
        returned = verified_by("K_c_fd_consistency", _ProperVerifyProcess)(_TargetProcess)
        # 元のクラスを返す
        assert returned is _TargetProcess
        assert self._fresh.verifier_of(_TargetProcess, "K_c_fd_consistency") is _ProperVerifyProcess

    def test_decorator_rejects_dummy_verify(self) -> None:
        contract = FDConsistencyContract(
            name="K_c_fd_consistency",
            equation_ref="m#e",
            vector_name="f_c",
            jacobian_name="K_c",
        )
        self._fresh.register_contracts(_TargetProcess, (contract,))

        with pytest.raises(DummyVerifyProcessError):
            verified_by("K_c_fd_consistency", _DummyPassVerifyProcess)(_TargetProcess)

    def test_decorator_rejects_unknown_contract(self) -> None:
        # _TargetProcess にはまだ契約が登録されていない
        with pytest.raises(ValueError, match="契約"):
            verified_by("no_such_contract", _ProperVerifyProcess)(_TargetProcess)


class TestAbstractProcessAutoRegistration:
    """`AbstractProcess.__init_subclass__` が契約を自動登録することを確認.

    class-level ``contracts`` ClassVar / ``meta.math_contracts`` のどちらか、
    または両方に宣言された契約が `ProcessContractRegistry.default()` に
    登録されることを検証する。
    """

    def setup_method(self) -> None:
        self._saved = ProcessContractRegistry.default()
        self._fresh = ProcessContractRegistry()
        ProcessContractRegistry._set_default(self._fresh)

    def teardown_method(self) -> None:
        ProcessContractRegistry._set_default(self._saved)

    def test_default_contracts_is_empty(self) -> None:
        """契約を宣言していないクラスは未登録（デフォルト空）."""
        # `_TargetProcess` はクラス定義時点で保存済みデフォルトレジストリに登録されるため、
        # ここでは fresh registry で contracts_of() = () となることだけ確認
        assert self._fresh.contracts_of(_TargetProcess) == ()
        # class-level ClassVar もデフォルト空
        assert _TargetProcess.contracts == ()
        # meta.math_contracts もデフォルト空
        assert _TargetProcess.meta.math_contracts == ()

    def test_classvar_contracts_auto_registered(self) -> None:
        """class-level `contracts = (...)` 宣言で fresh registry に自動登録."""
        identity = IdentityContract(name="auto_reg_classvar", equation_ref="m#e", lhs="A", rhs="B")

        class _ClassVarProcess(SolverProcess[object, object]):
            meta: ClassVar[ProcessMeta] = ProcessMeta(
                name="_ClassVarProcess",
                module="solve",
                version="0.1.0",
                document_path=_DOC_REL,
                stability="experimental",
                support_tier="dev-only",
            )
            contracts: ClassVar[tuple] = (identity,)
            _skip_registry = True

            def process(self, input_data: object) -> object:
                return input_data

        assert self._fresh.contracts_of(_ClassVarProcess) == (identity,)

    def test_meta_math_contracts_auto_registered(self) -> None:
        """`meta.math_contracts` 経由の宣言も自動登録される."""
        sym = SymmetryContract(name="auto_reg_meta", equation_ref="m#e", matrix_name="K")

        class _MetaContractsProcess(SolverProcess[object, object]):
            meta: ClassVar[ProcessMeta] = ProcessMeta(
                name="_MetaContractsProcess",
                module="solve",
                version="0.1.0",
                document_path=_DOC_REL,
                stability="experimental",
                support_tier="dev-only",
                math_contracts=(sym,),
            )
            _skip_registry = True

            def process(self, input_data: object) -> object:
                return input_data

        assert self._fresh.contracts_of(_MetaContractsProcess) == (sym,)

    def test_both_sources_combined(self) -> None:
        """class-level と meta 双方の契約が合算される."""
        identity = IdentityContract(name="combined_classvar", equation_ref="m#e", lhs="A", rhs="B")
        sym = SymmetryContract(name="combined_meta", equation_ref="m#e", matrix_name="K")

        class _CombinedProcess(SolverProcess[object, object]):
            meta: ClassVar[ProcessMeta] = ProcessMeta(
                name="_CombinedProcess",
                module="solve",
                version="0.1.0",
                document_path=_DOC_REL,
                stability="experimental",
                support_tier="dev-only",
                math_contracts=(sym,),
            )
            contracts: ClassVar[tuple] = (identity,)
            _skip_registry = True

            def process(self, input_data: object) -> object:
                return input_data

        registered = self._fresh.contracts_of(_CombinedProcess)
        # class-level が先、meta が後という順序（`_register_math_contracts` の実装に準拠）
        assert registered == (identity, sym)

    def test_duplicate_across_sources_rejected_at_subclass_time(self) -> None:
        """class-level と meta で同名契約を宣言するとクラス定義時点で ValueError."""
        identity_a = IdentityContract(name="dup_across", equation_ref="m#e", lhs="A", rhs="B")
        identity_b = IdentityContract(name="dup_across", equation_ref="m#e2", lhs="C", rhs="D")

        with pytest.raises(ValueError, match="重複"):

            class _BadProcess(SolverProcess[object, object]):
                meta: ClassVar[ProcessMeta] = ProcessMeta(
                    name="_BadProcess",
                    module="solve",
                    version="0.1.0",
                    document_path=_DOC_REL,
                    stability="experimental",
                    support_tier="dev-only",
                    math_contracts=(identity_b,),
                )
                contracts: ClassVar[tuple] = (identity_a,)
                _skip_registry = True

                def process(self, input_data: object) -> object:
                    return input_data


class TestPackageExports:
    """`xkep_cae.mathematics` の公開 API に Phase A-2 追加分が含まれていること."""

    def test_exports(self) -> None:
        import xkep_cae.mathematics as m

        expected = {
            "ProcessContractRegistry",
            "verified_by",
            "DummyVerifyProcessError",
        }
        assert expected.issubset(set(m.__all__))
        for name in expected:
            assert hasattr(m, name)
