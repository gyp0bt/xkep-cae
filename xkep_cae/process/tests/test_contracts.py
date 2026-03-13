"""process-architecture.md §13 契約違反テスト.

全 AbstractProcess サブクラスをレジストリから自動走査し、
§13 の契約を網羅的に検証する。

**設計意図**: Phase 7 未実装の項目は意図的に FAIL する。
後続AIセッションが FAIL を解消することで Phase 7 を完遂する。

テスト対象:
  C3  — 全プロセスに @binds_to テストが紐付けられていること
  C6  — Strategy Protocol 実装が意味論的契約を満たすこと
  C9  — execute() が入力データを変更しないこと
  C12 — BatchProcess 具象クラスが存在し順序制約を満たすこと
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from xkep_cae.process.base import AbstractProcess

# ============================================================
# レジストリ構築（全モジュールをインポート）
# ============================================================

_CONCRETE_MODULES = [
    "xkep_cae.process.concrete.pre_mesh",
    "xkep_cae.process.concrete.pre_contact",
    "xkep_cae.process.concrete.solve_ncp",
    "xkep_cae.process.concrete.post_export",
    "xkep_cae.process.concrete.post_render",
]
_STRATEGY_MODULES = [
    "xkep_cae.process.strategies.penalty",
    "xkep_cae.process.strategies.friction",
    "xkep_cae.process.strategies.contact_force",
    "xkep_cae.process.strategies.contact_geometry",
    "xkep_cae.process.strategies.time_integration",
]

for _mod in _CONCRETE_MODULES + _STRATEGY_MODULES:
    try:
        importlib.import_module(_mod)
    except Exception:
        pass


def _is_test_fixture(cls: type) -> bool:
    module = getattr(cls, "__module__", "")
    return ".tests." in module or module.startswith("tests.")


def _all_concrete_processes() -> list[tuple[str, type]]:
    """レジストリから全具象プロセス（テストフィクスチャ除外）を返す."""
    return [
        (name, cls)
        for name, cls in sorted(AbstractProcess._registry.items())
        if not _is_test_fixture(cls)
    ]


def _all_non_deprecated() -> list[tuple[str, type]]:
    """deprecated でない全具象プロセスを返す."""
    return [
        (name, cls)
        for name, cls in _all_concrete_processes()
        if not (hasattr(cls, "meta") and cls.meta.deprecated)
    ]


# ============================================================
# C3: 全プロセスに @binds_to テストが紐付けられていること
# ============================================================


class TestContractC3TestBinding:
    """C3: 全 non-deprecated プロセスに @binds_to テストが紐付けられていること.

    修正方法: concrete/test_*.py を作成し @binds_to(XxxProcess) で紐付ける。
    """

    @pytest.mark.parametrize(
        "name,cls",
        _all_non_deprecated(),
        ids=[name for name, _ in _all_non_deprecated()],
    )
    def test_has_test_binding(self, name: str, cls: type) -> None:
        assert cls._test_class is not None, (
            f"{name} にテストが紐付けられていない。"
            f"@binds_to({name}) デコレータでテストクラスを紐付けてください。"
        )


# ============================================================
# C6: Strategy Protocol 意味論的契約テスト
# ============================================================


def _strategy_processes_for(suffix: str) -> list[tuple[str, type]]:
    """指定サフィックスに一致する Strategy 具象クラスを返す.

    "Penalty" は "PenaltyProcess" で終わるクラスのみ（SmoothPenaltyContactForce を除外）。
    """
    return [
        (name, cls) for name, cls in _all_concrete_processes() if name.endswith(f"{suffix}Process")
    ]


class TestContractC6StrategySemantics:
    """C6: Strategy 具象クラスが Protocol の意味論的契約を満たすこと.

    修正方法: 各 Strategy 具象クラスの process() が仕様通りの
    入出力関係を満たすよう実装する。
    """

    @pytest.mark.parametrize(
        "name,cls",
        _strategy_processes_for("ContactForce"),
        ids=[n for n, _ in _strategy_processes_for("ContactForce")],
    )
    def test_contact_force_zero_displacement_gives_zero_force(self, name: str, cls: type) -> None:
        """ゼロ変位でゼロ接触力を返すこと."""
        ndof = 12
        # NCPContactForceProcess は ndof が必須引数
        try:
            instance = cls()
        except TypeError:
            instance = cls(ndof=ndof)
        if not hasattr(instance, "evaluate"):
            pytest.skip(f"{name} に evaluate メソッドがない")
        u = np.zeros(ndof)
        lambdas = np.zeros(ndof)

        class _MockManager:
            active_pairs = []

        try:
            force, residual = instance.evaluate(u, lambdas, _MockManager(), 1e6)
            # ゼロ変位 + ペアなし → ゼロ力
            assert np.allclose(force, 0.0), (
                f"{name}: ゼロ変位でゼロ接触力を期待するが、force={np.linalg.norm(force):.2e}"
            )
        except Exception as e:
            pytest.fail(f"{name}.evaluate() が例外: {e}")

    @pytest.mark.parametrize(
        "name,cls",
        _strategy_processes_for("Penalty"),
        ids=[n for n, _ in _strategy_processes_for("Penalty")],
    )
    def test_penalty_returns_positive(self, name: str, cls: type) -> None:
        """ペナルティ剛性が正の値を返すこと."""
        # Penalty クラスは必須引数を持つものがある
        try:
            instance = cls()
        except TypeError:
            # ManualPenaltyProcess(k_pen), ContinuationPenaltyProcess(k_pen_target)
            try:
                instance = cls(k_pen=1e6)
            except TypeError:
                instance = cls(k_pen_target=1e6)
        if not hasattr(instance, "compute_k_pen"):
            pytest.skip(f"{name} に compute_k_pen メソッドがない")
        try:
            k = instance.compute_k_pen(step=0, total_steps=10)
            assert k > 0, f"{name}: k_pen={k} は正でなければならない"
        except Exception as e:
            pytest.fail(f"{name}.compute_k_pen() が例外: {e}")


# ============================================================
# C9: execute() が入力データを変更しないこと
# ============================================================


class TestContractC9FrozenImmutability:
    """C9: execute() が入力の numpy 配列を変更しないこと.

    修正方法: AbstractProcess.execute() にチェックサム検証を追加する。
    入口でハッシュを記録し、出口で一致を確認する。
    """

    def test_execute_has_immutability_check(self) -> None:
        """execute() メソッドに入力不変性チェックが実装されていること."""
        import inspect

        source = inspect.getsource(AbstractProcess.execute)
        has_check = "checksum" in source or "hash" in source or "_immutable" in source
        assert has_check, (
            "AbstractProcess.execute() に入力データ不変性チェックが未実装。"
            "frozen dataclass の numpy 配列は in-place 変更可能なため、"
            "execute() 入口でチェックサムを記録し出口で検証する仕組みが必要。"
        )


# ============================================================
# C12: BatchProcess 具象クラスの存在と順序制約
# ============================================================


class TestContractC12BatchProcess:
    """C12: BatchProcess 具象クラスが存在し順序制約を満たすこと.

    修正方法: batch/ に StrandBendingBatchProcess を実装し、
    uses 宣言順が process() 内の実行順と一致するようにする。
    """

    def test_batch_process_exists(self) -> None:
        """BatchProcess の具象クラスが少なくとも1つ存在すること."""
        batch_classes = [
            (name, cls)
            for name, cls in _all_concrete_processes()
            if any(base.__name__ == "BatchProcess" and base is not cls for base in cls.__mro__)
        ]
        assert len(batch_classes) >= 1, (
            "BatchProcess の具象クラスが0件。"
            "process-architecture.md §6 の StrandBendingBatchProcess を実装してください。"
        )

    def test_verify_process_exists(self) -> None:
        """VerifyProcess の具象クラスが少なくとも1つ存在すること."""
        verify_classes = [
            (name, cls)
            for name, cls in _all_concrete_processes()
            if any(base.__name__ == "VerifyProcess" and base is not cls for base in cls.__mro__)
        ]
        assert len(verify_classes) >= 1, (
            "VerifyProcess の具象クラスが0件。"
            "process-architecture.md §6 の ConvergenceVerifyProcess を実装してください。"
        )


# ============================================================
# 補助: 全カテゴリカバレッジ
# ============================================================


class TestContractCategoryCoverage:
    """全5カテゴリに具象クラスが存在すること.

    修正方法: verify/ と batch/ ディレクトリに具象クラスを実装する。
    """

    @pytest.mark.parametrize(
        "category",
        ["PreProcess", "SolverProcess", "PostProcess", "VerifyProcess", "BatchProcess"],
    )
    def test_category_has_concrete(self, category: str) -> None:
        matching = [
            name
            for name, cls in _all_concrete_processes()
            if any(base.__name__ == category for base in cls.__mro__)
        ]
        assert len(matching) >= 1, (
            f"{category} の具象クラスが0件。process-architecture.md §3 を参照して実装してください。"
        )
