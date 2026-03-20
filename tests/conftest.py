"""pytest 共通設定 — 準静的ソルバー自動検知.

準静的ソルバー（NewtonUzawaStaticProcess）を使用したテストを自動検知し、
セッション終了時にレポートを出力する。

準静的ソルバーで通ったテストは「動的ソルバーでも動く」ことを意味しない。
準静的は動的ソルバーの足がかりに過ぎず、最終的には動的ソルバーで検証すべき。

意図的に準静的ソルバーを使用するテストは ``@pytest.mark.static_solver_ok``
でマークする。
"""

from __future__ import annotations

import warnings

import pytest

from xkep_cae.core.diagnostics import StaticSolverWarning

# セッション全体で検知結果を蓄積
_static_solver_tests: list[str] = []


@pytest.fixture(autouse=True)
def _detect_static_solver_usage(request: pytest.FixtureRequest) -> None:  # noqa: ANN001
    """全テストで StaticSolverWarning を監視し、検知したら記録."""
    # static_solver_ok マーカー付きは検知対象外
    if request.node.get_closest_marker("static_solver_ok"):
        yield
        return

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", StaticSolverWarning)
        yield
        for w in caught:
            if issubclass(w.category, StaticSolverWarning):
                _static_solver_tests.append(request.node.nodeid)
                break


def pytest_terminal_summary(
    terminalreporter: object,
    exitstatus: int,  # noqa: ARG001
    config: pytest.Config,  # noqa: ARG001
) -> None:
    """セッション終了時に準静的ソルバー使用テストのレポートを出力."""
    if not _static_solver_tests:
        return

    # 重複除去（parametrize 等で同一テストが複数回記録される場合）
    unique = sorted(set(_static_solver_tests))

    writer = terminalreporter  # type: ignore[assignment]
    writer.section("Static Solver Detection Report")  # type: ignore[attr-defined]
    writer.line(  # type: ignore[attr-defined]
        f"以下の {len(unique)} テストが準静的ソルバーを使用しています:"
    )
    writer.line(  # type: ignore[attr-defined]
        "（動的ソルバーでの検証が必要。意図的な場合は @pytest.mark.static_solver_ok を付与）"
    )
    writer.line("")  # type: ignore[attr-defined]
    for nodeid in unique:
        writer.line(f"  - {nodeid}")  # type: ignore[attr-defined]
