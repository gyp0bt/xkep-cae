"""接触グラフスナップショット生成 Process.

__xkep_cae_deprecated/contact/graph.py の snapshot_contact_graph を移植。
manager / ContactStatus は duck typing で受け取る。
"""

from __future__ import annotations

from dataclasses import dataclass

from xkep_cae.core import ProcessMeta, SolverProcess


@dataclass(frozen=True)
class _ContactEdgeOutput:
    """接触グラフのエッジ（1接触ペア）."""

    elem_a: int
    elem_b: int
    gap: float
    p_n: float
    status: str
    stick: bool
    dissipation: float
    s: float
    t: float


@dataclass(frozen=True)
class _ContactGraphOutput:
    """接触グラフの1スナップショット."""

    step: int
    load_factor: float
    nodes: set[int]
    edges: list[_ContactEdgeOutput]
    n_total_pairs: int


@dataclass(frozen=True)
class ContactGraphInput:
    """接触グラフスナップショットの入力."""

    manager: object
    step: int = 0
    load_factor: float = 0.0


@dataclass(frozen=True)
class ContactGraphOutput:
    """接触グラフスナップショットの出力."""

    graph: _ContactGraphOutput


def _snapshot_contact_graph(
    manager: object,
    step: int = 0,
    load_factor: float = 0.0,
) -> _ContactGraphOutput:
    """ContactManager の現在の状態からグラフスナップショットを生成する.

    manager は duck typing: .pairs 属性を持つオブジェクト。
    各 pair は .state.status, .elem_a, .elem_b, .state.gap, .state.p_n,
    .state.stick, .state.dissipation, .state.s, .state.t を持つ。
    """
    nodes: set[int] = set()
    edges: list[_ContactEdgeOutput] = []

    # ContactStatus.INACTIVE の文字列表現を使って判定
    for pair in manager.pairs:
        status = pair.state.status
        # INACTIVE ペアはスキップ（enum の name or value で判定）
        status_name = status.name if hasattr(status, "name") else str(status)
        if status_name == "INACTIVE":
            continue

        status_str = "SLIDING" if status_name == "SLIDING" else "ACTIVE"

        nodes.add(pair.elem_a)
        nodes.add(pair.elem_b)
        edges.append(
            _ContactEdgeOutput(
                elem_a=pair.elem_a,
                elem_b=pair.elem_b,
                gap=pair.state.gap,
                p_n=pair.state.p_n,
                status=status_str,
                stick=pair.state.stick,
                dissipation=pair.state.dissipation,
                s=pair.state.s,
                t=pair.state.t,
            )
        )

    return _ContactGraphOutput(
        step=step,
        load_factor=load_factor,
        nodes=nodes,
        edges=edges,
        n_total_pairs=len(manager.pairs),
    )


class ContactGraphProcess(SolverProcess[ContactGraphInput, ContactGraphOutput]):
    """接触グラフスナップショット生成 Process.

    現在の接触状態からグラフ構造を抽出する。
    """

    meta = ProcessMeta(
        name="ContactGraph",
        module="solve",
        version="1.0.0",
        document_path="docs/contact_friction.md",
    )

    def process(self, input_data: ContactGraphInput) -> ContactGraphOutput:
        graph = _snapshot_contact_graph(
            input_data.manager,
            input_data.step,
            input_data.load_factor,
        )
        return ContactGraphOutput(graph=graph)
