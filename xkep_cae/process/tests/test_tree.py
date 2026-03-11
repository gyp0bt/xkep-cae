"""ProcessTree の1:1テスト.

テスト対象: xkep_cae/process/tree.py
"""

from __future__ import annotations

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import PreProcess, SolverProcess
from xkep_cae.process.tree import NodeType, ProcessNode, ProcessTree

# ============================================================
# テスト用プロセス
# ============================================================


class TreePreProcess(PreProcess[str, str]):
    meta = ProcessMeta(name="Tree Pre", module="pre", document_path="docs/dummy.md")
    uses = []

    def process(self, input_data: str) -> str:
        return f"pre:{input_data}"


class TreeSolverProcess(SolverProcess[str, str]):
    meta = ProcessMeta(name="Tree Solver", module="solve", document_path="docs/dummy.md")
    uses = [TreePreProcess]

    def process(self, input_data: str) -> str:
        return f"solve:{input_data}"


class TreeBrokenProcess(SolverProcess[str, str]):
    """uses にツリー外のプロセスを宣言するプロセス."""

    meta = ProcessMeta(name="Tree Broken", module="solve", document_path="docs/dummy.md")
    uses = [TreePreProcess]

    def process(self, input_data: str) -> str:
        return ""


# ============================================================
# TestProcessTree
# ============================================================


class TestProcessTree:
    """ProcessTree のテスト."""

    def _build_valid_tree(self) -> ProcessTree:
        pre_node = ProcessNode(process_class=TreePreProcess)
        solver_node = ProcessNode(
            process_class=TreeSolverProcess,
            children=[pre_node],
        )
        return ProcessTree(root=solver_node, name="Test Pipeline")

    def test_valid_tree_no_errors(self) -> None:
        """正常なツリーはエラーなし."""
        tree = self._build_valid_tree()
        errors = tree.validate()
        assert errors == []

    def test_missing_dependency_detected(self) -> None:
        """uses 依存先がツリーに含まれない場合にエラー."""
        # TreeBrokenProcess は TreePreProcess を uses するが、ツリーに含めない
        broken_node = ProcessNode(process_class=TreeBrokenProcess)
        tree = ProcessTree(root=broken_node, name="Broken")
        errors = tree.validate()
        assert any("TreePreProcess" in e for e in errors)

    def test_circular_dependency_detected(self) -> None:
        """循環依存が検出されること."""
        node = ProcessNode(process_class=TreePreProcess)
        node.children = [node]  # 自己参照
        tree = ProcessTree(root=node)
        errors = tree.validate()
        assert any("循環依存" in e for e in errors)

    def test_to_markdown(self) -> None:
        """Markdown 出力が生成されること."""
        tree = self._build_valid_tree()
        md = tree.to_markdown()
        assert "TreeSolverProcess" in md
        assert "TreePreProcess" in md

    def test_to_mermaid(self) -> None:
        """Mermaid 出力が生成されること."""
        tree = self._build_valid_tree()
        mermaid = tree.to_mermaid()
        assert "graph TD" in mermaid
        assert "TreeSolverProcess --> TreePreProcess" in mermaid

    def test_node_types(self) -> None:
        """NodeType が正しく設定できること."""
        node = ProcessNode(
            process_class=TreePreProcess,
            node_type=NodeType.PARALLEL,
        )
        assert node.node_type == NodeType.PARALLEL
