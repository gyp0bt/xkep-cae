"""docs/math/ 台帳のアンカー抽出・参照解決レジストリ.

status-349 (MCDD Phase B-2) で新設。`docs/math/*.md` を走査して
`<a id="eq-..."></a>` / `<a id="inv-..."></a>` / `<a id="sym-..."></a>` /
`<a id="sec-..."></a>` 形式のアンカーを抽出し、
`MathematicalContract.equation_ref` の ``<file>#<anchor>`` 参照を機械的に
解決する API を提供する。

責務
----
- **アンカー抽出**: `docs/math/*.md` の全ファイルから ``<a id="...">`` を抽出
- **重複検出**: ファイル全体で同一アンカーが複数箇所に出現した場合を検出
- **参照解決**: `03_huber_contact_penalty.md#eq-kc-full-decomposition` のような
  参照を ``(file, anchor)`` に分解し、アンカーが存在するか検査
- **整合性検査** (`validate()`): `contracts/validate_process_contracts.py` の
  C15 拡張から呼ばれる単一のエントリポイント

非責務
------
- Markdown 構文の完全パース（`mistletoe` 等は導入しない。正規表現で十分）
- 実測値・tol の検査（本モジュールは「式の参照解決」のみ）
- 数式の TeX 構文検査

設計原則
--------
- **台帳ディレクトリはデフォルト `docs/math/`** だが `load()` で差し替え可能
  （テスト容易性）
- **循環 import 回避**: `contracts.py` には依存しない。`equation_ref` 文字列の
  形式のみに依存する
- **frozen dataclass**: 抽出結果は不変（C17 整合）

設計仕様: docs/mathematics.md, docs/math/README.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ``<a id="eq-kc"></a>`` 形式のアンカー抽出パターン
# 許可 prefix: eq- / inv- / sym- / sec- （docs/math/README.md 命名規約）
_ANCHOR_PATTERN = re.compile(r'<a\s+id="((?:eq|inv|sym|sec)-[A-Za-z0-9_-]+)"\s*></a>')

# ``03_huber_contact_penalty.md#eq-kc`` 形式の参照パターン
_REF_PATTERN = re.compile(
    r"^(?P<file>[A-Za-z0-9_.-]+\.md)#(?P<anchor>(?:eq|inv|sym|sec)-[A-Za-z0-9_-]+)$"
)

# プロジェクトルートからの台帳配置（実行ディレクトリ非依存の解決に使用）
_DEFAULT_LEDGER_RELATIVE = Path("docs/math")


@dataclass(frozen=True)
class DuplicateAnchorError:
    """同一ファイル内でアンカーが複数回定義されている不整合.

    Attributes:
        file_name: 該当ファイル名（例 ``"03_huber_contact_penalty.md"``）
        anchor: 重複したアンカー ID（例 ``"eq-kc"``）
        count: 出現回数（2 以上）
    """

    file_name: str
    anchor: str
    count: int

    def __str__(self) -> str:
        return (
            f"[DuplicateAnchor] {self.file_name}#{self.anchor} が {self.count} 回定義されています"
        )


@dataclass(frozen=True)
class UnresolvedReferenceError:
    """`equation_ref` が台帳に存在しない参照を指している不整合.

    Attributes:
        equation_ref: 解決失敗した参照文字列（例 ``"foo.md#eq-xx"``）
        reason: ``"bad_format"`` / ``"missing_file"`` / ``"missing_anchor"``
        detail: 人間可読な詳細（どのファイル/アンカーが見つからなかったか）
    """

    equation_ref: str
    reason: str
    detail: str

    def __str__(self) -> str:
        return f"[UnresolvedReference] {self.equation_ref}: {self.reason} — {self.detail}"


@dataclass(frozen=True)
class EquationIndex:
    """`docs/math/*.md` のアンカー抽出結果.

    ``load()`` で生成し、以降は読み取り専用。`resolve()` で `equation_ref`
    文字列を ``(file, anchor)`` に分解・検証する。

    Attributes:
        ledger_root: 台帳ディレクトリの絶対パス
        anchors_by_file: ``{ファイル名: {アンカー, ...}}`` マップ
        duplicates: 重複アンカー一覧（空なら一意性 OK）
    """

    ledger_root: Path
    anchors_by_file: dict[str, frozenset[str]] = field(default_factory=dict)
    duplicates: tuple[DuplicateAnchorError, ...] = ()

    # ------------------------------------------------------------------
    # 参照解決 API
    # ------------------------------------------------------------------

    def resolve(self, equation_ref: str) -> UnresolvedReferenceError | None:
        """`equation_ref` を解決し、不整合があればエラーを返す.

        Args:
            equation_ref: ``<file>#<anchor>`` 形式の参照文字列

        Returns:
            解決成功時は ``None``、失敗時は `UnresolvedReferenceError`
        """
        m = _REF_PATTERN.match(equation_ref)
        if m is None:
            return UnresolvedReferenceError(
                equation_ref=equation_ref,
                reason="bad_format",
                detail=(
                    "期待形式 '<file>.md#<prefix>-<id>'（prefix は eq/inv/sym/sec のいずれか）"
                ),
            )
        file_name = m.group("file")
        anchor = m.group("anchor")
        if file_name not in self.anchors_by_file:
            return UnresolvedReferenceError(
                equation_ref=equation_ref,
                reason="missing_file",
                detail=(
                    f"{file_name} が台帳 {self.ledger_root} に存在しません "
                    f"(known: {sorted(self.anchors_by_file)})"
                ),
            )
        if anchor not in self.anchors_by_file[file_name]:
            return UnresolvedReferenceError(
                equation_ref=equation_ref,
                reason="missing_anchor",
                detail=(
                    f"{file_name} に {anchor} が存在しません "
                    f"(既知 {len(self.anchors_by_file[file_name])} アンカー)"
                ),
            )
        return None

    def validate(self, equation_refs: list[str]) -> list[UnresolvedReferenceError]:
        """`equation_refs` を一括解決し、未解決分を返す.

        Args:
            equation_refs: `MathematicalContract.equation_ref` 収集リスト

        Returns:
            未解決参照のエラー一覧（空なら全解決 OK）
        """
        errs: list[UnresolvedReferenceError] = []
        for ref in equation_refs:
            err = self.resolve(ref)
            if err is not None:
                errs.append(err)
        return errs

    # ------------------------------------------------------------------
    # 情報参照 API
    # ------------------------------------------------------------------

    @property
    def total_anchors(self) -> int:
        """全ファイルのアンカー合計数."""
        return sum(len(s) for s in self.anchors_by_file.values())

    @property
    def is_unique(self) -> bool:
        """全ファイルで重複がないなら True."""
        return len(self.duplicates) == 0

    def anchors_in(self, file_name: str) -> frozenset[str]:
        """指定ファイルのアンカー集合（存在しない場合は空集合）."""
        return self.anchors_by_file.get(file_name, frozenset())


# ======================================================================
# ローダ
# ======================================================================


def _find_default_ledger_root() -> Path:
    """`docs/math/` の既定位置を探索する.

    実行ディレクトリから順に親方向に遡り、`docs/math/README.md` が
    存在する最初のディレクトリを採用する。
    """
    start = Path.cwd().resolve()
    for cand in (start, *start.parents):
        ledger = cand / _DEFAULT_LEDGER_RELATIVE
        if (ledger / "README.md").is_file():
            return ledger
    # フォールバック: 本ファイルからの相対位置
    return (Path(__file__).resolve().parents[2] / _DEFAULT_LEDGER_RELATIVE).resolve()


def load(ledger_root: Path | str | None = None) -> EquationIndex:
    """台帳ディレクトリをスキャンして `EquationIndex` を構築する.

    Args:
        ledger_root: 台帳ディレクトリ。`None` なら既定位置を探索。

    Returns:
        抽出結果を保持する `EquationIndex`
    """
    if ledger_root is None:
        root = _find_default_ledger_root()
    else:
        root = Path(ledger_root).resolve()

    anchors_by_file: dict[str, frozenset[str]] = {}
    duplicates: list[DuplicateAnchorError] = []

    if not root.is_dir():
        # 台帳が無い場合は空の EquationIndex を返す（テスト環境向け）
        return EquationIndex(
            ledger_root=root,
            anchors_by_file={},
            duplicates=(),
        )

    for md in sorted(root.glob("*.md")):
        # README.md はアンカー定義対象外（索引ファイル）
        if md.name == "README.md":
            continue
        text = md.read_text(encoding="utf-8")
        found = _ANCHOR_PATTERN.findall(text)
        # 同一ファイル内の重複検出
        counts: dict[str, int] = {}
        for anchor in found:
            counts[anchor] = counts.get(anchor, 0) + 1
        unique_anchors: set[str] = set()
        for anchor, cnt in counts.items():
            if cnt > 1:
                duplicates.append(DuplicateAnchorError(file_name=md.name, anchor=anchor, count=cnt))
            unique_anchors.add(anchor)
        anchors_by_file[md.name] = frozenset(unique_anchors)

    return EquationIndex(
        ledger_root=root,
        anchors_by_file=anchors_by_file,
        duplicates=tuple(duplicates),
    )


__all__ = [
    "DuplicateAnchorError",
    "UnresolvedReferenceError",
    "EquationIndex",
    "load",
]
