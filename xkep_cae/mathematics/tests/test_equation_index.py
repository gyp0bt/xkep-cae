"""`equation_index.load` / `EquationIndex.resolve` の単体テスト.

status-349（Phase B-2）で新設。`docs/math/*.md` のアンカー抽出・重複検出・
参照解決 API を検証する。

テスト方針
----------
- **実台帳への依存を排除**: `tmp_path` に疑似 md を書いて load() に渡す
- **実台帳の統合テスト**: `test_real_ledger_is_well_formed` でプロジェクト
  直下の `docs/math/` が実際に整合することも確認する
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xkep_cae.mathematics import (
    DuplicateAnchorError,
    EquationIndex,
    UnresolvedReferenceError,
    load_equation_index,
)
from xkep_cae.mathematics.equation_index import load


def _write_ledger(root: Path, files: dict[str, str]) -> None:
    """疑似台帳を作成（README.md も含める）."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# 索引（テスト用）\n", encoding="utf-8")
    for name, body in files.items():
        (root / name).write_text(body, encoding="utf-8")


# ======================================================================
# アンカー抽出
# ======================================================================


class TestLoadEquationIndexExtraction:
    """`load()` が `<a id>` タグを正しく抽出することを確認."""

    def test_single_file_single_anchor(self, tmp_path: Path) -> None:
        """最小ケース: 1 ファイル 1 アンカー."""
        ledger = tmp_path / "docs" / "math"
        _write_ledger(
            ledger,
            {"01_kin.md": '<a id="eq-foo"></a>\n\n## 1. Foo\n'},
        )
        idx = load(ledger)
        assert idx.total_anchors == 1
        assert idx.anchors_in("01_kin.md") == frozenset({"eq-foo"})

    def test_multiple_anchors_per_file(self, tmp_path: Path) -> None:
        """1 ファイルに複数アンカー（eq/inv/sym/sec 混在）."""
        ledger = tmp_path / "docs" / "math"
        _write_ledger(
            ledger,
            {
                "03_huber.md": (
                    '<a id="eq-pn"></a>\n\n'
                    '<a id="inv-pn-nonneg"></a>\n\n'
                    '<a id="sym-kmat"></a>\n\n'
                    '<a id="sec-ndir"></a>\n\n'
                )
            },
        )
        idx = load(ledger)
        assert idx.total_anchors == 4
        assert idx.anchors_in("03_huber.md") == frozenset(
            {"eq-pn", "inv-pn-nonneg", "sym-kmat", "sec-ndir"}
        )

    def test_multiple_files(self, tmp_path: Path) -> None:
        """複数ファイルの独立抽出."""
        ledger = tmp_path / "docs" / "math"
        _write_ledger(
            ledger,
            {
                "01_kin.md": '<a id="eq-a"></a>',
                "02_geo.md": '<a id="eq-b"></a>',
                "03_huber.md": '<a id="eq-c"></a><a id="eq-d"></a>',
            },
        )
        idx = load(ledger)
        assert idx.total_anchors == 4
        assert idx.anchors_in("01_kin.md") == frozenset({"eq-a"})
        assert idx.anchors_in("02_geo.md") == frozenset({"eq-b"})
        assert idx.anchors_in("03_huber.md") == frozenset({"eq-c", "eq-d"})

    def test_readme_not_scanned(self, tmp_path: Path) -> None:
        """README.md はアンカー抽出対象外."""
        ledger = tmp_path / "docs" / "math"
        _write_ledger(
            ledger,
            {"01_kin.md": '<a id="eq-a"></a>'},
        )
        # README にアンカーを追記しても無視される
        (ledger / "README.md").write_text('<a id="eq-should-be-ignored"></a>', encoding="utf-8")
        idx = load(ledger)
        assert idx.total_anchors == 1
        assert "eq-should-be-ignored" not in idx.anchors_in("README.md")

    def test_unsupported_prefix_ignored(self, tmp_path: Path) -> None:
        """`eq-/inv-/sym-/sec-` 以外の prefix は抽出されない."""
        ledger = tmp_path / "docs" / "math"
        _write_ledger(
            ledger,
            {
                "bad.md": (
                    '<a id="foo-bar"></a>\n'  # prefix 不一致
                    '<a id="eq-ok"></a>\n'
                    '<a id="my-custom"></a>\n'  # prefix 不一致
                )
            },
        )
        idx = load(ledger)
        assert idx.anchors_in("bad.md") == frozenset({"eq-ok"})


# ======================================================================
# 重複検出
# ======================================================================


class TestDuplicateDetection:
    """同一ファイル内でのアンカー重複検出."""

    def test_no_duplicates(self, tmp_path: Path) -> None:
        """重複なしなら `is_unique=True`."""
        ledger = tmp_path / "docs" / "math"
        _write_ledger(ledger, {"a.md": '<a id="eq-x"></a><a id="eq-y"></a>'})
        idx = load(ledger)
        assert idx.is_unique
        assert idx.duplicates == ()

    def test_single_duplicate(self, tmp_path: Path) -> None:
        """同一ファイル内で 2 回定義された場合の検出."""
        ledger = tmp_path / "docs" / "math"
        _write_ledger(
            ledger,
            {"a.md": '<a id="eq-x"></a>\n\n<a id="eq-x"></a>\n'},
        )
        idx = load(ledger)
        assert not idx.is_unique
        assert len(idx.duplicates) == 1
        dup = idx.duplicates[0]
        assert isinstance(dup, DuplicateAnchorError)
        assert dup.file_name == "a.md"
        assert dup.anchor == "eq-x"
        assert dup.count == 2

    def test_duplicate_count_accurate(self, tmp_path: Path) -> None:
        """3 回以上定義された場合も count が正しい."""
        ledger = tmp_path / "docs" / "math"
        _write_ledger(
            ledger,
            {"a.md": '<a id="eq-x"></a>' * 3},
        )
        idx = load(ledger)
        assert len(idx.duplicates) == 1
        assert idx.duplicates[0].count == 3

    def test_same_anchor_different_files_is_allowed(self, tmp_path: Path) -> None:
        """現状の規約ではファイル間の重複は duplicates に計上しない.

        命名規約（docs/math/README.md）はファイルごとの一意性のみを要求する。
        ファイル間の重複は `<file>#<anchor>` 参照で区別可能なため許容する。
        """
        ledger = tmp_path / "docs" / "math"
        _write_ledger(
            ledger,
            {
                "a.md": '<a id="eq-common"></a>',
                "b.md": '<a id="eq-common"></a>',
            },
        )
        idx = load(ledger)
        assert idx.is_unique


# ======================================================================
# 参照解決
# ======================================================================


class TestResolve:
    """`EquationIndex.resolve()` の参照解決検査."""

    @pytest.fixture
    def sample_index(self, tmp_path: Path) -> EquationIndex:
        ledger = tmp_path / "docs" / "math"
        _write_ledger(
            ledger,
            {
                "01_kin.md": '<a id="eq-hermite"></a>',
                "03_huber.md": ('<a id="eq-kc"></a>\n<a id="eq-kc-full-decomposition"></a>\n'),
            },
        )
        return load(ledger)

    def test_valid_reference(self, sample_index: EquationIndex) -> None:
        """正しい参照は None（エラーなし）を返す."""
        assert sample_index.resolve("03_huber.md#eq-kc") is None
        assert sample_index.resolve("01_kin.md#eq-hermite") is None

    def test_bad_format(self, sample_index: EquationIndex) -> None:
        """フォーマット不正は `bad_format` エラー."""
        err = sample_index.resolve("no_hash_here")
        assert isinstance(err, UnresolvedReferenceError)
        assert err.reason == "bad_format"

    def test_bad_format_no_md_suffix(self, sample_index: EquationIndex) -> None:
        """``.md`` がない参照は `bad_format`."""
        err = sample_index.resolve("03_huber#eq-kc")
        assert err is not None
        assert err.reason == "bad_format"

    def test_bad_prefix(self, sample_index: EquationIndex) -> None:
        """許可 prefix 外（例 `foo-`）は `bad_format`."""
        err = sample_index.resolve("03_huber.md#foo-bar")
        assert err is not None
        assert err.reason == "bad_format"

    def test_missing_file(self, sample_index: EquationIndex) -> None:
        """未知ファイル参照は `missing_file`."""
        err = sample_index.resolve("99_nowhere.md#eq-kc")
        assert err is not None
        assert err.reason == "missing_file"
        assert "99_nowhere.md" in err.detail

    def test_missing_anchor(self, sample_index: EquationIndex) -> None:
        """既知ファイル内の未知アンカーは `missing_anchor`."""
        err = sample_index.resolve("03_huber.md#eq-ghost")
        assert err is not None
        assert err.reason == "missing_anchor"
        assert "eq-ghost" in err.detail


# ======================================================================
# 一括検証
# ======================================================================


class TestValidate:
    """`EquationIndex.validate()` の一括解決."""

    def test_all_valid_returns_empty(self, tmp_path: Path) -> None:
        """全て正しい参照なら空リスト."""
        ledger = tmp_path / "docs" / "math"
        _write_ledger(
            ledger,
            {"a.md": '<a id="eq-x"></a><a id="eq-y"></a>'},
        )
        idx = load(ledger)
        errs = idx.validate(["a.md#eq-x", "a.md#eq-y"])
        assert errs == []

    def test_partial_failure_preserves_order(self, tmp_path: Path) -> None:
        """一部失敗時は失敗分のみ順序保持で返る."""
        ledger = tmp_path / "docs" / "math"
        _write_ledger(
            ledger,
            {"a.md": '<a id="eq-x"></a>'},
        )
        idx = load(ledger)
        errs = idx.validate(
            [
                "a.md#eq-x",  # OK
                "a.md#eq-missing",  # NG: missing_anchor
                "z.md#eq-x",  # NG: missing_file
            ]
        )
        assert len(errs) == 2
        assert errs[0].reason == "missing_anchor"
        assert errs[1].reason == "missing_file"


# ======================================================================
# エッジケース
# ======================================================================


class TestEdgeCases:
    def test_missing_ledger_root_returns_empty(self, tmp_path: Path) -> None:
        """存在しないディレクトリを渡しても例外を上げず空 Index を返す."""
        idx = load(tmp_path / "does_not_exist")
        assert idx.total_anchors == 0
        assert idx.is_unique

    def test_frozenset_is_immutable(self, tmp_path: Path) -> None:
        """抽出結果の frozenset は書き換え不可."""
        ledger = tmp_path / "docs" / "math"
        _write_ledger(ledger, {"a.md": '<a id="eq-x"></a>'})
        idx = load(ledger)
        anchors = idx.anchors_in("a.md")
        with pytest.raises(AttributeError):
            anchors.add("eq-new")  # type: ignore[attr-defined]


# ======================================================================
# 実台帳の統合テスト
# ======================================================================


class TestRealLedger:
    """プロジェクト直下の `docs/math/` が整合することを確認.

    Phase B-2 完了時点で 6 章 55 アンカーが重複ゼロで整備されている。
    """

    def test_real_ledger_loads_without_error(self) -> None:
        """実台帳が例外なくロードできる."""
        idx = load_equation_index()
        # 台帳が見つかった場合のみ以降を検査（CI の作業ディレクトリ依存を吸収）
        if idx.total_anchors == 0:
            pytest.skip("docs/math/ not found from cwd (likely running outside repo root)")
        assert idx.is_unique, f"実台帳に重複アンカー: {[str(d) for d in idx.duplicates]}"

    def test_real_ledger_contract_references_resolve(self) -> None:
        """既存の `MathematicalContract.equation_ref` docstring 例が全て解決可能."""
        idx = load_equation_index()
        if idx.total_anchors == 0:
            pytest.skip("docs/math/ not found from cwd")
        # status-348 の整合性確認と同じ参照集合
        docstring_examples = [
            "03_huber_contact_penalty.md#eq-kc",
            "03_huber_contact_penalty.md#eq-kc-def",
            "03_huber_contact_penalty.md#eq-pn",
            "03_huber_contact_penalty.md#eq-kc-fd",
            "03_huber_contact_penalty.md#eq-kmat",
            "03_huber_contact_penalty.md#eq-kc-full-decomposition",
        ]
        errs = idx.validate(docstring_examples)
        assert errs == [], f"解決失敗: {[str(e) for e in errs]}"
