"""contracts/validate_process_contracts.py::check_c15_equation_refs の単体テスト.

status-349（Phase B-2）で新設。`MathematicalContract.equation_ref` が
`docs/math/` 台帳に対して解決可能であることを検査する C15 拡張の挙動を、
**実台帳と独立** にフェイク契約で検証する。

テスト方針
----------
- **実台帳への依存を排除**: ``check_c15_equation_refs`` 内部が呼ぶ
  `load_equation_index()` を `monkeypatch` で差し替える
- **実行時契約のみ検査対象**: docstring 例題（未登録）が混入しないことを確認
  — Process class-level `contracts` に宣言したものだけが検出される
"""

from __future__ import annotations

import sys
from pathlib import Path

# 親ディレクトリを sys.path に追加して contracts/ スクリプトをロード
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from xkep_cae.mathematics import (  # noqa: E402
    EquationIndex,
    FDConsistencyContract,
    IdentityContract,
    InequalityContract,
)
from xkep_cae.mathematics.equation_index import load  # noqa: E402


def _load_check_module():
    """contracts.validate_process_contracts をパッケージ名で動的ロード."""
    # ファイル配置は /home/user/xkep-cae/contracts/validate_process_contracts.py
    import importlib.util

    path = _PROJECT_ROOT / "contracts" / "validate_process_contracts.py"
    spec = importlib.util.spec_from_file_location(
        "contracts_validate_process_contracts_test_copy", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fake_ledger(tmp_path: Path, files: dict[str, str]) -> EquationIndex:
    ledger = tmp_path / "docs" / "math"
    ledger.mkdir(parents=True, exist_ok=True)
    (ledger / "README.md").write_text("# index\n", encoding="utf-8")
    for name, body in files.items():
        (ledger / name).write_text(body, encoding="utf-8")
    return load(ledger)


class _FakeProcess:
    """`contracts` ClassVar を持つ最小の擬似 Process 型.

    `check_c15_equation_refs` は `registry: dict[str, type]` を受け取って
    `getattr(cls, "contracts", ())` を参照するだけなので、AbstractProcess を
    継承する必要はない。
    """

    contracts: tuple = ()


# ======================================================================
# 正常系
# ======================================================================


class TestC15EquationRefsValid:
    def test_valid_reference_passes(self, tmp_path: Path, monkeypatch) -> None:
        """正しい equation_ref を持つ契約は違反ゼロ."""
        check_mod = _load_check_module()
        idx = _write_fake_ledger(
            tmp_path,
            {"03_huber.md": '<a id="eq-kc"></a>\n'},
        )
        monkeypatch.setattr(
            "xkep_cae.mathematics.load_equation_index",
            lambda: idx,
        )

        class Good(_FakeProcess):
            contracts = (
                IdentityContract(
                    name="ok",
                    equation_ref="03_huber.md#eq-kc",
                    lhs="K_c",
                    rhs="dfc/du",
                ),
            )

        errors = check_mod.check_c15_equation_refs({"Good": Good})
        assert errors == []

    def test_empty_contracts_is_noop(self, tmp_path: Path, monkeypatch) -> None:
        """contracts 未宣言のクラスは検査対象外."""
        check_mod = _load_check_module()
        idx = _write_fake_ledger(tmp_path, {"a.md": '<a id="eq-x"></a>'})
        monkeypatch.setattr("xkep_cae.mathematics.load_equation_index", lambda: idx)

        class NoContracts(_FakeProcess):
            pass

        errors = check_mod.check_c15_equation_refs({"NoContracts": NoContracts})
        assert errors == []


# ======================================================================
# 異常系
# ======================================================================


class TestC15EquationRefsErrors:
    def test_missing_file_is_flagged(self, tmp_path: Path, monkeypatch) -> None:
        """台帳に存在しないファイルを指す参照を検出."""
        check_mod = _load_check_module()
        idx = _write_fake_ledger(tmp_path, {"03_huber.md": '<a id="eq-kc"></a>'})
        monkeypatch.setattr("xkep_cae.mathematics.load_equation_index", lambda: idx)

        class Bad(_FakeProcess):
            contracts = (
                InequalityContract(
                    name="badref",
                    equation_ref="99_nowhere.md#inv-foo",
                    expr="x",
                    kind="geq",
                    bound="0",
                ),
            )

        errors = check_mod.check_c15_equation_refs({"Bad": Bad})
        assert len(errors) == 1
        assert "missing_file" in errors[0]
        assert "99_nowhere.md#inv-foo" in errors[0]
        assert "Bad.contracts['badref']" in errors[0]

    def test_missing_anchor_is_flagged(self, tmp_path: Path, monkeypatch) -> None:
        """台帳にあるファイルの未定義アンカーを検出."""
        check_mod = _load_check_module()
        idx = _write_fake_ledger(tmp_path, {"03_huber.md": '<a id="eq-kc"></a>'})
        monkeypatch.setattr("xkep_cae.mathematics.load_equation_index", lambda: idx)

        class Bad(_FakeProcess):
            contracts = (
                FDConsistencyContract(
                    name="badanchor",
                    equation_ref="03_huber.md#eq-ghost",
                    vector_name="f",
                    jacobian_name="K",
                ),
            )

        errors = check_mod.check_c15_equation_refs({"Bad": Bad})
        assert len(errors) == 1
        assert "missing_anchor" in errors[0]

    def test_bad_format_is_flagged(self, tmp_path: Path, monkeypatch) -> None:
        """`<file>.md#<anchor>` 形式に合致しない参照を検出."""
        check_mod = _load_check_module()
        idx = _write_fake_ledger(tmp_path, {"a.md": '<a id="eq-x"></a>'})
        monkeypatch.setattr("xkep_cae.mathematics.load_equation_index", lambda: idx)

        class Bad(_FakeProcess):
            contracts = (
                IdentityContract(
                    name="badfmt",
                    equation_ref="no_hash_or_md",
                    lhs="x",
                    rhs="y",
                ),
            )

        errors = check_mod.check_c15_equation_refs({"Bad": Bad})
        assert len(errors) == 1
        assert "bad_format" in errors[0]

    def test_duplicate_anchor_is_flagged(self, tmp_path: Path, monkeypatch) -> None:
        """台帳内アンカー重複を C15(math) として計上."""
        check_mod = _load_check_module()
        idx = _write_fake_ledger(
            tmp_path,
            {"a.md": '<a id="eq-x"></a>\n<a id="eq-x"></a>'},
        )
        monkeypatch.setattr("xkep_cae.mathematics.load_equation_index", lambda: idx)

        errors = check_mod.check_c15_equation_refs({})
        # 台帳に重複があれば、契約未宣言でも C15(math) 違反が 1 件計上される
        assert len(errors) == 1
        assert "台帳アンカー重複" in errors[0]
        assert "eq-x" in errors[0]

    def test_empty_ledger_yields_error(self, tmp_path: Path, monkeypatch) -> None:
        """台帳が空なら explicit error を返す（cwd 依存バグの早期検出）."""
        check_mod = _load_check_module()
        # 存在するが README だけのディレクトリを指す
        empty_root = tmp_path / "empty_docs" / "math"
        empty_root.mkdir(parents=True)
        idx = load(empty_root)
        monkeypatch.setattr("xkep_cae.mathematics.load_equation_index", lambda: idx)

        errors = check_mod.check_c15_equation_refs({})
        assert len(errors) == 1
        assert "台帳が空" in errors[0]


# ======================================================================
# docstring 例題の除外
# ======================================================================


class TestDocstringExamplesExcluded:
    """`contracts.py` docstring 内の例題が検査対象に混入しないことを確認.

    docstring 例題は ClassVar 未宣言のため、`getattr(cls, "contracts", ())`
    経由では収集されない。これを実証的に確認する。
    """

    def test_only_classvar_contracts_collected(self, tmp_path: Path, monkeypatch) -> None:
        """docstring で例示された参照が検出対象外であることを確認."""
        check_mod = _load_check_module()
        # 存在しないアンカーを docstring で "例示" するクラス
        # （ただし ClassVar には何も設定しない）
        idx = _write_fake_ledger(
            tmp_path,
            {"03_huber.md": '<a id="eq-kc"></a>\n'},
        )
        monkeypatch.setattr("xkep_cae.mathematics.load_equation_index", lambda: idx)

        class HasDocstringExampleOnly(_FakeProcess):
            """サンプル docstring.

            Example:
                >>> # このような擬似コードは C15 拡張の対象外
                >>> IdentityContract(
                ...     name="example_only",
                ...     equation_ref="99_nowhere.md#eq-ghost",  # 未解決
                ... )
            """

            # contracts は空のまま
            contracts: tuple = ()

        errors = check_mod.check_c15_equation_refs(
            {"HasDocstringExampleOnly": HasDocstringExampleOnly}
        )
        # docstring 内の "99_nowhere.md#eq-ghost" は検査対象外
        assert errors == []
