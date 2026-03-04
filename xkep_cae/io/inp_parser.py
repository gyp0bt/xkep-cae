"""Abaqus .inp パーサー — OOP フレームワーク.

AbstractKeywordParser + __init_subclass__ による自動登録パターンで、
キーワード単位のパーサーをスケーラブルに追加できる設計。

設計原則:
  - 各キーワードは独立した Parser サブクラスで実装
  - ParseContext がブロックコンテキスト（Step, Material 等）を管理
  - *INCLUDE によるファイル再帰読み込みをサポート
  - 行単位でパースし、キーワードディスパッチで処理を委譲

使い方::

    from xkep_cae.io.inp_parser import InpReader
    mesh = InpReader().read("model.inp")

[← README](../../README.md)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from xkep_cae.io.abaqus_inp import (
    AbaqusBeamSection,
    AbaqusBoundary,
    AbaqusElementGroup,
    AbaqusFieldAnimation,
    AbaqusMaterial,
    AbaqusMesh,
    AbaqusNode,
    InpAnimationRequest,
    InpContactDef,
    InpInitialCondition,
    InpOutputRequest,
    InpStep,
    InpSurfaceDef,
    InpSurfaceInteraction,
)

logger = logging.getLogger(__name__)


# ====================================================================
# パーサーコンテキスト
# ====================================================================


@dataclass
class ParseContext:
    """パーサーに渡されるブロックコンテキスト.

    Attributes:
        mesh: パース結果を蓄積する AbaqusMesh
        current_step: 現在パース中のステップ（*STEP ～ *END STEP 内）
        current_material: 現在パース中の材料（*MATERIAL の後）
        current_surface_interaction: 現在の *SURFACE INTERACTION
        current_contact: 現在のコンタクト定義
        base_dir: *INCLUDE 解決用のベースディレクトリ
        include_depth: *INCLUDE のネスト深度（無限再帰防止）
    """

    mesh: AbaqusMesh
    current_step: InpStep | None = None
    current_material: AbaqusMaterial | None = None
    current_surface_interaction: InpSurfaceInteraction | None = None
    current_contact: InpContactDef | None = None
    base_dir: Path = field(default_factory=lambda: Path("."))
    include_depth: int = 0

    MAX_INCLUDE_DEPTH: int = 10


# ====================================================================
# AbstractKeywordParser — 基底クラス
# ====================================================================


class AbstractKeywordParser:
    """キーワードパーサーの基底クラス.

    サブクラスは ``keyword`` クラス変数を設定するだけで自動登録される。
    複数のキーワードに対応する場合は ``keyword_aliases`` も設定する。

    Example::

        class NodeParser(AbstractKeywordParser):
            keyword = "*NODE"

            def parse(self, lines, start_idx, opts, ctx):
                ...
                return next_idx
    """

    _registry: dict[str, type[AbstractKeywordParser]] = {}

    keyword: str = ""
    keyword_aliases: list[str] = []

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        kw = cls.keyword
        if kw:
            AbstractKeywordParser._registry[kw] = cls
            for alias in cls.keyword_aliases:
                AbstractKeywordParser._registry[alias] = cls

    def parse(
        self,
        lines: list[str],
        start_idx: int,
        opts: dict[str, str],
        ctx: ParseContext,
    ) -> int:
        """キーワードのデータ行をパースする.

        Args:
            lines: ファイル全行（INCLUDE 展開済み）
            start_idx: キーワード行の**次**の行のインデックス
            opts: キーワード行のオプション辞書（大文字正規化済み）
            ctx: パースコンテキスト

        Returns:
            次のキーワード行のインデックス
        """
        raise NotImplementedError

    @classmethod
    def get_parser(cls, keyword: str) -> AbstractKeywordParser | None:
        """キーワードに対応するパーサーインスタンスを返す."""
        parser_cls = cls._registry.get(keyword)
        if parser_cls is not None:
            return parser_cls()
        return None


# ====================================================================
# ユーティリティ
# ====================================================================


def parse_keyword_options(line: str) -> dict[str, str]:
    """キーワード行のオプションを辞書にパースする.

    例: ``*ELEMENT, TYPE=CPS4R, ELSET=solid`` →
        ``{"TYPE": "CPS4R", "ELSET": "solid"}``
    """
    parts = line.split(",")
    options: dict[str, str] = {}
    for part in parts[1:]:
        part = part.strip()
        if "=" in part:
            key, val = part.split("=", 1)
            options[key.strip().upper()] = val.strip()
        elif part:
            options[part.upper()] = ""
    return options


def _skip_to_next_keyword(lines: list[str], idx: int) -> int:
    """データ行をスキップし、次のキーワード行のインデックスを返す."""
    n = len(lines)
    while idx < n:
        line = lines[idx].strip()
        if line.startswith("*") and not line.startswith("**"):
            return idx
        idx += 1
    return idx


def _read_data_lines(lines: list[str], start_idx: int) -> tuple[list[str], int]:
    """キーワードに続くデータ行を全て読み取る.

    Returns:
        (data_lines, next_keyword_idx)
    """
    data: list[str] = []
    idx = start_idx
    n = len(lines)
    while idx < n:
        line = lines[idx].strip()
        if not line or line.startswith("**"):
            idx += 1
            continue
        if line.startswith("*"):
            break
        data.append(line)
        idx += 1
    return data, idx


# ====================================================================
# InpReader — メインリーダー
# ====================================================================


class InpReader:
    """Abaqus .inp ファイルリーダー.

    *INCLUDE を再帰的に展開し、AbstractKeywordParser で登録された
    パーサー群にキーワードをディスパッチする。
    """

    def read(self, filepath: str | Path) -> AbaqusMesh:
        """ファイルを読み込み AbaqusMesh を返す."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")

        mesh = AbaqusMesh()
        ctx = ParseContext(mesh=mesh, base_dir=filepath.parent)

        lines = self._read_lines_with_include(filepath, depth=0)
        self._parse_lines(lines, ctx)
        return mesh

    def _read_lines_with_include(self, filepath: Path, depth: int) -> list[str]:
        """*INCLUDE を再帰展開して全行を返す."""
        if depth > ParseContext.MAX_INCLUDE_DEPTH:
            logger.warning(
                "*INCLUDE のネスト深度が上限(%d)を超えました: %s",
                ParseContext.MAX_INCLUDE_DEPTH,
                filepath,
            )
            return []

        with filepath.open("r", encoding="utf-8", errors="replace") as f:
            raw_lines = f.readlines()

        result: list[str] = []
        for raw_line in raw_lines:
            stripped = raw_line.strip()
            keyword_upper = stripped.split(",")[0].strip().upper()
            if keyword_upper == "*INCLUDE":
                opts = parse_keyword_options(stripped)
                include_input = opts.get("INPUT", "")
                if include_input:
                    inc_path = filepath.parent / include_input
                    if inc_path.exists():
                        included = self._read_lines_with_include(inc_path, depth + 1)
                        result.extend(included)
                    else:
                        logger.warning("*INCLUDE ファイルが見つかりません: %s", inc_path)
            else:
                result.append(raw_line)

        return result

    def _parse_lines(self, lines: list[str], ctx: ParseContext) -> None:
        """行リストをパースしてコンテキストに蓄積する."""
        idx = 0
        n_lines = len(lines)

        while idx < n_lines:
            line = lines[idx].strip()

            # 空行スキップ
            if not line:
                idx += 1
                continue

            # xkep-cae 独自拡張コメント
            if line.startswith("** XKEP-CAE:"):
                _handle_xkep_comment(line, ctx)
                idx += 1
                continue

            # コメント行スキップ
            if line.startswith("**"):
                idx += 1
                continue

            # キーワード行
            if line.startswith("*"):
                keyword_raw = line.split(",")[0].strip().upper()
                # スペース除去版（*BEAM SECTION → *BEAMSECTION）
                keyword_nospace = keyword_raw.replace(" ", "")

                opts = parse_keyword_options(line)

                # パーサー検索: 正規形 → スペース除去形
                parser = AbstractKeywordParser.get_parser(keyword_raw)
                if parser is None:
                    parser = AbstractKeywordParser.get_parser(keyword_nospace)

                if parser is not None:
                    idx = parser.parse(lines, idx + 1, opts, ctx)
                else:
                    idx += 1
            else:
                idx += 1


# ====================================================================
# xkep-cae 独自コメント処理
# ====================================================================


def _handle_xkep_comment(line: str, ctx: ParseContext) -> None:
    """** XKEP-CAE: コメント行の拡張属性をパースする."""
    content = line[len("** XKEP-CAE:") :].strip()
    opts: dict[str, str] = {}
    for part in content.split(","):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            opts[k.strip().upper()] = v.strip()

    if "EXCLUDE_SAME_LAYER" in opts:
        val = opts["EXCLUDE_SAME_LAYER"].upper() == "YES"
        if ctx.current_step is not None and ctx.current_step.contact is not None:
            ctx.current_step.contact.exclude_same_layer = val
        elif ctx.current_contact is not None:
            ctx.current_contact.exclude_same_layer = val
        elif ctx.mesh.contact_defs:
            ctx.mesh.contact_defs[-1].exclude_same_layer = val

    if "ALGORITHM" in opts or "MORTAR" in opts:
        si = ctx.current_surface_interaction
        if si is not None:
            if "ALGORITHM" in opts:
                si.algorithm = opts["ALGORITHM"]
            if "MORTAR" in opts:
                si.mortar = opts["MORTAR"].upper() == "YES"


# ====================================================================
# キーワードパーサー実装
# ====================================================================


# --- ヘッダ ---


class HeadingParser(AbstractKeywordParser):
    keyword = "*HEADING"

    def parse(self, lines, start_idx, opts, ctx):
        idx = start_idx
        n = len(lines)
        heading_parts: list[str] = []

        while idx < n:
            line = lines[idx].strip()
            if line.startswith("*") and not line.startswith("**"):
                break
            if line.startswith("**"):
                heading_parts.append(line[2:].strip())
            elif line:
                heading_parts.append(line)
            idx += 1

        ctx.mesh.heading = "\n".join(heading_parts)
        return idx


# --- 節点 ---


class NodeParser(AbstractKeywordParser):
    keyword = "*NODE"

    def parse(self, lines, start_idx, opts, ctx):
        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 3:
                label = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3]) if len(parts) >= 4 else 0.0
                ctx.mesh.nodes.append(AbaqusNode(label=label, x=x, y=y, z=z))
            idx += 1

        return idx


# --- 要素 ---


class ElementParser(AbstractKeywordParser):
    keyword = "*ELEMENT"

    def parse(self, lines, start_idx, opts, ctx):
        elem_type = opts.get("TYPE", "UNKNOWN")
        elset = opts.get("ELSET")
        group = AbaqusElementGroup(elem_type=elem_type, elset=elset)

        idx = start_idx
        n = len(lines)
        pending_label: int | None = None
        pending_nodes: list[int] = []

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",")]
            # 末尾カンマ（継続行）対応
            while parts and parts[-1] == "":
                parts.pop()

            if pending_label is not None:
                # 前行の継続
                pending_nodes.extend(int(p) for p in parts if p)
                if not line.endswith(","):
                    group.elements.append((pending_label, pending_nodes))
                    pending_label = None
                    pending_nodes = []
            else:
                if len(parts) >= 2:
                    label = int(parts[0])
                    node_labels = [int(p) for p in parts[1:] if p]
                    if line.endswith(","):
                        pending_label = label
                        pending_nodes = node_labels
                    else:
                        group.elements.append((label, node_labels))
            idx += 1

        if pending_label is not None:
            group.elements.append((pending_label, pending_nodes))

        ctx.mesh.element_groups.append(group)

        # ELSET= で暗黙的に要素セットも定義
        if elset:
            labels = [label for label, _ in group.elements]
            ctx.mesh.elsets[elset] = labels

        return idx


# --- ノードセット ---


class NsetParser(AbstractKeywordParser):
    keyword = "*NSET"

    def parse(self, lines, start_idx, opts, ctx):
        name = opts.get("NSET", "UNNAMED")
        is_generate = "GENERATE" in opts

        idx = start_idx
        n = len(lines)
        labels: list[int] = ctx.mesh.nsets.get(name, [])

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",") if p.strip()]

            if is_generate and len(parts) >= 2:
                start = int(parts[0])
                stop = int(parts[1])
                step = int(parts[2]) if len(parts) >= 3 else 1
                labels.extend(range(start, stop + 1, step))
            else:
                labels.extend(int(p) for p in parts)
            idx += 1

        ctx.mesh.nsets[name] = labels
        return idx


# --- 要素セット ---


class ElsetParser(AbstractKeywordParser):
    keyword = "*ELSET"

    def parse(self, lines, start_idx, opts, ctx):
        name = opts.get("ELSET", "UNNAMED")
        is_generate = "GENERATE" in opts

        idx = start_idx
        n = len(lines)
        labels: list[int] = ctx.mesh.elsets.get(name, [])

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",") if p.strip()]

            if is_generate and len(parts) >= 2:
                start = int(parts[0])
                stop = int(parts[1])
                step = int(parts[2]) if len(parts) >= 3 else 1
                labels.extend(range(start, stop + 1, step))
            else:
                labels.extend(int(p) for p in parts)
            idx += 1

        ctx.mesh.elsets[name] = labels
        return idx


# --- 梁断面 ---


class BeamSectionParser(AbstractKeywordParser):
    keyword = "*BEAM SECTION"
    keyword_aliases = ["*BEAMSECTION"]

    def parse(self, lines, start_idx, opts, ctx):
        section_type = opts.get("SECTION", "RECT")
        elset = opts.get("ELSET", "")
        material = opts.get("MATERIAL", "")

        bsec = AbaqusBeamSection(
            section_type=section_type,
            elset=elset,
            material=material,
        )

        idx = start_idx
        n = len(lines)
        data_line_count = 0

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",") if p.strip()]
            if data_line_count == 0:
                bsec.dimensions = [float(p) for p in parts]
            elif data_line_count == 1:
                bsec.direction = [float(p) for p in parts]

            data_line_count += 1
            idx += 1

        ctx.mesh.beam_sections.append(bsec)
        return idx


# --- 横せん断剛性 ---


class TransverseShearParser(AbstractKeywordParser):
    keyword = "*TRANSVERSE SHEAR STIFFNESS"
    keyword_aliases = ["*TRANSVERSESHEARSTIFFNESS"]

    def parse(self, lines, start_idx, opts, ctx):
        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 2 and ctx.mesh.beam_sections:
                k11 = float(parts[0])
                k22 = float(parts[1])
                k12 = float(parts[2]) if len(parts) >= 3 else 0.0
                ctx.mesh.beam_sections[-1].transverse_shear = (k11, k22, k12)

            idx += 1
            break

        return idx


# --- 境界条件（モデルレベル）---


class BoundaryParser(AbstractKeywordParser):
    keyword = "*BOUNDARY"

    def parse(self, lines, start_idx, opts, ctx):
        # ステップ内ならステップ境界条件
        if ctx.current_step is not None:
            return self._parse_step_boundary(lines, start_idx, ctx.current_step)
        return self._parse_model_boundary(lines, start_idx, ctx.mesh)

    def _parse_model_boundary(self, lines, start_idx, mesh):
        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",")]
            parts = [p for p in parts if p]

            if not parts:
                idx += 1
                continue

            try:
                node_label = int(parts[0])
            except ValueError:
                # NSET 名の場合 — ノードセットから展開
                nset_name = parts[0]
                first_dof = int(parts[1]) if len(parts) > 1 else 1
                last_dof = int(parts[2]) if len(parts) > 2 else first_dof
                value = float(parts[3]) if len(parts) > 3 else 0.0
                nset_labels = mesh.nsets.get(nset_name, [])
                for nl in nset_labels:
                    mesh.boundaries.append(
                        AbaqusBoundary(
                            node_label=nl,
                            first_dof=first_dof,
                            last_dof=last_dof,
                            value=value,
                        )
                    )
                idx += 1
                continue

            if len(parts) == 2:
                dof = int(parts[1])
                mesh.boundaries.append(
                    AbaqusBoundary(
                        node_label=node_label,
                        first_dof=dof,
                        last_dof=dof,
                        value=0.0,
                    )
                )
            elif len(parts) == 3:
                first_dof = int(parts[1])
                last_dof = int(parts[2]) if parts[2] else first_dof
                mesh.boundaries.append(
                    AbaqusBoundary(
                        node_label=node_label,
                        first_dof=first_dof,
                        last_dof=last_dof,
                        value=0.0,
                    )
                )
            elif len(parts) >= 4:
                first_dof = int(parts[1])
                last_dof = int(parts[2])
                value = float(parts[3])
                mesh.boundaries.append(
                    AbaqusBoundary(
                        node_label=node_label,
                        first_dof=first_dof,
                        last_dof=last_dof,
                        value=value,
                    )
                )
            idx += 1

        return idx

    def _parse_step_boundary(self, lines, start_idx, step):
        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",")]
            parts = [p for p in parts if p]

            if len(parts) >= 4:
                node_or_nset: int | str
                try:
                    node_or_nset = int(parts[0])
                except ValueError:
                    node_or_nset = parts[0]
                first_dof = int(parts[1])
                last_dof = int(parts[2])
                value = float(parts[3])
                step.boundaries.append((node_or_nset, first_dof, last_dof, value))
            elif len(parts) >= 3:
                try:
                    node_or_nset = int(parts[0])
                except ValueError:
                    node_or_nset = parts[0]
                first_dof = int(parts[1])
                last_dof = int(parts[2]) if parts[2] else first_dof
                step.boundaries.append((node_or_nset, first_dof, last_dof, 0.0))
            elif len(parts) >= 2:
                try:
                    node_or_nset = int(parts[0])
                except ValueError:
                    node_or_nset = parts[0]
                first_dof = int(parts[1])
                step.boundaries.append((node_or_nset, first_dof, first_dof, 0.0))
            idx += 1

        return idx


# --- 材料 ---


class MaterialParser(AbstractKeywordParser):
    keyword = "*MATERIAL"

    def parse(self, lines, start_idx, opts, ctx):
        name = opts.get("NAME", "UNNAMED")
        mat = AbaqusMaterial(name=name)
        ctx.mesh.materials.append(mat)
        ctx.current_material = mat
        return start_idx


class ElasticParser(AbstractKeywordParser):
    keyword = "*ELASTIC"

    def parse(self, lines, start_idx, opts, ctx):
        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",") if p.strip()]
            E = float(parts[0])
            nu = float(parts[1]) if len(parts) > 1 else 0.0

            mat = ctx.current_material
            if mat is None and ctx.mesh.materials:
                mat = ctx.mesh.materials[-1]
            if mat is not None:
                mat.elastic = (E, nu)

            idx += 1
            break

        return idx


class DensityParser(AbstractKeywordParser):
    keyword = "*DENSITY"

    def parse(self, lines, start_idx, opts, ctx):
        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",") if p.strip()]
            rho = float(parts[0])

            mat = ctx.current_material
            if mat is None and ctx.mesh.materials:
                mat = ctx.mesh.materials[-1]
            if mat is not None:
                mat.density = rho

            idx += 1
            break

        return idx


class PlasticParser(AbstractKeywordParser):
    keyword = "*PLASTIC"

    def parse(self, lines, start_idx, opts, ctx):
        hardening = opts.get("HARDENING", "ISOTROPIC")
        table: list[tuple[float, float]] = []

        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",") if p.strip()]
            sigma_y = float(parts[0])
            eps_p = float(parts[1]) if len(parts) > 1 else 0.0
            table.append((sigma_y, eps_p))
            idx += 1

        mat = ctx.current_material
        if mat is None and ctx.mesh.materials:
            mat = ctx.mesh.materials[-1]
        if mat is not None:
            mat.plastic = table
            mat.plastic_hardening = hardening

        return idx


# --- サーフェス ---


class SurfaceParser(AbstractKeywordParser):
    keyword = "*SURFACE"

    def parse(self, lines, start_idx, opts, ctx):
        surf = InpSurfaceDef(
            name=opts.get("NAME", ""),
            type=opts.get("TYPE", "ELEMENT"),
        )

        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",") if p.strip()]
            if parts:
                stype = surf.type.upper()
                if stype == "ELEMENT":
                    surf.elset = parts[0]
                elif stype == "NODE":
                    surf.nset = parts[0]
            idx += 1
            break

        ctx.mesh.surfaces.append(surf)
        return idx


# --- サーフェスインタラクション ---


class SurfaceInteractionParser(AbstractKeywordParser):
    keyword = "*SURFACE INTERACTION"
    keyword_aliases = ["*SURFACEINTERACTION"]

    def parse(self, lines, start_idx, opts, ctx):
        si = InpSurfaceInteraction(name=opts.get("NAME", "CONTACT_PROP"))
        ctx.mesh.surface_interactions.append(si)
        ctx.current_surface_interaction = si
        return start_idx


class SurfaceBehaviorParser(AbstractKeywordParser):
    keyword = "*SURFACE BEHAVIOR"
    keyword_aliases = ["*SURFACEBEHAVIOR"]

    def parse(self, lines, start_idx, opts, ctx):
        po = opts.get("PRESSURE-OVERCLOSURE", "HARD")
        si = ctx.current_surface_interaction
        if si is None and ctx.mesh.surface_interactions:
            si = ctx.mesh.surface_interactions[-1]

        if si is not None:
            si.pressure_overclosure = po

        if po.upper() == "HARD":
            return start_idx

        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",") if p.strip()]
            if parts and po.upper() == "LINEAR" and si is not None:
                si.k_pen = float(parts[0])
            idx += 1
            break

        return idx


class FrictionParser(AbstractKeywordParser):
    keyword = "*FRICTION"

    def parse(self, lines, start_idx, opts, ctx):
        idx = start_idx
        n = len(lines)

        si = ctx.current_surface_interaction
        if si is None and ctx.mesh.surface_interactions:
            si = ctx.mesh.surface_interactions[-1]

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",") if p.strip()]
            if parts and si is not None:
                si.friction = float(parts[0])
            idx += 1
            break

        return idx


# --- コンタクト ---


class ContactParser(AbstractKeywordParser):
    keyword = "*CONTACT"

    def parse(self, lines, start_idx, opts, ctx):
        cdef = InpContactDef()
        cdef.surfaces = list(ctx.mesh.surfaces)
        cdef.surface_interactions = list(ctx.mesh.surface_interactions)

        if ctx.current_step is not None:
            ctx.current_step.contact = cdef
        else:
            ctx.mesh.contact_defs.append(cdef)

        ctx.current_contact = cdef
        return start_idx


class ContactInclusionsParser(AbstractKeywordParser):
    keyword = "*CONTACT INCLUSIONS"
    keyword_aliases = ["*CONTACTINCLUSIONS"]

    def parse(self, lines, start_idx, opts, ctx):
        idx = start_idx
        n = len(lines)
        inclusions: list[tuple[str, str]] = []

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                inclusions.append((parts[0], parts[1]))
            idx += 1

        cdef = ctx.current_contact
        if cdef is None:
            if ctx.current_step is not None and ctx.current_step.contact is not None:
                cdef = ctx.current_step.contact
            elif ctx.mesh.contact_defs:
                cdef = ctx.mesh.contact_defs[-1]

        if cdef is not None:
            cdef.inclusions = inclusions

        return idx


class ContactPropertyAssignmentParser(AbstractKeywordParser):
    keyword = "*CONTACT PROPERTY ASSIGNMENT"
    keyword_aliases = ["*CONTACTPROPERTYASSIGNMENT"]

    def parse(self, lines, start_idx, opts, ctx):
        idx = start_idx
        n = len(lines)
        interaction_name = ""

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                interaction_name = parts[2]
            elif len(parts) >= 1 and parts[0]:
                interaction_name = parts[0]
            idx += 1

        cdef = ctx.current_contact
        if cdef is None:
            if ctx.current_step is not None and ctx.current_step.contact is not None:
                cdef = ctx.current_step.contact
            elif ctx.mesh.contact_defs:
                cdef = ctx.mesh.contact_defs[-1]

        if cdef is not None and interaction_name:
            cdef.interaction = interaction_name

        return idx


# --- 初期条件 ---


class InitialConditionsParser(AbstractKeywordParser):
    keyword = "*INITIAL CONDITIONS"
    keyword_aliases = ["*INITIALCONDITIONS"]

    def parse(self, lines, start_idx, opts, ctx):
        ic_type = opts.get("TYPE", "VELOCITY")
        ic = InpInitialCondition(type=ic_type)

        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                node_or_nset: str | int
                try:
                    node_or_nset = int(parts[0])
                except ValueError:
                    node_or_nset = parts[0]
                dof = int(parts[1])
                value = float(parts[2])
                ic.data.append((node_or_nset, dof, value))
            idx += 1

        ctx.mesh.initial_conditions.append(ic)
        return idx


# --- ステップ ---


class StepParser(AbstractKeywordParser):
    keyword = "*STEP"

    def parse(self, lines, start_idx, opts, ctx):
        step = InpStep()
        if "INC" in opts:
            step.inc = int(opts["INC"])
        nlgeom = opts.get("NLGEOM", "").upper()
        step.nlgeom = nlgeom in ("YES", "")
        if "UNSYMM" in opts:
            step.unsymm = opts["UNSYMM"].upper() == "YES"

        # ステップ名: *STEP の直後のデータ行（キーワードでない行）
        idx = start_idx
        n = len(lines)
        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break
            # ステップ名行
            step.name = line
            idx += 1
            break

        ctx.current_step = step
        return idx


class EndStepParser(AbstractKeywordParser):
    keyword = "*END STEP"
    keyword_aliases = ["*ENDSTEP"]

    def parse(self, lines, start_idx, opts, ctx):
        if ctx.current_step is not None:
            ctx.mesh.steps.append(ctx.current_step)
            ctx.current_step = None
            ctx.current_contact = None
        return start_idx


class StaticParser(AbstractKeywordParser):
    keyword = "*STATIC"

    def parse(self, lines, start_idx, opts, ctx):
        if ctx.current_step is None:
            return start_idx

        ctx.current_step.procedure = "STATIC"
        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",") if p.strip()]
            if parts:
                ctx.current_step.time_params = tuple(float(p) for p in parts)
            idx += 1
            break

        return idx


class DynamicParser(AbstractKeywordParser):
    keyword = "*DYNAMIC"

    def parse(self, lines, start_idx, opts, ctx):
        if ctx.current_step is None:
            return start_idx

        ctx.current_step.procedure = "DYNAMIC"
        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",") if p.strip()]
            if parts:
                ctx.current_step.time_params = tuple(float(p) for p in parts)
            idx += 1
            break

        return idx


# --- 荷重 ---


class CloadParser(AbstractKeywordParser):
    keyword = "*CLOAD"

    def parse(self, lines, start_idx, opts, ctx):
        if ctx.current_step is None:
            return _skip_to_next_keyword(lines, start_idx)

        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                node_or_nset: int | str
                try:
                    node_or_nset = int(parts[0])
                except ValueError:
                    node_or_nset = parts[0]
                dof = int(parts[1])
                magnitude = float(parts[2])
                ctx.current_step.cloads.append((node_or_nset, dof, magnitude))
            idx += 1

        return idx


class DloadParser(AbstractKeywordParser):
    keyword = "*DLOAD"

    def parse(self, lines, start_idx, opts, ctx):
        if ctx.current_step is None:
            return _skip_to_next_keyword(lines, start_idx)

        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                elset = parts[0]
                load_type = parts[1]
                magnitude = float(parts[2])
                ctx.current_step.dloads.append((elset, load_type, magnitude))
            idx += 1

        return idx


# --- 出力 ---


class OutputParser(AbstractKeywordParser):
    keyword = "*OUTPUT"

    def parse(self, lines, start_idx, opts, ctx):
        # FIELD ANIMATION → 独自拡張（後方互換）
        if "FIELD ANIMATION" in opts or ("FIELD" in opts and "ANIMATION" in str(opts)):
            return self._parse_field_animation(lines, start_idx, opts, ctx)

        if ctx.current_step is None:
            return start_idx

        domain = "FIELD"
        if "HISTORY" in opts:
            domain = "HISTORY"
        elif "FIELD" in opts:
            domain = "FIELD"

        freq = 1
        if "FREQUENCY" in opts:
            try:
                freq = int(opts["FREQUENCY"])
            except ValueError:
                pass

        req = InpOutputRequest(domain=domain, frequency=freq)
        ctx.current_step.output_requests.append(req)
        return start_idx

    def _parse_field_animation(self, lines, start_idx, opts, ctx):
        output_dir = opts.get("DIR", "animation")
        anim = AbaqusFieldAnimation(output_dir=output_dir)

        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip().lower() for p in line.split(",") if p.strip()]
            if parts:
                anim.views = parts
            idx += 1
            break

        ctx.mesh.field_animation = anim
        return idx


class NodeOutputParser(AbstractKeywordParser):
    keyword = "*NODE OUTPUT"
    keyword_aliases = ["*NODEOUTPUT"]

    def parse(self, lines, start_idx, opts, ctx):
        return _parse_variable_output(lines, start_idx, opts, ctx)


class ElementOutputParser(AbstractKeywordParser):
    keyword = "*ELEMENT OUTPUT"
    keyword_aliases = ["*ELEMENTOUTPUT"]

    def parse(self, lines, start_idx, opts, ctx):
        return _parse_variable_output(lines, start_idx, opts, ctx)


class EnergyOutputParser(AbstractKeywordParser):
    keyword = "*ENERGY OUTPUT"
    keyword_aliases = ["*ENERGYOUTPUT"]

    def parse(self, lines, start_idx, opts, ctx):
        return _parse_variable_output(lines, start_idx, opts, ctx)


def _parse_variable_output(
    lines: list[str],
    start_idx: int,
    opts: dict[str, str],
    ctx: ParseContext,
) -> int:
    """*NODE OUTPUT / *ELEMENT OUTPUT / *ENERGY OUTPUT の共通パーサー."""
    if ctx.current_step is None:
        return _skip_to_next_keyword(lines, start_idx)

    idx = start_idx
    n = len(lines)
    variables: list[str] = []

    nset = opts.get("NSET")
    elset = opts.get("ELSET")

    while idx < n:
        line = lines[idx].strip()
        if not line or line.startswith("**"):
            idx += 1
            continue
        if line.startswith("*"):
            break

        parts = [p.strip().upper() for p in line.split(",") if p.strip()]
        variables.extend(parts)
        idx += 1

    if variables:
        matching = [r for r in ctx.current_step.output_requests if r.domain == "FIELD"]
        if matching:
            matching[-1].variables = variables
            if nset:
                matching[-1].nset = nset
            if elset:
                matching[-1].elset = elset
        else:
            req = InpOutputRequest(
                domain="FIELD",
                variables=variables,
                nset=nset,
                elset=elset,
            )
            ctx.current_step.output_requests.append(req)

    return idx


class AnimationParser(AbstractKeywordParser):
    keyword = "*ANIMATION"

    def parse(self, lines, start_idx, opts, ctx):
        if ctx.current_step is None:
            return start_idx

        output_dir = opts.get("DIR", "animation")
        anim = InpAnimationRequest(output_dir=output_dir)

        freq_str = opts.get("FREQUENCY", "1")
        try:
            anim.frequency = int(freq_str)
        except ValueError:
            pass

        idx = start_idx
        n = len(lines)

        while idx < n:
            line = lines[idx].strip()
            if not line or line.startswith("**"):
                idx += 1
                continue
            if line.startswith("*"):
                break

            parts = [p.strip().lower() for p in line.split(",") if p.strip()]
            if parts:
                anim.views = parts
            idx += 1
            break

        ctx.current_step.animation = anim
        return idx


# ====================================================================
# 公開API
# ====================================================================


def read_abaqus_inp_v2(filepath: str | Path) -> AbaqusMesh:
    """新パーサーで .inp を読み込む（InpReader ショートカット）."""
    return InpReader().read(filepath)
