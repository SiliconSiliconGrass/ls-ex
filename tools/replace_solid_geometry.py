"""把已有 .k 文件里的 solid 几何换成空心长方体（长方体壳）。

只替换 ``*NODE`` 和 ``*ELEMENT_SOLID`` 两个块，文件里其它关键字逐行原样保留，
所以 *PART / *SECTION_SOLID / *MAT_* / ALE / FSI / 起爆点这些设置都不受影响
（节点号、单元号是重新从 1 编的，前提是其它关键字没有引用它们）。

用法::

    python tools/replace_solid_geometry.py --source main.k --output hollow.k \
        --inner 40,40,40 --thickness 8 --element-size 2

摆放位置默认自动对齐原几何：x/y 方向居中对齐原包围盒，z 方向底面与原几何底面齐平，
也可以用 ``--origin x,y,z`` 直接指定外箱体最小角点。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ls_ex import (
    KCard,
    KWord,
    as_vec3,
    hollow_box_mesh,
)
from ls_ex.KeywordGenerator import ELEMENT_WIDTHS, NODE_WIDTHS


def block_name(line: str) -> str:
    """``*ELEMENT_SOLID (ten nodes format)`` -> ``ELEMENT_SOLID``。"""
    return line[1:].split()[0].split("(")[0].strip().upper()


def keyword_ranges(lines: list[str]) -> list[tuple[int, str]]:
    return [(index, block_name(line)) for index, line in enumerate(lines)
            if line.startswith("*")]


def find_block(lines: list[str], name: str) -> tuple[int, int]:
    """返回 ``[start, end)``：从 ``*NAME`` 那行到下一个关键字之前。"""
    starts = keyword_ranges(lines)
    for position, (index, current) in enumerate(starts):
        if current == name:
            end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
            return index, end
    raise KeyError(f"文件里找不到 *{name}")


def leading_comments(lines: list[str], start: int, end: int) -> list[str]:
    """关键字块开头连续的 ``$`` 注释行（例如 ``$#   nid ...``）。"""
    comments = []
    for line in lines[start + 1:end]:
        if not line.lstrip().startswith("$"):
            break
        comments.append(line)
    return comments


def source_bounds(lines: list[str], start: int, end: int) -> tuple[tuple[float, ...],
                                                                  tuple[float, ...]]:
    """原 *NODE 块里坐标的最小值 / 最大值。"""
    low = [float("inf")] * 3
    high = [float("-inf")] * 3
    for line in lines[start + 1:end]:
        fields = line.split()
        if len(fields) < 4 or fields[0].startswith("$"):
            continue
        for axis in range(3):
            value = float(fields[1 + axis])
            low[axis] = min(low[axis], value)
            high[axis] = max(high[axis], value)
    return tuple(low), tuple(high)


def node_block_lines(mesh, comment_lines: Sequence[str]) -> list[str]:
    """新的 *NODE 块；注释行沿用原文件（没有就用标准的字段名提示行）。"""
    cards = [KCard.parse(line) for line in comment_lines]
    if not cards:
        cards = [KCard.header(["nid", "x", "y", "z", "tc", "rc"], NODE_WIDTHS)]
    for nid in sorted(mesh.nodes):
        x, y, z = mesh.nodes[nid]
        cards.append(KCard(values=[nid, x, y, z, 0, 0], widths=NODE_WIDTHS,
                           kinds=["i", "f", "f", "f", "i", "i"]))
    return ["*NODE", *(card.render() for card in cards)]


def element_block_lines(mesh) -> list[str]:
    block = KWord("ELEMENT_SOLID", [
        KCard.header(["eid", "pid", "n1", "n2", "n3", "n4",
                      "n5", "n6", "n7", "n8"], ELEMENT_WIDTHS),
    ])
    for eid, pid, corners in mesh.solids:
        block.add(KCard(values=[eid, pid, *corners], widths=ELEMENT_WIDTHS, kinds=["i"] * 10))
    return ["*ELEMENT_SOLID", *(card.render() for card in block.cards)]


def replace_geometry(source: Path, output: Path, inner: object, thickness: float,
                     element_size: float, origin: object | None,
                     pid: int) -> None:
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()

    node_start, node_end = find_block(lines, "NODE")
    elem_start, elem_end = find_block(lines, "ELEMENT_SOLID")
    if node_end > elem_start:
        raise ValueError("*NODE 与 *ELEMENT_SOLID 的顺序不是预期的（先 NODE 后 ELEMENT_SOLID）")

    inner_size = as_vec3(inner, "inner")
    outer = tuple(side + 2 * thickness for side in inner_size)
    if origin is None:
        low, high = source_bounds(lines, node_start, node_end)
        origin = (0.5 * (low[0] + high[0]) - outer[0] / 2,
                  0.5 * (low[1] + high[1]) - outer[1] / 2,
                  low[2])
    mesh = hollow_box_mesh(inner_size, thickness, element_size=element_size,
                           origin=origin, pid=pid)

    new_lines = [
        *lines[:node_start],
        *node_block_lines(mesh, leading_comments(lines, node_start, node_end)),
        *lines[node_end:elem_start],
        *element_block_lines(mesh),
        *lines[elem_end:],
    ]

    output.write_text("\n".join(new_lines) + "\n", encoding="utf-8", newline="")
    print(f"wrote {output}")
    print(f"  inner={inner_size} thickness={thickness} element={element_size} "
          f"origin={tuple(round(v, 6) for v in origin)} outer={outer}")
    print(f"  replaced lines {node_start + 1}..{node_end} (*NODE, "
          f"{node_end - node_start - 1} lines) and {elem_start + 1}..{elem_end} "
          f"(*ELEMENT_SOLID, {elem_end - elem_start - 1} lines)")
    print(f"  kept lines: {node_end - node_start - 1 + elem_end - elem_start - 1} replaced, "
          f"{len(lines) - (node_end - node_start) - (elem_end - elem_start)} copied verbatim")
    print(f"  {mesh.summary()}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="replace_solid_geometry",
        description="把 .k 文件里的 solid 几何替换成空心长方体（只动 *NODE 和 *ELEMENT_SOLID）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="原始 .k 文件（不会被修改）")
    parser.add_argument("--output", required=True, help="输出 .k 文件")
    parser.add_argument("--inner", default="40,40,40", help="内腔尺寸 cx,cy,cz")
    parser.add_argument("-t", "--thickness", type=float, default=8.0, help="壳厚")
    parser.add_argument("-e", "--element-size", type=float, default=2.0, help="网格边长")
    parser.add_argument("--origin", default=None, help="外箱体最小角点 x,y,z（缺省对齐原几何）")
    parser.add_argument("--pid", type=int, default=2, help="新单元所属的 part id")
    args = parser.parse_args(argv)

    replace_geometry(
        source=Path(args.source),
        output=Path(args.output),
        inner=args.inner,
        thickness=args.thickness,
        element_size=args.element_size,
        origin=as_vec3(args.origin, "origin") if args.origin else None,
        pid=args.pid,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
