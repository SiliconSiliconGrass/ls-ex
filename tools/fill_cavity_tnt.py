"""把 S-ALE 的炸药填充从"结构本身的体积"改成"结构内腔的长方体区域"。

背景：``*ALE_STRUCTURED_MESH_VOLUME_FILLING`` 用 ``GEOM=PART`` 时，填的是那个
part 的**单元体积**。原来的 deck 里写的是 ``PART 2``（钢壳本身），于是炸药被填进
壁厚里、腔内一个都没有，起爆点落在空腔里 → 不会起爆（也不报错）。

这个脚本把 TNT 的填充块换成显式的长方体区域：

    python tools/fill_cavity_tnt.py --source sim/main_hollow_cube.k \
        --output main_hollow_cube_tnt.k --box -20,20,-20,20,8,48 --detonation 0,0,10

顺带可以把起爆点挪到炸药内部（贴着边界容易落空）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

FILLING = "*ALE_STRUCTURED_MESH_VOLUME_FILLING"


def find_blocks(lines: list[str], name: str) -> list[tuple[int, int]]:
    """找出所有 ``*NAME`` 块的行范围 ``[start, end)``。"""
    starts = [i for i, line in enumerate(lines) if line.startswith("*")]
    blocks = []
    for position, index in enumerate(starts):
        keyword = lines[index][1:].split()[0].upper()
        if keyword == name:
            end = starts[position + 1] if position + 1 < len(starts) else len(lines)
            blocks.append((index, end))
    return blocks


def box_filling_block(mshid: int, group: str, box_id: int, nsample: int) -> list[str]:
    """``GEOM=BOXCOR`` 的填充块：引用一个 ``*DEFINE_BOX``。

    LS-DYNA 的几何类型只有 ALL/PARTSET/PART/SEGSET/PLANE/CYLINDER/BOXCOR/BOXCPT/
    ELLIPSOID，没有 ``BOX``；长方体要用 ``BOXCOR`` + ``*DEFINE_BOX``。
    """
    return [
        FILLING,
        "$#   mshid         -    ammgto         -   nsample         -         -       vid",
        f"{mshid:>10}{'':10}{group:<10}{'':10}{nsample:>10}{'':20}{0:>10}",
        "$#    geom    in/out     boxid        e2        e3        e4        e5",
        f"{'BOXCOR':>10}{0:>10}{box_id:>10}{'':30}",
    ]


def define_box_block(box_id: int, box: Sequence[float]) -> list[str]:
    """``*DEFINE_BOX``：xmin,xmax,ymin,ymax,zmin,zmax。"""
    return [
        "*DEFINE_BOX",
        "$#    boxid        xmn        xmx        ymn        ymx        zmn        zmx",
        f"{box_id:>10}" + "".join(f"{value:>10.6g}" for value in box),
    ]


def replace_tnt_filling(source: Path, output: Path, group: str, box: Sequence[float],
                        detonation: Sequence[float] | None, box_id: int = 1,
                        nsample: int = 3) -> None:
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()

    target: tuple[int, int] | None = None
    for start, end in find_blocks(lines, "ALE_STRUCTURED_MESH_VOLUME_FILLING"):
        if any(group in line for line in lines[start:end]):
            target = (start, end)
    if target is None:
        raise KeyError(f"没找到填充 {group} 的 *ALE_STRUCTURED_MESH_VOLUME_FILLING 块")
    start, end = target

    mshid = int(lines[start + 2].split()[0])
    if find_blocks(lines, "DEFINE_BOX"):
        raise ValueError("deck 里已经有 *DEFINE_BOX，请换一个 --box-id 或手动合并")
    new_block = define_box_block(box_id, box) + box_filling_block(mshid, group, box_id, nsample)
    lines = lines[:start] + new_block + lines[end:]

    if detonation is not None:
        for index, line in enumerate(lines):
            if line.startswith("*INITIAL_DETONATION"):
                card = index + 2
                while lines[card].lstrip().startswith("$"):
                    card += 1
                line = lines[card]
                pid = line[0:10]
                lt = line[40:50]
                mmgset = line[60:70]
                lines[card] = (f"{pid}{str(f'{detonation[0]:.6g}'):>10}"
                               f"{str(f'{detonation[1]:.6g}'):>10}"
                               f"{str(f'{detonation[2]:.6g}'):>10}{lt}{'':10}{mmgset}")
                break

    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    print(f"wrote {output}")
    print(f"  {group}: GEOM=BOXCOR + *DEFINE_BOX {box_id}  "
          f"x[{box[0]:g}, {box[1]:g}] y[{box[2]:g}, {box[3]:g}] "
          f"z[{box[4]:g}, {box[5]:g}]  (mshid={mshid}, nsample={nsample})")
    if detonation is not None:
        print(f"  *INITIAL_DETONATION -> ({detonation[0]:g}, {detonation[1]:g}, {detonation[2]:g})")
    for line in new_block[2:]:
        if not line.startswith(("$", "*")):
            print("   |", line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fill_cavity_tnt",
        description="把 S-ALE 炸药填充改成内腔长方体区域",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="输入 deck")
    parser.add_argument("--output", required=True, help="输出 deck")
    parser.add_argument("--group", default="ammg-tnt", help="要重新填充的 MMG 名字")
    parser.add_argument("--box", default="-20,20,-20,20,8,48",
                        help="填充区域 xmin,xmax,ymin,ymax,zmin,zmax")
    parser.add_argument("--detonation", default="0,0,10",
                        help="新的起爆点 x,y,z（给 none 表示不改）")
    parser.add_argument("--box-id", type=int, default=1, help="新写的 *DEFINE_BOX 编号")
    parser.add_argument("--nsample", type=int, default=3, help="每个单元的采样点数")
    args = parser.parse_args(argv)

    replace_tnt_filling(
        source=Path(args.source),
        output=Path(args.output),
        group=args.group,
        box=tuple(float(v) for v in args.box.split(",")),
        detonation=None if args.detonation.lower() == "none"
            else tuple(float(v) for v in args.detonation.split(",")),
        box_id=args.box_id,
        nsample=args.nsample,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
