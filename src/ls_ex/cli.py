"""命令行入口：由参数生成空心长方体（壳）的 .k 文件。

等价于运行 ``examples/generate_hollow_box.py``，安装后用 ``ls-ex`` 调用。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .KeywordGenerator import (as_vec3, box_mesh, build_keyword_file,
                               hollow_box_mesh, hollow_box_shell_mesh)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ls-ex",
        description="生成空心长方体（长方体壳）的 LS-DYNA .k 文件。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kind", choices=("solid", "shell", "box"), default="solid",
                        help="solid=空心长方体实体网格；shell=壳单元；box=实心长方体")
    parser.add_argument("--inner", default="20,10,10",
                        help="内腔尺寸 cx,cy,cz（单个数字表示正方体空腔）")
    parser.add_argument("-t", "--thickness", type=float, default=2.0,
                        help="壳（壁）厚")
    parser.add_argument("-e", "--element-size", type=float, default=None,
                        help="目标单元尺寸，缺省等于壁厚")
    parser.add_argument("--n-wall", type=int, default=None,
                        help="壁厚方向的单元层数（覆盖由 element-size 推出的值）")
    parser.add_argument("--origin", default="0,0,0",
                        help="外箱体最小角点坐标")
    parser.add_argument("--title", default=None, help="写入 *TITLE / *PART 的标题")
    parser.add_argument("--endtim", type=float, default=1.0,
                        help="*CONTROL_TERMINATION 的 ENDTIM")
    parser.add_argument("-o", "--output", default="outputs/hollow_box.k",
                        help="输出文件")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    element_size = args.element_size if args.element_size else args.thickness

    if args.kind == "shell":
        mesh = hollow_box_shell_mesh(args.inner, args.thickness,
                                     element_size=element_size,
                                     origin=args.origin)
    elif args.kind == "box":
        mesh = box_mesh(args.inner, element_size,
                        origin=as_vec3(args.origin, "origin"))
    else:
        mesh = hollow_box_mesh(args.inner, args.thickness,
                               element_size=element_size,
                               n_wall=args.n_wall,
                               origin=args.origin)

    title = args.title or f"{args.kind}: inner={as_vec3(args.inner, 'inner')} t={args.thickness}"
    keyword_file = build_keyword_file(mesh, title=title, endtim=args.endtim)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = keyword_file.write(path)

    print(f"wrote {path}")
    print(f"  {mesh.summary()}")
    return 0
