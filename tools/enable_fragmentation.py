"""给 deck 加上"壳体失效后碎裂成破片"的设置。

现在这套 deck 的钢壳用的是 ``*MAT_PLASTIC_KINEMATIC``，``FS``（失效应变）是 0，
也就是**永远不会失效**：壳只会被撑大、扭曲，不会破。要出破片需要三件事：

1. ``FS``（失效应变）—— 单元达到失效应变就被侵蚀（删除）；没有它什么都发不
   了，这是最关键的一步。
2. ``*CONTACT_ERODING_SINGLE_SURFACE`` —— 单元被侵蚀后，剩下的碎片之间还能接
   触、互相碰撞（否则碎片会互相穿透）。
3. ``*DEFINE_ADAPTIVE_SOLID_TO_SPH``（可选）—— 让失效的实体单元**直接转成
   SPH 粒子**继续飞，这才是真正意义上的"破片"；配合 ``*CONTROL_SPH`` 使用。

用法::

    python tools/enable_fragmentation.py --source a.k --output b.k            # 只有侵蚀+接触
    python tools/enable_fragmentation.py --source a.k --output c.k --sph      # 再加 SPH 破片

``--failure-strain`` 是主要旋钮：越小越早破、碎片越多（钢一般 0.05~0.4）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def data_card(lines: list[str], keyword: str, card_index: int) -> int:
    """第 card_index 张非注释卡的行号（标题卡也算一张）。"""
    index = lines.index(keyword)
    seen = 0
    while True:
        index += 1
        if lines[index].lstrip().startswith("$"):
            continue
        seen += 1
        if seen == card_index:
            return index


def strip(lines: Sequence[str]) -> list[str]:
    """去掉空行/多余空白，返回自包含的关键字块。"""
    return [line.rstrip() for line in lines]


def contact_block(fs: float, fd: float) -> list[str]:
    """侵蚀感知的自接触。

    注意：ERODING 接触在 SFS 卡之后还有一张 ``isym/erosop/iadj`` 卡，漏掉它
    LS-DYNA 会把下一行当这张卡来读 → ``Error 10246 (KEY+246)``。
    ``erosop=1``：为侵蚀接触分配存储；``iadj=1``：允许实体内部侵蚀后继续接触。
    """
    return strip([
        "*CONTACT_ERODING_SINGLE_SURFACE",
        "$# shell + debris self-contact (erosion aware)",
        "$#   ssid      msid     sstyp     mstyp    sboxid    mboxid       spr       mpr",
        "         0         0         0         0         0         0         0         0",
        "$#     fs        fd        dc        vc       vdc    penchk        bt        dt",
        f"{fs:>10.6g}{fd:>10.6g}{0.0:>10.6g}{0.0:>10.6g}{0.0:>10.6g}{0:>10}{0.0:>10.6g}{0.0:>10.6g}",
        "$#    sfs       sfm       sst       mst      sfst      sfmt       fsf       vsf",
        "         0         0       0.0       0.0       0.0       0.0       0.0       0.0",
        "$#   isym    erosop      iadj                                               unused1",
        "         0         1         1",
        "$#   soft   softscl    lcidab    maxpar     sbopt     depth     bsort    frcfrq",
        "         1       0.1         0         0         0         5       100         0",
    ])


def adaptive_sph_block(did: int, solid_part: int, sph_part: int, sph_section: int,
                       nq: int) -> list[str]:
    fields = (solid_part, 0, nq, sph_part, sph_section, 0, 1, 0.0)
    return strip([
        "*DEFINE_ADAPTIVE_SOLID_TO_SPH_ID",
        "$#     did                                                               heading",
        "$#    ipid     itype        nq     ipsph     issph      icpl      iopt      cpcd",
        f"{did:>10}adaptive-solid-to-sph",
        "".join(f"{value:>10}" if isinstance(value, int) else f"{value:>10.6g}"
                for value in fields),
    ])


def control_sph_block() -> list[str]:
    return strip([
        "*CONTROL_SPH",
        "$#    ncbs     boxid        dt      idim   nmneigh      form     start      maxv",
        f"{1:>10}{0:>10}{1.0e20:>10.5E}{3:>10}{150:>10}{0:>10}{0.0:>10.5E}{1.0e15:>10.5E}",
        "$#    cont     deriv       ini     ishow     ierod     icont     iavis     isymp",
        f"{0:>10}{0:>10}{0:>10}{1:>10}{0:>10}{0:>10}{0:>10}{100:>10}",
    ])


def enable(source: Path, output: Path, failure_strain: float, friction: float,
           use_sph: bool, solid_part: int, sph_part: int, sph_section: int,
           nq: int) -> None:
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()

    # 1) 失效应变：MAT_PLASTIC_KINEMATIC 的第三张卡（src, srp, fs, vp）
    index = data_card(lines, "*MAT_PLASTIC_KINEMATIC_TITLE", 3)
    fields = [float(token) for token in lines[index].split()]
    old = fields[2]
    fields[2] = failure_strain
    lines[index] = "".join(f"{value:>10.6g}" for value in fields)
    print(f"MAT_PLASTIC_KINEMATIC fs: {old:g} -> {failure_strain:g}")

    # 2/3) 接触 + 可选的 SPH 破片
    extra: list[str] = []
    if use_sph:
        extra += adaptive_sph_block(1, solid_part, sph_part, sph_section, nq)
        extra += control_sph_block()
    extra += contact_block(friction, friction)

    end = lines.index("*END")
    lines = lines[:end] + extra + lines[end:]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")

    print(f"wrote {output}")
    for line in extra:
        if not line.startswith("$"):
            print("   |", line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="enable_fragmentation",
        description="给 deck 加失效/侵蚀（可选 SPH 破片）与侵蚀接触",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failure-strain", type=float, default=0.2,
                        help="失效应变，越小越早碎裂")
    parser.add_argument("--friction", type=float, default=0.3, help="碎片间摩擦系数")
    parser.add_argument("--sph", action="store_true", help="失效单元转 SPH 破片")
    parser.add_argument("--solid-part", type=int, default=2, help="要转换的实体 part")
    parser.add_argument("--sph-part", type=int, default=3, help="新生成的 SPH part")
    parser.add_argument("--sph-section", type=int, default=2, help="SPH part 的 section id")
    parser.add_argument("--nq", type=int, default=1, choices=(1, 2, 3),
                        help="每个实体单元转成 1/8/27 个 SPH 粒子（六面体）")
    args = parser.parse_args(argv)

    enable(Path(args.source), Path(args.output), args.failure_strain, args.friction,
           args.sph, args.solid_part, args.sph_part, args.sph_section, args.nq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
