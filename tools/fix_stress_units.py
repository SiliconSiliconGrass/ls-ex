"""把 deck 里"应力/压力"类的材料数据统一到 mm-g-ms（MPa）单位制。

这套 deck 的长度/质量/时间单位是 mm-g-ms（密度 0.00783 = 钢 7.83 g/cm³、
1.29e-6 = 空气、0.00163 = TNT，爆速 6930 mm/ms = 6930 m/s 都对得上），
但强度/压力类的数值却小得多：钢 E 写成 201（应 210000 MPa）、TNT 的
JWL 系数只有教科书值的 1/10^6，于是炸药几乎不产生压力，看起来"没爆"。

这个脚本按统一单位制改写这些字段（密度保持原值不动）::

    python tools/fix_stress_units.py --source main_hollow_cube_tnt.k \\
        --output main_hollow_cube_tnt_units.k

改写的数据都是钢/TNT 的标准值，可用 ``--scale`` 再整体缩放（默认 1）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

# 关键字 -> {字段位置(0 起, 不含标题卡): 新值}
UPDATES: dict[str, dict[int, float]] = {
    # MID, RO, E, PR, SIGY, ETAN, BETA
    "*MAT_PLASTIC_KINEMATIC_TITLE": {2: 210000.0, 4: 1650.0, 5: 300.0},
    # MID, RO, D, PCJ, BETA, K, G, SIGY
    "*MAT_HIGH_EXPLOSIVE_BURN_TITLE": {3: 21000.0},
    # EOSID, A, B, R1, R2, OMEG, E0, V0
    "*EOS_JWL_TITLE": {1: 371200.0, 2: 3231.0, 6: 7000.0},
}

HEADERS = {
    "*MAT_PLASTIC_KINEMATIC_TITLE": ["mid", "ro", "e", "pr", "sigy", "etan", "beta",
                                     "src", "srp", "fs", "vp"],
    "*MAT_HIGH_EXPLOSIVE_BURN_TITLE": ["mid", "ro", "d", "pcj", "beta", "k", "g", "sigy"],
    "*EOS_JWL_TITLE": ["eosid", "a", "b", "r1", "r2", "omeg", "e0", "vo"],
}


def data_card(lines: list[str], keyword: str, card_index: int = 2) -> int:
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


def rewrite(lines: list[str], keyword: str, scale: float) -> tuple[str, str]:
    index = data_card(lines, keyword)
    values = [float(token) for token in lines[index].split()]
    names = HEADERS[keyword][:len(values)]
    before = dict(zip(names, values))
    for position, value in UPDATES[keyword].items():
        values[position] = value * scale

    def field(value: float) -> str:
        if abs(value - round(value)) < 1e-12 and abs(value) < 1e11:
            return f"{int(round(value)):>10}"
        return f"{value:>10.6g}"

    lines[index] = "".join(field(value) for value in values)
    after = dict(zip(names, values))
    changed = ", ".join(f"{name}: {before[name]:g} -> {after[name]:g}"
                        for name in names
                        if before[name] != after[name])
    return lines[index], changed


def fix_units(source: Path, output: Path, scale: float = 1.0) -> None:
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"wrote {output}")
    for keyword in UPDATES:
        if keyword not in lines:
            raise KeyError(f"deck 里找不到 {keyword}")
        text, changed = rewrite(lines, keyword, scale)
        print(f"  {keyword:<32} {changed}")
        print(f"    |{text}|")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fix_stress_units",
        description="把钢/TNT 的强度与压力数据统一到 mm-g-ms(MPa) 单位制",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=float, default=1.0, help="对写入值再整体缩放")
    args = parser.parse_args(argv)
    fix_units(Path(args.source), Path(args.output), args.scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
