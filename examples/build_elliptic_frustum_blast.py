"""生成"椭圆台体壳 + S-ALE 炸药爆炸（侵蚀破片，无 SPH）"的完整 .k 文件。

用法::

    python examples/build_elliptic_frustum_blast.py
    python examples/build_elliptic_frustum_blast.py --ale-size 4   # 快速试算

几何参数（默认值即本次任务的参数）：
* 大端面半长轴 a0 = 43.20 mm，半短轴 b0 = 36 mm；
* 壳体厚度 d = 8 mm；
* 台体高度 H = 100 mm；
* 长轴方向锥角 86.6°（侧壁与端面的夹角）：
  ``tan(86.6°) = H / (a0 - a1)``，由此得到小端位似比
  ``k = a1 / a0 = 1 - H / (a0 * tan(86.6°)) = 0.85946``。

几何约定与 ``elliptical_frustum_shell_mesh`` 一致：给定的椭圆/高度/位似比是**内腔**；
外轮廓是每个截面的椭圆等距曲线；两端各有厚 d 的实心端盖；总高 = H + 2d。

炸药填充：默认 ``--fill cavity``，用 ``*INITIAL_VOLUME_FRACTION_NALEGP``
**逐单元**给出 air/TNT 体积分数，把 TNT 精确填满椭圆台体空腔（部分单元按
5×5×5 采样计算分数）。``--fill cylinder`` 则退回到腔内柱形装药（更快，无需大块数据）。

单位制：mm-g-ms（应力单位 MPa），与之前修正后的材料数据一致。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

from ls_ex import KCard, KWord, KeywordFile, elliptical_frustum_shell_mesh
from ls_ex.KeywordGenerator import ELEMENT_WIDTHS, NODE_WIDTHS


def _card(*values, widths=10, kinds=None) -> KCard:
    if isinstance(widths, int):
        widths = [widths] * len(values)
    return KCard(values=list(values), widths=list(widths),
                 kinds=list(kinds) if kinds else None)


def _word(name: str, cards: Sequence[KCard]) -> KWord:
    return KWord(name, list(cards))


def _header(names: Sequence[str], widths=10) -> KCard:
    return KCard.header(names, list(widths) if isinstance(widths, (list, tuple)) else [widths] * len(names))


def taper_ratio(a0: float, height: float, cone_angle_deg: float) -> float:
    """由"侧壁与端面夹角"求小端/大端位似比。"""
    ratio = 1.0 - height / (a0 * math.tan(math.radians(cone_angle_deg)))
    if not 0 < ratio <= 1.5:
        raise ValueError(f"锥角与尺寸不匹配：位似比 {ratio:g}（应接近 0~1）")
    return ratio


def symmetry_cut_axes(symmetry: str) -> tuple[str, ...]:
    """Return the global axes normal to the retained domain's symmetry cuts."""
    choices = {
        "full": (),
        "half-long": ("y",),
        "half-short": ("x",),
        "quarter": ("x", "y"),
    }
    try:
        return choices[symmetry]
    except KeyError:
        raise ValueError(
            "symmetry must be full, half-long, half-short, or quarter"
        ) from None


def cavity_fractions(a0: float, b0: float, height: float, ratio: float,
                     bounds: tuple[float, float, float, float, float, float],
                     counts: tuple[int, int, int], base_eid: int) -> list[tuple[int, float]]:
    """每个 S-ALE 单元的 (eid, TNT 体积分数)；其余为 air=1-vf (Steiner 空腔)。"""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    nx, ny, nz = counts
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    def inside(x: float, y: float, z: float) -> bool:
        if z < 0.0 or z > height:
            return False
        scale = 1.0 + (ratio - 1.0) * z / height
        return (x / (a0 * scale)) ** 2 + (y / (b0 * scale)) ** 2 <= 1.0

    result: list[tuple[int, float]] = []
    eid = base_eid
    for k in range(nz):
        zc = z_min + (k + 0.5) * dz
        for j in range(ny):
            yc = y_min + (j + 0.5) * dy
            for i in range(nx):
                xc = x_min + (i + 0.5) * dx
                # 8 个角点 + 中心
                flags = []
                for sx in (-0.5, 0.5):
                    for sy in (-0.5, 0.5):
                        for sz in (-0.5, 0.5):
                            flags.append(inside(xc + sx * dx, yc + sy * dy, zc + sz * dz))
                center = inside(xc, yc, zc)
                if all(flags):
                    fraction = 1.0
                elif not any(flags) and not center:
                    fraction = 0.0
                else:
                    # 部分填充：5x5x5 子采样
                    hits = 0
                    total = 0
                    for a in range(5):
                        sx = (a + 0.5) / 5 - 0.5
                        for b in range(5):
                            sy = (b + 0.5) / 5 - 0.5
                            for c in range(5):
                                sz = (c + 0.5) / 5 - 0.5
                                total += 1
                                if inside(xc + sx * dx, yc + sy * dy, zc + sz * dz):
                                    hits += 1
                    fraction = hits / total
                result.append((eid, fraction))
                eid += 1
    return result


def build_deck(major_semi: float = 43.2, minor_semi: float = 36.0,
               height: float = 100.0, thickness: float = 8.0,
               cone_angle: float = 86.6, element_size: float = 2.0,
               ale_size: float | None = None, endtim: float = 0.2,
               air_margin: float = 40.0, fill: str = "cavity",
               symmetry: str = "full") -> KeywordFile:
    if fill not in ("cavity", "cylinder"):
        raise ValueError("fill 只能是 'cavity' 或 'cylinder'")
    cut_axes = symmetry_cut_axes(symmetry)
    ale_size = ale_size or element_size
    ratio = taper_ratio(major_semi, height, cone_angle)
    mesh = elliptical_frustum_shell_mesh(
        2 * major_semi, 2 * minor_semi, height, ratio, thickness,
        element_size=element_size, pid=2, symmetry=symmetry,
    )
    # 柱形装药的轴线节点：优先复用端盖中心已有的节点，避免 massless node 警告
    def node_at(x: float, y: float, z: float) -> int:
        for nid, point in mesh.nodes.items():
            if (abs(point[0] - x) < 1e-9 and abs(point[1] - y) < 1e-9
                    and abs(point[2] - z) < 1e-9):
                return nid
        nid = len(mesh.nodes) + 1
        mesh.nodes[nid] = (x, y, z)
        return nid

    axis_node_1 = axis_node_2 = None
    if fill == "cylinder":
        axis_node_1 = node_at(0.0, 0.0, 0.0)
        axis_node_2 = node_at(0.0, 0.0, height)
    charge_radius = 30.0 if minor_semi > 30.0 else 0.8 * minor_semi
    outer_x = major_semi + thickness + air_margin
    outer_y = minor_semi + thickness + air_margin
    x_min = 0.0 if "x" in cut_axes else -outer_x
    x_max = outer_x
    y_min = 0.0 if "y" in cut_axes else -outer_y
    y_max = outer_y
    z_min = -thickness - 0.4 * height
    z_max = height + thickness + 0.4 * height

    deck = KeywordFile(header=[
        "$# LS-DYNA deck generated by ls_ex: elliptic frustum shell + S-ALE blast",
        f"$# a0={major_semi} b0={minor_semi} H={height} t={thickness} k={ratio:.6f} "
        f"cone={cone_angle} mesh={element_size}",
        f"$# symmetry={symmetry} cut_axes={','.join(cut_axes) or 'none'}",
    ])
    deck.add(KWord("KEYWORD"))
    deck.add(_word("TITLE", [
        _header(["title"], 80),
        _card(f"elliptic frustum shell blast a0={major_semi} b0={minor_semi} {symmetry}",
              widths=[80], kinds=["s"]),
    ]))

    # ------------------------------------------------------------- 控制
    deck.add(_word("CONTROL_ALE", [
        _header(["dct", "nadv", "meth", "afac", "bfac", "cfac", "dfac", "efac"]),
        _card(1, 1, 2, 1.0, 0.0, 0.0, 0.0, 0.0, kinds=["i", "i", "i", "f", "f", "f", "f", "f"]),
        _header(["start", "end", "aafac", "vfact", "prit", "ebc", "pref", "nsidebc"]),
        _card(0.0, 1.0e20, 1.0, 1.0e-6, 0, 0, 0.0, 0,
              kinds=["f", "f", "f", "f", "i", "i", "f", "i"]),
        _header(["ncpl", "nbkt", "imascl", "checkr", "beamin", "mmgpref", "pdifmx", "dtmufac"]),
        _card(1, 50, 0, 0.0, 0.0, 0, 0.0, 0.0, kinds=["i", "i", "i", "f", "f", "i", "f", "f"]),
        _header(["optimpp", "ialedr", "bndflx", "minmas"], [10] * 4),
        _card(0, 0, 0, 1.0e-5, kinds=["i", "i", "i", "f"]),
    ]))
    deck.add(_word("CONTROL_TERMINATION", [
        _header(["endtim", "endcyc", "dtmin", "endeng", "endmas", "nosol"]),
        _card(endtim, 0, 0.0, 0.0, 1.0e8, 0, kinds=["f", "i", "f", "f", "f", "i"]),
    ]))
    deck.add(_word("CONTROL_TIMESTEP", [
        _header(["dtinit", "tssfac", "isdo", "tslimt", "dt2ms", "lctm", "erode", "ms1st"]),
        _card(1.0e-4, 0.9, 0, 0.0, 0.0, 0, 0, 0,
              kinds=["f", "f", "i", "f", "f", "i", "i", "i"]),
    ]))
    deck.add(_word("DATABASE_BINARY_D3PLOT", [
        _header(["dt", "lcdt", "beam", "npltc", "psetid"]),
        _card(endtim / 50, 0, 0, 0, 0, kinds=["f", "i", "i", "i", "i"]),
    ]))
    deck.add(_word("DATABASE_GLSTAT", [
        _header(["dt", "binary", "lcur", "ioopt"]),
        _card(endtim / 50, 0, 0, 1, kinds=["f", "i", "i", "i"]),
    ]))
    deck.add(_word("DATABASE_MATSUM", [
        _header(["dt", "binary", "lcur", "ioopt"]),
        _card(endtim / 50, 0, 0, 1, kinds=["f", "i", "i", "i"]),
    ]))

    # ------------------------------------------------------------- 材料
    deck.add(_word("MAT_NULL_TITLE", [
        _card("mat-air", widths=[80], kinds=["s"]),
        _header(["mid", "ro", "pc", "mu", "terod", "cerod", "ym", "pr"]),
        _card(1, 1.29e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              kinds=["i"] + ["f"] * 7),
    ]))
    deck.add(_word("MAT_HIGH_EXPLOSIVE_BURN_TITLE", [
        _card("mat-tnt", widths=[80], kinds=["s"]),
        _header(["mid", "ro", "d", "pcj", "beta", "k", "g", "sigy"]),
        _card(2, 0.00163, 6930.0, 21000.0, 0.0, 0.0, 0.0, 0.0,
              kinds=["i"] + ["f"] * 7),
    ]))
    deck.add(_word("MAT_PLASTIC_KINEMATIC_TITLE", [
        _card("mat-shell", widths=[80], kinds=["s"]),
        _header(["mid", "ro", "e", "pr", "sigy", "etan", "beta"]),
        _card(3, 0.00783, 210000.0, 0.33, 1650.0, 300.0, 0.0,
              kinds=["i"] + ["f"] * 6),
        _header(["src", "srp", "fs", "vp"]),
        _card(0.0, 0.0, 0.2, 0.0, kinds=["f"] * 4),
    ]))
    deck.add(_word("EOS_LINEAR_POLYNOMIAL_TITLE", [
        _card("eos-air", widths=[80], kinds=["s"]),
        _header(["eosid", "c0", "c1", "c2", "c3", "c4", "c5", "c6"]),
        _card(1, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.0, kinds=["i"] + ["f"] * 7),
        _header(["e0", "v0"]),
        _card(0.0, 0.0, kinds=["f", "f"]),
    ]))
    deck.add(_word("EOS_JWL_TITLE", [
        _card("eos-tnt", widths=[80], kinds=["s"]),
        _header(["eosid", "a", "b", "r1", "r2", "omeg", "e0", "vo"]),
        _card(2, 371200.0, 3231.0, 4.15, 0.95, 0.3, 7000.0, 1.0,
              kinds=["i"] + ["f"] * 7),
    ]))

    # ------------------------------------------------------------- 网格
    node_block = KWord("NODE", [_header(["nid", "x", "y", "z", "tc", "rc"], NODE_WIDTHS)])
    for nid in sorted(mesh.nodes):
        x, y, z = mesh.nodes[nid]
        node_block.add(KCard(values=[nid, x, y, z, 0, 0], widths=NODE_WIDTHS,
                             kinds=["i", "f", "f", "f", "i", "i"]))
    deck.add(node_block)

    deck.add(_word("SECTION_SOLID_TITLE", [
        _card("section-shell", widths=[80], kinds=["s"]),
        _header(["secid", "elform", "aet"]),
        _card(1, 1, 0, kinds=["i", "i", "i"]),
    ]))
    deck.add(_word("PART", [
        _header(["title"], 80),
        _card("shell", widths=[80], kinds=["s"]),
        _header(["pid", "secid", "mid", "eosid", "hgid", "grav", "adpopt", "tmid"]),
        _card(2, 1, 3, 0, 0, 0, 0, 0, kinds=["i"] * 8),
    ]))

    element_block = KWord("ELEMENT_SOLID", [
        _header(["eid", "pid", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8"],
                ELEMENT_WIDTHS),
    ])
    for eid, pid, corners in mesh.solids:
        element_block.add(KCard(values=[eid, pid, *corners], widths=ELEMENT_WIDTHS,
                                kinds=["i"] * 10))
    deck.add(element_block)

    if cut_axes:
        symmetry_cards = [
            _header(["idsp", "pid", "x", "y", "z", "vx", "vy", "vz"]),
        ]
        for plane_id, axis in enumerate(cut_axes, 1):
            symmetry_cards.append(_card(
                plane_id, 2, 0.0, 0.0, 0.0,
                1.0 if axis == "x" else 0.0,
                1.0 if axis == "y" else 0.0,
                0.0,
                kinds=["i", "i"] + ["f"] * 6,
            ))
            symmetry_cards.append(_header(["tol"]))
            symmetry_cards.append(_card(1.0e-6, kinds=["f"]))
        deck.add(_word("BOUNDARY_SPC_SYMMETRY_PLANE", symmetry_cards))

    deck.add(_word("INITIAL_DETONATION", [
        _header(["ipid", "x", "y", "z", "tl", "-", "mmgset"]),
        _card(0, 0.0, 0.0, 10.0, 0.0, "", 1,
              kinds=["i", "f", "f", "f", "f", "s", "i"]),
    ]))

    # ------------------------------------------------------------- S-ALE
    deck.add(_word("ALE_STRUCTURED_MESH", [
        _header(["mshid", "dpid", "nbid", "ebid", "-", "-", "-", "tdeath"]),
        _card(1, 1, 0, 0, "", "", "", 1.0e16,
              kinds=["i", "i", "i", "i", "s", "s", "s", "f"]),
        _header(["cpidx", "cpidy", "cpidz", "nid0", "lcsid"]),
        _card(2, 3, 1, 0, 0, kinds=["i"] * 5),
    ]))
    n_x = max(1, round((x_max - x_min) / ale_size))
    n_y = max(1, round((y_max - y_min) / ale_size))
    n_z = max(1, round((z_max - z_min) / ale_size))
    deck.add(_word("ALE_STRUCTURED_MESH_CONTROL_POINTS", [
        _header(["cpid", "-", "icase", "sfo", "-", "offo"]),
        _card(1, "", 0, 1.0, "", 0.0, kinds=["i", "s", "i", "f", "s", "f"]),
        _header(["n", "x", "ratio/xl"], [20, 20, 20]),
        _card(1, z_min, 0.0, widths=[20, 20, 20], kinds=["i", "f", "f"]),
        _card(n_z + 1, z_max, 0.0, widths=[20, 20, 20], kinds=["i", "f", "f"]),
    ]))
    deck.add(_word("ALE_STRUCTURED_MESH_CONTROL_POINTS", [
        _header(["cpid", "-", "icase", "sfo", "-", "offo"]),
        _card(2, "", 0, 1.0, "", 0.0, kinds=["i", "s", "i", "f", "s", "f"]),
        _header(["n", "x", "ratio/xl"], [20, 20, 20]),
        _card(1, x_min, 0.0, widths=[20, 20, 20], kinds=["i", "f", "f"]),
        _card(n_x + 1, x_max, 0.0, widths=[20, 20, 20], kinds=["i", "f", "f"]),
    ]))
    deck.add(_word("ALE_STRUCTURED_MESH_CONTROL_POINTS", [
        _header(["cpid", "-", "icase", "sfo", "-", "offo"]),
        _card(3, "", 0, 1.0, "", 0.0, kinds=["i", "s", "i", "f", "s", "f"]),
        _header(["n", "x", "ratio/xl"], [20, 20, 20]),
        _card(1, y_min, 0.0, widths=[20, 20, 20], kinds=["i", "f", "f"]),
        _card(n_y + 1, y_max, 0.0, widths=[20, 20, 20], kinds=["i", "f", "f"]),
    ]))
    if cut_axes:
        deck.add(_word("BOUNDARY_SALE_MESH_FACE", [
            _header(["bctype", "mshid", "negx", "posx", "negy", "posy",
                     "negz", "posz"]),
            _card("SYM", 1,
                  int("x" in cut_axes), 0,
                  int("y" in cut_axes), 0,
                  0, 0, kinds=["s"] + ["i"] * 7),
        ]))
    deck.add(_word("ALE_STRUCTURED_MULTI-MATERIAL_GROUP", [
        _header(["ammgnm", "mid", "eosid", "-", "-", "-", "-", "pref"]),
        _card("ammg-air", 1, 1, "", "", "", "", 1.01325e-1,
              kinds=["s", "i", "i", "s", "s", "s", "s", "f"]),
        _header(["ammgnm", "mid", "eosid", "-", "-", "-", "-", "pref"]),
        _card("ammg-tnt", 2, 2, "", "", "", "", 0.0,
              kinds=["s", "i", "i", "s", "s", "s", "s", "f"]),
    ]))
    if fill == "cylinder":
        deck.add(_word("ALE_STRUCTURED_MESH_VOLUME_FILLING", [
            _header(["mshid", "-", "ammgto", "-", "nsample", "-", "-", "vid"]),
            _card(1, "", "ammg-air", "", 3, "", "", 0,
                  kinds=["i", "s", "s", "s", "i", "s", "s", "i"]),
            _header(["geom", "in/out", "e1", "e2", "e3", "e4", "e5"]),
            _card("ALL", 0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  kinds=["s", "i", "f", "f", "f", "f", "f"]),
        ]))
        deck.add(_word("ALE_STRUCTURED_MESH_VOLUME_FILLING", [
            _header(["mshid", "-", "ammgto", "-", "nsample", "-", "-", "vid"]),
            _card(1, "", "ammg-tnt", "", 3, "", "", 0,
                  kinds=["i", "s", "s", "s", "i", "s", "s", "i"]),
            _header(["geom", "in/out", "e1", "e2", "e3", "e4", "e5"]),
            _card("CYLINDER", 0, axis_node_1, axis_node_2, charge_radius, charge_radius, 0.0,
                  kinds=["s", "i", "i", "i", "f", "f", "f"]),
        ]))
    else:
        # 逐单元体积分数：TNT 精确填满内腔（air = 1 - TNT）
        fractions = cavity_fractions(
            major_semi, minor_semi, height, ratio,
            (x_min, x_max, y_min, y_max, z_min, z_max),
            (n_x, n_y, n_z), base_eid=len(mesh.solids) + 1,
        )
        fill_block = KWord("INITIAL_VOLUME_FRACTION_NALEGP", [
            _header(["nalegp"]),
            _card(2, kinds=["i"]),
        ])

        def _frac(value: float) -> str:
            if value >= 1.0:
                return "         1"
            if value <= 0.0:
                return "         0"
            return f"{value:>10.5f}"

        for eid, tnt_fraction in fractions:
            air_fraction = 1.0 - tnt_fraction
            fill_block.add(KCard(line=f"{eid:>10}{_frac(air_fraction)}{_frac(tnt_fraction)}"))
        deck.add(fill_block)
    deck.add(_word("SET_MULTI_MATERIAL_GROUP_TITLE", [
        _card("setmmg-tnt", widths=[80], kinds=["s"]),
        _header(["ammsid"]),
        _card(1, kinds=["i"]),
        _header(["ammgid1", "ammgid2", "ammgid3", "ammgid4",
                 "ammgid5", "ammgid6", "ammgid7", "ammgid8"], [10] * 8),
        _card("ammg-tnt", 0, 0, 0, 0, 0, 0, 0,
              kinds=["s"] + ["i"] * 7),
    ]))
    deck.add(_word("ALE_STRUCTURED_FSI", [
        _header(["lstrsid", "alesid", "lstrstyp", "alestyp", "-", "-", "-", "mcoup"]),
        _card(2, 1, 1, 1, "", "", "", -1,
              kinds=["i", "i", "i", "i", "s", "s", "s", "i"]),
        _header(["start", "end", "pfac", "fric", "ilvl", "flip", "decay", "offset"]),
        _card(0.0, 1.0e10, 0.1, 0.0, 0, 0, 0.0, 0.0,
              kinds=["f", "f", "f", "f", "i", "i", "f", "f"]),
    ]))

    # ------------------------------------------------------------- 破片
    deck.add(_word("CONTACT_ERODING_SINGLE_SURFACE", [
        _header(["ssid", "msid", "sstyp", "mstyp", "sboxid", "mboxid", "spr", "mpr"]),
        _card(0, 0, 0, 0, 0, 0, 0, 0, kinds=["i"] * 8),
        _header(["fs", "fd", "dc", "vc", "vdc", "penchk", "bt", "dt"]),
        _card(0.3, 0.3, 0.0, 0.0, 0.0, 0, 0.0, 0.0,
              kinds=["f", "f", "f", "f", "f", "i", "f", "f"]),
        _header(["sfs", "sfm", "sst", "mst", "sfst", "sfmt", "fsf", "vsf"]),
        _card(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, kinds=["f"] * 8),
        _header(["isym", "erosop", "iadj", "", "", "", "", "unused1"]),
        _card(0, 1, 1, "", "", "", "", "", kinds=["i", "i", "i", "s", "s", "s", "s", "s"]),
        _header(["soft", "softscl", "lcidab", "maxpar", "sbopt", "depth", "bsort", "frcfrq"]),
        _card(1, 0.1, 0, 0, 0, 5, 100, 0, kinds=["i", "f", "i", "i", "i", "i", "i", "i"]),
    ]))
    deck.add(KWord("END"))
    return deck


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_elliptic_frustum_blast",
        description="椭圆台体壳 + S-ALE 爆炸（侵蚀破片，无 SPH）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--major-semi", type=float, default=43.2)
    parser.add_argument("--minor-semi", type=float, default=36.0)
    parser.add_argument("--height", type=float, default=100.0)
    parser.add_argument("--thickness", type=float, default=8.0)
    parser.add_argument("--cone-angle", type=float, default=86.6)
    parser.add_argument("--element-size", type=float, default=2.0)
    parser.add_argument("--ale-size", type=float, default=None,
                        help="S-ALE 单元尺寸（缺省与结构网格相同）")
    parser.add_argument("--endtim", type=float, default=0.2)
    parser.add_argument("--air-margin", type=float, default=40.0)
    parser.add_argument("--fill", choices=("cavity", "cylinder"), default="cavity",
                        help="cavity=逐单元体积分数精确充满内腔；cylinder=腔内柱形装药")
    parser.add_argument("--symmetry",
                        choices=("full", "half-long", "half-short", "quarter"),
                        default="full",
                        help="full / half-long (along major axis) / "
                             "half-short (along minor axis) / quarter")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args(argv)

    deck = build_deck(
        major_semi=args.major_semi, minor_semi=args.minor_semi,
        height=args.height, thickness=args.thickness,
        cone_angle=args.cone_angle, element_size=args.element_size,
        ale_size=args.ale_size, endtim=args.endtim, air_margin=args.air_margin,
        fill=args.fill, symmetry=args.symmetry,
    )
    suffix = "" if args.symmetry == "full" else f"_{args.symmetry}"
    path = Path(args.output or f"outputs/elliptic_frustum_blast_{args.fill}{suffix}.k")
    path.parent.mkdir(parents=True, exist_ok=True)
    deck.write(path)

    ratio = taper_ratio(args.major_semi, args.height, args.cone_angle)
    print(f"wrote {path}")
    print(f"  inside big ellipse: a0={args.major_semi} b0={args.minor_semi} "
          f"H={args.height} t={args.thickness} k={ratio:.6f}")
    print(f"  shell mesh {args.element_size} mm, ALE {args.ale_size or args.element_size} mm, "
          f"endtim={args.endtim} ms")
    print(f"  symmetry={args.symmetry}, cut axes={symmetry_cut_axes(args.symmetry)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
