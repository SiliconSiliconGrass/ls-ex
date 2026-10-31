"""Convert the elliptic blast deck from S-ALE to explicit general ALE."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ls_ex import KCard, KWord, KeywordFile


SALE_KEYWORDS = {
    "ALE_STRUCTURED_MESH",
    "ALE_STRUCTURED_MESH_CONTROL_POINTS",
    "ALE_STRUCTURED_MULTI-MATERIAL_GROUP",
    "ALE_STRUCTURED_MESH_VOLUME_FILLING",
    "ALE_STRUCTURED_FSI",
    "BOUNDARY_SALE_MESH_FACE",
    "INITIAL_VOLUME_FRACTION_NALEGP",
    "SET_MULTI_MATERIAL_GROUP_TITLE",
}


def card(*values, widths=10, kinds=None) -> KCard:
    return KCard(values=list(values), widths=[widths] * len(values), kinds=kinds)


def header(*names) -> KCard:
    return KCard.header(list(names), [10] * len(names))


def word(name: str, *cards: KCard) -> KWord:
    return KWord(name, list(cards))


def raw(line: str) -> KCard:
    return KCard(line=line)


def before_end(deck: KeywordFile, blocks: Sequence[KWord]) -> None:
    index = next(i for i, keyword in enumerate(deck.keywords)
                 if keyword.name.upper() == "END")
    deck.keywords[index:index] = list(blocks)


def replace_keyword(deck: KeywordFile, replacement: KWord) -> None:
    for index, keyword in enumerate(deck.keywords):
        if keyword.name.upper() == replacement.name.upper():
            deck.keywords[index] = replacement
            return
    before_end(deck, [replacement])


def general_ale_mesh(
    node_base: int,
    element_base: int,
    symmetry: str = "half-long",
) -> tuple[
    KWord, KWord, list[list[int]], int, int, int
]:
    if symmetry not in {"full", "half-long", "half-short", "quarter"}:
        raise ValueError(f"unsupported symmetry {symmetry!r}")
    a0, b0, h, thickness = 43.2, 36.0, 100.0, 8.0
    small_ratio = 0.862475
    air_margin = 40.0
    outer_x = a0 + thickness + air_margin
    outer_y = b0 + thickness + air_margin
    z_min = -thickness - 0.4 * h
    z_max = h + thickness + 0.4 * h
    x_min = 0.0 if symmetry in {"half-short", "quarter"} else -outer_x
    x_max = outer_x
    y_min = 0.0 if symmetry in {"half-long", "quarter"} else -outer_y
    y_max = outer_y
    nx = round((x_max - x_min) / 2.0)
    ny = round((y_max - y_min) / 2.0)
    nz = round((z_max - z_min) / 2.0)
    xs = [x_min + 2.0 * i for i in range(nx + 1)]
    ys = [y_min + 2.0 * j for j in range(ny + 1)]
    zs = [z_min + 2.0 * k for k in range(nz + 1)]

    def node_id(i: int, j: int, k: int) -> int:
        return node_base + (k * (ny + 1) + j) * (nx + 1) + i

    node_cards = [raw("$#   nid               x               y               z      tc      rc")]
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                node_cards.append(raw(
                    f"{node_id(i, j, k):8d}{x:16.8f}{y:16.8f}{z:16.8f}{0:8d}{0:8d}"
                ))

    elem_cards = [raw("$#   eid     pid      n1      n2      n3      n4      n5      n6      n7      n8")]
    eid = element_base
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                xc = 0.5 * (xs[i] + xs[i + 1])
                yc = 0.5 * (ys[j] + ys[j + 1])
                zc = 0.5 * (zs[k] + zs[k + 1])
                scale = 1.0 + (small_ratio - 1.0) * zc / h
                is_tnt = (
                    0.0 <= zc <= h
                    and (xc / (a0 * scale)) ** 2
                    + (yc / (b0 * scale)) ** 2 <= 1.0
                )
                pid = 5 if is_tnt else 4
                nodes = (
                    node_id(i, j, k), node_id(i + 1, j, k),
                    node_id(i + 1, j + 1, k), node_id(i, j + 1, k),
                    node_id(i, j, k + 1), node_id(i + 1, j, k + 1),
                    node_id(i + 1, j + 1, k + 1), node_id(i, j + 1, k + 1),
                )
                elem_cards.append(raw(
                    f"{eid:8d}{pid:8d}" + "".join(f"{n:8d}" for n in nodes)
                ))
                eid += 1

    def nid(i: int, j: int, k: int) -> int:
        return node_id(i, j, k)

    outer_faces: list[list[int]] = []
    symmetry_faces: list[list[int]] = []
    for k in range(nz):
        for j in range(ny):
            x_min_face = [nid(0, j, k), nid(0, j + 1, k),
                          nid(0, j + 1, k + 1), nid(0, j, k + 1)]
            (symmetry_faces if x_min == 0.0 else outer_faces).append(x_min_face)
            outer_faces.append([nid(nx, j, k), nid(nx, j, k + 1),
                                nid(nx, j + 1, k + 1), nid(nx, j + 1, k)])
    for k in range(nz):
        for i in range(nx):
            y_min_face = [nid(i, 0, k), nid(i, 0, k + 1),
                          nid(i + 1, 0, k + 1), nid(i + 1, 0, k)]
            (symmetry_faces if y_min == 0.0 else outer_faces).append(y_min_face)
            outer_faces.append([nid(i, ny, k), nid(i + 1, ny, k),
                                nid(i + 1, ny, k + 1), nid(i, ny, k + 1)])
    for j in range(ny):
        for i in range(nx):
            outer_faces.append([nid(i, j, 0), nid(i, j + 1, 0),
                                nid(i + 1, j + 1, 0), nid(i + 1, j, 0)])
            outer_faces.append([nid(i, j, nz), nid(i + 1, j, nz),
                                nid(i + 1, j + 1, nz), nid(i, j + 1, nz)])

    return KWord("NODE", node_cards), KWord("ELEMENT_SOLID", elem_cards), \
        [outer_faces, symmetry_faces], nx, ny, nz


def segment_set(set_id: int, faces: Sequence[Sequence[int]]) -> KWord:
    cards = [
        raw("$#     sid       da1       da2       da3       da4    solver"),
        raw(f"{set_id:10d}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}MECH"),
        raw("$#      n1        n2        n3        n4        a1        a2        a3        a4"),
    ]
    for n1, n2, n3, n4 in faces:
        cards.append(raw(
            f"{n1:10d}{n2:10d}{n3:10d}{n4:10d}"
            f"{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}"
        ))
    return KWord("SET_SEGMENT", cards)


def convert(
    source: Path,
    output: Path,
    *,
    symmetry: str = "half-long",
    detonation_z: float = 1.0,
) -> None:
    deck = KeywordFile.load(source)
    deck.keywords = [keyword for keyword in deck.keywords
                     if keyword.name.upper() not in SALE_KEYWORDS]
    deck.keywords = [keyword for keyword in deck.keywords
                     if keyword.name.upper() != "ALE_COUPLING_NODAL_PENALTY"]
    nodes = deck.nodes()
    solids = deck.solid_elements()
    node_base = max(nodes, default=0) + 1
    element_base = max(solids, default=0) + 1
    node_block, elem_block, face_sets, nx, ny, nz = general_ale_mesh(
        node_base, element_base, symmetry
    )

    control_ale_word = word(
        "CONTROL_ALE",
        header("dct", "nadv", "meth", "afac", "bfac", "cfac", "dfac", "efac"),
        card(-1, 1, 2, -1.0, 0.0, 0.0, 0.0, 0.0,
             kinds=["i", "i", "i", "f", "f", "f", "f", "f"]),
        header("start", "end", "aafac", "vfact", "prit", "ebc", "pref", "nsidebc"),
        card(0.0, 1.0e20, 1.0, 1.0e-6, 0, 0, 0.0, 0,
             kinds=["f", "f", "f", "f", "i", "i", "f", "i"]),
        header("ncpl", "nbkt", "imascl", "checkr", "beamin", "mmgpref", "pdifmx", "dtmufac"),
        card(1, 50, 0, 0.0, 0.0, 0, 0.0, 0.0,
             kinds=["i", "i", "i", "f", "f", "i", "f", "f"]),
        header("optimpp", "ialedr", "bndflx", "minmas"),
        card(0, 0, 0, 1.0e-5, kinds=["i", "i", "i", "f"]),
    )
    detonation_word = word(
        "INITIAL_DETONATION",
        header("ipid", "x", "y", "z", "tl", "-", "mmgset"),
        card(5, 0.0, 0.0, detonation_z, 0.0, "", 0,
             kinds=["i", "f", "f", "f", "f", "s", "i"]),
    )
    replace_keyword(deck, control_ale_word)
    replace_keyword(deck, detonation_word)

    blocks = [
        word("CONTROL_MPP_DECOMPOSITION_DISTRIBUTE_ALE_ELEMENTS"),
        word(
            "CONTROL_BULK_VISCOSITY",
            header("q1", "q2", "type", "btype", "tstype"),
            card(1.5, 0.06, 1, 0, 0, kinds=["f", "f", "i", "i", "i"]),
        ),
        word(
            "CONTROL_HOURGLASS",
            header("ihq", "qh"),
            card(4, 0.1, kinds=["i", "f"]),
        ),
        word(
            "HOURGLASS",
            header("hgid", "ihq", "qm", "ibq", "q1", "q2", "qb/vdc", "qw"),
            card(1, 4, 0.1, 0, 1.5, 0.06, 0.1, 0.1,
                 kinds=["i", "i", "f", "i", "f", "f", "f", "f"]),
        ),
        word(
            "SECTION_SOLID_TITLE",
            card("section-ale", kinds=["s"]),
            header("secid", "elform", "aet"),
            card(4, 11, 0, kinds=["i", "i", "i"]),
        ),
        word(
            "PART",
            card("ale-air", kinds=["s"]),
            header("pid", "secid", "mid", "eosid", "hgid", "grav", "adpopt", "tmid"),
            card(4, 4, 1, 1, 1, 0, 0, 0, kinds=["i"] * 8),
        ),
        word(
            "PART",
            card("ale-tnt", kinds=["s"]),
            header("pid", "secid", "mid", "eosid", "hgid", "grav", "adpopt", "tmid"),
            card(5, 4, 2, 2, 1, 0, 0, 0, kinds=["i"] * 8),
        ),
        word(
            "ALE_MULTI-MATERIAL_GROUP",
            header("sid", "idtype"),
            card(4, 1, kinds=["i", "i"]),
            card(5, 1, kinds=["i", "i"]),
        ),
        word(
            "SET_PART_LIST_TITLE",
            card("ale-domain", kinds=["s"]),
            header("sid"),
            card(100, kinds=["i"]),
            header("pid1", "pid2", "pid3", "pid4", "pid5", "pid6", "pid7", "pid8"),
            card(4, 5, 0, 0, 0, 0, 0, 0, kinds=["i"] * 8),
        ),
        word(
            "CONSTRAINED_LAGRANGE_IN_SOLID",
            header("lstrsid", "alesid", "lstrstyp", "alestyp", "nquad", "ctype", "direc", "mcoup"),
            card(2, 100, 1, 0, 2, 5, 2, 0, kinds=["i"] * 8),
            header("start", "end", "pfac", "fric", "frcmin", "norm", "normtyp", "damp"),
            card(0.0, 1.0e10, 0.1, 0.0, 0.5, 0, 0, 0.0,
                 kinds=["f", "f", "f", "f", "f", "i", "i", "f"]),
            header("k", "hmin", "hmax", "ileak", "pleak", "lcidpor", "nvent", "iblock"),
            card(0.0, 0.0, 0.0, 2, 0.1, 0, 0, 0,
                 kinds=["f", "f", "f", "i", "f", "i", "i", "i"]),
        ),
        word(
            "ALE_COUPLING_NODAL_PENALTY",
            header("strsid", "alesid", "strstyp", "alestyp", "mcoup"),
            card(3, 100, 1, 0, 0, kinds=["i"] * 5),
            header("start", "end", "pform", "pfac", "", "frcmin"),
            card(0.0, 1.0e10, 0, 0.1, "", 0.5,
                 kinds=["f", "f", "i", "f", "s", "f"]),
        ),
        segment_set(1001, face_sets[0]),
        word(
            "BOUNDARY_NON_REFLECTING",
            header("ssid", "ad", "as"),
            card(1001, 0.0, 0.0, kinds=["i", "f", "f"]),
        ),
    ]
    if face_sets[1]:
        blocks.extend([
            segment_set(1002, face_sets[1]),
            word(
                "ALE_ESSENTIAL_BOUNDARY",
                header("id", "idtype", "ictype", "iexcl"),
                card(1002, 2, 2, 0, kinds=["i"] * 4),
            ),
        ])
    blocks.extend([node_block, elem_block])
    replace_keyword(deck, word(
        "DATABASE_BINARY_D3PLOT",
        header("dt", "lcdt", "beam", "npltc", "psetid"),
        card(0.002, 0, 0, 0, 0, kinds=["f", "i", "i", "i", "i"]),
    ))
    before_end(deck, blocks)
    deck.header.extend([
        f"$# general-ALE conversion source={source}",
        f"$# symmetry={symmetry}; explicit ALE mesh {nx}x{ny}x{nz}; "
        f"nodes start {node_base}; elements start {element_base}",
        f"$# detonation large-end center at z={detonation_z:g} mm "
        "(inside first 2 mm ALE layer)",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    deck.write(output)
    print(f"wrote {output} ALE elements={len(elem_block.cards)-1}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--symmetry",
        choices=("full", "half-long", "half-short", "quarter"),
        default="half-long",
    )
    parser.add_argument(
        "--detonation-z",
        type=float,
        default=1.0,
        help="z coordinate at large-end center; 1 mm is the first 2 mm ALE cell center",
    )
    args = parser.parse_args(argv)
    convert(
        args.source,
        args.output,
        symmetry=args.symmetry,
        detonation_z=args.detonation_z,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
