"""Generate 1 mm and ADAPT_TO_SPH blast variants without changing core code."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from ls_ex import KCard, KWord, KeywordFile  # noqa: E402
from build_elliptic_frustum_blast import build_deck, symmetry_cut_axes  # noqa: E402


SYMMETRIES = ("full", "half-long", "half-short", "quarter")


def card(*values, kinds=None) -> KCard:
    return KCard(values=list(values), widths=[10] * len(values), kinds=kinds)


def header(*names) -> KCard:
    return KCard.header(list(names), [10] * len(names))


def word(name: str, *cards: KCard) -> KWord:
    return KWord(name, list(cards))


def sph_words(symmetry: str) -> list[KWord]:
    """Return the complete SPH conversion, ALE coupling, and symmetry setup."""
    blocks = [
        word(
            "MAT_PLASTIC_KINEMATIC_TITLE",
            card("mat-sph", kinds=["s"]),
            header("mid", "ro", "e", "pr", "sigy", "etan", "beta"),
            card(4, 0.00783, 210000.0, 0.33, 1650.0, 300.0, 0.0,
                 kinds=["i"] + ["f"] * 6),
            header("src", "srp", "fs", "vp"),
            card(0.0, 0.0, 0.0, 0.0, kinds=["f"] * 4),
        ),
        word(
            "SECTION_SPH_TITLE",
            card("section-sph", kinds=["s"]),
            header("secid", "cslh", "hmin", "hmax", "sphini", "death", "start", "sphkern"),
            card(2, 1.2, 0.2, 2.0, 0.0, 1.0e20, 0.0, 0,
                 kinds=["i"] + ["f"] * 6 + ["i"]),
        ),
        word(
            "PART",
            card("part-sph", kinds=["s"]),
            header("pid", "secid", "mid", "eosid", "hgid", "grav", "adpopt", "tmid"),
            card(3, 2, 4, 0, 0, 0, 0, 0, kinds=["i"] * 8),
        ),
        word(
            "DEFINE_ADAPTIVE_SOLID_TO_SPH_ID",
            header("did", "heading"),
            card(1, "adaptive-solid-to-sph", kinds=["i", "s"]),
            header("ipid", "itype", "nq", "ipsph", "issph", "icpl", "iopt", "cpcd"),
            card(2, 0, 1, 3, 2, 0, 1, 0.0,
                 kinds=["i"] * 7 + ["f"]),
        ),
        word(
            "CONTROL_SPH",
            header("ncbs", "boxid", "dt", "idim", "nmneigh", "form", "start", "maxv"),
            card(1, 0, 1.0e20, 3, 150, 1, 0.0, 0.0,
                 kinds=["i", "i", "f", "i", "i", "i", "f", "f"]),
            header("cont", "deriv", "ini", "ishow", "ierod", "icont", "iavis", "isymp"),
            card(0, 0, 0, 1, 0, 0, 0, 100, kinds=["i"] * 8),
        ),
        word(
            "ALE_COUPLING_NODAL_PENALTY",
            header("strsid", "alesid", "strstyp", "alestyp", "mcoup"),
            card(3, 1, 1, 1, -1, kinds=["i"] * 5),
            header("start", "end", "pform", "pfac", "frcmin"),
            card(0.0, 1.0e10, 0, 0.1, "", 0.5,
                 kinds=["f", "f", "i", "f", "s", "f"]),
        ),
    ]

    for axis in symmetry_cut_axes(symmetry):
        vector = {
            "x": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            "y": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }[axis]
        blocks.append(word(
            "BOUNDARY_SPH_SYMMETRY_PLANE",
            header("vtx", "vty", "vtz", "vhx", "vhy", "vhz"),
            card(*vector, kinds=["f"] * 6),
        ))
    return blocks


def append_before_end(deck: KeywordFile, blocks: Sequence[KWord]) -> None:
    end_index = next(
        index for index, keyword in enumerate(deck.keywords)
        if keyword.name.upper() == "END"
    )
    deck.keywords[end_index:end_index] = list(blocks)


def generate_case(outdir: Path, element_size: float, symmetry: str, adapt_sph: bool) -> Path:
    deck = build_deck(element_size=element_size, fill="cylinder", symmetry=symmetry)
    variant = "sph" if adapt_sph else "solid"
    deck.header.append(
        f"$# study={variant} element_size={element_size:g} symmetry={symmetry}"
    )
    if adapt_sph:
        append_before_end(deck, sph_words(symmetry))

    suffix = "full" if symmetry == "full" else symmetry
    path = outdir / f"elliptic_frustum_blast_{variant}_{suffix}.k"
    path.parent.mkdir(parents=True, exist_ok=True)
    deck.write(path)
    print(
        f"wrote {path} nodes={len(deck.nodes())} solids={len(deck.solid_elements())} "
        f"adapt_sph={adapt_sph}"
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=ROOT / "outputs" / "blast_grid_study"
    )
    args = parser.parse_args(argv)

    for symmetry in SYMMETRIES:
        generate_case(args.root / "grid_1mm", 1.0, symmetry, adapt_sph=False)
    for symmetry in SYMMETRIES:
        generate_case(args.root / "grid_2mm_adapt_sph", 2.0, symmetry, adapt_sph=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
