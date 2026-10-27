"""Create an erosion-only general-ALE deck by removing all SPH definitions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ls_ex import KeywordFile, KWord

from convert_to_general_ale import card, header, replace_keyword, word


SPH_KEYWORDS = {
    "BOUNDARY_SPH_SYMMETRY_PLANE",
    "CONTROL_SPH",
    "DEFINE_ADAPTIVE_SOLID_TO_SPH_ID",
}


def first_title(keyword: KWord) -> str:
    for card_ in keyword.cards:
        text = card_.render().strip()
        if text and not text.startswith("$"):
            return text.lower()
    return ""


def is_sph_keyword(keyword: KWord) -> bool:
    name = keyword.name.upper()
    title = first_title(keyword)
    return (
        name in SPH_KEYWORDS
        or name.startswith("SECTION_SPH")
        or (name.startswith("MAT_") and title == "mat-sph")
        or (name == "PART" and title == "part-sph")
    )


def first_integer_card(keyword: KWord) -> list[int] | None:
    for card_ in keyword.data_cards:
        try:
            return [int(value) for value in card_.render().split()]
        except ValueError:
            continue
    return None


def sph_part_id(deck: KeywordFile) -> int | None:
    for keyword in deck.keywords:
        if keyword.name.upper() == "PART" and first_title(keyword) == "part-sph":
            values = first_integer_card(keyword)
            return values[0] if values else None
    return None


def couples_sph_part_to_ale(keyword: KWord, deleted_part_id: int | None) -> bool:
    if deleted_part_id is None:
        return False
    if keyword.name.upper() != "ALE_COUPLING_NODAL_PENALTY":
        return False
    values = first_integer_card(keyword)
    if values is None:
        return False
    strsid, strstyp = int(values[0]), int(values[2])
    return strstyp == 1 and strsid == deleted_part_id


def remove_sph(source: Path, output: Path) -> None:
    deck = KeywordFile.load(source)
    deleted_part_id = sph_part_id(deck)
    removed = [
        keyword.name
        for keyword in deck.keywords
        if is_sph_keyword(keyword)
        or couples_sph_part_to_ale(keyword, deleted_part_id)
    ]
    deck.keywords = [
        keyword
        for keyword in deck.keywords
        if not is_sph_keyword(keyword)
        and not couples_sph_part_to_ale(keyword, deleted_part_id)
    ]

    replace_keyword(
        deck,
        word(
            "DATABASE_BINARY_D3PLOT",
            header("dt", "lcdt", "beam", "npltc", "psetid"),
            card(0.002, 0, 0, 0, 0, kinds=["f", "i", "i", "i", "i"]),
        ),
    )
    deck.header.extend([
        f"$# no-SPH general-ALE conversion source={source}",
        "$# solid failure directly deletes elements; no SPH particle conversion",
        f"$# removed SPH keywords: {', '.join(removed)}",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    deck.write(output)
    print(f"wrote {output}; removed {len(removed)} SPH keyword blocks")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    remove_sph(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
