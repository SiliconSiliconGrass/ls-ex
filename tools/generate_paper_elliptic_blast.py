"""Export the four paper symmetry models in SPH and no-SPH variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "tools"))

from ls_ex import split_geometry_include  # noqa: E402
from build_elliptic_frustum_blast import build_deck  # noqa: E402
from convert_to_general_ale import convert  # noqa: E402
from generate_blast_grid_study import append_before_end, sph_words  # noqa: E402
from make_no_sph_general_ale import remove_sph  # noqa: E402
from stabilize_sph_case import stabilize  # noqa: E402


SYMMETRIES = ("full", "half-long", "half-short", "quarter")


def build_case(
    output_dir: Path,
    scratch_dir: Path,
    symmetry: str,
) -> tuple[Path, Path]:
    """Build one SPH model, its no-SPH counterpart, and a shared geometry."""
    suffix = symmetry
    deck = build_deck(
        element_size=2.0,
        fill="cylinder",
        symmetry=symmetry,
    )
    append_before_end(deck, sph_words(symmetry))
    deck.header.extend([
        "$# paper: Zhang Lu et al., 2025, elliptic variable-section warhead",
        "$# units: mm-g-ms; pressure=MPa; shell=30CrMnSi2A",
        "$# paper-table-unit-fix: sigma0=1650 MPa (the printed GPa header is invalid)",
        "$# paper-table-unit-fix: D_CJ=6487 mm/ms (1 m/s = 1 mm/ms)",
        "$# detonation: explosive large-end face center (0,0,0)",
    ])

    source = scratch_dir / f"sph_{suffix}_sale.k"
    stabilized = scratch_dir / f"sph_{suffix}_stabilized.k"
    sph_combined = scratch_dir / f"sph_{suffix}_general_ale.k"
    no_sph_combined = scratch_dir / f"no_sph_{suffix}_general_ale.k"
    deck.write(source)

    stabilize(
        source=source,
        output=stabilized,
        d3plot_dt=0.002,
        failure_frequency=100,
        coupling_start=0.0,
        coupling_pfac=0.1,
        coupling_frcmin=0.5,
        solid_dtmin=0.1,
        failure_strain=0.2,
        preserve_ale=False,
        contact_part_set=True,
        mpp_compatible=True,
        minimal_mpp=False,
    )
    convert(stabilized, sph_combined, symmetry=symmetry)
    remove_sph(sph_combined, no_sph_combined)

    geometry_name = f"geometry_{suffix}.k"
    geometry_path = output_dir / geometry_name
    sph_entry = output_dir / f"paper_sph_{suffix}.k"
    no_sph_entry = output_dir / f"paper_no-sph_{suffix}.k"
    split_geometry_include(
        sph_combined,
        geometry_path,
        sph_entry,
        include_name=geometry_name,
    )
    split_geometry_include(
        no_sph_combined,
        geometry_path,
        no_sph_entry,
        include_name=geometry_name,
        write_geometry=False,
    )
    return sph_entry, no_sph_entry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="generate_paper_elliptic_blast",
        description=(
            "Generate SPH/no-SPH full, half-long, half-short, and quarter "
            "decks with paper material data and included geometry"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "paper-elliptic-blast",
    )
    parser.add_argument(
        "--symmetry",
        choices=SYMMETRIES,
        action="append",
        help="restrict the export to one or more symmetry variants",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = tuple(args.symmetry or SYMMETRIES)
    with TemporaryDirectory(prefix="ls-ex-paper-blast-") as temporary:
        scratch = Path(temporary)
        for symmetry in requested:
            print(f"[{symmetry}] building SPH deck", flush=True)
            sph_entry, no_sph_entry = build_case(output_dir, scratch, symmetry)
            print(f"  wrote {sph_entry}", flush=True)
            print(f"  wrote {no_sph_entry}", flush=True)
            print(f"  shared include {output_dir / f'geometry_{symmetry}.k'}",
                  flush=True)
    print(f"exported {2 * len(requested)} entry decks to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
