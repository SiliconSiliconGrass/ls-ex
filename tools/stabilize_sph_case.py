"""Create a restartable diagnostic variant of a SPH blast keyword deck."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ls_ex import KCard, KWord, KeywordFile, parse_numbers


def card(*values, kinds=None) -> KCard:
    return KCard(values=list(values), widths=[10] * len(values), kinds=kinds)


def header(*names) -> KCard:
    return KCard.header(list(names), [10] * len(names))


def word(name: str, *cards: KCard) -> KWord:
    return KWord(name, list(cards))


def control_output(failure_frequency: int) -> KWord:
    return word(
        "CONTROL_OUTPUT",
        header("npopt", "neecho", "nrefup", "iaccop", "opifs", "ipnint", "ikedit", "iflush"),
        card(0, 0, 0, 0, 0.0, 0, 100, 5000,
             kinds=["i", "i", "i", "i", "f", "i", "i", "i"]),
        header("iprtf", "ierode", "tet10s8", "msgmax", "ipcurv", "gmdt", "ip1dblt", "eocs"),
        card(0, 0, 2, 50, 0, 0.0, 0, 0,
             kinds=["i", "i", "i", "i", "i", "f", "i", "i"]),
        header("tolev", "newleg", "frfreq", "minfo", "solsig", "msgflg", "cdetol", "igeom"),
        card(2, 0, failure_frequency, 0, 0, 0, 10.0, 1,
             kinds=["i", "i", "i", "i", "i", "i", "f", "i"]),
    )


def control_energy() -> KWord:
    return word(
        "CONTROL_ENERGY",
        header("hgen", "rwlen", "slnten", "rylen", "irgen", "maten", "drlen", "disen"),
        card(2, 2, 2, 1, 1, 2, 1, 1, kinds=["i"] * 8),
    )


def control_ale() -> KWord:
    return word(
        "CONTROL_ALE",
        header("dct", "nadv", "meth", "afac", "bfac", "cfac", "dfac", "efac"),
        card(-1, 1, 1, -1.0, 0.0, 0.0, 0.0, 0.0,
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


def control_ale_mpp() -> KWord:
    return word(
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


def control_timestep() -> KWord:
    return word(
        "CONTROL_TIMESTEP",
        header("dtinit", "tssfac", "isdo", "tslimt", "dt2ms", "lctm", "erode", "ms1st"),
        card(1.0e-4, 0.9, 0, 0.0, 0.0, 0, 0, 0,
             kinds=["f", "f", "i", "f", "f", "i", "i", "i"]),
        header("dt2msf", "dt2mslc", "imscl"),
        card(0.0, 0, 0, kinds=["f", "i", "i"]),
    )


def control_solid() -> KWord:
    return word(
        "CONTROL_SOLID",
        header("esort", "fmatrx", "niptets", "swlocl", "psfail", "t10jtol", "icoh", "tet13k"),
        card(0, 0, 4, 1, 1, 0.0, 0, 0,
             kinds=["i", "i", "i", "i", "i", "f", "i", "i"]),
    )


def plastic_kinematic(failure_strain: float) -> KWord:
    return word(
        "MAT_PLASTIC_KINEMATIC_TITLE",
        card("mat-shell", kinds=["s"]),
        header("mid", "ro", "e", "pr", "sigy", "etan", "beta"),
        card(3, 0.00783, 201000.0, 0.33, 1650.0, 300.0, 0.0,
             kinds=["i"] + ["f"] * 6),
        header("src", "srp", "fs", "vp"),
        card(0.0, 0.0, failure_strain, 0.0, kinds=["f"] * 4),
    )


def structural_part_set() -> KWord:
    return word(
        "SET_PART_LIST_TITLE",
        card("part-structural", kinds=["s"]),
        header("sid"),
        card(1, kinds=["i"]),
        header("pid1", "pid2", "pid3", "pid4", "pid5", "pid6", "pid7", "pid8"),
        card(2, 0, 0, 0, 0, 0, 0, 0, kinds=["i"] * 8),
    )


def contact_eroding(part_set_id: int) -> KWord:
    return word(
        "CONTACT_ERODING_SINGLE_SURFACE",
        header("ssid", "msid", "sstyp", "mstyp", "sboxid", "mboxid", "spr", "mpr"),
        card(part_set_id, 0, 2, 0, 0, 0, 0, 0, kinds=["i"] * 8),
        header("fs", "fd", "dc", "vc", "vdc", "penchk", "bt", "dt"),
        card(0.3, 0.3, 0.0, 0.0, 0.0, 0, 0.0, 0.0,
             kinds=["f", "f", "f", "f", "f", "i", "f", "f"]),
        header("sfs", "sfm", "sst", "mst", "sfst", "sfmt", "fsf", "vsf"),
        card(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, kinds=["f"] * 8),
        header("isym", "erosop", "iadj", "", "", "", "", "unused1"),
        card(0, 1, 1, "", "", "", "", "", kinds=["i", "i", "i"] + ["s"] * 5),
        header("soft", "softscl", "lcidab", "maxpar", "sbopt", "depth", "bsort", "frcfrq"),
        card(1, 0.1, 0, 0, 0, 5, 100, 0,
             kinds=["i", "f", "i", "i", "i", "i", "i", "i"]),
    )


def control_termination(dtmin: float) -> KWord:
    return word(
        "CONTROL_TERMINATION",
        header("endtim", "endcyc", "dtmin", "endeng", "endmas", "nosol"),
        card(0.2, 0, dtmin, 0.0, 1.0e8, 0,
             kinds=["f", "i", "f", "f", "f", "i"]),
    )


def d3plot_output(dt: float) -> KWord:
    return word(
        "DATABASE_BINARY_D3PLOT",
        header("dt", "lcdt", "beam", "npltc", "psetid"),
        card(dt, 0, 0, 0, 0, kinds=["f", "i", "i", "i", "i"]),
    )


def ale_nodal_coupling(start: float, pfac: float, frcmin: float) -> KWord:
    return word(
        "ALE_COUPLING_NODAL_PENALTY",
        header("strsid", "alesid", "strstyp", "alestyp", "mcoup"),
        card(3, 1, 1, 1, -1, kinds=["i"] * 5),
        header("start", "end", "pform", "pfac", "", "frcmin"),
        card(start, 1.0e10, 0, pfac, "", frcmin,
             kinds=["f", "f", "i", "f", "s", "f"]),
    )


def replace_keyword(deck: KeywordFile, replacement: KWord) -> None:
    for index, keyword in enumerate(deck.keywords):
        if keyword.name.upper() == replacement.name.upper():
            deck.keywords[index] = replacement
            return
    end_index = next(i for i, keyword in enumerate(deck.keywords)
                     if keyword.name.upper() == "END")
    deck.keywords.insert(end_index, replacement)


def stabilize(
    source: Path,
    output: Path,
    d3plot_dt: float,
    failure_frequency: int,
    coupling_start: float,
    coupling_pfac: float,
    coupling_frcmin: float,
    solid_dtmin: float,
    failure_strain: float | None,
    preserve_ale: bool,
    contact_part_set: bool,
    mpp_compatible: bool,
    minimal_mpp: bool,
) -> None:
    deck = KeywordFile.load(source)
    if mpp_compatible or minimal_mpp:
        replace_keyword(deck, control_ale_mpp())
    elif not preserve_ale:
        replace_keyword(deck, control_ale())

    if minimal_mpp:
        replace_keyword(deck, d3plot_output(d3plot_dt))
        deck.header.extend([
            f"$# mpp minimal source={source}",
            f"$# mpp minimal d3plot_dt={d3plot_dt:g}; no stabilization cards added",
        ])
        output.parent.mkdir(parents=True, exist_ok=True)
        deck.write(output)
        print(f"wrote {output}")
        return

    if failure_strain is not None:
        replace_keyword(deck, plastic_kinematic(failure_strain))
    replace_keyword(deck, control_output(failure_frequency))
    replace_keyword(deck, control_energy())
    replace_keyword(deck, control_timestep())
    if solid_dtmin > 0 or contact_part_set:
        replace_keyword(deck, structural_part_set())
    if solid_dtmin > 0:
        replace_keyword(deck, control_solid())
        replace_keyword(deck, control_termination(solid_dtmin))
    if contact_part_set:
        replace_keyword(deck, contact_eroding(1))
    replace_keyword(deck, d3plot_output(d3plot_dt))
    replace_keyword(deck, ale_nodal_coupling(
        coupling_start, coupling_pfac, coupling_frcmin
    ))

    deck.header.extend([
        f"$# stabilized_source={source}",
        f"$# stabilized d3plot_dt={d3plot_dt:g} frfreq={failure_frequency}",
        f"$# stabilized sph_ale start={coupling_start:g} "
        f"pfac={coupling_pfac:g} frcmin={coupling_frcmin:g}",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    deck.write(output)
    print(f"wrote {output}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--d3plot-dt", type=float, default=0.01)
    parser.add_argument("--failure-frequency", type=int, default=100)
    parser.add_argument("--coupling-start", type=float, default=0.0)
    parser.add_argument("--coupling-pfac", type=float, default=0.1)
    parser.add_argument("--coupling-frcmin", type=float, default=0.5)
    parser.add_argument("--solid-dtmin", type=float, default=0.0)
    parser.add_argument("--failure-strain", type=float)
    parser.add_argument("--preserve-ale", action="store_true")
    parser.add_argument("--contact-part-set", action="store_true")
    parser.add_argument("--mpp-compatible", action="store_true")
    parser.add_argument("--minimal-mpp", action="store_true")
    args = parser.parse_args(argv)

    stabilize(
        args.source,
        args.output,
        args.d3plot_dt,
        args.failure_frequency,
        args.coupling_start,
        args.coupling_pfac,
        args.coupling_frcmin,
        args.solid_dtmin,
        args.failure_strain,
        args.preserve_ale,
        args.contact_part_set,
        args.mpp_compatible,
        args.minimal_mpp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
