"""Voxelize an STL or GLB and optionally build a General ALE blast deck."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ls_ex import (  # noqa: E402
    VoxelGeometry,
    air_ale_voxel_count,
    build_general_ale_blast_deck,
    build_geometry_keyword_file,
    load_surface_mesh,
    surface_is_watertight,
    voxelize_surface,
)


def _detonation(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.replace(" ", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("detonation must be x,y,z")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as error:
        raise argparse.ArgumentTypeError("detonation must contain numbers") from error


def _print_summary(geometry: VoxelGeometry, output: Path) -> None:
    print(
        f"wrote {output}: mode={geometry.mode} "
        f"element_size={geometry.element_size:g} "
        f"grid={geometry.divisions[0]}x{geometry.divisions[1]}x{geometry.divisions[2]} "
        f"voxels={geometry.voxel_count}"
    )
    for region in geometry.regions:
        print(f"  PID {region.pid} {region.title}: {len(region.cells)} voxels")
    if geometry.open_surface:
        print("  non-watertight input: open surface patches are structural shell")
    if geometry.mode == "shell":
        detonation = geometry.detonation_point(3)
        print(
            f"  shell layers={geometry.shell_layers} "
            f"effective thickness={geometry.effective_shell_thickness:g}"
        )
        print(
            f"  default detonation=({detonation[0]:g}, "
            f"{detonation[1]:g}, {detonation[2]:g})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Voxelize a closed STL/GLB into equal cubic elements. "
            "solid mode creates one part; shell mode creates shell + interior."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("solid", "shell"), default="shell")
    parser.add_argument(
        "--element-size",
        type=float,
        help="cube edge length; default longest model dimension / 40",
    )
    parser.add_argument(
        "--shell-thickness",
        type=float,
        help="requested shell thickness; rounded up to whole voxel layers",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="uniformly scale model coordinates before voxelization",
    )
    parser.add_argument(
        "--simulation-output",
        type=Path,
        help="also write a General ALE blast deck that includes the geometry file",
    )
    parser.add_argument(
        "--detonation",
        type=_detonation,
        help="TNT detonation x,y,z; defaults to the central explosive voxel center",
    )
    parser.add_argument("--end-time", type=float, default=0.2)
    parser.add_argument(
        "--air-margin",
        type=float,
        default=20.0,
        help="air ALE margin around the model in model length units",
    )
    parser.add_argument("--max-voxels", type=int, default=2_000_000)
    args = parser.parse_args(argv)

    if args.simulation_output is not None:
        if args.mode != "shell":
            parser.error("--simulation-output requires --mode shell")
        if args.simulation_output.parent.resolve() != args.output.parent.resolve():
            parser.error(
                "--simulation-output and --output must be in the same directory "
                "so the include path stays portable"
            )
        if args.simulation_output.resolve() == args.output.resolve():
            parser.error("--simulation-output must differ from --output")

    surface = load_surface_mesh(args.source).scaled(args.scale)
    if args.mode == "solid" and not surface_is_watertight(surface):
        parser.error(
            "the input surface is not watertight; use --mode shell so open "
            "surface patches become structural shell"
        )
    geometry = voxelize_surface(
        surface,
        args.element_size,
        mode=args.mode,
        shell_thickness=args.shell_thickness,
        max_voxels=args.max_voxels,
    )
    title = args.title or f"{args.source.stem} voxel geometry"
    if args.scale != 1.0:
        title += f" (scale {args.scale:g}x)"
    geometry_deck = build_geometry_keyword_file(geometry, title=title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    geometry_deck.write(args.output)
    _print_summary(geometry, args.output)

    if args.simulation_output is not None:
        simulation = build_general_ale_blast_deck(
            geometry,
            args.output.name,
            title=f"{args.source.stem} General ALE TNT blast",
            detonation=args.detonation,
            end_time=args.end_time,
            air_margin=args.air_margin,
        )
        simulation.write(args.simulation_output)
        air_voxels = air_ale_voxel_count(geometry, args.air_margin)
        print(
            f"wrote {args.simulation_output}: General ALE, no SPH, "
            f"air_ALE_voxels={air_voxels}, includes {args.output.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
