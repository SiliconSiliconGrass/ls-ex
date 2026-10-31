"""Read-only diagnostics for the two elliptical warhead simulations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lasso.dyna import ArrayType, D3plot


ROOT = Path(r"D:\MyStuff\Research\sim-playground\elliptical20261030")
STATES = set(range(0, 102, 10)) | {101}


def quantiles(values: np.ndarray) -> np.ndarray:
    if not values.size:
        return np.full(7, np.nan)
    return np.quantile(values, [0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])


def inspect_no_sph(path: Path) -> None:
    reader = D3plot(
        str(path),
        state_filter=STATES,
        buffered_reading=True,
        state_array_filter=[
            ArrayType.global_timesteps,
            ArrayType.element_solid_is_alive,
            ArrayType.node_velocity,
            ArrayType.node_displacement,
        ],
    )
    alive = reader.arrays[ArrayType.element_solid_is_alive]
    part_indexes = reader.arrays[ArrayType.element_solid_part_indexes]
    node_indexes = reader.arrays[ArrayType.element_solid_node_indexes]
    velocity = reader.arrays[ArrayType.node_velocity]
    times = reader.arrays[ArrayType.global_timesteps]
    shell_elements = np.flatnonzero(part_indexes == 0)
    shell_nodes = node_indexes[shell_elements]

    print("\nNO SPH")
    for frame, time in enumerate(times):
        frame_alive = alive[frame, shell_elements] > 0.5
        nodes = np.unique(shell_nodes[frame_alive])
        speeds = np.linalg.norm(velocity[frame, nodes], axis=1)
        print(
            f"{frame:3d} t={time:.6f} ms "
            f"alive_elements={frame_alive.sum():6d} "
            f"nodes={len(nodes):6d} "
            f"speed_q={np.round(quantiles(speeds), 4)}"
        )


def inspect_sph(path: Path) -> None:
    reader = D3plot(
        str(path),
        state_filter=STATES,
        buffered_reading=True,
        state_array_filter=[
            ArrayType.global_timesteps,
            ArrayType.sph_deletion,
            ArrayType.sph_radius,
            ArrayType.sph_mass,
            ArrayType.node_velocity,
            ArrayType.node_displacement,
        ],
    )
    arrays = reader.arrays
    sph_nodes = arrays[ArrayType.sph_node_indexes]
    deletion = arrays[ArrayType.sph_deletion]
    radius = arrays[ArrayType.sph_radius]
    mass = arrays[ArrayType.sph_mass]
    velocity = arrays[ArrayType.node_velocity][:, sph_nodes, :]
    times = arrays[ArrayType.global_timesteps]
    print("\nSPH")
    print(
        "material index counts:",
        np.unique(arrays[ArrayType.sph_node_material_index], return_counts=True),
    )
    for frame, time in enumerate(times):
        speeds = np.linalg.norm(velocity[frame], axis=1)
        positive_radius = radius[frame] > 0.0
        active = positive_radius & (mass[frame] > 0.0) & (deletion[frame] == 0.0)
        finite = active & np.isfinite(speeds)
        plausible = finite & (speeds <= 20.0)
        print(
            f"{frame:3d} t={time:.6f} ms "
            f"radius>0={positive_radius.sum():5d} "
            f"active={active.sum():5d} "
            f"finite={finite.sum():5d} "
            f"speed<=20={plausible.sum():5d} "
            f"speed_q={np.round(quantiles(speeds[finite]), 3)}"
        )


def main() -> None:
    inspect_no_sph(ROOT / "no-sph-quarter" / "d3plot")
    inspect_sph(ROOT / "sph-quarter" / "d3plot")


if __name__ == "__main__":
    main()
