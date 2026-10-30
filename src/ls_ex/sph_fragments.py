"""Read, filter, analyze, and visualize SPH fragments from LS-DYNA d3plot.

The d3plot stores initial nodal coordinates and nodal displacement separately.
For a selected state, the current SPH position is therefore::

    position = node_coordinates + node_displacement[state]

``sph_node_indexes`` is a zero-based lookup into the nodal arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SPHParticleData:
    """SPH particle arrays for one d3plot state."""

    positions: np.ndarray
    velocities: np.ndarray
    speeds: np.ndarray
    node_ids: np.ndarray
    particle_indices: np.ndarray
    material_indices: np.ndarray
    alive: np.ndarray
    time: float

    def __len__(self) -> int:
        return len(self.speeds)

    def take(self, indexes: np.ndarray) -> "SPHParticleData":
        """Return a particle subset, keeping all metadata aligned."""

        return SPHParticleData(
            positions=self.positions[indexes],
            velocities=self.velocities[indexes],
            speeds=self.speeds[indexes],
            node_ids=self.node_ids[indexes],
            particle_indices=self.particle_indices[indexes],
            material_indices=self.material_indices[indexes],
            alive=self.alive[indexes],
            time=self.time,
        )


@dataclass(frozen=True)
class OutlierFilterReport:
    """Details of the particle mask applied before statistics."""

    method: str
    upper_speed: float | None
    lower_speed: float | None
    input_count: int
    alive_count: int
    finite_count: int
    output_count: int
    excluded_count: int
    excluded_speed_min: float | None
    excluded_speed_max: float | None


@dataclass(frozen=True)
class AngularStatistics:
    """Per-angle-bin fragment count and speed distribution."""

    bin_edges_deg: np.ndarray
    counts: np.ndarray
    fractions: np.ndarray
    speed_min: np.ndarray
    speed_mean: np.ndarray
    speed_std: np.ndarray
    speed_p05: np.ndarray
    speed_median: np.ndarray
    speed_p95: np.ndarray
    speed_max: np.ndarray

    @property
    def centers_deg(self) -> np.ndarray:
        return (self.bin_edges_deg[:-1] + self.bin_edges_deg[1:]) / 2.0


@dataclass(frozen=True)
class AngularSpeedHistogram:
    """Two-dimensional angular/velocity histogram."""

    angle_edges_deg: np.ndarray
    speed_edges: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class SPHParticleSequence:
    """Position and velocity time history for a fixed SPH particle subset."""

    positions: np.ndarray
    velocities: np.ndarray
    speeds: np.ndarray
    alive: np.ndarray
    times: np.ndarray
    node_ids: np.ndarray
    particle_indices: np.ndarray

    @property
    def n_frames(self) -> int:
        return len(self.times)

    @property
    def n_particles(self) -> int:
        return len(self.node_ids)


def load_sph_particle_data(
    d3plot_path: str | Path,
    *,
    state: int = -1,
    time: float | None = None,
) -> SPHParticleData:
    """Load SPH position and velocity arrays from an LS-DYNA d3plot.

    Only the selected state's displacement and velocity arrays are retained.
    If ``time`` is supplied, the state closest to that time is selected and
    ``state`` is ignored.
    """

    from lasso.dyna import ArrayType, D3plot

    path = Path(d3plot_path)
    if not path.is_file():
        raise FileNotFoundError(f"d3plot not found: {path}")

    selected_state = state
    if time is not None:
        timestep_reader = D3plot(
            str(path),
            state_array_filter=[ArrayType.global_timesteps],
        )
        timesteps = np.asarray(
            timestep_reader.arrays[ArrayType.global_timesteps], dtype=float
        )
        if timesteps.size == 0:
            raise ValueError(f"no states found in {path}")
        selected_state = int(np.argmin(np.abs(timesteps - float(time))))

    reader = D3plot(
        str(path),
        state_filter={selected_state},
        state_array_filter=[
            ArrayType.global_timesteps,
            ArrayType.node_displacement,
            ArrayType.node_velocity,
            ArrayType.sph_deletion,
        ],
        buffered_reading=True,
    )
    arrays = reader.arrays

    required = (
        ArrayType.global_timesteps,
        ArrayType.node_displacement,
        ArrayType.node_velocity,
        ArrayType.sph_deletion,
        ArrayType.node_coordinates,
        ArrayType.sph_node_indexes,
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(
            "d3plot does not contain the expected SPH/node arrays: "
            + ", ".join(missing)
        )

    sph_indexes = np.asarray(arrays[ArrayType.sph_node_indexes], dtype=np.int64)
    if sph_indexes.size == 0:
        raise ValueError(f"d3plot contains no SPH particles: {path}")

    node_coordinates = np.asarray(arrays[ArrayType.node_coordinates], dtype=float)
    if np.any(sph_indexes < 0) or np.any(sph_indexes >= len(node_coordinates)):
        raise ValueError("sph_node_indexes contains an out-of-range node index")

    displacement = _single_state_array(arrays[ArrayType.node_displacement])
    velocity = _single_state_array(arrays[ArrayType.node_velocity])
    deletion = _single_state_array(arrays[ArrayType.sph_deletion])
    timesteps = np.asarray(arrays[ArrayType.global_timesteps], dtype=float)

    current_positions = node_coordinates[sph_indexes] + displacement[sph_indexes]
    current_velocities = velocity[sph_indexes]
    speeds = np.linalg.norm(current_velocities, axis=1)

    node_ids = np.asarray(
        arrays.get(ArrayType.node_ids, sph_indexes + 1), dtype=np.int64
    )[sph_indexes]
    material_indices = np.asarray(
        arrays.get(
            ArrayType.sph_node_material_index,
            np.zeros(len(sph_indexes), dtype=np.int64),
        ),
        dtype=np.int64,
    )

    return SPHParticleData(
        positions=current_positions,
        velocities=current_velocities,
        speeds=speeds,
        node_ids=node_ids,
        particle_indices=sph_indexes,
        material_indices=material_indices,
        alive=deletion == 0,
        time=float(timesteps[0]) if timesteps.size else float("nan"),
    )


def load_sph_particle_sequence(
    d3plot_path: str | Path,
    *,
    states: Sequence[int] | None = None,
    max_frames: int = 60,
    particle_indices: Sequence[int] | None = None,
) -> SPHParticleSequence:
    """Load a fixed SPH particle subset over multiple d3plot states.

    ``particle_indices`` are zero-based node-array indexes from
    ``sph_node_indexes``. If omitted, every SPH particle is loaded. States are
    evenly sampled across the complete d3plot history unless explicitly given.
    """

    from lasso.dyna import ArrayType, D3plot

    path = Path(d3plot_path)
    if not path.is_file():
        raise FileNotFoundError(f"d3plot not found: {path}")
    if max_frames < 1:
        raise ValueError("max_frames must be positive")

    timestep_reader = D3plot(
        str(path),
        state_array_filter=[ArrayType.global_timesteps],
    )
    all_times = np.asarray(
        timestep_reader.arrays[ArrayType.global_timesteps], dtype=float
    )
    if all_times.size == 0:
        raise ValueError(f"no states found in {path}")

    if states is None:
        requested_states = np.unique(
            np.rint(np.linspace(0, len(all_times) - 1, min(max_frames, len(all_times))))
            .astype(np.int64)
        )
    else:
        requested_states = np.asarray(states, dtype=np.int64)
        if requested_states.ndim != 1:
            raise ValueError("states must be a one-dimensional sequence")
    if requested_states.size == 0:
        raise ValueError("at least one state must be selected")
    if np.any(requested_states < -len(all_times)) or np.any(requested_states >= len(all_times)):
        raise IndexError("selected state is outside the d3plot history")
    normalized_states = np.where(
        requested_states < 0,
        requested_states + len(all_times),
        requested_states,
    )

    first = load_sph_particle_data(path, state=int(normalized_states[0]))
    if particle_indices is None:
        lookup = np.arange(len(first), dtype=np.int64)
    else:
        requested_particles = np.unique(np.asarray(particle_indices, dtype=np.int64))
        lookup = np.searchsorted(first.particle_indices, requested_particles)
        if (
            np.any(lookup >= len(first.particle_indices))
            or not np.array_equal(first.particle_indices[lookup], requested_particles)
        ):
            raise ValueError("particle_indices contains an unknown SPH particle")

    selected = first.take(lookup)
    n_frames = len(normalized_states)
    n_particles = len(selected)
    positions = np.full((n_frames, n_particles, 3), np.nan, dtype=float)
    velocities = np.full((n_frames, n_particles, 3), np.nan, dtype=float)
    speeds = np.full((n_frames, n_particles), np.nan, dtype=float)
    alive = np.zeros((n_frames, n_particles), dtype=bool)
    times = np.empty(n_frames, dtype=float)

    for frame_index, state in enumerate(normalized_states):
        particles = first if frame_index == 0 else load_sph_particle_data(
            path,
            state=int(state),
        )
        if not np.array_equal(particles.particle_indices, first.particle_indices):
            raise RuntimeError("SPH particle ordering changed between d3plot states")
        frame_particles = particles.take(lookup)
        positions[frame_index] = frame_particles.positions
        velocities[frame_index] = frame_particles.velocities
        speeds[frame_index] = frame_particles.speeds
        alive[frame_index] = frame_particles.alive
        times[frame_index] = frame_particles.time

    return SPHParticleSequence(
        positions=positions,
        velocities=velocities,
        speeds=speeds,
        alive=alive,
        times=times,
        node_ids=selected.node_ids,
        particle_indices=selected.particle_indices,
    )


def filter_sph_outliers(
    particles: SPHParticleData,
    *,
    method: str = "tukey",
    iqr_factor: float = 3.0,
    mad_threshold: float = 6.0,
    percentile: float = 99.0,
    speed_min: float | None = None,
    speed_max: float | None = None,
) -> tuple[SPHParticleData, OutlierFilterReport]:
    """Remove dead, non-finite, and upper-tail abnormal-speed particles.

    ``tukey`` uses Q3 + iqr_factor * IQR. ``mad`` uses the median plus
    mad_threshold scaled median absolute deviations. ``percentile`` uses the
    requested speed percentile. ``speed_min`` and ``speed_max`` are optional
    hard limits applied in addition to the selected method.
    """

    speeds = np.asarray(particles.speeds, dtype=float)
    finite = (
        particles.alive
        & np.isfinite(speeds)
        & np.all(np.isfinite(particles.positions), axis=1)
        & np.all(np.isfinite(particles.velocities), axis=1)
    )
    input_count = len(particles)
    alive_count = int(np.count_nonzero(particles.alive))
    finite_count = int(np.count_nonzero(finite))

    upper_speed: float | None = None
    valid_speeds = speeds[finite]
    if valid_speeds.size:
        normalized_method = method.lower()
        if normalized_method == "tukey":
            q1, q3 = np.quantile(valid_speeds, [0.25, 0.75])
            upper_speed = float(q3 + iqr_factor * (q3 - q1))
        elif normalized_method == "mad":
            median = float(np.median(valid_speeds))
            mad = float(np.median(np.abs(valid_speeds - median)))
            upper_speed = median + mad_threshold * 1.4826 * mad
        elif normalized_method == "percentile":
            if not 0.0 < percentile <= 100.0:
                raise ValueError("percentile must be in (0, 100]")
            upper_speed = float(np.percentile(valid_speeds, percentile))
        elif normalized_method != "none":
            raise ValueError(
                "method must be one of: tukey, mad, percentile, none"
            )
    else:
        normalized_method = method.lower()
        if normalized_method not in {"tukey", "mad", "percentile", "none"}:
            raise ValueError(
                "method must be one of: tukey, mad, percentile, none"
            )

    if speed_max is not None:
        upper_speed = (
            float(speed_max)
            if upper_speed is None
            else min(upper_speed, float(speed_max))
        )

    keep = finite.copy()
    if upper_speed is not None:
        keep &= speeds <= upper_speed
    if speed_min is not None:
        keep &= speeds >= float(speed_min)

    excluded = finite & ~keep
    excluded_speeds = speeds[excluded]
    report = OutlierFilterReport(
        method=normalized_method,
        upper_speed=upper_speed,
        lower_speed=None if speed_min is None else float(speed_min),
        input_count=input_count,
        alive_count=alive_count,
        finite_count=finite_count,
        output_count=int(np.count_nonzero(keep)),
        excluded_count=int(np.count_nonzero(excluded)),
        excluded_speed_min=(
            float(np.min(excluded_speeds)) if excluded_speeds.size else None
        ),
        excluded_speed_max=(
            float(np.max(excluded_speeds)) if excluded_speeds.size else None
        ),
    )
    return particles.take(np.flatnonzero(keep)), report


def particle_angles_deg(
    particles: SPHParticleData,
    *,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    mode: str = "velocity",
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return the polar angle of each fragment direction relative to ``axis``.

    ``mode="velocity"`` measures the velocity direction. ``mode="position"``
    measures the position vector from ``origin``. Both are in degrees, with
    0 degrees along the positive axis and 180 degrees opposite to it.
    """

    axis_vector = np.asarray(axis, dtype=float)
    if axis_vector.shape != (3,) or not np.all(np.isfinite(axis_vector)):
        raise ValueError("axis must contain three finite values")
    axis_norm = float(np.linalg.norm(axis_vector))
    if axis_norm == 0.0:
        raise ValueError("axis must be non-zero")
    unit_axis = axis_vector / axis_norm

    if mode == "velocity":
        directions = particles.velocities
    elif mode == "position":
        origin_vector = np.asarray(origin, dtype=float)
        if origin_vector.shape != (3,) or not np.all(np.isfinite(origin_vector)):
            raise ValueError("origin must contain three finite values")
        directions = particles.positions - origin_vector
    else:
        raise ValueError("mode must be 'velocity' or 'position'")

    magnitudes = np.linalg.norm(directions, axis=1)
    usable = np.isfinite(magnitudes) & (magnitudes > 0.0)
    cosine = np.full(len(directions), np.nan, dtype=float)
    cosine[usable] = np.clip(
        directions[usable] @ unit_axis / magnitudes[usable], -1.0, 1.0
    )
    return np.degrees(np.arccos(cosine))


def make_angle_bins(bin_size_deg: float = 10.0) -> np.ndarray:
    """Create complete 0..180 degree bins with the requested width."""

    if not 0.0 < bin_size_deg <= 180.0:
        raise ValueError("angle bin size must be in (0, 180]")
    edges = np.arange(0.0, 180.0 + bin_size_deg * 0.5, bin_size_deg)
    if not np.isclose(edges[-1], 180.0):
        edges = np.append(edges, 180.0)
    return edges


def angular_statistics(
    angles_deg: np.ndarray,
    speeds: np.ndarray,
    *,
    bin_edges_deg: np.ndarray | None = None,
    bin_size_deg: float = 10.0,
) -> AngularStatistics:
    """Count fragments and summarize speed in each polar-angle bin."""

    angles = np.asarray(angles_deg, dtype=float)
    speed_values = np.asarray(speeds, dtype=float)
    if angles.shape != speed_values.shape:
        raise ValueError("angles_deg and speeds must have the same shape")
    edges = (
        make_angle_bins(bin_size_deg)
        if bin_edges_deg is None
        else np.asarray(bin_edges_deg, dtype=float)
    )
    _validate_angle_edges(edges)

    valid = (
        np.isfinite(angles)
        & np.isfinite(speed_values)
        & (angles >= edges[0])
        & (angles <= edges[-1])
    )
    counts, _ = np.histogram(angles[valid], bins=edges)
    n_bins = len(edges) - 1
    total = int(counts.sum())
    fractions = counts / total if total else np.zeros(n_bins, dtype=float)

    speed_min = np.full(n_bins, np.nan)
    speed_mean = np.full(n_bins, np.nan)
    speed_std = np.full(n_bins, np.nan)
    speed_p05 = np.full(n_bins, np.nan)
    speed_median = np.full(n_bins, np.nan)
    speed_p95 = np.full(n_bins, np.nan)
    speed_max = np.full(n_bins, np.nan)
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = valid & (angles >= edges[index]) & (angles <= edges[index + 1])
        else:
            mask = valid & (angles >= edges[index]) & (angles < edges[index + 1])
        values = speed_values[mask]
        if values.size == 0:
            continue
        speed_min[index] = np.min(values)
        speed_mean[index] = np.mean(values)
        speed_std[index] = np.std(values)
        speed_p05[index], speed_median[index], speed_p95[index] = np.quantile(
            values, [0.05, 0.5, 0.95]
        )
        speed_max[index] = np.max(values)

    return AngularStatistics(
        bin_edges_deg=edges,
        counts=counts,
        fractions=fractions,
        speed_min=speed_min,
        speed_mean=speed_mean,
        speed_std=speed_std,
        speed_p05=speed_p05,
        speed_median=speed_median,
        speed_p95=speed_p95,
        speed_max=speed_max,
    )


def angular_speed_histogram(
    angles_deg: np.ndarray,
    speeds: np.ndarray,
    *,
    angle_edges_deg: np.ndarray | None = None,
    angle_bin_size_deg: float = 10.0,
    speed_bin_width: float | None = None,
    speed_bin_count: int = 40,
) -> AngularSpeedHistogram:
    """Build a 2D histogram of fragment angle and speed."""

    angles = np.asarray(angles_deg, dtype=float)
    speed_values = np.asarray(speeds, dtype=float)
    if angles.shape != speed_values.shape:
        raise ValueError("angles_deg and speeds must have the same shape")
    angle_edges = (
        make_angle_bins(angle_bin_size_deg)
        if angle_edges_deg is None
        else np.asarray(angle_edges_deg, dtype=float)
    )
    _validate_angle_edges(angle_edges)
    speed_edges = _make_speed_bins(
        speed_values,
        bin_width=speed_bin_width,
        bin_count=speed_bin_count,
    )
    valid = (
        np.isfinite(angles)
        & np.isfinite(speed_values)
        & (angles >= angle_edges[0])
        & (angles <= angle_edges[-1])
        & (speed_values >= speed_edges[0])
        & (speed_values <= speed_edges[-1])
    )
    counts, _, _ = np.histogram2d(
        angles[valid],
        speed_values[valid],
        bins=(angle_edges, speed_edges),
    )
    return AngularSpeedHistogram(
        angle_edges_deg=angle_edges,
        speed_edges=speed_edges,
        counts=counts,
    )


def plot_sph_speed_3d(
    particles: SPHParticleData,
    output_html: str | Path,
    *,
    angles_deg: np.ndarray | None = None,
    length_unit: str = "model length",
    speed_unit: str = "model length/time",
    max_points: int = 50_000,
    marker_size: float = 2.6,
    title: str | None = None,
) -> Path:
    """Write an interactive 3D scatter plot colored by particle speed."""

    from plotly import graph_objects as go

    if len(particles) == 0:
        raise ValueError("cannot plot an empty SPH particle set")
    if max_points <= 0:
        raise ValueError("max_points must be positive")

    indexes = _display_indexes(len(particles), max_points)
    positions = particles.positions[indexes]
    speeds = particles.speeds[indexes]
    node_ids = particles.node_ids[indexes]
    material_ids = particles.material_indices[indexes]
    if angles_deg is None:
        angles = np.full(len(indexes), np.nan)
    else:
        angles = np.asarray(angles_deg, dtype=float)[indexes]

    color_min, color_max = np.quantile(speeds, [0.01, 0.99])
    if np.isclose(color_min, color_max):
        color_min = float(np.min(speeds))
        color_max = float(np.max(speeds))
    custom_data = np.column_stack((node_ids, material_ids, angles))
    plot_title = title or (
        f"SPH fragments at t = {particles.time:g} "
        f"({len(indexes):,}/{len(particles):,} shown)"
    )
    trace = go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker={
            "size": marker_size,
            "color": speeds,
            "colorscale": "Turbo",
            "cmin": float(color_min),
            "cmax": float(color_max),
            "opacity": 0.82,
            "colorbar": {
                "title": f"Speed<br>({speed_unit})",
                "thickness": 18,
                "len": 0.65,
            },
        },
        customdata=custom_data,
        hovertemplate=(
            "Node ID: %{customdata[0]:.0f}<br>"
            "Material index: %{customdata[1]:.0f}<br>"
            "Polar angle: %{customdata[2]:.2f} deg<br>"
            f"Position: (%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}}) {length_unit}<br>"
            f"Speed: %{{marker.color:.3f}} {speed_unit}"
            "<extra></extra>"
        ),
        name="SPH particles",
    )
    figure = go.Figure(data=[trace])
    figure.update_layout(
        title={"text": plot_title, "x": 0.5},
        template="plotly_white",
        width=1250,
        height=850,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        scene={
            "xaxis_title": f"X ({length_unit})",
            "yaxis_title": f"Y ({length_unit})",
            "zaxis_title": f"Z ({length_unit})",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.55, "y": 1.35, "z": 1.1}},
        },
    )

    path = Path(output_html)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(path), include_plotlyjs=True, full_html=True)
    return path


def plot_sph_speed_animation(
    sequence: SPHParticleSequence,
    output_html: str | Path,
    *,
    length_unit: str = "model length",
    speed_unit: str = "model length/time",
    speed_upper_bound: float | None = None,
    frame_duration_ms: int = 50,
    marker_size: float = 3.0,
    title: str = "SPH particle speed animation",
) -> Path:
    """Write a standalone time-slider 3D animation colored by particle speed."""

    from plotly import graph_objects as go

    if sequence.n_frames == 0 or sequence.n_particles == 0:
        raise ValueError("cannot animate an empty SPH sequence")
    if frame_duration_ms < 0:
        raise ValueError("frame_duration_ms must be non-negative")

    positions = np.asarray(sequence.positions, dtype=float)
    speeds = np.asarray(sequence.speeds, dtype=float)
    active = np.asarray(sequence.alive, dtype=bool) & np.isfinite(speeds)
    active &= np.isfinite(positions).all(axis=2)
    if speed_upper_bound is not None:
        active &= speeds <= float(speed_upper_bound)

    active_positions = positions[active]
    active_speeds = speeds[active]
    if active_positions.size == 0:
        raise ValueError("no valid particle data remain in the animation")

    color_min = float(np.min(active_speeds))
    color_max = float(np.max(active_speeds))
    if np.isclose(color_min, color_max):
        color_max = max(float(np.max(active_speeds)), 1.0)

    axis_ranges = []
    for dimension in range(3):
        coordinate = active_positions[:, dimension]
        low = float(np.min(coordinate))
        high = float(np.max(coordinate))
        padding = max(float(high - low) * 0.05, 1.0)
        axis_ranges.append([float(low - padding), float(high + padding)])

    def frame_trace(frame_index: int) -> go.Scatter3d:
        frame_active = active[frame_index]
        x = np.where(frame_active, positions[frame_index, :, 0], np.nan)
        y = np.where(frame_active, positions[frame_index, :, 1], np.nan)
        z = np.where(frame_active, positions[frame_index, :, 2], np.nan)
        color = np.where(frame_active, speeds[frame_index], color_min)
        return go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker={
                "size": marker_size,
                "color": color,
                "colorscale": "Turbo",
                "cmin": color_min,
                "cmax": color_max,
                "opacity": 0.84,
                "colorbar": {
                    "title": f"Speed<br>({speed_unit})",
                    "thickness": 18,
                    "len": 0.65,
                },
            },
            customdata=sequence.node_ids,
            hovertemplate=(
                "Node ID: %{customdata:.0f}<br>"
                f"Position: (%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}}) {length_unit}<br>"
                f"Speed: %{{marker.color:.3f}} {speed_unit}"
                "<extra></extra>"
            ),
            name="SPH particles",
        )

    frame_names = [f"{time_value:.6f}" for time_value in sequence.times]
    frames = [
        go.Frame(
            name=frame_names[index],
            data=[frame_trace(index)],
            layout={
                "title": {
                    "text": (
                        f"{title}<br><sup>t = {sequence.times[index]:.6f} "
                        f"(state {index + 1}/{sequence.n_frames})</sup>"
                    )
                }
            },
        )
        for index in range(sequence.n_frames)
    ]
    figure = go.Figure(
        data=[frame_trace(0)],
        frames=frames,
    )
    figure.update_layout(
        title={
            "text": (
                f"{title}<br><sup>t = {sequence.times[0]:.6f} "
                f"(state 1/{sequence.n_frames})</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        width=1250,
        height=850,
        margin={"l": 10, "r": 10, "t": 90, "b": 10},
        scene={
            "xaxis": {"title": f"X ({length_unit})", "range": axis_ranges[0]},
            "yaxis": {"title": f"Y ({length_unit})", "range": axis_ranges[1]},
            "zaxis": {"title": f"Z ({length_unit})", "range": axis_ranges[2]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.55, "y": 1.35, "z": 1.1}},
        },
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.05,
                "y": 1.12,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Time: ", "suffix": " ms"},
                "pad": {"t": 45, "b": 5},
                "steps": [
                    {
                        "label": name,
                        "method": "animate",
                        "args": [
                            [name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    }
                    for name in frame_names
                ],
            }
        ],
    )

    path = Path(output_html)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(path), include_plotlyjs=True, full_html=True)
    return path


def plot_angular_distribution(
    statistics: AngularStatistics,
    histogram: AngularSpeedHistogram,
    output_html: str | Path,
    *,
    speed_unit: str = "model length/time",
    title: str = "SPH fragment angular distribution",
) -> Path:
    """Write count, speed-summary, and count/speed-heatmap charts."""

    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    centers = statistics.centers_deg
    count_trace = go.Bar(
        x=centers,
        y=statistics.counts,
        marker_color="#3B82F6",
        name="Fragment count",
        hovertemplate="Angle: %{x:.1f} deg<br>Count: %{y}<extra></extra>",
    )

    p05 = statistics.speed_p05
    p95 = statistics.speed_p95
    band_x = np.concatenate((centers, centers[::-1]))
    band_y = np.concatenate((p95, p05[::-1]))
    band_trace = go.Scatter(
        x=band_x,
        y=band_y,
        fill="toself",
        fillcolor="rgba(249, 115, 22, 0.18)",
        line={"color": "rgba(255,255,255,0)"},
        hoverinfo="skip",
        showlegend=True,
        name="P05-P95 speed",
    )
    mean_trace = go.Scatter(
        x=centers,
        y=statistics.speed_mean,
        mode="lines+markers",
        line={"color": "#EA580C", "width": 2.5},
        marker={"size": 6},
        name="Mean speed",
        hovertemplate=(
            "Angle: %{x:.1f} deg<br>"
            f"Mean speed: %{{y:.3f}} {speed_unit}<extra></extra>"
        ),
    )

    figure = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Fragment count by polar angle",
            f"Speed distribution by polar angle ({speed_unit})",
            "Joint count distribution",
        ),
        vertical_spacing=0.09,
    )
    figure.add_trace(count_trace, row=1, col=1)
    figure.add_trace(band_trace, row=2, col=1)
    figure.add_trace(mean_trace, row=2, col=1)

    angle_centers = (
        histogram.angle_edges_deg[:-1] + histogram.angle_edges_deg[1:]
    ) / 2.0
    speed_centers = (histogram.speed_edges[:-1] + histogram.speed_edges[1:]) / 2.0
    heatmap_trace = go.Heatmap(
        x=angle_centers,
        y=speed_centers,
        z=histogram.counts.T,
        colorscale="Viridis",
        colorbar={"title": "Count", "len": 0.27, "y": 0.16},
        hovertemplate=(
            "Angle: %{x:.1f} deg<br>"
            f"Speed: %{{y:.3f}} {speed_unit}<br>"
            "Count: %{z}<extra></extra>"
        ),
    )
    figure.add_trace(heatmap_trace, row=3, col=1)
    figure.update_xaxes(title_text="Polar angle from axis (deg)", row=1, col=1)
    figure.update_yaxes(title_text="Count", row=1, col=1)
    figure.update_xaxes(title_text="Polar angle from axis (deg)", row=2, col=1)
    figure.update_yaxes(title_text="Speed", row=2, col=1)
    figure.update_xaxes(title_text="Polar angle from axis (deg)", row=3, col=1)
    figure.update_yaxes(title_text="Speed", row=3, col=1)
    figure.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_white",
        height=1150,
        width=1200,
        bargap=0.08,
        showlegend=True,
        margin={"l": 70, "r": 40, "t": 90, "b": 50},
    )

    path = Path(output_html)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(path), include_plotlyjs=True, full_html=True)
    return path


def analyze_sph_fragments(
    d3plot_path: str | Path,
    output_dir: str | Path,
    *,
    state: int = -1,
    time: float | None = None,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    angle_mode: str = "velocity",
    angle_bin_size_deg: float = 10.0,
    outlier_method: str = "tukey",
    iqr_factor: float = 3.0,
    mad_threshold: float = 6.0,
    speed_percentile: float = 99.0,
    speed_min: float | None = None,
    speed_max: float | None = None,
    speed_bin_width: float | None = None,
    speed_bin_count: int = 40,
    max_points: int = 50_000,
    animate: bool = True,
    max_animation_frames: int = 60,
    max_animation_points: int = 3_000,
    frame_duration_ms: int = 50,
    length_unit: str = "mm",
    speed_unit: str = "mm/ms",
) -> dict[str, Path | int | float | str | None]:
    """Run the complete SPH fragment extraction and visualization workflow."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_particles = load_sph_particle_data(
        d3plot_path,
        state=state,
        time=time,
    )
    particles, filter_report = filter_sph_outliers(
        all_particles,
        method=outlier_method,
        iqr_factor=iqr_factor,
        mad_threshold=mad_threshold,
        percentile=speed_percentile,
        speed_min=speed_min,
        speed_max=speed_max,
    )
    if len(particles) == 0:
        raise ValueError("all SPH particles were removed by the chosen filter")

    angles = particle_angles_deg(
        particles,
        axis=axis,
        mode=angle_mode,
        origin=origin,
    )
    angle_edges = make_angle_bins(angle_bin_size_deg)
    stats = angular_statistics(
        angles,
        particles.speeds,
        bin_edges_deg=angle_edges,
    )
    histogram = angular_speed_histogram(
        angles,
        particles.speeds,
        angle_edges_deg=angle_edges,
        speed_bin_width=speed_bin_width,
        speed_bin_count=speed_bin_count,
    )

    all_particles_csv = output_path / "sph_particles.csv"
    all_particles_npz = output_path / "sph_particles.npz"
    particles_csv = output_path / "sph_particles_filtered.csv"
    particles_npz = output_path / "sph_particles_filtered.npz"
    angle_csv = output_path / "angular_statistics.csv"
    histogram_csv = output_path / "angular_speed_histogram.csv"
    summary_json = output_path / "summary.json"
    plot_3d_html = output_path / "sph_speed_3d.html"
    angular_html = output_path / "angular_distribution.html"
    animation_html = output_path / "sph_speed_animation.html"
    sequence_npz = output_path / "sph_particle_sequence.npz"

    all_angles = particle_angles_deg(
        all_particles,
        axis=axis,
        mode=angle_mode,
        origin=origin,
    )
    keep_mask = np.isin(
        all_particles.particle_indices,
        particles.particle_indices,
        assume_unique=True,
    )
    if int(np.count_nonzero(keep_mask)) != len(particles):
        raise RuntimeError("filtered particle lookup is inconsistent")
    _write_particles_csv(
        all_particles,
        all_angles,
        all_particles_csv,
        kept=keep_mask,
    )
    _write_particle_npz(
        all_particles,
        all_angles,
        all_particles_npz,
        kept=keep_mask,
    )
    _write_particles_csv(particles, angles, particles_csv)
    _write_particle_npz(particles, angles, particles_npz)
    _write_angular_statistics_csv(stats, angle_csv)
    _write_histogram_csv(histogram, histogram_csv)
    plot_sph_speed_3d(
        particles,
        plot_3d_html,
        angles_deg=angles,
        length_unit=length_unit,
        speed_unit=speed_unit,
        max_points=max_points,
        title=(
            f"SPH speed at t = {particles.time:g} "
            f"({len(particles):,} filtered particles)"
        ),
    )
    plot_angular_distribution(
        stats,
        histogram,
        angular_html,
        speed_unit=speed_unit,
        title=(
            f"SPH fragment angular statistics, "
            f"{angle_mode} direction from axis"
        ),
    )

    animation_result: dict[str, Path | int | float | None] = {
        "html": None,
        "sequence_npz": None,
        "frame_count": 0,
        "particle_count": 0,
    }
    if animate:
        if max_animation_points <= 0:
            raise ValueError("max_animation_points must be positive")
        animation_indexes = _display_indexes(
            len(particles),
            max_animation_points,
        )
        sequence = load_sph_particle_sequence(
            d3plot_path,
            max_frames=max_animation_frames,
            particle_indices=particles.particle_indices[animation_indexes],
        )
        _write_sequence_npz(sequence, sequence_npz)
        plot_sph_speed_animation(
            sequence,
            animation_html,
            length_unit=length_unit,
            speed_unit=speed_unit,
            speed_upper_bound=filter_report.upper_speed,
            frame_duration_ms=frame_duration_ms,
            title=(
                "SPH particle speed animation | "
                f"{angle_mode} angle reference +{_format_axis(axis)}"
            ),
        )
        animation_result = {
            "html": animation_html,
            "sequence_npz": sequence_npz,
            "frame_count": sequence.n_frames,
            "particle_count": sequence.n_particles,
        }

    finite_angles = angles[np.isfinite(angles)]
    summary = {
        "d3plot": str(Path(d3plot_path).resolve()),
        "state": state,
        "time": particles.time,
        "axis": [float(value) for value in axis],
        "origin": [float(value) for value in origin],
        "angle_mode": angle_mode,
        "angle_bin_size_deg": angle_bin_size_deg,
        "speed_unit": speed_unit,
        "length_unit": length_unit,
        "sph_particle_count": len(all_particles),
        "filtered_particle_count": len(particles),
        "valid_angle_count": int(finite_angles.size),
        "invalid_angle_count": int(len(angles) - finite_angles.size),
        "filter": asdict(filter_report),
        "filtered_speed_statistics": _speed_statistics(particles.speeds),
        "outputs": {
            "all_particles_csv": str(all_particles_csv),
            "all_particles_npz": str(all_particles_npz),
            "filtered_particles_csv": str(particles_csv),
            "filtered_particles_npz": str(particles_npz),
            "angular_statistics_csv": str(angle_csv),
            "angular_speed_histogram_csv": str(histogram_csv),
            "speed_3d_html": str(plot_3d_html),
            "angular_distribution_html": str(angular_html),
            "animation_html": (
                str(animation_html) if animation_result["html"] else None
            ),
            "animation_sequence_npz": (
                str(sequence_npz) if animation_result["sequence_npz"] else None
            ),
            "animation_frame_count": animation_result["frame_count"],
            "animation_particle_count": animation_result["particle_count"],
        },
    }
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "particles": len(particles),
        "filtered_out": filter_report.excluded_count,
        "time": particles.time,
        "summary": summary_json,
        "plot_3d": plot_3d_html,
        "animation_plot": animation_result["html"],
        "angular_plot": angular_html,
        "particles_csv": particles_csv,
        "angular_csv": angle_csv,
    }


def _single_state_array(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 0:
        return values
    if values.shape[0] != 1:
        raise ValueError(
            f"expected exactly one loaded state, got array shape {values.shape}"
        )
    return values[0]


def _validate_angle_edges(edges: np.ndarray) -> None:
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("angle bin edges must be finite and strictly increasing")


def _make_speed_bins(
    speeds: np.ndarray,
    *,
    bin_width: float | None,
    bin_count: int,
) -> np.ndarray:
    values = np.asarray(speeds, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot create speed bins without finite speeds")
    low = min(0.0, float(np.min(values)))
    high = float(np.max(values))
    if np.isclose(low, high):
        high = low + 1.0
    if bin_width is not None:
        if bin_width <= 0.0:
            raise ValueError("speed bin width must be positive")
        start = np.floor(low / bin_width) * bin_width
        stop = np.ceil(high / bin_width) * bin_width
        if np.isclose(start, stop):
            stop = start + bin_width
        return np.arange(start, stop + bin_width * 0.5, bin_width)
    if bin_count < 1:
        raise ValueError("speed bin count must be positive")
    return np.linspace(low, high, bin_count + 1)


def _display_indexes(count: int, max_points: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count)
    rng = np.random.default_rng(20261028)
    indexes = rng.choice(count, size=max_points, replace=False)
    return np.sort(indexes)


def _write_particles_csv(
    particles: SPHParticleData,
    angles: np.ndarray,
    path: Path,
    *,
    kept: np.ndarray | None = None,
) -> None:
    kept_values = (
        np.ones(len(particles), dtype=bool)
        if kept is None
        else np.asarray(kept, dtype=bool)
    )
    if kept_values.shape != (len(particles),):
        raise ValueError("kept mask must match the particle count")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "node_id",
                "particle_index",
                "material_index",
                "alive",
                "kept",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "speed",
                "polar_angle_deg",
            ]
        )
        for index in range(len(particles)):
            writer.writerow(
                [
                    int(particles.node_ids[index]),
                    int(particles.particle_indices[index]),
                    int(particles.material_indices[index]),
                    bool(particles.alive[index]),
                    bool(kept_values[index]),
                    *particles.positions[index].tolist(),
                    *particles.velocities[index].tolist(),
                    float(particles.speeds[index]),
                    float(angles[index]),
                ]
            )


def _write_particle_npz(
    particles: SPHParticleData,
    angles: np.ndarray,
    path: Path,
    *,
    kept: np.ndarray | None = None,
) -> None:
    kept_values = (
        np.ones(len(particles), dtype=bool)
        if kept is None
        else np.asarray(kept, dtype=bool)
    )
    if kept_values.shape != (len(particles),):
        raise ValueError("kept mask must match the particle count")
    np.savez_compressed(
        path,
        positions=particles.positions,
        velocities=particles.velocities,
        speed=particles.speeds,
        polar_angle_deg=angles,
        node_ids=particles.node_ids,
        particle_index=particles.particle_indices,
        material_index=particles.material_indices,
        alive=particles.alive,
        kept=kept_values,
        time=np.asarray(particles.time),
    )


def _write_sequence_npz(sequence: SPHParticleSequence, path: Path) -> None:
    np.savez_compressed(
        path,
        positions=sequence.positions,
        velocities=sequence.velocities,
        speed=sequence.speeds,
        alive=sequence.alive,
        times=sequence.times,
        node_ids=sequence.node_ids,
        particle_index=sequence.particle_indices,
    )


def _write_angular_statistics_csv(
    statistics: AngularStatistics,
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "angle_start_deg",
                "angle_end_deg",
                "angle_center_deg",
                "count",
                "fraction",
                "speed_min",
                "speed_mean",
                "speed_std",
                "speed_p05",
                "speed_median",
                "speed_p95",
                "speed_max",
            ]
        )
        fields = (
            statistics.speed_min,
            statistics.speed_mean,
            statistics.speed_std,
            statistics.speed_p05,
            statistics.speed_median,
            statistics.speed_p95,
            statistics.speed_max,
        )
        for index in range(len(statistics.counts)):
            writer.writerow(
                [
                    statistics.bin_edges_deg[index],
                    statistics.bin_edges_deg[index + 1],
                    statistics.centers_deg[index],
                    int(statistics.counts[index]),
                    float(statistics.fractions[index]),
                    *[float(values[index]) for values in fields],
                ]
            )


def _write_histogram_csv(
    histogram: AngularSpeedHistogram,
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "angle_start_deg",
                "angle_end_deg",
                "speed_start",
                "speed_end",
                "count",
            ]
        )
        for angle_index in range(len(histogram.angle_edges_deg) - 1):
            for speed_index in range(len(histogram.speed_edges) - 1):
                writer.writerow(
                    [
                        histogram.angle_edges_deg[angle_index],
                        histogram.angle_edges_deg[angle_index + 1],
                        histogram.speed_edges[speed_index],
                        histogram.speed_edges[speed_index + 1],
                        int(histogram.counts[angle_index, speed_index]),
                    ]
                )


def _speed_statistics(speeds: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(speeds, dtype=float)
    quantiles = np.quantile(values, [0.0, 0.05, 0.5, 0.95, 1.0])
    return {
        "count": int(values.size),
        "min": float(quantiles[0]),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "max": float(quantiles[4]),
    }


def _parse_vector(value: str, name: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be x,y,z") from error
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{name} must be x,y,z")
    return parts


def _format_axis(axis: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(value):g}" for value in axis) + ")"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ls-ex-sph",
        description=(
            "Read LS-DYNA d3plot SPH particles, filter abnormal speeds, "
            "and export 3D/angular reports."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("d3plot", help="path to the first d3plot file")
    parser.add_argument("-o", "--output-dir", default="outputs/sph-fragments")
    parser.add_argument("--state", type=int, default=-1)
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="select the state nearest this time; overrides --state",
    )
    parser.add_argument("--axis", default="0,0,1", help="axis as x,y,z")
    parser.add_argument("--origin", default="0,0,0", help="position-angle origin")
    parser.add_argument(
        "--angle-mode", choices=("velocity", "position"), default="velocity"
    )
    parser.add_argument("--angle-bin-size", type=float, default=10.0)
    parser.add_argument(
        "--outlier-method",
        choices=("tukey", "mad", "percentile", "none"),
        default="tukey",
    )
    parser.add_argument("--iqr-factor", type=float, default=3.0)
    parser.add_argument("--mad-threshold", type=float, default=6.0)
    parser.add_argument("--speed-percentile", type=float, default=99.0)
    parser.add_argument("--speed-min", type=float, default=None)
    parser.add_argument("--speed-max", type=float, default=None)
    parser.add_argument("--speed-bin-width", type=float, default=None)
    parser.add_argument("--speed-bin-count", type=int, default=40)
    parser.add_argument("--max-points", type=int, default=50_000)
    parser.add_argument(
        "--animate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="generate the 3D time-slider animation",
    )
    parser.add_argument("--max-animation-frames", type=int, default=60)
    parser.add_argument("--max-animation-points", type=int, default=3_000)
    parser.add_argument("--frame-duration-ms", type=int, default=50)
    parser.add_argument("--length-unit", default="mm")
    parser.add_argument("--speed-unit", default="mm/ms")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    axis = _parse_vector(args.axis, "axis")
    origin = _parse_vector(args.origin, "origin")
    result = analyze_sph_fragments(
        args.d3plot,
        args.output_dir,
        state=args.state,
        time=args.time,
        axis=axis,
        origin=origin,
        angle_mode=args.angle_mode,
        angle_bin_size_deg=args.angle_bin_size,
        outlier_method=args.outlier_method,
        iqr_factor=args.iqr_factor,
        mad_threshold=args.mad_threshold,
        speed_percentile=args.speed_percentile,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        speed_bin_width=args.speed_bin_width,
        speed_bin_count=args.speed_bin_count,
        max_points=args.max_points,
        animate=args.animate,
        max_animation_frames=args.max_animation_frames,
        max_animation_points=args.max_animation_points,
        frame_duration_ms=args.frame_duration_ms,
        length_unit=args.length_unit,
        speed_unit=args.speed_unit,
    )
    print(f"time: {result['time']:.9g}")
    print(
        f"particles: {result['particles']} used, "
        f"{result['filtered_out']} removed as outliers"
    )
    print(f"3D plot: {result['plot_3d']}")
    print(f"3D animation: {result['animation_plot']}")
    print(f"angular plot: {result['angular_plot']}")
    print(f"particle data: {result['particles_csv']}")
    print(f"summary: {result['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
