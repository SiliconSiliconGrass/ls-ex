"""Post-process the 1/4 elliptical variable-section warhead simulations.

The two LS-DYNA runs use different fragmentation models:

* no-SPH: surviving shell solid elements are the fragment samples.
* SPH: adaptive shell elements become active SPH particles.

The script exports machine-readable particle/fragment data and paper-style
speed fields, speed distributions, quantity histories, radial distributions,
and axial speed profiles.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402


DEFAULT_ROOT = Path(
    r"D:\MyStuff\Research\sim-playground\elliptical20261030"
)
DEFAULT_TIMES_US = (0.0, 8.0, 10.0, 20.0, 40.0, 60.0, 100.0, 150.0, 200.0)
DEFAULT_SPEED_MAX = 6000.0
DISPLAY_SPEED_MAX = 2000.0
AZIMUTH_BIN_SIZE = 15.0


@dataclass
class ConnectedFragmentData:
    fragment_ids: np.ndarray
    initial_positions: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    speeds: np.ndarray
    element_counts: np.ndarray
    azimuth0_deg: np.ndarray


@dataclass
class FrameData:
    time_ms: float
    entity_ids: np.ndarray
    initial_positions: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    speeds: np.ndarray
    valid: np.ndarray
    azimuth0_deg: np.ndarray
    polar_deg: np.ndarray
    mass_g: np.ndarray | None = None
    radius_mm: np.ndarray | None = None
    connected: ConnectedFragmentData | None = None

    @property
    def active_count(self) -> int:
        return len(self.entity_ids)

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid))


@dataclass
class SimulationResult:
    case_name: str
    kind: str
    d3plot: Path
    output_dir: Path
    all_times_ms: np.ndarray
    activity_counts: np.ndarray
    frames: list[FrameData]
    speed_max: float
    sph_deletion_nonzero_count: int | None = None


def configure_matplotlib() -> None:
    candidates = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    )
    for path in candidates:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def nearest_state_indexes(times: np.ndarray, targets_us: Sequence[float]) -> list[int]:
    targets_ms = np.asarray(targets_us, dtype=float) / 1000.0
    return sorted({int(np.argmin(np.abs(times - target))) for target in targets_ms})


def read_all_times(d3plot: Path) -> np.ndarray:
    from lasso.dyna import ArrayType, D3plot

    reader = D3plot(
        str(d3plot),
        state_array_filter=[ArrayType.global_timesteps],
    )
    times = np.asarray(reader.arrays[ArrayType.global_timesteps], dtype=float)
    if times.size == 0:
        raise ValueError(f"d3plot has no states: {d3plot}")
    return times


def read_no_sph_activity(d3plot: Path) -> np.ndarray:
    from lasso.dyna import ArrayType, D3plot

    reader = D3plot(
        str(d3plot),
        state_array_filter=[
            ArrayType.global_timesteps,
            ArrayType.element_solid_is_alive,
        ],
    )
    arrays = reader.arrays
    part_indexes = np.asarray(arrays[ArrayType.element_solid_part_indexes])
    part_ids = np.asarray(arrays[ArrayType.part_ids], dtype=np.int64)
    shell_part_index = int(np.flatnonzero(part_ids == 2)[0])
    shell_elements = np.flatnonzero(part_indexes == shell_part_index)
    alive = np.asarray(arrays[ArrayType.element_solid_is_alive], dtype=float)
    return np.count_nonzero(alive[:, shell_elements] > 0.5, axis=1)


def read_sph_activity(d3plot: Path) -> tuple[np.ndarray, int]:
    from lasso.dyna import ArrayType, D3plot

    reader = D3plot(
        str(d3plot),
        state_array_filter=[
            ArrayType.global_timesteps,
            ArrayType.sph_deletion,
            ArrayType.sph_radius,
            ArrayType.sph_mass,
        ],
    )
    arrays = reader.arrays
    deletion = np.asarray(arrays[ArrayType.sph_deletion])
    radius = np.asarray(arrays[ArrayType.sph_radius])
    mass = np.asarray(arrays[ArrayType.sph_mass])
    active = (radius > 0.0) & (mass > 0.0) & (deletion == 0.0)
    return (
        np.count_nonzero(active, axis=1),
        int(np.count_nonzero(deletion != 0.0)),
    )


def read_no_sph_frames(
    d3plot: Path,
    state_indexes: Sequence[int],
    speed_max: float,
) -> list[FrameData]:
    from lasso.dyna import ArrayType, D3plot

    reader = D3plot(
        str(d3plot),
        state_filter=set(state_indexes),
        buffered_reading=True,
        state_array_filter=[
            ArrayType.global_timesteps,
            ArrayType.element_solid_is_alive,
            ArrayType.node_displacement,
            ArrayType.node_velocity,
        ],
    )
    arrays = reader.arrays
    times = np.asarray(arrays[ArrayType.global_timesteps], dtype=float)
    coordinates = np.asarray(arrays[ArrayType.node_coordinates], dtype=float)
    displacement = np.asarray(arrays[ArrayType.node_displacement], dtype=float)
    velocity = np.asarray(arrays[ArrayType.node_velocity], dtype=float)
    alive = np.asarray(arrays[ArrayType.element_solid_is_alive], dtype=float)
    element_nodes = np.asarray(
        arrays[ArrayType.element_solid_node_indexes], dtype=np.int64
    )
    element_ids = np.asarray(arrays[ArrayType.element_solid_ids], dtype=np.int64)
    part_indexes = np.asarray(arrays[ArrayType.element_solid_part_indexes])
    part_ids = np.asarray(arrays[ArrayType.part_ids], dtype=np.int64)
    shell_part_index = int(np.flatnonzero(part_ids == 2)[0])
    shell_elements = np.flatnonzero(part_indexes == shell_part_index)
    shell_nodes = element_nodes[shell_elements]
    shell_ids = element_ids[shell_elements]

    frames: list[FrameData] = []
    for frame_index, time in enumerate(times):
        active_local = alive[frame_index, shell_elements] > 0.5
        active_nodes = shell_nodes[active_local]
        current_nodes = coordinates + displacement[frame_index]
        initial_positions = coordinates[active_nodes].mean(axis=1)
        positions = current_nodes[active_nodes].mean(axis=1)
        velocities = velocity[frame_index, active_nodes].mean(axis=1)
        speeds = np.linalg.norm(velocities, axis=1)
        valid = (
            np.all(np.isfinite(positions), axis=1)
            & np.all(np.isfinite(velocities), axis=1)
            & np.isfinite(speeds)
            & (speeds <= speed_max)
        )
        azimuth = np.degrees(
            np.arctan2(initial_positions[:, 1], initial_positions[:, 0])
        )
        polar = velocity_polar_angle_deg(velocities)
        connected = connected_element_fragments(
            active_nodes=active_nodes,
            node_count=len(coordinates),
            initial_nodes=coordinates,
            current_nodes=current_nodes,
            current_velocities=velocity[frame_index],
            valid_elements=valid,
        )
        frames.append(
            FrameData(
                time_ms=float(time),
                entity_ids=shell_ids[active_local],
                initial_positions=initial_positions,
                positions=positions,
                velocities=velocities,
                speeds=speeds,
                valid=valid,
                azimuth0_deg=azimuth,
                polar_deg=polar,
                connected=connected,
            )
        )
    return frames


def read_sph_frames(
    d3plot: Path,
    state_indexes: Sequence[int],
    speed_max: float,
) -> list[FrameData]:
    from lasso.dyna import ArrayType, D3plot

    reader = D3plot(
        str(d3plot),
        state_filter=set(state_indexes),
        buffered_reading=True,
        state_array_filter=[
            ArrayType.global_timesteps,
            ArrayType.sph_deletion,
            ArrayType.sph_radius,
            ArrayType.sph_mass,
            ArrayType.node_displacement,
            ArrayType.node_velocity,
        ],
    )
    arrays = reader.arrays
    times = np.asarray(arrays[ArrayType.global_timesteps], dtype=float)
    sph_nodes = np.asarray(arrays[ArrayType.sph_node_indexes], dtype=np.int64)
    hierarchy_node_ids = np.asarray(arrays[ArrayType.node_ids], dtype=np.int64)
    coordinates = np.asarray(arrays[ArrayType.node_coordinates], dtype=float)
    displacement = np.asarray(arrays[ArrayType.node_displacement], dtype=float)
    velocity = np.asarray(arrays[ArrayType.node_velocity], dtype=float)
    deletion = np.asarray(arrays[ArrayType.sph_deletion], dtype=float)
    radius = np.asarray(arrays[ArrayType.sph_radius], dtype=float)
    mass = np.asarray(arrays[ArrayType.sph_mass], dtype=float)

    frames: list[FrameData] = []
    for frame_index, time in enumerate(times):
        active = (
            (radius[frame_index] > 0.0)
            & (mass[frame_index] > 0.0)
            & (deletion[frame_index] == 0.0)
        )
        active_nodes = sph_nodes[active]
        initial_positions = coordinates[active_nodes]
        positions = initial_positions + displacement[frame_index, active_nodes]
        velocities = velocity[frame_index, active_nodes]
        speeds = np.linalg.norm(velocities, axis=1)
        valid = (
            np.all(np.isfinite(positions), axis=1)
            & np.all(np.isfinite(velocities), axis=1)
            & np.isfinite(speeds)
            & (speeds <= speed_max)
        )
        azimuth = np.degrees(
            np.arctan2(initial_positions[:, 1], initial_positions[:, 0])
        )
        frames.append(
            FrameData(
                time_ms=float(time),
                entity_ids=hierarchy_node_ids[active_nodes],
                initial_positions=initial_positions,
                positions=positions,
                velocities=velocities,
                speeds=speeds,
                valid=valid,
                azimuth0_deg=azimuth,
                polar_deg=velocity_polar_angle_deg(velocities),
                mass_g=mass[frame_index, active],
                radius_mm=radius[frame_index, active],
            )
        )
    return frames


def velocity_polar_angle_deg(velocities: np.ndarray) -> np.ndarray:
    magnitudes = np.linalg.norm(velocities, axis=1)
    values = np.full(len(velocities), np.nan)
    usable = np.isfinite(magnitudes) & (magnitudes > 0.0)
    values[usable] = np.degrees(
        np.arccos(np.clip(velocities[usable, 2] / magnitudes[usable], -1.0, 1.0))
    )
    return values


def connected_element_fragments(
    *,
    active_nodes: np.ndarray,
    node_count: int,
    initial_nodes: np.ndarray,
    current_nodes: np.ndarray,
    current_velocities: np.ndarray,
    valid_elements: np.ndarray,
) -> ConnectedFragmentData:
    selected_nodes = active_nodes[valid_elements]
    if selected_nodes.size == 0:
        return ConnectedFragmentData(
            fragment_ids=np.full(np.count_nonzero(valid_elements), -1, dtype=int),
            initial_positions=np.empty((0, 3)),
            positions=np.empty((0, 3)),
            velocities=np.empty((0, 3)),
            speeds=np.empty(0),
            element_counts=np.empty(0, dtype=int),
            azimuth0_deg=np.empty(0),
        )

    rows = np.repeat(selected_nodes[:, 0], selected_nodes.shape[1] - 1)
    columns = selected_nodes[:, 1:].reshape(-1)
    adjacency = coo_matrix(
        (
            np.ones(rows.size, dtype=np.int8),
            (rows, columns),
        ),
        shape=(node_count, node_count),
    )
    _, node_labels = connected_components(adjacency, directed=False)

    element_component_labels = node_labels[selected_nodes[:, 0]]
    unique_labels, element_component_index = np.unique(
        element_component_labels, return_inverse=True
    )
    used_nodes = np.unique(selected_nodes.reshape(-1))
    used_node_component_labels = node_labels[used_nodes]
    _, used_node_component_index = np.unique(
        used_node_component_labels, return_inverse=True
    )
    n_fragments = len(unique_labels)
    denominators = np.bincount(
        used_node_component_index, minlength=n_fragments
    ).astype(float)
    initial_positions = np.column_stack(
        [
            np.bincount(
                used_node_component_index,
                weights=initial_nodes[used_nodes, axis],
                minlength=n_fragments,
            )
            / denominators
            for axis in range(3)
        ]
    )
    positions = np.column_stack(
        [
            np.bincount(
                used_node_component_index,
                weights=current_nodes[used_nodes, axis],
                minlength=n_fragments,
            )
            / denominators
            for axis in range(3)
        ]
    )
    velocities = np.column_stack(
        [
            np.bincount(
                used_node_component_index,
                weights=current_velocities[used_nodes, axis],
                minlength=n_fragments,
            )
            / denominators
            for axis in range(3)
        ]
    )
    speeds = np.linalg.norm(velocities, axis=1)
    element_counts = np.bincount(
        element_component_index, minlength=n_fragments
    )
    azimuth0_deg = np.degrees(
        np.arctan2(initial_positions[:, 1], initial_positions[:, 0])
    )
    return ConnectedFragmentData(
        fragment_ids=element_component_index,
        initial_positions=initial_positions,
        positions=positions,
        velocities=velocities,
        speeds=speeds,
        element_counts=element_counts,
        azimuth0_deg=azimuth0_deg,
    )


def analyze_case(
    case_name: str,
    kind: str,
    d3plot: Path,
    output_dir: Path,
    target_times_us: Sequence[float],
    speed_max: float,
) -> SimulationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_times = read_all_times(d3plot)
    state_indexes = nearest_state_indexes(all_times, target_times_us)
    if kind == "no-sph":
        activity = read_no_sph_activity(d3plot)
        frames = read_no_sph_frames(d3plot, state_indexes, speed_max)
        deletion_count = None
    elif kind == "sph":
        activity, deletion_count = read_sph_activity(d3plot)
        frames = read_sph_frames(d3plot, state_indexes, speed_max)
    else:
        raise ValueError(f"unknown case kind: {kind}")
    return SimulationResult(
        case_name=case_name,
        kind=kind,
        d3plot=d3plot,
        output_dir=output_dir,
        all_times_ms=all_times,
        activity_counts=activity,
        frames=frames,
        speed_max=speed_max,
        sph_deletion_nonzero_count=deletion_count,
    )


def display_limits(frames: Iterable[FrameData]) -> dict[str, float]:
    radial: list[np.ndarray] = []
    axial: list[np.ndarray] = []
    for frame in frames:
        positions = frame.positions[frame.valid]
        if len(positions) == 0:
            continue
        radial.append(np.hypot(positions[:, 0], positions[:, 1]))
        axial.append(positions[:, 2])
    if not radial:
        return {"r": 250.0, "zmin": -100.0, "zmax": 400.0}
    radius_values = np.concatenate(radial)
    z_values = np.concatenate(axial)
    radius_limit = float(np.quantile(radius_values, 0.995)) * 1.08
    zmin = float(np.quantile(z_values, 0.002)) - 10.0
    zmax = float(np.quantile(z_values, 0.998)) + 10.0
    return {
        "r": max(radius_limit, 60.0),
        "zmin": min(zmin, -20.0),
        "zmax": max(zmax, 140.0),
    }


def plot_field(
    result: SimulationResult,
    projection: str,
    output_path: Path,
) -> None:
    frames = result.frames
    limits = display_limits(frames)
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(11.0, 10.6),
        constrained_layout=True,
    )
    norm = Normalize(vmin=0.0, vmax=DISPLAY_SPEED_MAX, clip=True)
    scatter = None
    for axis, frame in zip(axes.flat, frames, strict=True):
        mask = frame.valid
        positions = frame.positions[mask]
        speeds = frame.speeds[mask]
        time_us = frame.time_ms * 1000.0
        if positions.size:
            if projection == "radial":
                x_values = np.concatenate(
                    (positions[:, 0], -positions[:, 0], positions[:, 0], -positions[:, 0])
                )
                y_values = np.concatenate(
                    (positions[:, 1], positions[:, 1], -positions[:, 1], -positions[:, 1])
                )
                color_values = np.tile(speeds, 4)
                scatter = axis.scatter(
                    x_values,
                    y_values,
                    c=color_values,
                    cmap="jet",
                    norm=norm,
                    s=3.2,
                    linewidths=0,
                    rasterized=True,
                )
                axis.set_xlim(-limits["r"], limits["r"])
                axis.set_ylim(-limits["r"], limits["r"])
                axis.set_xlabel("x / mm")
                axis.set_ylabel("y / mm")
                axis.set_aspect("equal", adjustable="box")
            elif projection == "axial":
                radius = np.hypot(positions[:, 0], positions[:, 1])
                x_values = np.concatenate((radius, -radius))
                y_values = np.concatenate((positions[:, 2], positions[:, 2]))
                color_values = np.concatenate((speeds, speeds))
                scatter = axis.scatter(
                    x_values,
                    y_values,
                    c=color_values,
                    cmap="jet",
                    norm=norm,
                    s=3.2,
                    linewidths=0,
                    rasterized=True,
                )
                axis.set_xlim(-limits["r"], limits["r"])
                axis.set_ylim(limits["zmin"], limits["zmax"])
                axis.set_xlabel("径向坐标 r / mm")
                axis.set_ylabel("轴向 z / mm")
                axis.set_aspect("equal", adjustable="box")
            else:
                raise ValueError(f"unknown projection: {projection}")
        else:
            axis.text(
                0.5,
                0.5,
                "该时刻无有效活动粒子",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="0.4",
            )
            axis.set_xlim(-limits["r"], limits["r"])
            axis.set_ylim(
                limits["zmin"] if projection == "axial" else -limits["r"],
                limits["zmax"] if projection == "axial" else limits["r"],
            )
            axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"t = {time_us:.1f} μs | 有效 {frame.valid_count:,} / "
            f"活动 {frame.active_count:,}"
        )
        axis.grid(color="0.9", linewidth=0.5)
        axis.tick_params(labelsize=8)

    subtitle = (
        "径向投影：沿轴向观察，1/4 模型按双平面对称展开"
        if projection == "radial"
        else "轴向投影：r = sqrt(x²+y²)，按双平面对称展开"
    )
    entity = "SPH 粒子" if result.kind == "sph" else "存活壳体单元质心"
    fig.suptitle(
        f"{result.case_name} 破片速度场（{entity}）\n{subtitle}",
        fontsize=14,
    )
    if scatter is not None:
        colorbar = fig.colorbar(
            scatter,
            ax=axes,
            orientation="horizontal",
            shrink=0.62,
            pad=0.025,
            extend="max",
        )
        colorbar.set_label(r"破片总速度 $|v|$ / (m s$^{-1}$)")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_speed_and_count(
    result: SimulationResult,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=True)
    bins = np.linspace(0.0, DISPLAY_SPEED_MAX, 41)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(result.frames)))
    for color, frame in zip(colors, result.frames, strict=True):
        speeds = frame.speeds[frame.valid]
        if speeds.size == 0:
            continue
        counts, _ = np.histogram(speeds, bins=bins)
        fractions = counts / speeds.size * 100.0
        axes[0, 0].plot(
            (bins[:-1] + bins[1:]) / 2.0,
            fractions,
            color=color,
            linewidth=1.4,
            label=f"{frame.time_ms * 1000:g} μs",
        )
    axes[0, 0].set_xlim(0.0, DISPLAY_SPEED_MAX)
    axes[0, 0].set_xlabel(r"总速度 $|v|$ / (m s$^{-1}$)")
    axes[0, 0].set_ylabel("粒子/单元占比 / %")
    axes[0, 0].set_title("不同时刻速度分布")
    axes[0, 0].legend(ncol=3, frameon=False)

    axes[0, 1].plot(
        result.all_times_ms * 1000.0,
        result.activity_counts,
        color="0.35",
        linewidth=1.7,
        label="活动实体",
    )
    selected_times = np.array([frame.time_ms * 1000.0 for frame in result.frames])
    selected_valid = np.array([frame.valid_count for frame in result.frames])
    axes[0, 1].plot(
        selected_times,
        selected_valid,
        "o-",
        color="#d1495b",
        markersize=4,
        linewidth=1.3,
        label="通过速度有效性筛选",
    )
    legend_handles, legend_labels = axes[0, 1].get_legend_handles_labels()
    if result.kind == "no-sph":
        connected_counts = np.array(
            [
                len(frame.connected.element_counts)
                if frame.connected is not None
                else 0
                for frame in result.frames
            ]
        )
        cluster_axis = axes[0, 1].twinx()
        connected_line = cluster_axis.plot(
            selected_times,
            connected_counts,
            "s-",
            color="#2a9d8f",
            markersize=4,
            linewidth=1.3,
            label="节点连通破片簇",
        )[0]
        cluster_axis.set_ylabel("节点连通破片簇数量")
        cluster_axis.tick_params(axis="y", colors="#2a9d8f")
        legend_handles.append(connected_line)
        legend_labels.append("节点连通破片簇")
    axes[0, 1].set_xlabel("t / μs")
    axes[0, 1].set_ylabel("数量")
    axes[0, 1].set_title("破片/有效采样数量随时间变化")
    axes[0, 1].legend(legend_handles, legend_labels, frameon=False, loc="upper left")

    for statistic, color in (("median", "#1f77b4"), ("p95", "#d1495b")):
        values = []
        for frame in result.frames:
            speeds = frame.speeds[frame.valid]
            values.append(
                float(np.quantile(speeds, 0.5 if statistic == "median" else 0.95))
                if speeds.size
                else np.nan
            )
        axes[1, 0].plot(
            selected_times,
            values,
            "o-",
            color=color,
            label="中位数" if statistic == "median" else "95% 分位数",
        )
    axes[1, 0].set_xlabel("t / μs")
    axes[1, 0].set_ylabel(r"速度 / (m s$^{-1}$)")
    axes[1, 0].set_title("速度统计量随时间变化")
    axes[1, 0].legend(frameon=False)

    final = result.frames[-1]
    final_speeds = final.speeds[final.valid]
    if final_speeds.size:
        axes[1, 1].hist(
            final_speeds,
            bins=np.linspace(0.0, DISPLAY_SPEED_MAX, 51),
            color="#457b9d",
            alpha=0.88,
            edgecolor="white",
            linewidth=0.3,
        )
    axes[1, 1].set_xlim(0.0, DISPLAY_SPEED_MAX)
    axes[1, 1].set_xlabel(r"总速度 $|v|$ / (m s$^{-1}$)")
    axes[1, 1].set_ylabel("数量")
    axes[1, 1].set_title(
        f"末帧速度直方图（t = {final.time_ms * 1000:.1f} μs）"
    )

    fig.suptitle(
        f"{result.case_name} 速度与数量分布\n"
        f"直方图仅显示 0–{DISPLAY_SPEED_MAX:g} m/s；CSV 保留全部有效值"
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_azimuth_distribution(
    result: SimulationResult,
    output_path: Path,
) -> None:
    usable_frames = [frame for frame in result.frames if frame.valid_count]
    if not usable_frames:
        raise ValueError(f"{result.case_name}: no valid frame for azimuth plot")
    frame = usable_frames[-1]
    azimuth = frame.azimuth0_deg[frame.valid]
    speeds = frame.speeds[frame.valid]
    edges = np.arange(0.0, 90.0 + AZIMUTH_BIN_SIZE, AZIMUTH_BIN_SIZE)
    indices = np.clip(
        np.digitize(azimuth, edges, right=False) - 1,
        0,
        len(edges) - 2,
    )
    counts = np.bincount(indices, minlength=len(edges) - 1)
    mean_speed = np.full(len(counts), np.nan)
    median_speed = np.full(len(counts), np.nan)
    for index in range(len(counts)):
        values = speeds[indices == index]
        if values.size:
            mean_speed[index] = np.mean(values)
            median_speed[index] = np.median(values)

    full_counts = np.concatenate((counts, counts[::-1], counts, counts[::-1]))
    full_mean_speed = np.concatenate(
        (mean_speed, mean_speed[::-1], mean_speed, mean_speed[::-1])
    )
    full_median_speed = np.concatenate(
        (median_speed, median_speed[::-1], median_speed, median_speed[::-1])
    )
    full_edges = np.arange(0.0, 360.0 + AZIMUTH_BIN_SIZE, AZIMUTH_BIN_SIZE)
    centers = (full_edges[:-1] + full_edges[1:]) / 2.0

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].bar(
        centers,
        full_counts,
        width=AZIMUTH_BIN_SIZE * 0.82,
        color="#4c78a8",
        edgecolor="white",
    )
    axes[0].set_ylabel("数量 / 全周等效")
    axes[0].set_title(
        f"{result.case_name} 径向方位角分布（t = {frame.time_ms * 1000:.1f} μs）"
    )
    axes[1].plot(centers, full_mean_speed, "o-", label="平均值", color="#d1495b")
    axes[1].plot(
        centers,
        full_median_speed,
        "s--",
        label="中位数",
        color="#2a9d8f",
    )
    axes[1].set_xlim(0.0, 360.0)
    axes[1].set_xticks(np.arange(0.0, 361.0, 30.0))
    axes[1].set_xlabel("径向方位角 θ / °")
    axes[1].set_ylabel(r"速度 / (m s$^{-1}$)")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(color="0.9", linewidth=0.6)
    named_angles = {
        0.0: "长轴 +x",
        90.0: "短轴 +y",
        180.0: "长轴 -x",
        270.0: "短轴 -y",
    }
    for axis in axes:
        for angle in named_angles:
            axis.axvline(angle, color="0.65", linewidth=0.7, linestyle=":")
    axes[0].set_xticks(np.arange(0.0, 361.0, 30.0))
    axes[0].set_xticklabels(
        [named_angles.get(float(value), "") for value in axes[0].get_xticks()]
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def azimuth_profile_rows(
    frame: FrameData,
) -> list[dict[str, float | int | str]]:
    valid = frame.valid
    azimuth = frame.azimuth0_deg[valid]
    speeds = frame.speeds[valid]
    edges = np.arange(0.0, 90.0 + AZIMUTH_BIN_SIZE, AZIMUTH_BIN_SIZE)
    rows: list[dict[str, float | int | str]] = []
    for index in range(len(edges) - 1):
        values = speeds[(azimuth >= edges[index]) & (azimuth < edges[index + 1])]
        rows.append(
            {
                "time_ms": frame.time_ms,
                "angle_start_deg": edges[index],
                "angle_end_deg": edges[index + 1],
                "angle_center_deg": (edges[index] + edges[index + 1]) / 2.0,
                "count_quarter": int(values.size),
                "fraction_quarter": (
                    float(values.size / np.count_nonzero(valid)) if np.any(valid) else 0.0
                ),
                "speed_mean_mps": float(np.mean(values)) if values.size else np.nan,
                "speed_median_mps": (
                    float(np.median(values)) if values.size else np.nan
                ),
                "speed_p95_mps": (
                    float(np.quantile(values, 0.95)) if values.size else np.nan
                ),
            }
        )
    return rows


def axial_profile_data(
    frame: FrameData,
) -> tuple[list[dict[str, float | int | str]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    valid = frame.valid
    z0 = frame.initial_positions[valid, 2]
    azimuth = frame.azimuth0_deg[valid]
    speeds = frame.speeds[valid]
    zmin = max(-10.0, np.floor(float(np.min(z0)) / 10.0) * 10.0)
    zmax = min(120.0, np.ceil(float(np.max(z0)) / 10.0) * 10.0)
    zmax = max(zmax, zmin + 10.0)
    edges = np.arange(zmin, zmax + 10.0, 10.0)
    sectors = (
        ("长轴附近", 0.0, 20.0),
        ("对角附近", 35.0, 55.0),
        ("短轴附近", 70.0, 90.0),
    )
    rows: list[dict[str, float | int | str]] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, angle_min, angle_max in sectors:
        sector = (azimuth >= angle_min) & (azimuth <= angle_max)
        centers: list[float] = []
        medians: list[float] = []
        for index in range(len(edges) - 1):
            upper_test = z0 <= edges[index + 1]
            lower_test = z0 >= edges[index]
            values = speeds[sector & lower_test & upper_test]
            rows.append(
                {
                    "time_ms": frame.time_ms,
                    "azimuth_sector": label,
                    "angle_start_deg": angle_min,
                    "angle_end_deg": angle_max,
                    "z_start_mm": edges[index],
                    "z_end_mm": edges[index + 1],
                    "z_center_mm": (edges[index] + edges[index + 1]) / 2.0,
                    "count": int(values.size),
                    "speed_mean_mps": float(np.mean(values)) if values.size else np.nan,
                    "speed_median_mps": (
                        float(np.median(values)) if values.size else np.nan
                    ),
                    "speed_p95_mps": (
                        float(np.quantile(values, 0.95)) if values.size else np.nan
                    ),
                }
            )
            if values.size >= 3:
                centers.append((edges[index] + edges[index + 1]) / 2.0)
                medians.append(float(np.median(values)))
        curves[label] = (np.asarray(centers), np.asarray(medians))
    return rows, curves


def plot_axial_profile(
    result: SimulationResult,
    output_path: Path,
) -> list[dict[str, float | int | str]]:
    usable_frames = [frame for frame in result.frames if frame.valid_count]
    if not usable_frames:
        raise ValueError(f"{result.case_name}: no valid frame for axial profile")
    frame = usable_frames[-1]
    rows, curves = axial_profile_data(frame)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), sharey=True, constrained_layout=True)
    colors = ("#d1495b", "#2a9d8f", "#457b9d")
    for axis, color, (label, (centers, medians)) in zip(
        axes, colors, curves.items(), strict=True
    ):
        axis.plot(centers, medians, "o-", color=color, markersize=4, linewidth=1.5)
        axis.set_title(label)
        axis.set_xlabel(r"初始轴向位置 $z_0$ / mm")
        axis.grid(color="0.9", linewidth=0.6)
    axes[0].set_ylabel(r"破片速度中位数 / (m s$^{-1}$)")
    fig.suptitle(
        f"{result.case_name} 轴向速度分布（t = {frame.time_ms * 1000:.1f} μs）"
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return rows


def write_frames_csv(result: SimulationResult, path: Path) -> None:
    fields = [
        "case",
        "model",
        "time_ms",
        "time_us",
        "entity_id",
        "valid",
        "connected_fragment_id",
        "x0_mm",
        "y0_mm",
        "z0_mm",
        "x_mm",
        "y_mm",
        "z_mm",
        "radius_from_axis_mm",
        "azimuth0_deg",
        "velocity_polar_angle_deg",
        "vx_mps",
        "vy_mps",
        "vz_mps",
        "speed_mps",
        "mass_g",
        "sph_radius_mm",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame in result.frames:
            if frame.connected is not None:
                fragment_ids = np.full(frame.active_count, -1, dtype=int)
                fragment_ids[np.flatnonzero(frame.valid)] = (
                    frame.connected.fragment_ids
                )
            else:
                fragment_ids = np.full(frame.active_count, -1, dtype=int)
            for index in range(frame.active_count):
                initial = frame.initial_positions[index]
                position = frame.positions[index]
                velocity = frame.velocities[index]
                writer.writerow(
                    {
                        "case": result.case_name,
                        "model": result.kind,
                        "time_ms": f"{frame.time_ms:.9f}",
                        "time_us": f"{frame.time_ms * 1000.0:.6f}",
                        "entity_id": int(frame.entity_ids[index]),
                        "valid": bool(frame.valid[index]),
                        "connected_fragment_id": int(fragment_ids[index]),
                        "x0_mm": f"{initial[0]:.9g}",
                        "y0_mm": f"{initial[1]:.9g}",
                        "z0_mm": f"{initial[2]:.9g}",
                        "x_mm": f"{position[0]:.9g}",
                        "y_mm": f"{position[1]:.9g}",
                        "z_mm": f"{position[2]:.9g}",
                        "radius_from_axis_mm": f"{np.hypot(position[0], position[1]):.9g}",
                        "azimuth0_deg": f"{frame.azimuth0_deg[index]:.9g}",
                        "velocity_polar_angle_deg": f"{frame.polar_deg[index]:.9g}",
                        "vx_mps": f"{velocity[0]:.9g}",
                        "vy_mps": f"{velocity[1]:.9g}",
                        "vz_mps": f"{velocity[2]:.9g}",
                        "speed_mps": f"{frame.speeds[index]:.9g}",
                        "mass_g": (
                            f"{frame.mass_g[index]:.9g}"
                            if frame.mass_g is not None
                            else ""
                        ),
                        "sph_radius_mm": (
                            f"{frame.radius_mm[index]:.9g}"
                            if frame.radius_mm is not None
                            else ""
                        ),
                    }
                )


def write_frames_npz(result: SimulationResult, path: Path) -> None:
    payload: dict[str, np.ndarray] = {
        "times_ms": np.array([frame.time_ms for frame in result.frames]),
        "frame_index": np.concatenate(
            [
                np.full(frame.active_count, index, dtype=np.int32)
                for index, frame in enumerate(result.frames)
            ]
        )
        if any(frame.active_count for frame in result.frames)
        else np.empty(0, dtype=np.int32),
    }
    for attribute in (
        "entity_ids",
        "initial_positions",
        "positions",
        "velocities",
        "speeds",
        "valid",
        "azimuth0_deg",
        "polar_deg",
    ):
        payload[attribute] = np.concatenate(
            [np.asarray(getattr(frame, attribute)) for frame in result.frames]
        )
    payload["mass_g"] = np.concatenate(
        [
            frame.mass_g
            if frame.mass_g is not None
            else np.full(frame.active_count, np.nan)
            for frame in result.frames
        ]
    )
    payload["sph_radius_mm"] = np.concatenate(
        [
            frame.radius_mm
            if frame.radius_mm is not None
            else np.full(frame.active_count, np.nan)
            for frame in result.frames
        ]
    )
    np.savez_compressed(path, **payload)


def write_connected_fragments(result: SimulationResult, path: Path) -> None:
    fields = [
        "time_ms",
        "time_us",
        "fragment_id",
        "element_count",
        "x0_mm",
        "y0_mm",
        "z0_mm",
        "azimuth0_deg",
        "x_mm",
        "y_mm",
        "z_mm",
        "vx_mps",
        "vy_mps",
        "vz_mps",
        "speed_mps",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame in result.frames:
            if frame.connected is None:
                continue
            for fragment_id, (
                initial_position,
                position,
                velocity,
                speed,
                n_elements,
                azimuth0,
            ) in enumerate(
                zip(
                    frame.connected.initial_positions,
                    frame.connected.positions,
                    frame.connected.velocities,
                    frame.connected.speeds,
                    frame.connected.element_counts,
                    frame.connected.azimuth0_deg,
                    strict=True,
                )
            ):
                writer.writerow(
                    {
                        "time_ms": f"{frame.time_ms:.9f}",
                        "time_us": f"{frame.time_ms * 1000.0:.6f}",
                        "fragment_id": fragment_id,
                        "element_count": int(n_elements),
                        "x0_mm": f"{initial_position[0]:.9g}",
                        "y0_mm": f"{initial_position[1]:.9g}",
                        "z0_mm": f"{initial_position[2]:.9g}",
                        "azimuth0_deg": f"{azimuth0:.9g}",
                        "x_mm": f"{position[0]:.9g}",
                        "y_mm": f"{position[1]:.9g}",
                        "z_mm": f"{position[2]:.9g}",
                        "vx_mps": f"{velocity[0]:.9g}",
                        "vy_mps": f"{velocity[1]:.9g}",
                        "vz_mps": f"{velocity[2]:.9g}",
                        "speed_mps": f"{speed:.9g}",
                    }
                )


def frame_statistics(frame: FrameData) -> dict[str, float | int | None]:
    speeds = frame.speeds[frame.valid]
    if speeds.size:
        quantiles = np.quantile(speeds, [0.0, 0.05, 0.5, 0.95, 1.0])
    else:
        quantiles = np.full(5, np.nan)
    def optional_float(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "active_count": frame.active_count,
        "valid_count": frame.valid_count,
        "filtered_count": frame.active_count - frame.valid_count,
        "connected_fragment_count": (
            len(frame.connected.element_counts)
            if frame.connected is not None
            else None
        ),
        "speed_min_mps": optional_float(quantiles[0]),
        "speed_p05_mps": optional_float(quantiles[1]),
        "speed_median_mps": optional_float(quantiles[2]),
        "speed_mean_mps": optional_float(np.mean(speeds)) if speeds.size else None,
        "speed_p95_mps": optional_float(quantiles[3]),
        "speed_max_mps": optional_float(quantiles[4]),
    }


def write_count_history(result: SimulationResult, path: Path) -> None:
    selected = {round(frame.time_ms, 9): frame for frame in result.frames}
    fields = [
        "case",
        "model",
        "time_ms",
        "time_us",
        "active_entity_count",
        "is_plot_frame",
        "valid_entity_count",
        "filtered_entity_count",
        "connected_fragment_count",
        "speed_median_mps",
        "speed_mean_mps",
        "speed_p95_mps",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for time, active_count in zip(
            result.all_times_ms, result.activity_counts, strict=True
        ):
            frame = selected.get(round(float(time), 9))
            stats = frame_statistics(frame) if frame is not None else {}
            writer.writerow(
                {
                    "case": result.case_name,
                    "model": result.kind,
                    "time_ms": f"{time:.9f}",
                    "time_us": f"{time * 1000.0:.6f}",
                    "active_entity_count": int(active_count),
                    "is_plot_frame": frame is not None,
                    "valid_entity_count": stats.get("valid_count", ""),
                    "filtered_entity_count": stats.get("filtered_count", ""),
                    "connected_fragment_count": stats.get(
                        "connected_fragment_count", ""
                    ),
                    "speed_median_mps": stats.get("speed_median_mps", ""),
                    "speed_mean_mps": stats.get("speed_mean_mps", ""),
                    "speed_p95_mps": stats.get("speed_p95_mps", ""),
                }
            )


def write_speed_bins(result: SimulationResult, path: Path) -> None:
    edges = np.linspace(0.0, DISPLAY_SPEED_MAX, 41)
    fields = [
        "case",
        "time_ms",
        "time_us",
        "speed_start_mps",
        "speed_end_mps",
        "count",
        "fraction_valid",
        "count_above_display_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame in result.frames:
            speeds = frame.speeds[frame.valid]
            counts, _ = np.histogram(speeds, bins=edges)
            total = len(speeds)
            above = int(np.count_nonzero(speeds > DISPLAY_SPEED_MAX))
            for index, count in enumerate(counts):
                writer.writerow(
                    {
                        "case": result.case_name,
                        "time_ms": f"{frame.time_ms:.9f}",
                        "time_us": f"{frame.time_ms * 1000.0:.6f}",
                        "speed_start_mps": f"{edges[index]:.9g}",
                        "speed_end_mps": f"{edges[index + 1]:.9g}",
                        "count": int(count),
                        "fraction_valid": count / total if total else "",
                        "count_above_display_max": above,
                    }
                )


def write_rows(
    rows: Iterable[dict[str, float | int | str]],
    path: Path,
) -> None:
    values = list(rows)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = list(values[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def write_case_outputs(
    result: SimulationResult,
    target_times_us: Sequence[float],
) -> dict[str, object]:
    out = result.output_dir
    plot_field(result, "axial", out / "axial_fragment_field.png")
    plot_field(result, "radial", out / "radial_fragment_field.png")
    plot_speed_and_count(result, out / "speed_and_count_distribution.png")
    plot_azimuth_distribution(result, out / "azimuth_distribution_final.png")
    axial_rows = plot_axial_profile(result, out / "axial_speed_profile_final.png")

    write_frames_csv(result, out / "fragment_points.csv")
    write_frames_npz(result, out / "fragment_points.npz")
    write_count_history(result, out / "fragment_count_history.csv")
    write_speed_bins(result, out / "fragment_speed_bins.csv")
    write_rows(
        (
            row
            for frame in result.frames
            for row in azimuth_profile_rows(frame)
        ),
        out / "radial_azimuth_statistics.csv",
    )
    write_rows(axial_rows, out / "axial_speed_profile.csv")
    connected_path = out / "connected_fragments.csv"
    if result.kind == "no-sph":
        write_connected_fragments(result, connected_path)

    warnings: list[str] = []
    if result.kind == "sph":
        if result.sph_deletion_nonzero_count == 0:
            warnings.append(
                "d3plot 中 sph_deletion 全程为 0；因此以 sph_radius>0、"
                "sph_mass>0 判定活动粒子，而不是只使用 deletion 标志。"
            )
        warnings.append(
            f"速度有效性阈值采用 {result.speed_max:g} m/s。超出阈值的活动粒子"
            "仍保留在 CSV 中并标记 valid=False，不参与速度场和分布统计。"
        )
        warnings.append(
            "SPH 速度尾部存在数十万至数亿 m/s 的非物理值；这些值对应的粒子"
            "应从物理统计中排除。"
        )
        warnings.append(
            "LS-DYNA 的 sldsph_frag 文本条数与 d3plot 活动粒子数明显不一致，"
            "可能只含最后输出周期/局部新增粒子，不能直接当作累计活动粒子数；"
            "本报告以 d3plot 的活动粒子数组为准。"
        )
    else:
        warnings.append(
            "无 SPH 模型中每个存活壳体实体单元作为一个场采样点；"
            "connected_fragments.csv 另给出按共享节点识别的连通破片簇。"
        )
    warnings.append(
        "模型单位制为 mm-g-ms；1 mm/ms 与 1 m/s 数值相同，因此速度列标签使用 m/s。"
    )
    warnings.append(
        "1/4 模型图像按 x=0、y=0 两个对称面扩展显示；统计 CSV 保留原始 1/4 数据。"
    )
    warnings.append(
        "图中的“数量”是离散场采样实体数（SPH 粒子或存活单元），"
        "不等同于靶板穿孔数；无 SPH 模型的连通破片簇数量见单独 CSV。"
    )

    summary = {
        "case": result.case_name,
        "model": result.kind,
        "d3plot": str(result.d3plot.resolve()),
        "state_count": len(result.all_times_ms),
        "time_start_ms": float(result.all_times_ms[0]),
        "time_end_ms": float(result.all_times_ms[-1]),
        "requested_times_us": list(target_times_us),
        "selected_times_us": [frame.time_ms * 1000.0 for frame in result.frames],
        "speed_validity_max_mps": result.speed_max,
        "speed_display_max_mps": DISPLAY_SPEED_MAX,
        "sph_deletion_nonzero_count": result.sph_deletion_nonzero_count,
        "frames": [
            {
                "time_ms": frame.time_ms,
                "time_us": frame.time_ms * 1000.0,
                **frame_statistics(frame),
            }
            for frame in result.frames
        ],
        "display_extent": display_limits(result.frames),
        "warnings": warnings,
        "outputs": {
            "axial_field_png": str((out / "axial_fragment_field.png").resolve()),
            "radial_field_png": str(
                (out / "radial_fragment_field.png").resolve()
            ),
            "speed_count_png": str(
                (out / "speed_and_count_distribution.png").resolve()
            ),
            "azimuth_png": str(
                (out / "azimuth_distribution_final.png").resolve()
            ),
            "axial_profile_png": str(
                (out / "axial_speed_profile_final.png").resolve()
            ),
            "fragment_points_csv": str((out / "fragment_points.csv").resolve()),
            "fragment_points_npz": str((out / "fragment_points.npz").resolve()),
            "count_history_csv": str(
                (out / "fragment_count_history.csv").resolve()
            ),
            "speed_bins_csv": str((out / "fragment_speed_bins.csv").resolve()),
            "connected_fragments_csv": (
                str(connected_path.resolve()) if result.kind == "no-sph" else None
            ),
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_case_readme(result, summary)
    return summary


def write_case_readme(result: SimulationResult, summary: dict[str, object]) -> None:
    out = result.output_dir
    text = f"""# {result.case_name} 后处理结果

输入：`{result.d3plot}`

模型：{result.kind}

时间范围：{float(result.all_times_ms[0]) * 1000:.3f}–{float(result.all_times_ms[-1]) * 1000:.3f} μs。

## 主要图件

### 轴向破片场

![轴向破片场](axial_fragment_field.png)

### 径向破片场

![径向破片场](radial_fragment_field.png)

### 速度与数量

![速度与数量](speed_and_count_distribution.png)

### 径向方位角

![径向方位角](azimuth_distribution_final.png)

### 轴向速度分布

![轴向速度分布](axial_speed_profile_final.png)

## 数据文件

- `fragment_points.csv` / `.npz`：所有活动实体的位置、速度、初始位置、
  方位角、有效性、质量/半径（SPH）或连通破片编号（无 SPH）。
- `fragment_count_history.csv`：全时间轴活动实体数量；抽样帧另含有效数量和速度统计。
- `fragment_speed_bins.csv`：各抽样帧 0–{DISPLAY_SPEED_MAX:g} m/s 速度分箱。
- `radial_azimuth_statistics.csv`：各抽样帧方位角分箱数量和速度。
- `axial_speed_profile.csv`：末个有效帧三个方位角区间的轴向速度分箱。
- `connected_fragments.csv`：无 SPH 模型按共享节点识别的连通破片簇。
- `summary.json`：输入、筛选阈值、逐帧统计和警告。

## 统计口径

- 无 SPH：存活结构实体单元质心作为场采样点；连通簇另按单元共享节点识别。
- SPH：`sph_radius>0`、`sph_mass>0`、`sph_deletion==0` 为活动粒子。
- 速度有效性：有限速度且 `|v| <= {result.speed_max:g} m/s`。
- 单位：坐标 mm，时间 ms/μs，速度 m/s。模型单位 mm/ms 与 m/s 的数值相同。
- 图中速度颜色范围 0–{DISPLAY_SPEED_MAX:g} m/s；CSV 保留全部有效速度。

## 数据质量说明

{chr(10).join(f"- {warning}" for warning in summary["warnings"])}
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def write_comparison(
    summaries: Sequence[dict[str, object]],
    output_root: Path,
) -> None:
    fields = [
        "case",
        "model",
        "time_us",
        "active_count",
        "valid_count",
        "filtered_count",
        "connected_fragment_count",
        "speed_mean_mps",
        "speed_median_mps",
        "speed_p95_mps",
        "speed_max_mps",
    ]
    with (output_root / "comparison_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            for frame in summary["frames"]:
                writer.writerow(
                    {
                        "case": summary["case"],
                        "model": summary["model"],
                        "time_us": f"{frame['time_us']:.6f}",
                        "active_count": frame["active_count"],
                        "valid_count": frame["valid_count"],
                        "filtered_count": frame["filtered_count"],
                        "connected_fragment_count": (
                            frame["connected_fragment_count"]
                            if frame["connected_fragment_count"] is not None
                            else ""
                        ),
                        "speed_mean_mps": frame["speed_mean_mps"],
                        "speed_median_mps": frame["speed_median_mps"],
                        "speed_p95_mps": frame["speed_p95_mps"],
                        "speed_max_mps": frame["speed_max_mps"],
                    }
                )
    lines = [
        "# 椭圆变截面战斗部 1/4 模型后处理",
        "",
        "本目录由 `tools/postprocess_elliptical_results.py` 生成。两个模型分别统计，"
        "图中的 1/4 几何已按双平面对称扩展。",
        "",
        "## 算例",
        "",
        "| 算例 | 末帧活动实体 | 末帧有效实体 | 末帧连通簇 | 中位速度 (m/s) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        frame = summary["frames"][-1]
        lines.append(
            "| {case} | {active_count:,} | {valid_count:,} | {connected} | "
            "{median:.1f} |".format(
                case=summary["case"],
                active_count=frame["active_count"],
                valid_count=frame["valid_count"],
                connected=(
                    frame["connected_fragment_count"]
                    if frame["connected_fragment_count"] is not None
                    else "—"
                ),
                median=frame["speed_median_mps"],
            )
        )
    lines.extend(
        [
            "",
            "## 查看结果",
            "",
            "- [无 SPH 结果](no-sph/README.md)",
            "- [SPH 结果](sph/README.md)",
            "- [两算例逐帧统计对比](comparison_summary.csv)",
            "",
            "## 复跑",
            "",
            "```powershell",
            ".venv\\Scripts\\python.exe tools\\postprocess_elliptical_results.py",
            "```",
            "",
            "关键统计口径和异常提示见各算例目录的 `README.md` 与 `summary.json`。",
            "",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-process the no-SPH and SPH quarter warhead simulations."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="directory containing no-sph-quarter and sph-quarter",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "elliptical20261030",
    )
    parser.add_argument(
        "--case",
        choices=("no-sph", "sph", "both"),
        default="both",
    )
    parser.add_argument(
        "--times-us",
        default=",".join(f"{value:g}" for value in DEFAULT_TIMES_US),
        help="comma-separated post-processing times in microseconds",
    )
    parser.add_argument(
        "--speed-max",
        type=float,
        default=DEFAULT_SPEED_MAX,
        help="hard physical speed-validity threshold in m/s",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target_times = tuple(
        float(value.strip()) for value in args.times_us.split(",") if value.strip()
    )
    if not target_times:
        raise ValueError("at least one post-processing time is required")
    if args.speed_max <= 0.0:
        raise ValueError("--speed-max must be positive")
    configure_matplotlib()
    args.output_root.mkdir(parents=True, exist_ok=True)

    cases = ("no-sph", "sph") if args.case == "both" else (args.case,)
    summaries: list[dict[str, object]] = []
    for kind in cases:
        case_name = "无 SPH（单元失效删除）" if kind == "no-sph" else "SPH（自适应粒子）"
        d3plot = args.input_root / f"{kind}-quarter" / "d3plot"
        if not d3plot.is_file():
            raise FileNotFoundError(f"d3plot not found: {d3plot}")
        output_dir = args.output_root / kind
        print(f"Reading {case_name}: {d3plot}")
        result = analyze_case(
            case_name=case_name,
            kind=kind,
            d3plot=d3plot,
            output_dir=output_dir,
            target_times_us=target_times,
            speed_max=args.speed_max,
        )
        summary = write_case_outputs(result, target_times)
        summaries.append(summary)
        print(f"Wrote {output_dir}")

    write_comparison(summaries, args.output_root)
    print(f"Comparison: {args.output_root / 'comparison_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
