"""STL/GLB surface loading and equal-sized voxel mesh generation."""

from __future__ import annotations

import json
import math
import re
import struct
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .KCard import KCard
from .KeywordFile import KeywordFile, Vec3
from .KWord import KWord

Triangle = tuple[Vec3, Vec3, Vec3]
Cell = tuple[int, int, int]

_ASCII_VERTEX_RE = re.compile(
    r"^\s*vertex\s+"
    r"([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)",
    re.IGNORECASE | re.MULTILINE,
)
_HEX_OFFSETS = (
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
)


@dataclass
class SurfaceMesh:
    """A triangulated closed surface in the source coordinate system."""

    triangles: list[Triangle]
    name: str = ""

    def __post_init__(self) -> None:
        self.triangles = [
            triangle for triangle in self.triangles
            if _triangle_area_squared(triangle) > 0.0
        ]
        if not self.triangles:
            raise ValueError("surface mesh contains no non-degenerate triangles")

    @property
    def bounds(self) -> tuple[Vec3, Vec3]:
        points = [point for triangle in self.triangles for point in triangle]
        lower = tuple(min(point[axis] for point in points) for axis in range(3))
        upper = tuple(max(point[axis] for point in points) for axis in range(3))
        return (lower, upper)  # type: ignore[return-value]

    def scaled(self, factor: float) -> "SurfaceMesh":
        """Return a uniformly scaled copy of the surface."""
        factor = float(factor)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("scale factor must be a finite positive value")
        return SurfaceMesh(
            triangles=[
                tuple(
                    tuple(value * factor for value in point)
                    for point in triangle
                )
                for triangle in self.triangles
            ],  # type: ignore[list-item]
            name=self.name,
        )


@dataclass
class VoxelRegion:
    """One PID's occupied cells in a shared regular voxel grid."""

    pid: int
    title: str
    cells: set[Cell] = field(default_factory=set)


@dataclass
class VoxelGeometry:
    """Voxelized occupancy split into one or more geometry regions."""

    regions: list[VoxelRegion]
    origin: Vec3
    divisions: Cell
    element_size: float
    bounds: tuple[Vec3, Vec3]
    mode: str
    shell_layers: int = 0
    open_surface: bool = False

    @property
    def voxel_count(self) -> int:
        return sum(len(region.cells) for region in self.regions)

    @property
    def effective_shell_thickness(self) -> float:
        return self.shell_layers * self.element_size

    def region(self, pid: int) -> VoxelRegion:
        for region in self.regions:
            if region.pid == pid:
                return region
        raise KeyError(f"no voxel region with PID {pid}")

    def cell_center(self, cell: Cell) -> Vec3:
        size = self.element_size
        return tuple(
            self.origin[axis] + (cell[axis] + 0.5) * size
            for axis in range(3)
        )  # type: ignore[return-value]

    def detonation_point(self, pid: int) -> Vec3:
        """Return the occupied cell center nearest to the region centroid."""
        region = self.region(pid)
        if not region.cells:
            raise ValueError(f"PID {pid} has no voxels")
        centers = [self.cell_center(cell) for cell in region.cells]
        centroid = tuple(
            sum(point[axis] for point in centers) / len(centers)
            for axis in range(3)
        )
        return min(
            centers,
            key=lambda point: sum(
                (point[axis] - centroid[axis]) ** 2 for axis in range(3)
            ),
        )

    def contains_point(self, pid: int, point: Sequence[float]) -> bool:
        """Return whether a point lies in one voxel occupied by the PID."""
        if len(point) != 3 or not all(math.isfinite(float(value)) for value in point):
            return False
        cell = tuple(
            math.floor((float(point[axis]) - self.origin[axis])
                       / self.element_size)
            for axis in range(3)
        )
        if any(
            cell[axis] < 0 or cell[axis] >= self.divisions[axis]
            for axis in range(3)
        ):
            return False
        return cell in self.region(pid).cells


def _parse_float(text: str) -> float:
    return float(text.replace("d", "e").replace("D", "E"))


def _triangle_area_squared(triangle: Triangle) -> float:
    p0, p1, p2 = triangle
    ux, uy, uz = (p1[i] - p0[i] for i in range(3))
    vx, vy, vz = (p2[i] - p0[i] for i in range(3))
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    return cx * cx + cy * cy + cz * cz


def _finite_triangle(triangle: Triangle) -> bool:
    return all(math.isfinite(value) for point in triangle for value in point)


def surface_is_watertight(
    surface: SurfaceMesh,
    *,
    weld_digits: int = 9,
) -> bool:
    """Return whether every welded triangle edge belongs to exactly two faces."""
    vertex_ids: dict[tuple[float, float, float], int] = {}
    triangle_ids: list[tuple[int, int, int]] = []
    for triangle in surface.triangles:
        ids = []
        for point in triangle:
            key = tuple(round(value, weld_digits) for value in point)
            vertex_id = vertex_ids.get(key)
            if vertex_id is None:
                vertex_id = len(vertex_ids)
                vertex_ids[key] = vertex_id
            ids.append(vertex_id)
        triangle_ids.append((ids[0], ids[1], ids[2]))

    edge_counts: Counter[tuple[int, int]] = Counter()
    for ids in triangle_ids:
        for left, right in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            edge_counts[tuple(sorted((left, right)))] += 1
    return bool(edge_counts) and all(count == 2 for count in edge_counts.values())


def load_surface_mesh(path: str | Path) -> SurfaceMesh:
    """Load a binary/ASCII STL or binary glTF 2.0 (``.glb``) surface."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return load_stl(path)
    if suffix == ".glb":
        return load_glb(path)
    raise ValueError(f"unsupported geometry format {suffix!r}; use .stl or .glb")


def load_stl(path: str | Path) -> SurfaceMesh:
    """Load an ASCII or binary STL file."""
    path = Path(path)
    data = path.read_bytes()
    triangles: list[Triangle] = []

    is_binary = False
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        is_binary = len(data) == 84 + triangle_count * 50

    if is_binary:
        for index in range(triangle_count):
            offset = 84 + index * 50 + 12
            values = struct.unpack_from("<9f", data, offset)
            triangle = (
                (values[0], values[1], values[2]),
                (values[3], values[4], values[5]),
                (values[6], values[7], values[8]),
            )
            if _finite_triangle(triangle):
                triangles.append(triangle)
    else:
        text = data.decode("utf-8", errors="replace")
        coordinates = [
            (_parse_float(match.group(1)),
             _parse_float(match.group(2)),
             _parse_float(match.group(3)))
            for match in _ASCII_VERTEX_RE.finditer(text)
        ]
        if len(coordinates) % 3:
            raise ValueError(f"{path}: ASCII STL vertex count is not divisible by 3")
        triangles = [
            (coordinates[index], coordinates[index + 1], coordinates[index + 2])
            for index in range(0, len(coordinates), 3)
            if _finite_triangle((
                coordinates[index], coordinates[index + 1], coordinates[index + 2]
            ))
        ]

    return SurfaceMesh(triangles=triangles, name=path.stem)


def _mat_identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mat_multiply(left: Sequence[Sequence[float]],
                  right: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [
            sum(left[row][k] * right[k][column] for k in range(4))
            for column in range(4)
        ]
        for row in range(4)
    ]


def _node_matrix(node: dict) -> list[list[float]]:
    if "matrix" in node:
        values = [float(value) for value in node["matrix"]]
        if len(values) != 16:
            raise ValueError("glTF node matrix must have 16 values")
        return [
            [values[0], values[4], values[8], values[12]],
            [values[1], values[5], values[9], values[13]],
            [values[2], values[6], values[10], values[14]],
            [values[3], values[7], values[11], values[15]],
        ]

    tx, ty, tz = (float(value) for value in node.get("translation", (0, 0, 0)))
    qx, qy, qz, qw = (float(value) for value in node.get("rotation", (0, 0, 0, 1)))
    sx, sy, sz = (float(value) for value in node.get("scale", (1, 1, 1)))

    x2, y2, z2 = 2.0 * qx, 2.0 * qy, 2.0 * qz
    xx, xy, xz = qx * x2, qx * y2, qx * z2
    yy, yz, zz = qy * y2, qy * z2, qz * z2
    wx, wy, wz = qw * x2, qw * y2, qw * z2
    return [
        [1 - (yy + zz), xy - wz, xz + wy, tx],
        [xy + wz, 1 - (xx + zz), yz - wx, ty],
        [xz - wy, yz + wx, 1 - (xx + yy), tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _apply_matrix(matrix: Sequence[Sequence[float]],
                  point: Sequence[float]) -> Vec3:
    x, y, z = point
    return tuple(
        matrix[row][0] * x + matrix[row][1] * y
        + matrix[row][2] * z + matrix[row][3]
        for row in range(3)
    )  # type: ignore[return-value]


_COMPONENT_FORMATS = {
    5120: "b",
    5121: "B",
    5122: "h",
    5123: "H",
    5125: "I",
    5126: "f",
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _accessor_values(document: dict, binary: bytes, accessor_index: int) -> list:
    accessor = document["accessors"][accessor_index]
    if "sparse" in accessor:
        raise ValueError("sparse glTF accessors are not supported")
    view = document["bufferViews"][accessor["bufferView"]]
    if int(view.get("buffer", 0)) != 0:
        raise ValueError("only the GLB embedded binary buffer is supported")

    component_format = _COMPONENT_FORMATS[int(accessor["componentType"])]
    component_count = _TYPE_COMPONENTS[str(accessor["type"])]
    component_size = struct.calcsize("<" + component_format)
    item_size = component_size * component_count
    stride = int(view.get("byteStride", item_size))
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    count = int(accessor["count"])
    unpack_format = "<" + component_format * component_count
    if offset + (count - 1) * stride + item_size > len(binary):
        raise ValueError("glTF accessor exceeds the embedded binary chunk")

    values: list = []
    for index in range(count):
        item = struct.unpack_from(unpack_format, binary, offset + index * stride)
        values.append(item[0] if component_count == 1 else item)
    return values


def _glb_chunks(data: bytes) -> tuple[dict, bytes]:
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError("not a binary glTF 2.0 file")
    version, total_length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise ValueError(f"unsupported glTF version {version}")
    if total_length > len(data):
        raise ValueError("truncated GLB file")

    json_chunk: bytes | None = None
    binary_chunk = b""
    offset = 12
    while offset + 8 <= total_length:
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        start = offset + 8
        end = start + chunk_length
        if end > total_length:
            raise ValueError("truncated GLB chunk")
        if chunk_type == 0x4E4F534A:
            json_chunk = data[start:end]
        elif chunk_type == 0x004E4942 and not binary_chunk:
            binary_chunk = data[start:end]
        offset = end

    if json_chunk is None:
        raise ValueError("GLB has no JSON chunk")
    document = json.loads(json_chunk.decode("utf-8").rstrip("\x00 "))
    return document, binary_chunk


def load_glb(path: str | Path) -> SurfaceMesh:
    """Load triangle primitives from a GLB with an embedded binary buffer."""
    path = Path(path)
    document, binary = _glb_chunks(path.read_bytes())
    triangles: list[Triangle] = []

    def add_node(node_index: int, parent_matrix: list[list[float]],
                 ancestors: set[int]) -> None:
        if node_index in ancestors:
            raise ValueError("cyclic node hierarchy in GLB")
        node = document["nodes"][node_index]
        matrix = _mat_multiply(parent_matrix, _node_matrix(node))
        mesh_index = node.get("mesh")
        if mesh_index is not None:
            mesh = document["meshes"][int(mesh_index)]
            for primitive in mesh.get("primitives", ()):
                if int(primitive.get("mode", 4)) != 4:
                    continue
                position_index = primitive.get("attributes", {}).get("POSITION")
                if position_index is None:
                    continue
                positions = [
                    _apply_matrix(matrix, point)
                    for point in _accessor_values(document, binary, int(position_index))
                ]
                if "indices" in primitive:
                    indices = [
                        int(value)
                        for value in _accessor_values(
                            document, binary, int(primitive["indices"])
                        )
                    ]
                else:
                    indices = list(range(len(positions)))
                if len(indices) % 3:
                    raise ValueError("GLB triangle index count is not divisible by 3")
                for index in range(0, len(indices), 3):
                    triangle = (
                        positions[indices[index]],
                        positions[indices[index + 1]],
                        positions[indices[index + 2]],
                    )
                    if _finite_triangle(triangle):
                        triangles.append(triangle)

        next_ancestors = ancestors | {node_index}
        for child_index in node.get("children", ()):
            add_node(int(child_index), matrix, next_ancestors)

    nodes = document.get("nodes", ())
    scenes = document.get("scenes", ())
    if scenes:
        scene_index = int(document.get("scene", 0))
        roots = scenes[scene_index].get("nodes", ())
    else:
        children = {
            int(child)
            for node in nodes
            for child in node.get("children", ())
        }
        roots = [index for index in range(len(nodes)) if index not in children]

    for root_index in roots:
        add_node(int(root_index), _mat_identity(), set())

    return SurfaceMesh(triangles=triangles, name=path.stem)


def _inside_cells(surface: SurfaceMesh, origin: Vec3, divisions: Cell,
                  element_size: float) -> set[Cell]:
    """Voxelize the closed volume by filling ray-casting intervals."""
    weighted_normals = [0.0, 0.0, 0.0]
    for p0, p1, p2 in surface.triangles:
        ux, uy, uz = (p1[i] - p0[i] for i in range(3))
        vx, vy, vz = (p2[i] - p0[i] for i in range(3))
        cross = (
            uy * vz - uz * vy,
            uz * vx - ux * vz,
            ux * vy - uy * vx,
        )
        for axis in range(3):
            weighted_normals[axis] += abs(cross[axis])

    ray_axis = max(range(3), key=weighted_normals.__getitem__)
    projected_axes = [axis for axis in range(3) if axis != ray_axis]
    u_axis, v_axis = projected_axes
    hits: dict[tuple[int, int], list[float]] = {}

    for triangle in surface.triangles:
        p0, p1, p2 = triangle
        x0, y0 = p0[u_axis], p0[v_axis]
        x1, y1 = p1[u_axis], p1[v_axis]
        x2, y2 = p2[u_axis], p2[v_axis]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denominator) <= 1e-15:
            continue

        min_u, max_u = min(x0, x1, x2), max(x0, x1, x2)
        min_v, max_v = min(y0, y1, y2), max(y0, y1, y2)
        i_start = max(
            0, math.ceil((min_u - origin[u_axis]) / element_size - 0.5)
        )
        i_stop = min(
            divisions[u_axis] - 1,
            math.floor((max_u - origin[u_axis]) / element_size - 0.5),
        )
        j_start = max(
            0, math.ceil((min_v - origin[v_axis]) / element_size - 0.5)
        )
        j_stop = min(
            divisions[v_axis] - 1,
            math.floor((max_v - origin[v_axis]) / element_size - 0.5),
        )
        if i_start > i_stop or j_start > j_stop:
            continue

        for i in range(i_start, i_stop + 1):
            sample_u = origin[u_axis] + (i + 0.5) * element_size
            for j in range(j_start, j_stop + 1):
                sample_v = origin[v_axis] + (j + 0.5) * element_size
                weight0 = (
                    (y1 - y2) * (sample_u - x2)
                    + (x2 - x1) * (sample_v - y2)
                ) / denominator
                weight1 = (
                    (y2 - y0) * (sample_u - x2)
                    + (x0 - x2) * (sample_v - y2)
                ) / denominator
                weight2 = 1.0 - weight0 - weight1
                tolerance = 1e-9
                if min(weight0, weight1, weight2) < -tolerance:
                    continue
                ray_value = (
                    weight0 * p0[ray_axis]
                    + weight1 * p1[ray_axis]
                    + weight2 * p2[ray_axis]
                )
                if math.isfinite(ray_value):
                    hits.setdefault((i, j), []).append(ray_value)

    occupied: set[Cell] = set()
    merge_tolerance = max(element_size * 1e-8, 1e-10)
    for (i, j), intersections in hits.items():
        intersections.sort()
        unique: list[float] = []
        for value in intersections:
            if not unique or abs(value - unique[-1]) > merge_tolerance:
                unique.append(value)
        for lower, upper in zip(unique[0::2], unique[1::2]):
            if upper <= lower:
                continue
            k_start = max(
                0,
                math.ceil((lower - origin[ray_axis]) / element_size - 0.5),
            )
            k_stop = min(
                divisions[ray_axis] - 1,
                math.ceil((upper - origin[ray_axis]) / element_size - 0.5) - 1,
            )
            for k in range(k_start, k_stop + 1):
                cell = [0, 0, 0]
                cell[u_axis], cell[v_axis], cell[ray_axis] = i, j, k
                occupied.add(tuple(cell))
    return occupied


def _erode(cells: set[Cell], layers: int) -> set[Cell]:
    current = set(cells)
    for _ in range(layers):
        current = {
            cell for cell in current
            if all(
                (cell[0] + dx, cell[1] + dy, cell[2] + dz) in current
                for dx, dy, dz in (
                    (-1, 0, 0), (1, 0, 0),
                    (0, -1, 0), (0, 1, 0),
                    (0, 0, -1), (0, 0, 1),
                )
            )
        }
    return current


def _point_triangle_distance_squared(point: Vec3, triangle: Triangle) -> float:
    """Squared Euclidean distance from a point to a triangle."""
    a, b, c = triangle
    abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    acx, acy, acz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    apx, apy, apz = point[0] - a[0], point[1] - a[1], point[2] - a[2]
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return apx * apx + apy * apy + apz * apz

    bpx, bpy, bpz = point[0] - b[0], point[1] - b[1], point[2] - b[2]
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return bpx * bpx + bpy * bpy + bpz * bpz

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        value = d1 / (d1 - d3)
        qx = a[0] + value * abx - point[0]
        qy = a[1] + value * aby - point[1]
        qz = a[2] + value * abz - point[2]
        return qx * qx + qy * qy + qz * qz

    cpx, cpy, cpz = point[0] - c[0], point[1] - c[1], point[2] - c[2]
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return cpx * cpx + cpy * cpy + cpz * cpz

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        value = d2 / (d2 - d6)
        qx = a[0] + value * acx - point[0]
        qy = a[1] + value * acy - point[1]
        qz = a[2] + value * acz - point[2]
        return qx * qx + qy * qy + qz * qz

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        value = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        qx = b[0] + value * (c[0] - b[0]) - point[0]
        qy = b[1] + value * (c[1] - b[1]) - point[1]
        qz = b[2] + value * (c[2] - b[2]) - point[2]
        return qx * qx + qy * qy + qz * qz

    denominator = 1.0 / (va + vb + vc)
    value_b = vb * denominator
    value_c = vc * denominator
    qx = a[0] + abx * value_b + acx * value_c - point[0]
    qy = a[1] + aby * value_b + acy * value_c - point[1]
    qz = a[2] + abz * value_b + acz * value_c - point[2]
    return qx * qx + qy * qy + qz * qz


def _surface_shell_cells(
    surface: SurfaceMesh,
    origin: Vec3,
    divisions: Cell,
    element_size: float,
    shell_layers: int,
) -> set[Cell]:
    """Rasterize every triangle into a structural shell of target thickness."""
    radius = 0.5 * shell_layers * element_size
    radius_squared = radius * radius
    cells: set[Cell] = set()
    for triangle in surface.triangles:
        mins = [
            min(point[axis] for point in triangle) - radius
            for axis in range(3)
        ]
        maxs = [
            max(point[axis] for point in triangle) + radius
            for axis in range(3)
        ]
        starts = [
            max(0, math.ceil((mins[axis] - origin[axis]) / element_size - 0.5))
            for axis in range(3)
        ]
        stops = [
            min(
                divisions[axis] - 1,
                math.floor((maxs[axis] - origin[axis]) / element_size - 0.5),
            )
            for axis in range(3)
        ]
        if any(starts[axis] > stops[axis] for axis in range(3)):
            continue
        for i in range(starts[0], stops[0] + 1):
            x = origin[0] + (i + 0.5) * element_size
            for j in range(starts[1], stops[1] + 1):
                y = origin[1] + (j + 0.5) * element_size
                for k in range(starts[2], stops[2] + 1):
                    z = origin[2] + (k + 0.5) * element_size
                    if _point_triangle_distance_squared(
                        (x, y, z), triangle
                    ) <= radius_squared:
                        cells.add((i, j, k))
    return cells


def _flood_fill_enclosed_cells(
    divisions: Cell,
    shell: set[Cell],
) -> set[Cell]:
    """Fill all non-shell cells not connected to the grid boundary."""
    outside: set[Cell] = set()
    queue: deque[Cell] = deque()
    for i in range(divisions[0]):
        for j in range(divisions[1]):
            for k in range(divisions[2]):
                if (
                    i in (0, divisions[0] - 1)
                    or j in (0, divisions[1] - 1)
                    or k in (0, divisions[2] - 1)
                ):
                    cell = (i, j, k)
                    if cell not in shell:
                        outside.add(cell)
                        queue.append(cell)

    while queue:
        cell = queue.popleft()
        for dx, dy, dz in (
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ):
            candidate = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            if (
                0 <= candidate[0] < divisions[0]
                and 0 <= candidate[1] < divisions[1]
                and 0 <= candidate[2] < divisions[2]
                and candidate not in shell
                and candidate not in outside
            ):
                outside.add(candidate)
                queue.append(candidate)

    all_cells = {
        (i, j, k)
        for i in range(divisions[0])
        for j in range(divisions[1])
        for k in range(divisions[2])
    }
    return all_cells - shell - outside


def voxelize_surface(
    surface: SurfaceMesh,
    element_size: float | None = None,
    *,
    mode: str = "shell",
    shell_thickness: float | None = None,
    divisions_along_longest: int = 40,
    max_voxels: int = 2_000_000,
) -> VoxelGeometry:
    """Voxelize a closed surface into a solid or shell-plus-interior mesh.

    ``mode="solid"`` creates one occupied region with PID 1.
    ``mode="shell"`` creates a shell with PID 2 and an interior with PID 3.
    The shell thickness is rounded up to whole voxel layers.
    """
    if mode not in {"solid", "shell"}:
        raise ValueError("mode must be 'solid' or 'shell'")
    if divisions_along_longest < 2:
        raise ValueError("divisions_along_longest must be at least 2")

    lower, upper = surface.bounds
    dimensions = tuple(upper[axis] - lower[axis] for axis in range(3))
    longest = max(dimensions)
    if not longest > 0:
        raise ValueError("surface has zero extent")
    if element_size is None:
        element_size = longest / divisions_along_longest
    element_size = float(element_size)
    if not math.isfinite(element_size) or element_size <= 0:
        raise ValueError("element_size must be a finite positive value")

    divisions = tuple(
        max(1, math.ceil(dimensions[axis] / element_size - 1e-12))
        for axis in range(3)
    )
    watertight = surface_is_watertight(surface)
    if not watertight and mode == "solid":
        raise ValueError("solid mode requires a watertight surface; use shell mode")

    if mode == "shell" and not watertight:
        requested_thickness = (
            element_size if shell_thickness is None else float(shell_thickness)
        )
        if not math.isfinite(requested_thickness) or requested_thickness <= 0:
            raise ValueError("shell_thickness must be a finite positive value")
        shell_layers = max(
            1, math.ceil(requested_thickness / element_size - 1e-12)
        )
        padding = math.ceil(0.5 * shell_layers) + 1
        padded_divisions = tuple(
            divisions[axis] + 2 * padding for axis in range(3)
        )
        if math.prod(padded_divisions) > max_voxels:
            raise ValueError(
                f"voxel grid {padded_divisions} has "
                f"{math.prod(padded_divisions)} cells, "
                f"exceeding max_voxels={max_voxels}"
            )
        padded_origin = tuple(
            lower[axis] - padding * element_size for axis in range(3)
        )
        shell = _surface_shell_cells(
            surface,
            padded_origin,
            padded_divisions,
            element_size,
            shell_layers,
        )
        interior = _flood_fill_enclosed_cells(padded_divisions, shell)
        if not interior:
            raise ValueError(
                "the thickened surface does not enclose any voxel volume; "
                "use a smaller element_size or a thinner shell"
            )
        return VoxelGeometry(
            regions=[
                VoxelRegion(pid=2, title="shell", cells=shell),
                VoxelRegion(pid=3, title="interior", cells=interior),
            ],
            origin=padded_origin,
            divisions=padded_divisions,
            element_size=element_size,
            bounds=(lower, upper),
            mode=mode,
            shell_layers=shell_layers,
            open_surface=True,
        )

    if math.prod(divisions) > max_voxels:
        raise ValueError(
            f"voxel grid {divisions} has {math.prod(divisions)} cells, "
            f"exceeding max_voxels={max_voxels}"
        )
    occupied = _inside_cells(surface, lower, divisions, element_size)
    if not occupied:
        raise ValueError("voxelization produced no occupied cells")

    if mode == "solid":
        return VoxelGeometry(
            regions=[VoxelRegion(pid=1, title="voxel-solid", cells=occupied)],
            origin=lower,
            divisions=divisions,
            element_size=element_size,
            bounds=(lower, upper),
            mode=mode,
            open_surface=False,
        )

    requested_thickness = (
        element_size if shell_thickness is None else float(shell_thickness)
    )
    if not math.isfinite(requested_thickness) or requested_thickness <= 0:
        raise ValueError("shell_thickness must be a finite positive value")
    shell_layers = max(1, math.ceil(requested_thickness / element_size - 1e-12))
    interior = _erode(occupied, shell_layers)
    if not interior:
        raise ValueError(
            "shell thickness eroded the entire geometry; use a thinner shell "
            "or a smaller element_size"
        )
    shell = occupied - interior
    return VoxelGeometry(
        regions=[
            VoxelRegion(pid=2, title="shell", cells=shell),
            VoxelRegion(pid=3, title="interior", cells=interior),
        ],
        origin=lower,
        divisions=divisions,
        element_size=element_size,
        bounds=(lower, upper),
        mode=mode,
        shell_layers=shell_layers,
        open_surface=False,
    )


def _header(*names: str, widths: Iterable[int] = ()) -> KCard:
    width_list = list(widths) or [10] * len(names)
    return KCard.header(names, width_list)


def _card(*values: object, kinds: Sequence[str] | None = None,
          widths: Sequence[int] | None = None) -> KCard:
    value_list = list(values)
    return KCard(
        values=value_list,
        widths=list(widths) if widths else [10] * len(value_list),
        kinds=list(kinds) if kinds else None,
    )


def _word(name: str, *cards: KCard) -> KWord:
    return KWord(name, list(cards))


def _title_card(title: str) -> KCard:
    return KCard(values=[title], widths=[80], kinds=["s"])


def _region_node_maps(
    geometry: VoxelGeometry,
) -> dict[int, dict[tuple[int, int, int], int]]:
    """Reproduce the deterministic node numbering used by the geometry file."""
    next_node_id = 1
    result: dict[int, dict[tuple[int, int, int], int]] = {}
    for region in geometry.regions:
        node_ids: dict[tuple[int, int, int], int] = {}
        for cell in sorted(region.cells):
            for dx, dy, dz in _HEX_OFFSETS:
                grid_node = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                if grid_node not in node_ids:
                    node_ids[grid_node] = next_node_id
                    next_node_id += 1
        result[region.pid] = node_ids
    return result


def build_geometry_keyword_file(
    geometry: VoxelGeometry,
    *,
    title: str = "voxel geometry",
) -> KeywordFile:
    """Build an include-compatible keyword file with nodes, parts and solids.

    Nodes are intentionally duplicated between regions. This keeps ALE and
    Lagrangian parts topologically independent at their coincident interface.
    """
    deck = KeywordFile(header=[
        "$# LS-DYNA voxel geometry generated by ls_ex",
        f"$# {title}",
        f"$# mode={geometry.mode} element_size={geometry.element_size:g} "
        f"voxels={geometry.voxel_count}",
    ])
    deck.add(KWord("NODE", [
        _header("nid", "x", "y", "z", "tc", "rc",
                widths=(8, 16, 16, 16, 8, 8)),
    ]))
    next_node_id = 1
    solids: list[tuple[int, int, tuple[int, ...]]] = []
    next_element_id = 1
    expected_node_maps = _region_node_maps(geometry)

    for region in geometry.regions:
        deck.add(_word(
            "PART",
            _title_card(f"{title} {region.title}"),
            _header("pid", "secid", "mid", "eosid", "hgid", "grav", "adpopt", "tmid"),
            _card(
                region.pid, region.pid, region.pid,
                region.pid if region.pid == 3 else 0,
                0, 0, 0, 0,
                kinds=["i"] * 8,
            ),
        ))

        node_ids: dict[tuple[int, int, int], int] = {}
        for cell in sorted(region.cells):
            corners: list[int] = []
            for dx, dy, dz in _HEX_OFFSETS:
                grid_node = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                node_id = node_ids.get(grid_node)
                if node_id is None:
                    node_id = next_node_id
                    next_node_id += 1
                    node_ids[grid_node] = node_id
                    point = tuple(
                        geometry.origin[axis] + grid_node[axis]
                        * geometry.element_size
                        for axis in range(3)
                    )
                    deck.get("NODE").add(
                        _card(
                            node_id, *point, 0, 0,
                            kinds=["i", "f", "f", "f", "i", "i"],
                            widths=[8, 16, 16, 16, 8, 8],
                        )
                    )
                corners.append(node_id)
            solids.append((next_element_id, region.pid, tuple(corners)))
            next_element_id += 1
        assert node_ids == expected_node_maps[region.pid]

    element_block = KWord("ELEMENT_SOLID", [
        _header(
            "eid", "pid", "n1", "n2", "n3", "n4",
            "n5", "n6", "n7", "n8", widths=(8,) * 10,
        ),
    ])
    element_block.extend(
        _card(eid, pid, *corners, kinds=["i"] * 10, widths=(8,) * 10)
        for eid, pid, corners in solids
    )
    deck.add(element_block)
    return deck


def air_ale_voxel_count(geometry: VoxelGeometry, air_margin: float) -> int:
    """Return the number of air cells in the surrounding regular ALE domain."""
    margin_cells = math.ceil(float(air_margin) / geometry.element_size - 1e-12)
    if margin_cells <= 0:
        raise ValueError("air_margin must be positive")
    expanded = math.prod(
        geometry.divisions[axis] + 2 * margin_cells for axis in range(3)
    )
    return expanded - len(geometry.region(3).cells)


def _build_air_ale_mesh(
    geometry: VoxelGeometry,
    air_margin: float,
    start_node_id: int,
    start_element_id: int,
) -> tuple[KWord, KWord, list[tuple[int, int, int, int]], int]:
    """Build air cells around/through the shell and on the TNT interface.

    Interior TNT nodes from the included geometry file are reused. This makes
    the air and TNT elements one connected General ALE domain while keeping the
    Lagrangian shell topologically independent.
    """
    margin_cells = math.ceil(float(air_margin) / geometry.element_size - 1e-12)
    if margin_cells <= 0:
        raise ValueError("air_margin must be positive")
    node_maps = _region_node_maps(geometry)
    interior_nodes = node_maps[3]
    all_geometry_nodes = {
        node_id for mapping in node_maps.values() for node_id in mapping.values()
    }
    if start_node_id <= max(all_geometry_nodes):
        raise ValueError("air mesh node IDs overlap the included geometry")

    ranges = [
        range(-margin_cells, geometry.divisions[axis] + margin_cells)
        for axis in range(3)
    ]
    interior_cells = geometry.region(3).cells
    air_cells = {
        cell
        for cell in (
            (i, j, k)
            for i in ranges[0]
            for j in ranges[1]
            for k in ranges[2]
        )
        if cell not in interior_cells
    }

    node_block = KWord("NODE", [
        _header("nid", "x", "y", "z", "tc", "rc",
                widths=(8, 16, 16, 16, 8, 8)),
    ])
    node_ids = dict(interior_nodes)
    next_node_id = start_node_id

    def node_id(grid_node: tuple[int, int, int]) -> int:
        nonlocal next_node_id
        existing = node_ids.get(grid_node)
        if existing is not None:
            return existing
        existing = next_node_id
        next_node_id += 1
        node_ids[grid_node] = existing
        point = tuple(
            geometry.origin[axis] + grid_node[axis] * geometry.element_size
            for axis in range(3)
        )
        node_block.add(_card(
            existing, *point, 0, 0,
            kinds=["i", "f", "f", "f", "i", "i"],
            widths=[8, 16, 16, 16, 8, 8],
        ))
        return existing

    element_block = KWord("ELEMENT_SOLID", [
        _header(
            "eid", "pid", "n1", "n2", "n3", "n4",
            "n5", "n6", "n7", "n8", widths=(8,) * 10,
        ),
    ])
    next_element_id = start_element_id
    for cell in sorted(air_cells):
        corners = tuple(
            node_id((cell[0] + dx, cell[1] + dy, cell[2] + dz))
            for dx, dy, dz in _HEX_OFFSETS
        )
        element_block.add(_card(
            next_element_id, 4, *corners,
            kinds=["i"] * 10, widths=(8,) * 10,
        ))
        next_element_id += 1

    nx, ny, nz = geometry.divisions
    lo, hi = -margin_cells, margin_cells
    outer_faces: list[tuple[int, int, int, int]] = []
    for j in range(lo, ny + margin_cells):
        for k in range(lo, nz + margin_cells):
            outer_faces.append((
                node_id((lo, j, k)),
                node_id((lo, j + 1, k)),
                node_id((lo, j + 1, k + 1)),
                node_id((lo, j, k + 1)),
            ))
            outer_faces.append((
                node_id((nx + margin_cells, j, k)),
                node_id((nx + margin_cells, j, k + 1)),
                node_id((nx + margin_cells, j + 1, k + 1)),
                node_id((nx + margin_cells, j + 1, k)),
            ))
    for i in range(lo, nx + margin_cells):
        for k in range(lo, nz + margin_cells):
            outer_faces.append((
                node_id((i, lo, k)),
                node_id((i, lo, k + 1)),
                node_id((i + 1, lo, k + 1)),
                node_id((i + 1, lo, k)),
            ))
            outer_faces.append((
                node_id((i, ny + margin_cells, k)),
                node_id((i + 1, ny + margin_cells, k)),
                node_id((i + 1, ny + margin_cells, k + 1)),
                node_id((i, ny + margin_cells, k + 1)),
            ))
        for j in range(lo, ny + margin_cells):
            outer_faces.append((
                node_id((i, j, lo)),
                node_id((i, j + 1, lo)),
                node_id((i + 1, j + 1, lo)),
                node_id((i + 1, j, lo)),
            ))
            outer_faces.append((
                node_id((i, j, nz + margin_cells)),
                node_id((i + 1, j, nz + margin_cells)),
                node_id((i + 1, j + 1, nz + margin_cells)),
                node_id((i, j + 1, nz + margin_cells)),
            ))

    return node_block, element_block, outer_faces, len(air_cells)


def _segment_set(
    set_id: int,
    faces: Sequence[tuple[int, int, int, int]],
) -> KWord:
    cards = [
        _header("sid", "da1", "da2", "da3", "da4", "solver"),
        _card(set_id, 0.0, 0.0, 0.0, 0.0, "MECH",
              kinds=["i", "f", "f", "f", "f", "s"]),
        _header("n1", "n2", "n3", "n4", "a1", "a2", "a3", "a4"),
    ]
    cards.extend(
        _card(*face, 0.0, 0.0, 0.0, 0.0, kinds=["i"] * 4 + ["f"] * 4)
        for face in faces
    )
    return KWord("SET_SEGMENT", cards)


def build_general_ale_blast_deck(
    geometry: VoxelGeometry,
    include_name: str,
    *,
    title: str = "voxel shell TNT general ALE blast",
    detonation: Vec3 | None = None,
    end_time: float = 0.2,
    d3plot_dt: float = 0.002,
    air_margin: float = 20.0,
) -> KeywordFile:
    """Build a General ALE deck with a surrounding air domain."""
    if geometry.mode != "shell":
        raise ValueError("General ALE blast decks require mode='shell'")
    if end_time <= 0 or d3plot_dt <= 0:
        raise ValueError("end_time and d3plot_dt must be positive")
    air_voxels = air_ale_voxel_count(geometry, air_margin)
    if detonation is None:
        detonation = geometry.detonation_point(3)
    elif not geometry.contains_point(3, detonation):
        raise ValueError("detonation point must lie inside the explosive voxels")

    deck = KeywordFile(header=[
        "$# LS-DYNA General ALE / no SPH deck generated by ls_ex",
        f"$# {title}",
        f"$# element_size={geometry.element_size:g} "
        f"shell_thickness={geometry.effective_shell_thickness:g}",
        f"$# air_margin={air_margin:g} air_ALE_voxels={air_voxels}",
    ])
    deck.add(KWord("KEYWORD"))
    deck.add(_word("TITLE", _title_card(title)))

    deck.add(_word(
        "CONTROL_TERMINATION",
        _header("endtim", "endcyc", "dtmin", "endeng", "endmas", "nosol"),
        _card(end_time, 0, 0.0, 0.0, 1.0e8, 0,
              kinds=["f", "i", "f", "f", "f", "i"]),
    ))
    deck.add(_word(
        "CONTROL_TIMESTEP",
        _header("dtinit", "tssfac", "isdo", "tslimt", "dt2ms", "lctm", "erode", "ms1st"),
        _card(1.0e-4, 0.9, 0, 0.0, 0.0, 0, 0, 0,
              kinds=["f", "f", "i", "f", "f", "i", "i", "i"]),
    ))
    deck.add(_word(
        "CONTROL_ALE",
        _header("dct", "nadv", "meth", "afac", "bfac", "cfac", "dfac", "efac"),
        _card(-1, 1, 2, -1.0, 0.0, 0.0, 0.0, 0.0,
              kinds=["i", "i", "i", "f", "f", "f", "f", "f"]),
        _header("start", "end", "aafac", "vfact", "prit", "ebc", "pref", "nsidebc"),
        _card(0.0, 1.0e20, 1.0, 1.0e-6, 0, 0, 0.0, 0,
              kinds=["f", "f", "f", "f", "i", "i", "f", "i"]),
        _header("ncpl", "nbkt", "imascl", "checkr", "beamin", "mmgpref", "pdifmx", "dtmufac"),
        _card(1, 50, 0, 0.0, 0.0, 0, 0.0, 0.0,
              kinds=["i", "i", "i", "f", "f", "i", "f", "f"]),
        _header("optimpp", "ialedr", "bndflx", "minmas"),
        _card(0, 0, 0, 1.0e-5, kinds=["i", "i", "i", "f"]),
    ))
    deck.add(KWord("CONTROL_MPP_DECOMPOSITION_DISTRIBUTE_ALE_ELEMENTS"))
    deck.add(_word(
        "CONTROL_ENERGY",
        _header("hgen", "rwlen", "slnten", "rylen", "irgen", "maten", "drlen", "disen"),
        _card(2, 2, 2, 1, 1, 2, 1, 1, kinds=["i"] * 8),
    ))
    deck.add(_word(
        "CONTROL_SOLID",
        _header("esort", "fmatrx", "niptets", "swlocl", "psfail", "t10jtol", "icoh", "tet13k"),
        _card(0, 0, 4, 1, 0, 0.0, 0, 0,
              kinds=["i", "i", "i", "i", "i", "f", "i", "i"]),
    ))
    deck.add(_word(
        "CONTROL_BULK_VISCOSITY",
        _header("q1", "q2", "type", "btype", "tstype"),
        _card(1.5, 0.06, 1, 0, 0, kinds=["f", "f", "i", "i", "i"]),
    ))
    deck.add(_word(
        "HOURGLASS",
        _header("hgid", "ihq", "qm", "ibq", "q1", "q2", "qb/vdc", "qw"),
        _card(1, 4, 0.1, 0, 1.5, 0.06, 0.1, 0.1,
              kinds=["i", "i", "f", "i", "f", "f", "f", "f"]),
    ))

    deck.add(_word(
        "SECTION_SOLID_TITLE",
        _title_card("structural steel shell"),
        _header("secid", "elform", "aet"),
        _card(2, 1, 0, kinds=["i", "i", "i"]),
    ))
    deck.add(_word(
        "SECTION_SOLID_TITLE",
        _title_card("TNT ALE domain"),
        _header("secid", "elform", "aet"),
        _card(3, 11, 0, kinds=["i", "i", "i"]),
    ))
    deck.add(_word(
        "SECTION_SOLID_TITLE",
        _title_card("air ALE domain"),
        _header("secid", "elform", "aet"),
        _card(4, 11, 0, kinds=["i", "i", "i"]),
    ))
    deck.add(_word(
        "MAT_PLASTIC_KINEMATIC_TITLE",
        _title_card("structural steel shell"),
        _header("mid", "ro", "e", "pr", "sigy", "etan", "beta"),
        _card(2, 0.00783, 210000.0, 0.33, 1650.0, 300.0, 0.0,
              kinds=["i"] + ["f"] * 6),
        _header("src", "srp", "fs", "vp"),
        _card(0.0, 0.0, 0.2, 0.0, kinds=["f"] * 4),
    ))
    deck.add(_word(
        "MAT_HIGH_EXPLOSIVE_BURN_TITLE",
        _title_card("TNT"),
        _header("mid", "ro", "d", "pcj", "beta", "k", "g", "sigy"),
        _card(3, 0.00163, 6930.0, 21000.0, 0.0, 0.0, 0.0, 0.0,
              kinds=["i"] + ["f"] * 7),
    ))
    deck.add(_word(
        "EOS_JWL_TITLE",
        _title_card("TNT JWL"),
        _header("eosid", "a", "b", "r1", "r2", "omeg", "e0", "vo"),
        _card(3, 371200.0, 3231.0, 4.15, 0.95, 0.3, 7000.0, 1.0,
              kinds=["i"] + ["f"] * 7),
    ))
    deck.add(_word(
        "MAT_NULL_TITLE",
        _title_card("air"),
        _header("mid", "ro", "pc", "mu", "terod", "cerod", "ym", "pr"),
        _card(4, 1.29e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              kinds=["i"] + ["f"] * 7),
    ))
    deck.add(_word(
        "EOS_LINEAR_POLYNOMIAL_TITLE",
        _title_card("air ideal gas"),
        _header("eosid", "c0", "c1", "c2", "c3", "c4", "c5", "c6"),
        _card(4, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.0,
              kinds=["i"] + ["f"] * 7),
        _header("e0", "v0"),
        _card(0.0, 0.0, kinds=["f", "f"]),
    ))
    deck.add(_word(
        "PART",
        _title_card("air ALE domain"),
        _header("pid", "secid", "mid", "eosid", "hgid", "grav", "adpopt", "tmid"),
        _card(4, 4, 4, 4, 0, 0, 0, 0, kinds=["i"] * 8),
    ))

    deck.add(_word("INCLUDE", _title_card(include_name)))
    deck.add(_word(
        "ALE_MULTI-MATERIAL_GROUP",
        _header("sid", "idtype"),
        _card(3, 1, kinds=["i", "i"]),
        _card(4, 1, kinds=["i", "i"]),
    ))
    deck.add(_word(
        "SET_PART_LIST_TITLE",
        _title_card("Lagrangian shell"),
        _header("sid"),
        _card(100, kinds=["i"]),
        _header("pid1", "pid2", "pid3", "pid4", "pid5", "pid6", "pid7", "pid8"),
        _card(2, 0, 0, 0, 0, 0, 0, 0, kinds=["i"] * 8),
    ))
    deck.add(_word(
        "SET_PART_LIST_TITLE",
        _title_card("ALE TNT"),
        _header("sid"),
        _card(101, kinds=["i"]),
        _header("pid1", "pid2", "pid3", "pid4", "pid5", "pid6", "pid7", "pid8"),
        _card(3, 4, 0, 0, 0, 0, 0, 0, kinds=["i"] * 8),
    ))
    deck.add(_word(
        "INITIAL_DETONATION",
        _header("ipid", "x", "y", "z", "tl", "-", "mmgset"),
        _card(3, detonation[0], detonation[1], detonation[2], 0.0, "", 0,
              kinds=["i", "f", "f", "f", "f", "s", "i"]),
    ))
    deck.add(_word(
        "CONSTRAINED_LAGRANGE_IN_SOLID",
        _header("lstrsid", "alesid", "lstrstyp", "alestyp", "nquad", "ctype", "direc", "mcoup"),
        _card(2, 101, 1, 0, 2, 5, 2, 0, kinds=["i"] * 8),
        _header("start", "end", "pfac", "fric", "frcmin", "norm", "normtyp", "damp"),
        _card(0.0, 1.0e10, 0.1, 0.0, 0.5, 0, 0, 0.0,
              kinds=["f", "f", "f", "f", "f", "i", "i", "f"]),
        _header("k", "hmin", "hmax", "ileak", "pleak", "lcidpor", "nvent", "iblock"),
        _card(0.0, 0.0, 0.0, 2, 0.1, 0, 0, 0,
              kinds=["f", "f", "f", "i", "f", "i", "i", "i"]),
    ))
    deck.add(_word(
        "CONTACT_ERODING_SINGLE_SURFACE",
        _header("ssid", "msid", "sstyp", "mstyp", "sboxid", "mboxid", "spr", "mpr"),
        _card(100, 0, 2, 0, 0, 0, 0, 0, kinds=["i"] * 8),
        _header("fs", "fd", "dc", "vc", "vdc", "penchk", "bt", "dt"),
        _card(0.3, 0.3, 0.0, 0.0, 0.0, 0, 0.0, 0.0,
              kinds=["f", "f", "f", "f", "f", "i", "f", "f"]),
        _header("sfs", "sfm", "sst", "mst", "sfst", "sfmt", "fsf", "vsf"),
        _card(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, kinds=["f"] * 8),
        _header("isym", "erosop", "iadj", "", "", "", "", "unused1"),
        _card(0, 1, 1, "", "", "", "", "",
              kinds=["i", "i", "i"] + ["s"] * 5),
        _header("soft", "softscl", "lcidab", "maxpar", "sbopt", "depth", "bsort", "frcfrq"),
        _card(1, 0.1, 0, 0, 0, 5, 100, 0,
              kinds=["i", "f", "i", "i", "i", "i", "i", "i"]),
    ))
    max_geometry_node = max(
        node_id
        for mapping in _region_node_maps(geometry).values()
        for node_id in mapping.values()
    )
    air_node_block, air_element_block, outer_faces, _ = _build_air_ale_mesh(
        geometry,
        air_margin,
        start_node_id=max_geometry_node + 1,
        start_element_id=geometry.voxel_count + 1,
    )
    deck.add(air_node_block)
    deck.add(air_element_block)
    deck.add(_segment_set(1001, outer_faces))
    deck.add(_word(
        "BOUNDARY_NON_REFLECTING",
        _header("ssid", "ad", "as"),
        _card(1001, 0.0, 0.0, kinds=["i", "f", "f"]),
    ))
    deck.add(_word(
        "DATABASE_BINARY_D3PLOT",
        _header("dt", "lcdt", "beam", "npltc", "psetid"),
        _card(d3plot_dt, 0, 0, 0, 0, kinds=["f", "i", "i", "i", "i"]),
    ))
    deck.add(_word(
        "DATABASE_GLSTAT",
        _header("dt", "binary", "lcur", "ioopt"),
        _card(end_time / 50.0, 0, 0, 1, kinds=["f", "i", "i", "i"]),
    ))
    deck.add(_word(
        "DATABASE_MATSUM",
        _header("dt", "binary", "lcur", "ioopt"),
        _card(end_time / 50.0, 0, 0, 1, kinds=["f", "i", "i", "i"]),
    ))
    deck.add(KWord("END"))
    return deck
