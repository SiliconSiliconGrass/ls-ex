"""Geometric quality checks for conforming eight-node hexahedral meshes.

The checks in this module are independent of a particular LS-DYNA deck.  They
operate on :class:`ls_ex.KeywordGenerator.Mesh`, so generated and loaded meshes
can be assessed through the same API.

Two complementary distortion measures are reported:

* ``scaled_jacobian`` is evaluated at all eight corners.  Its ideal absolute
  value is 1 and 0 indicates a locally collapsed mapping.  Signed element
  volume is checked separately for inversion.
* ``skewness`` is the largest normalized deviation from a 90 degree angle
  between two incident element edges at a corner.  It is 0 for an orthogonal
  hexahedron and approaches 1 for a degenerate corner.

The skewness value is an explicit geometric proxy, not a claim that every
preprocessor reports exactly the same nonlinear skewness metric.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .KeywordGenerator import Mesh


_EDGE_PAIRS = (
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7),
)

# At each corner, the three incident edges used for Jacobian and angle checks.
_CORNER_EDGES = (
    (1, 3, 4), (2, 0, 5), (3, 1, 6), (0, 2, 7),
    (5, 7, 0), (6, 4, 1), (7, 5, 2), (4, 6, 3),
)

_HEX_FACES = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)

_TET_SPLIT = (
    (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
    (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
)


@dataclass(frozen=True)
class DistributionSummary:
    """Selected percentiles and extrema of a scalar element distribution."""

    minimum: float
    p1: float
    p5: float
    median: float
    p95: float
    p99: float
    maximum: float

    @classmethod
    def from_values(cls, values: Sequence[float] | np.ndarray) -> "DistributionSummary":
        data = np.asarray(values, dtype=float)
        if data.size == 0:
            raise ValueError("cannot summarize an empty distribution")
        if not np.all(np.isfinite(data)):
            raise ValueError("quality distribution contains non-finite values")
        minimum, p1, p5, median, p95, p99, maximum = np.percentile(
            data, (0, 1, 5, 50, 95, 99, 100)
        )
        return cls(
            minimum=float(minimum),
            p1=float(p1),
            p5=float(p5),
            median=float(median),
            p95=float(p95),
            p99=float(p99),
            maximum=float(maximum),
        )

    @property
    def max_min_ratio(self) -> float:
        return _safe_ratio(self.maximum, self.minimum)

    @property
    def p95_p5_ratio(self) -> float:
        return _safe_ratio(self.p95, self.p5)


@dataclass(frozen=True)
class MeshQualityThresholds:
    """Configurable acceptance limits for hexahedral mesh quality.

    The defaults are useful screening values for explicit-dynamics models.  A
    project with stricter requirements can provide its own thresholds.
    """

    # Skewness is normalized by 90 degrees, so 0.5 means 45 degrees.
    min_scaled_jacobian: float = 0.3
    max_skewness: float = 0.5
    max_characteristic_size_ratio: float = 2.0
    max_characteristic_size_p95_p5: float = 2.0
    max_adjacent_size_ratio: float = 2.0


@dataclass(frozen=True)
class HexMeshQualityReport:
    """Quality statistics for one PID (or for all selected solid elements)."""

    pid: int | None
    element_count: int
    target_size: float | None
    signed_volume: DistributionSummary
    scaled_jacobian: DistributionSummary
    skewness: DistributionSummary
    edge_length: DistributionSummary
    characteristic_size: DistributionSummary
    aspect_ratio: DistributionSummary
    size_to_target: DistributionSummary | None
    adjacent_size_ratio: DistributionSummary | None
    negative_volume_count: int
    degenerate_volume_count: int
    scaled_jacobian_below_limit: int
    skewness_above_limit: int
    adjacent_size_ratio_above_limit: int
    worst_scaled_jacobian_eid: int
    worst_skewness_eid: int
    worst_aspect_ratio_eid: int
    worst_adjacent_size_ratio_eid: int | None
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    @property
    def max_angle_deviation_deg(self) -> float:
        """Largest pairwise corner-edge deviation from 90 degrees."""

        return 90.0 * self.skewness.maximum

    def summary(self) -> str:
        """Return a compact human-readable report."""

        label = "all" if self.pid is None else str(self.pid)
        lines = [
            f"pid={label} elements={self.element_count} passed={self.passed}",
            (
                f"scaled_jacobian min={self.scaled_jacobian.minimum:.6g} "
                f"p1={self.scaled_jacobian.p1:.6g} "
                f"median={self.scaled_jacobian.median:.6g}"
            ),
            (
                f"skewness max={self.skewness.maximum:.6g} "
                f"(angle deviation={self.max_angle_deviation_deg:.6g} deg) "
                f"p95={self.skewness.p95:.6g} "
                f"p99={self.skewness.p99:.6g}"
            ),
            (
                f"size min={self.characteristic_size.minimum:.6g} "
                f"p5={self.characteristic_size.p5:.6g} "
                f"p95={self.characteristic_size.p95:.6g} "
                f"max={self.characteristic_size.maximum:.6g} "
                f"p95/p5={self.characteristic_size.p95_p5_ratio:.6g}"
            ),
            (
                f"aspect p95={self.aspect_ratio.p95:.6g} "
                f"max={self.aspect_ratio.maximum:.6g}"
            ),
        ]
        if self.adjacent_size_ratio is not None:
            lines.append(
                f"adjacent_size_ratio p95={self.adjacent_size_ratio.p95:.6g} "
                f"max={self.adjacent_size_ratio.maximum:.6g}"
            )
        if self.violations:
            lines.append("violations: " + "; ".join(self.violations))
        return "\n".join(lines)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return numerator / denominator
    return np.finfo(float).max if numerator > 0 else 1.0


def _selected_elements(
    mesh: Mesh, pid: int | None
) -> tuple[np.ndarray, np.ndarray]:
    eids: list[int] = []
    corners: list[tuple[int, ...]] = []
    for eid, element_pid, nodes in mesh.solids:
        if pid is not None and element_pid != pid:
            continue
        if len(nodes) != 8:
            raise ValueError(f"element {eid} is not an eight-node hexahedron")
        eids.append(eid)
        corners.append(tuple(nodes))
    if not eids:
        label = "all solid elements" if pid is None else f"PID {pid}"
        raise ValueError(f"mesh contains no hexahedral elements for {label}")
    return np.asarray(eids, dtype=np.int64), np.asarray(corners, dtype=np.int64)


def _element_coordinates(
    mesh: Mesh, element_nodes: np.ndarray
) -> np.ndarray:
    used_ids = np.unique(element_nodes)
    try:
        coordinates = np.asarray([mesh.nodes[int(nid)] for nid in used_ids], dtype=float)
    except KeyError as error:
        raise ValueError(f"element references missing node {error.args[0]}") from error
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("mesh contains non-finite coordinates")
    indices = np.searchsorted(used_ids, element_nodes)
    return coordinates[indices]


def _signed_hex_volumes(points: np.ndarray) -> np.ndarray:
    volumes = np.zeros(len(points), dtype=float)
    for ia, ib, ic, id_ in _TET_SPLIT:
        a = points[:, ib] - points[:, ia]
        b = points[:, ic] - points[:, ia]
        c = points[:, id_] - points[:, ia]
        volumes += np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    return volumes


def _distortion_metrics(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edge_vectors = np.stack(
        [points[:, b] - points[:, a] for a, b in _EDGE_PAIRS], axis=1
    )
    edge_lengths = np.linalg.norm(edge_vectors, axis=2)
    local_minimum = edge_lengths.min(axis=1)
    local_maximum = edge_lengths.max(axis=1)
    aspect_ratio = np.divide(
        local_maximum,
        local_minimum,
        out=np.full_like(local_maximum, np.finfo(float).max),
        where=local_minimum > 0,
    )

    scaled_jacobian = np.ones(len(points), dtype=float)
    skewness = np.zeros(len(points), dtype=float)
    degenerate_corner = np.zeros(len(points), dtype=bool)
    for corner, neighbours in enumerate(_CORNER_EDGES):
        vectors = [points[:, nid] - points[:, corner] for nid in neighbours]
        norms = [np.linalg.norm(vector, axis=1) for vector in vectors]
        denominator = norms[0] * norms[1] * norms[2]
        nonzero = denominator > 0
        determinant = np.einsum(
            "ij,ij->i", vectors[0], np.cross(vectors[1], vectors[2])
        )
        corner_jacobian = np.divide(
            determinant,
            denominator,
            out=np.zeros_like(determinant),
            where=nonzero,
        )
        corner_jacobian = np.clip(np.abs(corner_jacobian), 0.0, 1.0)
        scaled_jacobian = np.minimum(scaled_jacobian, corner_jacobian)
        degenerate_corner |= ~nonzero

        for first, second in ((0, 1), (0, 2), (1, 2)):
            pair_denominator = norms[first] * norms[second]
            pair_nonzero = pair_denominator > 0
            cosine = np.divide(
                np.einsum("ij,ij->i", vectors[first], vectors[second]),
                pair_denominator,
                out=np.ones_like(pair_denominator),
                where=pair_nonzero,
            )
            angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            corner_skewness = np.abs(angle - 90.0) / 90.0
            skewness = np.maximum(skewness, corner_skewness)
            degenerate_corner |= ~pair_nonzero

    skewness[degenerate_corner] = 1.0
    return edge_lengths.ravel(), scaled_jacobian, skewness, aspect_ratio


def _adjacent_size_ratios(
    element_nodes: np.ndarray, characteristic_size: np.ndarray
) -> np.ndarray:
    """Return each element's maximum face-neighbour characteristic-size ratio."""

    ratios = np.full(len(element_nodes), -1.0, dtype=float)
    faces: dict[tuple[int, int, int, int], int] = {}
    for index, corners in enumerate(element_nodes):
        for face_nodes in _HEX_FACES:
            key = tuple(sorted(int(corners[nid]) for nid in face_nodes))
            other = faces.pop(key, None)
            if other is None:
                faces[key] = index
                continue
            ratio = _safe_ratio(
                max(characteristic_size[index], characteristic_size[other]),
                min(characteristic_size[index], characteristic_size[other]),
            )
            if ratio > ratios[index]:
                ratios[index] = ratio
            if ratio > ratios[other]:
                ratios[other] = ratio
    return ratios


def _threshold_violations(
    report: "HexMeshQualityReport",
    thresholds: MeshQualityThresholds,
) -> tuple[str, ...]:
    violations: list[str] = []
    if report.negative_volume_count:
        violations.append(f"{report.negative_volume_count} negative-volume elements")
    if report.degenerate_volume_count:
        violations.append(f"{report.degenerate_volume_count} zero-volume elements")
    if report.scaled_jacobian_below_limit:
        violations.append(
            f"{report.scaled_jacobian_below_limit} elements below "
            f"scaled Jacobian {thresholds.min_scaled_jacobian:g}"
        )
    if report.skewness_above_limit:
        violations.append(
            f"{report.skewness_above_limit} elements above "
            f"skewness {thresholds.max_skewness:g}"
        )
    if report.characteristic_size.max_min_ratio > thresholds.max_characteristic_size_ratio:
        violations.append(
            f"characteristic size ratio {report.characteristic_size.max_min_ratio:.6g} "
            f"> {thresholds.max_characteristic_size_ratio:g}"
        )
    if (
        report.characteristic_size.p95_p5_ratio
        > thresholds.max_characteristic_size_p95_p5
    ):
        violations.append(
            f"characteristic size p95/p5 {report.characteristic_size.p95_p5_ratio:.6g} "
            f"> {thresholds.max_characteristic_size_p95_p5:g}"
        )
    if report.adjacent_size_ratio_above_limit:
        violations.append(
            f"{report.adjacent_size_ratio_above_limit} elements with an "
            f"adjacent-size ratio above "
            f"size ratio {thresholds.max_adjacent_size_ratio:g}"
        )
    return tuple(violations)


def assess_hex_mesh_quality(
    mesh: Mesh,
    *,
    pid: int | None = None,
    target_size: float | None = None,
    thresholds: MeshQualityThresholds | None = None,
    include_adjacent_size: bool = True,
) -> HexMeshQualityReport:
    """Measure geometric quality of eight-node hexahedral elements.

    Args:
        mesh: Nodes and solid elements to assess.
        pid: Assess only this part ID.  All solid elements are used by default.
        target_size: Optional requested element size.  When supplied, the ratio
            of each equal-volume characteristic size to this value is reported.
        thresholds: Optional configurable acceptance limits.
        include_adjacent_size: Calculate face-neighbour size jumps in addition
            to global size distributions.  Disable this for very large meshes
            when only global statistics are needed.

    Returns:
        A report containing distributions, worst element IDs and violations.

    Raises:
        ValueError: If no matching hexahedra exist or required nodes are missing.
    """

    if target_size is not None:
        target_size = float(target_size)
        if not np.isfinite(target_size) or target_size <= 0:
            raise ValueError("target_size must be a finite positive value")
    limits = thresholds or MeshQualityThresholds()
    eids, element_nodes = _selected_elements(mesh, pid)
    points = _element_coordinates(mesh, element_nodes)

    signed_volumes = _signed_hex_volumes(points)
    abs_volumes = np.abs(signed_volumes)
    characteristic_size = np.cbrt(abs_volumes)
    edge_lengths, scaled_jacobian, skewness, aspect_ratio = _distortion_metrics(points)

    degenerate_volume = abs_volumes <= 1e-14
    negative_volume = signed_volumes < 0
    jacobian_below_limit = scaled_jacobian < limits.min_scaled_jacobian
    skewness_above_limit = skewness > limits.max_skewness

    adjacent = None
    adjacent_above_limit = 0
    worst_adjacent_eid = None
    if include_adjacent_size:
        adjacent_ratios = _adjacent_size_ratios(element_nodes, characteristic_size)
        adjacent_values = adjacent_ratios[adjacent_ratios >= 0]
        if adjacent_values.size:
            adjacent = DistributionSummary.from_values(adjacent_values)
            adjacent_above_limit = int(np.count_nonzero(
                adjacent_values > limits.max_adjacent_size_ratio
            ))
            worst_adjacent_eid = int(eids[int(np.argmax(adjacent_ratios))])

    report = HexMeshQualityReport(
        pid=pid,
        element_count=len(eids),
        target_size=target_size,
        signed_volume=DistributionSummary.from_values(signed_volumes),
        scaled_jacobian=DistributionSummary.from_values(scaled_jacobian),
        skewness=DistributionSummary.from_values(skewness),
        edge_length=DistributionSummary.from_values(edge_lengths),
        characteristic_size=DistributionSummary.from_values(characteristic_size),
        aspect_ratio=DistributionSummary.from_values(aspect_ratio),
        size_to_target=(
            DistributionSummary.from_values(characteristic_size / target_size)
            if target_size is not None else None
        ),
        adjacent_size_ratio=adjacent,
        negative_volume_count=int(np.count_nonzero(negative_volume)),
        degenerate_volume_count=int(np.count_nonzero(degenerate_volume)),
        scaled_jacobian_below_limit=int(np.count_nonzero(jacobian_below_limit)),
        skewness_above_limit=int(np.count_nonzero(skewness_above_limit)),
        adjacent_size_ratio_above_limit=adjacent_above_limit,
        worst_scaled_jacobian_eid=int(eids[int(np.argmin(scaled_jacobian))]),
        worst_skewness_eid=int(eids[int(np.argmax(skewness))]),
        worst_aspect_ratio_eid=int(eids[int(np.argmax(aspect_ratio))]),
        worst_adjacent_size_ratio_eid=worst_adjacent_eid,
        violations=(),
    )
    violations = _threshold_violations(report, limits)
    return replace(report, violations=violations)
