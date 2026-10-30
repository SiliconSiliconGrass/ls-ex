from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ls_ex.sph_fragments import (
    SPHParticleData,
    SPHParticleSequence,
    angular_speed_histogram,
    angular_statistics,
    filter_sph_outliers,
    make_angle_bins,
    particle_angles_deg,
    plot_angular_distribution,
    plot_sph_speed_animation,
    plot_sph_speed_3d,
)


def _particles(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    alive: np.ndarray | None = None,
) -> SPHParticleData:
    count = len(velocities)
    return SPHParticleData(
        positions=np.asarray(positions, dtype=float),
        velocities=np.asarray(velocities, dtype=float),
        speeds=np.linalg.norm(velocities, axis=1),
        node_ids=np.arange(1, count + 1),
        particle_indices=np.arange(count),
        material_indices=np.ones(count, dtype=np.int64),
        alive=np.ones(count, dtype=bool) if alive is None else alive,
        time=1.25,
    )


class SPHFragmentTests(unittest.TestCase):
    def test_velocity_angles_cover_cardinal_directions(self) -> None:
        particles = _particles(
            np.zeros((4, 3)),
            np.array(
                [
                    [0.0, 0.0, 2.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 0.0, -4.0],
                    [0.0, 5.0, 0.0],
                ]
            ),
        )
        angles = particle_angles_deg(particles)
        np.testing.assert_allclose(angles, [0.0, 90.0, 180.0, 90.0])

    def test_position_angles_use_origin(self) -> None:
        particles = _particles(
            np.array([[0.0, 0.0, 5.0], [5.0, 0.0, 0.0]]),
            np.ones((2, 3)),
        )
        angles = particle_angles_deg(
            particles,
            mode="position",
            origin=(0.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(angles, [0.0, 90.0])

    def test_tukey_filter_removes_only_upper_speed_tail(self) -> None:
        particles = _particles(
            np.zeros((5, 3)),
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [1000.0, 0.0, 0.0],
                ]
            ),
        )
        filtered, report = filter_sph_outliers(particles, iqr_factor=3.0)
        self.assertEqual(report.excluded_count, 1)
        self.assertEqual(report.output_count, 4)
        np.testing.assert_allclose(filtered.speeds, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(report.excluded_speed_min, 1000.0)
        self.assertEqual(report.excluded_speed_max, 1000.0)

    def test_angular_statistics_and_histogram_are_consistent(self) -> None:
        angles = np.array([0.0, 20.0, 90.0, 100.0, 180.0])
        speeds = np.array([100.0, 200.0, 300.0, 500.0, 700.0])
        stats = angular_statistics(
            angles,
            speeds,
            bin_edges_deg=make_angle_bins(90.0),
        )
        np.testing.assert_array_equal(stats.counts, [2, 3])
        np.testing.assert_allclose(stats.fractions, [0.4, 0.6])
        np.testing.assert_allclose(stats.speed_mean, [150.0, 500.0])

        histogram = angular_speed_histogram(
            angles,
            speeds,
            angle_edges_deg=make_angle_bins(90.0),
            speed_bin_count=4,
        )
        self.assertEqual(int(histogram.counts.sum()), len(angles))
        np.testing.assert_array_equal(histogram.counts.sum(axis=1), stats.counts)

    def test_invalid_axis_and_angle_bins_are_rejected(self) -> None:
        particles = _particles(np.zeros((1, 3)), np.ones((1, 3)))
        with self.assertRaises(ValueError):
            particle_angles_deg(particles, axis=(0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            make_angle_bins(0.0)

    def test_plot_functions_write_standalone_html(self) -> None:
        particles = _particles(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                ]
            ),
            np.array(
                [
                    [0.0, 0.0, 10.0],
                    [10.0, 0.0, 0.0],
                    [0.0, 0.0, -20.0],
                ]
            ),
        )
        angles = particle_angles_deg(particles)
        stats = angular_statistics(angles, particles.speeds)
        histogram = angular_speed_histogram(angles, particles.speeds)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plot_3d = plot_sph_speed_3d(particles, root / "3d.html", angles_deg=angles)
            angular = plot_angular_distribution(
                stats,
                histogram,
                root / "angles.html",
            )
            self.assertIn("plotly", plot_3d.read_text(encoding="utf-8").lower())
            self.assertIn("plotly", angular.read_text(encoding="utf-8").lower())

    def test_animation_writes_frames_and_play_controls(self) -> None:
        times = np.array([0.0, 0.1, 0.2])
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]],
            ]
        )
        velocities = np.zeros_like(positions)
        velocities[:, :, 2] = 10.0
        speeds = np.linalg.norm(velocities, axis=2)
        sequence = SPHParticleSequence(
            positions=positions,
            velocities=velocities,
            speeds=speeds,
            alive=np.ones((3, 2), dtype=bool),
            times=times,
            node_ids=np.array([10, 11]),
            particle_indices=np.array([100, 101]),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = plot_sph_speed_animation(
                sequence,
                Path(directory) / "animation.html",
                frame_duration_ms=25,
            )
            html = output.read_text(encoding="utf-8").lower()
            self.assertIn("play", html)
            self.assertIn("pause", html)
            self.assertIn("frame", html)
            self.assertIn("scatter3d", html)


if __name__ == "__main__":
    unittest.main()
