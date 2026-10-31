"""Tests for geometric hexahedral mesh quality checks."""

from __future__ import annotations

import unittest

from ls_ex import Mesh, MeshQualityThresholds, assess_hex_mesh_quality


def cube_mesh() -> Mesh:
    nodes = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (1.0, 1.0, 0.0),
        4: (0.0, 1.0, 0.0),
        5: (0.0, 0.0, 1.0),
        6: (1.0, 0.0, 1.0),
        7: (1.0, 1.0, 1.0),
        8: (0.0, 1.0, 1.0),
    }
    return Mesh(nodes=nodes, solids=[(1, 1, (1, 2, 3, 4, 5, 6, 7, 8))])


class MeshQualityTest(unittest.TestCase):
    def test_unit_cube_is_ideal(self) -> None:
        report = assess_hex_mesh_quality(cube_mesh(), target_size=1.0)

        self.assertTrue(report.passed)
        self.assertAlmostEqual(report.signed_volume.minimum, 1.0)
        self.assertAlmostEqual(report.scaled_jacobian.minimum, 1.0)
        self.assertAlmostEqual(report.skewness.maximum, 0.0)
        self.assertAlmostEqual(report.characteristic_size.max_min_ratio, 1.0)
        self.assertIsNone(report.adjacent_size_ratio)
        self.assertAlmostEqual(report.size_to_target.median, 1.0)

    def test_sheared_cube_reports_skewness_and_jacobian(self) -> None:
        mesh = cube_mesh()
        for nid in (5, 6, 7, 8):
            x, y, z = mesh.nodes[nid]
            mesh.nodes[nid] = (x + 1.0, y, z)

        report = assess_hex_mesh_quality(
            mesh,
            thresholds=MeshQualityThresholds(max_skewness=0.49),
        )

        self.assertAlmostEqual(report.scaled_jacobian.minimum, 2 ** -0.5)
        self.assertAlmostEqual(report.skewness.maximum, 0.5)
        self.assertEqual(report.skewness_above_limit, 1)
        self.assertFalse(report.passed)
        self.assertIn("skewness", report.violations[0])

    def test_inverted_element_is_invalid(self) -> None:
        mesh = cube_mesh()
        corners = (1, 4, 3, 2, 5, 8, 7, 6)
        mesh.solids = [(1, 1, corners)]

        report = assess_hex_mesh_quality(mesh)

        self.assertEqual(report.negative_volume_count, 1)
        self.assertFalse(report.passed)

    def test_collapsed_element_is_reported_without_breaking_summary(self) -> None:
        mesh = cube_mesh()
        for nid in (5, 6, 7, 8):
            mesh.nodes[nid] = (0.0, 0.0, 0.0)

        report = assess_hex_mesh_quality(mesh)

        self.assertEqual(report.degenerate_volume_count, 1)
        self.assertEqual(report.scaled_jacobian.maximum, 0.0)
        self.assertEqual(report.skewness.maximum, 1.0)
        self.assertTrue(report.aspect_ratio.maximum > 1e100)
        self.assertFalse(report.passed)

    def test_adjacent_and_global_size_differences_are_reported(self) -> None:
        mesh = cube_mesh()
        mesh.nodes.update({
            9: (3.0, 0.0, 0.0),
            10: (3.0, 1.0, 0.0),
            11: (3.0, 0.0, 1.0),
            12: (3.0, 1.0, 1.0),
        })
        mesh.solids.append((2, 1, (2, 9, 10, 3, 6, 11, 12, 7)))

        report = assess_hex_mesh_quality(mesh, target_size=1.0)

        expected_ratio = 2 ** (1 / 3)
        self.assertAlmostEqual(report.characteristic_size.max_min_ratio, expected_ratio)
        self.assertAlmostEqual(report.adjacent_size_ratio.maximum, expected_ratio)
        self.assertEqual(report.worst_adjacent_size_ratio_eid, 1)

    def test_pid_filter_and_missing_pid(self) -> None:
        mesh = cube_mesh()
        mesh.solids[0] = (1, 7, mesh.solids[0][2])

        report = assess_hex_mesh_quality(mesh, pid=7, include_adjacent_size=False)
        self.assertEqual(report.pid, 7)
        self.assertEqual(report.element_count, 1)

        with self.assertRaises(ValueError):
            assess_hex_mesh_quality(mesh, pid=8)


if __name__ == "__main__":
    unittest.main()
