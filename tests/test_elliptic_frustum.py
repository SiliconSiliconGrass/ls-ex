"""椭圆台体壳网格的几何与网格质量检查。"""

from __future__ import annotations

import unittest
from collections import Counter, defaultdict
from math import sqrt

from ls_ex import KeywordFile, build_keyword_file, hex_volume
from ls_ex.EllipticFrustum import (analytic_shell_volume, ellipse_perimeter,
                                   elliptical_frustum_shell_mesh)

MAJOR, MINOR, HEIGHT, RATIO, THICK = 40.0, 30.0, 50.0, 0.6, 3.0
ELEMENT = 2.0

FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
         (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def face_key(quad):
    rotations = [tuple(quad[i:] + quad[:i]) for i in range(4)]
    return min(rotations + [rotation[::-1] for rotation in rotations])


def face_counts(mesh) -> Counter:
    counts: Counter = Counter()
    for _, _, corners in mesh.solids:
        for face in FACES:
            counts[face_key([corners[i] for i in face])] += 1
    return counts


def boundary_components(mesh) -> tuple[int, Counter]:
    """返回 (封闭边界分量数, 边界边出现次数)。"""
    free = [k for k, v in face_counts(mesh).items() if v == 1]
    edges: dict = defaultdict(list)
    for index, face in enumerate(free):
        for i in range(4):
            edges[tuple(sorted((face[i], face[(i + 1) % 4])))].append(index)
    parent = list(range(len(free)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ids in edges.values():
        root = find(ids[0])
        for other in ids[1:]:
            other_root = find(other)
            if other_root != root:
                parent[other_root] = root
    return len({find(i) for i in range(len(free))}), Counter(len(v) for v in edges.values())


class EllipticFrustumTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = elliptical_frustum_shell_mesh(MAJOR, MINOR, HEIGHT, RATIO, THICK,
                                                  element_size=ELEMENT)

    def test_counts_and_positive_volume(self) -> None:
        self.assertGreater(len(self.mesh.solids), 1000)
        for eid, _, corners in self.mesh.solids:
            self.assertGreater(hex_volume([self.mesh.nodes[n] for n in corners]), 0,
                               f"单元 {eid} 体积为负")

    def test_volume_matches_analytic(self) -> None:
        exact = analytic_shell_volume(MAJOR, MINOR, HEIGHT, RATIO, THICK)
        self.assertLess(abs(self.mesh.solid_volume - exact) / exact, 0.01)

    def test_surface_is_closed(self) -> None:
        counts = face_counts(self.mesh)
        self.assertLessEqual(max(counts.values()), 2, "有面被三个以上单元共用")
        components, boundary_edges = boundary_components(self.mesh)
        self.assertEqual(components, 2, "边界应为两个封闭曲面：外表面 + 内腔面")
        self.assertEqual(set(boundary_edges), {2},
                         "边界曲面有裂缝（边不属于两个面片）")

    def test_cavity_is_empty(self) -> None:
        a0, b0 = MAJOR / 2, MINOR / 2
        for eid, _, corners in self.mesh.solids:
            points = [self.mesh.nodes[n] for n in corners]
            center = [sum(p[i] for p in points) / 8 for i in range(3)]
            z = center[2]
            if 0 < z < HEIGHT:
                scale = 1 + (RATIO - 1) * z / HEIGHT
                if (center[0] / (a0 * scale)) ** 2 + (center[1] / (b0 * scale)) ** 2 < 0.999:
                    self.fail(f"单元 {eid} 落在内腔里")

    def test_caps_are_solid(self) -> None:
        """端盖中心必须有材料：端盖是实心椭圆等距区域。"""
        for z_low, z_high in ((-THICK, 0.0), (HEIGHT, HEIGHT + THICK)):
            near_center = []
            for _, _, corners in self.mesh.solids:
                points = [self.mesh.nodes[n] for n in corners]
                center = [sum(p[i] for p in points) / 8 for i in range(3)]
                if z_low < center[2] < z_high and abs(center[0]) < 2 and abs(center[1]) < 2:
                    near_center.append(center)
            self.assertTrue(near_center, f"z∈[{z_low},{z_high}] 端盖中心没有单元")

    def test_overall_extents(self) -> None:
        xs = [p[0] for p in self.mesh.nodes.values()]
        ys = [p[1] for p in self.mesh.nodes.values()]
        zs = [p[2] for p in self.mesh.nodes.values()]
        a0, b0 = MAJOR / 2, MINOR / 2
        # 多边形逼近等距曲线，外扩量略小于 h
        self.assertAlmostEqual(max(xs), a0 + THICK, delta=0.15)
        self.assertAlmostEqual(max(ys), b0 + THICK, delta=0.15)
        self.assertAlmostEqual(min(zs), -THICK, places=9)
        self.assertAlmostEqual(max(zs), HEIGHT + THICK, places=9)

    def test_element_size_is_reasonable(self) -> None:
        edges = set()
        for _, _, corners in self.mesh.solids:
            for a, b in zip(corners, corners[1:] + corners[:1]):
                edges.add(tuple(sorted((a, b))))
        lengths = sorted(sqrt(sum((self.mesh.nodes[a][i] - self.mesh.nodes[b][i]) ** 2
                                  for i in range(3))) for a, b in edges)
        median = lengths[len(lengths) // 2]
        # 台体整体缩放到 0.6，环向尺寸本身就会在小端变细；中位数应接近目标值
        self.assertAlmostEqual(median, ELEMENT, delta=0.35)
        self.assertGreater(lengths[0], 0.3 * ELEMENT)

    def test_origin_is_applied(self) -> None:
        mesh = elliptical_frustum_shell_mesh(MAJOR, MINOR, HEIGHT, RATIO, THICK,
                                             element_size=ELEMENT, origin=(100, -50, 7))
        xs = [p[0] for p in mesh.nodes.values()]
        zs = [p[2] for p in mesh.nodes.values()]
        self.assertGreater(min(xs), 75)
        self.assertAlmostEqual(min(zs), 7 - THICK, places=9)

    def test_symmetry_sectors_have_exact_volume_fractions(self) -> None:
        exact = analytic_shell_volume(MAJOR, MINOR, HEIGHT, RATIO, THICK)
        cases = {
            "half-long": 0.5,
            "half-short": 0.5,
            "quarter": 0.25,
        }
        for symmetry, fraction in cases.items():
            with self.subTest(symmetry=symmetry):
                mesh = elliptical_frustum_shell_mesh(
                    MAJOR, MINOR, HEIGHT, RATIO, THICK,
                    element_size=ELEMENT, symmetry=symmetry,
                )
                self.assertAlmostEqual(mesh.solid_volume, exact * fraction,
                                       delta=exact * 0.002)
                xs = [p[0] for p in mesh.nodes.values()]
                ys = [p[1] for p in mesh.nodes.values()]
                if symmetry in {"half-short", "quarter"}:
                    self.assertGreaterEqual(min(xs), -1e-9)
                    self.assertTrue(any(abs(x) < 1e-9 for x in xs))
                if symmetry in {"half-long", "quarter"}:
                    self.assertGreaterEqual(min(ys), -1e-9)
                    self.assertTrue(any(abs(y) < 1e-9 for y in ys))
                for _, _, corners in mesh.solids:
                    center_x = sum(mesh.nodes[nid][0] for nid in corners) / 8
                    center_y = sum(mesh.nodes[nid][1] for nid in corners) / 8
                    if symmetry in {"half-short", "quarter"}:
                        self.assertGreater(center_x, 0.0)
                    if symmetry in {"half-long", "quarter"}:
                        self.assertGreater(center_y, 0.0)

    def test_unknown_symmetry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            elliptical_frustum_shell_mesh(
                MAJOR, MINOR, HEIGHT, RATIO, THICK,
                element_size=ELEMENT, symmetry="eighth",
            )

    def test_offset_curve_limit_is_checked(self) -> None:
        # 非常扁的椭圆 + 厚壁：等距曲线会自交
        with self.assertRaises(ValueError):
            elliptical_frustum_shell_mesh(100, 10, 20, 0.5, 5, element_size=2)

    def test_keyword_file_roundtrip(self) -> None:
        deck = build_keyword_file(self.mesh, title="elliptic frustum shell")
        reloaded = KeywordFile.loads(deck.dumps())
        self.assertEqual(len(reloaded.nodes()), len(self.mesh.nodes))
        self.assertEqual(len(reloaded.solid_elements()), len(self.mesh.solids))
        self.assertIn("ELEMENT_SOLID", list(reloaded.iter_keyword_names()))

    def test_analytic_helpers(self) -> None:
        self.assertAlmostEqual(ellipse_perimeter(10, 10), 20 * 3.141592653589793, places=2)


if __name__ == "__main__":
    unittest.main()
