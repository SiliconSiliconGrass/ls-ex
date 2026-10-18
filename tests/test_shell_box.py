"""壳单元"长方体壳"网格的检查：封闭、法向朝外、尺寸正确。"""

from __future__ import annotations

import unittest
from collections import Counter

from ls_ex import KeywordFile, build_keyword_file, hollow_box_shell_mesh

INNER = (20.0, 10.0, 10.0)
THICKNESS = 2.0
ELEMENT_SIZE = 2.0
OUTER = tuple(side + 2 * THICKNESS for side in INNER)          # (24, 14, 14)
MID = tuple(side + THICKNESS for side in INNER)                # (22, 12, 12)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def subtract(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


class HollowBoxShellTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = hollow_box_shell_mesh(INNER, THICKNESS, element_size=ELEMENT_SIZE)

    def test_counts(self) -> None:
        nx, ny, nz = (round(side / ELEMENT_SIZE) for side in MID)  # 11, 6, 6
        self.assertEqual(len(self.mesh.shells),
                         2 * (nx * ny + nx * nz + ny * nz))
        # 曲面上的节点：网格点减去内部的点
        self.assertEqual(len(self.mesh.nodes),
                         (nx + 1) * (ny + 1) * (nz + 1) - (nx - 1) * (ny - 1) * (nz - 1))

    def test_nodes_lie_on_the_mid_surface(self) -> None:
        planes = {0: (0.0, MID[0]), 1: (0.0, MID[1]), 2: (0.0, MID[2])}
        for nid, point in self.mesh.nodes.items():
            on_face = any(point[axis] in planes[axis] for axis in range(3))
            self.assertTrue(on_face, f"节点 {nid} 不在中面上: {point}")

    def test_surface_is_watertight(self) -> None:
        counter: Counter = Counter()
        for _, _, corners in self.mesh.shells:
            for index in range(4):
                edge = tuple(sorted((corners[index], corners[(index + 1) % 4])))
                counter[edge] += 1
        self.assertTrue(all(count == 2 for count in counter.values()),
                        "存在只被一个壳单元使用的边，曲面没有封闭")

    def test_normals_point_outwards(self) -> None:
        center = tuple(side / 2 for side in MID)
        for eid, _, corners in self.mesh.shells:
            p0, p1, p2 = (self.mesh.nodes[nid] for nid in corners[:3])
            normal = cross(subtract(p1, p0), subtract(p2, p0))
            centroid = [sum(self.mesh.nodes[nid][axis] for nid in corners) / 4
                        for axis in range(3)]
            outward = sum(normal[axis] * (centroid[axis] - center[axis])
                          for axis in range(3))
            self.assertGreater(outward, 0.0, f"单元 {eid} 的法向朝内")

    def test_total_area(self) -> None:
        mx, my, mz = MID
        self.assertAlmostEqual(self.mesh.shell_area,
                               2 * (mx * my + mx * mz + my * mz), places=9)

    def test_section_shell_thickness(self) -> None:
        deck = build_keyword_file(self.mesh, title="shell box")
        reloaded = KeywordFile.loads(deck.dumps())
        section = reloaded.get("SECTION_SHELL")
        self.assertIsNotNone(section)
        # 第二张 card 的前 4 个字段是 T1..T4
        thickness = [float(token) for token in section.data_cards[1].render().split()]
        self.assertEqual(thickness[:4], [THICKNESS] * 4)
        self.assertEqual(len(reloaded.shell_elements()), len(self.mesh.shells))
        self.assertEqual(reloaded.parts()[1]["secid"], 1)

    def test_element_size_defaults_to_thickness(self) -> None:
        mesh = hollow_box_shell_mesh((4.0, 4.0, 4.0), 1.0)
        nx = ny = nz = 5  # 中面 5×5×5，单元尺寸 = 壁厚
        self.assertEqual(len(mesh.shells), 2 * (nx * ny + nx * nz + ny * nz))


if __name__ == "__main__":
    unittest.main()
