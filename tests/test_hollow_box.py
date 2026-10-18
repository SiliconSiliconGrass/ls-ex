"""空心长方体（实体单元）网格的正确性检查。"""

from __future__ import annotations

import unittest
from collections import Counter

from ls_ex import KeywordFile, build_keyword_file, hex_volume, hollow_box_mesh

INNER = (20.0, 10.0, 10.0)
THICKNESS = 2.0
OUTER = tuple(side + 2 * THICKNESS for side in INNER)  # (24, 14, 14)

# 单元面的 6 个组合，用来判断网格是否封闭
FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
         (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def face_counts(mesh) -> Counter:
    counter: Counter = Counter()
    for _, _, corners in mesh.solids:
        for face in FACES:
            counter[face_key([corners[i] for i in face])] += 1
    return counter


def face_key(quad) -> tuple:
    """面的规范化 key：保留节点环状顺序，顺时针/逆时针和起点都不影响结果。"""
    rotations = [tuple(quad[index:] + quad[:index]) for index in range(4)]
    return min(rotations + [rotation[::-1] for rotation in rotations])


def free_faces(mesh) -> Counter:
    """只被一个单元使用的面，也就是网格的表面。"""
    return Counter({face: count for face, count in face_counts(mesh).items() if count == 1})


class HollowBoxTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = hollow_box_mesh(INNER, THICKNESS, element_size=2.0)

    def test_counts(self) -> None:
        # 每个方向：1 层壁 + 内腔单元 + 1 层壁，即 (12, 7, 7) 个单元
        self.assertEqual(len(self.mesh.solids), 12 * 7 * 7 - 10 * 5 * 5)
        # 网格点 13×8×8，空腔内部 9×4×4 个点没有被任何单元引用
        self.assertEqual(len(self.mesh.nodes), 13 * 8 * 8 - 9 * 4 * 4)

    def test_volume_matches_inner_and_outer_box(self) -> None:
        expected = OUTER[0] * OUTER[1] * OUTER[2] - INNER[0] * INNER[1] * INNER[2]
        self.assertAlmostEqual(self.mesh.solid_volume, expected, places=9)

    def test_every_element_has_positive_volume(self) -> None:
        for eid, _, corners in self.mesh.solids:
            volume = hex_volume([self.mesh.nodes[nid] for nid in corners])
            self.assertGreater(volume, 0.0, f"单元 {eid} 体积为负，LS-DYNA 会报错")

    def test_mesh_is_watertight(self) -> None:
        # 内部面被两个单元共用，表面（外壁 + 空腔）只被一个单元使用
        self.assertTrue(all(count <= 2 for count in face_counts(self.mesh).values()),
                        "有面被三个以上单元共用，单元连接重复了")
        surface = free_faces(self.mesh)
        # 2×2 的面单元，外表面积 2*(24*14 + 24*14 + 14*14)，空腔 2*(20*10 + 20*10 + 10*10)
        expected = (2 * (OUTER[0] * OUTER[1] + OUTER[0] * OUTER[2] + OUTER[1] * OUTER[2])
                    + 2 * (INNER[0] * INNER[1] + INNER[0] * INNER[2] + INNER[1] * INNER[2])) / 4
        self.assertEqual(len(surface), expected)

        # 表面本身必须是封闭的：每条边正好属于两个表面面片
        edges: Counter = Counter()
        for face in surface:
            for index in range(4):
                edges[tuple(sorted((face[index], face[(index + 1) % 4])))] += 1
        self.assertTrue(all(count == 2 for count in edges.values()),
                        "表面有裂缝（边只属于一个面片）")

    def test_no_element_inside_the_cavity(self) -> None:
        for eid, _, corners in self.mesh.solids:
            coordinates = [self.mesh.nodes[nid] for nid in corners]
            center = [sum(point[axis] for point in coordinates) / 8 for axis in range(3)]
            inside = all(THICKNESS < center[axis] < THICKNESS + INNER[axis]
                         for axis in range(3))
            self.assertFalse(inside, f"单元 {eid} 落在空腔里")

    def test_bounding_box(self) -> None:
        xs = [point[0] for point in self.mesh.nodes.values()]
        ys = [point[1] for point in self.mesh.nodes.values()]
        zs = [point[2] for point in self.mesh.nodes.values()]
        self.assertEqual((min(xs), max(xs)), (0.0, OUTER[0]))
        self.assertEqual((min(ys), max(ys)), (0.0, OUTER[1]))
        self.assertEqual((min(zs), max(zs)), (0.0, OUTER[2]))

    def test_wall_layer_count_can_be_forced(self) -> None:
        mesh = hollow_box_mesh((12.0, 12.0, 12.0), 3.0, element_size=3.0, n_wall=3)
        # 每个方向 3+4+3 = 10 个单元，中心 4×4×4 是空腔
        self.assertEqual(len(mesh.solids), 10 ** 3 - 4 ** 3)
        self.assertAlmostEqual(mesh.solid_volume, 18 ** 3 - 12 ** 3, places=9)

    def test_n_wall_alone_sets_the_element_size(self) -> None:
        # 没给 element_size 时取 thickness / n_wall = 1，于是 18×18×18 的网格
        mesh = hollow_box_mesh((12.0, 12.0, 12.0), 3.0, n_wall=3)
        self.assertEqual(len(mesh.solids), 18 ** 3 - 12 ** 3)
        self.assertAlmostEqual(mesh.solid_volume, 18 ** 3 - 12 ** 3, places=9)

    def test_uneven_element_size_keeps_the_wall_thickness(self) -> None:
        # 单元尺寸 2 不整除壁厚 1.5：壁面单元是 1.5，空腔单元是 2.0
        mesh = hollow_box_mesh((10.0, 10.0, 10.0), 1.5, element_size=2.0)
        self.assertAlmostEqual(mesh.solid_volume, 13 ** 3 - 10 ** 3, places=9)
        self.assertEqual(len(mesh.solids), 7 ** 3 - 5 ** 3)
        # 壁厚方向只有一层单元（round(1.5 / 2) = 1）
        xs = sorted({point[0] for point in mesh.nodes.values()})
        self.assertAlmostEqual(xs[1] - xs[0], 1.5, places=9)

    def test_default_element_size_is_thickness(self) -> None:
        mesh = hollow_box_mesh((4.0, 4.0, 4.0), 1.0)
        self.assertEqual(len(mesh.solids), 6 * 6 * 6 - 4 * 4 * 4)

    def test_origin_offsets_the_mesh(self) -> None:
        mesh = hollow_box_mesh(INNER, THICKNESS, element_size=2.0, origin=(10.0, -5.0, 0.0))
        xs = [point[0] for point in mesh.nodes.values()]
        self.assertEqual(min(xs), 10.0)

    def test_keyword_file_round_trip(self) -> None:
        deck = build_keyword_file(self.mesh, title="hollow box")
        text = deck.dumps()
        reloaded = KeywordFile.loads(text)
        self.assertEqual(reloaded.nodes(), self.mesh.nodes)
        self.assertEqual(len(reloaded.solid_elements()), len(self.mesh.solids))
        self.assertEqual(reloaded.parts()[1]["secid"], 1)
        self.assertEqual(reloaded.parts()[1]["mid"], 1)
        self.assertIn("SECTION_SOLID", list(reloaded.iter_keyword_names()))
        self.assertIn("MAT_ELASTIC", list(reloaded.iter_keyword_names()))
        self.assertIn("CONTROL_TERMINATION", list(reloaded.iter_keyword_names()))

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            hollow_box_mesh((10.0, 10.0, 10.0), 0.0)
        with self.assertRaises(ValueError):
            hollow_box_mesh((0.0, 10.0, 10.0), 1.0)


if __name__ == "__main__":
    unittest.main()
