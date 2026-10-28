"""Tests for STL/GLB loading, voxelization and the General ALE deck."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from ls_ex import (
    KeywordFile,
    SurfaceMesh,
    air_ale_voxel_count,
    build_general_ale_blast_deck,
    build_geometry_keyword_file,
    load_glb,
    parse_numbers,
    surface_is_watertight,
    voxelize_surface,
)


def box_triangles(size: float = 4.0) -> list[tuple]:
    ids = {
        (0, 0, 0): 0, (0, 0, 1): 1, (0, 1, 0): 2, (0, 1, 1): 3,
        (1, 0, 0): 4, (1, 0, 1): 5, (1, 1, 0): 6, (1, 1, 1): 7,
    }
    points = {
        key: (key[0] * size, key[1] * size, key[2] * size)
        for key in ids
    }
    faces = (
        ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)),
        ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)),
        ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)),
        ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)),
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
        ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)),
    )
    triangles = []
    for a, b, c, d in faces:
        triangles.append((points[a], points[b], points[c]))
        triangles.append((points[a], points[c], points[d]))
    return triangles


def glb_cube(size: float = 4.0) -> bytes:
    positions = [
        (x, y, z)
        for x in (0.0, size)
        for y in (0.0, size)
        for z in (0.0, size)
    ]
    triangles = box_triangles(size)
    lookup = {point: index for index, point in enumerate(positions)}
    indices = [lookup[point] for triangle in triangles for point in triangle]
    position_data = b"".join(
        struct.pack("<3f", *point) for point in positions
    )
    index_data = struct.pack(f"<{len(indices)}I", *indices)
    binary = position_data + index_data
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_data)},
            {
                "buffer": 0,
                "byteOffset": len(position_data),
                "byteLength": len(index_data),
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [size, size, size],
            },
            {
                "bufferView": 1,
                "componentType": 5125,
                "count": len(indices),
                "type": "SCALAR",
            },
        ],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1,
                "mode": 4,
            }],
        }],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    json_bytes = json.dumps(document, separators=(",", ":")).encode()
    json_bytes += b" " * (-len(json_bytes) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A)
        + json_bytes
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


class VoxelizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = SurfaceMesh(box_triangles())

    def test_solid_voxelization(self) -> None:
        geometry = voxelize_surface(self.surface, 1.0, mode="solid")
        self.assertEqual(geometry.divisions, (4, 4, 4))
        self.assertEqual(geometry.voxel_count, 64)
        self.assertEqual([region.pid for region in geometry.regions], [1])

    def test_watertight_detection_and_open_surface_shell(self) -> None:
        self.assertTrue(surface_is_watertight(self.surface))
        open_surface = SurfaceMesh(
            box_triangles(12.0)
            + [
                ((20.0, 0.0, 0.0), (20.0, 4.0, 0.0), (20.0, 0.0, 4.0)),
            ]
        )
        self.assertFalse(surface_is_watertight(open_surface))
        geometry = voxelize_surface(
            open_surface, 1.0, mode="shell", shell_thickness=4.0
        )
        self.assertTrue(geometry.open_surface)
        self.assertEqual(geometry.shell_layers, 4)
        self.assertTrue(geometry.region(2).cells)
        self.assertTrue(geometry.region(3).cells)

    def test_shell_and_interior_are_disjoint_and_complete(self) -> None:
        geometry = voxelize_surface(
            self.surface, 1.0, mode="shell", shell_thickness=1.0
        )
        shell = geometry.region(2).cells
        interior = geometry.region(3).cells
        self.assertFalse(shell & interior)
        self.assertEqual(len(shell) + len(interior), 64)
        self.assertEqual(len(interior), 8)
        self.assertEqual(geometry.shell_layers, 1)

    def test_shell_can_be_four_voxel_layers(self) -> None:
        surface = SurfaceMesh(box_triangles(12.0))
        geometry = voxelize_surface(
            surface, 1.0, mode="shell", shell_thickness=4.0
        )
        self.assertEqual(geometry.shell_layers, 4)
        self.assertEqual(geometry.effective_shell_thickness, 4.0)
        self.assertEqual(len(geometry.region(3).cells), 4 ** 3)

    def test_geometry_has_independent_region_nodes(self) -> None:
        geometry = voxelize_surface(
            self.surface, 1.0, mode="shell", shell_thickness=1.0
        )
        deck = build_geometry_keyword_file(geometry, title="box")
        elements = deck.solid_elements()
        shell_nodes = {
            node
            for pid, nodes in elements.values()
            if pid == 2
            for node in nodes
        }
        interior_nodes = {
            node
            for pid, nodes in elements.values()
            if pid == 3
            for node in nodes
        }
        self.assertFalse(shell_nodes & interior_nodes)
        self.assertEqual(len(deck.nodes()), len(shell_nodes) + len(interior_nodes))
        self.assertEqual(set(deck.parts()), {2, 3})
        tnt_part = parse_numbers(
            deck.get_all("PART")[1].data_cards[-1].render()
        )
        self.assertEqual(tnt_part[3], 3)
        self.assertNotIn("*KEYWORD", deck.dumps())
        self.assertNotIn("*END", deck.dumps())

    def test_glb_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cube.glb"
            path.write_bytes(glb_cube())
            surface = load_glb(path)
        self.assertEqual(len(surface.triangles), 12)
        self.assertEqual(surface.bounds, ((0.0, 0.0, 0.0), (4.0, 4.0, 4.0)))

    def test_surface_scaling_keeps_element_size(self) -> None:
        scaled = SurfaceMesh(box_triangles(6.0)).scaled(2.0)
        self.assertEqual(scaled.bounds, ((0.0, 0.0, 0.0), (12.0, 12.0, 12.0)))
        geometry = voxelize_surface(
            scaled, 1.0, mode="shell", shell_thickness=4.0
        )
        self.assertEqual(geometry.divisions, (12, 12, 12))
        self.assertEqual(geometry.shell_layers, 4)

    def test_general_ale_deck(self) -> None:
        geometry = voxelize_surface(
            self.surface, 1.0, mode="shell", shell_thickness=1.0
        )
        deck = build_general_ale_blast_deck(
            geometry, "box_geometry.k", air_margin=2.0
        )
        names = set(deck.iter_keyword_names())
        self.assertIn("INCLUDE", names)
        self.assertIn("CONTROL_ALE", names)
        self.assertIn("ALE_MULTI-MATERIAL_GROUP", names)
        self.assertIn("INITIAL_DETONATION", names)
        self.assertFalse(any("SPH" in name for name in names))
        self.assertIn("MAT_NULL_TITLE", names)
        self.assertIn("EOS_LINEAR_POLYNOMIAL_TITLE", names)
        self.assertIn("BOUNDARY_NON_REFLECTING", names)
        include = deck.get("INCLUDE")
        self.assertIn("box_geometry.k", include.render())
        detonation = deck.get("INITIAL_DETONATION").data_cards[-1]
        coordinates = parse_numbers(detonation.render())[1:4]
        self.assertTrue(all(0.0 < value < 4.0 for value in coordinates))
        self.assertEqual(set(deck.parts()), {4})
        self.assertIn("MAT_PLASTIC_KINEMATIC_TITLE", names)
        self.assertIn("MAT_HIGH_EXPLOSIVE_BURN_TITLE", names)
        air_part = parse_numbers(deck.get("PART").data_cards[-1].render())
        self.assertEqual(air_part[3], 4)
        coupling = deck.get("CONSTRAINED_LAGRANGE_IN_SOLID").data_cards[-3]
        self.assertEqual(parse_numbers(coupling.render())[:4], [2.0, 101.0, 1.0, 0.0])
        air_elements = deck.solid_elements()
        self.assertTrue(air_elements)
        self.assertEqual({pid for pid, _ in air_elements.values()}, {4})
        self.assertEqual(len(air_elements), air_ale_voxel_count(geometry, 2.0))
        geometry_deck = build_geometry_keyword_file(geometry)
        interior_nodes = {
            node
            for pid, nodes in geometry_deck.solid_elements().values()
            if pid == 3
            for node in nodes
        }
        air_nodes = {
            node
            for _, nodes in air_elements.values()
            for node in nodes
        }
        self.assertTrue(interior_nodes & air_nodes)

    def test_detonation_must_be_inside_a_voxel(self) -> None:
        geometry = voxelize_surface(
            self.surface, 1.0, mode="shell", shell_thickness=1.0
        )
        with self.assertRaises(ValueError):
            build_general_ale_blast_deck(
                geometry, "box.k", detonation=(0.1, 0.1, 0.1)
            )
        deck = build_general_ale_blast_deck(
            geometry, "box.k", detonation=(1.1, 1.1, 1.1)
        )
        detonation = deck.get("INITIAL_DETONATION").data_cards[-1]
        coordinates = parse_numbers(detonation.render())[1:4]
        self.assertTrue(all(abs(value - 1.1) < 1e-9 for value in coordinates))

    def test_invalid_options(self) -> None:
        with self.assertRaises(ValueError):
            voxelize_surface(self.surface, 1.0, mode="other")
        with self.assertRaises(ValueError):
            voxelize_surface(self.surface, 0.0)
        with self.assertRaises(ValueError):
            voxelize_surface(
                self.surface, 1.0, mode="shell", shell_thickness=100.0
            )


if __name__ == "__main__":
    unittest.main()
