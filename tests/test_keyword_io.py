"""解析 test1.k，并检查读写往返。"""

from __future__ import annotations

import unittest
from pathlib import Path

from ls_ex import KeywordFile, hex_volume, parse_numbers, split_geometry_include

ROOT = Path(__file__).resolve().parents[1]
TEST1 = ROOT / "test1.k"


class KeywordFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = TEST1.read_text(encoding="utf-8")
        self.deck = KeywordFile.loads(self.text)

    def test_keyword_order(self) -> None:
        self.assertEqual(list(self.deck.iter_keyword_names()),
                         ["KEYWORD", "TITLE", "NODE", "PART", "ELEMENT_SOLID", "END"])

    def test_nodes(self) -> None:
        nodes = self.deck.nodes()
        self.assertEqual(len(nodes), 125)  # 5×5×5
        self.assertEqual(nodes[1], (1.0, 1.0, 1.0))
        self.assertEqual(nodes[512], (5.0, 5.0, 5.0))

    def test_solid_elements(self) -> None:
        solids = self.deck.solid_elements()
        self.assertEqual(len(solids), 64)  # 4×4×4
        pid, corners = solids[1]
        self.assertEqual(pid, 1)
        self.assertEqual(corners, (1, 5, 37, 3, 2, 13, 45, 4))

    def test_volume_is_4x4x4(self) -> None:
        nodes = self.deck.nodes()
        volume = sum(hex_volume([nodes[nid] for nid in corners])
                     for _, corners in self.deck.solid_elements().values())
        self.assertAlmostEqual(volume, 64.0, places=9)

    def test_parts(self) -> None:
        parts = self.deck.parts()
        self.assertEqual(parts[1]["title"], "BlockMesh1")
        self.assertEqual(parts[1]["secid"], 0)

    def test_round_trip_keeps_everything(self) -> None:
        normalized = "\n".join(line.rstrip() for line in self.text.splitlines()) + "\n"
        self.assertEqual(self.deck.dumps(), normalized)

    def test_parse_numbers_handles_fixed_width(self) -> None:
        # 定长格式下两个数字可能紧挨着，没有空格分隔
        self.assertEqual(parse_numbers("  1.0-2.0   3.5"), [1.0, -2.0, 3.5])
        self.assertEqual(parse_numbers("  2.0D-03 4.0"), [2e-3, 4.0])
        self.assertEqual(parse_numbers("      1       1       1"), [1.0, 1.0, 1.0])

    def test_split_geometry_and_resolve_include(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry, geometry = self.deck.split_geometry("mesh.k")
            self.assertNotIn("NODE", list(entry.iter_keyword_names()))
            self.assertNotIn("ELEMENT_SOLID", list(entry.iter_keyword_names()))
            self.assertIn("INCLUDE", list(entry.iter_keyword_names()))
            geometry.write(root / "mesh.k")
            entry.write(root / "entry.k")

            resolved = KeywordFile.load_resolved(root / "entry.k")
            self.assertEqual(len(resolved.nodes()), len(self.deck.nodes()))
            self.assertEqual(
                len(resolved.solid_elements()),
                len(self.deck.solid_elements()),
            )

    def test_nested_include_and_cycle_detection(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "leaf.k").write_text("*NODE\n       1     0.0     0.0     0.0\n*END\n")
            middle = KeywordFile()
            middle.add_include("leaf.k")
            middle.add(type(middle).loads("*END").keywords[0])
            middle.write(root / "middle.k")
            top = KeywordFile()
            top.add_include("middle.k")
            top.add(type(top).loads("*END").keywords[0])
            top.write(root / "top.k")
            self.assertEqual(KeywordFile.load_resolved(root / "top.k").nodes(),
                             {1: (0.0, 0.0, 0.0)})

            first = KeywordFile()
            first.add_include("second.k")
            first.add(type(first).loads("*END").keywords[0])
            first.write(root / "first.k")
            second = KeywordFile()
            second.add_include("first.k")
            second.add(type(second).loads("*END").keywords[0])
            second.write(root / "second.k")
            with self.assertRaises(ValueError):
                KeywordFile.load_resolved(root / "first.k")

    def test_streaming_geometry_split(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.k"
            source.write_text(self.text, encoding="utf-8")
            entry_path, geometry_path = split_geometry_include(
                source, root / "geometry.k", root / "entry.k"
            )
            entry = KeywordFile.load(entry_path)
            geometry = KeywordFile.load(geometry_path)
            self.assertIn("INCLUDE", list(entry.iter_keyword_names()))
            self.assertIn("NODE", list(geometry.iter_keyword_names()))
            self.assertNotIn("NODE", list(entry.iter_keyword_names()))
            resolved = KeywordFile.load_resolved(entry_path)
            self.assertEqual(len(resolved.nodes()), len(self.deck.nodes()))


if __name__ == "__main__":
    unittest.main()
