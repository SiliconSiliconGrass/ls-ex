"""Include and streaming geometry-split tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ls_ex import KCard, KWord, KeywordFile, split_geometry_include


def deck() -> KeywordFile:
    result = KeywordFile(header=["$# test"])
    result.add(KWord("KEYWORD"))
    nodes = KWord("NODE")
    for node_id, point in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        1,
    ):
        nodes.add(KCard(values=[node_id, *point, 0, 0],
                        kinds=["i", "f", "f", "f", "i", "i"]))
    result.add(nodes)
    result.add(KWord("ELEMENT_SOLID", [
        KCard(values=[1, 1, 1, 2, 3, 4, 1, 2, 3, 4],
              kinds=["i"] * 10),
    ]))
    result.add(KWord("END"))
    return result


class KeywordIncludeTest(unittest.TestCase):
    def test_split_and_resolve_include(self) -> None:
        original = deck()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry, geometry = original.split_geometry("geometry.k")
            entry.write(root / "entry.k")
            geometry.write(root / "geometry.k")
            resolved = KeywordFile.load_resolved(root / "entry.k")
            self.assertEqual(len(resolved.nodes()), 4)
            self.assertEqual(len(resolved.solid_elements()), 1)

    def test_streaming_split(self) -> None:
        original = deck()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.k"
            original.write(source)
            entry_path, geometry_path = split_geometry_include(
                source, root / "geometry.k", root / "entry.k"
            )
            entry = KeywordFile.load(entry_path)
            geometry = KeywordFile.load(geometry_path)
            self.assertIn("INCLUDE", entry.iter_keyword_names())
            self.assertNotIn("NODE", entry.iter_keyword_names())
            self.assertIn("NODE", geometry.iter_keyword_names())
            resolved = KeywordFile.load_resolved(entry_path)
            self.assertEqual(len(resolved.nodes()), 4)

    def test_cycle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = KeywordFile()
            first.add_include("second.k")
            first.add(KWord("END"))
            first.write(root / "first.k")
            second = KeywordFile()
            second.add_include("first.k")
            second.add(KWord("END"))
            second.write(root / "second.k")
            with self.assertRaises(ValueError):
                KeywordFile.load_resolved(root / "first.k")


if __name__ == "__main__":
    unittest.main()
