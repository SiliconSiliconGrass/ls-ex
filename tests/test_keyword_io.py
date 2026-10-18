"""解析 test1.k，并检查读写往返。"""

from __future__ import annotations

import unittest
from pathlib import Path

from ls_ex import KeywordFile, hex_volume, parse_numbers

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


if __name__ == "__main__":
    unittest.main()
