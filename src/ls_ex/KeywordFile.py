"""整个 .k 文件的读写，以及常用数据（节点、单元）的提取。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .KCard import KCard
from .KWord import KWord

#: 从定长字段里抠数字用的正则（兼容 ``1.0-2.0`` 这种没有空格的写法）
NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")

Vec3 = tuple[float, float, float]
Solid = tuple[int, int, tuple[int, ...]]
Shell = tuple[int, int, tuple[int, ...]]


def parse_numbers(line: str) -> list[float]:
    """把一行 card 里的数值都读出来。

    先按空白切分；只要有字段不是一个合法数字（定长格式下两个数可能连着写），
    就退回用正则扫描整行。``1.5d-3`` 这种 LS-DYNA 的 D 指数也会被接受。
    """
    text = line.split("$")[0]
    tokens = text.split()
    try:
        return [float(token.replace("d", "e").replace("D", "E")) for token in tokens]
    except ValueError:
        return [float(token.replace("d", "e").replace("D", "E"))
                for token in NUMBER_RE.findall(text)]


@dataclass
class KeywordFile:
    """一个 .k 文件：开头的注释行 + 若干关键字块。"""

    keywords: list[KWord] = field(default_factory=list)
    header: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ 基本操作
    def get(self, name: str) -> KWord | None:
        """按名字取第一个关键字块（名字大小写不敏感）。"""
        wanted = name.upper()
        for keyword in self.keywords:
            if keyword.name.upper() == wanted:
                return keyword
        return None

    def get_all(self, name: str) -> list[KWord]:
        wanted = name.upper()
        return [keyword for keyword in self.keywords if keyword.name.upper() == wanted]

    def add(self, keyword: KWord) -> KWord:
        self.keywords.append(keyword)
        return keyword

    # --------------------------------------------------------------- 渲染
    def dumps(self) -> str:
        blocks = [line.rstrip() for line in self.header]
        blocks.extend(keyword.render() for keyword in self.keywords)
        return "\n".join(blocks) + "\n"

    def write(self, path: str | Path, newline: str = "\n") -> Path:
        """写文件，默认用 LF 换行（和 LS-PrePost 导出的 test1.k 一致）。"""
        path = Path(path)
        text = self.dumps()
        if newline != "\n":
            text = text.replace("\n", newline)
        path.write_text(text, encoding="utf-8", newline="")
        return path

    # --------------------------------------------------------------- 解析
    @classmethod
    def loads(cls, text: str) -> "KeywordFile":
        header: list[str] = []
        keywords: list[KWord] = []
        current: KWord | None = None
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if line.startswith("*"):
                current = KWord(name=line[1:].strip())
                keywords.append(current)
            elif current is None:
                if line.strip():
                    header.append(line)
            else:
                current.cards.append(KCard.parse(line))
        return cls(keywords=keywords, header=header)

    @classmethod
    def load(cls, path: str | Path) -> "KeywordFile":
        return cls.loads(Path(path).read_text(encoding="utf-8", errors="replace"))

    # ------------------------------------------------------ 常用数据提取
    def nodes(self) -> dict[int, Vec3]:
        """``*NODE`` 里的节点坐标：``{nid: (x, y, z)}``。"""
        keyword = self.get("NODE")
        result: dict[int, Vec3] = {}
        if keyword is None:
            return result
        for card in keyword.data_cards:
            values = parse_numbers(card.render())
            if len(values) >= 4:
                result[int(values[0])] = (values[1], values[2], values[3])
        return result

    def _elements(self, name: str, node_count: int) -> dict[int, tuple[int, tuple[int, ...]]]:
        keyword = self.get(name)
        result: dict[int, tuple[int, tuple[int, ...]]] = {}
        if keyword is None:
            return result
        for card in keyword.data_cards:
            values = parse_numbers(card.render())
            if len(values) >= 2 + node_count:
                eid = int(values[0])
                pid = int(values[1])
                nodes = tuple(int(value) for value in values[2:2 + node_count])
                result[eid] = (pid, nodes)
        return result

    def solid_elements(self) -> dict[int, tuple[int, tuple[int, ...]]]:
        """``*ELEMENT_SOLID``：``{eid: (pid, (n1..n8))}``。"""
        return self._elements("ELEMENT_SOLID", 8)

    def shell_elements(self) -> dict[int, tuple[int, tuple[int, ...]]]:
        """``*ELEMENT_SHELL``：``{eid: (pid, (n1..n4))}``。"""
        return self._elements("ELEMENT_SHELL", 4)

    def parts(self) -> dict[int, dict[str, int]]:
        """``*PART`` 定义：``{pid: {"secid": .., "mid": .., "title": ..}}``。"""
        result: dict[int, dict[str, int]] = {}
        for keyword in self.get_all("PART"):
            cards = keyword.data_cards
            if len(cards) < 2:
                continue
            title = cards[0].render()
            values = parse_numbers(cards[1].render())
            if not values:
                continue
            pid = int(values[0])
            result[pid] = {
                "title": title.strip(),
                "secid": int(values[1]) if len(values) > 1 else 0,
                "mid": int(values[2]) if len(values) > 2 else 0,
            }
        return result

    def iter_keyword_names(self) -> Iterator[str]:
        for keyword in self.keywords:
            yield keyword.name

    def summary(self) -> str:
        """给命令行用的一行摘要。"""
        return (f"keywords={len(self.keywords)} nodes={len(self.nodes())} "
                f"solids={len(self.solid_elements())} "
                f"shells={len(self.shell_elements())}")
