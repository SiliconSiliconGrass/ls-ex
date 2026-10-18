"""一个 LS-DYNA 关键字块，例如 ``*NODE`` 整块。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .KCard import KCard


@dataclass
class KWord:
    """由 ``*NAME`` 行和若干 card 组成的关键字块。

    ``name`` 不含开头的 ``*``，例如 ``"ELEMENT_SOLID"``。``$`` 开头的注释行也
    作为 card 存在列表里，读写时可以原样带回去。
    """

    name: str
    cards: list[KCard] = field(default_factory=list)

    # ------------------------------------------------------------------ 渲染
    def render(self) -> str:
        lines = [f"*{self.name}"]
        lines.extend(card.render() for card in self.cards)
        return "\n".join(lines)

    def add(self, *cards: KCard) -> "KWord":
        self.cards.extend(cards)
        return self

    def extend(self, cards: Iterable[KCard]) -> "KWord":
        self.cards.extend(cards)
        return self

    # -------------------------------------------------------------- 数据读取
    @property
    def data_cards(self) -> list[KCard]:
        """去掉注释行之后的数据 card。"""
        return [card for card in self.cards if not card.is_comment]

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试
        return f"KWord({self.name!r}, cards={len(self.cards)})"


def comment_block(lines: Sequence[str]) -> list[KCard]:
    """把若干行文本变成注释 card。"""
    return [KCard.comment(line) for line in lines]
