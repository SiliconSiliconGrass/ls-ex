"""LS-DYNA 关键字块中的一行（一张 "card"）。

LS-DYNA 的 .k 文件是按列宽排列的定长格式：每一张 card 占一行，字段宽度
固定（大多数关键字是 10 字符一个字段，*NODE / *ELEMENT_* 是 8 字符）。
:class:`KCard` 同时保存字段值和列宽，因此可以重新渲染成定长格式；从文件里
解析出来的行则原样保存在 ``line`` 里，保证读写往返不丢失信息。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

#: 大多数关键字的字段宽度
DEFAULT_FIELD_WIDTH = 10
#: 浮点数最多保留的有效位数（超宽时会自动降低精度）
FLOAT_PRECISION = 10

_NUMBER_CHARS = set("0123456789.+-eEdD")


def format_int(value: object, width: int = DEFAULT_FIELD_WIDTH) -> str:
    """右对齐的整数字段。"""
    return f"{int(value):>{width}d}"


def format_float(value: object, width: int = DEFAULT_FIELD_WIDTH,
                 precision: int = FLOAT_PRECISION) -> str:
    """右对齐的浮点数字段。

    优先用 ``%g`` 风格的有效数字；字段超宽时逐步降低精度，必要时退化成科学
    计数法，保证结果一定塞得进 ``width`` 个字符。整数值会补上 ``.0``
    （列宽不够时保持原样，LS-DYNA 同样能识别）。
    """
    number = float(value)
    text = f"{number:.{precision}g}"
    if len(text) > width:
        for digits in range(precision, 0, -1):
            text = f"{number:.{digits}g}"
            if len(text) <= width:
                break
        else:
            text = f"{number:.1e}"
    if "." not in text and "e" not in text.lower() and len(text) + 2 <= width:
        text += ".0"
    return f"{text:>{width}}"


def format_str(value: object, width: int = DEFAULT_FIELD_WIDTH) -> str:
    """左对齐的字符串字段（用于标题一类的文本字段）。"""
    return f"{str(value):<{width}}"


def _guess_kind(value: object) -> str:
    if isinstance(value, bool):
        return "i"
    if isinstance(value, int):
        return "i"
    if isinstance(value, float):
        return "f"
    text = str(value)
    if text and all(char in _NUMBER_CHARS for char in text):
        return "f"
    return "s"


@dataclass
class KCard:
    """关键字块里的一行。

    参数
    ----
    values:
        按列顺序排列的字段值。
    widths:
        每个字段的列宽，缺省用 :data:`DEFAULT_FIELD_WIDTH`。
    kinds:
        每个字段的类型，``"i"`` 整数 / ``"f"`` 浮点 / ``"s"`` 文本；缺省按值推断。
    line:
        原样保存的整行文本（解析得到的 card 用它），此时渲染直接返回该行。
    """

    values: Sequence[object] = field(default_factory=list)
    widths: Sequence[int] = field(default_factory=list)
    kinds: Sequence[str] = field(default_factory=list)
    line: str | None = None

    # ------------------------------------------------------------------ 渲染
    def render(self) -> str:
        if self.line is not None:
            return self.line.rstrip()
        chunks = []
        for index, value in enumerate(self.values):
            width = self.widths[index] if index < len(self.widths) else DEFAULT_FIELD_WIDTH
            kind = self.kinds[index] if index < len(self.kinds) else _guess_kind(value)
            if kind == "i":
                chunks.append(format_int(value, width))
            elif kind == "f":
                chunks.append(format_float(value, width))
            else:
                chunks.append(format_str(value, width))
        return "".join(chunks).rstrip()

    @property
    def is_comment(self) -> bool:
        """``$`` 开头的行对 LS-DYNA 来说只是注释。"""
        return self.render().lstrip().startswith("$")

    # ------------------------------------------------------------ 构造快捷方式
    @classmethod
    def parse(cls, line: str) -> "KCard":
        """把文件里的一行原样收下来。"""
        return cls(line=line)

    @classmethod
    def comment(cls, text: str) -> "KCard":
        """注释行；文本已以 ``$`` 开头时保持原样。"""
        text = text if text.lstrip().startswith("$") else f"$# {text}"
        return cls(line=text)

    @classmethod
    def header(cls, names: Sequence[str], widths: Sequence[int]) -> "KCard":
        """生成 ``$#`` 开头的字段名提示行（LS-PrePost 的写法）。

        字段名右对齐在自己的列宽里；第一列前面有 ``$#``，所以要少占 2 个字符。
        """
        chunks = ["$#"]
        for index, name in enumerate(names):
            width = widths[index] if index < len(widths) else DEFAULT_FIELD_WIDTH
            effective = width - 2 if index == 0 else width
            chunks.append(f"{name:>{effective}}")
        return cls(line="".join(chunks))


def typed_card(values: Sequence[object], kinds: Sequence[str],
               widths: Sequence[int] | None = None) -> KCard:
    """显式指定字段类型的 card。"""
    widths = list(widths) if widths else [DEFAULT_FIELD_WIDTH] * len(values)
    return KCard(values=list(values), widths=widths, kinds=list(kinds))
