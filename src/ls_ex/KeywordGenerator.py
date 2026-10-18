"""由几何参数生成网格，并把网格组装成 LS-DYNA 关键字文件。

目前提供两个生成器：

``box_mesh``
    实心长方体的六面体网格（``*ELEMENT_SOLID``），用来复现 test1.k 那种网格。
``hollow_box_mesh``
    空心长方体（长方体壳）的六面体网格：外箱体内挖掉一个长方体空腔，空腔尺寸
    和壁厚都可以指定。
``hollow_box_shell_mesh``
    只用壳单元（``*ELEMENT_SHELL``）表示的长方体壳：网格画在壁厚的中间面上，
    壁厚通过 ``*SECTION_SHELL`` 给出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Sequence

from .KCard import KCard
from .KWord import KWord
from .KeywordFile import KeywordFile, Vec3

#: *NODE 的列宽：nid + xyz + tc + rc
NODE_WIDTHS = [8, 16, 16, 16, 8, 8]
#: *ELEMENT_SOLID / *ELEMENT_SHELL 的列宽：eid + pid + 节点号
ELEMENT_WIDTHS = [8] * 12

#: 六面体单元 8 个角点在 (i, j, k) 网格上的偏移，顺序与 LS-DYNA 要求一致
#: （n1..n4 为一层，n5..n8 为同序的上一层，test1.k 就是这么排的）
HEX_OFFSETS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))

#: 六面体拆成 6 个四面体的方式（绕 1-7 对角线），用来算体积、判断正负体积
TET_SPLIT = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
             (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))

#: 坐标四舍五入到小数点后多少位（清掉 2.0000000000000004 这类浮点噪声）
COORD_DIGITS = 12


# --------------------------------------------------------------------- 工具
def _clean(value: float) -> float:
    return round(value, COORD_DIGITS)


def as_vec3(value: object, name: str = "value") -> Vec3:
    """把 ``3`` 或 ``(3, 4, 5)`` 或 ``"3,4,5"`` 统一成三元组。"""
    if isinstance(value, str):
        parts = [part for part in value.replace("x", ",").replace(" ", ",").split(",") if part]
        numbers = [float(part) for part in parts]
    elif isinstance(value, (int, float)):
        numbers = [float(value)] * 3
    else:
        numbers = [float(item) for item in value]  # type: ignore[arg-type]
    if len(numbers) == 1:
        numbers *= 3
    if len(numbers) != 3:
        raise ValueError(f"{name} 需要 1 个或 3 个数值，得到 {value!r}")
    return (numbers[0], numbers[1], numbers[2])


def _check_positive(value: float, name: str) -> float:
    if not value > 0:
        raise ValueError(f"{name} 必须大于 0，得到 {value!r}")
    return float(value)


def _linspace(start: float, stop: float, count: int) -> list[float]:
    """``count`` 个单元的等分线，返回 ``count + 1`` 个坐标。"""
    if count < 1:
        raise ValueError("count 至少为 1")
    return [_clean(start + (stop - start) * index / count) for index in range(count + 1)]


def _divisions(size: float, element_size: float) -> int:
    return max(1, round(size / element_size))


def _tet_volume(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> float:
    ax, ay, az = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    bx, by, bz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    cx, cy, cz = p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]
    determinant = (ax * (by * cz - bz * cy)
                   - ay * (bx * cz - bz * cx)
                   + az * (bx * cy - by * cx))
    return determinant / 6.0


def hex_volume(points: Sequence[Vec3]) -> float:
    """六面体的带符号体积；节点顺序右手系时为正（LS-DYNA 要求为正）。"""
    return sum(_tet_volume(points[a], points[b], points[c], points[d])
               for a, b, c, d in TET_SPLIT)


# --------------------------------------------------------------------- 网格
@dataclass
class Mesh:
    """一份网格：节点坐标 + 实体单元 / 壳单元。"""

    nodes: dict[int, Vec3] = field(default_factory=dict)
    solids: list[tuple[int, int, tuple[int, ...]]] = field(default_factory=list)
    shells: list[tuple[int, int, tuple[int, ...]]] = field(default_factory=list)
    pid: int = 1
    shell_thickness: float | None = None
    description: str = ""

    @property
    def solid_volume(self) -> float:
        """所有实体单元的体积之和。"""
        return sum(hex_volume([self.nodes[nid] for nid in nodes])
                   for _, _, nodes in self.solids)

    @property
    def shell_area(self) -> float:
        """所有壳单元的面积之和。"""
        total = 0.0
        for _, _, nodes in self.shells:
            coordinates = [self.nodes[nid] for nid in nodes]
            # 四边形面积 = 两个三角形面积之和
            total += _triangle_area(coordinates[0], coordinates[1], coordinates[2])
            total += _triangle_area(coordinates[0], coordinates[2], coordinates[3])
        return total

    def summary(self) -> str:
        text = f"nodes={len(self.nodes)}"
        if self.solids:
            text += f" solids={len(self.solids)} volume={self.solid_volume:.6g}"
        if self.shells:
            text += f" shells={len(self.shells)} area={self.shell_area:.6g}"
        return text


def _triangle_area(p0: Vec3, p1: Vec3, p2: Vec3) -> float:
    ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return 0.5 * (cx * cx + cy * cy + cz * cz) ** 0.5


def box_mesh(size: object = (4.0, 4.0, 4.0), element_size: float = 1.0,
             origin: object = (0.0, 0.0, 0.0), pid: int = 1) -> Mesh:
    """实心长方体（复现 test1.k 那种 4×4×4 的立方体网格）。"""
    length = as_vec3(size, "size")
    start = as_vec3(origin, "origin")
    element_size = _check_positive(float(element_size), "element_size")
    axes = [
        _linspace(start[axis], start[axis] + length[axis],
                  _divisions(length[axis], element_size))
        for axis in range(3)
    ]
    return _structured_box(axes, pid=pid,
                           description=f"solid box {length} h={element_size}")


def hollow_box_mesh(inner_size: object, thickness: float,
                    element_size: float | None = None,
                    n_wall: int | None = None,
                    origin: object = (0.0, 0.0, 0.0),
                    pid: int = 1) -> Mesh:
    """空心长方体的六面体网格。

    参数
    ----
    inner_size:
        内腔尺寸 ``(cx, cy, cz)``（也可以只给一个数表示正方体空腔）。
    thickness:
        壳（壁）厚，六个面一样厚。
    element_size:
        目标单元尺寸；缺省等于 ``thickness``。内腔和壁面都是它的整数等分，
        这样壁面正好落在网格线上，不会出现"阶梯"状的边界。
    n_wall:
        壁厚方向的单元层数，指定它会覆盖由 ``element_size`` 推出的层数。
    origin:
        外箱体最小角点的坐标，缺省在原点。

    返回的网格外轮廓尺寸是 ``inner_size + 2 * thickness``。
    """
    inner = as_vec3(inner_size, "inner_size")
    thickness = _check_positive(float(thickness), "thickness")
    for index, side in enumerate(inner):
        _check_positive(side, f"inner_size[{index}]")
    wall_layers = int(n_wall) if n_wall else None
    if element_size is None:
        # 只给了层数就按层数取单元尺寸，这样壁面和空腔的单元大小比较接近
        element_size = thickness / wall_layers if wall_layers else thickness
    element_size = _check_positive(float(element_size), "element_size")
    if wall_layers is None:
        wall_layers = _divisions(thickness, element_size)
    cavity_layers = [_divisions(side, element_size) for side in inner]
    lines = [_hollow_axis(side, thickness, wall_layers, layers)
             for side, layers in zip(inner, cavity_layers)]
    start = as_vec3(origin, "origin")
    axes = [[_clean(start[axis] + value) for value in lines[axis]] for axis in range(3)]
    return _structured_box(
        axes,
        pid=pid,
        description=(f"hollow box inner={inner} thickness={thickness} "
                     f"h={element_size} wall_layers={wall_layers}"),
        cavity=(wall_layers, cavity_layers),
    )


def _hollow_axis(inner: float, thickness: float, wall_layers: int,
                 cavity_layers: int) -> list[float]:
    """一个方向上的网格线：壁 - 空腔 - 壁，且网格线一定落在空腔边界上。"""
    lines = _linspace(0.0, thickness, wall_layers)
    lines += _linspace(thickness, thickness + inner, cavity_layers)[1:]
    lines += _linspace(thickness + inner, 2 * thickness + inner, wall_layers)[1:]
    return lines


def _structured_box(axes: Sequence[Sequence[float]], pid: int = 1, description: str = "",
                    cavity: tuple[int, Sequence[int]] | None = None) -> Mesh:
    """按规则的 (i, j, k) 网格铺满长方体，可选地挖掉中间的空腔。

    ``axes`` 是三个方向上已经排好的网格线坐标（可以是不等距的，"壁-空腔-壁"
    就是这么来的），单元就是相邻网格线围成的小方块。

    ``cavity = (wall_layers, [nx, ny, nz])`` 表示空腔在三个方向上各占几个单元；
    完全落在空腔里的单元不输出，其余单元组成"壳"。
    """
    divisions = [len(axis) - 1 for axis in axes]
    node_ids: dict[tuple[int, int, int], int] = {}
    nodes: dict[int, Vec3] = {}

    def node_id(cell: tuple[int, int, int]) -> int:
        existing = node_ids.get(cell)
        if existing is None:
            existing = len(node_ids) + 1
            node_ids[cell] = existing
            nodes[existing] = (axes[0][cell[0]], axes[1][cell[1]], axes[2][cell[2]])
        return existing

    if cavity is None:
        in_cavity = lambda cell: False  # noqa: E731 - 简单直白
    else:
        wall_layers, layers = cavity
        cavity_range = [range(wall_layers, wall_layers + layers[axis]) for axis in range(3)]

        def in_cavity(cell: tuple[int, int, int]) -> bool:
            return all(cell[axis] in cavity_range[axis] for axis in range(3))

    solids: list[tuple[int, int, tuple[int, ...]]] = []
    cells = product(range(divisions[0]), range(divisions[1]), range(divisions[2]))
    for cell in cells:
        if in_cavity(cell):
            continue
        corners = tuple(node_id((cell[0] + dx, cell[1] + dy, cell[2] + dz))
                        for dx, dy, dz in HEX_OFFSETS)
        solids.append((len(solids) + 1, pid, corners))

    return Mesh(nodes=nodes, solids=solids, pid=pid, description=description)


def hollow_box_shell_mesh(inner_size: object, thickness: float,
                          element_size: float | None = None,
                          origin: object = (0.0, 0.0, 0.0),
                          pid: int = 1) -> Mesh:
    """只画壳单元的"长方体壳"：6 个面都画在壁厚的中面上。

    中面尺寸 = ``inner_size + thickness``，因此加上 ``*SECTION_SHELL`` 的壁厚
    ``thickness`` 之后，内腔正好是 ``inner_size``、外形是 ``inner_size + 2 * thickness``。
    单元法向全部朝外。
    """
    inner = as_vec3(inner_size, "inner_size")
    thickness = _check_positive(float(thickness), "thickness")
    for index, side in enumerate(inner):
        _check_positive(side, f"inner_size[{index}]")
    element_size = thickness if element_size is None else _check_positive(float(element_size), "element_size")

    mid = tuple(side + thickness for side in inner)
    nx, ny, nz = (_divisions(side, element_size) for side in mid)
    mx, my, mz = mid

    # (起点, u 方向矢量, v 方向矢量, u 向单元数, v 向单元数)；u × v 指向面外
    faces = (
        ((0.0, 0.0, 0.0), (0.0, 0.0, mz), (0.0, my, 0.0), nz, ny),  # -x
        ((mx, 0.0, 0.0), (0.0, my, 0.0), (0.0, 0.0, mz), ny, nz),   # +x
        ((0.0, 0.0, 0.0), (mx, 0.0, 0.0), (0.0, 0.0, mz), nx, nz),  # -y
        ((0.0, my, 0.0), (0.0, 0.0, mz), (mx, 0.0, 0.0), nz, nx),   # +y
        ((0.0, 0.0, 0.0), (0.0, my, 0.0), (mx, 0.0, 0.0), ny, nx),  # -z
        ((0.0, 0.0, mz), (mx, 0.0, 0.0), (0.0, my, 0.0), nx, ny),   # +z
    )

    shift = as_vec3(origin, "origin")
    node_ids: dict[tuple[float, float, float], int] = {}
    nodes: dict[int, Vec3] = {}

    def node_id(point: Vec3) -> int:
        key = (round(point[0], 9), round(point[1], 9), round(point[2], 9))
        existing = node_ids.get(key)
        if existing is None:
            existing = len(node_ids) + 1
            node_ids[key] = existing
            nodes[existing] = (_clean(point[0] + shift[0]),
                               _clean(point[1] + shift[1]),
                               _clean(point[2] + shift[2]))
        return existing

    shells: list[tuple[int, int, tuple[int, ...]]] = []
    for start, u_axis, v_axis, n_u, n_v in faces:
        for i, j in product(range(n_u), range(n_v)):
            corners = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                u = (i + du) / n_u
                v = (j + dv) / n_v
                corners.append(node_id((
                    _clean(start[0] + u * u_axis[0] + v * v_axis[0]),
                    _clean(start[1] + u * u_axis[1] + v * v_axis[1]),
                    _clean(start[2] + u * u_axis[2] + v * v_axis[2]),
                )))
            shells.append((len(shells) + 1, pid, tuple(corners)))

    return Mesh(nodes=nodes, shells=shells, pid=pid, shell_thickness=thickness,
                description=(f"hollow box shell inner={inner} thickness={thickness} "
                             f"h={element_size}"))


# ------------------------------------------------------- 组装成关键字文件
DEFAULT_MATERIAL = {"ro": 7.85e-9, "e": 210000.0, "pr": 0.3}  # mm / ton / s 单位制


def build_keyword_file(mesh: Mesh, *, title: str | None = None,
                       pid: int | None = None, secid: int = 1, mid: int = 1,
                       material: dict | None = None,
                       endtim: float | None = 1.0) -> KeywordFile:
    """把 :class:`Mesh` 组装成可以直接提交给 LS-DYNA 的 :class:`KeywordFile`。

    生成的块依次是 ``*KEYWORD``、``*TITLE``、``*NODE``、``*SECTION_*``、
    ``*MAT_ELASTIC``、``*PART``、``*ELEMENT_*``、``*CONTROL_TERMINATION``、``*END``。
    ``material=None`` 时用 :data:`DEFAULT_MATERIAL`（mm-ton-s 单位制的钢）。
    """
    if not mesh.solids and not mesh.shells:
        raise ValueError("网格里没有任何单元")
    pid = mesh.pid if pid is None else pid
    title = title or mesh.description or "ls_ex mesh"
    material = dict(DEFAULT_MATERIAL if material is None else material)

    file = KeywordFile(header=[
        "$# LS-DYNA keyword file generated by ls_ex",
        f"$# {title}",
    ])
    file.add(KWord("KEYWORD"))
    file.add(KWord("TITLE", [
        KCard.header(["title"], [80]),
        KCard(values=[title], widths=[80], kinds=["s"]),
    ]))

    # --------------------------------------------------------------- 节点
    node_block = KWord("NODE", [KCard.header(["nid", "x", "y", "z", "tc", "rc"], NODE_WIDTHS)])
    ids = sorted(mesh.nodes)
    assert ids == list(range(1, len(ids) + 1)), "节点编号必须是 1..N 的连续整数"
    for nid in ids:
        x, y, z = mesh.nodes[nid]
        node_block.add(KCard(values=[nid, x, y, z, 0, 0], widths=NODE_WIDTHS,
                             kinds=["i", "f", "f", "f", "i", "i"]))
    file.add(node_block)

    # ----------------------------------------------------------- 截面/材料
    if mesh.solids:
        file.add(KWord("SECTION_SOLID", [
            KCard.header(["secid", "elform", "aet"], [10, 10, 10]),
            KCard(values=[secid, 1, 0], widths=[10, 10, 10], kinds=["i", "i", "i"]),
        ]))
    if mesh.shells:
        thickness = mesh.shell_thickness or 1.0
        file.add(KWord("SECTION_SHELL", [
            KCard.header(["secid", "elform", "shrf", "nip",
                          "propt", "qr/irid", "icomp", "setyp"], [10] * 8),
            KCard(values=[secid, 2, 1.0, 2, 0.0, 0.0, 0.0, 0.0],
                  widths=[10] * 8, kinds=["i", "i", "f", "i", "f", "f", "f", "i"]),
            KCard.header(["t1", "t2", "t3", "t4", "nloc", "marea", "idor", "edgset"], [10] * 8),
            KCard(values=[thickness] * 4 + [0.0, 0.0, 0.0, 0.0],
                  widths=[10] * 8, kinds=["f", "f", "f", "f", "f", "f", "i", "i"]),
        ]))

    file.add(KWord("MAT_ELASTIC", [
        KCard.header(["mid", "ro", "e", "pr", "da", "db"], [10, 10, 10, 10, 10, 10]),
        KCard(values=[mid, material["ro"], material["e"], material["pr"], 0.0, 0.0],
              widths=[10, 10, 10, 10, 10, 10],
              kinds=["i", "f", "f", "f", "f", "f"]),
    ]))

    file.add(KWord("PART", [
        KCard.header(["title"], [80]),
        KCard(values=[title], widths=[80], kinds=["s"]),
        KCard.header(["pid", "secid", "mid", "eosid", "hgid", "grav", "adpopt", "tmid"],
                     [10] * 8),
        KCard(values=[pid, secid, mid, 0, 0, 0, 0, 0], widths=[10] * 8,
              kinds=["i", "i", "i", "i", "i", "i", "i", "i"]),
    ]))

    # --------------------------------------------------------------- 单元
    if mesh.solids:
        block = KWord("ELEMENT_SOLID", [
            KCard.header(["eid", "pid", "n1", "n2", "n3", "n4",
                          "n5", "n6", "n7", "n8"], ELEMENT_WIDTHS),
        ])
        for eid, element_pid, corners in mesh.solids:
            block.add(KCard(values=[eid, element_pid, *corners], widths=ELEMENT_WIDTHS,
                            kinds=["i"] * 10))
        file.add(block)
    if mesh.shells:
        block = KWord("ELEMENT_SHELL", [
            KCard.header(["eid", "pid", "n1", "n2", "n3", "n4"], ELEMENT_WIDTHS),
        ])
        for eid, element_pid, corners in mesh.shells:
            block.add(KCard(values=[eid, element_pid, *corners], widths=ELEMENT_WIDTHS,
                            kinds=["i"] * 6))
        file.add(block)

    # ------------------------------------------------------------- 控制卡
    if endtim is not None:
        file.add(KWord("CONTROL_TERMINATION", [
            KCard.header(["endtim", "endcyc", "dtmin", "endeng", "endmas", "nosol"],
                         [10] * 6),
            KCard(values=[endtim, 0, 0.0, 0.0, 0.0, 0], widths=[10] * 6,
                  kinds=["f", "i", "f", "f", "f", "i"]),
        ]))

    file.add(KWord("END"))
    return file
