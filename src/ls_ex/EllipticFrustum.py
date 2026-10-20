"""椭圆台体壳的纯六面体网格生成器。

几何约定
--------
* 给定的 ``major_axis`` / ``minor_axis``（全轴长）和 ``height`` / ``small_ratio``
  定义的是**内腔**：z=0 处椭圆半轴为 ``(a0, b0)``，z=H 处为
  ``(a0*k, b0*k)``，中间线性过渡。
* 外轮廓是每个截面上内椭圆的**等距曲线**（沿外法向偏移 ``thickness``），
  不是 ``(a+h, b+h)`` 的缩放椭圆。
* 上下两个端盖各厚 ``thickness``，截面是外等距曲线围成的实心区域；
  z ∈ [0, H] 的中间段是"内椭圆 → 外等距曲线"的环形侧壁。
* 总高度 = ``H + 2*thickness``。

网格分块
--------
* 侧壁：环向 × 径向 × 轴向的单块结构化网格。
* 端盖：外环（内椭圆 → 外等距曲线）沿用侧壁的节点分布；内部椭圆盘用
  "蝶形"五块分块（中心矩形 + 四个过渡块），再沿轴向扫掠。
* 环向节点按各象限的弧长均匀分布，单元尺寸接近 ``element_size``；
  由于台体是整体缩放的，小端环向尺寸约为大端的 ``small_ratio`` 倍。
"""

from __future__ import annotations

from math import cos, hypot, pi, sin
from typing import Sequence

from .KeywordGenerator import Mesh, Vec3, _clean, as_vec3


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not value > 0:
        raise ValueError(f"{name} 必须大于 0，得到 {value!r}")
    return value


def ellipse_perimeter(a: float, b: float) -> float:
    """Ramanujan 近似（相对误差 < 1e-5，用于估算环向网格数）。"""
    return pi * (3 * (a + b) - ((3 * a + b) * (a + 3 * b)) ** 0.5)


def ellipse_area(a: float, b: float) -> float:
    return pi * a * b


def offset_ellipse_area(a: float, b: float, distance: float) -> float:
    """等距曲线围成的面积（Steiner 公式，要求曲线不自交）。"""
    return ellipse_area(a, b) + ellipse_perimeter(a, b) * distance + pi * distance ** 2


def analytic_shell_volume(major_axis: float, minor_axis: float, height: float,
                          small_ratio: float, thickness: float) -> float:
    """壳体体积的解析值（用于检查网格）。"""
    a0, b0 = major_axis / 2, minor_axis / 2
    a1, b1 = a0 * small_ratio, b0 * small_ratio
    h = thickness
    l0 = ellipse_perimeter(a0, b0)
    # 侧壁：∫(L(z)h + pi h^2) dz，L(z) 随截面线性缩放
    wall = l0 * h * height * (1 + small_ratio) / 2 + pi * h ** 2 * height
    # 两个端盖：外等距曲线围成的实心面积 × 厚度
    bottom = h * offset_ellipse_area(a0, b0, h)
    top = h * offset_ellipse_area(a1, b1, h)
    return wall + bottom + top


def _quadrant_angles(a: float, b: float, start: float, stop: float,
                     cells: int) -> list[float]:
    """在 [start, stop] 内按弧长等分，返回 cells+1 个参数角。"""
    fine = max(256, cells * 64)
    thetas = [start + (stop - start) * index / fine for index in range(fine + 1)]
    points = [(a * cos(theta), b * sin(theta)) for theta in thetas]
    cumulative = [0.0]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + hypot(x1 - x0, y1 - y0))
    total = cumulative[-1]

    result = []
    cursor = 0
    for index in range(cells + 1):
        target = total * index / cells
        while cursor < fine and cumulative[cursor + 1] < target:
            cursor += 1
        if cursor >= fine:
            result.append(stop)
            continue
        segment = cumulative[cursor + 1] - cumulative[cursor]
        fraction = 0.0 if segment <= 0 else (target - cumulative[cursor]) / segment
        result.append(thetas[cursor] + (thetas[cursor + 1] - thetas[cursor]) * fraction)
    result[0], result[-1] = start, stop
    return result


def _offset_point(a: float, b: float, theta: float, distance: float) -> tuple[float, float]:
    """椭圆 (a cosθ, b sinθ) 沿外法向偏移 distance。"""
    ct, st = cos(theta), sin(theta)
    nx, ny = b * ct, a * st
    norm = hypot(nx, ny)
    return (a * ct + distance * nx / norm, b * st + distance * ny / norm)


def _lerp(p: Sequence[float], q: Sequence[float], t: float) -> tuple[float, float]:
    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def _keep_symmetry_sector(x: float, y: float, symmetry: str) -> bool:
    """Return whether an element centroid belongs to the requested symmetry sector."""
    if symmetry == "full":
        return True
    if symmetry == "half-long":
        # A cut along the major (x) axis lies in the x-z plane and keeps y >= 0.
        return y > 0.0
    if symmetry == "half-short":
        # A cut along the minor (y) axis lies in the y-z plane and keeps x >= 0.
        return x > 0.0
    if symmetry == "quarter":
        return x > 0.0 and y > 0.0
    raise ValueError(f"unknown symmetry model: {symmetry!r}")


def _truncate_symmetry(builder: "_Builder", symmetry: str) -> None:
    """Discard complementary elements and compact the remaining node IDs."""
    if symmetry == "full":
        return

    kept_solids = []
    used_nodes = set()
    for _, pid, corners in builder.solids:
        center_x = sum(builder.nodes[nid][0] for nid in corners) / len(corners)
        center_y = sum(builder.nodes[nid][1] for nid in corners) / len(corners)
        if _keep_symmetry_sector(center_x, center_y, symmetry):
            kept_solids.append((pid, corners))
            used_nodes.update(corners)

    remap = {old_nid: new_nid for new_nid, old_nid in enumerate(sorted(used_nodes), 1)}
    builder.nodes = {remap[old_nid]: builder.nodes[old_nid] for old_nid in sorted(used_nodes)}
    builder.ids = {(round(x, 9), round(y, 9), round(z, 9)): nid
                   for nid, (x, y, z) in builder.nodes.items()}
    builder.solids = [
        (new_eid, pid, tuple(remap[nid] for nid in corners))
        for new_eid, (pid, corners) in enumerate(kept_solids, 1)
    ]


def _coons_block(inner: Sequence[tuple[float, float]],
                 outer: Sequence[tuple[float, float]],
                 radial_cells: int) -> list[list[tuple[float, float]]]:
    """内/外边之间用 Coons 插值生成 [径向][环向] 的结构化二维网格。"""
    cells = len(inner) - 1
    grid = []
    for v_index in range(radial_cells + 1):
        v = v_index / radial_cells
        left = _lerp(inner[0], outer[0], v)
        right = _lerp(inner[-1], outer[-1], v)
        row = []
        for u_index in range(cells + 1):
            u = u_index / cells
            corner = ((1 - u) * (1 - v) * inner[0][0] + u * (1 - v) * inner[-1][0]
                      + (1 - u) * v * outer[0][0] + u * v * outer[-1][0]), (
                      (1 - u) * (1 - v) * inner[0][1] + u * (1 - v) * inner[-1][1]
                      + (1 - u) * v * outer[0][1] + u * v * outer[-1][1])
            x = ((1 - v) * inner[u_index][0] + v * outer[u_index][0]
                 + (1 - u) * left[0] + u * right[0] - corner[0])
            y = ((1 - v) * inner[u_index][1] + v * outer[u_index][1]
                 + (1 - u) * left[1] + u * right[1] - corner[1])
            row.append((x, y))
        grid.append(row)
    return grid


class _Builder:
    """按坐标去重节点，并按扫掠块生成六面体。"""

    def __init__(self) -> None:
        self.nodes: dict[int, Vec3] = {}
        self.ids: dict[tuple[float, float, float], int] = {}
        self.solids: list[tuple[int, int, tuple[int, ...]]] = []

    def node(self, x: float, y: float, z: float) -> int:
        # 9 位小数足够（mm 级几何），同时能容忍 Coons 插值与直接椭圆点之间
        # ~1e-12 的浮点差异，避免同一位置生成重复节点。
        key = (round(x, 9), round(y, 9), round(z, 9))
        existing = self.ids.get(key)
        if existing is None:
            existing = len(self.nodes) + 1
            self.ids[key] = existing
            self.nodes[existing] = (key[0], key[1], key[2])
        return existing

    def sweep(self, grids: Sequence[Sequence[Sequence[tuple[float, float]]]],
              zs: Sequence[float], pid: int, cyclic_second: bool = False) -> None:
        """把一组二维网格沿 zs 扫掠成六面体。

        约定：``grid[i][j]`` 中 i 方向 × j 方向 = +z（保证体积为正）；
        ``cyclic_second=True`` 时 j 方向首尾相接。
        """
        if len(grids) != len(zs):
            raise ValueError("grids 与 zs 数量不一致")
        ids: list[list[list[int]]] = []
        for grid, z in zip(grids, zs):
            layer = [[self.node(x, y, z) for (x, y) in row] for row in grid]
            ids.append(layer)

        for k in range(len(zs) - 1):
            first, second = ids[k], ids[k + 1]
            ni = len(first)
            nj = len(first[0])
            j_limit = nj if cyclic_second else nj - 1
            for i in range(ni - 1):
                for j in range(j_limit):
                    jn = (j + 1) % nj
                    corners = (first[i][j], first[i + 1][j], first[i + 1][jn], first[i][jn],
                               second[i][j], second[i + 1][j], second[i + 1][jn], second[i][jn])
                    self.solids.append((len(self.solids) + 1, pid, corners))


def _annulus_grid(a: float, b: float, angles: Sequence[float], thickness: float,
                  radial_cells: int) -> list[list[tuple[float, float]]]:
    rows = []
    for r_index in range(radial_cells + 1):
        f = r_index / radial_cells
        row = []
        for theta in angles:
            inner = (a * cos(theta), b * sin(theta))
            outer = _offset_point(a, b, theta, thickness)
            row.append(_lerp(inner, outer, f))
        rows.append(row)
    return rows


def _butterfly_blocks(a: float, b: float, m_right: int, m_top: int,
                      angles_right: Sequence[float], angles_top: Sequence[float],
                      angles_left: Sequence[float], angles_bottom: Sequence[float],
                      element_size: float) -> tuple[list, list]:
    """返回 (中心矩形网格, 四个 Coons 过渡块网格)，坐标已映射到物理椭圆。"""
    core = 0.50  # 归一化坐标里中心矩形的半宽；须 < 1/sqrt(2)
    # 四个过渡块共用同一个径向层数：相邻块在对角线上的节点必须一一对应，
    # 否则块与块之间会出现不共形的"零厚度裂缝"（单元不连通）。
    radial_cells = max(1, round((1 - core) * (a + b) / 2 / element_size))
    square = [
        [(a * (-core + 2 * core * i / m_top), b * (-core + 2 * core * j / m_right))
         for j in range(m_right + 1)]
        for i in range(m_top + 1)
    ]

    def block(inner_normalized, outer_angles, radial_cells):
        outer = [(cos(t), sin(t)) for t in outer_angles]
        grid = _coons_block(inner_normalized, outer, radial_cells)
        return [[(a * x, b * y) for (x, y) in row] for row in grid]

    # 右块：内边 = 中心矩形右边 (c,-c)->(c,c)
    right_inner = [(core, -core + 2 * core * i / m_right) for i in range(m_right + 1)]
    right = block(right_inner, angles_right, radial_cells)
    # 上块：内边 = 中心矩形上边 (c,c)->(-c,c)
    top_inner = [(core - 2 * core * i / m_top, core) for i in range(m_top + 1)]
    top = block(top_inner, angles_top, radial_cells)
    # 左块：(-c,c)->(-c,-c)
    left_inner = [(-core, core - 2 * core * i / m_right) for i in range(m_right + 1)]
    left = block(left_inner, angles_left, radial_cells)
    # 下块：(-c,-c)->(c,-c)
    bottom_inner = [(-core + 2 * core * i / m_top, -core) for i in range(m_top + 1)]
    bottom = block(bottom_inner, angles_bottom, radial_cells)
    return square, [right, top, left, bottom]


def elliptical_frustum_shell_mesh(
    major_axis: float,
    minor_axis: float,
    height: float,
    small_ratio: float,
    thickness: float,
    element_size: float = 2.0,
    origin: object = (0.0, 0.0, 0.0),
    pid: int = 1,
    symmetry: str = "full",
) -> Mesh:
    """生成椭圆台体壳的纯六面体网格。

    参数
    ----
    major_axis, minor_axis:
        大端面椭圆的长轴 / 短轴（全长，不是半轴）。
    height:
        内腔台体高度。
    small_ratio:
        小端面相对大端面的位似比。
    thickness:
        壳厚（侧壁与端盖相同）。
    element_size:
        目标单元尺寸，缺省 2（和模型单位一致，例如 mm）。
    """
    major_axis = _positive(major_axis, "major_axis")
    minor_axis = _positive(minor_axis, "minor_axis")
    height = _positive(height, "height")
    small_ratio = _positive(small_ratio, "small_ratio")
    thickness = _positive(thickness, "thickness")
    element_size = _positive(element_size, "element_size")
    shift = as_vec3(origin, "origin")
    if symmetry not in {"full", "half-long", "half-short", "quarter"}:
        raise ValueError("symmetry must be full, half-long, half-short, or quarter")

    a0, b0 = major_axis / 2, minor_axis / 2
    a1, b1 = a0 * small_ratio, b0 * small_ratio
    # 等距曲线在长轴顶点处曲率半径 = b^2/a；偏移量超过它曲线会自交
    curvature_radius = min(b0 * b0 / a0, b1 * b1 / a1)
    if thickness >= curvature_radius:
        raise ValueError(
            f"壳厚 {thickness:g} 超过椭圆等距曲线的极限：长轴顶点处最小曲率半径约 "
            f"{curvature_radius:.6g}，等距曲线会自交。请减小壳厚或增大小端位似比。"
        )

    # 环向：按中截面（大端/小端的平均）的象限弧长分配单元数，
    # 这样大端略粗、小端略细，整体尺寸围绕 element_size 分布。
    def quadrant_cells(a: float, b: float, start: float, stop: float) -> int:
        sample = _quadrant_angles(a, b, start, stop, 32)
        length = sum(hypot(a * cos(t1) - a * cos(t0), b * sin(t1) - b * sin(t0))
                     for t0, t1 in zip(sample, sample[1:]))
        # 外等距曲线上同一象限的弧长 = 内弧长 + (π/2)·thickness（总转角 90°）
        return max(1, round((length + 0.5 * pi * thickness) / element_size))

    a_mid, b_mid = (a0 + a1) / 2, (b0 + b1) / 2
    m_right = quadrant_cells(a_mid, b_mid, -pi / 4, pi / 4)
    m_top = quadrant_cells(a_mid, b_mid, pi / 4, 3 * pi / 4)
    if symmetry != "full":
        # Keep x=0 and y=0 on mesh faces so the symmetry cuts have no jagged edges.
        m_right = max(2, 2 * round(m_right / 2))
        m_top = max(2, 2 * round(m_top / 2))
    right_angles = _quadrant_angles(a0, b0, -pi / 4, pi / 4, m_right)
    top_angles = _quadrant_angles(a0, b0, pi / 4, 3 * pi / 4, m_top)
    left_angles = _quadrant_angles(a0, b0, 3 * pi / 4, 5 * pi / 4, m_right)
    bottom_angles = _quadrant_angles(a0, b0, 5 * pi / 4, 7 * pi / 4, m_top)
    angles = right_angles[:-1] + top_angles[:-1] + left_angles[:-1] + bottom_angles[:-1]

    radial_cells = max(1, round(thickness / element_size))
    wall_cells = max(1, round(height / element_size))
    cap_cells = max(1, round(thickness / element_size))
    z_wall = [height * i / wall_cells for i in range(wall_cells + 1)]

    builder = _Builder()

    def make_annulus(a: float, b: float, z: float) -> list[list[tuple[float, float]]]:
        return _annulus_grid(a, b, angles, thickness, radial_cells)

    # 侧壁：逐层截面（内腔随 z 线性收缩）
    wall_grids = []
    for z in z_wall:
        scale = 1 + (small_ratio - 1) * z / height
        wall_grids.append(make_annulus(a0 * scale, b0 * scale, z))
    builder.sweep(wall_grids, z_wall, pid, cyclic_second=True)

    # 两端端盖：外环 + 蝶形内盘，截面在厚度方向不变
    for a, b, z_start in ((a0, b0, -thickness), (a1, b1, height)):
        z_cap = [z_start + thickness * i / cap_cells for i in range(cap_cells + 1)]
        annulus = make_annulus(a, b, 0.0)
        builder.sweep([annulus] * len(z_cap), z_cap, pid, cyclic_second=True)

        # 蝶形分块用大端角度分布（各端有自己的 m 数）
        qr = _quadrant_angles(a, b, -pi / 4, pi / 4, m_right)
        qt = _quadrant_angles(a, b, pi / 4, 3 * pi / 4, m_top)
        ql = _quadrant_angles(a, b, 3 * pi / 4, 5 * pi / 4, m_right)
        qb = _quadrant_angles(a, b, 5 * pi / 4, 7 * pi / 4, m_top)
        square, blocks = _butterfly_blocks(a, b, m_right, m_top, qr, qt, ql, qb,
                                           element_size)
        builder.sweep([square] * len(z_cap), z_cap, pid)
        for block in blocks:
            builder.sweep([block] * len(z_cap), z_cap, pid)

    # 平移到 origin
    _truncate_symmetry(builder, symmetry)

    if shift != (0.0, 0.0, 0.0):
        builder.nodes = {nid: (x + shift[0], y + shift[1], z + shift[2])
                         for nid, (x, y, z) in builder.nodes.items()}

    mesh = Mesh(nodes=builder.nodes, solids=builder.solids, pid=pid,
                description=(f"elliptic frustum shell major={major_axis} minor={minor_axis} "
                             f"H={height} k={small_ratio} t={thickness} h={element_size} "
                             f"symmetry={symmetry}"))
    return mesh
