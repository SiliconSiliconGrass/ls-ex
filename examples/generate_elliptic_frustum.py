"""示例：生成椭圆台体壳（纯六面体网格）的 LS-DYNA .k 文件。

    python examples/generate_elliptic_frustum.py \
        --major-axis 40 --minor-axis 30 --height 50 --taper 0.6 \
        --thickness 3 --element-size 2 -o outputs/elliptic_frustum.k

几何约定：``major-axis`` / ``minor-axis`` / ``height`` / ``taper`` 定义的是**内腔**
椭圆台体；外轮廓是每个截面的椭圆等距曲线（法向偏移 = 壳厚）；两端各有厚度为
壳厚的实心端盖。总高度 = height + 2 * thickness。

在脚本里直接调用库接口：

    from ls_ex import elliptical_frustum_shell_mesh, build_keyword_file

    mesh = elliptical_frustum_shell_mesh(40, 30, 50, 0.6, 3, element_size=2)
    build_keyword_file(mesh).write("outputs/elliptic_frustum.k")
"""

from __future__ import annotations

import sys

from ls_ex.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["--kind", "elliptic-frustum", *sys.argv[1:]]))
