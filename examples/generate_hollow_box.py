"""示例：生成空心长方体（长方体壳）的 LS-DYNA .k 文件。

只是 ``ls-ex`` 命令行入口的一层薄封装，方便直接 ``python examples/generate_hollow_box.py``。

    python examples/generate_hollow_box.py --inner 20,10,10 --thickness 2 -o outputs/hollow_box.k
    python examples/generate_hollow_box.py --kind shell --inner 20,10,10 --thickness 2

如果要在自己的脚本里调用库接口，核心就三行::

    from ls_ex import hollow_box_mesh, build_keyword_file

    mesh = hollow_box_mesh(inner_size=(20, 10, 10), thickness=2.0, element_size=1.0)
    build_keyword_file(mesh, title="hollow box").write("outputs/hollow_box.k")
"""

from __future__ import annotations

from ls_ex.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
