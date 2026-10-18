# ls-ex

读写 LS-DYNA `.k`（keyword）文件的小工具，外加两个网格生成器：实心长方体和**空心长方体（长方体壳）**。

## 快速开始

```powershell
# 生成空心长方体（实体网格）：内腔 20×10×10，壁厚 2
uv run ls-ex --inner 20,10,10 --thickness 2 -e 2 -o outputs/hollow_box.k

# 只用壳单元表示的长方体壳，壁厚写进 *SECTION_SHELL
uv run ls-ex --kind shell --inner 20,10,10 --thickness 2 -e 2 -o outputs/hollow_box_shell.k
```

也可以用项目自带的虚拟环境直接跑（不需要重新安装）：

```powershell
.venv\Scripts\python.exe examples\generate_hollow_box.py --inner 20,10,10 --thickness 2 -o outputs/hollow_box.k
```

命令行参数：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--kind` | `solid` 实体空心长方体 / `shell` 壳单元 / `box` 实心长方体 | `solid` |
| `--inner` | 内腔尺寸 `cx,cy,cz`（只写一个数表示正方体空腔） | `20,10,10` |
| `--thickness` | 壳（壁）厚，六个面一样 | `2.0` |
| `--element-size` | 目标单元尺寸，缺省等于壁厚 | `= thickness` |
| `--n-wall` | 壁厚方向的单元层数，覆盖由 `element-size` 推出的值 | 自动 |
| `--origin` | 外箱体最小角点坐标 | `0,0,0` |
| `--title` | 写进 `*TITLE` / `*PART` 的标题 | 自动 |
| `--endtim` | `*CONTROL_TERMINATION` 的 ENDTIM | `1.0` |
| `-o, --output` | 输出文件 | `outputs/hollow_box.k` |

`--thickness 2 --element-size 2` 时壁厚方向只有一层单元；想要更细就把 `--element-size` 调小，
或者用 `--n-wall` 直接指定层数（例如 `--n-wall 3` 得到 3 层壁单元）。

## 几何是怎么定的

给定内腔尺寸 `inner_size` 和壁厚 `thickness`：

```
外轮廓尺寸 = inner_size + 2 * thickness
```

实体网格的做法是：先把外箱体按三个方向分格，每个方向都切成 **壁 - 空腔 - 壁** 三段，
保证网格线正好落在空腔的六个面上；然后丢掉完全落在空腔内部的单元，剩下的单元就组成“壳”。
所以空腔是一个规整的直角长方体，内壁不会出现“阶梯”状的锯齿边界。

```
        thickness（1 层单元）
     ┌────┬───────────────────┬────┐
     │    │                   │    │
     │    │   inner_size      │    │  ← 空腔（被挖掉）
     │    │                   │    │
     └────┴───────────────────┴────┘
```

壳单元版本（`--kind shell`）把网格画在壁厚的**中面**上，中面尺寸 = `inner_size + thickness`，
再通过 `*SECTION_SHELL` 的 `T1..T4` 给出壁厚。单元法向统一朝外。

## 生成的文件里有什么

```
*KEYWORD
*TITLE
*NODE
*SECTION_SOLID        （或 *SECTION_SHELL，带壁厚）
*MAT_ELASTIC
*PART
*ELEMENT_SOLID        （或 *ELEMENT_SHELL）
*CONTROL_TERMINATION
*END
```

节点编号从 1 连续排到 N，只输出被单元引用到的节点；`*NODE` / `*ELEMENT_*` 的列宽和
`$#` 提示行都按 LS-PrePost 的风格写，可以直接丢进 LS-PrePost 查看模型。

默认材料是 mm-ton-s 单位制的钢：`RO=7.85e-9`、`E=210000`、`PR=0.3`。如果你用的是其它
单位制（比如 m-kg-s），在 `build_keyword_file(..., material={...})` 里替换即可。

## 当库用

```python
from ls_ex import KeywordFile, hollow_box_mesh, build_keyword_file

# 1) 生成几何
mesh = hollow_box_mesh(inner_size=(20, 10, 10), thickness=2.0, element_size=1.0)
print(mesh.summary())          # nodes=... solids=... volume=...

# 2) 组装成关键字文件
deck = build_keyword_file(mesh, title="hollow box")
deck.write("outputs/hollow_box.k")
```

解析现成的 `.k`（例如 `test1.k`）：

```python
from ls_ex import KeywordFile

deck = KeywordFile.load("test1.k")
print(list(deck.iter_keyword_names()))   # ['KEYWORD', 'TITLE', 'NODE', 'PART', 'ELEMENT_SOLID', 'END']
deck.nodes()                             # {1: (1.0, 1.0, 1.0), ...}
deck.solid_elements()                    # {1: (1, (1, 5, 37, 3, 2, 13, 45, 4)), ...}
deck.parts()                             # {1: {'title': 'BlockMesh1', 'secid': 0, 'mid': 0}}
deck.write("outputs/copy.k")             # 原样写回
```

自己搭关键字块（`KWord` = 一个 `*XXX` 块，`KCard` = 块里的一行）：

```python
from ls_ex import KCard, KWord, KeywordFile

deck = KeywordFile(header=["$# my deck"])
deck.add(KWord("KEYWORD"))
deck.add(KWord("CONTROL_TERMINATION", [
    KCard(values=[0.5, 0, 0.0, 0.0, 0.0, 0]),   # 按 10 字符字段自动排版
]))
deck.add(KWord("END"))
print(deck.dumps())
```

## 目录结构

```
src/ls_ex/
    KCard.py            一行 card 的定长格式读写
    KWord.py            一个 *XXX 关键字块
    KeywordFile.py      整个 .k 的读写 + 节点/单元/part 提取
    KeywordGenerator.py 长方体 / 空心长方体 / 壳单元网格生成
    cli.py              命令行入口（ls-ex）
examples/
    generate_hollow_box.py  示例脚本
tests/                  单元测试（网格体积、封闭性、法向、读写往返）
outputs/                所有生成的 .k 文件（已 gitignore，只保留 .gitkeep）
test1.k                 原始示例文件（测试用的输入，保留在 git 里）
```

## 测试

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
```

测试会检查：单元体积之和等于「外箱体 − 空腔」、每个六面体体积为正（否则 LS-DYNA 会报
negative volume）、网格表面封闭无裂缝、壳单元法向朝外，以及读写往返不丢数据。

## 改造已有的 deck（tools/）

把自己的 `.k`（例如 `shell-explosion-no-sph` 那种圆筒壳 + S-ALE 装药的 deck）改造成
"方壳 + 内腔装药"时，可以用这三个脚本，它们都只动该动的关键字，其余逐行原样保留：

```powershell
# 1) 把 *NODE + *ELEMENT_SOLID 换成空心长方体（内腔 40、壁厚 8、网格 2）
.venv\Scripts\python.exe tools\replace_solid_geometry.py `
    --source D:\path\to\main.k --output outputs\main_hollow_cube.k `
    --inner 40,40,40 --thickness 8 --element-size 2 --pid 2

# 2) 把炸药填进内腔：改写成 *DEFINE_BOX + GEOM=BOXCOR 的体积填充
.venv\Scripts\python.exe tools\fill_cavity_tnt.py `
    --source outputs\main_hollow_cube.k --output outputs\main_hollow_cube_tnt.k `
    --box=-20,20,-20,20,8,48 --detonation 0,0,10

# 3)（可选）把强度/压力数据统一到 mm-g-ms(MPa) 单位制，炸药才会真的产生压力
.venv\Scripts\python.exe tools\fix_stress_units.py `
    --source outputs\main_hollow_cube_tnt.k --output outputs\main_hollow_cube_tnt_units.k
```

三个脚本对应的三个坑：

1. **`GEOM=PART` 填的是那个 part 的单元体积**，不是它的内腔。用钢壳 part 去填，炸药会跑到
   壁厚里，腔内一个都没有，起爆点落空 → 不报错但也不爆。
2. **填充几何类型里没有 `BOX`**：可选值是 `ALL/PARTSET/PART/SEGSET/PLANE/CYLINDER/BOXCOR/BOXCPT/ELLIPSOID`，
   长方体要用 `BOXCOR` + `*DEFINE_BOX`（写成 `BOX` 会直接 `Error 10246/10450`）。
3. **单位制要自洽**：这套 deck 的长度/质量/时间单位是 mm-g-ms（密度、爆速都对得上），
   但钢的 `E=201`、TNT 的 `A/B/E0/PCJ` 却是另一套单位下的数（小了 10³～10⁶ 倍），
   于是 JWL 完全产生不了压力，看起来就是"没爆"。

### 让壳体失效后碎裂成破片

```powershell
# 侵蚀版（推荐，实测稳定）：加失效应变 + 侵蚀感知自接触
.venv\Scripts\python.exe tools\enable_fragmentation.py `
    --source outputs\main_hollow_cube_tnt_units.k --output outputs\main_hollow_cube_frag.k `
    --failure-strain 0.2

# 实验版：让失效单元转成 SPH 破片
.venv\Scripts\python.exe tools\enable_fragmentation.py `
    --source outputs\main_hollow_cube_tnt_units.k --output outputs\main_hollow_cube_frag_sph.k `
    --failure-strain 0.2 --sph
```

要点：

1. `*MAT_PLASTIC_KINEMATIC` 的 `FS` **默认是 0 = 永不失效**，所以壳只会被撑开、扭曲。
   给个失效应变（钢 0.05~0.4）单元就会失效/侵蚀，壳才会"破"。
2. `*CONTACT_ERODING_SINGLE_SURFACE` 在 SFS 卡之后**还有一张 `isym/erosop/iadj` 卡**，
   漏掉它 LS-DYNA 会把下一行当这张卡读 → `Error 10246 (KEY+246)`；`erosop=1`（为侵蚀
   分配存储）和 `iadj=1`（允许实体内部侵蚀后继续接触）是破片接触必须的。
3. 侵蚀接触会自动给所有实体 part 打开"负体积失效判据"（LS-DYNA `Warning 30364`），
   这也是为什么加了接触以后相机单元可能比 `FS` 更早失效。
4. `--sph` 用 `*DEFINE_ADAPTIVE_SOLID_TO_SPH` 把失效单元转成 SPH 粒子（`icpl=0` 即
   "failure without coupling"= 破片模式）。实测 LS-DYNA 会先建好 13952 个粒子
   （每个实体单元一个），但**加了侵蚀接触时粒子不会被激活**（负体积判据抢先删单元）；
   去掉接触后粒子会激活，但 SPH 解在 t≈0.01 发散。要走这条路还需要调
   `SECTION_SPH` 的 `CSLH/HMIN/HMAX/SPHINI`、`*CONTROL_SPH` 的 `DT/FORM/IAVIS`、
   `NQ`（1/8/27 个粒子每单元）以及失效判据。
