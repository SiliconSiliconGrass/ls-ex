# ls-ex

## 论文椭圆战斗部导出

`tools/generate_paper_elliptic_blast.py` 以现有 2 mm、General ALE 基线导出
8 个入口文件：

```text
paper_sph_{full,half-long,half-short,quarter}.k
paper_no-sph_{full,half-long,half-short,quarter}.k
```

每个入口通过 `*INCLUDE` 引用同目录的 `geometry_{对称形式}.k`。同一对称形式的
SPH/NO-SPH 模型共用一份几何，网格、节点和 segment set 不会复制到入口文件中。

```powershell
python tools/generate_paper_elliptic_blast.py
```

材料值来自论文表 3–5，并转换为 `mm-g-ms` / MPa。表 3 中
`σ0=1650 GPa` 与 30CrMnSi2A 的物理量级不符，模型采用 `1650 MPa`。
炸药的 A/B 为表中 4.157/−0.4587 GPa 转为 4157/−458.7 MPa；论文未给出的
`E0` 保持原模型值。起爆位置统一为炸药大端面中心 `(0, 0, 0)`。

`KeywordFile.load_resolved()` 会递归解析 `*INCLUDE`，并检测循环引用；
`split_geometry_include()` 可流式拆分大型 deck，避免构造逐单元 Python 对象。

## d3plot SPH 破片分析

```powershell
uv run ls-ex-sph D:\path\to\run\d3plot `
    --output-dir outputs\sph-fragments `
    --state -1 --axis 0,0,1 `
    --angle-mode velocity --angle-bin-size 10 `
    --outlier-method tukey --iqr-factor 3 `
    --speed-unit mm/ms --length-unit mm
```

输出包括：

- `sph_speed_animation.html`：带 Play/Pause 和拖动时间轴的动态 3D 粒子图，
  所有时间帧共用同一速度色标。
- `sph_speed_3d.html`：最后一个状态的按速度着色交互式 3D 粒子图。
- `sph_particle_sequence.npz`：动画中固定粒子子集的位置、速度和存活状态时间序列。
- `angular_distribution.html`：极角分箱的粒子数量、速度均值/分位数和二维分布。
- `sph_particles.csv` / `.npz`：全部 SPH 粒子的位置、速度和过滤标记。
- `sph_particles_filtered.csv` / `.npz`：实际用于统计和可视化的粒子。
- `angular_statistics.csv`：每个角度箱的数量、占比和速度统计。
- `angular_speed_histogram.csv`：角度 × 速度的二维计数。
- `summary.json`：状态、时间、过滤阈值、样本数量和输出索引。

位置按 `node_coordinates + node_displacement` 重建，速度读取
`node_velocity`，粒子编号通过 `sph_node_indexes` 映射。默认角度是速度方向
相对 `+Z` 轴的极角：0° 为沿 `+Z`，180° 为沿 `-Z`。可以用 `--axis`
和 `--angle-mode position --origin x,y,z` 修改定义。

默认 `tukey` 过滤以 Q3 + 3×IQR 截除速度上尾异常值，并始终排除
非有限值和 `sph_deletion != 0` 的粒子。若已知模型单位下的速度上限，
建议显式使用 `--speed-max`；也可以改用 `--outlier-method mad` 或
`--outlier-method percentile --speed-percentile 99`。

动画默认抽取 60 个状态和 3000 个粒子，可用
`--max-animation-frames`、`--max-animation-points` 调整。提高粒子数会让
HTML 明显变大；`--no-animate` 可关闭动画生成。

在 Python 中调用：

```python
from ls_ex import (
    analyze_sph_fragments,
    load_sph_particle_data,
    load_sph_particle_sequence,
)

particles = load_sph_particle_data("path/to/d3plot", state=-1)
sequence = load_sph_particle_sequence(
    "path/to/d3plot",
    max_frames=60,
    particle_indices=particles.particle_indices[:3000],
)
report = analyze_sph_fragments(
    "path/to/d3plot",
    "outputs/sph-fragments",
    axis=(0.0, 0.0, 1.0),
    angle_bin_size_deg=10.0,
    outlier_method="tukey",
)
```

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

## 椭圆台体壳（纯六面体）

```powershell
uv run ls-ex --kind elliptic-frustum `
    --major-axis 40 --minor-axis 30 --height 50 --taper 0.6 `
    --thickness 3 --element-size 2
# -> outputs/elliptic_frustum.k
```

参数：

| 参数 | 含义 |
| --- | --- |
| `--major-axis` / `--minor-axis` | 大端面椭圆的长轴 / 短轴（全长，不是半轴） |
| `--height` | 内腔椭圆台体高度 H |
| `--taper` | 小端面相对大端面的位似比 k |
| `--thickness` | 壳厚 h（侧壁和端盖相同） |
| `--element-size` | 目标单元尺寸，缺省 2 |

几何约定（重要）：

```
内腔：z=0 处椭圆 (a0,b0)，z=H 处椭圆 (k*a0,k*b0)，中间线性过渡
外轮廓：每个截面上内椭圆的等距曲线（法向偏移 h），不是 (a+h,b+h) 缩放椭圆
端盖：z∈[-h,0] 与 z∈[H,H+h]，截面是外等距曲线围成的实心区域，厚度 h
侧壁：z∈[0,H]，内椭圆与外等距曲线之间的环形区域
总高度 = H + 2h
```

网格采用"分区扫掠"：

- 侧壁：环向 × 径向 × 轴向的结构化扫掠；
- 端盖：外环（内椭圆 → 外等距曲线）沿用侧壁节点，内部椭圆盘用中心矩形 +
  四个 Coons 过渡块（蝶形分块），再沿轴向扫掠；
- 环向节点按象限弧长均匀分布，环向单元数由"中截面外轮廓"估算，让大端略粗、
  小端略细，尺寸整体围绕 `element-size` 分布（台体缩到 k 倍，环向尺寸也会按
  截面缩放；k=0.6 时大约 0.6~1.2 倍目标值）。

限制：等距曲线在椭圆长轴顶点的曲率半径为 `b²/a`，当壳厚接近该值（尤其在小端）
曲线会自交，程序会直接报错提示。示例（40×30、k=0.6、h=3）共约 4.4k 个六面体，
网格体积与解析值（Steiner 公式）误差 < 0.3%。

### 椭圆台体壳 + S-ALE 爆炸（侵蚀破片，无 SPH）

```powershell
.venv\Scripts\python.exe examples\build_elliptic_frustum_blast.py `
    --major-semi 43.2 --minor-semi 36 --height 100 --thickness 8 `
    --cone-angle 86.6 --element-size 2 --endtim 0.15 --fill cavity
# -> outputs/elliptic_frustum_blast_cavity.k
```

- `--cone-angle` 按**侧壁与端面夹角**换算小端位似比：`k = 1 - H/(a0*tan(θ))`。
- `--fill cavity`（默认）：用 `*INITIAL_VOLUME_FRACTION_NALEGP` 逐单元给出 air/TNT
  体积分数，TNT **精确填满椭圆内腔**（部分单元 5×5×5 采样）。文件较大（2mm 时约 32 MB）。
- `--fill cylinder`：腔内柱形装药（`GEOM=CYLINDER`，更快、文件小）。
- 空气域 = 结构外轮廓每侧再加 `--air-margin`（默认 40 mm）；2mm 网格约 75 万 S-ALE 单元。
- 壳体：`MAT_PLASTIC_KINEMATIC`（FS=0.2 失效应变）+ `*CONTACT_ERODING_SINGLE_SURFACE`
  （`erosop=1, iadj=1`）；`*ALE_STRUCTURED_FSI` 耦合壳体；起爆点 (0,0,10)。
- 单位制 mm-g-ms（应力 MPa）：空气 `MAT_NULL`+多项式 EOS；TNT `MAT_HIGH_EXPLOSIVE_BURN`+JWL
  （A=371200, B=3231, E0=7000, PCJ=21000, D=6930）。

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
    EllipticFrustum.py  椭圆台体壳（纯六面体）网格生成
    cli.py              命令行入口（ls-ex）
examples/
    generate_hollow_box.py  示例脚本
    generate_elliptic_frustum.py  椭圆台体壳示例
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
