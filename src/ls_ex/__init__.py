"""ls_ex —— 一个很小的 LS-DYNA (.k) 关键字文件读写 + 网格生成工具包。

读写：

    from ls_ex import KeywordFile

    deck = KeywordFile.load("test1.k")          # 解析
    nodes = deck.nodes()                        # {nid: (x, y, z)}
    solids = deck.solid_elements()              # {eid: (pid, (n1..n8))}
    deck.write("copy.k")                        # 再写回去

生成空心长方体（壳）：

    from ls_ex import hollow_box_mesh, build_keyword_file

    mesh = hollow_box_mesh(inner_size=(20, 10, 10), thickness=2.0, element_size=1.0)
    build_keyword_file(mesh, title="hollow box").write("hollow_box.k")
"""

from .KCard import KCard, format_float, format_int, format_str, typed_card
from .KWord import KWord
from .KeywordFile import KeywordFile, parse_numbers
from .KeywordGenerator import (
    Mesh,
    as_vec3,
    box_mesh,
    build_keyword_file,
    hex_volume,
    hollow_box_mesh,
    hollow_box_shell_mesh,
)
from .EllipticFrustum import (
    analytic_shell_volume,
    ellipse_perimeter,
    elliptical_frustum_shell_mesh,
)
from .Voxelization import (
    SurfaceMesh,
    VoxelGeometry,
    VoxelRegion,
    air_ale_voxel_count,
    build_general_ale_blast_deck,
    build_geometry_keyword_file,
    load_glb,
    load_stl,
    load_surface_mesh,
    surface_is_watertight,
    voxelize_surface,
)
from .sph_fragments import (
    AngularSpeedHistogram,
    AngularStatistics,
    OutlierFilterReport,
    SPHParticleData,
    SPHParticleSequence,
    analyze_sph_fragments,
    angular_speed_histogram,
    angular_statistics,
    filter_sph_outliers,
    load_sph_particle_data,
    load_sph_particle_sequence,
    make_angle_bins,
    particle_angles_deg,
    plot_angular_distribution,
    plot_sph_speed_3d,
    plot_sph_speed_animation,
)
from .cli import main

__version__ = "0.1.0"

__all__ = [
    "KCard",
    "KWord",
    "KeywordFile",
    "Mesh",
    "AngularSpeedHistogram",
    "AngularStatistics",
    "OutlierFilterReport",
    "SPHParticleData",
    "SPHParticleSequence",
    "SurfaceMesh",
    "VoxelGeometry",
    "VoxelRegion",
    "analytic_shell_volume",
    "analyze_sph_fragments",
    "angular_speed_histogram",
    "angular_statistics",
    "air_ale_voxel_count",
    "as_vec3",
    "box_mesh",
    "build_general_ale_blast_deck",
    "build_geometry_keyword_file",
    "build_keyword_file",
    "ellipse_perimeter",
    "elliptical_frustum_shell_mesh",
    "format_float",
    "format_int",
    "format_str",
    "filter_sph_outliers",
    "hex_volume",
    "hollow_box_mesh",
    "hollow_box_shell_mesh",
    "load_glb",
    "load_sph_particle_data",
    "load_sph_particle_sequence",
    "load_stl",
    "load_surface_mesh",
    "main",
    "make_angle_bins",
    "parse_numbers",
    "particle_angles_deg",
    "plot_angular_distribution",
    "plot_sph_speed_3d",
    "plot_sph_speed_animation",
    "surface_is_watertight",
    "typed_card",
    "voxelize_surface",
]
