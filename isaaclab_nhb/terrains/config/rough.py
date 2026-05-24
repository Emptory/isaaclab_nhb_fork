# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
import isaaclab_nhb.terrains.trimesh as terrain_gen_nhb

from isaaclab.terrains import TerrainGeneratorCfg

# 在lab默认地形的基础上去除了上下楼梯和随机方块地形
ROUGH_TERRAINS_SIMPLE_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=30.0, # border是在地形区域外扩充一圈平地，设置平地的宽度
    border_height = 1.0, # border的高度，负数是向上砌墙，正数改了没有反应
    num_rows=10, # 列数，一共多少级
    num_cols=20, # 行数，一共多少个赛道
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # # 随机方块地形
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # 碎石地
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        # 上坡平顶金字塔
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.3, slope_range=(0.0, 0.2), platform_width=2.0, border_width=0.25
        ),
        # 下坡平底金字塔
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.3, slope_range=(0.0, 0.2), platform_width=2.0, border_width=0.25
        ),
        # 平地
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.2,
        ),
    },
)


# 只保留台阶和box地形
ROUGH_TERRAINS_STAIRS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=30.0, # border是在地形区域外扩充一圈平地，设置平地的宽度
    border_height = 1.0, # border的高度，负数是向上砌墙，正数改了没有反应
    num_rows=10, # 列数，一共多少级
    num_cols=20, # 行数，一共多少个赛道
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.0, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.0, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # 随机方块地形
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0
        ),
        # 平地
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.2,
        ),
    },
)

# 纯平地
ROUGH_TERRAINS_PLANE_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=30.0, # border是在地形区域外扩充一圈平地，设置平地的宽度
    border_height = 1.0, # border的高度，负数是向上砌墙，正数改了没有反应
    num_rows=10, # 列数，一共多少级
    num_cols=20, # 行数，一共多少个赛道
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0,
        ),
    },
)


# 给高程图训练用
ROUGH_ELEVATION_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0), # 每个地形的大小
    border_width=30.0, # border是在地形区域外扩充一圈平地，设置平地的宽度
    border_height = 1.0, # border的高度，负数是向上砌墙，正数改了没有反应
    num_rows=10, # 列数，一共多少级
    num_cols=20, # 行数，一共多少个赛道
    horizontal_scale=0.1, # 地形水平分辨率
    vertical_scale=0.005, # 地形高度分辨率
    slope_threshold=0.75, # tan超过此值的斜坡会变成墙
    use_cache=False, # 不能开，可能导致地形修改不成功
    curriculum=True,
    sub_terrains={
        # 上楼梯
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2, 
            step_height_range=(0.0, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # 下楼梯
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        # 同心沟壑地形 (concentric moats)
        "concentric_moats": terrain_gen_nhb.MeshConcentricMoatsTerrainCfg(
            proportion=0.2,
            platform_width=3.0,
            moat_width_range=(0.0, 0.3),
            moat_depth_range=(0.0, 1.5),
            num_moats=3,
            platform_ring_width=0.7,
        ),
        # 平地
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.1,
        ),
    },
)

ROUGH_ELEVATION_RICH_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0), # 每个地形的大小
    border_width=30.0, # border是在地形区域外扩充一圈平地，设置平地的宽度
    border_height = 1.0, # border的高度，负数是向上砌墙，正数改了没有反应
    num_rows=10, # 列数，一共多少级
    num_cols=16, # 行数，一共多少个赛道
    horizontal_scale=0.1, # 地形水平分辨率
    vertical_scale=0.005, # 地形高度分辨率
    slope_threshold=0.75, # tan超过此值的斜坡会变成墙
    use_cache=False, # 不能开，可能导致地形修改不成功
    curriculum=True,
    sub_terrains={
        # 上楼梯
        "pyramid_stairs_narrow": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05, 
            step_height_range=(0.0, 0.23),
            step_width=0.25,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),

        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05, 
            step_height_range=(0.0, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_wide": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05, 
            step_height_range=(0.0, 0.23),
            step_width=0.4,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_wider": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05, 
            step_height_range=(0.0, 0.23),
            step_width=0.5,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_widest": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05, 
            step_height_range=(0.0, 0.23),
            step_width=0.6,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # 下楼梯
        "pyramid_stairs_inv_narrow": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.23),
            step_width=0.25,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv_wide": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.23),
            step_width=0.4,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv_wider": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.23),
            step_width=0.5,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv_widest": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.23),
            step_width=0.6,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        # 同心沟壑地形 (concentric moats)
        "concentric_moats_narrow": terrain_gen_nhb.MeshConcentricMoatsTerrainCfg(
            proportion=0.05,
            platform_width=3.0,
            moat_width_range=(0.0, 0.4),
            moat_depth_range=(0.0, 0.4),
            num_moats=3,
            platform_ring_width=0.7,
        ),
        "concentric_moats": terrain_gen_nhb.MeshConcentricMoatsTerrainCfg(
            proportion=0.05,
            platform_width=3.0,
            moat_width_range=(0.0, 0.4),
            moat_depth_range=(0.3, 0.8),
            num_moats=3,
            platform_ring_width=0.7,
        ),
        "concentric_moats_wide": terrain_gen_nhb.MeshConcentricMoatsTerrainCfg(
            proportion=0.05,
            platform_width=3.0,
            moat_width_range=(0.0, 0.4),
            moat_depth_range=(0.7, 1.2),
            num_moats=3,
            platform_ring_width=0.7,
        ),
        # 平地
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.05,
        ),
    },
)

# 测试用
ROUGH_ELEVATION_RICH_FOR_TEST_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0), # 每个地形的大小
    border_width=30.0, # border是在地形区域外扩充一圈平地，设置平地的宽度
    border_height = 1.0, # border的高度，负数是向上砌墙，正数改了没有反应
    num_rows=5, # 列数，一共多少级
    num_cols=5, # 行数，一共多少个赛道
    horizontal_scale=0.1, # 地形水平分辨率
    vertical_scale=0.005, # 地形高度分辨率
    slope_threshold=0.75, # tan超过此值的斜坡会变成墙
    use_cache=False, # 不能开，可能导致地形修改不成功
    curriculum=True,
    sub_terrains={
        # 下楼梯
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.05, 
        #     step_height_range=(0.15, 0.15),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # 上楼梯
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.15, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.3, 0.3), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.3, 0.3), platform_width=2.0, border_width=0.25
        # ),

    },
)
# 给Go2高程图训练用
ROUGH_ELEVATION_CFG_GO2 = TerrainGeneratorCfg(
    size=(10.0, 10.0), # 每个地形的大小
    border_width=30.0, # border是在地形区域外扩充一圈平地，设置平地的宽度
    border_height = 1.0, # border的高度，负数是向上砌墙，正数改了没有反应
    num_rows=10, # 列数，一共多少级
    num_cols=20, # 行数，一共多少个赛道
    horizontal_scale=0.1, # 地形水平分辨率
    vertical_scale=0.005, # 地形高度分辨率
    slope_threshold=0.75, # tan超过此值的斜坡会变成墙
    use_cache=False, # 不能开，可能导致地形修改不成功
    curriculum=True,
    sub_terrains={
        # 上楼梯
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # 下楼梯
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        # 同心沟壑地形 (concentric moats)
        "concentric_moats": terrain_gen_nhb.MeshConcentricMoatsTerrainCfg(
            proportion=0.2,
            platform_width=3.0,
            moat_width_range=(0.0, 0.4),
            moat_depth_range=(0.0, 1.5),
            num_moats=3,
            platform_ring_width=0.7,
        ),
        # 平地
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.1,
        ),
    },
)