# Copyright (c) 2024, isaaclab_nhb Project Developers.
# All rights reserved.

"""Configuration classes for custom trimesh terrains."""

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg

import isaaclab_nhb.terrains.trimesh.mesh_terrains as mesh_terrains


@configclass
class MeshConcentricMoatsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with concentric moats around a central platform.
    
    This terrain creates a challenging environment with:
    - A central square platform
    - Alternating moats and platform rings
    
    The moat width, depth, and number of moats are configurable.
    """

    function = mesh_terrains.concentric_moats_terrain

    platform_width: float = 2.0
    """The width of the central square platform (in m). Defaults to 2.0."""

    moat_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of each moat (in m).
    
    The actual width is interpolated based on the difficulty parameter.
    """

    moat_depth_range: tuple[float, float] = MISSING
    """The minimum and maximum depth of the moats (in m).
    
    The actual depth is interpolated based on the difficulty parameter.
    """

    num_moats: int = 1
    """The number of moats around the central platform. Defaults to 1."""

    platform_ring_width: float = MISSING
    """The width of each platform ring between moats (in m)."""
