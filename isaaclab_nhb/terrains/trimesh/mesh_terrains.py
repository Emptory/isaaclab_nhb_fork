# Copyright (c) 2024, isaaclab_nhb Project Developers.
# All rights reserved.

"""Functions to generate custom terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import trimesh
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def concentric_moats_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshConcentricMoatsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with concentric moats around a central platform.
    
    The terrain structure from center outward:
    1. Central square platform
    2. Moat 1
    3. Platform ring
    4. Moat 2
    5. Platform ring
    6. ... (repeats based on num_moats)
    
    Uses boolean difference operations to create moats from a base platform.
    
    .. note::
        The moat width, depth, and number of moats are configurable.
    
    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.
    
    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration based on difficulty
    moat_depth = cfg.moat_depth_range[0] + difficulty * (cfg.moat_depth_range[1] - cfg.moat_depth_range[0])
    moat_width = cfg.moat_width_range[0] + difficulty * (cfg.moat_width_range[1] - cfg.moat_width_range[0])
    
    # compute the position of the terrain center
    terrain_center = [cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0]
    
    # Calculate available space for moats and platforms
    max_radius = min(cfg.size[0], cfg.size[1]) / 2.0
    
    # Calculate the required radius starting from center platform
    # Center platform is a square with side length cfg.platform_width
    # Use half the side length as the "radius" for box calculations
    center_platform_half = cfg.platform_width / 2.0
    
    # Required radius = center + moats + platform rings between moats
    # Structure from center: platform -> moat1 -> ring1 -> moat2 -> ring2 -> ...
    required_radius = center_platform_half + cfg.num_moats * moat_width + (cfg.num_moats - 1) * cfg.platform_ring_width
    
    # Make sure we don't exceed the terrain size
    if required_radius > max_radius:
        raise ValueError(
            f"Terrain configuration exceeds available space. "
            f"Required radius: {required_radius:.2f}m, "
            f"Available radius: {max_radius:.2f}m. "
            f"Please reduce moat_width, num_moats, or platform_width."
        )
    
    # Create the base platform (entire terrain)
    # Make it thicker than moat_depth so there's material below the moats
    # Top surface at z=0, bottom deeper than the moats will dig
    platform_base_thickness = 0.5  # Extra thickness below moats
    platform_total_height = moat_depth + platform_base_thickness
    result_platform = trimesh.creation.box(
        (cfg.size[0], cfg.size[1], platform_total_height),
        trimesh.transformations.translation_matrix([terrain_center[0], terrain_center[1], -platform_total_height / 2.0])
    )
    
    # Dig moats from center outward to maintain fixed center platform size
    # Structure from center: center_platform -> moat1 -> ring1 -> moat2 -> ring2 -> ...
    current_inner_radius = center_platform_half  # Start from the edge of center platform
    
    for i in range(cfg.num_moats):
        # Calculate the i-th moat from center (i=0 is closest to center)
        moat_inner_radius = current_inner_radius
        moat_outer_radius = current_inner_radius + moat_width
        
        # Create the moat ring by subtracting inner box from outer box
        # The moat depth is from z=0 (top surface) down to z=-moat_depth
        # We add small epsilon to ensure clean boolean operations
        outer_box = trimesh.creation.box(
            (2 * moat_outer_radius, 2 * moat_outer_radius, moat_depth + 0.01),
            trimesh.transformations.translation_matrix([terrain_center[0], terrain_center[1], -moat_depth / 2.0])
        )
        
        inner_box = trimesh.creation.box(
            (2 * moat_inner_radius, 2 * moat_inner_radius, moat_depth + 0.02),
            trimesh.transformations.translation_matrix([terrain_center[0], terrain_center[1], -moat_depth / 2.0])
        )
        
        # Create moat ring by subtracting inner from outer (this is the material to remove)
        moat_ring = outer_box.difference(inner_box)
        
        # Dig the moat by subtracting the ring from the platform
        result_platform = result_platform.difference(moat_ring)
        
        # Move to next layer: add moat width + platform ring width to get to next moat's inner radius
        current_inner_radius = moat_outer_radius + cfg.platform_ring_width
    
    # Origin of the terrain (spawn point on the central platform at height 0)
    origin = np.array([terrain_center[0], terrain_center[1], 0.0])
    
    return [result_platform], origin
