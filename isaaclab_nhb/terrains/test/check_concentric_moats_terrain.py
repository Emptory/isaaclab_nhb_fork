# Copyright (c) 2024, isaaclab_nhb Project Developers.
# All rights reserved.

"""
Test script for visualizing a single concentric moats terrain configuration.

This script allows you to easily modify terrain parameters and observe the results.

Example usage:

.. code-block:: bash

    # Test with default parameters
    python isaaclab_nhb/terrains/test/check_concentric_moats_terrain.py
    
    # Test in headless mode (no visualization window)
    python isaaclab_nhb/terrains/test/check_concentric_moats_terrain.py --headless

"""

"""Launch Isaac Sim Simulator first."""

import argparse

parser = argparse.ArgumentParser(description="Test concentric moats terrain using trimesh")
parser.add_argument(
    "--headless", action="store_true", default=False, help="Don't create a window to display each output."
)
args_cli = parser.parse_args()

from isaaclab.app import AppLauncher

# launch omniverse app
# note: we only need to do this because of `TerrainImporter` which uses Omniverse functions
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import os
import trimesh

import isaaclab_nhb.terrains.trimesh as terrain_gen_nhb
from isaaclab.terrains.utils import color_meshes_by_height


# ============================================================================
# TERRAIN CONFIGURATION - Modify these parameters to test different terrains
# ============================================================================

# Terrain size
TERRAIN_SIZE = (10.0, 10.0)  # Width x Length (in meters)

# Central platform
PLATFORM_WIDTH = 2.0  # Width of the central square platform (in meters)

# Moat parameters
MOAT_WIDTH_RANGE = (0.2, 0.6)  # Min and max width of each moat (in meters)
MOAT_DEPTH_RANGE = (0.3, 0.6)  # Min and max depth (in meters)
NUM_MOATS = 3  # Number of concentric moats around the central platform
PLATFORM_RING_WIDTH = 0.8  # Width of each platform ring between moats (in meters)

# Difficulty level (0.0 = easiest, 1.0 = hardest)
DIFFICULTY = 0.5  # Controls the actual moat depth within MOAT_DEPTH_RANGE

# ============================================================================
# ============================================================================


def test_concentric_moats_terrain():
    """Test and visualize a single concentric moats terrain configuration."""
    print("\n" + "=" * 80)
    print("CONCENTRIC MOATS TERRAIN TEST")
    print("=" * 80)
    
    # Create terrain configuration from the parameters defined above
    cfg = terrain_gen_nhb.MeshConcentricMoatsTerrainCfg(
        size=TERRAIN_SIZE,
        platform_width=PLATFORM_WIDTH,
        moat_width_range=MOAT_WIDTH_RANGE,
        moat_depth_range=MOAT_DEPTH_RANGE,
        num_moats=NUM_MOATS,
        platform_ring_width=PLATFORM_RING_WIDTH,
    )
    
    # Print configuration
    print("\nTerrain Configuration:")
    print(f"  Size: {TERRAIN_SIZE[0]}m x {TERRAIN_SIZE[1]}m")
    print(f"  Central Platform Width: {PLATFORM_WIDTH}m")
    print(f"  Moat Width Range: {MOAT_WIDTH_RANGE[0]}m - {MOAT_WIDTH_RANGE[1]}m")
    print(f"  Platform Ring Width: {PLATFORM_RING_WIDTH}m")
    print(f"  Moat Depth Range: {MOAT_DEPTH_RANGE[0]}m - {MOAT_DEPTH_RANGE[1]}m")
    print(f"  Number of Moats: {NUM_MOATS}")
    print(f"  Difficulty: {DIFFICULTY}")
    
    # Calculate actual moat depth based on difficulty
    actual_depth = MOAT_DEPTH_RANGE[0] + DIFFICULTY * (MOAT_DEPTH_RANGE[1] - MOAT_DEPTH_RANGE[0])
    actual_width = MOAT_WIDTH_RANGE[0] + DIFFICULTY * (MOAT_WIDTH_RANGE[1] - MOAT_WIDTH_RANGE[0])
    print(f"  Actual Moat Depth (at difficulty {DIFFICULTY}): {actual_depth:.2f}m")
    print(f"  Actual Moat Width (at difficulty {DIFFICULTY}): {actual_width:.2f}m")
    
    # Generate the terrain
    print("\nGenerating terrain...")
    try:
        meshes, origin = cfg.function(difficulty=DIFFICULTY, cfg=cfg)
        print(f"✓ Successfully generated {len(meshes)} mesh components")
        print(f"✓ Terrain origin: ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})")
    except Exception as e:
        print(f"✗ Failed to generate terrain: {e}")
        raise
    
    # Add colors to the meshes based on height
    colored_mesh = color_meshes_by_height(meshes)
    
    # Add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.3, transform=origin_transform)
    
    # Create the scene
    scene = trimesh.Scene([colored_mesh, origin_marker])
    
    # Show the scene in a window (if not headless)
    if not args_cli.headless:
        print("\n" + "=" * 80)
        print("Opening visualization window...")
        print("TIP: Modify the parameters at the top of this script to test different configurations")
        print("=" * 80 + "\n")
        try:
            scene.show(caption=f"Concentric Moats Terrain (Difficulty: {DIFFICULTY})")
        except Exception as e:
            print(f"Warning: Could not open visualization window: {e}")
            print("Visualization requires a display environment.")
    else:
        print("\n✓ Test completed successfully (headless mode - no visualization)")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80 + "\n")


def main():
    """Main function to run all tests."""
    print("\n" + "=" * 80)
    print("CONCENTRIC MOATS TERRAIN TEST SUITE")
    print("=" * 80)
    
    # Run the test
    test_concentric_moats_terrain()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
