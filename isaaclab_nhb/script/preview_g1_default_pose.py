"""Preview the fixed-base default pose of the 29-DoF G1 asset.

Run from the package directory:

    python script/preview_g1_default_pose.py --device cuda:7
"""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Preview the static default G1 pose.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from isaaclab_nhb.tasks.humanoid.G1.G1_asset_cfg import G1_29DOF_CFG


def design_scene() -> Articulation:
    """Spawn a fixed-base G1 in its configured default pose."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = G1_29DOF_CFG.replace(prim_path="/World/G1")
    robot_cfg.spawn.articulation_props.fix_root_link = True
    return Articulation(robot_cfg)


def main():
    """Run the static GUI preview."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    sim.set_camera_view(eye=[1.8, -2.0, 1.4], target=[0.1, 0.0, 0.75])

    robot = design_scene()
    sim.reset()

    sim_dt = sim.get_physics_dt()
    robot.update(sim_dt)
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)

    print("[INFO] Previewing fixed-base G1 default pose from G1_29DOF_CFG.init_state.")
    print("[INFO] Default pose is the reference used by joint_deviation_l1 rewards.")
    print("[INFO] Close the Isaac Sim window or press Ctrl+C to stop.")

    while simulation_app.is_running():
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
