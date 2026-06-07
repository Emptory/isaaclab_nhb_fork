"""Preview the fixed-base G1 pose used by CoopG1S1 rewards.

Run from the package directory:

    python script/preview_s1_hold_pose.py --device cuda:7
"""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Preview the static CoopG1S1 target pose.")
parser.add_argument("--hide_box", action="store_true", help="Hide the virtual hold-box preview.")
parser.add_argument(
    "--print_hand_targets",
    action="store_true",
    help="Print current hand positions in torso_link frame and exit.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_nhb.tasks.humanoid.G1.G1_asset_cfg import G1_29DOF_CFG
from isaaclab_nhb.tasks.humanoid.coopG1S1.coopG1S1_env_cfg import (
    HOLD_ARM_JOINT_POS,
    HOLD_BOX_REL_POS,
    HOLD_BOX_SIZE,
    HOLD_HAND_TARGET_POS,
)


def design_scene() -> tuple[Articulation, VisualizationMarkers, VisualizationMarkers | None]:
    """Spawn a fixed-base G1 and non-physical reward-target markers."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    joint_pos = dict(G1_29DOF_CFG.init_state.joint_pos)
    joint_pos.pop(".*_elbow_joint", None)
    joint_pos.update(HOLD_ARM_JOINT_POS)
    robot_cfg = G1_29DOF_CFG.replace(
        prim_path="/World/G1",
        init_state=G1_29DOF_CFG.init_state.replace(joint_pos=joint_pos),
    )
    robot_cfg.spawn.articulation_props.fix_root_link = True
    robot = Articulation(robot_cfg)

    hand_targets = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/S1HandTargets",
            markers={
                "left": sim_utils.SphereCfg(
                    radius=0.045,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "right": sim_utils.SphereCfg(
                    radius=0.045,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.2, 1.0)),
                ),
            },
        )
    )

    box_preview = None
    if not args_cli.hide_box:
        box_preview = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/S1HoldBox",
                markers={
                    "box": sim_utils.CuboidCfg(
                        size=HOLD_BOX_SIZE,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.9, 0.35, 0.05),
                            opacity=0.35,
                        ),
                    )
                },
            )
        )

    return robot, hand_targets, box_preview


def desired_joint_state(robot: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the full G1 target state while preserving the default leg pose."""
    joint_pos = robot.data.default_joint_pos.clone()
    for joint_name, target in HOLD_ARM_JOINT_POS.items():
        joint_pos[:, robot.joint_names.index(joint_name)] = target
    return joint_pos, torch.zeros_like(joint_pos)


def visualize_targets(
    robot: Articulation,
    hand_targets: VisualizationMarkers,
    box_preview: VisualizationMarkers | None,
):
    """Draw reward targets in the torso frame."""
    torso_id = robot.body_names.index("torso_link")
    torso_pos_w = robot.data.body_link_pos_w[:, torso_id, :]
    torso_quat_w = robot.data.body_link_quat_w[:, torso_id, :]

    hand_targets_b = torch.tensor(HOLD_HAND_TARGET_POS, dtype=torch.float32, device=robot.device)
    hand_targets_w = torso_pos_w.expand(2, -1) + quat_apply(torso_quat_w.expand(2, -1), hand_targets_b)
    hand_targets.visualize(hand_targets_w, marker_indices=[0, 1])

    if box_preview is not None:
        box_pos_b = torch.tensor([HOLD_BOX_REL_POS], dtype=torch.float32, device=robot.device)
        box_pos_w = torso_pos_w + quat_apply(torso_quat_w, box_pos_b)
        box_preview.visualize(box_pos_w, orientations=torso_quat_w)


def print_current_hand_targets(robot: Articulation):
    """Print current left/right hand positions in the torso frame."""
    torso_id = robot.body_names.index("torso_link")
    hand_ids = [robot.body_names.index("left_rubber_hand"), robot.body_names.index("right_rubber_hand")]

    torso_pos_w = robot.data.body_link_pos_w[:, torso_id, :]
    torso_quat_w = robot.data.body_link_quat_w[:, torso_id, :]
    hand_pos_w = robot.data.body_link_pos_w[:, hand_ids, :]
    hand_rel_w = hand_pos_w - torso_pos_w.unsqueeze(1)
    hand_rel_b = quat_apply_inverse(
        torso_quat_w.unsqueeze(1).expand(-1, len(hand_ids), -1).reshape(-1, 4),
        hand_rel_w.reshape(-1, 3),
    ).reshape(1, len(hand_ids), 3)

    left = hand_rel_b[0, 0].detach().cpu().tolist()
    right = hand_rel_b[0, 1].detach().cpu().tolist()
    print("[INFO] Current hand positions in torso_link frame:")
    print(f"    left_rubber_hand:  ({left[0]:.6f}, {left[1]:.6f}, {left[2]:.6f})")
    print(f"    right_rubber_hand: ({right[0]:.6f}, {right[1]:.6f}, {right[2]:.6f})")


def main():
    """Run the static GUI preview."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    sim.set_camera_view(eye=[1.8, -2.0, 1.5], target=[0.2, 0.0, 0.85])

    robot, hand_targets, box_preview = design_scene()
    sim.reset()

    sim_dt = sim.get_physics_dt()
    robot.update(sim_dt)
    joint_pos, joint_vel = desired_joint_state(robot)

    print("[INFO] Previewing the fixed-base CoopG1S1 target pose.")
    print("[INFO] Marker legend: red=left hand, blue=right hand, orange=virtual hold box.")
    print("[INFO] Close the Isaac Sim window or press Ctrl+C to stop.")

    if args_cli.print_hand_targets:
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        print_current_hand_targets(robot)
        return

    while simulation_app.is_running():
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        visualize_targets(robot, hand_targets, box_preview)


if __name__ == "__main__":
    main()
    simulation_app.close()
