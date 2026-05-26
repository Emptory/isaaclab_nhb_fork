# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

# 设置 CUDA 设备（必须在导入 CUDA 相关库之前设置）
import os

import argparse
import sys

from isaaclab.app import AppLauncher
import isaaclab_nhb 

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps).")
parser.add_argument("--csv_length", type=int, default=200, help="Length of CSV data recording (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--show_s1_hand_targets",
    action="store_true",
    default=False,
    help="Visualize CoopG1S1 virtual hand target positions in the torso_link frame.",
)

parser.add_argument("--csv_export", action="store_true", default=False, help="Export data to CSV.")
parser.add_argument("--amp_stats", action="store_true", default=True, help="Enable AMP data statistics collection and output.")
parser.add_argument("--stats_interval", type=int, default=100, help="Interval (in steps) for AMP statistics output.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()




# 测试奖励用。不要在这里覆盖命令行参数；按 CLI 传入 task/checkpoint。
# args_cli.checkpoint = "/home/jyz/project/isaaclab_nhb_ws/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/G1_ElevHist_ECMM_rough/2026-04-15_15-03-13_new_edge_pal_test2/model_24000.pt"
# args_cli.task = "G1-ElevHist-ECMM-Rough"

# ours
# args_cli.checkpoint = "/home/jyz/project/isaaclab_nhb_ws/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/g1_mode13A5_rough/2026-02-05_11-39-36_mode13A5-NoAMP-colli500-edge5-Nostill/model_28000.pt"
# args_cli.task = "G1-Elevation-Net-Mode13A5-AMP-Rough"

# E1 光头
# args_cli.checkpoint = "/home/jyz/project/isaaclab_nhb_ws/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/g1_E1/2026-02-05_14-00-07_E1-history-elevation/model_4000.pt"
# args_cli.task = "G1-terrain-E1"

# E2 不使用Rew的CNN
# args_cli.checkpoint = "/home/jyz/project/isaaclab_nhb_ws/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/g1_mode13A5_rough/2026-02-10_12-11-17_mode13A5-NobigAMP-E2-NoNewRew/model_28000.pt"
# args_cli.task = "G1-Elevation-Net-Mode13A5-AMP-Rough"

# BeyondMimic
# args_cli.checkpoint = "/home/andew/RL/RL_robot_ws/isaaclab_lyj/isaaclab_nhb/logs/rsl_rl/beyond_mimic_g1/2025-12-16_14-17-52_default/model_2000.pt"
# args_cli.task = "BeyondMimic-G1-Flat"

# 使用WebRTC串流
# args_cli.livestream=2

args_cli.device = "cuda:0"
# args_cli.headless = True
# args_cli.real_time = True
# args_cli.video = True
# args_cli.video_length = 710
# args_cli.csv_export = True
# args_cli.csv_length = 700


# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

if args_cli.headless:
    isaaclab_nhb.HEADLESS_FLAG = True 

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

# Import for getting camera position
import isaacsim.core.utils.stage as stage_utils
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_apply
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    try:
        from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    except ModuleNotFoundError:
        get_published_pretrained_checkpoint = None

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit
from rsl_rl.utils.vecenv_wrapper import RslRlVecEnvWrapperDictAction

from rsl_rl.utils.exporter import export_policy_as_onnx
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_nhb import * # 识别isaaclab_nhb库的内容
# from rsl_rl.utils.exporter import EstNetOnnxPolicyExporter, DWAQOnnxPolicyExporter  
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
# PLACEHOLDER: Extension template (do not remove this comment)

from isaaclab_nhb.script.rsl_rl.data_logger import DataLogger


def get_camera_position():
    """Get the current camera position from the USD stage.

    Returns:
        tuple: (x, y, z) camera position or None if not available
    """
    try:
        stage = stage_utils.get_current_stage()
        if stage is not None:
            # Get the viewport camera prim
            camera_prim_path = "/OmniverseKit_Persp"
            camera_prim = stage.GetPrimAtPath(camera_prim_path)

            if camera_prim and camera_prim.IsValid():
                # Get the camera's world transform
                camera_xform = UsdGeom.Xformable(camera_prim)
                world_transform = camera_xform.ComputeLocalToWorldTransform(0)  # 0 = current time

                # Extract position from the transform matrix
                camera_pos = world_transform.ExtractTranslation()
                return (camera_pos[0], camera_pos[1], camera_pos[2])
        return None
    except Exception as e:
        print(f"[ERROR]: Failed to get camera position: {e}")
        return None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # Honor CLI-provided overrides and defaults; avoid hard-coded debug presets
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if hasattr(env_cfg, "rebuild_dynamic_cfg"):
        env_cfg.rebuild_dynamic_cfg()

    # env_cfg.scene.env_spacing = 20.0
    
    
    # env_cfg.commands.base_velocity.ranges.lin_vel_x = [-0.7, -0.7]
    # env_cfg.commands.base_velocity.ranges.lin_vel_y = [0.0, 0.0]
    # env_cfg.commands.base_velocity.ranges.lin_vel_x = [0.0, 0.0]
    # env_cfg.commands.base_velocity.ranges.lin_vel_y = [0.7, 0.7] # y正向左
    # env_cfg.commands.base_velocity.ranges.ang_vel_z = [-1.0, 1.0]
    # env_cfg.commands.base_velocity.ranges.ang_vel_z = [-1.5, 1.5] # 给0则yawKp失效
    # env_cfg.commands.base_velocity.ranges.heading = [1.57, 1.57]
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.3)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    if hasattr(env_cfg.events, "reset_base"):
        env_cfg.events.reset_base.params["pose_range"]["yaw"] = (1.57,1.57)
        env_cfg.events.reset_base.params["pose_range"]["x"] = (0,0)
        env_cfg.events.reset_base.params["pose_range"]["y"] = (0,0)
    # env_cfg.viewer.eye = (2.0, 0.0, 2.0)
    # env_cfg.viewer.eye = (-1.0, 2.0, 2.0)
    # env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    # env_cfg.viewer.resolution = (1920,1080)
    # env_cfg.viewer.resolution = (1280,720)
    # env_cfg.viewer.origin_type = "asset_root"
    # env_cfg.viewer.asset_name = "robot"
    # env_cfg.viewer.env_index = 0

    # env_cfg.commands.base_velocity.debug_vis=False
    # env_cfg.commands.base_velocity.heading_control_stiffness = 1.0

    # 去除随机化
    # env_cfg.terminations = None
    # env_cfg.events = None
    env_cfg.curriculum = None    

    # set the environment seed
    # note: certain randomizations occur in environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    # Prefer project-local logs under isaaclab_nhb/isaaclab_nhb/logs, then fallback to repo root logs
    script_dir = os.path.dirname(__file__)
    project_logs = os.path.abspath(os.path.join(script_dir, "..", "..", "logs", "rsl_rl", agent_cfg.experiment_name))
    repo_logs = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    # pick first existing; if neither exists, default to project_logs
    if os.path.isdir(project_logs):
        log_root_path = project_logs
    elif os.path.isdir(repo_logs):
        log_root_path = repo_logs
    else:
        log_root_path = project_logs
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        if get_published_pretrained_checkpoint is None:
            print("[INFO] Pre-trained checkpoint lookup is unavailable in this IsaacLab installation.")
            return
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        # find checkpoint under the chosen log root
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapperDictAction(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    
    # 使用日志文件夹名称作为onnx文件名,并添加模型步数
    log_folder_name = os.path.basename(log_dir)
    # 从checkpoint文件名中提取步数 (例如: model_18000.pt -> 18000)
    checkpoint_filename = os.path.basename(resume_path)
    model_step = checkpoint_filename.replace("model_", "").replace(".pt", "")
    onnx_filename = f"{log_folder_name}_step{model_step}.onnx"
    
    # export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    try:
        export_policy_as_onnx(
            policy_nn,
            policy_type=agent_cfg.policy.class_name,
            normalizer=normalizer,
            path=export_model_dir,
            filename=onnx_filename,
        )
    except Exception as err:
        print(f"[WARNING] Skipping ONNX export during play: {err}")

    # 将obs和action到出为CSV
    if args_cli.csv_export:
        export_csv_dir = os.path.join(os.path.dirname(resume_path), "exported/csv")
        # 排除不需要记录的观测组：高程图和AMP观测
        datalogger = DataLogger(
            export_csv_dir, 
            env, 
            log_height_scan=False,
            exclude_obs_groups=['height_scan_policy', 'height_scan_critic', 'amp']
        )


    dt = env.unwrapped.step_dt

    s1_hand_target_viz = None
    if args_cli.show_s1_hand_targets:
        left_target_viz = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/S1LeftHandTarget",
                markers={
                    "marker": sim_utils.SphereCfg(
                        radius=0.06,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    )
                },
            )
        )
        right_target_viz = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/S1RightHandTarget",
                markers={
                    "marker": sim_utils.SphereCfg(
                        radius=0.06,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.2, 1.0)),
                    )
                },
            )
        )
        s1_hand_target_viz = {
            "left_viz": left_target_viz,
            "right_viz": right_target_viz,
            "left_local": torch.tensor([[0.44, 0.24, 0.03]], device=env.unwrapped.device),
            "right_local": torch.tensor([[0.44, -0.24, 0.03]], device=env.unwrapped.device),
        }
        print("[INFO] Showing S1 hand target markers: red=left target, blue=right target.")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    
    # determine if we need to track timesteps for limiting recording
    should_limit_timesteps = args_cli.video or args_cli.csv_export
    max_timesteps = None

    if should_limit_timesteps:
        if args_cli.video and args_cli.csv_export:
            # if both are enabled, use the minimum
            max_timesteps = min(args_cli.video_length, args_cli.csv_length)
        elif args_cli.video:
            max_timesteps = args_cli.video_length
        elif args_cli.csv_export:
            max_timesteps = args_cli.csv_length

        print(f"[INFO] Recording limited to {max_timesteps} timesteps "
              f"(video: {args_cli.video_length if args_cli.video else 'disabled'}, "
              f"csv: {args_cli.csv_length if args_cli.csv_export else 'disabled'})")

    # Add a flag to skip camera position check in first few steps
    initial_steps = 5
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, extra_info = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions, extra_info)

            if s1_hand_target_viz is not None:
                robot = env.unwrapped.scene["robot"]
                torso_id = robot.body_names.index("torso_link")
                torso_pos_w = robot.data.body_link_pos_w[:, torso_id, :]
                torso_quat_w = robot.data.body_link_quat_w[:, torso_id, :]

                left_target_w = torso_pos_w + quat_apply(
                    torso_quat_w,
                    s1_hand_target_viz["left_local"].expand(torso_pos_w.shape[0], -1),
                )
                right_target_w = torso_pos_w + quat_apply(
                    torso_quat_w,
                    s1_hand_target_viz["right_local"].expand(torso_pos_w.shape[0], -1),
                )
                s1_hand_target_viz["left_viz"].visualize(left_target_w)
                s1_hand_target_viz["right_viz"].visualize(right_target_w)
        
        # Get and print camera position (skip first few steps to ensure physics is initialized)
        if timestep >= initial_steps:
            try:
                camera_pos = get_camera_position()
                if camera_pos is not None:
                    print(f"[Camera] Position: x={camera_pos[0]:.3f}, y={camera_pos[1]:.3f}, z={camera_pos[2]:.3f}")
            except Exception as e:
                # Silently skip if camera position retrieval fails
                pass
        
        # record CSV data
        if args_cli.csv_export:
            datalogger.log(obs, actions)

        # increment timestep and check if we should stop
        if should_limit_timesteps:
            timestep += 1
            if timestep >= max_timesteps:
                print(f"[INFO] Reached recording limit of {max_timesteps} timesteps, stopping...")
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if args_cli.csv_export:
        datalogger.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
