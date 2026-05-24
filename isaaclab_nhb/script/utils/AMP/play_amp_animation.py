# file: play_amp_animation.py (refactored)
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.

"""一个使用 Isaac Lab 标准��作流来可视化和导出 AMP 动画的脚本。"""

import argparse
import os
import sys
import numpy as np
import gymnasium as gym
from isaaclab_nhb.dataset.amp_data_cfg import G1AmpDataCfg
from rsl_rl.utils import AMPLoaderDisplay

# --- 仅导入 AppLauncher（可安全在脚本顶部导入） ---
from isaaclab.app import AppLauncher


def main():
    """主函数，负责解析参数、创建环境并运行动画循环。"""
    try:
        # --- 1. 参数解析 (模仿 train.py) ---
        print("\n" + "="*60)
        print("🎬 AMP 动画可视化播放器")
        print("="*60)
        print("正在加载配置...")

        try:
            DataCfg = G1AmpDataCfg()
            print(f"✅ 配置加载成功")
            print(f"配置目录: {DataCfg.cfg_dir}")
            print(f"可视化文件夹: {DataCfg.visualization_path}")
        except Exception as e:
            print(f"❌ 配置加载失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # 获取所有可视化文件
        try:
            visualization_files = DataCfg.get_visualization_files()
            if not visualization_files:
                print("\n❌ 错误: 在可视化文件夹中未找到任何 .txt 文件。")
                print(f"文件夹路径: {DataCfg.visualization_path}")
                print(f"请检查该目录是否存在以及是否包含 .txt 文件")
                return
        except Exception as e:
            print(f"❌ 获取可视化文件失败: {e}")
            import traceback
            traceback.print_exc()
            return

        print(f"\n{'='*60}")
        print(f"📂 找到 {len(visualization_files)} 个可视化文件:")
        print(f"{'='*60}")
        for idx, file_path in enumerate(visualization_files, 1):
            file_name = os.path.basename(file_path)
            print(f"  {idx}. {file_name}")
        print(f"{'='*60}")

        # === 用户选择播放模式 ===
        print("\n🎮 请选择播放模式:")
        print("  1 - 循环播放所有数据集")
        print("  2 - 手动选择播放")
        print("  (直接按 Enter 默认使用模式 1)")

        mode_choice = '1'  # 默认使用循环播放模式

        try:
            user_input = input("\n请输入选项 (1 或 2，直接回车使用默认): ").strip()
            if user_input in ['1', '2']:
                mode_choice = user_input
            elif user_input == '':
                print("✅ 使用默认模式：循环播放所有数据集")
            else:
                print(f"⚠️  无效选项 '{user_input}'，使用默认模式：循环播放所有数据集")
        except (EOFError, KeyboardInterrupt):
            print("\n✅ 使用默认模式：循环播放所有数据集")
        except Exception as e:
            print(f"⚠️  输入异常: {e}，使用默认模式：循环播放所有数据集")

        auto_loop = (mode_choice == '1')

        if auto_loop:
            print("\n🔄 已选择：循环播放模式")
        else:
            print("\n🎯 已选择：手动选择模式")

        print("\n⏳ 正在初始化 Isaac Sim 环境...")
        print("="*60 + "\n")

        parser = argparse.ArgumentParser(description="Visualize and export AMP animation using Isaac Lab.")
        # 添加此脚本特有的参数
        parser.add_argument("--task", type=str, help="Name of the registered environment task.")
        parser.add_argument("--save_path", type=str,
                            default=DataCfg.save_motion_expert_path, help="Path to save the exported motion file.")
        parser.add_argument("--fps", type=float, default=30.0, help="Target FPS for the exported motion.")
        parser.add_argument("--num_envs", type=int, default=1,
                            help="Number of environments to simulate (should be 1 for viz).")
        # 添加标准的 AppLauncher 参数
        AppLauncher.add_app_launcher_args(parser)
        args_cli, hydra_args = parser.parse_known_args()

        # 清理 sys.argv 以便 Hydra 接管
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.task = getattr(args_cli, "task", "G1-AMP-Walk-Flat") or "G1-AMP-Walk-Flat"

        # --- 2. 创建并启动 Isaac Sim 内核（必须 launch() 以加载 omni.* 模块） ---
        app_launcher = AppLauncher(args_cli)
        # call launch() to bootstrap the Omniverse/IsaacSim kernel
        simulation_app = app_launcher.app

        # ---------- IMPORTANT: Delay imports that require IsaacSim kernel ----------
        # 只有在 launch() 完成后再导入这些会触发 omni.physics 等的模块
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.hydra import hydra_task_config
        from isaaclab.envs import ManagerBasedRLEnvCfg

        # 你的自定义 env 导入也必须延迟，保证 Omniverse 运行时已准备好
        from isaaclab_nhb.envs.AMP_manager_based_rl_env import AMPManagerBasedEnv

        # --- 3. 使用 Hydra 加载配置并运行核心逻辑 ---
        @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
        def run_animation(env_cfg: ManagerBasedRLEnvCfg, _):
            """使用加载的配置来创建环境并运行动画。"""

            # --- 4. 修改配置以适应可视化需求 (与旧脚本逻辑相同) ---
            env_cfg.scene.num_envs = args_cli.num_envs
            env_cfg.scene.env_spacing = 2.5
            # 设置为平坦地面
            if hasattr(env_cfg.scene, "terrain"):
                env_cfg.scene.terrain.terrain_type = "plane"
                env_cfg.scene.terrain.terrain_generator = None
            # 禁用随机化事件
            if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"):
                env_cfg.events.push_robot = None
            # 禁用观测噪声
            if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
                env_cfg.observations.policy.enable_corruption = False
            # 禁用指令的UI调试窗口
            if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
                env_cfg.commands.base_velocity.debug_vis = False

            # --- 4.2 在创建环境前初始化第一个动画文件 ---
            # 这样可以防止 AMPLoaderDisplay 收到空列表
            if len(visualization_files) > 0:
                env_cfg.amp_motion_files_display = [visualization_files[0]]
                print(f"🎬 初始化第一个动画文件: {os.path.basename(visualization_files[0])}")
            else:
                print("❌ 错误: 没有找到可视化文件，无法创建环境")
                return

            # --- 4.3 启用动画播放模式 ---
            env_cfg.play_amp_animation_mode = True
            print("✅ 已启用动画播放模式")

            # --- 5. 标准化创建环境 (与 train.py 相同) ---
            print("\n✅ Isaac Sim 环境已启动，正在创建场景...")
            env = gym.make(args_cli.task, cfg=env_cfg)
            AMPwrapped_env: AMPManagerBasedEnv = env.unwrapped
            print("✅ 场景创建完成，开始播放动画...\n")

            # --- 6. 根据模式播放文件 ---
            if auto_loop:
                # 模式1：循环播放所有数据集
                run_auto_loop_mode(AMPwrapped_env, visualization_files, args_cli, DataCfg, simulation_app)
            else:
                # 模式2：手动选择模式
                run_manual_select_mode(AMPwrapped_env, visualization_files, args_cli, DataCfg, simulation_app)

            # 关闭环境
            env.close()

        # 调用被hydra装饰的函数
        run_animation()

        # 关闭模拟器（确保在所有任务完成后调用）
        simulation_app.close()

    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_auto_loop_mode(AMPwrapped_env, visualization_files, args_cli, DataCfg, simulation_app):
    """循环播放模式：自动依次播放所有数据集"""
    current_file_idx = 0

    while simulation_app.is_running():
        # 获取当前文件
        current_file = visualization_files[current_file_idx]
        file_name = os.path.basename(current_file)

        print(f"\n{'='*60}")
        print(f"🎬 正在播放 [{current_file_idx + 1}/{len(visualization_files)}]: {file_name}")
        print(f"{'='*60}\n")

        # 播放当前文件
        play_single_file(AMPwrapped_env, current_file, args_cli, DataCfg, simulation_app)

        # 移动到下一个文件
        current_file_idx += 1
        if current_file_idx >= len(visualization_files):
            print(f"\n{'='*60}")
            print("✅ 所有文件播放完成！重新开始循环...")
            print(f"{'='*60}\n")
            current_file_idx = 0


def run_manual_select_mode(AMPwrapped_env, visualization_files, args_cli, DataCfg, simulation_app):
    """手动选择模式：用户选择要播放的数据集"""
    while simulation_app.is_running():
        # 显示文件列表
        print(f"\n{'='*60}")
        print("📂 可用的数据集:")
        print(f"{'='*60}")
        for idx, file_path in enumerate(visualization_files, 1):
            file_name = os.path.basename(file_path)
            print(f"  {idx}. {file_name}")
        print(f"  0. 退出程序")
        print(f"{'='*60}\n")

        # 获取用户选择
        try:
            choice = input("请选择要播放的数据集编号 (0 退出): ").strip()

            if not choice:
                continue

            choice_idx = int(choice)

            if choice_idx == 0:
                print("\n👋 退出播放器...")
                break
            elif 1 <= choice_idx <= len(visualization_files):
                selected_file = visualization_files[choice_idx - 1]
                file_name = os.path.basename(selected_file)

                print(f"\n{'='*60}")
                print(f"🎬 正在播放: {file_name}")
                print(f"{'='*60}\n")

                # 播放选中的文件
                play_single_file(AMPwrapped_env, selected_file, args_cli, DataCfg, simulation_app)

                print(f"\n{'='*60}")
                print(f"✅ 播放完成: {file_name}")
                print(f"{'='*60}\n")
            else:
                print(f"❌ 无效选项，请输入 0-{len(visualization_files)} 之间的数字")

        except ValueError:
            print("❌ 请输入有效的数字")
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 检测到中断，退出播放器...")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            break


def play_single_file(AMPwrapped_env, file_path, args_cli, DataCfg, simulation_app):
    """播放单个数据集文件"""
    file_name = os.path.basename(file_path)

    # 更新环境的可视化文件
    AMPwrapped_env.cfg.amp_motion_files_display = [file_path]
    AMPwrapped_env.amp_loader_display = AMPLoaderDisplay(
        motion_files=[file_path],
        device=AMPwrapped_env.device,
        time_between_frames=AMPwrapped_env.physics_dt
    )
    AMPwrapped_env.motion_len = AMPwrapped_env.amp_loader_display.trajectory_num_frames[0]

    if not hasattr(AMPwrapped_env, "motion_len") or AMPwrapped_env.motion_len <= 0:
        print("❌ 错误: `env.motion_len` 未定义或为零。无法播放动画。")
        return

    frame_cnt = 0
    all_frames = []

    # 播放当前文件的所有帧
    while frame_cnt < AMPwrapped_env.motion_len and simulation_app.is_running():
        t = frame_cnt * (1.0 / args_cli.fps)

        if not hasattr(AMPwrapped_env, "visualize_motion"):
            print("❌ 错误: 在您的环境中未找到 `visualize_motion(time)` 方法。")
            break

        frame = AMPwrapped_env.visualize_motion(t)

        if DataCfg.save_motion_expert_amp_data and args_cli.save_path and frame is not None:
            all_frames.append(frame.cpu().numpy().reshape(-1))

        frame_cnt += 1

    print(f"\n📊 播放统计: {file_name} - 共 {frame_cnt} 帧")

    # --- 文件保存逻辑 (仅在开启保存选项时) ---
    if DataCfg.save_motion_expert_amp_data and args_cli.save_path and len(all_frames) > 0:
        # 根据当前文件名生成保存路径
        base_name = os.path.splitext(file_name)[0]

        # 修复：如果 save_path 是目录，直接使用；如果是文件，则取其父目录
        if os.path.isdir(args_cli.save_path):
            save_dir = args_cli.save_path
        else:
            save_dir = os.path.dirname(args_cli.save_path)

        current_save_path = os.path.join(save_dir, f"{base_name}.txt")

        print(f"💾 正在保存动画数据到 {current_save_path}...")
        all_frames_np = np.stack(all_frames, axis=0)

        # 定义关节顺序和末端执行器顺序
        joint_order = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]

        # 末端执行器顺序必须与visualize_motion函数中的拼接顺序一致
        end_effector_order = [
            "left_wrist_yaw_link", "right_wrist_yaw_link",
            "left_ankle_roll_link", "right_ankle_roll_link"
        ]

        temp_path = current_save_path + ".tmp"
        np.savetxt(temp_path, all_frames_np, fmt='%f', delimiter=', ')
        with open(temp_path, 'r') as f:
            frames_data = f.readlines()
        frames_data_len = len(frames_data)

        with open(current_save_path, 'w') as f:
            f.write('{\n')
            f.write('"LoopMode": "Wrap",\n')
            f.write(f'"FrameDuration": {1.0 / args_cli.fps:.6f},\n')
            f.write('"EnableCycleOffsetPosition": true,\n')
            f.write('"EnableCycleOffsetRotation": true,\n')
            f.write('"MotionWeight": 0.5,\n\n')

            # ✅ 添加关节顺序
            f.write('"JointOrder": [\n')
            for i, joint_name in enumerate(joint_order):
                if i < len(joint_order) - 1:
                    f.write(f'  "{joint_name}",\n')
                else:
                    f.write(f'  "{joint_name}"\n')
            f.write('],\n\n')

            # ✅ 添加末端执行器顺序
            f.write('"EndEffectorOrder": [\n')
            for i, ee_name in enumerate(end_effector_order):
                if i < len(end_effector_order) - 1:
                    f.write(f'  "{ee_name}",\n')
                else:
                    f.write(f'  "{ee_name}"\n')
            f.write('],\n\n')

            f.write('"Frames":\n[\n')
            for i, line in enumerate(frames_data):
                line_start_str = '  ['
                if i == frames_data_len - 1:
                    f.write(line_start_str + line.rstrip() + ']\n')
                else:
                    f.write(line_start_str + line.rstrip() + '],\n')
            f.write(']\n}')
        os.remove(temp_path)
        print(f"✅ 成功转换并保存到 {current_save_path}")
        print(f"   - 包含 {len(joint_order)} 个关节")
        print(f"   - 包含 {len(end_effector_order)} 个末端执行器")
        print(f"   - 共 {frames_data_len} 帧数据")
    elif DataCfg.save_motion_expert_amp_data and len(all_frames) == 0:
        print(f"⚠️  跳过保存 {file_name}: 未捕获到任何帧")


if __name__ == "__main__":
    main()
