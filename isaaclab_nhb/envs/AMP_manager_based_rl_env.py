# file: tienkung_env.py (The new one)

import torch
from rsl_rl.utils import AMPLoaderDisplay
from scipy.spatial.transform import Rotation
from isaaclab.utils.math import quat_apply, quat_conjugate
from isaaclab.assets.articulation import Articulation
from isaaclab_nhb.tasks.humanoid.G1.G1_AMP_env_cfg import G1AmpFlatEnvCfg # Import the new Cfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from isaaclab_nhb.dataset.amp_data_cfg.G1_amp_data_cfg import G1AmpDataCfg
from isaaclab_nhb.envs.manager_debug_rl_env import ManagerDebugRLEnv
import os


class AMPManagerBasedEnv(ManagerDebugRLEnv):
    """
    在play播放amp数据集的时候才使用
    The TienKung environment, migrated to the ManagerBasedRLEnv framework.
    This class handles custom logic such as gait parameter calculation,
    initialization of body/joint IDs, and visualization utilities that do not
    fit into the standard manager-based paradigm.
    """
    cfg: G1AmpFlatEnvCfg

    def __init__(self, cfg: G1AmpFlatEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.DataCfg = G1AmpDataCfg()
        self.play_amp_animation_mode = False
        # -- Initialize custom state variables from the old `init_buffers` --
        self.robot: Articulation = self.scene["robot"]
        # Local vectors for hand position calculation
        self.hand_body_ids, _ = self.robot.find_bodies(["left_wrist_yaw_link", "right_wrist_yaw_link"],preserve_order=True)
        self.feet_body_ids, _ = self.robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"],preserve_order=True)
        self.elbow_body_ids, _ = self.robot.find_bodies(["left_elbow_link", "right_elbow_link"],preserve_order=True)
        # ... Find and store all other necessary joint and body IDs ...
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.waist_ids, _ = self.robot.find_joints(
            name_keys=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            preserve_order=True,
        )
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ],
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
            preserve_order=True,
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=["left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint",
                       "right_ankle_roll_joint"],
            preserve_order=True,
        )


        # -- AMP visualization loader (if needed) --
        # You need to define cfg.amp_motion_files_display in your TienKungEnvCfg
        # if hasattr(self.cfg, "amp_motion_files_display"):
        if self.play_amp_animation_mode:
            print("G1AmpRoughEnvCfgdebug", cfg.DataCfg.save_motion_expert_path)
            amp_motion_files = cfg.DataCfg.save_motion_expert_path

            self.amp_loader_display = AMPLoaderDisplay(
                motion_files=cfg.amp_motion_files_display, device=self.device, time_between_frames=self.physics_dt
            )
            self.motion_len = self.amp_loader_display.trajectory_num_frames[0]

        # [!!!] 修复：初始化 4 个 EE 可视化工具
        if self.cfg.enable_ee_visualizations:
            # 检查配置是否存在
            if not all(hasattr(self.cfg, name) for name in
                       ["ee_viz_lh_cfg", "ee_viz_rh_cfg", "ee_viz_lf_cfg", "ee_viz_rf_cfg"]):
                print("[警告] 缺少一个或多个 ee_viz_..._cfg 配置。末端可视化将被禁用。")
                self.ee_viz_lh = self.ee_viz_rh = self.ee_viz_lf = self.ee_viz_rf = None
            else:
                self.ee_viz_lh = VisualizationMarkers(self.cfg.ee_viz_lh_cfg)
                self.ee_viz_rh = VisualizationMarkers(self.cfg.ee_viz_rh_cfg)
                self.ee_viz_lf = VisualizationMarkers(self.cfg.ee_viz_lf_cfg)
                self.ee_viz_rf = VisualizationMarkers(self.cfg.ee_viz_rf_cfg)
        else:
            self.ee_viz_lh = self.ee_viz_rh = self.ee_viz_lf = self.ee_viz_rf = None

    # #
    def visualize_motion(self, time):
        """
        [用于可视化]：根据 mocap 数据设置机器人 t 时刻的状态，
        步进到 t+dt，然后计算并返回 t+dt 时刻的末端执行器位置。

        注意：此函数返回的 ee_pos 时间戳 (t+dt) 与 dof_pos/vel (t) 不同步。
        这对于可视化是正确的，但不能用于生成训练数据。

        Args:
            time (float): The time (in seconds) at which to fetch the AMP motion frame.

        Returns:
            torch.Tensor: 包含 dof_pos(t), dof_vel(t) 和 ee_pos(t+dt) 的张量。
        """
        visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        device = self.device

        # --- 1. 准备 t 时刻的状态 (来自 mocap 文件) ---
        dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)

        dof_pos[:, self.left_leg_ids] = visual_motion_frame[6:12]
        dof_pos[:, self.right_leg_ids] = visual_motion_frame[12:18]
        dof_pos[:, self.waist_ids] = visual_motion_frame[18:21]
        dof_pos[:, self.left_arm_ids] = visual_motion_frame[21:28]
        dof_pos[:, self.right_arm_ids] = visual_motion_frame[28:35]

        dof_vel[:, self.left_leg_ids] = visual_motion_frame[41:47]
        dof_vel[:, self.right_leg_ids] = visual_motion_frame[47:53]
        dof_vel[:, self.waist_ids] = visual_motion_frame[53:56]
        dof_vel[:, self.left_arm_ids] = visual_motion_frame[56:63]
        dof_vel[:, self.right_arm_ids] = visual_motion_frame[63:70]

        env_ids = torch.arange(self.num_envs, device=device)

        root_pos = visual_motion_frame[:3].clone()
        root_pos[2] += 0.3  # (z 轴偏移)

        euler = visual_motion_frame[3:6].cpu().numpy()
        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()  # [x, y, z, w]
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=device
        )

        lin_vel = visual_motion_frame[35:38].clone()
        ang_vel = visual_motion_frame[38:41].clone()

        # === 打印每一帧的 root 状态信息（单行格式，带数据集名称和突出x方向速度）===
        if self.DataCfg.print_frame_root_info:
            # 获取当前数据集名称（从配置路径中提取，限制7个字符）
            dataset_name = "Unknown"
            if hasattr(self.cfg, 'amp_motion_files_display') and len(self.cfg.amp_motion_files_display) > 0:
                full_path = self.cfg.amp_motion_files_display[0]
                file_name = os.path.basename(full_path)
                # 移除扩展名并限制长度
                dataset_name = os.path.splitext(file_name)[0][:8]

            # 格式化输出，包含线速度和角速度
            print(f"[{dataset_name:>8s}] Frame t={time:.3f}s | "
                  f"Pos:[{root_pos[0]:7.4f}, {root_pos[1]:7.4f}, {root_pos[2]:7.4f}] | "
                  f"Euler:[{euler[0]:7.4f}, {euler[1]:7.4f}, {euler[2]:7.4f}] | "
                  f"LinVel:[{lin_vel[0]:7.4f}, {lin_vel[1]:7.4f}, {lin_vel[2]:7.4f}] | "
                  f"AngVel:[{ang_vel[0]:7.4f}, {ang_vel[1]:7.4f}, {ang_vel[2]:7.4f}]")

        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        # --- 2. 写入 t 时刻的状态到缓冲区 ---
        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(dof_vel)
        self.robot.write_root_state_to_sim(root_state, env_ids)

        # --- 3. 步进 (t -> t+dt), 更新 (t+dt), 渲染 (t+dt) ---
        self.sim.render()
        self.sim.step()  # 使用 self.physics_dt 步进
        self.scene.update(dt=self.step_dt)  # 将 t+dt 的状态读入 self.robot.data

        # if self.cfg.enable_ee_visualizations:
        #     self._update_ee_visualization(
        #         self.robot.data.body_state_w[:, self.hand_body_ids[0], :3],
        #         self.robot.data.body_state_w[:, self.hand_body_ids[1], :3],
        #         self.robot.data.body_state_w[:, self.feet_body_ids[0], :3],
        #         self.robot.data.body_state_w[:, self.feet_body_ids[1], :3],)
        # --- 4. 计算 t+dt 时刻的末端位置 ---
        # (self.robot.data 现在包含 t+dt 的数据)
        # left_hand_pos = (
        #         self.robot.data.body_state_w[:, self.hand_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        # )
        # right_hand_pos = (
        #         self.robot.data.body_state_w[:, self.hand_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        # )
        # left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        # right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        #
        # left_foot_pos = (
        #         self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        # )
        # right_foot_pos = (
        #         self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        # )
        # left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        # right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)

        bodies_order = ["left_wrist_yaw_link","right_wrist_yaw_link","left_ankle_roll_link","right_ankle_roll_link"]
        function_output = self.bodies_pos_order_r(self.robot, bodies_order)
        num_envs = function_output.shape[0]
        bodies_pos_local = function_output.reshape(num_envs, 4, 3)
        left_hand_pos = bodies_pos_local[:, 0, :]  # Shape: (num_envs, 3)
        right_hand_pos = bodies_pos_local[:, 1, :]  # Shape: (num_envs, 3)
        left_foot_pos = bodies_pos_local[:, 2, :]  # Shape: (num_envs, 3)
        right_foot_pos = bodies_pos_local[:, 3, :]  # Shape: (num_envs, 3)

        # 🔍 调试：打印专家数据播放时的脚部Y坐标
        if not hasattr(self, '_viz_coord_printed'):
            print(f"\n[坐标系诊断] 专家数据可视化时的脚部Y坐标（局部坐标系）:")
            print(f"  左脚Y: {left_foot_pos[0, 1].item():+.4f}")
            print(f"  右脚Y: {right_foot_pos[0, 1].item():+.4f}")
            if left_foot_pos[0, 1] > 0 and right_foot_pos[0, 1] < 0:
                print(f"  ✓ 右手系：左正右负")
            elif left_foot_pos[0, 1] < 0 and right_foot_pos[0, 1] > 0:
                print(f"  ✗ 左手系：左负右正")
            self._viz_coord_printed = True

        # root2world
        ## 将局部坐标转换回世界坐标（用于可视化）

        # 1. 获取机器人的根节点世界位置和姿态
        root_pos_w = self.robot.data.root_state_w[:, 0:3]
        root_quat_w = self.robot.data.root_state_w[:, 3:7]

        # 2. 将局部坐标向量从"根节点坐标系"旋转回"世界坐标系"
        #    (注意：这里用的是原始四元数 root_quat_w，而不是共轭)
        left_hand_vec_in_world = quat_apply(root_quat_w, left_hand_pos)
        right_hand_vec_in_world = quat_apply(root_quat_w, right_hand_pos)
        left_foot_vec_in_world = quat_apply(root_quat_w, left_foot_pos)
        right_foot_vec_in_world = quat_apply(root_quat_w, right_foot_pos)

        # 3. 加上根节点的世界坐标，得到最终的、复现出来的世界坐标
        #    这些 '..._reconstructed' 变量就是你可以用来可视化的
        left_hand_pos_world_reconstructed = left_hand_vec_in_world + root_pos_w
        right_hand_pos_world_reconstructed = right_hand_vec_in_world + root_pos_w
        left_foot_pos_world_reconstructed = left_foot_vec_in_world + root_pos_w
        right_foot_pos_world_reconstructed = right_foot_vec_in_world + root_pos_w

        if self.cfg.enable_ee_visualizations:
            self._update_ee_visualization(
                left_hand_pos_world_reconstructed,
                right_hand_pos_world_reconstructed,
                left_foot_pos_world_reconstructed,
                right_foot_pos_world_reconstructed,)

        # --- 5. 准备 t 时刻的关节数据 (用于返回/打印) ---
        self.left_leg_dof_pos = dof_pos[:, self.left_leg_ids]
        self.right_leg_dof_pos = dof_pos[:, self.right_leg_ids]
        self.left_leg_dof_vel = dof_vel[:, self.left_leg_ids]
        self.right_leg_dof_vel = dof_vel[:, self.right_leg_ids]
        self.left_arm_dof_pos = dof_pos[:, self.left_arm_ids]
        self.right_arm_dof_pos = dof_pos[:, self.right_arm_ids]
        self.left_arm_dof_vel = dof_vel[:, self.left_arm_ids]
        self.right_arm_dof_vel = dof_vel[:, self.right_arm_ids]
        self.waist_dof_pos = dof_pos[:, self.waist_ids]
        self.waist_dof_vel = dof_vel[:, self.waist_ids]

        # --- 6. 拼接并返回 (dof(t), vel(t), ee_pos(t+dt)) ---
        joint_order_with_end_order = torch.cat(
            (
                self.left_leg_dof_pos, self.right_leg_dof_pos,
                self.waist_dof_pos,
                self.left_arm_dof_pos, self.right_arm_dof_pos,

                self.left_leg_dof_vel, self.right_leg_dof_vel,
                self.waist_dof_vel,
                self.left_arm_dof_vel, self.right_arm_dof_vel,

                left_hand_pos, right_hand_pos,
                left_foot_pos, right_foot_pos
            ),
            dim=-1,
        )
        if (self.DataCfg.print_joint_order_with_end_order):
            print("末端部分:\n", joint_order_with_end_order[0, :])

        return joint_order_with_end_order

    def _update_ee_visualization(self,left_hand_pos_w,right_hand_pos_w, left_foot_pos_w, right_foot_pos_w):
        """
        计算并可视化末端执行器在*root坐标系*中的位置。
        这个函数应该在 self.scene.update() 之后被调用。
        """
        self.ee_viz_lh.visualize(left_hand_pos_w)
        self.ee_viz_rh.visualize(right_hand_pos_w)
        self.ee_viz_lf.visualize(left_foot_pos_w)
        self.ee_viz_rf.visualize(right_foot_pos_w)

    def bodies_pos_order_r(
            self,
            robot: Articulation,
            bodies_order: str | list[str] = [],
    ) -> torch.Tensor:
        """
        Computes the bodies pos observation order in robot root frame from the environment's state.
        """
        body_ids, _ = robot.find_bodies(name_keys=bodies_order, preserve_order=True)
        bodies_pos_rel = robot.data.body_pos_w[:, body_ids, :3] - robot.data.root_pos_w.unsqueeze(1)
        num_envs, num_bodies, _ = bodies_pos_rel.shape
        bodies_pos_flat = bodies_pos_rel.reshape(num_envs * num_bodies, 3)

        root_quat = robot.data.root_state_w[:, 3:7]
        root_quat_expanded = root_quat.unsqueeze(1).repeat(1, num_bodies, 1).reshape(num_envs * num_bodies, 4)

        bodies_pos_root_flat = math_utils.quat_apply_inverse(root_quat_expanded, bodies_pos_flat)

        bodies_pos_root = bodies_pos_root_flat.reshape(num_envs, num_bodies, 3)

        return bodies_pos_root.reshape(num_envs, -1)  # Flatten for observation
