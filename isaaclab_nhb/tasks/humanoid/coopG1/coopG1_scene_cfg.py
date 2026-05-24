import math

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils


def _quat_wxyz_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """Returns quaternion (w, x, y, z) for a rotation around +Z by `yaw` radians."""
    half = 0.5 * yaw
    return (math.cos(half), 0.0, 0.0, math.sin(half))

def get_coop_scene_cfg(num_robots: int = 2, include_box: bool = True):
    """
    动态生成多智能体协作场景配置
    :param num_robots: 初始化的 G1 机器人数量 (1, 2, 3...)
    :param include_box: 是否在一开始生成被搬运的木箱
    """

    # NOTE: 延迟导入，避免在任务注册/包初始化阶段触发 G1 资产模块的重依赖。
    from isaaclab_nhb.tasks.humanoid.G1.G1_asset_cfg import G1_29DOF_CFG

    coop_joint_pos = dict(G1_29DOF_CFG.init_state.joint_pos)
    coop_joint_pos.pop(".*_elbow_joint", None)
    # 给定一个像正常人类准备搬东西的姿态 (胸前水平环抱)
    coop_joint_pos.update(
        {
            "left_shoulder_pitch_joint": 0.25,
            "left_shoulder_roll_joint": 0.55,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 1.05,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.25,
            "right_shoulder_roll_joint": -0.55,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 1.05,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }
    )
    
    @configclass
    class GeneratedCoopG1SceneCfg(InteractiveSceneCfg):
        # 1. 基础地形
        terrain = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=GroundPlaneCfg(color=(0.1, 0.1, 0.1)),
        )

        # 2. 环境光照
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)),
        )

    # 3. 动态计算并注入机器人和对应接触力传感器
    radius = 0.40  # 所有机器人围绕原点的半径(米)
    for i in range(num_robots):
        # 均匀分布于圆周
        angle = i * (2 * math.pi / num_robots)
        pos_x = radius * math.cos(angle)
        pos_y = radius * math.sin(angle)
        
        # 让机器人面向圆心 (绕Z轴旋转)
        # 面对圆心的偏航角(yaw)为 angle + pi
        yaw = angle + math.pi
        rot_wxyz = _quat_wxyz_from_yaw(yaw)

        # 动态创建配置对象（遵循仓库现有写法：对已有 cfg 调用 .replace(...)）
        # 只在 coopG1 中覆盖 pos/rot 和手臂初始姿态，不改全局 G1 asset cfg。
        robot_cfg = G1_29DOF_CFG.replace(
            prim_path=f"{{ENV_REGEX_NS}}/Robot_{i}",
            init_state=G1_29DOF_CFG.init_state.replace(
                pos=(pos_x, pos_y, 0.75),
                rot=rot_wxyz,
                joint_pos=coop_joint_pos,
            ),
        )
        # 动态挂载到类属性上
        setattr(GeneratedCoopG1SceneCfg, f"robot_{i}", robot_cfg)
        setattr(
            GeneratedCoopG1SceneCfg,
            f"contact_forces_{i}",
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot_{i}/.*",
                history_length=3,
                track_air_time=True,
                debug_vis=False,
            ),
        )

    # 4. 可选是否生成目标木箱
    if include_box:
        box_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.4, 0.4, 0.4),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.90)),
        )
        setattr(GeneratedCoopG1SceneCfg, "box", box_cfg)

    return GeneratedCoopG1SceneCfg
