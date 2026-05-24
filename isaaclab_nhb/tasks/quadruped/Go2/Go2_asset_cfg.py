import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_nhb import ISAACLAB_ROBOT_DESCRIPTION_PATH

# 设置最大最小命令延迟时间
JOINT_MIN_DELAY_STEP = 1
JOINT_MAX_DELAY_STEP = 2

GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ROBOT_DESCRIPTION_PATH}/Go2/usd/Go2/Go2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*hip.*":  0.0,
            ".*thigh.*":  0.8,
            ".*calf.*":  -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_thigh": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint",".*_thigh_joint"],
            effort_limit=23.7,
            velocity_limit=30.1,
            stiffness=50.0,
            damping=1.0,
            friction=0.0,
            armature=0.01,
            min_delay=JOINT_MIN_DELAY_STEP,
            max_delay=JOINT_MAX_DELAY_STEP,
        ),
        "calf": DelayedPDActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=45.43,
            velocity_limit=15.70,
            stiffness=50.0,
            damping=1.0,
            friction=0.0,
            armature=0.01, 
            min_delay=JOINT_MIN_DELAY_STEP,
            max_delay=JOINT_MAX_DELAY_STEP,
        ),
    },
)

GO2_JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint"
]