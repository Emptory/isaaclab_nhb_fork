import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_nhb import ISAACLAB_ROBOT_DESCRIPTION_PATH

JOINT_MIN_DELAY_STEP = 1
JOINT_MAX_DELAY_STEP = 3

S3_12DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path= f"{ISAACLAB_ROBOT_DESCRIPTION_PATH}/S3/usd/S3_22dof/S3_22dof.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        # joint_pos={
        #     ".*_hip_roll_.*": 0.0,
        #     ".*_hip_yaw_.*": 0.0,
        #     ".*_hip_pitch_.*": -0.2,
        #     ".*_knee_.*": 0.4,
        #     ".*_foot_pitch_.*": -0.2,
        #     ".*_foot_roll_.*": 0.0,
        # },
        joint_pos={
            ".*_hip_roll_.*": 0.0,
            ".*_hip_yaw_.*": 0.0,
            ".*_hip_pitch_.*": -0.35,
            ".*_knee_.*": 0.7,
            ".*_foot_pitch_.*": -0.35,
            ".*_foot_roll_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "leg":DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_.*",
                ".*_hip_yaw_.*",
                ".*_hip_pitch_.*",
                ".*_knee_.*",
                ".*_foot_pitch_.*",
                ".*_foot_roll_.*",
            ],
            effort_limit={
                ".*_hip_roll_.*": 200.0,
                ".*_hip_yaw_.*": 120.0,
                ".*_hip_pitch_.*": 200.0,
                ".*_knee_.*": 300.0,
                ".*_foot_pitch_.*": 50.0,
                ".*_foot_roll_.*": 50.0,
            },
            velocity_limit={
                ".*_hip_roll_.*": 23.04,
                ".*_hip_yaw_.*": 23.04,
                ".*_hip_pitch_.*": 23.04,
                ".*_knee_.*": 8.38,
                ".*_foot_pitch_.*": 17.28,
                ".*_foot_roll_.*": 20.94,
            },
            stiffness={
                ".*_hip_roll_.*": 80.0,
                ".*_hip_yaw_.*": 80.0,
                ".*_hip_pitch_.*": 100.0,
                ".*_knee_.*": 100.0,
                ".*_foot_pitch_.*": 50.0,
                ".*_foot_roll_.*": 50.0,
                # ".*_foot_pitch_.*": 35.0,
                # ".*_foot_roll_.*": 12.0,
            },
            damping={
                ".*_hip_roll_.*": 2.0,
                ".*_hip_yaw_.*": 2.0,
                ".*_hip_pitch_.*": 3.0,
                ".*_knee_.*": 3.0,
                ".*_foot_pitch_.*": 1.5,
                ".*_foot_roll_.*": 1.5,
                # ".*_foot_pitch_.*": 1.5,
                # ".*_foot_roll_.*": 0.3,
            },
            min_delay=JOINT_MIN_DELAY_STEP,
            max_delay=JOINT_MAX_DELAY_STEP,
            armature=0.01,
        ),
    },
)

S3_12DOF_FIX_BASE_CFG = S3_12DOF_CFG.copy()
S3_12DOF_FIX_BASE_CFG.spawn.articulation_props.fix_root_link = True
S3_12DOF_FIX_BASE_CFG.init_state.pos = (0.0, 0.0, 1.0)

S3_12DOF_JOINT_LOWER_ORDER = [
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_foot_pitch_joint",
    "left_foot_roll_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_foot_pitch_joint",
    "right_foot_roll_joint",
]

S3_12DOF_JOINT_UPPER_ORDER = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_hand_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_hand_joint",
]

S3_22DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path= f"{ISAACLAB_ROBOT_DESCRIPTION_PATH}/S3/usd/S3_22dof_noshell/S3_22dof_noshell.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            ".*_hip_roll_.*": 0.0,
            ".*_hip_yaw_.*": 0.0,
            ".*_hip_pitch_.*": -0.35,
            ".*_knee_.*": 0.70,
            ".*_foot_pitch_.*": -0.35,
            ".*_foot_roll_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "leg":DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_.*",
                ".*_hip_yaw_.*",
                ".*_hip_pitch_.*",
                ".*_knee_.*",
                ".*_foot_pitch_.*",
                ".*_foot_roll_.*",
            ],
            effort_limit={
                ".*_hip_roll_.*": 200.0,
                ".*_hip_yaw_.*": 120.0,
                ".*_hip_pitch_.*": 200.0,
                ".*_knee_.*": 300.0,
                ".*_foot_pitch_.*": 50.0,
                ".*_foot_roll_.*": 50.0,
            },
            velocity_limit={
                ".*_hip_roll_.*": 23.04,
                ".*_hip_yaw_.*": 23.04,
                ".*_hip_pitch_.*": 23.04,
                ".*_knee_.*": 8.38,
                ".*_foot_pitch_.*": 17.28,
                ".*_foot_roll_.*": 20.94,
            },
            stiffness={
                ".*_hip_roll_.*": 80.0,
                ".*_hip_yaw_.*": 80.0,
                ".*_hip_pitch_.*": 100.0,
                ".*_knee_.*": 100.0,
                ".*_foot_pitch_.*": 50.0,
                ".*_foot_roll_.*": 50.0,
            },
            damping={
                ".*_hip_roll_.*": 2.0,
                ".*_hip_yaw_.*": 2.0,
                ".*_hip_pitch_.*": 3.0,
                ".*_knee_.*": 3.0,
                ".*_foot_pitch_.*": 1.5,
                ".*_foot_roll_.*": 1.5,
            },
            min_delay=JOINT_MIN_DELAY_STEP,
            max_delay=JOINT_MAX_DELAY_STEP,
            armature=0.01,
        ),
        "arm":DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*shoulder_pitch.*",
                ".*shoulder_roll.*",
                ".*shoulder_yaw.*",
                ".*elbow.*",
                ".*hand.*",
            ],
            effort_limit={
                ".*shoulder_pitch.*": 66.0,
                ".*shoulder_roll.*": 66.0,
                ".*shoulder_yaw.*": 66.0,
                ".*elbow.*": 66.0,
                ".*hand.*": 18.0,
            },
            velocity_limit={
                ".*shoulder_pitch.*": 11.41,
                ".*shoulder_roll.*": 11.41,
                ".*shoulder_yaw.*": 11.41,
                ".*elbow.*": 11.41,
                ".*hand.*": 7.32,
            },
            stiffness={
                ".*shoulder_pitch.*": 80.0,
                ".*shoulder_roll.*": 80.0,
                ".*shoulder_yaw.*": 80.0,
                ".*elbow.*": 80.0,
                ".*hand.*": 80.0,
            },
            damping={
                ".*shoulder_pitch.*": 2.0,
                ".*shoulder_roll.*": 2.0,
                ".*shoulder_yaw.*": 2.0,
                ".*elbow.*": 2.0,
                ".*hand.*": 2.0,
            },
            min_delay=JOINT_MIN_DELAY_STEP,
            max_delay=JOINT_MAX_DELAY_STEP,
            armature=0.01,
        ),
    },
)

S3_22DOF_FIX_JOINTLIMIT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path= f"{ISAACLAB_ROBOT_DESCRIPTION_PATH}/S3/usd/S3_22dof_noshell_fixJointLimit/S3_22dof_noshell_fixJointLimit.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.1),
        joint_pos={
            ".*_hip_roll_.*": 0.0,
            ".*_hip_yaw_.*": 0.0,
            ".*_hip_pitch_.*": -0.35,
            ".*_knee_.*": 0.70,
            ".*_foot_pitch_.*": -0.35,
            ".*_foot_roll_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "leg":DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_.*",
                ".*_hip_yaw_.*",
                ".*_hip_pitch_.*",
                ".*_knee_.*",
                ".*_foot_pitch_.*",
                ".*_foot_roll_.*",
            ],
            effort_limit={
                ".*_hip_roll_.*": 100.0,
                ".*_hip_yaw_.*": 61.0,
                ".*_hip_pitch_.*": 100.0,
                ".*_knee_.*": 258.0,
                ".*_foot_pitch_.*": 50, # 需要修改
                ".*_foot_roll_.*": 50, # 需要修改
            },
            velocity_limit={
                ".*_hip_roll_.*": 17.62,
                ".*_hip_yaw_.*": 31.72,
                ".*_hip_pitch_.*": 17.62,
                ".*_knee_.*": 9.31,
                ".*_foot_pitch_.*": 17.28, # 需要修改
                ".*_foot_roll_.*": 17.28, # 需要修改
            },
            stiffness={
                ".*_hip_roll_.*": 80.0,
                ".*_hip_yaw_.*": 80.0,
                ".*_hip_pitch_.*": 100.0,
                ".*_knee_.*": 100.0,
                ".*_foot_pitch_.*": 35.0,
                ".*_foot_roll_.*": 12.0,
            },
            damping={
                ".*_hip_roll_.*": 2.0,
                ".*_hip_yaw_.*": 2.0,
                ".*_hip_pitch_.*": 3.0,
                ".*_knee_.*": 3.0,
                ".*_foot_pitch_.*": 1.5,
                ".*_foot_roll_.*": 0.3,
            },
            min_delay=JOINT_MIN_DELAY_STEP,
            max_delay=JOINT_MAX_DELAY_STEP,
            armature=0.01,
        ),
        "arm":DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*shoulder_pitch.*",
                ".*shoulder_roll.*",
                ".*shoulder_yaw.*",
                ".*elbow.*",
                ".*hand.*",
            ],
            effort_limit={
                ".*shoulder_pitch.*": 66.0,
                ".*shoulder_roll.*": 66.0,
                ".*shoulder_yaw.*": 66.0,
                ".*elbow.*": 66.0,
                ".*hand.*": 18.0,
            },
            velocity_limit={
                ".*shoulder_pitch.*": 11.41,
                ".*shoulder_roll.*": 11.41,
                ".*shoulder_yaw.*": 11.41,
                ".*elbow.*": 11.41,
                ".*hand.*": 7.32,
            },
            stiffness={
                ".*shoulder_pitch.*": 80.0,
                ".*shoulder_roll.*": 80.0,
                ".*shoulder_yaw.*": 80.0,
                ".*elbow.*": 80.0,
                ".*hand.*": 80.0,
            },
            damping={
                ".*shoulder_pitch.*": 2.0,
                ".*shoulder_roll.*": 2.0,
                ".*shoulder_yaw.*": 2.0,
                ".*elbow.*": 2.0,
                ".*hand.*": 2.0,
            },
            min_delay=JOINT_MIN_DELAY_STEP,
            max_delay=JOINT_MAX_DELAY_STEP,
            armature=0.01,
        ),
    },
)

S3_22DOF_JOINT_ORDER = [
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_foot_pitch_joint",
    "left_foot_roll_joint",

    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_foot_pitch_joint",
    "right_foot_roll_joint",

    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_hand_joint",

    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_hand_joint",
]

S3_22DOF_FIX_BASE_CFG = S3_22DOF_CFG.copy()
S3_22DOF_FIX_BASE_CFG.spawn.articulation_props.fix_root_link = True
S3_22DOF_FIX_BASE_CFG.init_state.pos = (0.0, 0.0, 1.0)