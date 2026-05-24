import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_nhb import ISAACLAB_ROBOT_DESCRIPTION_PATH

PEGASUS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ROBOT_DESCRIPTION_PATH}/Pegasus/usd/Pegasus/Pegasus.usd",
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
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*1_Joint.*": 0.0,
            "lb2.*":  0.53,
            "lb3.*":  1.22,
            "lf2.*":  0.53,
            "lf3.*":  1.22,

            "rb2.*":  0.53,
            "rb3.*":  1.22,
            "rf2.*":  0.53,
            "rf3.*":  1.22,

        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=120.0,
            velocity_limit=20.0,
            stiffness=50.0,
            damping=1.0,
            friction=0.0,
            min_delay = 0,
            max_delay = 2,
        ),
    },
)

PEGASUS_JOINT_NAMES = [
    "lf1_Joint",
    "lf2_Joint",
    "lf3_Joint",
    "rf1_Joint",
    "rf2_Joint",
    "rf3_Joint",
    "lb1_Joint",
    "lb2_Joint",
    "lb3_Joint",
    "rb1_Joint",
    "rb2_Joint",
    "rb3_Joint"
]