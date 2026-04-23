# robot_zoo/envs/stretch_cfg.py
# Isaac Lab ArticulationCfg for Hello Robot Stretch 3

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

STRETCH_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/shareef/robot_zoo/assets/stretch/stretch.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint_left_wheel"          : 0.0,
            "joint_right_wheel"         : 0.0,
            "joint_lift"                : 0.5,
            "joint_arm_l3"              : 0.0,
            "joint_arm_l2"              : 0.0,
            "joint_arm_l1"              : 0.0,
            "joint_arm_l0"              : 0.0,
            "joint_wrist_yaw"           : 0.0,
            "joint_head_pan"            : 0.0,
            "joint_head_tilt"           : 0.0,
            "joint_gripper_finger_left" : 0.05,
            "joint_gripper_finger_right": 0.05,
        },
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["joint_left_wheel", "joint_right_wheel"],
            effort_limit=100.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=1000.0,
        ),
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["joint_lift"],
            effort_limit=100.0,
            velocity_limit=0.5,
            stiffness=1000.0,
            damping=100.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_arm_l.*"],
            effort_limit=50.0,
            velocity_limit=0.5,
            stiffness=1000.0,
            damping=100.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint_wrist_yaw"],
            effort_limit=10.0,
            velocity_limit=1.0,
            stiffness=100.0,
            damping=10.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["joint_head_.*"],
            effort_limit=10.0,
            velocity_limit=1.0,
            stiffness=100.0,
            damping=10.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint_gripper_finger_.*"],
            effort_limit=10.0,
            velocity_limit=1.0,
            stiffness=200.0,
            damping=20.0,
        ),
    },
)
