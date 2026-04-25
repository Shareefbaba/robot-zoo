from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

FETCH_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/shareef/robot_zoo/assets/fetch/fetch.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "torso_lift_joint":       0.2,
            "shoulder_pan_joint":     0.0,
            "shoulder_lift_joint":    0.0,
            "upperarm_roll_joint":    0.0,
            "elbow_flex_joint":       1.5,
            "forearm_roll_joint":     0.0,
            "wrist_flex_joint":       1.0,
            "wrist_roll_joint":       0.0,
            "l_gripper_finger_joint": 0.05,
            "r_gripper_finger_joint": 0.05,
            "head_pan_joint":         0.0,
            "head_tilt_joint":        0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint",
                              "upperarm_roll_joint", "elbow_flex_joint",
                              "forearm_roll_joint", "wrist_flex_joint",
                              "wrist_roll_joint"],
            stiffness=800.0, damping=40.0,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_lift_joint"],
            stiffness=800.0, damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["l_gripper_finger_joint", "r_gripper_finger_joint"],
            stiffness=1e5, damping=1e3,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_pan_joint", "head_tilt_joint"],
            stiffness=100.0, damping=10.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["l_wheel_joint", "r_wheel_joint", "bellows_joint"],
            stiffness=0.0, damping=100.0,
        ),
    },
)
